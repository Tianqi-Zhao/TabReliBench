"""Compare XGBoost meta-models fitted on X-only, Y-only, and X--Y features.

The analyzer deliberately reuses :class:`RFImportanceAnalyzer` for fitting.
Consequently the three feature-group models use the same per-seed dataset-level
GroupKFold splits, XGBoost hyper-parameters, native-NaN handling, and held-out
R² scoring as the existing full-feature ``rf`` analysis.  Only the predictor
columns differ.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ...features.dataset.selected_features import DATASET_FEATURE_GROUPS_BY_TASK
from ..base import DatasetAnalyzer, InputKind, _alpha_to_cell, group_keys_by_ratio
from .rf import RFImportanceAnalyzer, RFResult


FeatureGroupResults = dict[str, dict[tuple[str, str], RFResult]]


class FeatureGroupRFAnalyzer(DatasetAnalyzer):
    """Fit three otherwise-identical meta-models using dependency groups.

    This analyzer evaluates absolute metric values only.  Its output is compact
    fit-quality data rather than per-feature importance because its estimand is
    the predictive contribution of each *feature set*.
    """

    name: ClassVar[str] = "feature_group_rf"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs",)
    GROUP_ORDER: ClassVar[tuple[str, ...]] = (
        "x_only",
        "y_only",
        "xy_relation",
    )

    def __init__(
        self,
        *,
        dataset_task: str,
        n_estimators: int = 200,
        n_splits: int = 5,
        random_state: int = 0,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_obs: int = 15,
    ) -> None:
        if dataset_task not in DATASET_FEATURE_GROUPS_BY_TASK:
            raise ValueError(
                "dataset_task must be 'regression' or 'classification'; "
                f"got {dataset_task!r}"
            )
        self.dataset_task = dataset_task
        self.feature_groups = DATASET_FEATURE_GROUPS_BY_TASK[dataset_task]
        # All analyzed responses are continuous evaluation metrics, even when
        # the underlying benchmark task is classification.  Therefore the
        # reused meta-model is always an XGBRegressor scored by held-out R².
        self._rf = RFImportanceAnalyzer(
            n_estimators=n_estimators,
            n_splits=n_splits,
            random_state=random_state,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_obs=min_obs,
            task="regression",
            compute_permutation=False,
            compute_shap=False,
            compute_shap_interactions=False,
            save_models=False,
        )

    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> FeatureGroupResults:
        available = set(feature_cols) & set(df.columns)
        out: FeatureGroupResults = {}
        for group in self.GROUP_ORDER:
            group_cols = [
                feature for feature in self.feature_groups[group]
                if feature in available
            ]
            if not group_cols:
                continue
            missing = sorted(set(self.feature_groups[group]) - set(group_cols))
            if missing:
                print(
                    f"  GROUP-RF | {group}: ignoring unavailable columns "
                    f"{missing}"
                )
            print(
                f"  GROUP-RF | {group}: fitting with {len(group_cols)} features"
            )
            out[group] = self._rf.run(
                df,
                feature_cols=group_cols,
                models=models,
                responses=responses,
            )
        return out

    @staticmethod
    def _fit_quality_long(results: FeatureGroupResults) -> pd.DataFrame:
        rows: list[dict] = []
        for group, group_results in results.items():
            for (model, response), result in group_results.items():
                rows.append({
                    "feature_group": group,
                    "n_features": len(result.feature_order),
                    "model": model,
                    "response": response,
                    "cv_r2_mean": float(result.cv_r2_mean),
                    "cv_r2_std_across_seeds": float(result.cv_r2_std),
                    "train_r2_mean": float(result.train_r2),
                    "n_obs": int(result.n_obs),
                    "cv_n_splits": int(result.cv_n_splits),
                    "n_seeds": len(result.seed_fits),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _per_seed_long(results: FeatureGroupResults) -> pd.DataFrame:
        rows: list[dict] = []
        for group, group_results in results.items():
            for (model, response), result in group_results.items():
                for fit in result.seed_fits:
                    rows.append({
                        "feature_group": group,
                        "n_features": len(fit.feature_order),
                        "model": model,
                        "response": response,
                        "seed": fit.seed,
                        "cv_r2_mean": float(fit.cv_r2_mean),
                        "cv_r2_std_across_folds": float(fit.cv_r2_std),
                        "train_r2": float(fit.train_r2),
                        "n_obs": int(fit.n_obs),
                        "cv_n_splits": int(fit.cv_n_splits),
                    })
        return pd.DataFrame(rows)

    @classmethod
    def _paired_differences(
        cls, per_seed: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return per-seed and aggregated paired group R² differences."""
        if per_seed.empty:
            return pd.DataFrame(), pd.DataFrame()
        id_cols = ["ratio", "alpha", "model", "response", "seed"]
        work = per_seed.copy()
        # pandas pivot_table drops rows whose index contains NaN.  NaN is the
        # intentional alpha-free marker, so preserve it with a temporary value.
        alpha_free_sentinel = "__alpha_free__"
        work["alpha"] = work["alpha"].fillna(alpha_free_sentinel)
        wide = work.pivot_table(
            index=id_cols,
            columns="feature_group",
            values="cv_r2_mean",
            aggfunc="first",
        ).reset_index()
        wide["alpha"] = wide["alpha"].map(
            lambda value: np.nan if value == alpha_free_sentinel else value
        )
        per_seed_rows: list[dict] = []
        # A positive delta means the first-named group achieved higher held-out
        # R² than the second on the same experimental seed.
        for group_a, group_b in combinations(cls.GROUP_ORDER, 2):
            if group_a not in wide.columns or group_b not in wide.columns:
                continue
            for _, row in wide.dropna(subset=[group_a, group_b]).iterrows():
                per_seed_rows.append({
                    **{col: row[col] for col in id_cols},
                    "group_a": group_a,
                    "group_b": group_b,
                    "delta_cv_r2": float(row[group_a] - row[group_b]),
                })
        paired = pd.DataFrame(per_seed_rows)
        if paired.empty:
            return paired, pd.DataFrame()

        agg_rows: list[dict] = []
        agg_ids = ["ratio", "alpha", "model", "response", "group_a", "group_b"]
        for keys, sub in paired.groupby(agg_ids, dropna=False, sort=False):
            vals = sub["delta_cv_r2"].to_numpy(dtype=float)
            n = len(vals)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
            se = sd / np.sqrt(n) if n > 1 else float("nan")
            critical = (
                float(student_t.ppf(0.975, df=n - 1))
                if n > 1 else float("nan")
            )
            agg_rows.append({
                **dict(zip(agg_ids, keys)),
                "delta_cv_r2_mean": mean,
                "delta_cv_r2_std": sd,
                "delta_cv_r2_se": se,
                "delta_cv_r2_ci_low": mean - critical * se,
                "delta_cv_r2_ci_high": mean + critical * se,
                "n_paired_seeds": n,
            })
        return paired, pd.DataFrame(agg_rows)

    @classmethod
    def _render_comparison_html(
        cls, summary: pd.DataFrame, out_path: Path,
    ) -> None:
        if summary.empty:
            return
        work = summary.copy()
        work["alpha"] = work["alpha"].map(
            lambda value: "alpha_free" if pd.isna(value) else f"alpha={value:g}"
        )
        pivot = work.pivot_table(
            index=["ratio", "alpha", "response"],
            columns=["model", "feature_group"],
            values="cv_r2_mean",
            aggfunc="first",
        )
        if pivot.empty:
            return
        desired_cols = [
            (model, group)
            for model in sorted(work["model"].unique())
            for group in cls.GROUP_ORDER
            if (model, group) in pivot.columns
        ]
        pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(desired_cols))
        styled = (
            pivot.style
            .background_gradient(cmap="RdYlGn", axis=None, vmin=0.0, vmax=1.0)
            .format("{:.3f}", na_rep="—")
            .set_caption(
                "Held-out CV R² by meta-feature dependency group "
                "(same XGBoost settings and dataset folds)"
            )
        )
        out_path.write_text(styled.to_html(), encoding="utf-8")

    def save(
        self,
        results_by_key: dict[
            tuple[float, str | float | None], FeatureGroupResults
        ],
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_frames: list[pd.DataFrame] = []
        per_seed_frames: list[pd.DataFrame] = []
        for ratio, pairs in group_keys_by_ratio(results_by_key).items():
            for alpha_label, results in pairs:
                alpha = _alpha_to_cell(alpha_label)
                summary = self._fit_quality_long(results)
                if not summary.empty:
                    summary.insert(0, "ratio", float(ratio))
                    summary.insert(1, "alpha", alpha)
                    summary_frames.append(summary)
                per_seed = self._per_seed_long(results)
                if not per_seed.empty:
                    per_seed.insert(0, "ratio", float(ratio))
                    per_seed.insert(1, "alpha", alpha)
                    per_seed_frames.append(per_seed)

        if not summary_frames:
            return
        summary_all = pd.concat(summary_frames, ignore_index=True)
        summary_all.to_csv(out_dir / "fit_quality.csv", index=False)
        self._render_comparison_html(summary_all, out_dir / "fit_quality.html")

        if per_seed_frames:
            per_seed_all = pd.concat(per_seed_frames, ignore_index=True)
            per_seed_all.to_csv(out_dir / "fit_quality_per_seed.csv", index=False)
            paired, paired_agg = self._paired_differences(per_seed_all)
            if not paired.empty:
                paired.to_csv(
                    out_dir / "paired_differences_per_seed.csv", index=False,
                )
            if not paired_agg.empty:
                paired_agg.to_csv(
                    out_dir / "paired_differences.csv", index=False,
                )

        manifest_rows = [
            {"feature_group": group, "feature": feature}
            for group in self.GROUP_ORDER
            for feature in self.feature_groups[group]
        ]
        pd.DataFrame(manifest_rows).to_csv(
            out_dir / "feature_groups.csv", index=False,
        )
