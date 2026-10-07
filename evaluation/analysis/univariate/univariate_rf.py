"""Univariate per-feature random-forest CV importance.

Split out of the original ``analysis/univariate.py``. For each feature
this analyzer fits a 1-feature random forest under K-fold (default 5) CV
and reports the mean / std held-out score (R² for regression, ROC-AUC
for classification).

Cross-validation defaults to :class:`~sklearn.model_selection.GroupKFold`
when the input frame has a ``dataset_id`` column (and the number of
distinct datasets allows it), so the held-out fold is always a *new
dataset*. This is the right default for dataset-level analyses where
the rows of ``df`` correspond to (dataset_id × seed × ratio × model)
observations: without grouping, the same dataset appears in train and
test of every fold, inflating R². Pass an explicit ``cv_splitter`` to
override.

Per-seed two-stage fitting
--------------------------
The analyzer consumes seed-level tables (``"long_abs"`` / ``"long_rel"``).
For each ``(model, response, feature)`` we fit a 1-feature CV
**independently per seed**, then aggregate the resulting CV scores
across seeds via :class:`MeanSDAggregator` (default). Per-feature CV has
no analytical p-value, so the aggregated row carries ``mean ± SD/√K``
and a t-CI but ``p`` / ``t`` are left NaN.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from .._aggregation import MeanSDAggregator, SeedAggregator
from ..base import (
    _alpha_to_cell,
    alpha_col_for_html,
    DatasetAnalyzer,
    FeatureImportance,
    group_keys_by_ratio,
    InputKind,
    render_feature_pivot_html,
)

log = logging.getLogger(__name__)


def _clean_xy(
    df: pd.DataFrame, feature_cols: list[str], response: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Return ``(X_df, y, groups)`` with NaN/inf rows removed.

    ``groups`` is the matching ``dataset_id`` array when the column is
    present, else ``None``.
    """
    cols = list(feature_cols) + [response]
    if "dataset_id" in df.columns:
        cols = cols + ["dataset_id"]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return sub.iloc[:, :0], np.empty(0), None
    groups = (
        sub["dataset_id"].to_numpy() if "dataset_id" in sub.columns else None
    )
    return sub[feature_cols], sub[response].values.astype(float), groups


@dataclass
class UnivariateResult:
    """Per-(model, response) aggregated + per-seed CV scores.

    ``importances`` is the cross-seed *aggregated* DataFrame::

        feature, cv_r2_mean, cv_r2_std, se, ci_low, ci_high,
        n_seeds, between_var, train_r2, n_obs, cv_n_splits

    ``per_seed`` is the raw per-seed DataFrame::

        seed, feature, cv_r2_mean, cv_r2_std, train_r2, n_obs

    Column names ``cv_r2_*`` are kept for backward compatibility — the
    values are R² for ``task="regression"`` and ROC-AUC for
    ``task="classification"``.
    """

    importances:  pd.DataFrame
    per_seed:     pd.DataFrame
    n_obs:        int
    cv_n_splits:  int


def _univariate_cv_one_feature(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: Literal["regression", "classification"],
    cv_splitter,
    groups: Optional[np.ndarray] = None,
    n_estimators: int = 300,
    random_state: int = 0,
) -> tuple[float, float, float, np.ndarray]:
    """Run a single-feature K-fold CV and return ``(cv_mean, cv_std, train, fold_scores)``."""
    X = x.reshape(-1, 1)
    if task == "regression":
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        scoring = "r2"
    else:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        scoring = "roc_auc"

    try:
        if groups is not None:
            fold_scores = cross_val_score(
                rf, X, y, cv=cv_splitter, scoring=scoring,
                n_jobs=1, groups=groups,
            )
        else:
            fold_scores = cross_val_score(
                rf, X, y, cv=cv_splitter, scoring=scoring, n_jobs=1,
            )
    except Exception as exc:
        n_splits = getattr(cv_splitter, "n_splits", 5)
        log.warning("  univariate RF CV failed: %s", exc)
        fold_scores = np.array([np.nan] * n_splits)

    rf.fit(X, y)
    if task == "regression":
        train = float(rf.score(X, y))
    else:
        try:
            from sklearn.metrics import roc_auc_score
            train = float(roc_auc_score(y, rf.predict_proba(X)[:, 1]))
        except (ValueError, IndexError):
            train = float("nan")

    return (
        float(np.nanmean(fold_scores)),
        float(np.nanstd(fold_scores)),
        train,
        np.asarray(fold_scores, dtype=float),
    )


class UnivariateRFAnalyzer(DatasetAnalyzer, FeatureImportance):
    """Per-feature 1-feature random-forest with K-fold CV, per seed.

    Defaults to :class:`GroupKFold` over ``dataset_id`` whenever the
    input frame carries that column (and the number of distinct groups
    is at least ``n_splits``). Otherwise falls back to a plain
    :class:`KFold`. Pass an explicit ``cv_splitter`` and ``groups`` to
    ``_fit_one_seed`` to override.
    """

    name: ClassVar[str] = "univariate"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs", "long_rel")

    def __init__(
        self,
        n_estimators: int = 300,
        n_splits: int = 5,
        random_state: int = 0,
        min_obs: int = 30,
        task: Literal["regression", "classification"] = "regression",
        aggregator: Optional[SeedAggregator] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.n_splits     = int(n_splits)
        self.random_state = int(random_state)
        self.min_obs      = int(min_obs)
        if task not in {"regression", "classification"}:
            raise ValueError(
                f"task must be 'regression' or 'classification', got {task!r}"
            )
        self.task = task
        self.aggregator = aggregator if aggregator is not None else MeanSDAggregator()

    # ── single-seed fit ─────────────────────────────────────────────────────

    def _fit_one_seed(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str,
        *,
        cv_splitter=None,
        groups: Optional[np.ndarray] = None,
    ) -> Optional[tuple[pd.DataFrame, int, int]]:
        """Run univariate CV for one seed slice.

        Returns ``(per_feature_df, n_obs, n_splits)`` where
        ``per_feature_df`` has columns
        ``(feature, cv_r2_mean, cv_r2_std, train_r2, n_obs)`` — or
        ``None`` if the seed slice is too small.
        """
        X_df, y, auto_groups = _clean_xy(df, feature_cols, response)
        n = len(y)
        if n < self.min_obs:
            return None
        if self.task == "classification":
            y = y.astype(int)
            if len(np.unique(y)) < 2:
                return None

        if groups is None:
            groups = auto_groups
        if cv_splitter is None:
            if groups is not None and len(np.unique(groups)) >= 2:
                n_splits = max(2, min(self.n_splits, len(np.unique(groups))))
                cv_splitter = GroupKFold(n_splits=n_splits)
            else:
                n_splits = max(2, min(self.n_splits, n))
                cv_splitter = KFold(
                    n_splits=n_splits, shuffle=True,
                    random_state=self.random_state,
                )
                groups = None
        else:
            n_splits = int(getattr(cv_splitter, "n_splits", self.n_splits))

        rows: list[dict] = []
        for f in feature_cols:
            x = X_df[f].values.astype(float)
            if not np.isfinite(x).any() or np.std(x) < 1e-12:
                rows.append({
                    "feature":    f,
                    "cv_r2_mean": float("nan"),
                    "cv_r2_std":  float("nan"),
                    "train_r2":   float("nan"),
                    "n_obs":      int(n),
                })
                continue
            cv_mean, cv_std, train, _ = _univariate_cv_one_feature(
                x, y,
                task=self.task,
                cv_splitter=cv_splitter,
                groups=groups,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            rows.append({
                "feature":    f,
                "cv_r2_mean": cv_mean,
                "cv_r2_std":  cv_std,
                "train_r2":   train,
                "n_obs":      int(n),
            })
        if not rows:
            return None
        return pd.DataFrame(rows), int(n), int(n_splits)

    # ── per-seed loop + aggregator combine ──────────────────────────────────

    def _aggregate(self, per_seed: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for feature, grp in per_seed.groupby("feature", dropna=False):
            per_seed_list = [
                {"estimate": float(r["cv_r2_mean"])}
                for _, r in grp.iterrows()
                if np.isfinite(r["cv_r2_mean"])
            ]
            agg = self.aggregator.combine(per_seed_list)
            rows.append({
                "feature":     feature,
                "cv_r2_mean":  agg["mean"],
                "cv_r2_std":   float(np.sqrt(agg["between_var"]))
                                 if np.isfinite(agg["between_var"]) and agg["between_var"] >= 0
                                 else float("nan"),
                "se":          agg["se"],
                "ci_low":      agg["ci_low"],
                "ci_high":     agg["ci_high"],
                "n_seeds":     agg["n_seeds"],
                "between_var": agg["between_var"],
                "train_r2":    float(grp["train_r2"].mean(skipna=True)),
            })
        if not rows:
            return pd.DataFrame(
                columns=[
                    "feature", "cv_r2_mean", "cv_r2_std", "se",
                    "ci_low", "ci_high", "n_seeds", "between_var", "train_r2",
                ]
            )
        return (
            pd.DataFrame(rows)
            .sort_values("cv_r2_mean", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    def fit_per_model_response(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], UnivariateResult]:
        out: dict[tuple[str, str], UnivariateResult] = {}
        if "seed" in df.columns:
            seeds = sorted(s for s in df["seed"].dropna().unique())
        else:
            seeds = [None]

        for model in models:
            sub_m = df[df["model"] == model]
            for resp in responses:
                log.info("  UNIV RF | model=%-10s response=%s", model, resp)
                per_seed_frames: list[pd.DataFrame] = []
                n_obs_max = 0
                n_splits_max = 0
                for seed in seeds:
                    if seed is None:
                        sub_s = sub_m
                        seed_val: object = float("nan")
                    else:
                        sub_s = sub_m[sub_m["seed"] == seed]
                        seed_val = seed
                    res = self._fit_one_seed(sub_s, feature_cols, resp)
                    if res is None:
                        continue
                    seed_df, n_obs_k, n_splits_k = res
                    n_obs_max = max(n_obs_max, n_obs_k)
                    n_splits_max = max(n_splits_max, n_splits_k)
                    seed_df = seed_df.copy()
                    seed_df.insert(0, "seed", seed_val)
                    per_seed_frames.append(seed_df)
                if not per_seed_frames:
                    log.info("    skipped (insufficient data)")
                    continue
                per_seed = pd.concat(per_seed_frames, ignore_index=True)
                aggregated = self._aggregate(per_seed)
                if aggregated.empty:
                    log.info("    skipped (no feature after aggregation)")
                    continue
                out[(model, resp)] = UnivariateResult(
                    importances=aggregated,
                    per_seed=per_seed,
                    n_obs=int(n_obs_max),
                    cv_n_splits=int(n_splits_max),
                )
        return out

    # ── DatasetAnalyzer interface ────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], UnivariateResult]:
        return self.fit_per_model_response(df, feature_cols, models, responses)

    def importance_long(
        self,
        results: dict[tuple[str, str], UnivariateResult],
    ) -> pd.DataFrame:
        """Long-format rows of *aggregated* CV scores."""
        rows: list[dict] = []
        for (model, resp), result in results.items():
            for _, c in result.importances.iterrows():
                rows.append({
                    "model":       model,
                    "response":    resp,
                    "feature":     c["feature"],
                    "cv_r2_mean":  float(c["cv_r2_mean"]) if pd.notna(c["cv_r2_mean"]) else float("nan"),
                    "cv_r2_std":   float(c["cv_r2_std"]) if pd.notna(c["cv_r2_std"]) else float("nan"),
                    "se":          float(c["se"]) if pd.notna(c["se"]) else float("nan"),
                    "ci_low":      float(c["ci_low"]) if pd.notna(c["ci_low"]) else float("nan"),
                    "ci_high":     float(c["ci_high"]) if pd.notna(c["ci_high"]) else float("nan"),
                    "n_seeds":     int(c["n_seeds"]),
                    "between_var": float(c["between_var"]) if pd.notna(c["between_var"]) else float("nan"),
                    "train_r2":    float(c["train_r2"]) if pd.notna(c["train_r2"]) else float("nan"),
                    "n_obs":       int(result.n_obs),
                    "cv_n_splits": int(result.cv_n_splits),
                })
        return pd.DataFrame(rows)

    def per_seed_long(
        self,
        results: dict[tuple[str, str], UnivariateResult],
    ) -> pd.DataFrame:
        """Long-format rows of *per-seed* CV scores."""
        rows: list[dict] = []
        for (model, resp), result in results.items():
            for _, c in result.per_seed.iterrows():
                rows.append({
                    "model":      model,
                    "response":   resp,
                    "feature":    c["feature"],
                    "seed":       c["seed"],
                    "cv_r2_mean": float(c["cv_r2_mean"]) if pd.notna(c["cv_r2_mean"]) else float("nan"),
                    "cv_r2_std":  float(c["cv_r2_std"]) if pd.notna(c["cv_r2_std"]) else float("nan"),
                    "train_r2":   float(c["train_r2"]) if pd.notna(c["train_r2"]) else float("nan"),
                    "n_obs":      int(c["n_obs"]),
                })
        return pd.DataFrame(rows)

    def save(
        self,
        results_by_key: dict[
            tuple[float, "str | float | None"],
            dict[tuple[str, str], UnivariateResult],
        ],
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        grouped = group_keys_by_ratio(results_by_key)

        # 1. Long aggregated + per-seed CSVs.
        long_frames: list[pd.DataFrame] = []
        per_seed_frames: list[pd.DataFrame] = []
        for ratio, pairs in grouped.items():
            for alpha_label, results in pairs:
                df_agg = self.importance_long(results)
                if not df_agg.empty:
                    df_agg = df_agg.copy()
                    df_agg.insert(0, "ratio", float(ratio))
                    df_agg.insert(1, "alpha", _alpha_to_cell(alpha_label))
                    long_frames.append(df_agg)
                df_ps = self.per_seed_long(results)
                if not df_ps.empty:
                    df_ps = df_ps.copy()
                    df_ps.insert(0, "ratio", float(ratio))
                    df_ps.insert(1, "alpha", _alpha_to_cell(alpha_label))
                    per_seed_frames.append(df_ps)
        if long_frames:
            (
                pd.concat(long_frames, ignore_index=True)
                .to_csv(out_dir / "summary.csv", index=False)
            )
        if per_seed_frames:
            (
                pd.concat(per_seed_frames, ignore_index=True)
                .to_csv(out_dir / "summary_per_seed.csv", index=False)
            )

        # 2. Per-ratio HTML pivot of cv_r2_mean (clip negatives to 0).
        for ratio, pairs in grouped.items():
            sub_frames: list[pd.DataFrame] = []
            for alpha_label, results in pairs:
                df = self.importance_long(results)
                if df.empty:
                    continue
                df = df.copy()
                df["cv_r2_mean"] = df["cv_r2_mean"].clip(lower=0.0)
                df["alpha_label"] = (
                    "alpha_free" if (alpha_label is None
                                     or alpha_label == "alpha_free")
                    else f"alpha={float(alpha_label)}"
                )
                sub_frames.append(df)
            if not sub_frames:
                continue
            combined = pd.concat(sub_frames, ignore_index=True)
            render_feature_pivot_html(
                combined,
                out_dir / f"summary_ratio_{ratio}.html",
                value_col="cv_r2_mean",
                caption=(
                    f"Univariate single-feature CV R² (clipped to ≥ 0; "
                    f"ratio={ratio}; rows = meta-feature, "
                    f"cols = alpha × response × model)"
                ),
                cmap="viridis",
                vmin=0.0, vmax=1.0,
                diverging=False,
                fmt="{:.3f}",
                alpha_col=alpha_col_for_html(pairs),
            )
