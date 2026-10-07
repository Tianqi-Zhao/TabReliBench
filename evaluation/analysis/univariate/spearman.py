"""Per-feature Spearman r — direction-aware companion to Chatterjee ξ.

The other univariate analyzers (:class:`ChatterjeeAnalyzer`,
:class:`UnivariateRFAnalyzer`) and the multivariate
:class:`~evaluation.analysis.multivariate.RFImportanceAnalyzer` mostly report
**non-negative magnitudes** (Chatterjee ξ, univariate CV-R², RF gain,
``shap_mean_abs``) — *how much* dependence, not direction. RF **permutation**
importance and OOF SHAP are **signed**; use Spearman when you need a clean
monotone direction on a single feature.

This analyzer fills that gap: for every ``(model, response, feature)``
triple it reports

* ``r`` — Spearman rank correlation in ``[-1, +1]``,
* ``p`` — combined two-sided p-value (cross-seed aggregated), and
* ``n`` — max pairs used (after NaN/inf filtering) across seeds.

Sign is what makes Spearman useful here: pairing ``|r|`` with the
direction-blind ``ξ`` (see :class:`ChatterjeeAnalyzer`) reproduces the
read-out used by the existing heat-map analysis in
``eval_results/plot_dataset_pattern_heatmap.py``.

Per-seed two-stage fitting
--------------------------
Consumed on seed-level tables (``"long_abs"`` / ``"long_rel"``). For each
``(model, response, feature)`` we compute Spearman r per seed, then
combine across seeds via :class:`FisherZAggregator` (default) — Rubin's
rules applied in Fisher-z space, back-transformed to r at the end. The
reported SE / CI / p reflect both within-fit and between-seed variance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aggregation import FisherZAggregator, SeedAggregator
from ..base import (
    _alpha_to_cell,
    alpha_col_for_html,
    DatasetAnalyzer,
    FeatureImportance,
    group_keys_by_ratio,
    InputKind,
    render_feature_pivot_html,
)


# Pairs with fewer than this many finite observations are skipped.
# Mirrors ``MIN_SPEARMAN_N`` in eval_results/plot_dataset_pattern_heatmap.py
# and ``_MIN_SPEARMAN_N`` in the per-dataset summary pipeline.
MIN_SPEARMAN_N: int = 4


@dataclass
class SpearmanResult:
    """Per-(model, response) aggregated + per-seed Spearman correlations.

    ``correlations`` is the cross-seed *aggregated* DataFrame::

        feature, r, se, ci_low, ci_high, p, t, df,
        n_seeds, within_var, between_var, n

    ``per_seed`` is the raw per-seed DataFrame::

        seed, feature, r, p, n
    """

    correlations: pd.DataFrame
    per_seed:     pd.DataFrame
    n_obs:        int


class SpearmanAnalyzer(DatasetAnalyzer, FeatureImportance):
    """Spearman r per ``(feature, response)`` pair, per model, per-seed.

    Consumed on seed-level tables (``"long_abs"`` / ``"long_rel"``).
    Aggregated across seeds via :class:`FisherZAggregator` (Rubin in
    Fisher-z space).
    """

    name: ClassVar[str] = "spearman"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs", "long_rel")

    def __init__(
        self,
        min_obs: int = MIN_SPEARMAN_N,
        aggregator: Optional[SeedAggregator] = None,
    ) -> None:
        self.min_obs = int(min_obs)
        self.aggregator = aggregator if aggregator is not None else FisherZAggregator()

    # ── single-seed fit ─────────────────────────────────────────────────────

    def _fit_one_seed(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str,
    ) -> Optional[pd.DataFrame]:
        """Run Spearman r for one seed slice and return per-feature rows
        ``(feature, r, p, n)`` or ``None`` if no feature was fittable."""
        if response not in df.columns:
            return None
        rows: list[dict] = []
        y_all = df[response].to_numpy(dtype=float)
        for f in feature_cols:
            if f == response or f not in df.columns:
                continue
            x_all = df[f].to_numpy(dtype=float)
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            n = int(mask.sum())
            if n < self.min_obs:
                continue
            xm = x_all[mask]
            ym = y_all[mask]
            if np.unique(xm).size < 2 or np.unique(ym).size < 2:
                continue
            r, p = sp_stats.spearmanr(xm, ym)
            rows.append({
                "feature": f,
                "r":       float(r),
                "p":       float(p),
                "n":       n,
            })
        if not rows:
            return None
        return pd.DataFrame(rows)

    # ── per-seed loop + Fisher-z combine ────────────────────────────────────

    def _aggregate(self, per_seed: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for feature, grp in per_seed.groupby("feature", dropna=False):
            per_seed_list = []
            for _, r in grp.iterrows():
                n_k = int(r["n"])
                if n_k < 4:
                    continue
                per_seed_list.append({
                    "estimate": float(r["r"]),
                    "se":       1.0 / float(np.sqrt(max(n_k - 3, 1))),
                    "r":        float(r["r"]),
                    "n":        n_k,
                    "p":        float(r["p"]),
                })
            agg = self.aggregator.combine(per_seed_list)
            rows.append({
                "feature":     feature,
                "r":           agg["mean"],
                "se":          agg["se"],
                "ci_low":      agg["ci_low"],
                "ci_high":     agg["ci_high"],
                "p":           agg["p"],
                "p_acat":      agg["p_acat"],
                "t":           agg["t"],
                "df":          agg["df"],
                "n_seeds":     agg["n_seeds"],
                "within_var":  agg["within_var"],
                "between_var": agg["between_var"],
                "n":           int(grp["n"].max()),
            })
        if not rows:
            return pd.DataFrame(
                columns=[
                    "feature", "r", "se", "ci_low", "ci_high",
                    "p", "p_acat", "t", "df", "n_seeds",
                    "within_var", "between_var", "n",
                ]
            )
        out = pd.DataFrame(rows)
        return out.reindex(
            out["r"].abs().sort_values(ascending=False, na_position="last").index
        ).reset_index(drop=True)

    def fit_per_model_response(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], SpearmanResult]:
        out: dict[tuple[str, str], SpearmanResult] = {}
        if "seed" in df.columns:
            seeds = sorted(s for s in df["seed"].dropna().unique())
        else:
            seeds = [None]

        for model in models:
            sub_m = df[df["model"] == model]
            for resp in responses:
                print(f"  SPR  | model={model:10s} response={resp}")
                per_seed_frames: list[pd.DataFrame] = []
                for seed in seeds:
                    if seed is None:
                        sub_s = sub_m
                        seed_val: object = float("nan")
                    else:
                        sub_s = sub_m[sub_m["seed"] == seed]
                        seed_val = seed
                    seed_df = self._fit_one_seed(sub_s, feature_cols, resp)
                    if seed_df is None:
                        continue
                    seed_df = seed_df.copy()
                    seed_df.insert(0, "seed", seed_val)
                    per_seed_frames.append(seed_df)
                if not per_seed_frames:
                    print("    skipped (insufficient data)")
                    continue
                per_seed = pd.concat(per_seed_frames, ignore_index=True)
                aggregated = self._aggregate(per_seed)
                if aggregated.empty:
                    print("    skipped (no feature after aggregation)")
                    continue
                out[(model, resp)] = SpearmanResult(
                    correlations=aggregated,
                    per_seed=per_seed,
                    n_obs=int(per_seed["n"].max()),
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
    ) -> dict[tuple[str, str], SpearmanResult]:
        return self.fit_per_model_response(df, feature_cols, models, responses)

    def importance_long(
        self,
        results: dict[tuple[str, str], SpearmanResult],
    ) -> pd.DataFrame:
        """Long-format rows of *aggregated* r for one (ne, kind, ratio) slice."""
        rows: list[dict] = []
        for (model, resp), result in results.items():
            for _, c in result.correlations.iterrows():
                rows.append({
                    "model":       model,
                    "response":    resp,
                    "feature":     c["feature"],
                    "r":           float(c["r"]) if pd.notna(c["r"]) else float("nan"),
                    "se":          float(c["se"]) if pd.notna(c["se"]) else float("nan"),
                    "ci_low":      float(c["ci_low"]) if pd.notna(c["ci_low"]) else float("nan"),
                    "ci_high":     float(c["ci_high"]) if pd.notna(c["ci_high"]) else float("nan"),
                    "p":           float(c["p"]) if pd.notna(c["p"]) else float("nan"),
                    "p_acat":      float(c["p_acat"]) if pd.notna(c.get("p_acat", float("nan"))) else float("nan"),
                    "t":           float(c["t"]) if pd.notna(c["t"]) else float("nan"),
                    "df":          float(c["df"]) if pd.notna(c["df"]) else float("nan"),
                    "n_seeds":     int(c["n_seeds"]),
                    "within_var":  float(c["within_var"]) if pd.notna(c["within_var"]) else float("nan"),
                    "between_var": float(c["between_var"]) if pd.notna(c["between_var"]) else float("nan"),
                    "n":           int(c["n"]),
                })
        return pd.DataFrame(rows)

    def per_seed_long(
        self,
        results: dict[tuple[str, str], SpearmanResult],
    ) -> pd.DataFrame:
        """Long-format rows of *per-seed* r for one (ne, kind, ratio) slice."""
        rows: list[dict] = []
        for (model, resp), result in results.items():
            for _, c in result.per_seed.iterrows():
                rows.append({
                    "model":    model,
                    "response": resp,
                    "feature":  c["feature"],
                    "seed":     c["seed"],
                    "r":        float(c["r"]),
                    "p":        float(c["p"]),
                    "n":        int(c["n"]),
                })
        return pd.DataFrame(rows)

    def save(
        self,
        results_by_key: dict[
            tuple[float, "str | float | None"],
            dict[tuple[str, str], SpearmanResult],
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

        # 2. Per-ratio HTML pivot of r ∈ [-1, +1] (columns: alpha × response × model).
        for ratio, pairs in grouped.items():
            sub_frames: list[pd.DataFrame] = []
            for alpha_label, results in pairs:
                df = self.importance_long(results)
                if df.empty:
                    continue
                df = df.copy()
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
                value_col="r",
                p_col="p",
                caption=(
                    f"Spearman rank correlation r "
                    f"(ratio={ratio}; rows = meta-feature, "
                    f"cols = alpha × response × model; cell = r ∈ [-1, +1]; "
                    f"* p<0.05, ** p<0.01)"
                ),
                cmap="RdBu_r",
                vmin=-1.0, vmax=1.0,
                diverging=True,
                fmt="{:+.3f}",
                alpha_col=alpha_col_for_html(pairs),
            )
