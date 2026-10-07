"""Chatterjee ξ correlation, ``ξ(feature → response)``, per pair.

One analyzer per file under :mod:`evaluation.analysis.univariate`.

ξ ∈ [0, 1] (asymptotically) is a recently-introduced rank correlation
(Chatterjee, 2021) that captures *any* functional dependence — including
non-monotonic ones — and equals 0 iff the two variables are independent.

Two practical caveats this analyzer takes care of:

* **ξ is asymmetric** (``ξ(X, Y) ≠ ξ(Y, X)``). For our use-case we want
  to know how informative each ``feature`` is *about* the ``response``,
  so we only compute and report ``ξ(feature → response)``.

* **x ties** matter. The Chatterjee statistic ranks observations *by x*
  and ties on x have to be broken; :func:`scipy.stats.chatterjeexi`
  breaks them with an internal RNG, which makes the result
  non-reproducible across calls. When ``tie_jitter > 0`` (default) and
  ``x`` contains ties, this analyzer adds tiny deterministic uniform
  noise of scale ``tie_jitter * (1 + std(x))`` *before* calling scipy
  so the answer is reproducible run-to-run. Set ``tie_jitter=0.0`` to
  fall back to scipy's internal random tie-breaking.

Per-seed two-stage fitting
--------------------------
The analyzer consumes seed-level tables (``"long_abs"`` and ``"long_rel"``):
for each ``(model, response, feature)`` triple it computes ξ
*independently per seed*, then combines across seeds via
:class:`evaluation.analysis._aggregation.RubinAggregator` so the reported
SE and p-value reflect both within-fit and between-seed uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aggregation import RubinAggregator, SeedAggregator
from ..base import (
    _alpha_to_cell,
    alpha_col_for_html,
    DatasetAnalyzer,
    FeatureImportance,
    group_keys_by_ratio,
    InputKind,
    render_feature_pivot_html,
)


@dataclass
class ChatterjeeResult:
    """Per-(model, response) aggregated + per-seed ξ correlations.

    ``correlations`` is the cross-seed *aggregated* DataFrame::

        feature, xi, se, ci_low, ci_high, p, t, df,
        n_seeds, within_var, between_var, n

    ``per_seed`` is the raw per-seed DataFrame::

        seed, feature, xi, p, n
    """

    correlations: pd.DataFrame
    per_seed:     pd.DataFrame
    n_obs:        int


class ChatterjeeAnalyzer(DatasetAnalyzer, FeatureImportance):
    """Chatterjee ξ correlation, ``ξ(feature → response)``, per pair.

    Consumed on seed-level tables: ``eval_long`` (kind=``"long_abs"``)
    and ``eval_rel_long`` (kind=``"long_rel"``). For each
    ``(model, response, feature)`` it fits ξ once per seed then
    aggregates across seeds (default: :class:`RubinAggregator`).

    Output written by :meth:`save`:

    * ``summary.csv`` — long: ``(ratio, alpha, model, response, feature,
      xi, se, ci_low, ci_high, p, t, df, n_seeds, within_var,
      between_var, n)``. ``xi`` and ``p`` columns are the
      cross-seed-aggregated values (kept under their old names so the
      heatmap / topk plot scripts continue to work unchanged).
    * ``summary_per_seed.csv`` — long: ``(ratio, alpha, model, response,
      feature, seed, xi, p, n)`` — raw per-seed fits, for diagnostics
      and forest-style visualisations.
    * ``summary_ratio_<r>.html`` — feature × (response, model) pivot of
      aggregated ``xi`` with sequential ``viridis`` colour scale on
      ``[0, 1]``.
    """

    name: ClassVar[str] = "chatterjee"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs", "long_rel")

    def __init__(
        self,
        y_continuous: bool = True,
        method: str = "asymptotic",
        min_obs: int = 10,
        tie_jitter: float = 1e-10,
        random_state: int = 0,
        aggregator: Optional[SeedAggregator] = None,
    ) -> None:
        self.y_continuous = bool(y_continuous)
        self.method       = str(method)
        self.min_obs      = int(min_obs)
        self.tie_jitter   = float(tie_jitter)
        self.random_state = int(random_state)
        self.aggregator   = aggregator if aggregator is not None else RubinAggregator()

    # ── single-seed fit ─────────────────────────────────────────────────────

    def _fit_one_seed(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str,
    ) -> Optional[pd.DataFrame]:
        """Run Chatterjee ξ for one seed slice and return per-feature rows
        ``(feature, xi, p, n)`` or ``None`` if no feature was fittable."""
        if response not in df.columns:
            return None
        rng = np.random.default_rng(self.random_state)
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
            xm, ym = x_all[mask], y_all[mask]
            if np.std(xm) < 1e-12 or np.std(ym) < 1e-12:
                continue

            xm = self._maybe_break_ties(xm, rng)
            res = sp_stats.chatterjeexi(
                xm, ym,
                y_continuous=self.y_continuous,
                method=self.method,
            )
            rows.append({
                "feature": f,
                "n":       n,
                "xi":      float(res.statistic),
                "p":       float(res.pvalue),
            })
        if not rows:
            return None
        return pd.DataFrame(rows)

    # ── per-seed loop + Rubin combine ───────────────────────────────────────

    def _aggregate(self, per_seed: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-seed ξ across seeds for every feature.

        Each seed's asymptotic null SE is ``sqrt(2 / (5 n_k))`` (Chatterjee
        2021, continuous-y case). The configured aggregator combines
        ``(xi_k, se_k)`` into a single row per feature.
        """
        rows: list[dict] = []
        for feature, grp in per_seed.groupby("feature", dropna=False):
            per_seed_list = []
            for _, r in grp.iterrows():
                n_k = int(r["n"])
                if n_k <= 0:
                    continue
                se_k = float(np.sqrt(2.0 / (5.0 * n_k)))
                per_seed_list.append({
                    "estimate": float(r["xi"]),
                    "se":       se_k,
                    "r":        float(r["xi"]),
                    "n":        n_k,
                    "p":        float(r["p"]),
                })
            agg = self.aggregator.combine(per_seed_list)
            rows.append({
                "feature":     feature,
                "xi":          agg["mean"],
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
                    "feature", "xi", "se", "ci_low", "ci_high",
                    "p", "p_acat", "t", "df", "n_seeds",
                    "within_var", "between_var", "n",
                ]
            )
        return (
            pd.DataFrame(rows)
            .sort_values("xi", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    def fit_per_model_response(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], ChatterjeeResult]:
        out: dict[tuple[str, str], ChatterjeeResult] = {}
        if "seed" in df.columns:
            seeds = sorted(s for s in df["seed"].dropna().unique())
        else:
            seeds = [None]

        for model in models:
            sub_m = df[df["model"] == model]
            for resp in responses:
                print(f"  XI   | model={model:10s} response={resp}")
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
                out[(model, resp)] = ChatterjeeResult(
                    correlations=aggregated,
                    per_seed=per_seed,
                    n_obs=int(per_seed["n"].max()),
                )
        return out

    def _maybe_break_ties(
        self, x: np.ndarray, rng: np.random.Generator,
    ) -> np.ndarray:
        """Add tiny uniform noise to ``x`` if it has ties."""
        if self.tie_jitter <= 0.0 or len(np.unique(x)) == len(x):
            return x
        scale = self.tie_jitter * (1.0 + float(np.std(x)))
        return x + rng.uniform(-scale, scale, size=x.shape)

    # ── DatasetAnalyzer interface ────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], ChatterjeeResult]:
        return self.fit_per_model_response(df, feature_cols, models, responses)

    def importance_long(
        self,
        results: dict[tuple[str, str], ChatterjeeResult],
    ) -> pd.DataFrame:
        """Long-format rows of *aggregated* ξ for one (ne, kind, ratio) slice."""
        rows: list[dict] = []
        for (model, resp), result in results.items():
            for _, c in result.correlations.iterrows():
                rows.append({
                    "model":       model,
                    "response":    resp,
                    "feature":     c["feature"],
                    "xi":          float(c["xi"]) if pd.notna(c["xi"]) else float("nan"),
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
        results: dict[tuple[str, str], ChatterjeeResult],
    ) -> pd.DataFrame:
        """Long-format rows of *per-seed* ξ for one (ne, kind, ratio) slice."""
        rows: list[dict] = []
        for (model, resp), result in results.items():
            for _, c in result.per_seed.iterrows():
                rows.append({
                    "model":    model,
                    "response": resp,
                    "feature":  c["feature"],
                    "seed":     c["seed"],
                    "xi":       float(c["xi"]),
                    "p":        float(c["p"]),
                    "n":        int(c["n"]),
                })
        return pd.DataFrame(rows)

    def save(
        self,
        results_by_key: dict[
            tuple[float, "str | float | None"],
            dict[tuple[str, str], ChatterjeeResult],
        ],
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        grouped = group_keys_by_ratio(results_by_key)

        # 1. Long aggregated summary CSV (all ratios × alphas).
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

        # 2. Per-ratio HTML pivot of ξ ∈ [0, 1] (columns: alpha × response × model).
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
                value_col="xi",
                caption=(
                    f"Chatterjee ξ(feature → response) "
                    f"(ratio={ratio}; rows = meta-feature, "
                    f"cols = alpha × response × model; cell = ξ ∈ [0, 1])"
                ),
                cmap="viridis",
                vmin=0.0, vmax=1.0,
                diverging=False,
                fmt="{:.3f}",
                alpha_col=alpha_col_for_html(pairs),
            )
