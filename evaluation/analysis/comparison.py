"""Cross-model calibration comparisons.

All "between-model" analyses live here so they're reusable from notebooks
without going through the CLI script. Counterpart to ``LMEAnalyzer`` /
``RFImportanceAnalyzer`` which answer the within-model "what features
predict miscalibration" question — this answers the "which model is more
calibrated" question.

Uses the long DataFrame produced by :class:`MetricsTable`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..metrics import RESPONSE_COLS


# Default column set used by ``model_comparison`` / ``per_dataset_table``.
# Mirrors the per_dataset keys produced by CoverageMetric, WSCMetric, and
# TotalAbsDevMetric, plus the ECE scalar from the classification side.
DEFAULT_METRIC_COLS: list[str] = [
    "marginal_coverage", "cov_abs_dev",
    "worst_slab_coverage", "wsc_abs_dev",
    "avg_width_norm", "interval_score",
    "cov_dev_signed", "wsc_dev_signed", "total_abs_dev",
    "calibration_error",
]


@dataclass
class WilcoxonResult:
    """Result of one paired Wilcoxon between two models."""
    model_a: str
    model_b: str
    n_shared: int
    n_paired_finite: int
    mean_diff: float
    wilcoxon_p: float

    def as_dict(self) -> dict:
        return {
            "model_a": self.model_a, "model_b": self.model_b,
            "n_shared": self.n_shared,
            "n_paired_finite": self.n_paired_finite,
            "mean_diff": self.mean_diff,
            "wilcoxon_p": self.wilcoxon_p,
        }


# ─────────────────────────────────────────────────────────────────────────────

class ModelComparator:
    """Cross-model calibration comparisons on a long metrics DataFrame.

    The input ``df`` must have at minimum the columns produced by
    :class:`MetricsTable`:

      ``dataset_id``, ``model``, ``seed``, ``ratio``, ``alpha``,
      ``marginal_coverage``, ``worst_slab_coverage``,
      ``cov_dev_signed``, ``wsc_dev_signed``, ``total_abs_dev``,
      ``avg_width_norm``, ``interval_score``,
      ``calibration_error`` (alias of ``total_abs_dev``).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        metric_cols: list[str] = DEFAULT_METRIC_COLS,
    ) -> None:
        if "calibration_error" not in df.columns and "total_abs_dev" in df.columns:
            df = df.assign(calibration_error=df["total_abs_dev"])
        self.df = df
        self.metric_cols = [c for c in metric_cols if c in df.columns]

    # ── Aggregate tables ────────────────────────────────────────────────────

    def model_comparison(self) -> pd.DataFrame:
        """Mean / median / std per (model, alpha) across datasets and seeds."""
        return (
            self.df.groupby(["model", "alpha"])[self.metric_cols]
            .agg(["mean", "median", "std"])
            .round(4)
        )

    def model_comparison_simple(self) -> pd.DataFrame:
        """Mean per (model, alpha) — used in the human-readable summary."""
        return (
            self.df.groupby(["model", "alpha"])[self.metric_cols]
            .mean().round(4)
        )

    def per_dataset_table(self) -> pd.DataFrame:
        """Mean per (dataset_id, model, alpha) across seeds and ratios."""
        return (
            self.df.groupby(["dataset_id", "model", "alpha"])[self.metric_cols]
            .mean().round(4).reset_index()
        )

    def coverage_bias(self) -> pd.DataFrame:
        """Per (model, alpha): fraction over- / under-covered + signed dev."""
        rows = []
        for (model, alpha), grp in self.df.groupby(["model", "alpha"]):
            mean_dev = grp.groupby("dataset_id")["marginal_coverage"].mean()
            nominal  = 1.0 - alpha
            rows.append({
                "model": model, "alpha": alpha,
                "frac_over_covered":  round(float((mean_dev > nominal).mean()), 3),
                "frac_under_covered": round(float((mean_dev < nominal).mean()), 3),
                "mean_signed_dev":    round(float((mean_dev - nominal).mean()), 4),
            })
        return pd.DataFrame(rows).sort_values(["alpha", "model"])

    def wsc_availability(self) -> pd.DataFrame:
        """How many rows have a finite WSC per (model, alpha)."""
        diag = (
            self.df.assign(_ok=np.isfinite(self.df["worst_slab_coverage"]))
            .groupby(["model", "alpha"])["_ok"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "n_finite", "count": "n_total"})
        )
        diag["finite_frac"] = (diag["n_finite"] / diag["n_total"]).round(4)
        return diag

    def interval_efficiency(self) -> pd.DataFrame:
        """Sharpness vs calibration trade-off per (model, alpha).

        ``avg_width_norm`` (= width / std(y_eval)) and ``interval_score``
        are reported with mean and median across datasets. Smaller is
        better on all three columns.
        """
        cols = [c for c in ("avg_width_norm", "interval_score", "cov_abs_dev")
                if c in self.df.columns]
        agg = (
            self.df.groupby(["model", "alpha"])[cols]
            .agg(["mean", "median"]).round(4)
        )
        agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
        return (
            agg.reset_index()
            .sort_values(["alpha", "avg_width_norm_median"])
        )

    # ── Per-alpha rankings ──────────────────────────────────────────────────

    def model_wins(
        self,
        alpha: float,
        metric: str = "calibration_error",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """For each dataset, which model has the lowest metric?

        Datasets where every model is NaN (no estimable WSC at all) are
        skipped; otherwise NaN cells are ignored when picking the winner.

        Returns ``(tally, best_per_dataset)``.
        """
        sub = self._at_alpha(alpha)
        mean_per = (
            sub.groupby(["dataset_id", "model"])[metric]
            .mean().dropna().reset_index()
        )
        if mean_per.empty:
            return pd.DataFrame(columns=["n_datasets_best"]), mean_per
        best = mean_per.loc[mean_per.groupby("dataset_id")[metric].idxmin()]
        tally = best["model"].value_counts().rename("n_datasets_best").to_frame()
        return tally, best

    def worst_datasets(
        self,
        alpha: float,
        n: int = 10,
        metric: str = "calibration_error",
    ) -> pd.DataFrame:
        """Top-n worst (dataset, model) pairs ranked by *metric*.

        Pairs with NaN metric (case-B WSC remainder) are dropped so the
        ranking only reflects estimable cases.
        """
        sub = self._at_alpha(alpha)
        return (
            sub.groupby(["dataset_id", "model"])[metric]
            .mean().dropna().reset_index()
            .sort_values(metric, ascending=False)
            .head(n)
        )

    def per_model_extremes(
        self,
        alpha: float,
        n: int = 5,
        metric: str = "calibration_error",
    ) -> dict[str, dict[str, dict]]:
        """Best / worst N datasets per model on *metric* at *alpha*."""
        sub = self._at_alpha(alpha)
        out: dict[str, dict[str, dict]] = {}
        for model in sorted(sub["model"].unique()):
            mean_err = (
                sub[sub["model"] == model]
                .groupby("dataset_id")[metric]
                .mean().dropna().sort_values()
            )
            out[model] = {
                "best":  mean_err.head(n).round(4).to_dict(),
                "worst": mean_err.tail(n).round(4).to_dict(),
            }
        return out

    # ── Pairwise tests ──────────────────────────────────────────────────────

    def paired_wilcoxon(
        self,
        alpha: float,
        model_a: str,
        model_b: str,
        metric: str = "calibration_error",
    ) -> WilcoxonResult:
        """Wilcoxon signed-rank test on shared datasets at *alpha*.

        Positive ``mean_diff`` means ``model_a`` is *worse* than ``model_b``.
        """
        sub = self._at_alpha(alpha)
        mean_a = sub[sub["model"] == model_a].groupby("dataset_id")[metric].mean()
        mean_b = sub[sub["model"] == model_b].groupby("dataset_id")[metric].mean()
        shared = mean_a.index.intersection(mean_b.index)
        diff = (mean_a.loc[shared] - mean_b.loc[shared]).values
        finite = np.isfinite(diff)
        diff = diff[finite]
        n_eff = int(finite.sum())
        if n_eff < 5:
            return WilcoxonResult(model_a, model_b, len(shared), n_eff,
                                  float("nan"), float("nan"))
        try:
            _stat, p = sp_stats.wilcoxon(diff, alternative="two-sided")
        except Exception:
            p = float("nan")
        return WilcoxonResult(
            model_a=model_a, model_b=model_b,
            n_shared=len(shared), n_paired_finite=n_eff,
            mean_diff=float(np.mean(diff)),
            wilcoxon_p=float(p) if p == p else float("nan"),
        )

    def all_pairwise_wilcoxon(
        self,
        alpha: float,
        metric: str = "calibration_error",
    ) -> pd.DataFrame:
        sub = self._at_alpha(alpha)
        models = sorted(sub["model"].unique())
        rows = []
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                rows.append(
                    self.paired_wilcoxon(alpha, ma, mb, metric=metric).as_dict()
                )
        return pd.DataFrame(rows)

    def ranking_agreement(
        self,
        alpha: float,
        metric: str = "calibration_error",
    ) -> list[dict]:
        """Spearman ρ between models on per-dataset *metric*.

        Returns one dict per model pair — empty list if fewer than 5
        shared datasets are available for any pair.
        """
        sub = self._at_alpha(alpha)
        pivot = (
            sub.groupby(["dataset_id", "model"])[metric]
            .mean().unstack("model")
        )
        out: list[dict] = []
        if pivot.shape[1] < 2:
            return out
        for i, ma in enumerate(pivot.columns):
            for mb in list(pivot.columns)[i + 1:]:
                both = pivot[[ma, mb]].dropna()
                if len(both) < 5:
                    continue
                rho, p = sp_stats.spearmanr(both[ma], both[mb])
                out.append({
                    "model_a": str(ma), "model_b": str(mb),
                    "n": int(len(both)),
                    "spearman_rho": float(rho),
                    "spearman_p":   float(p),
                })
        return out

    # ── Internal ────────────────────────────────────────────────────────────

    def _at_alpha(self, alpha: float) -> pd.DataFrame:
        a = float(alpha)
        sub = self.df[self.df["alpha"] == a]
        if sub.empty:
            available = sorted(self.df["alpha"].unique())
            raise ValueError(
                f"No rows at alpha={a!r}; available alphas: {available}"
            )
        return sub
