"""Regression calibration metrics.

Concrete :class:`RegressionMetric` subclasses
----------------------------------------------

* :class:`CoverageMetric`  — marginal coverage + coverage deviation scalars
  (``cov_dev_signed``, ``cov_abs_dev``, ``nominal``) + Winkler interval
  score + normalised interval width; writes the full :class:`IntervalArrays`
  bundle as per-instance output.

* :class:`WSCMetric`       — worst-slab coverage via split-half search;
  enforces minimum search support, preserves undefined coverage, and outputs
  ``worst_slab_coverage``, ``wsc_dev_signed``, ``wsc_abs_dev``;
  per-dataset only (no per-instance arrays).

* :class:`TotalAbsDevMetric` — derived metric (subclass of
  :class:`~evaluation.metrics.base.DerivedRegressionMetric`) that computes
  ``total_abs_dev = cov_abs_dev + wsc_abs_dev`` after the primary pass.

* :class:`PinballMetric`   — Pinball loss at the two quantile endpoints
  ``τ = α/2`` and ``τ = 1 − α/2`` of the central (1−α) interval;
  alpha-dependent.

* :class:`CRPSMetric`      — Continuous Ranked Probability Score via
  numerical integration over the PPD quantile grid; alpha-free.

* :class:`PITMetric`       — Probability Integral Transform uniformity
  (KS statistic / p-value + PIT-ECE + PIT-histogram L1); alpha-free.

Orchestrator
------------
:class:`RegressionMetricsCalculator` iterates over a configurable list
of :class:`RegressionMetric` instances (default:
:data:`DEFAULT_REGRESSION_METRICS`) and merges their output into the 2×2
nested schema documented in :mod:`evaluation.metrics.base`.

Primary metrics run first; :class:`DerivedRegressionMetric` subclasses
run in a second pass so they can read values written by primary metrics.

Pure helper functions
---------------------
:func:`pinball_at_quantile`, :func:`crps_per_row`,
:func:`pit_calibration_scalars` are exposed as module-level functions so
they can be imported and used independently (e.g. in notebooks).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Optional, Sequence

import numpy as np
import pandas as pd

from ..ppd import PPDQuantileGrid
from ..spec import TASK_REGRESSION
from .base import (
    BaseMetricsCalculator,
    DerivedRegressionMetric,
    MetricOutput,
    RegressionContext,
    RegressionMetric,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance interval bundle (regression / central PI only)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntervalArrays:
    """Five aligned (n_test,) arrays describing one alpha's interval."""

    lower:   np.ndarray   # central interval lower bound
    upper:   np.ndarray   # central interval upper bound
    width:   np.ndarray   # = upper - lower
    covered: np.ndarray   # int8: 1 if y_test in [lower, upper] else 0
    winkler: np.ndarray   # per-point Winkler interval-score contribution

    @classmethod
    def build(
        cls,
        lower: np.ndarray,
        upper: np.ndarray,
        y_test: np.ndarray,
        alpha: float,
    ) -> "IntervalArrays":
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        y     = np.asarray(y_test, dtype=np.float64)
        width = upper - lower
        covered = ((y >= lower) & (y <= upper)).astype(np.int8)
        winkler = (
            width
            + (2.0 / alpha) * np.maximum(lower - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - upper, 0.0)
        )
        return cls(
            lower=lower, upper=upper, width=width,
            covered=covered, winkler=winkler.astype(np.float64),
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions (can be used independently)
# ─────────────────────────────────────────────────────────────────────────────

def pinball_at_quantile(
    ppd: PPDQuantileGrid,
    y: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Per-row Pinball (quantile) loss at level *tau*.

    ``PL_τ(y, q̂) = max(τ · (y − q̂), (τ − 1) · (y − q̂))``

    Parameters
    ----------
    ppd:
        Posterior predictive distribution grid.
    y:
        True target values, shape ``(n_test,)``.
    tau:
        Quantile level in ``(0, 1)``.

    Returns
    -------
    np.ndarray of shape ``(n_test,)`` — per-row pinball loss.
    """
    y    = np.asarray(y, dtype=np.float64)
    qhat = ppd.quantile_at(tau)        # (n_test,)
    diff = y - qhat
    return np.where(diff >= 0.0, tau * diff, (tau - 1.0) * diff)


def crps_per_row(ppd: PPDQuantileGrid, y: np.ndarray) -> np.ndarray:
    """Per-row Continuous Ranked Probability Score (CRPS).

    Uses the quantile-decomposition identity:

        CRPS_i = 2 · ∫₀¹ PL_τ(y_i, F_i⁻¹(τ)) dτ

    The integral is approximated with the trapezoidal rule over the
    existing quantile grid (handles non-uniform spacing correctly).

    Parameters
    ----------
    ppd:
        Posterior predictive distribution grid.
    y:
        True target values, shape ``(n_test,)``.

    Returns
    -------
    np.ndarray of shape ``(n_test,)`` — per-row CRPS ≥ 0.
    """
    y      = np.asarray(y, dtype=np.float64)
    levels = ppd.levels                # (n_levels,)
    q_grid = ppd.ppd                   # (n_test, n_levels)

    diff = y[:, None] - q_grid         # (n_test, n_levels)
    pl   = np.where(
        diff >= 0.0,
        levels[None, :] * diff,
        (levels[None, :] - 1.0) * diff,
    )                                  # (n_test, n_levels)  — PL_τ per cell

    # NumPy 2 removed ``trapz`` in favour of ``trapezoid``; keep the metrics
    # runnable on both modern and older server environments.
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(pl, levels, axis=1)
    else:  # pragma: no cover - exercised only on NumPy < 2
        integral = np.trapz(pl, levels, axis=1)
    return 2.0 * integral


def pit_calibration_scalars(
    pit: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute PIT-based calibration scalars.

    Parameters
    ----------
    pit:
        Per-instance PIT values in ``(0, 1)``, shape ``(n_test,)``.
    n_bins:
        Number of equally spaced quantile levels at which to evaluate the
        empirical PIT CDF for PIT-ECE, and the number of equal-width bins for
        PIT-histogram L1.  The historical argument name is kept for API
        compatibility.

    Returns
    -------
    dict with keys ``pit_ks_stat``, ``pit_ks_pvalue``, ``pit_ece``, and
    ``pit_hist_l1``.
    """
    pit = np.asarray(pit, dtype=np.float64)
    n = len(pit)

    # KS test against Uniform(0, 1)
    if n == 0:
        ks_stat = float("nan")
        ks_pvalue = float("nan")
    else:
        try:
            from scipy import stats as _stats
            ks_stat, ks_pvalue = _stats.kstest(pit, "uniform")
            ks_stat   = float(ks_stat)
            ks_pvalue = float(ks_pvalue)
        except ImportError:
            # Fallback: Glivenko-Cantelli KS statistic (no p-value)
            sorted_pit = np.sort(pit)
            emp_cdf    = np.arange(1, n + 1) / n
            ks_stat    = float(np.max(np.abs(emp_cdf - sorted_pit)))
            ks_pvalue  = float("nan")

    # PIT-ECE / probabilistic calibration error: at equally spaced quantile
    # levels alpha_j, compare the empirical PIT CDF F_hat(alpha_j) with the
    # uniform CDF alpha_j.  This is the regression analogue of comparing
    # empirical frequency with predicted probability in classification ECE:
    #
    #   (1 / M) * sum_j |F_hat(alpha_j) - alpha_j|.
    #
    # It is a grid approximation to the 1-Wasserstein distance
    # integral_0^1 |F_hat(alpha) - alpha| d alpha.
    if n == 0 or n_bins < 2:
        pit_ece = float("nan")
        pit_hist_l1 = float("nan")
    else:
        sorted_pit = np.sort(pit)
        alpha = np.linspace(0.0, 1.0, n_bins + 2, dtype=np.float64)[1:-1]
        empirical_cdf = np.searchsorted(
            sorted_pit, alpha, side="right"
        ).astype(np.float64) / float(n)
        pit_ece = float(np.mean(np.abs(empirical_cdf - alpha)))

        # Equal-width PIT-histogram L1 distance from the discrete uniform
        # distribution.  Empty bins are deliberately retained and contribute
        # |0 - 1/B| to the distance.
        counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
        bin_probability = counts.astype(np.float64) / float(n)
        pit_hist_l1 = float(
            np.sum(np.abs(bin_probability - 1.0 / float(n_bins)))
        )

    return {
        "pit_ks_stat":   ks_stat,
        "pit_ks_pvalue": ks_pvalue,
        "pit_ece":       pit_ece,
        "pit_hist_l1":   pit_hist_l1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WSC helpers (minimum-support candidates, independent evaluation half)
# ─────────────────────────────────────────────────────────────────────────────

def _categorical_groups(xj: pd.Series, min_size: int) -> list[frozenset]:
    cats, counts = np.unique(xj.dropna().values, return_counts=True)
    groups: list[list] = [[c] for c in cats]
    sizes:  list[int]  = list(map(int, counts))
    while len(groups) > 1:
        min_idx = int(np.argmin(sizes))
        if sizes[min_idx] >= min_size:
            break
        other = [i for i in range(len(groups)) if i != min_idx]
        merge_idx = min(other, key=lambda i: sizes[i])
        merged      = groups[min_idx] + groups[merge_idx]
        merged_size = sizes[min_idx]  + sizes[merge_idx]
        keep   = [i for i in range(len(groups)) if i not in (min_idx, merge_idx)]
        groups = [groups[i] for i in keep] + [merged]
        sizes  = [sizes[i]  for i in keep] + [merged_size]
    return [frozenset(g) for g in groups]


_SlabSpec = tuple[str, frozenset] | tuple[str, float, float]


@dataclass(frozen=True)
class _WSCResult:
    coverage: float = float("nan")
    feature: int = -1
    search_n: int = 0
    eval_n: int = 0
    reason: str = "no_eligible_subgroup"


def _slab_mask(x: pd.Series, spec: _SlabSpec) -> np.ndarray:
    """Use identical membership rules for search, bin merging, and evaluation."""
    if spec[0] == "cat":
        return x.isin(spec[1]).to_numpy()
    _, lo, hi = spec
    values = x.to_numpy(dtype=np.float64, na_value=np.nan)
    return (values > lo) & (values <= hi)


def _numeric_groups(x: pd.Series, min_size: int, n_bins: int) -> list[_SlabSpec]:
    """Merge undersized quantile bins using search covariates only.

    Merge the smallest bin into its smaller neighbor; ties favor the lower
    neighbor. Keep the original outer bounds, including their nextafter padding.
    """
    values = x.dropna().to_numpy(dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    while len(edges) > 2:
        counts = [
            int(_slab_mask(x, ("num", lo, hi)).sum())
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        smallest = int(np.argmin(counts))
        if counts[smallest] >= min_size:
            break
        neighbors = [i for i in (smallest - 1, smallest + 1) if 0 <= i < len(counts)]
        neighbor = min(neighbors, key=lambda i: (counts[i], i))
        edges = np.delete(edges, max(smallest, neighbor))
    return [("num", lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]


def _worst_conditional_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    y_test: np.ndarray,
    X_test: pd.DataFrame,
    n_bins: int = 5,
) -> _WSCResult:
    """Select a supported slab on the first half, evaluate on the second.

    Implements the audited minimum-support policy. Both candidate types need
    at least max(1, floor(n_search / n_bins)) search observations. Coverage
    ties retain the first candidate, including ties at 1. Evaluation labels
    and support never influence selection. Undefined coverage stays NaN.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    covered = (y_test >= lower) & (y_test <= upper)
    mid = len(covered) // 2
    search, evaluation = X_test.iloc[:mid], X_test.iloc[mid:]
    min_group = max(1, mid // n_bins)
    best_coverage = float("inf")
    best_spec = None
    best_feature, best_count = -1, 0

    for j in range(search.shape[1]):
        x = search.iloc[:, j]
        valid = x.dropna()
        if len(valid) < 2 or valid.nunique() < 2:
            continue
        if pd.api.types.is_numeric_dtype(x):
            specs = _numeric_groups(x, min_group, n_bins)
        else:
            if valid.nunique() > 0.5 * len(valid):
                continue
            specs = [("cat", group) for group in _categorical_groups(x, min_group)]

        for spec in specs:
            mask = _slab_mask(x, spec)
            count = int(mask.sum())
            if count < min_group:
                continue
            coverage = float(covered[:mid][mask].mean())
            if coverage < best_coverage:
                best_coverage = coverage
                best_spec, best_feature, best_count = spec, j, count

    if best_spec is None:
        return _WSCResult()

    mask = _slab_mask(evaluation.iloc[:, best_feature], best_spec)
    count = int(mask.sum())
    return _WSCResult(
        coverage=float(covered[mid:][mask].mean()) if count else float("nan"),
        feature=best_feature,
        search_n=best_count,
        eval_n=count,
        reason="finite" if count else "empty_evaluation_subgroup",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Concrete RegressionMetric subclasses
# ─────────────────────────────────────────────────────────────────────────────

class CoverageMetric(RegressionMetric):
    """Marginal coverage, normalised width, and Winkler interval score.

    ``alpha_dependent = True`` — one result per alpha.

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``marginal_coverage``, ``nominal``, ``cov_dev_signed``, ``cov_abs_dev``,
    ``avg_length``, ``avg_width_norm``, ``interval_score``

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``lower``, ``upper``, ``width``, ``covered``, ``winkler``
    (all from :class:`IntervalArrays`)
    """
    name: ClassVar[str] = "coverage"
    alpha_dependent: ClassVar[bool] = True

    def compute(self, ctx: RegressionContext) -> MetricOutput:
        lower, upper = ctx.ppd.interval(ctx.alpha)
        iv = IntervalArrays.build(lower, upper, ctx.y_test, ctx.alpha)

        e     = ctx.eval_slice
        y_e   = ctx.y_test[e]
        y_std = max(float(np.std(y_e)), 1e-10)

        nominal    = 1.0 - ctx.alpha
        mc         = float(iv.covered[e].mean())
        cov_signed = mc - nominal

        return MetricOutput(
            per_dataset={
                "marginal_coverage": mc,
                "nominal":           nominal,
                "cov_dev_signed":    cov_signed,
                "cov_abs_dev":       abs(cov_signed),
                "avg_length":        float((upper[e] - lower[e]).mean()),
                "avg_width_norm":    float((upper[e] - lower[e]).mean() / y_std),
                "interval_score":    float(iv.winkler[e].mean()),
            },
            per_instance=iv.as_dict(),
        )


class WSCMetric(RegressionMetric):
    """Worst-slab conditional coverage (split-half search).

    ``alpha_dependent = True`` — one result per alpha.

    Uses ``wsc_min_support_v1``: undersized numeric bins are merged and
    both numeric and categorical candidates require minimum search support.
    No eligible slab or an empty evaluation slab yields NaN, including in
    the derived deviations. Perfect search coverage still selects a slab.

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``worst_slab_coverage``, ``wsc_dev_signed``, ``wsc_abs_dev``,
    ``worst_slab_feature_idx``, ``worst_slab_feature_name``, ``wsc_version``,
    ``wsc_search_n``, ``wsc_eval_n``, ``wsc_reason``.
    Feature index and support counts are diagnostics, excluded from analysis
    metric columns by ``_METADATA_SKIP``.

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    *(none)*
    """
    version: ClassVar[str] = "wsc_min_support_v1"
    name: ClassVar[str] = "worst_slab"
    alpha_dependent: ClassVar[bool] = True

    def compute(self, ctx: RegressionContext) -> MetricOutput:
        lower, upper = ctx.ppd.interval(ctx.alpha)
        result = _worst_conditional_coverage(
            lower, upper, ctx.y_test, ctx.X_test,
        )

        wcc, feat = result.coverage, result.feature

        nominal    = 1.0 - ctx.alpha
        wsc_signed = wcc - nominal if np.isfinite(wcc) else float("nan")
        wsc_abs    = abs(wsc_signed) if np.isfinite(wsc_signed) else float("nan")

        feat_name = ctx.feature_names[feat] if feat != -1 else None

        return MetricOutput(
            per_dataset={
                "worst_slab_coverage":      wcc,
                "wsc_dev_signed":           wsc_signed,
                "wsc_abs_dev":              wsc_abs,
                "worst_slab_feature_idx":   int(feat),
                "worst_slab_feature_name":  feat_name,
                "wsc_version":              self.version,
                "wsc_search_n":             result.search_n,
                "wsc_eval_n":               result.eval_n,
                "wsc_reason":               result.reason,
            },
            per_instance={},
        )


class PinballMetric(RegressionMetric):
    """Pinball loss at the two endpoints of the (1−α) central interval.

    ``alpha_dependent = True`` — one result per alpha.

    Computes PL at τ = α/2 (lower tail) and τ = 1 − α/2 (upper tail).

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``pinball_lower``, ``pinball_upper``, ``pinball_mean``
    (means computed on the eval half, consistent with ``marginal_coverage``)

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``pinball_lower_row``, ``pinball_upper_row``  (full test-set arrays)
    """
    name: ClassVar[str] = "pinball"
    alpha_dependent: ClassVar[bool] = True

    def compute(self, ctx: RegressionContext) -> MetricOutput:
        tau_lo  = ctx.alpha / 2.0
        tau_hi  = 1.0 - tau_lo
        pl_low  = pinball_at_quantile(ctx.ppd, ctx.y_test, tau_lo)
        pl_up   = pinball_at_quantile(ctx.ppd, ctx.y_test, tau_hi)
        pl_mean = 0.5 * (pl_low + pl_up)

        e = ctx.eval_slice
        return MetricOutput(
            per_dataset={
                "pinball_lower": float(pl_low[e].mean()),
                "pinball_upper": float(pl_up[e].mean()),
                "pinball_mean":  float(pl_mean[e].mean()),
            },
            per_instance={
                "pinball_lower_row": pl_low,
                "pinball_upper_row": pl_up,
            },
        )


class CRPSMetric(RegressionMetric):
    """Continuous Ranked Probability Score (alpha-free).

    Integrates over the entire PPD quantile grid so it summarises the
    full predictive distribution, not just a single interval.

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``crps_mean``

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``crps_row``
    """
    name: ClassVar[str] = "crps"
    alpha_dependent: ClassVar[bool] = False

    def compute(self, ctx: RegressionContext) -> MetricOutput:
        crps_row = crps_per_row(ctx.ppd, ctx.y_test)
        return MetricOutput(
            per_dataset={"crps_mean": float(crps_row.mean())},
            per_instance={"crps_row": crps_row},
        )


class PITMetric(RegressionMetric):
    """Probability Integral Transform uniformity (alpha-free).

    Tests whether the PIT values F_i(y_i) are uniformly distributed,
    which is a necessary condition for calibration.

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``pit_ks_stat``, ``pit_ks_pvalue``, ``pit_ece``, ``pit_hist_l1``

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``pit``
    """
    name: ClassVar[str] = "pit"
    alpha_dependent: ClassVar[bool] = False

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins

    def compute(self, ctx: RegressionContext) -> MetricOutput:
        pit = ctx.ppd.cdf_at(ctx.y_test)
        return MetricOutput(
            per_dataset=pit_calibration_scalars(pit, self.n_bins),
            per_instance={"pit": pit},
        )


def point_accuracy(y, prediction):
    """Canonical point metrics shared by full and R²-only evaluation."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or y.shape != prediction.shape or y.size == 0:
        raise ValueError("Targets and point predictions must be matching nonempty vectors")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Targets and point predictions must be finite")
    residual = y - prediction
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else float("nan")
    return {"r2": r2, "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual)))}, residual


class PointAccuracyMetric(RegressionMetric):
    """Point-prediction accuracy metrics (alpha-free).

    Requires ``ctx.point_pred`` (the model's scalar point prediction,
    e.g. the posterior mean/median).  Skipped gracefully when
    ``point_pred`` is ``None`` (legacy records that pre-date this field).

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``r2``, ``rmse``, ``mae``

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``residual``  (y_test − point_pred, signed)
    """
    name: ClassVar[str] = "point_accuracy"
    alpha_dependent: ClassVar[bool] = False

    def compute(self, ctx: RegressionContext) -> MetricOutput:
        if ctx.point_pred is None:
            return MetricOutput(per_dataset={}, per_instance={})

        scalars, residual = point_accuracy(ctx.y_test, ctx.point_pred)
        return MetricOutput(per_dataset=scalars, per_instance={"residual": residual})


# ─────────────────────────────────────────────────────────────────────────────
# Derived metrics
# ─────────────────────────────────────────────────────────────────────────────

class TotalAbsDevMetric(DerivedRegressionMetric):
    """Total absolute deviation = cov_abs_dev + wsc_abs_dev.

    Requires :class:`CoverageMetric` and :class:`WSCMetric` to have run
    first (both must be in the same metrics list).

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``total_abs_dev``
    """
    name: ClassVar[str] = "total_abs_dev"
    alpha_dependent: ClassVar[bool] = True

    def derive(self, per_dataset: dict, alpha: float) -> dict:
        cov = per_dataset.get("cov_abs_dev", float("nan"))
        wsc = per_dataset.get("wsc_abs_dev", float("nan"))
        return {"total_abs_dev": cov + wsc}


# Default response columns analyzed by the dataset-level pipeline
# (LME / Spearman / Chatterjee / RF / Univariate).
#
# Mirrors the metric coverage of ``eval_results/plot_dataset_pattern_heatmap.py``
# (REG_PREDICTION + REG_CALIBRATION_FREE + REG_INTERVAL), minus the two
# raw coverage rates (``marginal_coverage`` / ``worst_slab_coverage``)
# which are exact transforms of the signed deviations and therefore
# redundant — see ``evaluation/metrics/directions.py``.
#
# Two notes on what gets analyzed where (see RelTable / RelBiasTable for
# the details):
#
#   * The default ``eval_rel`` / ``eval_rel_long`` is axis-inclusive:
#     proper / monotone quality metrics keep the usual "positive = better
#     than peers" meaning, while single-axis lower-better metrics
#     (``cov_abs_dev`` / ``wsc_abs_dev`` / ``total_abs_dev`` /
#     ``avg_width_norm``) are included with
#     "positive = smaller / more favourable on this axis". These axis
#     responses are useful for pattern analysis, but not honest standalone
#     model-ranking targets because coverage and width trade off.
#
#   * Bidirectional metrics (``cov_dev_signed`` / ``wsc_dev_signed``)
#     remain excluded from axis-relative tables. Configured responses are
#     available in ``eval_rel_bias`` / ``eval_rel_long_bias`` as raw peer
#     z-scores, where positive means the raw response is larger than peers,
#     not better (for signed deviations: more over-covering / less
#     under-covering).
#
#   * Honest model ranking lives on the proper scoring rules
#     (``interval_score`` / ``pinball_*`` / ``crps_mean`` for intervals,
#     ``r2`` / ``rmse`` / ``mae`` for point prediction, ``pit_*`` for
#     distributional calibration), all of which survive into ``eval_rel``.
RESPONSE_COLS: list[str] = [
    # ── Prediction accuracy (alpha-free) ─────────────────────────────────
    "r2",                                       # higher-better, scale-free
    "rmse", "mae", "crps_mean",                 # lower-better, scale-dependent
    # ── Distributional calibration (alpha-free, scale-free) ──────────────
    "pit_ece", "pit_hist_l1", "pit_ks_stat",
    # ── Coverage-axis deviations (alpha-dependent; trade-off incomplete) ─
    "cov_dev_signed", "wsc_dev_signed",         # bidirectional, optimum at 0
    "cov_abs_dev", "wsc_abs_dev", "total_abs_dev",
    # ── Interval quality (alpha-dependent) ───────────────────────────────
    "interval_score",                           # proper score: lower-better
    "pinball_lower", "pinball_mean", "pinball_upper",  # proper scores
    # ── Width-axis (alpha-dependent; trade-off incomplete) ───────────────
    # ``avg_length`` is intentionally omitted from dataset-level response
    # analysis because it is in raw target units and not comparable across
    # datasets. The metric is still computed in metrics PKLs for compatibility.
    "avg_width_norm",
]


# ─────────────────────────────────────────────────────────────────────────────
# Default metric list (the extension point)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REGRESSION_METRICS: list[RegressionMetric] = [
    CoverageMetric(),
    WSCMetric(),
    TotalAbsDevMetric(),
    PinballMetric(),
    CRPSMetric(),
    PITMetric(),
    PointAccuracyMetric(),
]


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RegressionMetricsCalculator(BaseMetricsCalculator):
    """Compute all regression calibration metrics for one prediction record.

    Iterates over a configurable list of :class:`RegressionMetric` objects
    (default: :data:`DEFAULT_REGRESSION_METRICS`) and merges their
    :class:`MetricOutput` results into the 2×2 nested schema.

    Parameters
    ----------
    alphas:
        Miscoverage levels; each must lie in ``(0, 1)``.
    metrics:
        List of :class:`RegressionMetric` instances to evaluate.
        Defaults to :data:`DEFAULT_REGRESSION_METRICS`.

    Output schema (per ``compute_for_record``)
    ------------------------------------------
    See :mod:`evaluation.metrics.base` for the full schema description.
    """

    task: ClassVar[str] = TASK_REGRESSION

    def __init__(
        self,
        alphas: list[float],
        metrics: Optional[Sequence[RegressionMetric]] = None,
    ) -> None:
        if any(not (0.0 < a < 1.0) for a in alphas):
            raise ValueError("alphas must all lie in (0, 1)")
        self.alphas  = list(map(float, alphas))
        self.metrics = (
            list(metrics) if metrics is not None
            else list(DEFAULT_REGRESSION_METRICS)
        )

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _merge_into(
        dest_pd: dict,
        dest_pi: dict,
        output: MetricOutput,
        metric_name: str,
    ) -> None:
        """Merge MetricOutput into destination dicts, asserting no key clash."""
        for key in output.per_dataset:
            if key in dest_pd:
                raise KeyError(
                    f"Metric '{metric_name}' tried to write per_dataset key "
                    f"'{key}' which was already produced by a previous metric."
                )
            dest_pd[key] = output.per_dataset[key]

        for key in output.per_instance:
            if key in dest_pi:
                raise KeyError(
                    f"Metric '{metric_name}' tried to write per_instance key "
                    f"'{key}' which was already produced by a previous metric."
                )
            dest_pi[key] = output.per_instance[key]

    def _run_primary_alpha(
        self,
        ctx: RegressionContext,
        alpha: float,
    ) -> tuple[dict, dict]:
        """Run primary and derived alpha-dependent metrics for one alpha.

        Returns ``(per_dataset, per_instance)`` dicts with all keys merged.
        Primary metrics are run first, then :class:`DerivedRegressionMetric`
        instances receive the accumulated ``per_dataset`` and write derived
        keys in a second pass.
        """
        ad_pd: dict = {}
        ad_pi: dict = {}

        for m in self.metrics:
            if not m.alpha_dependent or isinstance(m, DerivedRegressionMetric):
                continue
            out = m.compute(ctx)
            self._merge_into(ad_pd, ad_pi, out, m.name)

        for m in self.metrics:
            if not m.alpha_dependent or not isinstance(m, DerivedRegressionMetric):
                continue
            extra = m.derive(ad_pd, alpha)
            self._merge_into(ad_pd, ad_pi, MetricOutput(extra, {}), m.name)

        return ad_pd, ad_pi

    # ── Public API ──────────────────────────────────────────────────────

    def compute_for_alpha(
        self,
        ppd: "PPDQuantileGrid",
        y_test: np.ndarray,
        X_test: "pd.DataFrame",
        feature_names: list,
        alpha: float,
    ) -> "tuple[dict, IntervalArrays]":
        """Compute alpha-dependent metrics for a single alpha.

        Convenience method for the simulation runner, which builds its own
        PPD and iterates over alphas without a full prediction record.

        Parameters
        ----------
        ppd:
            Posterior predictive distribution grid.
        y_test:
            True target values (full test set).
        X_test:
            Test feature matrix.
        feature_names:
            Column names of *X_test*.
        alpha:
            Miscoverage level.

        Returns
        -------
        per_dataset : dict
            All scalar outputs for this alpha (primary + derived).
        intervals : :class:`IntervalArrays`
            Full-test-set interval bundle (useful for saving raw outputs).
        """
        n_eval = len(y_test) // 2
        ctx = RegressionContext(
            ppd=ppd, y_test=y_test, X_test=X_test,
            feature_names=feature_names,
            alpha=alpha,
            eval_slice=slice(n_eval, None),
        )
        ad_pd, _ad_pi = self._run_primary_alpha(ctx, alpha)

        lower, upper = ppd.interval(alpha)
        intervals = IntervalArrays.build(lower, upper, y_test, alpha)
        return ad_pd, intervals

    def compute_for_record(self, record: dict) -> dict:
        """Compute every registered metric for one prediction PKL record.

        Returns the dict to pickle as the metrics PKL.
        """
        if record.get("output_kind") == "point":
            if record.get("regression_metrics") != ["r2"]:
                raise ValueError("Point-only records must explicitly request regression_metrics=['r2']")
            scalars, _ = point_accuracy(record["y_test"], record["point_pred"])
            result = {key: record.get(key) for key in (
                "model", "dataset_id", "seed", "ratio", "n_total", "n_train",
                "n_test", "n_context", "n_features", "feature_names",
            )}
            result.update(
                task=TASK_REGRESSION, alphas=[], alpha_dependent={},
                alpha_free={"per_dataset": {"r2": scalars["r2"]}, "per_instance": {}},
                y_test=np.asarray(record["y_test"]),
                distribution_status="not_requested",
                metric_status={"r2": "ok" if np.isfinite(scalars["r2"]) else "undefined_constant_target"},
            )
            return result
        ppd           = PPDQuantileGrid.from_record(record)
        y_test        = np.asarray(record["y_test"])
        X_test        = record["X_test"]
        feature_names = record["feature_names"]
        n_eval        = len(y_test) // 2

        # point_pred may be absent in legacy records
        raw_pp = record.get("point_pred")
        point_pred = np.asarray(raw_pp, dtype=np.float64) if raw_pp is not None else None

        # ── Alpha-free metrics (computed once over the full test set) ──
        ctx_af = RegressionContext(
            ppd=ppd, y_test=y_test, X_test=X_test,
            feature_names=feature_names,
            alpha=None,
            eval_slice=slice(None),
            point_pred=point_pred,
        )
        af_pd: dict = {}
        af_pi: dict = {}
        for m in self.metrics:
            if m.alpha_dependent:
                continue
            out = m.compute(ctx_af)
            self._merge_into(af_pd, af_pi, out, m.name)

        alpha_free = {"per_dataset": af_pd, "per_instance": af_pi}

        # ── Alpha-dependent metrics (once per alpha) ───────────────────
        alpha_dep: dict[float, dict] = {}
        for alpha in self.alphas:
            ctx_ad = RegressionContext(
                ppd=ppd, y_test=y_test, X_test=X_test,
                feature_names=feature_names,
                alpha=alpha,
                eval_slice=slice(n_eval, None),
            )
            ad_pd, ad_pi = self._run_primary_alpha(ctx_ad, alpha)
            alpha_dep[alpha] = {"per_dataset": ad_pd, "per_instance": ad_pi}

        return {
            "task":            TASK_REGRESSION,
            "model":           record.get("model"),
            "dataset_id":      record.get("dataset_id"),
            "seed":            record.get("seed"),
            "ratio":           record.get("ratio"),
            "n_total":         record.get("n_total"),
            "n_train":         record.get("n_train"),
            "n_test":          record.get("n_test"),
            "n_context":       record.get("n_context"),
            "n_features":      record.get("n_features"),
            "feature_names":   feature_names,
            "alphas":          list(self.alphas),
            "alpha_dependent": alpha_dep,
            "alpha_free":      alpha_free,
            "y_test":          y_test,
        }


# Back-compat alias
MetricsCalculator = RegressionMetricsCalculator
