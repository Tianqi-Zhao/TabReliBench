"""Metric ranking direction registry.

Single source of truth for "what ranking semantics does this metric have?".
Used by:

* peer-relative z-scoring in :mod:`evaluation.analysis.tables`
  (sign-flip so that positive z = better than peers across all
  metric families);
* rank / norm-excess plots in :mod:`plots.results`;
* heatmap colour-direction in :mod:`eval_results.plot_dataset_pattern_heatmap`.

The taxonomy
------------
Every metric used in the project falls into one of three top-level
categories.  The third has a sub-category.

1. **Lower-better** — monotone, optimum at ``-∞``.  E.g. ``rmse``,
   ``log_loss``, ``crps_mean``, ``interval_score``, ``pit_ece``,
   classification ECE / MCE.  ``METRIC_ASCENDING[m] = True``.

2. **Higher-better** — monotone, optimum at ``+∞``.  E.g. ``r2``,
   ``accuracy``, ``macro_f1``, ``roc_auc_ovr_macro``.
   ``METRIC_ASCENDING[m] = False``.

3. **Trade-off incomplete** — describes one axis of a known trade-off,
   so single-direction ranking is misleading.  Concretely for
   regression intervals: coverage and width trade off — pushing
   intervals wider always improves coverage, so ranking a model by
   coverage alone (or width alone) is gameable.  All members listed in
   :data:`TRADE_OFF_METRICS`; ``METRIC_ASCENDING[m] = None``.

   3a. **Bidirectional sub-case** — when the trade-off axis additionally
       has its optimum at the *interior* (typically ``0``), so neither
       "lower" nor "higher" is uniformly closer to it.  E.g.
       ``cov_dev_signed`` (positive = over-coverage, negative =
       under-coverage; **both** signs are biased).  Listed in
       :data:`BIDIRECTIONAL_METRICS`, which is a strict subset of
       :data:`TRADE_OFF_METRICS`.

For honest model ranking, use proper scoring rules instead
(``interval_score`` / ``pinball_*`` / ``crps_mean`` — these penalise
both axes of the coverage-width trade-off simultaneously).

All trade-off-incomplete metrics report ``None`` (and
``metric_direction == 0``); the bidirectional sub-case is surfaced only
in error messages and downstream documentation when the extra
"interior optimum" caveat matters.

A note on independence
~~~~~~~~~~~~~~~~~~~~~~
In principle, "bidirectional" and "trade-off incomplete" are
independent properties — a bidirectional metric could exist without a
trade-off partner.  In this codebase, however, every bidirectional
metric is *also* trade-off incomplete (the signed coverage deviations
describe only the coverage axis), so the subset relationship holds.

What was deleted
----------------
``marginal_coverage`` and ``worst_slab_coverage`` used to be marked
"joint-only".  They have been removed from the registry entirely because
they are exact transforms of the signed deviations:

    marginal_coverage    = cov_dev_signed + (1 − α)
    worst_slab_coverage  = wsc_dev_signed + (1 − α)

The data columns are still computed by ``CoverageMetric`` / ``WSCMetric``
for compatibility with old PKLs and external scripts that reference them
directly, but the analysis pipeline no longer treats them as analysis
targets — the signed deviations carry the same information without the
``α``-shift.

Convention summary
------------------
``METRIC_ASCENDING[m]``
    * ``True``  → lower-better (monotone).
    * ``False`` → higher-better (monotone).
    * ``None``  → no single-direction ranking; see
      :data:`BIDIRECTIONAL_METRICS` vs :data:`TRADE_OFF_METRICS` for the
      sub-category.

``metric_direction(m)``
    Maps the same registry to ``{+1, -1, 0}``.  ``0`` covers all
    direction-less cases (both ``None`` entries and unknown metrics).
    Convenient for sign-flipping peer-relative z-scores.

``JOINT_ONLY_METRICS``
    Legacy alias kept for backward compat — equal to
    :data:`TRADE_OFF_METRICS` (which is the full set of metrics whose
    direction is ``None``, including the bidirectional sub-case).  All
    existing callers that use it as a "skip these" filter continue to
    work; the only thing that changed is the set now correctly includes
    the trade-off-incomplete metrics that used to claim a (misleading)
    direction.
"""
from __future__ import annotations


METRIC_ASCENDING: dict[str, bool | None] = {
    # ─── REGRESSION ────────────────────────────────────────────────────────

    # Prediction (alpha-free)
    "r2":                   False,  # higher better, scale-free
    "rmse":                 True,   # lower better, scale-dependent
    "mae":                  True,   # lower better, scale-dependent
    "crps_mean":            True,   # proper score: penalises sharpness & calibration

    # Distributional calibration (alpha-free, scale-free; no
    # coverage-width trade-off, so direction is well-defined)
    "pit_ece":              True,
    "pit_hist_l1":          True,
    "pit_ks_stat":          True,

    # Coverage-axis metrics (alpha-dependent) — all trade-off incomplete:
    # they describe only how often the interval covers, not how wide it
    # had to be to achieve that coverage. Direction is undefined.
    "cov_dev_signed":       None,   # bidirectional, optimum at 0
    "wsc_dev_signed":       None,   # bidirectional, optimum at 0
    "cov_abs_dev":          None,   # single-axis
    "wsc_abs_dev":          None,   # single-axis
    "total_abs_dev":        None,   # single-axis (sum of two coverage gaps)

    # Width-axis metrics (alpha-dependent) — also single-axis; can be
    # made trivially small by under-covering, so no honest ranking
    # without the coverage side.
    "avg_length":           None,
    "avg_width_norm":       None,

    # Proper interval scoring (alpha-dependent) — internalise the
    # coverage-width trade-off, so direction is well-defined.
    "interval_score":       True,   # Winkler score
    "pinball_lower":        True,
    "pinball_mean":         True,
    "pinball_upper":        True,

    # ─── CLASSIFICATION ────────────────────────────────────────────────────
    # (classification has no analog of the coverage-width trade-off;
    #  ECE / MCE measure calibration which isn't mechanically traded off
    #  against accuracy, and proper scoring rules cover both axes.)

    # Prediction (scale-free)
    "accuracy":             False,
    "macro_f1":             False,
    "roc_auc_ovr_macro":    False,

    # Proper scoring (penalise both accuracy and calibration)
    "brier_score":          True,
    "log_loss":             True,

    # Confidence / classwise / cumulative calibration (lower better)
    "confidence_ece_em":    True,
    "confidence_ece_ew":    True,
    "confidence_mce_em":    True,
    "confidence_mce_ew":    True,
    "classwise_ece_em":     True,
    "classwise_ece_ew":     True,
    "classwise_mce_em":     True,
    "classwise_mce_ew":     True,
    "cmce_em":              True,
    "cmce_ew":              True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pair-delta form: absolute (raw) difference vs relative difference.
#
# ``PairDeltaTable`` (evaluation/analysis/tables.py) builds the between-model
# response for one metric. Its default is the *relative* delta
# ``direction * (M_b - M_a) / M_a``. For non-negative metrics whose baseline
# value ``M_a`` can approach 0 (a tiny-scale target, or a near-perfect
# baseline model), that division explodes — e.g. a baseline RMSE ~1e-4 sends
# the relative delta to ~-1000, and the R²-scored permutation importance then
# amplifies it further.
#
# For metrics listed here we use the *absolute* difference
# ``direction * (M_b - M_a)`` instead, which is well-behaved because these
# metrics are scale-free and bounded (e.g. the r2 difference stays in
# ~[-0.7, +3]). Members MUST be scale-free + bounded so the raw difference is
# comparable across datasets.
#
# Calibration / proper-scoring metrics are eligible too when they are bounded
# and scale-free: ``brier_score`` ∈ [0, 2] is included so a near-perfect
# baseline (Brier → 0) doesn't blow up the relative form. ``crps_mean`` is
# deliberately *excluded* — it is in target units (scale-dependent), so it
# keeps the relative form, like ``rmse``.
# ─────────────────────────────────────────────────────────────────────────────

PAIR_DELTA_ABSOLUTE_METRICS: frozenset[str] = frozenset({
    "r2",                   # regression prediction accuracy (scale-free, ≤1)
    "accuracy",             # classification prediction accuracy (∈ [0, 1])
    "macro_f1",             # classification (∈ [0, 1])
    "roc_auc_ovr_macro",    # classification (∈ [0, 1])
    "brier_score",          # classification proper score (∈ [0, 2]); bounded → absolute pair-delta despite scale-dependent raw agg
})


# ─────────────────────────────────────────────────────────────────────────────
# Categorisation of the ``None``-direction metrics.
#
# TRADE_OFF_METRICS is the full set of metrics where
# ``METRIC_ASCENDING[m] is None`` — every direction-undefined metric in
# the codebase falls into the coverage-width trade-off.
#
# BIDIRECTIONAL_METRICS ⊂ TRADE_OFF_METRICS is the sub-case where the
# trade-off axis additionally has its optimum at the interior (signed
# deviations: both positive and negative signs are biased).  See the
# module docstring for the full taxonomy.
# ─────────────────────────────────────────────────────────────────────────────

TRADE_OFF_METRICS: frozenset[str] = frozenset({
    # Coverage axis — abs / total deviations
    "cov_abs_dev",
    "wsc_abs_dev",
    "total_abs_dev",
    # Coverage axis — signed deviations (also bidirectional)
    "cov_dev_signed",
    "wsc_dev_signed",
    # Width axis
    "avg_length",
    "avg_width_norm",
})

AXIS_RELATIVE_METRICS: frozenset[str] = frozenset({
    # Single-axis lower-better quantities.  These are not honest standalone
    # model-ranking targets, but they are useful for peer-relative pattern
    # analysis when labelled as axis improvements rather than overall quality.
    "cov_abs_dev",
    "wsc_abs_dev",
    "total_abs_dev",
    "avg_width_norm",
})

BIDIRECTIONAL_METRICS: frozenset[str] = frozenset({
    "cov_dev_signed",
    "wsc_dev_signed",
})

# Subset invariant (see module docstring).
assert BIDIRECTIONAL_METRICS <= TRADE_OFF_METRICS, (
    "BIDIRECTIONAL_METRICS must be a subset of TRADE_OFF_METRICS"
)
assert AXIS_RELATIVE_METRICS <= TRADE_OFF_METRICS, (
    "AXIS_RELATIVE_METRICS must be a subset of TRADE_OFF_METRICS"
)
assert AXIS_RELATIVE_METRICS.isdisjoint(BIDIRECTIONAL_METRICS), (
    "AXIS_RELATIVE_METRICS and BIDIRECTIONAL_METRICS must stay disjoint"
)

# Legacy alias — see module docstring. Equal to TRADE_OFF_METRICS (the
# full set of direction-undefined metrics).  Kept so existing callers
# that import ``JOINT_ONLY_METRICS`` continue to work and automatically
# pick up the trade-off metrics.
JOINT_ONLY_METRICS: frozenset[str] = TRADE_OFF_METRICS

# Sanity check at import time: the registry and the trade-off set agree.
_none_in_registry = frozenset(
    m for m, asc in METRIC_ASCENDING.items() if asc is None
)
assert _none_in_registry == TRADE_OFF_METRICS, (
    "METRIC_ASCENDING None entries must match TRADE_OFF_METRICS; "
    f"missing: {_none_in_registry - TRADE_OFF_METRICS}; "
    f"extra: {TRADE_OFF_METRICS - _none_in_registry}"
)
del _none_in_registry


def metric_ascending(metric: str) -> bool:
    """Return the ascending flag for *metric*.

    Raises
    ------
    KeyError
        If the metric is not registered.
    ValueError
        If the metric has no single-direction ranking — either
        bidirectional (optimum at 0) or trade-off incomplete (single axis
        of a known trade-off).  The exception message reports which
        sub-category the metric belongs to.
    """
    if metric not in METRIC_ASCENDING:
        raise KeyError(
            f"Metric {metric!r} is not in METRIC_ASCENDING. "
            "Add it to evaluation/metrics/directions.py."
        )
    asc = METRIC_ASCENDING[metric]
    if asc is None:
        if metric in TRADE_OFF_METRICS:
            bidir_note = (
                " It is also bidirectional — the optimum is at 0, so "
                "both positive and negative values indicate bias."
                if metric in BIDIRECTIONAL_METRICS else ""
            )
            raise ValueError(
                f"Metric {metric!r} is trade-off incomplete: it describes "
                "one axis of the coverage-width trade-off and ranking by "
                "it alone is misleading because the other axis can "
                f"compensate.{bidir_note} Use a proper scoring rule "
                "(interval_score / pinball_* / crps_mean) for honest "
                "model ranking."
            )
        raise ValueError(
            f"Metric {metric!r} has no single-direction ranking "
            "and is not categorised."
        )
    return asc


def metric_direction(metric: str) -> int:
    """Return ``+1`` (higher-better) / ``-1`` (lower-better) / ``0`` (no direction).

    The ``0`` return value covers both bidirectional and trade-off-
    incomplete metrics, plus any metric not in the registry — callers
    iterating over a mixed bag can skip them without exception handling.
    Use :func:`metric_ascending` for strict lookup that distinguishes
    the sub-categories.
    """
    asc = METRIC_ASCENDING.get(metric)
    if asc is None:
        return 0
    return -1 if asc else +1


# ─────────────────────────────────────────────────────────────────────────────
# Response (metric) category + scale taxonomy.
#
# Single source of truth for "which semantic bucket does this response belong
# to?" and "is it expressed in the target's raw units?".  Consumed by the RF
# fit-quality summary, and folded into the plotting groupings
# (``plot_dataset_pattern_heatmap``, ``plot_pairwise_corr_scatter._category``) and the
# feature-selection categories (``selected_features.RESPONSE_CATEGORIES_BY_TASK_KIND``
# is derived from this map).
#
#   * prediction   — point/accuracy quality.
#   * proper_score — proper scoring rules that penalise both accuracy and
#     calibration (regression: crps / interval_score / pinball_*; classification:
#     brier_score / log_loss).  This is the "prediction + calibration" bucket.
#   * calibration  — distributional / coverage / probability calibration.
#
# ``scale_dependent`` metrics should not be raw-aggregated across datasets.
# Regression members are in the target's raw units (magnitude ∝ σ_d).  Classification
# proper scores ``brier_score`` / ``log_loss`` are included too: they are not in
# target units, but their raw values are still not meaningfully comparable across
# heterogeneous datasets (class count, imbalance, difficulty), so plots and
# ``summary_long`` expose only rank / norm-excess for them.  Everything else is
# scale-free (a bounded probability, a ratio, or per-dataset normalized like
# ``r2`` / ``avg_width_norm``).
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_CATEGORY_METRICS: dict[str, dict[str, list[str]]] = {
    "regression": {
        "prediction":   ["r2", "rmse", "mae"],
        "proper_score": ["crps_mean", "interval_score",
                         "pinball_lower", "pinball_mean", "pinball_upper"],
        "calibration":  ["pit_ece", "pit_hist_l1", "pit_ks_stat",
                         "cov_abs_dev", "cov_dev_signed",
                         "wsc_abs_dev", "wsc_dev_signed",
                         "total_abs_dev", "avg_width_norm", "avg_length"],
    },
    "classification": {
        "prediction":   ["accuracy", "macro_f1", "roc_auc_ovr_macro"],
        "proper_score": ["brier_score", "log_loss"],
        "calibration":  ["confidence_ece_em", "confidence_ece_ew",
                         "confidence_mce_em", "confidence_mce_ew",
                         "classwise_ece_em", "classwise_ece_ew",
                         "classwise_mce_em", "classwise_mce_ew",
                         "cmce_em", "cmce_ew"],
    },
}

CATEGORY_ORDER: tuple[str, ...] = (
    "prediction", "proper_score", "calibration", "other",
)

# Metrics whose raw values must not be pooled across datasets (see comment
# above).  Consumed by :func:`metric_scale`, :mod:`plots.results`, and the RF
# fit-quality summary in :mod:`evaluation.analysis.base`.
SCALE_DEPENDENT_METRICS: frozenset[str] = frozenset({
    "rmse", "mae", "crps_mean", "interval_score",
    "pinball_lower", "pinball_mean", "pinball_upper", "avg_length",
    "brier_score", "log_loss",
})


def metric_category(metric: str) -> str:
    """Return the response category for *metric* — one of :data:`CATEGORY_ORDER`,
    or ``"other"`` if unregistered.

    Task-agnostic on purpose: metric names are unique across tasks (``accuracy``
    is classification-only, ``rmse`` regression-only), and a metric's category is
    intrinsic to the metric — it does not depend on the model task.  In
    dataset-level analysis the regressor/LME always *regress a continuous metric*
    on the meta-features, so there is no per-task category split to thread here.
    """
    for task_map in RESPONSE_CATEGORY_METRICS.values():
        for cat, metrics in task_map.items():
            if metric in metrics:
                return cat
    return "other"


def metric_scale(metric: str) -> str:
    """Return ``"scale_dependent"`` or ``"scale_free"``.

    Scale-dependent metrics must not be raw-aggregated across datasets; use
    rank / norm-excess (or per-dataset tables) instead.  See
    :data:`SCALE_DEPENDENT_METRICS`.
    """
    return "scale_dependent" if metric in SCALE_DEPENDENT_METRICS else "scale_free"
