"""Calibration-metrics sub-package.

Public API
----------
All symbols that were previously importable from ``evaluation.metrics``
(the old single-file module) are re-exported here unchanged, so every
existing call-site continues to work without modification::

    from evaluation.metrics import (
        BaseMetricsCalculator,
        DerivedRegressionMetric,
        RegressionMetricsCalculator,
        ClassificationMetricsCalculator,
        ConcreteClassificationMetricsCalculator,  # deprecated alias
        IntervalArrays,
        RESPONSE_COLS,
        MetricsCalculator,          # legacy alias
        upgrade_legacy_schema,
    )

Extension points
----------------
To add a new regression metric, subclass :class:`RegressionMetric`,
implement :meth:`compute`, and append an instance to
:data:`DEFAULT_REGRESSION_METRICS`.

To add a derived regression metric (one that reads values produced by
other metrics), subclass :class:`DerivedRegressionMetric`, implement
:meth:`derive`, and append an instance to
:data:`DEFAULT_REGRESSION_METRICS`.

To add a new classification metric, subclass :class:`ClassificationMetric`
and append an instance to :data:`DEFAULT_CLASSIFICATION_METRICS`.
"""

# ── Base layer ────────────────────────────────────────────────────────────────
from .base import (
    BaseMetricsCalculator,
    ClassificationContext,
    ClassificationMetric,
    DerivedRegressionMetric,
    MetricOutput,
    RegressionContext,
    RegressionMetric,
    upgrade_legacy_schema,
)

# ── Regression layer ──────────────────────────────────────────────────────────
from .regression import (
    CoverageMetric,
    CRPSMetric,
    crps_per_row,
    DEFAULT_REGRESSION_METRICS,
    IntervalArrays,
    MetricsCalculator,           # back-compat alias for RegressionMetricsCalculator
    PITMetric,
    pinball_at_quantile,
    PinballMetric,
    pit_calibration_scalars,
    PointAccuracyMetric,
    RegressionMetricsCalculator,
    RESPONSE_COLS,
    TotalAbsDevMetric,
    WSCMetric,
)

# ── Classification layer ──────────────────────────────────────────────────────
from .classification import (
    BrierMetric,
    CLASSIFICATION_RESPONSE_COLS,
    ClassificationAccuracyMetric,
    ClassificationMetricsCalculator,
    ClasswiseECEMetric,
    CMCEMetric,
    ConcreteClassificationMetricsCalculator,  # deprecated alias
    DEFAULT_CLASSIFICATION_METRICS,
    LogLossMetric,
    TopLabelMetric,
)

# ── Direction registry (peer-relative z, ranking) ─────────────────────────────
from .directions import (
    AXIS_RELATIVE_METRICS,
    BIDIRECTIONAL_METRICS,
    JOINT_ONLY_METRICS,
    METRIC_ASCENDING,
    PAIR_DELTA_ABSOLUTE_METRICS,
    SCALE_DEPENDENT_METRICS,
    TRADE_OFF_METRICS,
    metric_ascending,
    metric_direction,
    metric_scale,
)

__all__ = [
    # base
    "BaseMetricsCalculator",
    "ClassificationContext",
    "ClassificationMetric",
    "DerivedRegressionMetric",
    "MetricOutput",
    "RegressionContext",
    "RegressionMetric",
    "upgrade_legacy_schema",
    # regression
    "CoverageMetric",
    "CRPSMetric",
    "crps_per_row",
    "DEFAULT_REGRESSION_METRICS",
    "IntervalArrays",
    "MetricsCalculator",
    "PITMetric",
    "pinball_at_quantile",
    "PinballMetric",
    "pit_calibration_scalars",
    "PointAccuracyMetric",
    "RegressionMetricsCalculator",
    "RESPONSE_COLS",
    "TotalAbsDevMetric",
    "WSCMetric",
    # classification
    "BrierMetric",
    "CLASSIFICATION_RESPONSE_COLS",
    "ClassificationAccuracyMetric",
    "ClassificationMetricsCalculator",
    "ClasswiseECEMetric",
    "CMCEMetric",
    "ConcreteClassificationMetricsCalculator",
    "DEFAULT_CLASSIFICATION_METRICS",
    "LogLossMetric",
    "TopLabelMetric",
    # direction registry
    "AXIS_RELATIVE_METRICS",
    "BIDIRECTIONAL_METRICS",
    "JOINT_ONLY_METRICS",
    "METRIC_ASCENDING",
    "PAIR_DELTA_ABSOLUTE_METRICS",
    "SCALE_DEPENDENT_METRICS",
    "TRADE_OFF_METRICS",
    "metric_ascending",
    "metric_direction",
    "metric_scale",
]
