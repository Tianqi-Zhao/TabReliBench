"""Evaluation package: TabPFN/TabICL calibration analysis pipeline.

Public entry points:

  ExperimentSpec                  - frozen identifier for one (dataset_id, seed, ratio[, task])
  ArtifactStore                   - filesystem layout + pickle I/O for every artefact
  DatasetLoader                   - OpenML loading + canonical splits (train/test/context)
  PPDQuantileGrid                 - posterior-predictive-distribution helper

Metrics:

  BaseMetricsCalculator           - abstract calculator (one per task)
  RegressionMetricsCalculator     - regression calibration metrics
  DerivedRegressionMetric         - base class for derived/post-hoc regression metrics
  ClassificationMetricsCalculator - classification metrics (Brier, ECE, CMCE, …)
  ConcreteClassificationMetricsCalculator - deprecated alias of the above
  MetricsCalculator               - legacy alias for the regression class
"""
from .data import DatasetLoader, SplitData
from .preprocessing import FeaturePreprocessor, strategy_for
from .metrics import (
    BaseMetricsCalculator,
    BrierMetric,
    ClassificationMetric,
    ClassificationMetricsCalculator,
    ClassificationContext,
    ClasswiseECEMetric,
    CMCEMetric,
    ConcreteClassificationMetricsCalculator,  # deprecated alias
    CoverageMetric,
    CRPSMetric,
    DEFAULT_CLASSIFICATION_METRICS,
    DEFAULT_REGRESSION_METRICS,
    DerivedRegressionMetric,
    IntervalArrays,
    LogLossMetric,
    MetricOutput,
    MetricsCalculator,
    PITMetric,
    PinballMetric,
    RegressionContext,
    RegressionMetric,
    RegressionMetricsCalculator,
    RESPONSE_COLS,
    TotalAbsDevMetric,
    TopLabelMetric,
    upgrade_legacy_schema,
    WSCMetric,
)
from .ppd import PPDQuantileGrid, make_quantile_grid, quantile_grid_integral
from .spec import (
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TASKS,
    ExperimentSpec,
)
from .store import ArtifactStore, load_dataset_ids

__all__ = [
    "ArtifactStore",
    "BaseMetricsCalculator",
    "BrierMetric",
    "ClassificationContext",
    "ClassificationMetric",
    "ClassificationMetricsCalculator",
    "ClasswiseECEMetric",
    "CMCEMetric",
    "ConcreteClassificationMetricsCalculator",
    "CoverageMetric",
    "CRPSMetric",
    "DatasetLoader",
    "DEFAULT_CLASSIFICATION_METRICS",
    "DEFAULT_REGRESSION_METRICS",
    "DerivedRegressionMetric",
    "ExperimentSpec",
    "FeaturePreprocessor",
    "IntervalArrays",
    "LogLossMetric",
    "MetricOutput",
    "MetricsCalculator",
    "PITMetric",
    "PinballMetric",
    "PPDQuantileGrid",
    "make_quantile_grid",
    "quantile_grid_integral",
    "RegressionContext",
    "RegressionMetric",
    "RegressionMetricsCalculator",
    "RESPONSE_COLS",
    "SplitData",
    "TASK_CLASSIFICATION",
    "TASK_REGRESSION",
    "TASKS",
    "TotalAbsDevMetric",
    "TopLabelMetric",
    "WSCMetric",
    "load_dataset_ids",
    "strategy_for",
    "upgrade_legacy_schema",
]
