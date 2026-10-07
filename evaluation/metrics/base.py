"""Shared abstractions for calibration metrics.

Schema (single source of truth)
--------------------------------
Every metrics PKL written by this package uses the following 2×2 nested
structure. Position encodes semantics — no separate registry is needed.

Regression metrics PKL::

    {
        "task": "regression",
        "model", "dataset_id", "seed", "ratio",
        "n_total", "n_train", "n_test", "n_context", "n_features",
        "feature_names",  "alphas",  "y_test",

        "alpha_dependent": {
            alpha: {
                "per_dataset":  { scalar_key: float, ... },
                "per_instance": { array_key:  np.ndarray(n_test,), ... },
            }, ...
        },

        "alpha_free": {
            "per_dataset":  { scalar_key: float, ... },
            "per_instance": { array_key:  np.ndarray(n_test,), ... },
        },
    }

Classification metrics PKL::

    {
        "task": "classification",
        "model", "dataset_id", "seed", "ratio",
        "n_total", "n_train", "n_test", "n_context", "n_features",
        "feature_names", "classes_", "y_test",

        "alpha_dependent": {},          # always empty for classification

        "alpha_free": {
            "per_dataset":  { scalar_key: float, ... },
            "per_instance": { array_key:  np.ndarray(n_test,), ... },
        },
    }

Metric strategy interface
-------------------------
:class:`RegressionMetric` and :class:`ClassificationMetric` are the
extension points.  Each concrete subclass owns a semantically coherent
group of keys (scalars + arrays) and declares whether it requires a
specific ``alpha`` (``alpha_dependent = True``) or works on the full
predictive distribution (``alpha_dependent = False``).

For metrics that derive scalars from values already computed by other
metrics, use :class:`DerivedRegressionMetric` instead.  Its
:meth:`~DerivedRegressionMetric.derive` method receives the accumulated
``per_dataset`` dict and runs in a second pass after all primary metrics.

The orchestrators (:class:`RegressionMetricsCalculator`,
:class:`ClassificationMetricsCalculator`) iterate over a
configurable list of :class:`Metric` objects and merge their
:class:`MetricOutput` results into the 2×2 nested dict.

Legacy adapter
--------------
:func:`upgrade_legacy_schema` transparently converts old-format dicts
(``metrics[alpha]`` / ``intervals[alpha]``) into the new schema.
Call it in :meth:`ArtifactStore.load_metrics` to keep historical PKLs
readable without any migration step.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from ..spec import TASK_CLASSIFICATION, TASK_REGRESSION
from ..ppd import PPDQuantileGrid


# ─────────────────────────────────────────────────────────────────────────────
# Context objects — input bundles passed to every Metric.compute()
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionContext:
    """All inputs a regression metric might need."""
    ppd: PPDQuantileGrid
    y_test: np.ndarray
    X_test: pd.DataFrame
    feature_names: list[str]
    alpha: Optional[float] = None       # None ⇒ alpha-free phase
    eval_slice: slice = field(default_factory=lambda: slice(None))
    point_pred: Optional[np.ndarray] = None  # model's point prediction (mean/median)


@dataclass
class ClassificationContext:
    """All inputs a classification metric might need."""
    proba: np.ndarray       # (N, K)  float64, rows sum to ~1
    classes_: np.ndarray    # (K,)
    y_idx: np.ndarray       # (N,)  integer indices into classes_
    n_bins: int = 15


# ─────────────────────────────────────────────────────────────────────────────
# MetricOutput — one metric's two-path output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricOutput:
    """Two-path output from a single :class:`Metric`.

    ``per_dataset`` holds scalar summaries (float / int / str / None).
    ``per_instance`` holds aligned arrays of length ``n_test``.
    Either dict may be empty when that path is not produced.
    """
    per_dataset: dict
    per_instance: dict

    @classmethod
    def empty(cls) -> "MetricOutput":
        return cls({}, {})


# ─────────────────────────────────────────────────────────────────────────────
# Metric abstract base classes
# ─────────────────────────────────────────────────────────────────────────────

class RegressionMetric(abc.ABC):
    """Abstract base for one regression calibration metric.

    A concrete subclass should override:

    * ``name``               — short identifier (used in logs / debug).
    * ``alpha_dependent``    — ``True`` means the metric needs ``ctx.alpha``;
                               the orchestrator will put its output under
                               ``alpha_dependent[α]``.  ``False`` → ``alpha_free``.
    * :meth:`compute`        — return a :class:`MetricOutput`.

    Keys written into ``per_dataset`` / ``per_instance`` must not collide
    with those of other metrics registered in the same list; the
    orchestrator asserts this at runtime.
    """
    name: ClassVar[str]
    alpha_dependent: ClassVar[bool]

    @abc.abstractmethod
    def compute(self, ctx: RegressionContext) -> MetricOutput:
        raise NotImplementedError


class DerivedRegressionMetric(RegressionMetric, abc.ABC):
    """Derives per_dataset scalars from already-computed scalars.

    Subclasses implement :meth:`derive` instead of :meth:`compute`.  The
    orchestrator (:class:`~evaluation.metrics.RegressionMetricsCalculator`)
    runs all :class:`DerivedRegressionMetric` instances in a second pass,
    **after** all primary :class:`RegressionMetric` instances have
    populated ``per_dataset``.  This allows derived metrics to read values
    written by primary metrics without duplicating computation.

    The default ``alpha_dependent = True`` covers the common case; override
    in subclasses if a derived metric should be alpha-free.
    """
    alpha_dependent: ClassVar[bool] = True

    def compute(self, ctx: RegressionContext) -> MetricOutput:
        """Not called by the orchestrator; delegates to :meth:`derive`."""
        return MetricOutput(per_dataset={}, per_instance={})

    @abc.abstractmethod
    def derive(self, per_dataset: dict, alpha: float) -> dict:
        """Return new ``per_dataset`` keys derived from already-computed values.

        Parameters
        ----------
        per_dataset:
            The accumulated ``per_dataset`` dict after all primary metrics
            have run for this alpha.
        alpha:
            The miscoverage level for this pass.

        Returns
        -------
        dict
            New key/value pairs to merge into ``per_dataset``.  Keys must
            not clash with existing ones (the orchestrator asserts this).
        """
        raise NotImplementedError


class ClassificationMetric(abc.ABC):
    """Abstract base for one classification calibration metric.

    Classification has no ``alpha`` dimension, so all metrics are
    implicitly alpha-free.
    """
    name: ClassVar[str]
    alpha_dependent: ClassVar[bool] = False

    @abc.abstractmethod
    def compute(self, ctx: ClassificationContext) -> MetricOutput:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# BaseMetricsCalculator (abstract orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class BaseMetricsCalculator(abc.ABC):
    """Abstract calibration-metrics calculator.

    Subclasses bind ``task`` to one of ``'regression'`` /
    ``'classification'`` and implement :meth:`compute_for_record`, which
    consumes one prediction-PKL record (as produced by
    :class:`evaluation.pipelines.PredictionPipeline`) and returns the
    payload to pickle as that run's metrics PKL.
    """
    task: ClassVar[str]

    @abc.abstractmethod
    def compute_for_record(self, record: dict) -> dict:
        """Return the dict payload to pickle as the metrics PKL."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Legacy schema adapter
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_legacy_schema(d: dict) -> dict:
    """Convert an old-format metrics PKL dict to the new 2×2 nested schema.

    Old format (pre-refactor)::

        { "metrics": {alpha: {...}}, "intervals": {alpha: {...}}, ... }

    New format::

        { "alpha_dependent": {alpha: {"per_dataset": ..., "per_instance": ...}},
          "alpha_free": {"per_dataset": {}, "per_instance": {}}, ... }

    Already-upgraded dicts are returned unchanged.  The conversion is
    done **in-place** on the dict (the caller owns it after ``pickle.load``),
    and the same dict is also returned for convenience.
    """
    if "alpha_dependent" in d or "alpha_free" in d:
        return d

    metrics_legacy   = d.pop("metrics",   {})
    intervals_legacy = d.pop("intervals", {})

    ad: dict = {}
    for a, m in metrics_legacy.items():
        ad[a] = {
            "per_dataset":  m,
            "per_instance": intervals_legacy.get(a, {}),
        }

    d["alpha_dependent"] = ad
    d["alpha_free"]      = {"per_dataset": {}, "per_instance": {}}
    return d
