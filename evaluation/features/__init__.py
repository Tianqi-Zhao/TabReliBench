"""Meta-feature extraction primitives.

Two abstract bases:

* :class:`DatasetFeatureGroup` returns a flat ``dict[str, float]`` for one
  (dataset, seed) — used by ``DatasetFeatureExtractor``.
* :class:`InstanceFeatureGroup` returns a per-row ``pd.DataFrame`` for one
  (dataset, model, alpha) — used by ``InstanceFeatureExtractor``.

Both are simple stateless classes with a ``name``, declared ``feature_names``
and a ``compute`` method, so the extractor can run them independently and
report which features came from which group.

Dataset-level context hierarchy::

    DatasetFeatureContext              (shared fields for all tasks)
    ├── RegressionDatasetFeatureContext    (+ y_mean, y_std, z)
    └── ClassificationDatasetFeatureContext (+ y_int, classes, class_counts, n_classes)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Dataset level
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetFeatureContext:
    """Shared pre-computed scratchpad for all tasks.

    Built once per (X_train, y_train, seed) by ``DatasetFeatureExtractor`` so
    every group can read what it needs without recomputing standardisation
    or NaN-dropping.

    Task-specific fields live in the subclasses:
    :class:`RegressionDatasetFeatureContext` and
    :class:`ClassificationDatasetFeatureContext`.
    """
    X_df:        pd.DataFrame
    y:           np.ndarray
    seed:        int
    n:           int
    d:           int
    X_num:       pd.DataFrame
    d_num:       int
    X_num_clean: pd.DataFrame
    y_clean:     np.ndarray
    n_clean:     int
    X_sc:        np.ndarray             # standardized X_num_clean (or empty)


@dataclass
class RegressionDatasetFeatureContext(DatasetFeatureContext):
    """Context for regression tasks.

    Adds continuous-target statistics used exclusively by regression groups.
    """
    y_mean: float = 0.0
    y_std:  float = 1.0
    z:      np.ndarray = None           # type: ignore[assignment]  # (y - mean) / std


@dataclass
class ClassificationDatasetFeatureContext(DatasetFeatureContext):
    """Context for classification tasks.

    Pre-computes label-encoding and class-frequency statistics so individual
    groups do not each repeat ``np.unique`` / ``astype(int)`` over ``y_clean``.

    Note: ``y_int`` / ``classes`` / ``class_counts`` / ``n_classes`` are all
    derived from ``y_clean`` (the NaN-dropped subset), which is what most
    classification groups need.  ``ClassDistribution`` still uses ``ctx.y``
    directly for the full-sample class distribution.
    """
    y_int:        np.ndarray = None     # type: ignore[assignment]  # y_clean cast to int
    classes:      np.ndarray = None     # type: ignore[assignment]  # unique class labels
    class_counts: np.ndarray = None     # type: ignore[assignment]  # per-class counts
    n_classes:    int = 0


class DatasetFeatureGroup(ABC):
    """One group of dataset-level meta-features."""

    name: str = ""
    feature_names: Tuple[str, ...] = ()

    @abstractmethod
    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Instance level
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceFeatureContext:
    """Per-experiment scratchpad shared by every instance-level group."""
    X_ctx:       pd.DataFrame
    y_ctx:       np.ndarray
    X_test:      pd.DataFrame
    n_context:   int
    X_ctx_sc:    np.ndarray              # standardized numeric context
    X_te_sc:     np.ndarray              # standardized numeric test
    ppd:         object                   # PPDQuantileGrid (forward ref)
    intervals:   dict                     # IntervalArrays-shaped dict


class InstanceFeatureGroup(ABC):
    name: str = ""
    feature_names: Tuple[str, ...] = ()

    @abstractmethod
    def compute(self, ctx: InstanceFeatureContext) -> pd.DataFrame:
        ...
