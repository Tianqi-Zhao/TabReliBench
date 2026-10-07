"""Dataset-level meta-feature extraction.

Ten groups for regression, eleven (partly different) groups for
classification.  Two concrete extractor subclasses, one per task.

Regression groups → module:

  target_distribution               target.py
  signal_quality                    signal.py
  complexity_nonlinearity           complexity.py
  tree_shape                        tree_shape.py
  hs_tree_complexity                hs_tree_complexity.py
  heteroscedasticity_local          heteroscedasticity.py
  dimensionality_capacity           dimensionality.py
  feature_importance_structure      correlation.py
  feature_moments                   feature_moments.py
  categorical_features              categorical.py

Classification groups → module:

  class_distribution                class_distribution.py    (replaces target_distribution)
  class_signal_quality              class_signal.py           (replaces signal_quality)
  classification_complexity         class_complexity.py       (replaces complexity_nonlinearity)
  class_tree_shape                  class_tree_shape.py       (replaces tree_shape)
  hs_tree_classification_complexity hs_tree_class_complexity.py (replaces hs_tree_complexity)
  class_local_structure             class_local.py            (replaces heteroscedasticity_local)
  clustering_structure              class_clustering.py       (classification-only, no regression analogue)
  dimensionality_capacity           dimensionality.py         (reused as-is)
  feature_importance_structure      correlation.py            (reused as-is)
  feature_moments                   feature_moments.py        (reused as-is)
  categorical_features              categorical.py             (reused as-is)

Context hierarchy:
  :class:`~evaluation.features.DatasetFeatureContext` (shared base)
  ├── :class:`~evaluation.features.RegressionDatasetFeatureContext`
  └── :class:`~evaluation.features.ClassificationDatasetFeatureContext`

Extractor hierarchy:
  :class:`DatasetFeatureExtractor` (abstract base)
  ├── :class:`RegressionDatasetFeatureExtractor`
  └── :class:`ClassificationDatasetFeatureExtractor`

Public entry points::

    feats = RegressionDatasetFeatureExtractor().compute(X_train, y_train, seed)
    feats = ClassificationDatasetFeatureExtractor().compute(X_train, y_train, seed)

Task-keyed routing::

    from evaluation.features.dataset import TASK_TO_EXTRACTOR_CLS
    extractor = TASK_TO_EXTRACTOR_CLS["regression"]()
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import (
    ClassificationDatasetFeatureContext,
    DatasetFeatureContext,
    DatasetFeatureGroup,
    RegressionDatasetFeatureContext,
)
from .categorical import CategoricalFeatures
from .class_clustering import ClusteringStructure
from .class_complexity import ClassificationComplexity
from .class_distribution import ClassDistribution
from .class_local import ClassLocalStructure
from .class_signal import ClassSignalQuality
from .class_tree_shape import ClassTreeShape
from .complexity import ComplexityNonlinearity
from .correlation import FeatureImportanceStructure
from .feature_moments import FeatureMoments
from .hs_tree_class_complexity import HSTreeClassificationComplexity
from .hs_tree_complexity import HSTreeComplexity
from .dimensionality import DimensionalityCapacity
from .heteroscedasticity import HeteroscedasticityLocal
from .signal import SignalQuality
from .target import TargetDistribution
from .tree_shape import TreeShape

warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_DATASET_GROUPS: tuple[DatasetFeatureGroup, ...] = (
    TargetDistribution(),
    SignalQuality(),
    ComplexityNonlinearity(),
    TreeShape(),
    HSTreeComplexity(),
    HeteroscedasticityLocal(),
    DimensionalityCapacity(),
    FeatureImportanceStructure(),
    FeatureMoments(),
    CategoricalFeatures(),
)

DEFAULT_CLASSIFICATION_DATASET_GROUPS: tuple[DatasetFeatureGroup, ...] = (
    ClassDistribution(),
    ClassSignalQuality(),
    ClassificationComplexity(),
    ClassTreeShape(),
    HSTreeClassificationComplexity(),
    ClassLocalStructure(),
    ClusteringStructure(),
    DimensionalityCapacity(),
    FeatureImportanceStructure(),
    FeatureMoments(),
    CategoricalFeatures(),
)


class DatasetFeatureExtractor(ABC):
    """Abstract base: run every configured group and merge their dicts.

    Do not instantiate directly.  Use :class:`RegressionDatasetFeatureExtractor`
    or :class:`ClassificationDatasetFeatureExtractor` instead.

    Subclasses must declare:

    * ``task: ClassVar[str]`` — ``'regression'`` or ``'classification'``
    * ``DEFAULT_GROUPS: ClassVar[tuple[DatasetFeatureGroup, ...]]``
    * ``_build_context(X_train, y_train, seed) -> DatasetFeatureContext``

    Parameters
    ----------
    groups:
        Explicit list of groups to run.  When omitted, ``DEFAULT_GROUPS`` of
        the concrete subclass is used.
    max_samples:
        Cap the training set to this many rows (random subsample) before
        computing features.
    max_numeric_missing_frac:
        Numeric columns whose missing-value fraction exceeds this threshold
        are dropped before standardisation.
    """

    task: ClassVar[str]
    DEFAULT_GROUPS: ClassVar[tuple[DatasetFeatureGroup, ...]]

    def __init__(
        self,
        groups: Sequence[DatasetFeatureGroup] | None = None,
        max_samples: int = 10_000,
        max_numeric_missing_frac: float = 0.5,
    ) -> None:
        if not (0.0 <= max_numeric_missing_frac <= 1.0):
            raise ValueError(
                "max_numeric_missing_frac must be in [0, 1], got "
                f"{max_numeric_missing_frac}",
            )
        self.groups = list(groups) if groups is not None else list(type(self).DEFAULT_GROUPS)
        self.max_samples = max_samples
        self.max_numeric_missing_frac = float(max_numeric_missing_frac)

    @property
    def feature_groups(self) -> dict[str, list[str]]:
        return {g.name: list(g.feature_names) for g in self.groups}

    def compute(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        seed: int,
    ) -> dict[str, float]:
        ctx = self._build_context(X_train, y_train, seed)
        feats: dict[str, float] = {}
        for grp in self.groups:
            feats.update(grp.compute(ctx))
        return feats

    def _build_base_fields(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        seed: int,
    ) -> dict:
        """Compute fields shared by both regression and classification contexts."""
        rng = np.random.default_rng(seed)
        X_df = X_train
        y = y_train
        if len(y) > self.max_samples:
            idx = rng.choice(len(y), size=self.max_samples, replace=False)
            X_df = X_df.iloc[idx].reset_index(drop=True)
            y = y[idx]

        n = len(y)
        d = X_df.shape[1]
        X_num_raw = X_df.select_dtypes(include="number")
        if X_num_raw.shape[1] > 0:
            keep_cols = X_num_raw.isna().mean() <= self.max_numeric_missing_frac
            X_num = X_num_raw.loc[:, keep_cols]
        else:
            X_num = X_num_raw
        d_num = X_num.shape[1]

        X_num_clean = X_num.dropna()
        y_clean = y[X_num_clean.index.values]
        n_clean = len(X_num_clean)

        if d_num > 0 and n_clean >= 4:
            X_sc = StandardScaler().fit_transform(X_num_clean.values)
        else:
            X_sc = np.empty((0, 0))

        return dict(
            X_df=X_df, y=y, seed=seed,
            n=n, d=d,
            X_num=X_num, d_num=d_num,
            X_num_clean=X_num_clean, y_clean=y_clean, n_clean=n_clean,
            X_sc=X_sc,
        )

    @abstractmethod
    def _build_context(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        seed: int,
    ) -> DatasetFeatureContext:
        """Build the task-specific context dataclass from training data."""


class RegressionDatasetFeatureExtractor(DatasetFeatureExtractor):
    """Extractor for regression tasks.

    Builds a :class:`~evaluation.features.RegressionDatasetFeatureContext`
    (adds ``y_mean``, ``y_std``, ``z``) and runs
    :data:`DEFAULT_DATASET_GROUPS`.
    """

    task: ClassVar[str] = "regression"
    DEFAULT_GROUPS: ClassVar[tuple[DatasetFeatureGroup, ...]] = DEFAULT_DATASET_GROUPS

    def _build_context(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        seed: int,
    ) -> RegressionDatasetFeatureContext:
        base = self._build_base_fields(X_train, y_train, seed)
        y = base["y"]
        y_mean = float(np.mean(y))
        y_std  = float(np.std(y)) or 1.0
        return RegressionDatasetFeatureContext(
            **base,
            y_mean=y_mean,
            y_std=y_std,
            z=(y - y_mean) / y_std,
        )


class ClassificationDatasetFeatureExtractor(DatasetFeatureExtractor):
    """Extractor for classification tasks.

    Builds a :class:`~evaluation.features.ClassificationDatasetFeatureContext`
    (adds ``y_int``, ``classes``, ``class_counts``, ``n_classes``) and runs
    :data:`DEFAULT_CLASSIFICATION_DATASET_GROUPS`.
    """

    task: ClassVar[str] = "classification"
    DEFAULT_GROUPS: ClassVar[tuple[DatasetFeatureGroup, ...]] = DEFAULT_CLASSIFICATION_DATASET_GROUPS

    def _build_context(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        seed: int,
    ) -> ClassificationDatasetFeatureContext:
        base = self._build_base_fields(X_train, y_train, seed)
        y_clean = base["y_clean"]
        y_int = (
            y_clean.astype(int)
            if not np.issubdtype(y_clean.dtype, np.integer)
            else y_clean
        )
        classes, class_counts = np.unique(y_int, return_counts=True)
        return ClassificationDatasetFeatureContext(
            **base,
            y_int=y_int,
            classes=classes,
            class_counts=class_counts,
            n_classes=int(len(classes)),
        )


TASK_TO_EXTRACTOR_CLS: dict[str, type[DatasetFeatureExtractor]] = {
    "regression":     RegressionDatasetFeatureExtractor,
    "classification": ClassificationDatasetFeatureExtractor,
}
