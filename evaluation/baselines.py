"""Baselines for the uncertainty benchmark.

The classes in this module deliberately produce the same prediction payloads
as :class:`evaluation.pipelines.PredictionPipeline`:

* regression -> a dense conditional-quantile grid plus a point prediction;
* classification -> probabilities on the dataset's global class space.

Consequently, the existing ``RegressionMetricsCalculator`` and
``ClassificationMetricsCalculator`` can evaluate these baselines without a
second metrics implementation.

``pytabkit``, ``xgboost``, and ``bartz`` are optional dependencies and are
imported only when their respective runners need them. Random Forest remains
usable in a lightweight local environment that has none of these packages.

``PosteriorPredictiveBaseline`` is the reusable adapter for Bayesian models.
An implementation supplies posterior predictive draws and a reloadable final
model artifact; the adapter orchestrates injectable preprocessing and owns
output normalization. ``BARTBaseline`` implements that interface with
``bartz.Bart`` while keeping the optional JAX dependency lazy.
"""
from __future__ import annotations

import gc
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .models import _require_zero_based_class_subset
from .ppd import quantile_grid_integral


def _is_categorical(series: pd.Series) -> bool:
    """Return whether a column should be treated as categorical."""
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
        or not pd.api.types.is_numeric_dtype(series)
    )


def _categorical_columns(
    X_context: pd.DataFrame,
    X_test: pd.DataFrame,
) -> list[str]:
    """Categorical columns in either split, retaining input column order."""
    return [
        str(column)
        for column in X_context.columns
        if _is_categorical(X_context[column]) or _is_categorical(X_test[column])
    ]


def _stringify_categorical(series: pd.Series) -> pd.Series:
    """Convert observed categories to strings and missing cells to a token.

    ``OrdinalEncoder`` otherwise rejects columns containing a mixture of, for
    example, integers and strings.  The token is intentionally fitted only on
    the context split; test-only labels still become the unknown value ``-1``.
    """
    obj = series.astype("object")
    return obj.where(obj.notna(), "__MISSING__").astype(str)


class BaselinePreprocessor(Protocol):
    """Injectable preprocessing contract used by benchmark adapters.

    Implementations receive the already-created benchmark context/test split.
    Any fitted state must be learned from ``X_context`` only, then applied
    unchanged to ``X_test``.  They must preserve row count and row order.
    ``categorical_columns_`` refers to categorical-ID columns in the returned
    feature matrix.
    """

    categorical_columns_: list[str] | None

    def fit_transform(
        self,
        X_context: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[Any, Any]: ...


BaselinePreprocessorFactory = Callable[[], BaselinePreprocessor]


@dataclass
class BaselineFeaturePreprocessor:
    """Context-fitted ordinal preprocessing used by RealMLP.

    Numerical columns are coerced to floating point and filled with their
    context median (or zero when the entire context column is missing).
    Categorical columns are ordinal encoded using the context vocabulary;
    test-only categories receive ``-1``.  Returned DataFrames contain no
    missing values.  ``categorical_columns_`` is passed to RealMLP so its
    categorical embeddings remain active despite the numeric representation.
    Random Forest and BART deliberately use their own named preprocessors so
    each model's categorical semantics remain explicit.
    """

    categorical_columns_: list[str] | None = None
    numeric_columns_: list[str] | None = None
    numeric_fill_values_: dict[str, float] | None = None
    encoder_: Optional[OrdinalEncoder] = None
    feature_columns_: list[str] | None = None

    def fit_transform(
        self,
        X_context: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(X_context, pd.DataFrame) or not isinstance(
            X_test, pd.DataFrame
        ):
            raise TypeError("BaselineFeaturePreprocessor expects pandas DataFrames")
        if list(X_context.columns) != list(X_test.columns):
            raise ValueError(
                "X_context and X_test must have identical columns in identical order"
            )

        # OpenML column labels are normalized to strings by DatasetLoader, but
        # keep this conversion here so the preprocessor is safe in direct use.
        context = X_context.copy()
        test = X_test.copy()
        context.columns = [str(c) for c in context.columns]
        test.columns = [str(c) for c in test.columns]
        self.feature_columns_ = list(context.columns)

        cat_cols = _categorical_columns(context, test)
        num_cols = [c for c in context.columns if c not in cat_cols]
        self.categorical_columns_ = cat_cols
        self.numeric_columns_ = num_cols
        self.numeric_fill_values_ = {}

        out_context = pd.DataFrame(index=context.index)
        out_test = pd.DataFrame(index=test.index)

        for column in num_cols:
            context_col = pd.to_numeric(context[column], errors="coerce")
            test_col = pd.to_numeric(test[column], errors="coerce")
            if context_col.notna().any():
                fill_value = float(context_col.median(skipna=True))
            else:
                fill_value = 0.0
            self.numeric_fill_values_[column] = fill_value
            out_context[column] = context_col.fillna(fill_value).astype(np.float32)
            out_test[column] = test_col.fillna(fill_value).astype(np.float32)

        if cat_cols:
            context_cat = pd.DataFrame(
                {
                    column: _stringify_categorical(context[column])
                    for column in cat_cols
                },
                index=context.index,
            )
            test_cat = pd.DataFrame(
                {
                    column: _stringify_categorical(test[column])
                    for column in cat_cols
                },
                index=test.index,
            )
            self.encoder_ = OrdinalEncoder(
                dtype=np.float32,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            context_encoded = self.encoder_.fit_transform(context_cat)
            test_encoded = self.encoder_.transform(test_cat)
            for index, column in enumerate(cat_cols):
                out_context[column] = context_encoded[:, index]
                out_test[column] = test_encoded[:, index]

        # Restore the original column ordering after constructing numeric and
        # categorical blocks separately.
        columns = list(context.columns)
        return (
            out_context.loc[:, columns].reset_index(drop=True),
            out_test.loc[:, columns].reset_index(drop=True),
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted context transformation to new rows."""
        if self.feature_columns_ is None or self.numeric_columns_ is None:
            raise RuntimeError("BaselineFeaturePreprocessor has not been fitted")
        frame = X.copy()
        frame.columns = [str(c) for c in frame.columns]
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("X columns do not match the fitted feature schema")
        out = pd.DataFrame(index=frame.index)
        for column in self.numeric_columns_:
            fill_value = (self.numeric_fill_values_ or {})[column]
            out[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(fill_value)
                .astype(np.float32)
            )
        cat_cols = self.categorical_columns_ or []
        if cat_cols:
            if self.encoder_ is None:
                raise RuntimeError("categorical encoder is missing")
            cat = pd.DataFrame(
                {
                    column: _stringify_categorical(frame[column])
                    for column in cat_cols
                },
                index=frame.index,
            )
            encoded = self.encoder_.transform(cat)
            for index, column in enumerate(cat_cols):
                out[column] = encoded[:, index]
        return out.loc[:, self.feature_columns_].reset_index(drop=True)


@dataclass
class BARTFeaturePreprocessor:
    """Context-fitted numeric and one-hot features for ``bartz``.

    ``bartz.Bart`` uses ordered threshold splits, so assigning arbitrary
    ordinal IDs to nominal categories would introduce a false ordering.  The
    default BART representation therefore one-hot encodes categorical columns.
    Unknown test categories map to an all-zero indicator block.  Numerical
    columns use the same context-median imputation as the other baselines.

    One-hot expansion would normally make a high-cardinality source feature
    more likely to be selected for a split.  ``feature_prior_weights_`` gives
    every original input column equal total mass; the BART adapter passes these
    weights to ``bartz.Bart(varprob=...)``.  The transformed matrices also carry
    the weights in ``DataFrame.attrs`` so the narrow posterior-baseline hook
    contract does not need to change.
    """

    categorical_columns_: list[str] | None = None
    source_categorical_columns_: list[str] | None = None
    numeric_columns_: list[str] | None = None
    numeric_fill_values_: dict[str, float] | None = None
    encoder_: Optional[OneHotEncoder] = None
    feature_columns_: list[str] | None = None
    transformed_columns_: list[str] | None = None
    feature_prior_weights_: Optional[np.ndarray] = None
    categorical_output_slices_: dict[str, slice] | None = None

    def _validate_frames(
        self,
        X_context: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(X_context, pd.DataFrame) or not isinstance(
            X_test, pd.DataFrame
        ):
            raise TypeError("BARTFeaturePreprocessor expects pandas DataFrames")
        if list(X_context.columns) != list(X_test.columns):
            raise ValueError(
                "X_context and X_test must have identical columns in identical order"
            )
        context = X_context.copy()
        test = X_test.copy()
        context.columns = [str(column) for column in context.columns]
        test.columns = [str(column) for column in test.columns]
        return context, test

    def _numeric_block(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        columns: list[np.ndarray] = []
        for column in self.numeric_columns_ or []:
            fill_value = (self.numeric_fill_values_ or {})[column]
            columns.append(
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(fill_value)
                .to_numpy(dtype=np.float32)
            )
        if not columns:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.column_stack(columns).astype(np.float32, copy=False)

    def _categorical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = self.source_categorical_columns_ or []
        return pd.DataFrame(
            {
                column: _stringify_categorical(frame[column])
                for column in columns
            },
            index=frame.index,
        )

    def _assemble(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = self._numeric_block(frame)
        categorical_columns = self.source_categorical_columns_ or []
        if categorical_columns:
            if self.encoder_ is None:
                raise RuntimeError("categorical encoder is missing")
            categorical = self.encoder_.transform(
                self._categorical_frame(frame)
            ).astype(np.float32, copy=False)
        else:
            categorical = np.empty((len(frame), 0), dtype=np.float32)
        values = np.concatenate([numeric, categorical], axis=1)
        output = pd.DataFrame(values, columns=self.transformed_columns_)
        output.attrs["bartz_feature_prior_weights"] = np.asarray(
            self.feature_prior_weights_, dtype=np.float32
        )
        return output

    def fit_transform(
        self,
        X_context: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        context, test = self._validate_frames(X_context, X_test)
        self.feature_columns_ = list(context.columns)
        source_categorical = _categorical_columns(context, test)
        numeric = [
            column
            for column in self.feature_columns_
            if column not in source_categorical
        ]
        self.source_categorical_columns_ = source_categorical
        # The returned columns are numeric/binary, not categorical IDs.
        self.categorical_columns_ = []
        self.numeric_columns_ = numeric
        self.numeric_fill_values_ = {}
        self.encoder_ = None
        for column in numeric:
            context_column = pd.to_numeric(context[column], errors="coerce")
            self.numeric_fill_values_[column] = (
                float(context_column.median(skipna=True))
                if context_column.notna().any()
                else 0.0
            )

        if source_categorical:
            self.encoder_ = OneHotEncoder(
                dtype=np.float32,
                handle_unknown="ignore",
                sparse_output=False,
            )
            self.encoder_.fit(self._categorical_frame(context))

        transformed_columns = [
            f"numeric::{index}" for index in range(len(numeric))
        ]
        prior_weights = [1.0] * len(numeric)
        self.categorical_output_slices_ = {}
        output_index = len(numeric)
        for column_index, column in enumerate(source_categorical):
            if self.encoder_ is None:
                raise RuntimeError("categorical encoder is missing")
            width = len(self.encoder_.categories_[column_index])
            start = output_index
            output_index += width
            self.categorical_output_slices_[column] = slice(start, output_index)
            transformed_columns.extend(
                f"categorical::{column_index}::{level_index}"
                for level_index in range(width)
            )
            prior_weights.extend([1.0 / width] * width)
        self.transformed_columns_ = transformed_columns
        self.feature_prior_weights_ = np.asarray(
            prior_weights, dtype=np.float32
        )
        return self._assemble(context), self._assemble(test)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted context transformation to new rows."""
        if self.feature_columns_ is None or self.transformed_columns_ is None:
            raise RuntimeError("BARTFeaturePreprocessor has not been fitted")
        frame = X.copy()
        frame.columns = [str(column) for column in frame.columns]
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("X columns do not match the fitted feature schema")
        return self._assemble(frame)


@dataclass
class RandomForestFeaturePreprocessor(BaselineFeaturePreprocessor):
    """TabArena-style context-fitted ordinal preprocessing for sklearn RF.

    Each categorical feature remains one column, so preprocessing width stays
    equal to the input width regardless of category cardinality.  The encoder
    is fitted on context rows only, unseen test categories map to ``-1``, and
    numerical features use context-median imputation without scaling.
    """


def _prepare_baseline_features(
    X_context: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    preprocessor: Optional[BaselinePreprocessor] = None,
) -> tuple[
    Any,
    Any,
    Any,
    float,
]:
    """Run and time preprocessing without changing the benchmark split."""
    start = time.perf_counter()
    if preprocessor is None:
        preprocessor = BaselineFeaturePreprocessor()
    X_context_in, X_test_in = preprocessor.fit_transform(X_context, X_test)
    if not hasattr(preprocessor, "categorical_columns_"):
        raise TypeError(
            "baseline preprocessing must expose categorical_columns_ after fit"
        )
    n_context_out = int(X_context_in.shape[0])
    n_test_out = int(X_test_in.shape[0])
    if n_context_out != len(X_context):
        raise ValueError(
            "baseline preprocessing must preserve the context row count: "
            f"expected {len(X_context)}, got {n_context_out}"
        )
    if n_test_out != len(X_test):
        raise ValueError(
            "baseline preprocessing must preserve the test row count: "
            f"expected {len(X_test)}, got {n_test_out}"
        )
    return (
        preprocessor,
        X_context_in,
        X_test_in,
        time.perf_counter() - start,
    )


def _local_class_labels(
    y_context: np.ndarray,
    n_classes_global: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel context classes using the foundation-runner contract.

    In particular, a non-empty single-class context is valid at the adapter
    layer.  Whether a specific estimator can fit it is left to that estimator,
    matching :class:`evaluation.models.ClassificationModelRunner`.
    """
    y = np.asarray(y_context)
    context_classes, n_classes_global = _require_zero_based_class_subset(
        "baseline", y, n_classes_global,
    )
    context_classes = context_classes.astype(np.int64, copy=False)
    global_to_local = np.full(n_classes_global, -1, dtype=np.int64)
    global_to_local[context_classes] = np.arange(context_classes.size)
    return global_to_local[y.astype(np.int64)], context_classes


def _pad_probabilities(
    local_probabilities: np.ndarray,
    context_classes: np.ndarray,
    n_classes_global: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad probabilities for classes absent from the context with zeros."""
    local = np.asarray(local_probabilities, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != len(context_classes):
        raise ValueError(
            "probability shape does not match the number of context classes: "
            f"shape={local.shape}, classes={context_classes.tolist()}"
        )
    probabilities = np.zeros(
        (local.shape[0], int(n_classes_global)), dtype=np.float64
    )
    probabilities[:, context_classes] = local
    return probabilities, np.arange(n_classes_global, dtype=np.int64)


def _quantile_mean(ppd: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Approximate a distributional mean from its conditional quantiles.

    The expectation equals the integral of the quantile function over
    ``[0, 1]``.  The dense benchmark grid excludes exactly 0 and 1, so the
    two tiny tails are extended as constants at the outermost predictions.
    """
    values = np.asarray(ppd, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"ppd must be 2D; got shape {values.shape}")
    return quantile_grid_integral(values, levels)


def _coerce_quantile_predictions(
    predictions: np.ndarray,
    *,
    n_test: int,
    n_quantiles: int,
    enforce_monotone: bool = True,
) -> np.ndarray:
    """Normalize multi-quantile output to ``(n_test, n_quantiles)``."""
    arr = np.asarray(predictions, dtype=np.float64)
    while arr.ndim > 2:
        singleton_axes = [axis for axis, size in enumerate(arr.shape) if size == 1]
        if not singleton_axes:
            break
        arr = np.squeeze(arr, axis=singleton_axes[0])
    if arr.shape == (n_quantiles, n_test):
        arr = arr.T
    if arr.shape != (n_test, n_quantiles):
        raise ValueError(
            "Multi-quantile prediction has unexpected shape "
            f"{arr.shape}; expected ({n_test}, {n_quantiles})"
        )
    return np.maximum.accumulate(arr, axis=1) if enforce_monotone else arr


def _shared_regression_holdout_splits(
    n_rows: int,
    *,
    seed: int,
    n_splits: int = 1,
    validation_fraction: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic holdouts shared by XGBoost and RealMLP."""
    if n_rows < 2:
        raise ValueError("A validation split requires at least two context rows")
    splitter = ShuffleSplit(
        n_splits=n_splits,
        test_size=validation_fraction,
        random_state=seed,
    )
    dummy = np.empty((n_rows, 1), dtype=np.float32)
    return list(splitter.split(dummy))


def _shared_classification_holdout_splits(
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int = 1,
    validation_fraction: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shared class-safe holdouts for checkpoint/HPO selection."""
    labels = np.asarray(y, dtype=np.int64)
    n_rows = len(labels)
    if n_rows < 2:
        raise ValueError("A validation split requires at least two context rows")
    classes, counts = np.unique(labels, return_counts=True)
    n_validation = max(1, int(np.ceil(validation_fraction * n_rows)))
    n_train = n_rows - n_validation
    can_stratify = (
        counts.min() >= 2
        and n_validation >= len(classes)
        and n_train >= len(classes)
    )
    dummy = np.empty((n_rows, 1), dtype=np.float32)
    if can_stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=n_validation,
            random_state=seed,
        )
        return list(splitter.split(dummy, labels))

    # Keep at least one example of every context class in training. This is
    # essential when a rare class occurs only once in the outer context split.
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split_index in range(n_splits):
        rng = np.random.default_rng(seed + split_index)
        validation_candidates: list[int] = []
        for label in classes:
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            validation_candidates.extend(indices[1:].tolist())
        if not validation_candidates:
            raise ValueError(
                "Cannot create a validation set while keeping every context "
                "class in training"
            )
        rng.shuffle(validation_candidates)
        validation = np.asarray(
            validation_candidates[:n_validation], dtype=np.int64
        )
        training = np.setdiff1d(
            np.arange(n_rows, dtype=np.int64), validation
        )
        splits.append((training, validation))
    return splits


def _shared_kfold_splits(
    n_rows: int,
    *,
    seed: int,
    n_splits: int,
    y: Optional[np.ndarray] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create equal-width K-fold indices accepted by PyTabKit ``val_idxs``.

    PyTabKit requires an explicit multi-fold validation array to be
    rectangular.  Consequently, at most ``n_splits - 1`` shuffled remainder
    rows stay in every training fold when ``n_rows`` is not divisible by
    ``n_splits``.  For classification, singleton-class rows also stay in
    training so every fitted classifier retains the full context class space.
    """
    if n_splits <= 1:
        raise ValueError("K-fold validation requires n_splits >= 2")
    if n_splits > n_rows:
        raise ValueError(
            "K-fold validation requires n_splits <= number of context rows"
        )

    all_indices = np.arange(n_rows, dtype=np.int64)
    eligible_indices = all_indices
    labels: Optional[np.ndarray] = None
    if y is not None:
        labels = np.asarray(y, dtype=np.int64)
        if labels.shape != (n_rows,):
            raise ValueError(
                f"Expected {n_rows} classification labels; got {labels.shape}"
            )
        classes, counts = np.unique(labels, return_counts=True)
        non_singleton_classes = classes[counts >= 2]
        eligible_indices = all_indices[
            np.isin(labels, non_singleton_classes)
        ]

    fold_size = len(eligible_indices) // n_splits
    if fold_size == 0:
        raise ValueError(
            "Not enough validation-eligible context rows for the requested "
            f"{n_splits}-fold split"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible_indices)
    fold_length = fold_size * n_splits
    fold_members = shuffled[:fold_length]
    if labels is not None:
        # Sorting after shuffling preserves random order within each class;
        # striding then distributes each class across the folds.
        order = np.argsort(labels[fold_members], kind="stable")
        fold_members = fold_members[order]
    validation_folds = [
        np.asarray(fold_members[offset::n_splits], dtype=np.int64)
        for offset in range(n_splits)
    ]
    return [
        (np.setdiff1d(all_indices, validation), validation)
        for validation in validation_folds
    ]


def _shared_regression_cv_splits(
    n_rows: int,
    *,
    seed: int,
    n_cv: int,
    validation_fraction: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use one holdout for ``n_cv=1`` and K-fold for ``n_cv>1``."""
    if n_cv == 1:
        return _shared_regression_holdout_splits(
            n_rows,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    return _shared_kfold_splits(n_rows, seed=seed, n_splits=n_cv)


def _shared_classification_cv_splits(
    y: np.ndarray,
    *,
    seed: int,
    n_cv: int,
    validation_fraction: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use one class-safe holdout or class-safe stratified K-fold."""
    labels = np.asarray(y, dtype=np.int64)
    if n_cv == 1:
        return _shared_classification_holdout_splits(
            labels,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    return _shared_kfold_splits(
        len(labels),
        seed=seed,
        n_splits=n_cv,
        y=labels,
    )


@dataclass(frozen=True)
class BaselinePrediction:
    """One model's normalized output and phase-level wall-clock timings."""

    task: str
    ppd_quantiles: Optional[np.ndarray] = None
    point_pred: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None
    classes_: Optional[np.ndarray] = None
    timing_seconds: dict[str, float] | None = None
    model_metadata: dict[str, Any] | None = None
    model_bundle: dict[str, Any] | None = None
    external_checkpoint_dir: str | None = None


@dataclass(frozen=True)
class BaselineModelArtifact:
    """Library-specific final fitted state returned to the artifact store.

    ``model_bundle`` must be sufficient to reload the selected/final model for
    later prediction and must include its context-fitted preprocessor.  A
    backend that keeps required sidecar files may additionally return their
    durable directory via ``external_checkpoint_dir``.
    """

    model_bundle: dict[str, Any]
    model_metadata: dict[str, Any] | None = None
    external_checkpoint_dir: str | None = None


class PosteriorPredictiveBaseline:
    """Reusable benchmark adapter for models with posterior predictive draws.

    Subclasses implement five narrow, library-specific hooks: fit a regression
    or classification model, draw from its posterior predictive distribution,
    and package the final fitted state for durable storage. This class turns
    those draws into the exact payload consumed by the existing metrics.

    The outer evaluation runner remains the sole owner of train/test/context
    splitting.  This adapter receives that exact ``X_context``/``X_test`` pair,
    creates a fresh preprocessor for each fit, and never resamples or repartitions
    rows.  A custom preprocessor can be injected with ``preprocessor_factory``;
    its fitted state must come only from context data.

    Regression implementations must return draws from ``p(y* | X*, data)``
    with shape ``(n_draws, n_test)``.  These should be *posterior predictive*
    response draws, including observation noise, rather than draws of only the
    latent conditional mean.  The adapter converts them to the requested dense
    quantile grid and uses their mean as the point prediction.

    Classification implementations must return posterior class-probability
    draws with shape ``(n_draws, n_test, n_context_classes)``.  The final
    probability is the posterior mean, padded onto the dataset's global class
    space by the same helper used by RF and RealMLP.
    """

    name: str

    def __init__(
        self,
        *,
        preprocessor_factory: Optional[BaselinePreprocessorFactory] = None,
    ) -> None:
        self._preprocessor_factory = (
            BaselineFeaturePreprocessor
            if preprocessor_factory is None
            else preprocessor_factory
        )

    def _make_preprocessor(self) -> BaselinePreprocessor:
        preprocessor = self._preprocessor_factory()
        if not callable(getattr(preprocessor, "fit_transform", None)):
            raise TypeError(
                "preprocessor_factory must return an object exposing "
                "fit_transform(X_context, X_test)"
            )
        return preprocessor

    def _fit_regression_model(
        self,
        X_context: Any,
        y_context: np.ndarray,
        categorical_columns: list[str],
    ) -> Any:
        raise NotImplementedError

    def _sample_regression_posterior_predictive(
        self,
        model: Any,
        X_test: Any,
    ) -> np.ndarray:
        raise NotImplementedError

    def _fit_classification_model(
        self,
        X_context: Any,
        y_context: np.ndarray,
        categorical_columns: list[str],
    ) -> Any:
        raise NotImplementedError

    def _sample_classification_probabilities(
        self,
        model: Any,
        X_test: Any,
    ) -> np.ndarray:
        raise NotImplementedError

    def _build_model_artifact(
        self,
        *,
        task: str,
        model: Any,
        preprocessor: BaselinePreprocessor,
        context_classes: Optional[np.ndarray] = None,
        n_classes_global: Optional[int] = None,
    ) -> BaselineModelArtifact:
        """Package the final/reloadable model state after a successful fit.

        This is intentionally a required library-specific hook. Some Bayesian
        libraries can serialize a fitted object directly, while others need a
        posterior trace plus model specification or durable checkpoint files.
        """
        raise NotImplementedError

    def fit_predict_regression(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        q_grid: np.ndarray,
    ) -> BaselinePrediction:
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=self._make_preprocessor(),
        )

        fit_start = time.perf_counter()
        model = self._fit_regression_model(
            X_context_in,
            np.asarray(y_context, dtype=np.float64),
            preprocessor.categorical_columns_ or [],
        )
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        draws = np.asarray(
            self._sample_regression_posterior_predictive(model, X_test_in),
            dtype=np.float64,
        )
        if draws.ndim != 2 or draws.shape[1] != len(X_test_in):
            raise ValueError(
                f"{self.name}: regression posterior predictive draws must have "
                f"shape (n_draws, n_test={len(X_test_in)}); got {draws.shape}"
            )
        if draws.shape[0] < 2:
            raise ValueError(
                f"{self.name}: at least two posterior predictive draws are required"
            )
        if not np.isfinite(draws).all():
            raise ValueError(
                f"{self.name}: regression posterior predictive draws contain "
                "non-finite values"
            )
        q = np.asarray(q_grid, dtype=np.float64)
        ppd = np.quantile(draws, q, axis=0).T
        ppd = np.maximum.accumulate(ppd, axis=1)
        point_pred = np.mean(draws, axis=0)
        predict_seconds = time.perf_counter() - predict_start
        model_artifact = self._build_model_artifact(
            task="regression",
            model=model,
            preprocessor=preprocessor,
        )

        return BaselinePrediction(
            task="regression",
            ppd_quantiles=ppd,
            point_pred=point_pred,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata=model_artifact.model_metadata,
            model_bundle=model_artifact.model_bundle,
            external_checkpoint_dir=model_artifact.external_checkpoint_dir,
        )

    def fit_predict_classification(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        *,
        n_classes_global: int,
    ) -> BaselinePrediction:
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=self._make_preprocessor(),
        )
        y_local, context_classes = _local_class_labels(
            y_context, n_classes_global
        )

        fit_start = time.perf_counter()
        model = self._fit_classification_model(
            X_context_in,
            y_local,
            preprocessor.categorical_columns_ or [],
        )
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        draws = np.asarray(
            self._sample_classification_probabilities(model, X_test_in),
            dtype=np.float64,
        )
        expected_tail = (len(X_test_in), len(context_classes))
        if draws.ndim != 3 or draws.shape[1:] != expected_tail:
            raise ValueError(
                f"{self.name}: classification posterior probability draws must "
                "have shape (n_draws, "
                f"n_test={expected_tail[0]}, n_context_classes={expected_tail[1]}); "
                f"got {draws.shape}"
            )
        if draws.shape[0] < 2:
            raise ValueError(
                f"{self.name}: at least two posterior probability draws are required"
            )
        if not np.isfinite(draws).all() or np.any(draws < 0.0):
            raise ValueError(
                f"{self.name}: posterior class probabilities must be finite "
                "and non-negative"
            )
        row_sums = draws.sum(axis=2, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError(
                f"{self.name}: posterior class probabilities must have positive "
                "row sums"
            )
        # Normalize defensively: MCMC implementations may differ by harmless
        # floating-point error or return unnormalized positive class weights.
        local_proba = np.mean(draws / row_sums, axis=0)
        proba, classes_ = _pad_probabilities(
            local_proba, context_classes, n_classes_global
        )
        predict_seconds = time.perf_counter() - predict_start
        model_artifact = self._build_model_artifact(
            task="classification",
            model=model,
            preprocessor=preprocessor,
            context_classes=context_classes,
            n_classes_global=n_classes_global,
        )

        return BaselinePrediction(
            task="classification",
            proba=proba,
            classes_=classes_,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata=model_artifact.model_metadata,
            model_bundle=model_artifact.model_bundle,
            external_checkpoint_dir=model_artifact.external_checkpoint_dir,
        )


@dataclass(frozen=True)
class _BARTFittedModel:
    """Small task-aware wrapper around a fitted ``bartz.Bart`` object."""

    estimator: Any | None
    task: str
    n_save_per_chain: int
    classification_mode: str | None = None
    n_context_classes: int | None = None


def _as_bartz_predictor_array(X: Any) -> np.ndarray:
    """Convert row-oriented benchmark features to bartz's ``(p, n)`` form.

    Using bartz's array interface also avoids a pandas 3 incompatibility in
    bartz 0.12, where comparing two pandas ``Index`` objects during prediction
    produces an array of booleans instead of a single format-match result.
    """
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(dtype=np.float32)
    else:
        values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"BART features must be two-dimensional; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("BART features contain non-finite values")
    return np.ascontiguousarray(values.T)


@dataclass(frozen=True)
class _BARTPredictorAdapter:
    """Present the repository's row-oriented input API around ``bartz.Bart``."""

    estimator: Any

    def predict(self, X: Any, **kwargs: Any) -> Any:
        predictors = X if isinstance(X, str) else _as_bartz_predictor_array(X)
        return self.estimator.predict(predictors, **kwargs)

    def dump(self, path: Any) -> None:
        self.estimator.dump(path)


class BARTBaseline(PosteriorPredictiveBaseline):
    """BART baseline implemented with the high-level ``bartz.Bart`` API.

    Regression uses posterior outcome samples, which include the sampled
    Gaussian observation error. Binary classification uses probit BART.
    Multiclass classification uses a multivariate one-vs-rest probit outcome;
    the reusable adapter normalizes each posterior probability vector before
    averaging it and padding it onto the dataset's global class space.

    ``n_draws`` is the total number of draws returned to the benchmark across
    all chains. ``bartz`` saves the same number per chain, so this adapter saves
    ``ceil(n_draws / n_chains)`` per chain and selects an approximately equal
    number from every chain.
    """

    name = "bart"

    def __init__(
        self,
        *,
        seed: int,
        n_trees: int = 200,
        n_draws: int = 1_000,
        n_burn: int = 1_000,
        n_chains: int = 4,
        device: str = "auto",
        preprocessor_factory: Optional[BaselinePreprocessorFactory] = None,
    ) -> None:
        if n_trees < 1:
            raise ValueError("BART needs at least one tree")
        if n_draws < 2:
            raise ValueError("BART needs at least two posterior draws")
        if n_burn < 0:
            raise ValueError("BART burn-in cannot be negative")
        if n_chains < 1:
            raise ValueError("BART needs at least one chain")
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("BART device must be 'auto', 'cpu', or 'gpu'")
        super().__init__(
            preprocessor_factory=(
                BARTFeaturePreprocessor
                if preprocessor_factory is None
                else preprocessor_factory
            )
        )
        self.seed = int(seed)
        self.n_trees = int(n_trees)
        self.n_draws = int(n_draws)
        self.n_burn = int(n_burn)
        self.n_chains = int(n_chains)
        self.device = str(device)

    @staticmethod
    def _bartz_modules() -> tuple[Any, Any, Any]:
        try:
            import bartz
            import jax
            from bartz import Bart
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "BART requires bartz>=0.12 and Python>=3.10. Install the "
                "CPU package with `pip install 'bartz>=0.12,<0.13'`, or use "
                "the bartz CUDA extra appropriate for the server."
            ) from exc
        version_info = getattr(bartz, "__version_info__", None)
        if version_info is not None and tuple(version_info)[:2] < (0, 12):
            raise ImportError(
                "BART requires bartz>=0.12 for the tested prediction and "
                "serialization APIs"
            )
        return bartz, Bart, jax

    @property
    def _n_save_per_chain(self) -> int:
        return (self.n_draws + self.n_chains - 1) // self.n_chains

    def _fit_bartz(
        self,
        X_context: Any,
        y_context: np.ndarray,
        *,
        outcome_type: str | list[str],
        categorical_columns: list[str],
    ) -> Any:
        if categorical_columns:
            raise ValueError(
                "bartz uses ordered threshold splits; a custom BART "
                "preprocessor must expand nominal categorical columns and "
                "report categorical_columns_=[]"
            )
        _, Bart, _ = self._bartz_modules()
        feature_prior_weights = None
        if isinstance(X_context, pd.DataFrame):
            feature_prior_weights = X_context.attrs.get(
                "bartz_feature_prior_weights"
            )
        predictors = _as_bartz_predictor_array(X_context)
        kwargs: dict[str, Any] = {
            "outcome_type": outcome_type,
            "num_trees": self.n_trees,
            "n_save": self._n_save_per_chain,
            "n_burn": self.n_burn,
            "num_chains": self.n_chains,
            "seed": self.seed,
            "pbar": False,
        }
        if feature_prior_weights is not None:
            weights = np.asarray(feature_prior_weights, dtype=np.float32)
            if weights.shape != (predictors.shape[0],) or np.any(weights <= 0.0):
                raise ValueError(
                    "BART feature prior weights must be positive with one "
                    "entry per transformed feature"
                )
            kwargs["varprob"] = weights
        if self.device != "auto":
            kwargs["devices"] = self.device
        return _BARTPredictorAdapter(
            Bart(
                predictors,
                np.asarray(y_context, dtype=np.float32),
                **kwargs,
            )
        )

    def _select_draws(self, values: Any) -> np.ndarray:
        draws = np.asarray(values)
        expected = self.n_chains * self._n_save_per_chain
        if draws.ndim < 1 or draws.shape[0] != expected:
            raise ValueError(
                "bartz returned an unexpected posterior sample count: "
                f"expected {expected}, got {draws.shape}"
            )
        by_chain = draws.reshape(
            self.n_chains,
            self._n_save_per_chain,
            *draws.shape[1:],
        )
        base, remainder = divmod(self.n_draws, self.n_chains)
        selected = [
            by_chain[chain, : base + (chain < remainder)]
            for chain in range(self.n_chains)
            if base + (chain < remainder) > 0
        ]
        return np.concatenate(selected, axis=0)

    def _prediction_key(self, jax: Any) -> Any:
        prediction_seed = (self.seed + 1_000_003) % (2**31 - 1)
        key_factory = getattr(jax.random, "key", None)
        if key_factory is None:  # pragma: no cover - older compatible JAX
            key_factory = jax.random.PRNGKey
        return key_factory(prediction_seed)

    def _fit_regression_model(
        self,
        X_context: Any,
        y_context: np.ndarray,
        categorical_columns: list[str],
    ) -> Any:
        estimator = self._fit_bartz(
            X_context,
            y_context,
            outcome_type="continuous",
            categorical_columns=categorical_columns,
        )
        return _BARTFittedModel(
            estimator=estimator,
            task="regression",
            n_save_per_chain=self._n_save_per_chain,
        )

    def _sample_regression_posterior_predictive(
        self,
        model: Any,
        X_test: Any,
    ) -> np.ndarray:
        if not isinstance(model, _BARTFittedModel) or model.task != "regression":
            raise TypeError("invalid fitted BART regression model")
        _, _, jax = self._bartz_modules()
        draws = model.estimator.predict(
            X_test,
            kind="outcome_samples",
            key=self._prediction_key(jax),
        )
        return self._select_draws(draws)

    def _fit_classification_model(
        self,
        X_context: Any,
        y_context: np.ndarray,
        categorical_columns: list[str],
    ) -> Any:
        y = np.asarray(y_context, dtype=np.int64)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("BART classification needs a non-empty 1D target")
        n_context_classes = int(np.max(y)) + 1
        if not np.array_equal(np.unique(y), np.arange(n_context_classes)):
            raise ValueError("BART classification labels must be contiguous")
        if n_context_classes == 1:
            return _BARTFittedModel(
                estimator=None,
                task="classification",
                n_save_per_chain=self._n_save_per_chain,
                classification_mode="constant",
                n_context_classes=1,
            )
        if n_context_classes == 2:
            estimator = self._fit_bartz(
                X_context,
                y.astype(np.float32),
                outcome_type="binary",
                categorical_columns=categorical_columns,
            )
            mode = "binary_probit"
        else:
            one_vs_rest = np.eye(
                n_context_classes, dtype=np.float32
            )[y].T
            estimator = self._fit_bartz(
                X_context,
                one_vs_rest,
                outcome_type=["binary"] * n_context_classes,
                categorical_columns=categorical_columns,
            )
            mode = "multivariate_one_vs_rest_probit"
        return _BARTFittedModel(
            estimator=estimator,
            task="classification",
            n_save_per_chain=self._n_save_per_chain,
            classification_mode=mode,
            n_context_classes=n_context_classes,
        )

    def _sample_classification_probabilities(
        self,
        model: Any,
        X_test: Any,
    ) -> np.ndarray:
        if (
            not isinstance(model, _BARTFittedModel)
            or model.task != "classification"
            or model.n_context_classes is None
        ):
            raise TypeError("invalid fitted BART classification model")
        n_test = len(X_test)
        if model.classification_mode == "constant":
            return np.ones((self.n_draws, n_test, 1), dtype=np.float32)
        raw = self._select_draws(
            model.estimator.predict(
                X_test, kind="mean_samples"
            )
        )
        if model.classification_mode == "binary_probit":
            if raw.shape != (self.n_draws, n_test):
                raise ValueError(
                    "bartz binary probabilities have unexpected shape "
                    f"{raw.shape}"
                )
            success = np.clip(raw, 0.0, 1.0)
            return np.stack([1.0 - success, success], axis=2)
        expected = (self.n_draws, model.n_context_classes, n_test)
        if raw.shape != expected:
            raise ValueError(
                "bartz multiclass probabilities have unexpected shape "
                f"{raw.shape}; expected {expected}"
            )
        probabilities = np.moveaxis(raw, 1, 2)
        return np.clip(probabilities, 1e-12, 1.0)

    def _build_model_artifact(
        self,
        *,
        task: str,
        model: Any,
        preprocessor: BaselinePreprocessor,
        context_classes: Optional[np.ndarray] = None,
        n_classes_global: Optional[int] = None,
    ) -> BaselineModelArtifact:
        if not isinstance(model, _BARTFittedModel) or model.task != task:
            raise TypeError(f"invalid fitted BART model for task={task}")
        bartz, _, jax = self._bartz_modules()
        serialized_model: bytes | None = None
        if model.estimator is not None:
            with tempfile.TemporaryDirectory(prefix="bartz-artifact-") as tmp:
                path = Path(tmp) / "model.pkl"
                model.estimator.dump(path)
                serialized_model = path.read_bytes()
        version_info = getattr(bartz, "__version_info__", None)
        bartz_version = getattr(bartz, "__version__", None)
        if bartz_version is None and version_info is not None:
            bartz_version = ".".join(str(part) for part in version_info)
        final_hyperparameters = {
            "seed": self.seed,
            "n_trees": self.n_trees,
            "n_draws_total": self.n_draws,
            "n_save_per_chain": self._n_save_per_chain,
            "n_saved_by_bartz": self._n_save_per_chain * self.n_chains,
            "n_burn_per_chain": self.n_burn,
            "n_chains": self.n_chains,
            "device": self.device,
        }
        bundle = {
            "backend": "bartz.Bart",
            "serialization": "bartz.Bart.dump",
            "serialized_model": serialized_model,
            "preprocessor": preprocessor,
            "task": task,
            "classification_mode": model.classification_mode,
            "context_classes": (
                None
                if context_classes is None
                else np.asarray(context_classes, dtype=np.int64)
            ),
            "n_classes_global": (
                None if n_classes_global is None else int(n_classes_global)
            ),
            "final_hyperparameters": final_hyperparameters,
            "bartz_version": bartz_version or "unknown",
            "jax_version": getattr(jax, "__version__", "unknown"),
        }
        return BaselineModelArtifact(
            model_bundle=bundle,
            model_metadata={
                "implementation": "bartz.Bart",
                "bartz_version": bundle["bartz_version"],
                "jax_version": bundle["jax_version"],
                "classification_mode": model.classification_mode,
                "final_hyperparameters": final_hyperparameters,
            },
        )


def load_bartz_model(model_bundle: dict[str, Any]) -> Any | None:
    """Reload the fitted BART predictor stored in a model bundle.

    The returned value is ``None`` only for a single-class constant
    classification artifact. The returned adapter accepts the row-oriented
    output of ``model_bundle['preprocessor']`` directly.
    """
    if model_bundle.get("backend") != "bartz.Bart":
        raise ValueError("model bundle is not a bartz BART artifact")
    payload = model_bundle.get("serialized_model")
    if payload is None:
        if model_bundle.get("classification_mode") == "constant":
            return None
        raise ValueError("BART model bundle has no serialized model")
    if not isinstance(payload, bytes):
        raise TypeError("serialized BART model must be bytes")
    _, Bart, _ = BARTBaseline._bartz_modules()
    with tempfile.TemporaryDirectory(prefix="bartz-reload-") as tmp:
        path = Path(tmp) / "model.pkl"
        path.write_bytes(payload)
        return _BARTPredictorAdapter(Bart.load(path))


class RandomForestBaseline:
    """Random Forest with TabArena-style ordinal categorical preprocessing."""

    name = "random_forest"

    def __init__(
        self,
        *,
        seed: int,
        n_estimators: int = 500,
        n_jobs: int = -1,
        max_features: Optional[str | float] = None,
    ) -> None:
        if n_estimators < 2:
            raise ValueError("Random Forest needs at least two trees")
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.n_jobs = int(n_jobs)
        self.max_features = max_features

    def fit_predict_regression(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        q_grid: np.ndarray,
    ) -> BaselinePrediction:
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=RandomForestFeaturePreprocessor(),
        )

        kwargs: dict[str, object] = {
            "n_estimators": self.n_estimators,
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
        }
        if self.max_features is not None:
            kwargs["max_features"] = self.max_features
        model = RandomForestRegressor(**kwargs)

        fit_start = time.perf_counter()
        model.fit(X_context_in, np.asarray(y_context, dtype=np.float64))
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        point_pred = np.asarray(model.predict(X_test_in), dtype=np.float64)

        # Each tree prediction is one draw from the fitted forest's empirical
        # predictive distribution.  Parallelize over trees with threads, as
        # sklearn's tree prediction releases the GIL.
        from joblib import Parallel, delayed

        per_tree = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(tree.predict)(X_test_in) for tree in model.estimators_
        )
        tree_predictions = np.asarray(per_tree, dtype=np.float64)
        q = np.asarray(q_grid, dtype=np.float64)
        ppd = np.quantile(tree_predictions, q, axis=0).T
        ppd = np.maximum.accumulate(ppd, axis=1)
        predict_seconds = time.perf_counter() - predict_start

        return BaselinePrediction(
            task="regression",
            ppd_quantiles=ppd,
            point_pred=point_pred,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata={
                "final_hyperparameters": model.get_params(deep=False),
            },
            model_bundle={
                "estimator": model,
                "preprocessor": preprocessor,
                "categorical_columns": list(
                    preprocessor.categorical_columns_ or []
                ),
                "serialization_device": "cpu",
            },
        )

    def fit_predict_classification(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        *,
        n_classes_global: int,
    ) -> BaselinePrediction:
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=RandomForestFeaturePreprocessor(),
        )
        y_local, context_classes = _local_class_labels(
            y_context, n_classes_global
        )

        kwargs: dict[str, object] = {
            "n_estimators": self.n_estimators,
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
        }
        if self.max_features is not None:
            kwargs["max_features"] = self.max_features
        model = RandomForestClassifier(**kwargs)

        fit_start = time.perf_counter()
        model.fit(X_context_in, y_local)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        local_proba = model.predict_proba(X_test_in)
        proba, classes_ = _pad_probabilities(
            local_proba, context_classes, n_classes_global
        )
        predict_seconds = time.perf_counter() - predict_start

        return BaselinePrediction(
            task="classification",
            proba=proba,
            classes_=classes_,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata={
                "final_hyperparameters": model.get_params(deep=False),
            },
            model_bundle={
                "estimator": model,
                "preprocessor": preprocessor,
                "categorical_columns": list(
                    preprocessor.categorical_columns_ or []
                ),
                "context_classes": np.asarray(
                    context_classes, dtype=np.int64
                ),
                "n_classes_global": int(n_classes_global),
                "serialization_device": "cpu",
            },
        )


class XGBoostQuantileBaseline:
    """XGBoost probability classifier and direct quantile regressor.

    Regression uses XGBoost's native ``reg:quantileerror`` objective to learn
    the complete benchmark quantile grid in one fitted estimator.  The scalar
    point prediction is the integral of that fitted quantile function, matching
    the RealMLP baseline's semantics.  Classification uses the corresponding
    XGBoost classifier probabilities so both tasks share one benchmark model
    name and the existing metrics pipeline.

    XGBoost is imported lazily so the Random Forest baseline remains usable in
    lightweight environments without the optional dependency.
    """

    name = "xgboost_quantile"
    validation_fraction = 0.2

    def __init__(
        self,
        *,
        seed: int,
        n_estimators: int = 256,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        n_jobs: int = -1,
        device: str = "cpu",
        multi_strategy: str = "one_output_per_tree",
    ) -> None:
        if n_estimators < 1:
            raise ValueError("XGBoost needs at least one boosting round")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("XGBoost learning_rate must lie in (0, 1]")
        if max_depth < 0:
            raise ValueError("XGBoost max_depth must be non-negative")
        if multi_strategy not in {"one_output_per_tree", "multi_output_tree"}:
            raise ValueError(
                "XGBoost multi_strategy must be 'one_output_per_tree' or "
                "'multi_output_tree'"
            )
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.n_jobs = int(n_jobs)
        self.device = str(device)
        self.multi_strategy = multi_strategy

    @staticmethod
    def _xgboost():
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "xgboost_quantile requires xgboost>=2.0; install or upgrade "
                "the optional xgboost dependency"
            ) from exc
        return xgb

    def _common_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "tree_method": "hist",
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
            "device": self.device,
            "verbosity": 0,
        }

    @staticmethod
    def _classification_eval_metric(y: np.ndarray) -> str:
        """Pair the eval metric with XGBClassifier's automatic objective.

        ``XGBClassifier.fit`` keeps ``binary:logistic`` for two classes and
        switches to ``multi:softprob`` when ``n_classes_ > 2``.  Binary
        ``logloss`` cannot score the latter, so multiclass uses ``mlogloss``.
        """
        n_classes = int(np.unique(np.asarray(y)).size)
        return "mlogloss" if n_classes > 2 else "logloss"

    @staticmethod
    def _select_validation_round(model: Any, metric_name: str) -> int:
        """Select and persist the best checkpoint after a full-budget fit."""
        history = np.asarray(
            model.evals_result()["validation_0"][metric_name],
            dtype=np.float64,
        )
        if history.size == 0 or not np.any(np.isfinite(history)):
            raise RuntimeError(
                f"XGBoost produced no finite validation {metric_name} values"
            )
        best_index = int(np.nanargmin(history))
        # The sklearn predict methods honor this booster attribute, including
        # after the selected validation model has been serialized and reloaded.
        model.get_booster().set_attr(
            best_iteration=str(best_index),
            best_score=str(float(history[best_index])),
        )
        return best_index + 1

    def fit_predict_regression(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        q_grid: np.ndarray,
    ) -> BaselinePrediction:
        xgb = self._xgboost()
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=RandomForestFeaturePreprocessor(),
        )

        q = np.asarray(q_grid, dtype=np.float64)
        if q.ndim != 1 or q.size < 2 or not np.all(np.diff(q) > 0.0):
            raise ValueError("q_grid must contain at least two increasing levels")
        if np.any((q <= 0.0) | (q >= 1.0)):
            raise ValueError("q_grid levels must lie strictly inside (0, 1)")

        y = np.asarray(y_context, dtype=np.float64)
        train_indices, validation_indices = _shared_regression_holdout_splits(
            len(y),
            seed=self.seed,
            validation_fraction=self.validation_fraction,
        )[0]

        model = xgb.XGBRegressor(
            **self._common_kwargs(),
            objective="reg:quantileerror",
            quantile_alpha=q,
            multi_strategy=self.multi_strategy,
        )
        fit_start = time.perf_counter()
        try:
            model.fit(
                X_context_in.iloc[train_indices],
                y[train_indices],
                eval_set=[(
                    X_context_in.iloc[validation_indices],
                    y[validation_indices],
                )],
                verbose=False,
            )
        except xgb.core.XGBoostError as exc:
            if "quantile" in str(exc).lower():
                raise RuntimeError(
                    "Native multi-quantile regression requires xgboost>=2.0"
                ) from exc
            raise
        selected_round = self._select_validation_round(model, "quantile")
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        raw_ppd = np.asarray(model.predict(X_test_in), dtype=np.float64)
        raw_ppd = _coerce_quantile_predictions(
            raw_ppd,
            n_test=len(X_test_in),
            n_quantiles=len(q),
            enforce_monotone=False,
        )
        # XGBoost warns that independently fitted quantiles can cross.  Reuse
        # the benchmark's canonical monotonicity convention before computing
        # the distributional mean or any reliability metric.
        crossing_rows = np.any(np.diff(raw_ppd, axis=1) < 0.0, axis=1)
        ppd = np.maximum.accumulate(raw_ppd, axis=1)
        point_pred = _quantile_mean(ppd, q)
        predict_seconds = time.perf_counter() - predict_start

        return BaselinePrediction(
            task="regression",
            ppd_quantiles=ppd,
            point_pred=point_pred,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata={
                "final_hyperparameters": model.get_params(deep=False),
                "xgboost_version": xgb.__version__,
                "checkpoint_selection_metric": "mean_pinball",
                "max_training_rounds": self.n_estimators,
                "selected_round": selected_round,
                "validation_fraction": self.validation_fraction,
                "validation_indices": validation_indices,
                "prediction_source": "selected_validation_model",
                "full_context_refit": False,
                "quantile_crossing_rows_before_correction": int(
                    crossing_rows.sum()
                ),
                "quantile_crossing_fraction_before_correction": float(
                    crossing_rows.mean()
                ),
            },
            model_bundle={
                "estimator": model,
                "preprocessor": preprocessor,
                "categorical_columns": list(
                    preprocessor.categorical_columns_ or []
                ),
                "quantile_levels": q,
                "selected_round": selected_round,
                "validation_indices": validation_indices,
                "serialization_device": self.device,
            },
        )

    def fit_predict_classification(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        *,
        n_classes_global: int,
    ) -> BaselinePrediction:
        xgb = self._xgboost()
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=RandomForestFeaturePreprocessor(),
        )
        y_local, context_classes = _local_class_labels(
            y_context, n_classes_global
        )
        train_indices, validation_indices = (
            _shared_classification_holdout_splits(
                y_local,
                seed=self.seed,
                validation_fraction=self.validation_fraction,
            )[0]
        )

        eval_metric = self._classification_eval_metric(y_local[train_indices])
        model = xgb.XGBClassifier(
            **self._common_kwargs(),
            eval_metric=eval_metric,
        )
        fit_start = time.perf_counter()
        model.fit(
            X_context_in.iloc[train_indices],
            y_local[train_indices],
            eval_set=[(
                X_context_in.iloc[validation_indices],
                y_local[validation_indices],
            )],
            verbose=False,
        )
        selected_round = self._select_validation_round(model, eval_metric)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        if len(context_classes) == 1:
            # XGBClassifier exposes a synthetic two-column binary output for a
            # one-class fit.  The benchmark contract instead expects exactly
            # one local column before padding to the global class space.
            local_proba = np.ones((len(X_test_in), 1), dtype=np.float64)
        else:
            local_proba = np.asarray(
                model.predict_proba(X_test_in), dtype=np.float64
            )
        proba, classes_ = _pad_probabilities(
            local_proba, context_classes, n_classes_global
        )
        predict_seconds = time.perf_counter() - predict_start

        return BaselinePrediction(
            task="classification",
            proba=proba,
            classes_=classes_,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata={
                "final_hyperparameters": model.get_params(deep=False),
                "xgboost_version": xgb.__version__,
                "checkpoint_selection_metric": "log_loss",
                "max_training_rounds": self.n_estimators,
                "selected_round": selected_round,
                "validation_fraction": self.validation_fraction,
                "validation_indices": validation_indices,
                "prediction_source": "selected_validation_model",
                "full_context_refit": False,
            },
            model_bundle={
                "estimator": model,
                "preprocessor": preprocessor,
                "categorical_columns": list(
                    preprocessor.categorical_columns_ or []
                ),
                "context_classes": np.asarray(
                    context_classes, dtype=np.int64
                ),
                "n_classes_global": int(n_classes_global),
                "selected_round": selected_round,
                "validation_indices": validation_indices,
                "serialization_device": self.device,
            },
        )


XGBOOST_TABARENA_ADAPTED_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {
        "distribution": "loguniform",
        "low": 0.02,
        "high": 0.1,
    },
    "max_depth": {
        "distribution": "loguniform_int",
        "low": 4,
        "high": 10,
    },
    "min_child_weight": {
        "distribution": "loguniform",
        "low": 0.001,
        "high": 5.0,
    },
    "subsample": {
        "distribution": "uniform",
        "low": 0.6,
        "high": 1.0,
    },
    "colsample_bylevel": {
        "distribution": "uniform",
        "low": 0.6,
        "high": 1.0,
    },
    "colsample_bynode": {
        "distribution": "uniform",
        "low": 0.6,
        "high": 1.0,
    },
    "reg_alpha": {
        "distribution": "uniform",
        "low": 1e-4,
        "high": 5.0,
    },
    "reg_lambda": {
        "distribution": "uniform",
        "low": 1e-4,
        "high": 5.0,
    },
    "grow_policy": {
        "distribution": "choice",
        "values": ["depthwise", "lossguide"],
    },
    "max_leaves": {
        "distribution": "loguniform_int",
        "low": 8,
        "high": 1024,
    },
}


class XGBoostQuantileHPOBaseline(XGBoostQuantileBaseline):
    """TabArena-adapted random search with the RealMLP-HPO budget.

    The default protocol samples 50 configurations and evaluates every one on
    a shared 80/20 validation split, matching ``RealMLPHPOBaseline``'s default
    search magnitude.  Every candidate is trained for 256 boosting rounds and
    its best validation round is retained, matching RealMLP-HPO's 256-epoch
    best-checkpoint protocol.  Regression selects by mean pinball loss over the
    full benchmark quantile grid; classification selects by log loss.  As in
    the current RealMLP-HPO adapter, predictions come directly from the
    selected validation model(s), without a full-context refit.  Multiple
    validation splits are combined by arithmetic averaging.

    TabArena's native-categorical-only ``max_cat_to_onehot`` parameter is
    intentionally omitted because this benchmark ordinal-encodes categories
    before XGBoost sees them.  Its learning-rate lower bound is raised from
    0.005 to 0.02 to avoid undertraining under the 256-round budget instead of
    TabArena's much larger boosting-round cap.
    """

    name = "xgboost_quantile_hpo"
    validation_fraction = 0.2

    def __init__(
        self,
        *,
        seed: int,
        n_hyperopt_steps: int = 50,
        n_cv: int = 1,
        max_n_estimators: int = 256,
        n_jobs: int = -1,
        device: str = "cpu",
        multi_strategy: str = "one_output_per_tree",
    ) -> None:
        if n_hyperopt_steps < 1:
            raise ValueError("XGBoost HPO requires n_hyperopt_steps >= 1")
        if n_cv < 1:
            raise ValueError("XGBoost HPO requires n_cv >= 1")
        if max_n_estimators < 1:
            raise ValueError("XGBoost HPO max_n_estimators must be positive")
        super().__init__(
            seed=seed,
            n_estimators=max_n_estimators,
            learning_rate=0.3,
            max_depth=6,
            n_jobs=n_jobs,
            device=device,
            multi_strategy=multi_strategy,
        )
        self.n_hyperopt_steps = int(n_hyperopt_steps)
        self.n_cv = int(n_cv)
        self.max_n_estimators = int(max_n_estimators)

    @staticmethod
    def _loguniform(
        rng: np.random.Generator, low: float, high: float
    ) -> float:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    @classmethod
    def _sample_search_space(
        cls, rng: np.random.Generator
    ) -> dict[str, Any]:
        def loguniform_int(low: int, high: int) -> int:
            sampled = int(round(cls._loguniform(rng, float(low), float(high))))
            return int(np.clip(sampled, low, high))

        return {
            "learning_rate": cls._loguniform(rng, 0.02, 0.1),
            "max_depth": loguniform_int(4, 10),
            "min_child_weight": cls._loguniform(rng, 0.001, 5.0),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
            "colsample_bynode": float(rng.uniform(0.6, 1.0)),
            "reg_alpha": float(rng.uniform(1e-4, 5.0)),
            "reg_lambda": float(rng.uniform(1e-4, 5.0)),
            "grow_policy": str(rng.choice(["depthwise", "lossguide"])),
            "max_leaves": loguniform_int(8, 1024),
        }

    def _sample_configurations(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        return [
            self._sample_search_space(rng)
            for _ in range(self.n_hyperopt_steps)
        ]

    def _regression_splits(self, n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
        return _shared_regression_cv_splits(
            n_rows,
            seed=self.seed,
            n_cv=self.n_cv,
            validation_fraction=self.validation_fraction,
        )

    def _classification_splits(
        self, y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return _shared_classification_cv_splits(
            y,
            seed=self.seed,
            n_cv=self.n_cv,
            validation_fraction=self.validation_fraction,
        )

    def _candidate_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        kwargs = self._common_kwargs()
        kwargs.update(params)
        kwargs["n_estimators"] = self.max_n_estimators
        return kwargs

    @staticmethod
    def _mean_pinball_loss(
        y_true: np.ndarray,
        quantiles: np.ndarray,
        levels: np.ndarray,
    ) -> float:
        residual = np.asarray(y_true, dtype=np.float64)[:, None] - quantiles
        loss = np.maximum(
            levels[None, :] * residual,
            (levels[None, :] - 1.0) * residual,
        )
        return float(np.mean(loss))

    @staticmethod
    def _log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
        rows = np.arange(len(y_true))
        selected = probabilities[rows, np.asarray(y_true, dtype=np.int64)]
        return float(-np.mean(np.log(np.clip(selected, 1e-15, 1.0))))

    def fit_predict_regression(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        q_grid: np.ndarray,
    ) -> BaselinePrediction:
        xgb = self._xgboost()
        start = time.perf_counter()
        q = np.asarray(q_grid, dtype=np.float64)
        if q.ndim != 1 or q.size < 2 or not np.all(np.diff(q) > 0.0):
            raise ValueError("q_grid must contain at least two increasing levels")
        if np.any((q <= 0.0) | (q >= 1.0)):
            raise ValueError("q_grid levels must lie strictly inside (0, 1)")
        y = np.asarray(y_context, dtype=np.float64)
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=RandomForestFeaturePreprocessor(),
        )
        splits = self._regression_splits(len(y))
        configurations = self._sample_configurations()
        search_start = time.perf_counter()
        trials: list[dict[str, Any]] = []
        best_score = np.inf
        best_params: Optional[dict[str, Any]] = None
        best_rounds: list[int] = []
        best_models: list[Any] = []

        for candidate_index, params in enumerate(configurations):
            fold_scores: list[float] = []
            fold_rounds: list[int] = []
            candidate_models: list[Any] = []
            for train_indices, validation_indices in splits:
                X_train = X_context_in.iloc[train_indices]
                X_validation = X_context_in.iloc[validation_indices]
                model = xgb.XGBRegressor(
                    **self._candidate_kwargs(params),
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    multi_strategy=self.multi_strategy,
                )
                model.fit(
                    X_train,
                    y[train_indices],
                    eval_set=[(X_validation, y[validation_indices])],
                    verbose=False,
                )
                selected_round = self._select_validation_round(
                    model, "quantile"
                )
                raw = np.asarray(model.predict(X_validation), dtype=np.float64)
                predictions = _coerce_quantile_predictions(
                    raw,
                    n_test=len(validation_indices),
                    n_quantiles=len(q),
                )
                fold_scores.append(
                    self._mean_pinball_loss(
                        y[validation_indices], predictions, q
                    )
                )
                fold_rounds.append(selected_round)
                candidate_models.append(model)
            score = float(np.mean(fold_scores))
            trials.append(
                {
                    "candidate": candidate_index,
                    "score": score,
                    "trained_rounds": [self.max_n_estimators] * len(splits),
                    "selected_rounds": fold_rounds,
                    "params": dict(params),
                }
            )
            if score < best_score:
                best_score = score
                best_params = dict(params)
                best_rounds = list(fold_rounds)
                best_models = candidate_models

        if best_params is None:  # pragma: no cover - guarded by validation
            raise RuntimeError("XGBoost HPO did not evaluate any configuration")
        search_seconds = time.perf_counter() - search_start
        predict_seconds = 0.0
        fold_predictions: list[np.ndarray] = []
        for model in best_models:
            predict_start = time.perf_counter()
            raw = np.asarray(model.predict(X_test_in), dtype=np.float64)
            predict_seconds += time.perf_counter() - predict_start
            fold_predictions.append(
                _coerce_quantile_predictions(
                    raw,
                    n_test=len(X_test),
                    n_quantiles=len(q),
                    enforce_monotone=False,
                )
            )
        raw_ppd = np.mean(np.stack(fold_predictions, axis=0), axis=0)
        crossing_rows = np.any(np.diff(raw_ppd, axis=1) < 0.0, axis=1)
        ppd = np.maximum.accumulate(raw_ppd, axis=1)
        point_pred = _quantile_mean(ppd, q)
        one_fold = len(best_models) == 1
        validation_indices_payload = (
            splits[0][1]
            if self.n_cv == 1
            else np.stack(
                [validation for _, validation in splits], axis=0
            )
        )
        return BaselinePrediction(
            task="regression",
            ppd_quantiles=ppd,
            point_pred=point_pred,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": search_seconds,
                "predict": predict_seconds,
                "hpo_search": search_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata={
                "xgboost_version": xgb.__version__,
                "hpo_space_name": "tabarena_xgboost_adapted",
                "hpo_selection_metric": "mean_pinball",
                "hpo_selected_params": best_params,
                "hpo_selection_score": best_score,
                "hpo_selected_rounds": best_rounds,
                "hpo_max_training_rounds": self.max_n_estimators,
                "hpo_round_selection": "full_budget_then_best_validation_round",
                "hpo_n_hyperopt_steps": self.n_hyperopt_steps,
                "hpo_n_cv": self.n_cv,
                "hpo_validation_scheme": (
                    "holdout" if self.n_cv == 1 else "k_fold"
                ),
                "hpo_validation_fraction": (
                    self.validation_fraction
                    if self.n_cv == 1
                    else len(splits[0][1]) / len(y)
                ),
                "hpo_validation_indices": validation_indices_payload,
                "prediction_source": "selected_configuration_cv_ensemble",
                "full_context_refit": False,
                "hpo_trials": trials,
                "quantile_crossing_rows_before_correction": int(
                    crossing_rows.sum()
                ),
                "quantile_crossing_fraction_before_correction": float(
                    crossing_rows.mean()
                ),
            },
            model_bundle={
                "estimator": best_models[0] if one_fold else None,
                "preprocessor": preprocessor,
                "ensemble_estimators": best_models,
                "ensemble_selected_rounds": best_rounds,
                "ensemble_method": "arithmetic_mean",
                "quantile_levels": q,
                "validation_indices": validation_indices_payload,
                "serialization_device": self.device,
            },
        )

    def fit_predict_classification(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        *,
        n_classes_global: int,
    ) -> BaselinePrediction:
        xgb = self._xgboost()
        start = time.perf_counter()
        y, context_classes = _local_class_labels(
            y_context, n_classes_global
        )
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(
            X_context,
            X_test,
            preprocessor=RandomForestFeaturePreprocessor(),
        )
        splits = self._classification_splits(y)
        configurations = self._sample_configurations()
        search_start = time.perf_counter()
        trials: list[dict[str, Any]] = []
        best_score = np.inf
        best_params: Optional[dict[str, Any]] = None
        best_rounds: list[int] = []
        best_models: list[Any] = []

        for candidate_index, params in enumerate(configurations):
            fold_scores: list[float] = []
            fold_rounds: list[int] = []
            candidate_models: list[Any] = []
            for train_indices, validation_indices in splits:
                X_train = X_context_in.iloc[train_indices]
                X_validation = X_context_in.iloc[validation_indices]
                eval_metric = self._classification_eval_metric(
                    y[train_indices]
                )
                model = xgb.XGBClassifier(
                    **self._candidate_kwargs(params),
                    eval_metric=eval_metric,
                )
                model.fit(
                    X_train,
                    y[train_indices],
                    eval_set=[(X_validation, y[validation_indices])],
                    verbose=False,
                )
                selected_round = self._select_validation_round(
                    model, eval_metric
                )
                if len(np.unique(y[train_indices])) == 1:
                    probabilities = np.ones(
                        (len(validation_indices), 1), dtype=np.float64
                    )
                else:
                    probabilities = np.asarray(
                        model.predict_proba(X_validation), dtype=np.float64
                    )
                fold_scores.append(
                    self._log_loss(y[validation_indices], probabilities)
                )
                fold_rounds.append(selected_round)
                candidate_models.append(model)
            score = float(np.mean(fold_scores))
            trials.append(
                {
                    "candidate": candidate_index,
                    "score": score,
                    "trained_rounds": [self.max_n_estimators] * len(splits),
                    "selected_rounds": fold_rounds,
                    "params": dict(params),
                }
            )
            if score < best_score:
                best_score = score
                best_params = dict(params)
                best_rounds = list(fold_rounds)
                best_models = candidate_models

        if best_params is None:  # pragma: no cover - guarded by validation
            raise RuntimeError("XGBoost HPO did not evaluate any configuration")
        search_seconds = time.perf_counter() - search_start
        predict_seconds = 0.0
        fold_probabilities: list[np.ndarray] = []
        for model in best_models:
            predict_start = time.perf_counter()
            if len(context_classes) == 1:
                local_proba = np.ones((len(X_test), 1), dtype=np.float64)
            else:
                local_proba = np.asarray(
                    model.predict_proba(X_test_in), dtype=np.float64
                )
            predict_seconds += time.perf_counter() - predict_start
            fold_probabilities.append(local_proba)
        local_proba = np.mean(np.stack(fold_probabilities, axis=0), axis=0)
        proba, classes_ = _pad_probabilities(
            local_proba, context_classes, n_classes_global
        )
        one_fold = len(best_models) == 1
        validation_indices_payload = (
            splits[0][1]
            if self.n_cv == 1
            else np.stack(
                [validation for _, validation in splits], axis=0
            )
        )
        return BaselinePrediction(
            task="classification",
            proba=proba,
            classes_=classes_,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": search_seconds,
                "predict": predict_seconds,
                "hpo_search": search_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata={
                "xgboost_version": xgb.__version__,
                "hpo_space_name": "tabarena_xgboost_adapted",
                "hpo_selection_metric": "log_loss",
                "hpo_selected_params": best_params,
                "hpo_selection_score": best_score,
                "hpo_selected_rounds": best_rounds,
                "hpo_max_training_rounds": self.max_n_estimators,
                "hpo_round_selection": "full_budget_then_best_validation_round",
                "hpo_n_hyperopt_steps": self.n_hyperopt_steps,
                "hpo_n_cv": self.n_cv,
                "hpo_validation_scheme": (
                    "holdout" if self.n_cv == 1 else "k_fold"
                ),
                "hpo_validation_fraction": (
                    self.validation_fraction
                    if self.n_cv == 1
                    else len(splits[0][1]) / len(y)
                ),
                "hpo_validation_indices": validation_indices_payload,
                "prediction_source": "selected_configuration_cv_ensemble",
                "full_context_refit": False,
                "hpo_trials": trials,
            },
            model_bundle={
                "estimator": best_models[0] if one_fold else None,
                "preprocessor": preprocessor,
                "ensemble_estimators": best_models,
                "ensemble_selected_rounds": best_rounds,
                "ensemble_method": "arithmetic_mean",
                "context_classes": np.asarray(
                    context_classes, dtype=np.int64
                ),
                "n_classes_global": int(n_classes_global),
                "validation_indices": validation_indices_payload,
                "serialization_device": self.device,
            },
        )


def _import_realmlp_classes(*, hpo: bool = False):
    """Import public PyTabKit TD or HPO estimators with a useful error."""
    try:
        if hpo:
            from pytabkit import RealMLP_HPO_Classifier, RealMLP_HPO_Regressor

            return RealMLP_HPO_Classifier, RealMLP_HPO_Regressor
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
    except ImportError as exc:
        raise ImportError(
            "RealMLP requires PyTabKit. Install it in the server environment "
            "with `pip install pytabkit` (and install a CUDA-enabled PyTorch "
            "build separately)."
        ) from exc
    return RealMLP_TD_Classifier, RealMLP_TD_Regressor


class RealMLPBaseline:
    """RealMLP-TD baseline using PyTabKit's sklearn-compatible interface."""

    name = "realmlp"
    validation_fraction = 0.2

    def __init__(
        self,
        *,
        seed: int,
        device: str = "cuda",
        n_epochs: Optional[int] = 256,
        n_cv: int = 1,
        n_refit: int = 0,
        n_ens: Optional[int] = 1,
        n_threads: Optional[int] = None,
        verbosity: int = 0,
    ) -> None:
        self.seed = int(seed)
        self.device = device
        self.n_epochs = None if n_epochs is None else int(n_epochs)
        self.n_cv = int(n_cv)
        self.n_refit = int(n_refit)
        self.n_ens = None if n_ens is None else int(n_ens)
        self.n_threads = None if n_threads is None else int(n_threads)
        self.verbosity = int(verbosity)

    def _common_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "device": self.device,
            "random_state": self.seed,
            "n_cv": self.n_cv,
            "n_refit": self.n_refit,
            "verbosity": self.verbosity,
        }
        if self.n_epochs is not None:
            kwargs["n_epochs"] = self.n_epochs
        if self.n_ens is not None:
            kwargs["n_ens"] = self.n_ens
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        return kwargs

    @contextmanager
    def _model_kwargs(self):
        """Yield constructor kwargs while any model backing store is alive."""
        yield self._common_kwargs()

    @staticmethod
    def _prediction_metadata(model: Any) -> dict[str, Any]:
        fit_params = getattr(model, "fit_params_", None)
        return {} if fit_params is None else {"selected_fit_params": fit_params}

    def _prediction_source(self) -> str:
        return (
            "selected_validation_model"
            if self.n_cv == 1
            else "cv_ensemble"
        )

    def _validation_metadata(
        self,
        validation_indices: np.ndarray,
        *,
        n_rows: int,
    ) -> dict[str, Any]:
        fold_size = (
            len(validation_indices)
            if validation_indices.ndim == 1
            else validation_indices.shape[1]
        )
        return {
            "validation_scheme": (
                "holdout" if self.n_cv == 1 else "k_fold"
            ),
            "validation_fraction": (
                self.validation_fraction
                if self.n_cv == 1
                else fold_size / n_rows
            ),
            "validation_indices": validation_indices,
            "prediction_source": self._prediction_source(),
            "full_context_refit": False,
        }

    @staticmethod
    def _prepare_model_for_serialization(model: Any) -> str:
        """Prefer portable CPU checkpoints when supported by PyTabKit."""
        move = getattr(model, "to", None)
        if callable(move):
            try:
                move("cpu")
                return "cpu"
            except Exception:
                # Older PyTabKit releases did not reliably support moving a
                # fitted sklearn wrapper. Its pickle is still valid on the
                # original training device, so retain it with explicit metadata.
                return "training_device"
        return "unspecified"

    def _finalize_fitted_model(self, model: Any) -> None:
        """Hook for model families with external checkpoint housekeeping."""

    def _external_checkpoint_dir(self) -> str | None:
        path = getattr(self, "checkpoint_dir", None)
        return None if path is None else str(Path(path).resolve())

    @staticmethod
    def _model_classes():
        return _import_realmlp_classes()

    def _shared_validation_indices(
        self,
        y_context: np.ndarray,
        *,
        classification: bool,
    ) -> np.ndarray:
        """Return the same holdout/K-fold validation indices used by XGBoost."""
        if classification:
            splits = _shared_classification_cv_splits(
                y_context,
                seed=self.seed,
                n_cv=self.n_cv,
                validation_fraction=self.validation_fraction,
            )
        else:
            splits = _shared_regression_cv_splits(
                len(np.asarray(y_context)),
                seed=self.seed,
                n_cv=self.n_cv,
                validation_fraction=self.validation_fraction,
            )
        if self.n_cv == 1:
            return splits[0][1]
        return np.stack(
            [validation for _, validation in splits], axis=0
        )

    @staticmethod
    def _fit(
        model,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        categorical_columns: list[str],
        validation_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Fit while remaining compatible with older PyTabKit releases."""
        fit_kwargs: dict[str, Any] = {}
        if validation_indices is not None:
            fit_kwargs["val_idxs"] = validation_indices
        try:
            model.fit(
                X_context,
                y_context,
                cat_col_names=categorical_columns,
                **fit_kwargs,
            )
        except TypeError as exc:
            # Older releases did not expose cat_col_names in fit().  Do not
            # hide arbitrary TypeErrors thrown *inside* training.
            if "cat_col_names" not in str(exc):
                raise
            model.fit(X_context, y_context, **fit_kwargs)

    def fit_predict_regression(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        q_grid: np.ndarray,
    ) -> BaselinePrediction:
        _, regressor_cls = self._model_classes()
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(X_context, X_test)

        q = np.asarray(q_grid, dtype=np.float64)
        quantile_text = ",".join(f"{level:.12g}" for level in q)
        metric = f"multi_pinball({quantile_text})"
        validation_indices = self._shared_validation_indices(
            y_context,
            classification=False,
        )
        with self._model_kwargs() as model_kwargs:
            model = regressor_cls(
                **model_kwargs,
                train_metric_name=metric,
                val_metric_name=metric,
            )

            fit_start = time.perf_counter()
            self._fit(
                model,
                X_context_in,
                np.asarray(y_context, dtype=np.float64),
                preprocessor.categorical_columns_ or [],
                validation_indices,
            )
            fit_seconds = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            raw_prediction = model.predict(X_test_in)
            ppd = _coerce_quantile_predictions(
                raw_prediction,
                n_test=len(X_test_in),
                n_quantiles=len(q),
            )
            point_pred = _quantile_mean(ppd, q)
            predict_seconds = time.perf_counter() - predict_start
            model_metadata = self._prediction_metadata(model)
            if validation_indices is not None:
                model_metadata.update(
                    self._validation_metadata(
                        validation_indices,
                        n_rows=len(y_context),
                    )
                )
            serialization_device = self._prepare_model_for_serialization(model)
            self._finalize_fitted_model(model)
            model_bundle = {
                "estimator": model,
                "preprocessor": preprocessor,
                "categorical_columns": list(
                    preprocessor.categorical_columns_ or []
                ),
                "quantile_levels": q,
                "serialization_device": serialization_device,
            }
        self._release_gpu_cache()
        return BaselinePrediction(
            task="regression",
            ppd_quantiles=ppd,
            point_pred=point_pred,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata=model_metadata,
            model_bundle=model_bundle,
            external_checkpoint_dir=self._external_checkpoint_dir(),
        )

    def fit_predict_classification(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        *,
        n_classes_global: int,
    ) -> BaselinePrediction:
        classifier_cls, _ = self._model_classes()
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(X_context, X_test)
        y_local, context_classes = _local_class_labels(
            y_context, n_classes_global
        )
        validation_indices = self._shared_validation_indices(
            y_local,
            classification=True,
        )

        with self._model_kwargs() as model_kwargs:
            model = classifier_cls(
                **model_kwargs,
                val_metric_name="cross_entropy",
                use_ls=False,
            )
            fit_start = time.perf_counter()
            self._fit(
                model,
                X_context_in,
                y_local,
                preprocessor.categorical_columns_ or [],
                validation_indices,
            )
            fit_seconds = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            local_proba = np.asarray(
                model.predict_proba(X_test_in), dtype=np.float64
            )
            proba, classes_ = _pad_probabilities(
                local_proba, context_classes, n_classes_global
            )
            predict_seconds = time.perf_counter() - predict_start
            model_metadata = self._prediction_metadata(model)
            if validation_indices is not None:
                model_metadata.update(
                    self._validation_metadata(
                        validation_indices,
                        n_rows=len(y_local),
                    )
                )
            serialization_device = self._prepare_model_for_serialization(model)
            self._finalize_fitted_model(model)
            model_bundle = {
                "estimator": model,
                "preprocessor": preprocessor,
                "categorical_columns": list(
                    preprocessor.categorical_columns_ or []
                ),
                "context_classes": np.asarray(
                    context_classes, dtype=np.int64
                ),
                "n_classes_global": int(n_classes_global),
                "serialization_device": serialization_device,
            }
        self._release_gpu_cache()
        return BaselinePrediction(
            task="classification",
            proba=proba,
            classes_=classes_,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata=model_metadata,
            model_bundle=model_bundle,
            external_checkpoint_dir=self._external_checkpoint_dir(),
        )

    @staticmethod
    def _release_gpu_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class RealMLPHPOBaseline(RealMLPBaseline):
    """RealMLP random-search HPO using PyTabKit's ``default`` space.

    The benchmark objectives remain unchanged: cross-entropy selects the
    classification configuration, while the dense multi-pinball objective
    used by RealMLP-TD selects the regression configuration. Predictions use
    only the selected configuration rather than a weighted HPO ensemble.
    """

    name = "realmlp_hpo"

    def __init__(
        self,
        *,
        seed: int,
        device: str = "cuda",
        n_epochs: Optional[int] = 256,
        n_cv: int = 1,
        n_refit: int = 0,
        n_hyperopt_steps: int = 50,
        n_threads: Optional[int] = None,
        verbosity: int = 0,
        tmp_root: Optional[str | Path] = None,
        time_limit_s: Optional[float] = None,
    ) -> None:
        super().__init__(
            seed=seed,
            device=device,
            n_epochs=n_epochs,
            n_cv=n_cv,
            n_refit=n_refit,
            n_ens=None,
            n_threads=n_threads,
            verbosity=verbosity,
        )
        if n_hyperopt_steps < 1:
            raise ValueError("RealMLP HPO requires n_hyperopt_steps >= 1")
        if n_refit != 0:
            raise ValueError(
                "PyTabKit RealMLP-HPO does not currently implement refitting; "
                "use n_refit=0 so predictions come from the selected CV model"
            )
        if time_limit_s is not None and time_limit_s <= 0:
            raise ValueError("RealMLP HPO time_limit_s must be positive")
        self.n_hyperopt_steps = int(n_hyperopt_steps)
        self.tmp_root = None if tmp_root is None else Path(tmp_root)
        self.time_limit_s = (
            None if time_limit_s is None else float(time_limit_s)
        )
        self.checkpoint_dir: Optional[Path] = None

    def set_checkpoint_dir(self, path: str | Path) -> None:
        """Use a persistent store-owned directory for the selected checkpoint."""
        self.checkpoint_dir = Path(path)

    @staticmethod
    def _model_classes():
        return _import_realmlp_classes(hpo=True)

    @contextmanager
    def _model_kwargs(self):
        # PyTabKit otherwise retains every candidate in memory. The selected
        # model is predicted before this temporary backing store is removed.
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            kwargs = self._common_kwargs()
            kwargs.update(
                {
                    "hpo_space_name": "default",
                    "n_hyperopt_steps": self.n_hyperopt_steps,
                    "tmp_folder": str(self.checkpoint_dir),
                    "use_caruana_ensembling": False,
                }
            )
            if self.time_limit_s is not None:
                kwargs["time_limit_s"] = self.time_limit_s
            yield kwargs
            return

        if self.tmp_root is not None:
            self.tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="realmlp_hpo_",
            dir=None if self.tmp_root is None else str(self.tmp_root),
        ) as tmp_folder:
            kwargs = self._common_kwargs()
            kwargs.update(
                {
                    "hpo_space_name": "default",
                    "n_hyperopt_steps": self.n_hyperopt_steps,
                    "tmp_folder": tmp_folder,
                    "use_caruana_ensembling": False,
                }
            )
            if self.time_limit_s is not None:
                kwargs["time_limit_s"] = self.time_limit_s
            yield kwargs

    @staticmethod
    def _prediction_metadata(model: Any) -> dict[str, Any]:
        fit_params = getattr(model, "fit_params_", None)
        if fit_params is None:
            return {}
        return {"hpo_selected_fit_params": fit_params}

    def _prediction_source(self) -> str:
        return "selected_configuration_cv_ensemble"

    @staticmethod
    def _prepare_model_for_serialization(model: Any) -> str:
        """Move only the selected HPO candidate to CPU when possible."""
        fit_params = getattr(model, "fit_params_", None) or {}
        best_idx = fit_params.get("best_alg_idx")
        interface = getattr(model, "alg_interface_", None)
        contexts = getattr(interface, "alg_contexts_", None)
        if best_idx is None or contexts is None:
            return "training_device"
        try:
            with contexts[int(best_idx)] as selected:
                move = getattr(selected, "to", None)
                if callable(move):
                    move("cpu")
                    return "cpu"
        except Exception:
            pass
        return "training_device"

    def _prune_unselected_checkpoints(self, model: Any) -> None:
        """Remove candidate weights other than the selected HPO configuration."""
        if self.checkpoint_dir is None:
            return
        fit_params = getattr(model, "fit_params_", None) or {}
        best_idx = fit_params.get("best_alg_idx")
        if best_idx is None:
            raise ValueError("RealMLP HPO did not expose best_alg_idx")
        cv_dir = self.checkpoint_dir / "cv"
        if not cv_dir.is_dir():
            return
        keep = {str(int(best_idx)), f"model_{int(best_idx)}"}
        for child in cv_dir.iterdir():
            name = child.name
            is_candidate = name.isdigit() or (
                name.startswith("model_") and name[6:].isdigit()
            )
            if not is_candidate or name in keep:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _finalize_fitted_model(self, model: Any) -> None:
        self._prune_unselected_checkpoints(model)

    def fit_predict_classification(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_test: pd.DataFrame,
        *,
        n_classes_global: int,
    ) -> BaselinePrediction:
        """Run the unmodified default space, including its ls_eps search."""
        classifier_cls, _ = self._model_classes()
        start = time.perf_counter()
        (
            preprocessor,
            X_context_in,
            X_test_in,
            preprocess_seconds,
        ) = _prepare_baseline_features(X_context, X_test)
        y_local, context_classes = _local_class_labels(
            y_context, n_classes_global
        )
        validation_indices = self._shared_validation_indices(
            y_local,
            classification=True,
        )

        with self._model_kwargs() as model_kwargs:
            model = classifier_cls(
                **model_kwargs,
                val_metric_name="cross_entropy",
            )
            fit_start = time.perf_counter()
            self._fit(
                model,
                X_context_in,
                y_local,
                preprocessor.categorical_columns_ or [],
                validation_indices,
            )
            fit_seconds = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            local_proba = np.asarray(
                model.predict_proba(X_test_in), dtype=np.float64
            )
            proba, classes_ = _pad_probabilities(
                local_proba, context_classes, n_classes_global
            )
            predict_seconds = time.perf_counter() - predict_start
            model_metadata = self._prediction_metadata(model)
            model_metadata.update(
                self._validation_metadata(
                    validation_indices,
                    n_rows=len(y_local),
                )
            )
            serialization_device = self._prepare_model_for_serialization(model)
            self._finalize_fitted_model(model)
            model_bundle = {
                "estimator": model,
                "preprocessor": preprocessor,
                "categorical_columns": list(
                    preprocessor.categorical_columns_ or []
                ),
                "context_classes": np.asarray(
                    context_classes, dtype=np.int64
                ),
                "n_classes_global": int(n_classes_global),
                "serialization_device": serialization_device,
            }

        self._release_gpu_cache()
        return BaselinePrediction(
            task="classification",
            proba=proba,
            classes_=classes_,
            timing_seconds={
                "preprocess": preprocess_seconds,
                "fit": fit_seconds,
                "predict": predict_seconds,
                "model_total": time.perf_counter() - start,
            },
            model_metadata=model_metadata,
            model_bundle=model_bundle,
            external_checkpoint_dir=self._external_checkpoint_dir(),
        )


__all__ = [
    "BARTBaseline",
    "BARTFeaturePreprocessor",
    "BaselineFeaturePreprocessor",
    "BaselineModelArtifact",
    "BaselinePreprocessor",
    "BaselinePreprocessorFactory",
    "BaselinePrediction",
    "PosteriorPredictiveBaseline",
    "RandomForestFeaturePreprocessor",
    "RandomForestBaseline",
    "RealMLPBaseline",
    "RealMLPHPOBaseline",
    "XGBOOST_TABARENA_ADAPTED_SEARCH_SPACE",
    "XGBoostQuantileBaseline",
    "XGBoostQuantileHPOBaseline",
    "load_bartz_model",
]
