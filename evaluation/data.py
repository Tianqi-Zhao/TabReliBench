"""OpenML loading + canonical train/test/context splitting.

A single ``DatasetLoader`` reproduces the splits that
``evaluate_vanilla_tabpfn.py`` historically performed inline. All four
upstream scripts (predict / metrics / dataset-features / instance-features)
must agree on these splits - this module is the only place where the
relevant RNG seeds and slicing live.

Task awareness
--------------
The loader auto-detects whether the OpenML dataset is a regression or
classification task from the dtype of its default-target column:

* boolean target            -> ``task='classification'``  (e.g. OpenML bool labels)
* numeric target            -> ``task='regression'``  (legacy behaviour)
* categorical / string / int label codes -> ``task='classification'``

For classification the target is integer-encoded with
:class:`sklearn.preprocessing.LabelEncoder`; the resulting ``classes_``
array (in encoder order) is exposed on :class:`SplitData` so downstream
code can map the integer indices back to the original labels stored on
the model's ``classes_`` attribute.

Classification sampling defaults to ``stratified_v1``; override via
``CLASSIFICATION_SAMPLING_PROTOCOL`` or ``classification_sampling``. It
retains all classes in the capped sample, training, test, and context sets
when feasible; impossible coverage raises rather than dropping a class.
Regression and the explicitly selected ``legacy`` protocol preserve their original RNGs.

LRU cache: the raw loading helper uses ``functools.lru_cache`` so that
repeated calls for the same ``(dataset_id, seed)`` do not re-download from
OpenML. The key includes ``max_n``, task, and sampling protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from openml.datasets import get_dataset
from sklearn.preprocessing import LabelEncoder

from .spec import TASK_CLASSIFICATION, TASK_REGRESSION, ExperimentSpec
from .sampling import LEGACY, STRATIFIED, classification_protocol, stratified_indices


@dataclass
class SplitData:
    """Bundle returned by ``DatasetLoader.load_for_spec``.

    For classification the ``y_*`` arrays are *integer-encoded* labels
    (``np.int64``); the original class labels in encoder order are
    available as :attr:`classes_`. For regression ``classes_`` is ``None``
    and ``y_*`` are float64 as before.
    """
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test:  pd.DataFrame
    y_test:  np.ndarray
    X_ctx:   pd.DataFrame
    y_ctx:   np.ndarray
    feature_names: list[str]
    task:    str = TASK_REGRESSION
    classes_: Optional[np.ndarray] = None
    sampling_metadata: dict = field(default_factory=dict)


def context_size(n_train: int, ratio: float) -> int:
    """Context row count implied by *ratio* (matches :meth:`DatasetLoader.context_split`)."""
    return max(1, int(ratio * n_train))


class DatasetLoader:
    """Load + split an OpenML dataset deterministically from a seed."""

    def __init__(self, max_n: int = 10_000, *, classification_sampling: str | None = None) -> None:
        self.max_n = max_n
        self.classification_sampling = classification_protocol(classification_sampling)

    def load_raw(
        self,
        dataset_id: int,
        seed: int,
        task: str = "auto",
    ) -> tuple[pd.DataFrame, np.ndarray, list[str], str, Optional[np.ndarray]]:
        """Return ``(X_df, y, feature_names, task, classes_)``.

        ``task='auto'`` (default) detects regression vs classification from
        the OpenML target dtype. ``task='regression'`` / ``'classification'``
        forces the corresponding behaviour.
        """
        return _load_raw_cached(dataset_id, seed, self.max_n, task, self.classification_sampling)

    @staticmethod
    def train_test_split(
        X_df: pd.DataFrame, y: np.ndarray, seed: int,
        *, task: str = TASK_REGRESSION, classification_sampling: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """50/50 split, optionally stratified for classification; seed + 1.

        Direct callers must pass ``task='classification'`` to stratify.
        ``load_for_spec`` supplies the resolved task automatically.
        """
        rng = np.random.default_rng(seed + 1)
        mid = len(y) // 2
        if task == TASK_CLASSIFICATION and classification_protocol(classification_sampling) == STRATIFIED:
            train_idx = stratified_indices(y, mid, rng, minimum=1, reserve=1)
            in_test = np.ones(len(y), dtype=bool)
            in_test[train_idx] = False
            test_idx = rng.permutation(np.flatnonzero(in_test))
        else:
            idx = rng.permutation(len(y))
            train_idx, test_idx = idx[:mid], idx[mid:]
        X_train = X_df.iloc[train_idx].reset_index(drop=True)
        X_test  = X_df.iloc[test_idx].reset_index(drop=True)
        return X_train, y[train_idx], X_test, y[test_idx]

    @staticmethod
    def context_split(
        X_train: pd.DataFrame, y_train: np.ndarray, ratio: float, seed: int,
        *, task: str = TASK_REGRESSION, classification_sampling: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Subsample ``ratio`` of the training set as the model's context.

        RNG seed = ``seed + 2`` (legacy, matches evaluate_vanilla_tabpfn).
        """
        rng = np.random.default_rng(seed + 2)
        n_ctx = context_size(len(y_train), ratio)
        if task == TASK_CLASSIFICATION and classification_protocol(classification_sampling) == STRATIFIED:
            if not 0 < ratio <= 1:
                raise ValueError("Context ratio must lie in (0, 1].")
            ctx_idx = stratified_indices(y_train, n_ctx, rng, minimum=1)
        else:
            ctx_idx = rng.choice(len(y_train), size=n_ctx, replace=False)
        return X_train.iloc[ctx_idx].reset_index(drop=True), y_train[ctx_idx]

    def load_for_spec(
        self, spec: ExperimentSpec, *, task: str = "auto",
    ) -> SplitData:
        """One-shot helper returning every split a downstream pipeline needs.

        ``task`` defaults to ``'auto'`` (detect from target dtype). Pass
        an explicit task to force a particular interpretation; an explicit
        task that disagrees with what the data looks like raises
        :class:`ValueError`.
        """
        X_df, y, names, detected_task, classes_ = self.load_raw(
            spec.dataset_id, spec.seed, task=task,
        )
        split_options = dict(task=detected_task, classification_sampling=self.classification_sampling)
        X_train, y_train, X_test, y_test = self.train_test_split(X_df, y, spec.seed, **split_options)
        X_ctx, y_ctx = self.context_split(X_train, y_train, spec.ratio, spec.seed, **split_options)
        metadata = dict(X_df.attrs.get("sampling_metadata", {}))
        if detected_task == TASK_CLASSIFICATION:
            k = len(classes_)
            metadata.update({
                "train_class_counts": np.bincount(y_train, minlength=k).tolist(),
                "test_class_counts": np.bincount(y_test, minlength=k).tolist(),
                "context_class_counts": np.bincount(y_ctx, minlength=k).tolist(),
            })
        return SplitData(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            X_ctx=X_ctx,     y_ctx=y_ctx,
            feature_names=names,
            task=detected_task,
            classes_=classes_,
            sampling_metadata=metadata,
        )


def _detect_task(y_series: pd.Series) -> str:
    """Heuristic: bool / non-numeric -> classification; other numeric -> regression.

    OpenML stores categorical targets as pandas ``category`` dtype (or
    object/string for older datasets); some binary classification sets use
    ``bool`` (e.g. TabArena ``online_shoppers_intention``). Regression
    targets come through as a numeric dtype that is not boolean.
    """
    if pd.api.types.is_bool_dtype(y_series):
        return TASK_CLASSIFICATION
    if pd.api.types.is_numeric_dtype(y_series) and not isinstance(
        y_series.dtype, pd.CategoricalDtype,
    ):
        return TASK_REGRESSION
    return TASK_CLASSIFICATION


@lru_cache(maxsize=64)
def _load_raw_cached(
    dataset_id: int, seed: int, max_n: int, task: str, sampling_protocol: str = STRATIFIED,
) -> tuple[pd.DataFrame, np.ndarray, list[str], str, Optional[np.ndarray]]:
    dataset = get_dataset(dataset_id, download_data=True)
    target = dataset.default_target_attribute
    X_df, y_series, _, _ = dataset.get_data(target=target)

    columns = [str(c) for c in X_df.columns]

    detected = _detect_task(y_series)
    if task == "auto":
        resolved = detected
    elif task in (TASK_REGRESSION, TASK_CLASSIFICATION):
        if task == TASK_REGRESSION and detected == TASK_CLASSIFICATION:
            raise ValueError(
                f"Dataset {dataset_id}: target column {target!r} is "
                f"non-numeric (dtype={y_series.dtype}); cannot force "
                f"task='regression'."
            )
        resolved = task
    else:
        raise ValueError(
            f"task must be 'auto', 'regression', or 'classification'; "
            f"got {task!r}"
        )

    if resolved == TASK_REGRESSION:
        y = y_series.values.astype(np.float64)
        finite = np.isfinite(y)
        if not finite.all():
            n_bad = int((~finite).sum())
            X_df = X_df.loc[finite].reset_index(drop=True)
            y = y[finite]
            if len(y) == 0:
                raise ValueError(
                    f"Dataset {dataset_id}: all {n_bad} target values "
                    f"are NaN/inf."
                )
        classes_: Optional[np.ndarray] = None
    else:
        # Drop rows whose label is missing (NaN / pd.NA) before encoding.
        valid = y_series.notna().to_numpy()
        if not valid.all():
            X_df = X_df.loc[valid].reset_index(drop=True)
            y_series = y_series.loc[valid].reset_index(drop=True)
            if len(y_series) == 0:
                raise ValueError(
                    f"Dataset {dataset_id}: every target label is missing."
                )
        # Cast categorical / object to a plain object array so the encoder
        # always sees the underlying labels (not the category codes).
        raw_labels = np.asarray(y_series.values)
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_labels).astype(np.int64)
        classes_ = np.asarray(encoder.classes_)
        if classes_.size < 2:
            raise ValueError(
                f"Dataset {dataset_id}: classification task has "
                f"{classes_.size} unique label(s); need >= 2."
            )

    raw_counts = np.bincount(y, minlength=len(classes_)) if classes_ is not None else None
    rng = np.random.default_rng(seed)
    if len(y) > max_n:
        if resolved == TASK_CLASSIFICATION and sampling_protocol == STRATIFIED:
            idx = stratified_indices(y, max_n, rng, minimum=2)
        else:
            idx = rng.choice(len(y), size=max_n, replace=False)
        X_df = X_df.iloc[idx].reset_index(drop=True)
        y = y[idx]

    if resolved == TASK_CLASSIFICATION:
        # A private copy keeps the offline/OpenML cache frame unmodified.
        X_df = X_df.copy()
        X_df.attrs["sampling_metadata"] = {
            "sampling_protocol": sampling_protocol,
            "max_n": int(max_n),
            "seed": int(seed),
            "class_labels": [str(label) for label in classes_],
            "raw_class_counts": raw_counts.tolist(),
            "sample_class_counts": np.bincount(y, minlength=len(classes_)).tolist(),
            "class_minimum_per_sample": 2 if sampling_protocol == STRATIFIED else 0,
            "class_minimum_per_train_test": 1 if sampling_protocol == STRATIFIED else 0,
        }

    return X_df, y, columns, resolved, classes_
