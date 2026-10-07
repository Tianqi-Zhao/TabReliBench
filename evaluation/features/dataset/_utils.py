"""Small numeric helpers shared by dataset-level feature groups."""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.neighbors import NearestNeighbors

from .. import (
    ClassificationDatasetFeatureContext,
    DatasetFeatureContext,
)


_CATEGORICAL_DTYPES: tuple[str, ...] = ("object", "string", "category", "bool")


def gini(arr: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (0 = uniform, 1 = concentrated)."""
    arr = np.abs(arr)
    total = arr.sum()
    if total < 1e-10 or len(arr) == 0:
        return 0.0
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr_sorted).sum()) / (n * total) - (n + 1) / n)


def shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


def mean_knn_distance(X: np.ndarray, *, n_neighbors: int = 5) -> float:
    """Mean non-self k-NN distance in the given feature space."""
    n = int(X.shape[0])
    if n < 2:
        return float("nan")
    k_nb = min(n_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k_nb + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    return float(dists[:, 1:].mean())


def knn_distance_null_ratio(
    X: np.ndarray,
    observed: float,
    *,
    seed: int,
    n_neighbors: int = 5,
    n_repeats: int = 5,
) -> float:
    """Observed k-NN distance divided by a column-permutation null baseline.

    The null keeps each standardised column's marginal distribution and the
    dataset's ``n``/``d`` fixed, but breaks multivariate local geometry by
    independently permuting rows within each column.
    """
    if not np.isfinite(observed) or X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    null_vals: list[float] = []
    for _ in range(n_repeats):
        x_null = np.asarray(X, dtype=float).copy()
        for j in range(x_null.shape[1]):
            x_null[:, j] = rng.permutation(x_null[:, j])
        val = mean_knn_distance(x_null, n_neighbors=n_neighbors)
        if np.isfinite(val):
            null_vals.append(val)

    if not null_vals:
        return float("nan")
    baseline = float(np.mean(null_vals))
    return observed / baseline if baseline > 1e-10 else float("nan")


def all_columns_target_mi(ctx: DatasetFeatureContext) -> np.ndarray:
    """Per-column MI with the target, over **numeric + categorical** columns.

    Numeric columns use the row-level complete-case subset (``X_num_clean`` /
    ``y_clean``) with ``discrete_features=False`` — k-NN (KSG / Ross)
    estimator.  Categorical columns use the full ``X_df`` rows with
    ``__NA__`` as a sentinel level and ``discrete_features=True`` —
    plug-in / Ross discrete-side estimator, so the integer codes act as
    **labels only** and the choice of encoding does not affect the result.

    Returns an empty array when neither part can be computed.  Numeric and
    categorical sub-vectors are concatenated; each is internally consistent
    even though they come from different row subsets.
    """
    is_clf = isinstance(ctx, ClassificationDatasetFeatureContext)
    mi_parts: list[np.ndarray] = []

    if ctx.d_num > 0 and ctx.n_clean >= 4:
        if is_clf:
            if ctx.n_classes >= 2:
                mi_num = mutual_info_classif(
                    ctx.X_num_clean.values, ctx.y_int,
                    discrete_features=False, random_state=ctx.seed,
                )
                mi_parts.append(np.asarray(mi_num, dtype=float))
        else:
            if float(np.std(ctx.y_clean)) > 1e-10:
                mi_num = mutual_info_regression(
                    ctx.X_num_clean.values, ctx.y_clean,
                    discrete_features=False, random_state=ctx.seed,
                )
                mi_parts.append(np.asarray(mi_num, dtype=float))

    X_cat = ctx.X_df.select_dtypes(include=list(_CATEGORICAL_DTYPES))
    if X_cat.shape[1] > 0 and ctx.n >= 4:
        codes = np.empty((ctx.n, X_cat.shape[1]), dtype=np.int64)
        for j, col_name in enumerate(X_cat.columns):
            ser = (
                X_cat[col_name]
                .astype("string")
                .fillna("__NA__")
                .astype("category")
            )
            codes[:, j] = ser.cat.codes.to_numpy()
        try:
            if is_clf:
                if len(np.unique(ctx.y)) >= 2:
                    y_full = (
                        ctx.y.astype(int)
                        if not np.issubdtype(ctx.y.dtype, np.integer)
                        else ctx.y
                    )
                    mi_cat = mutual_info_classif(
                        codes, y_full,
                        discrete_features=True, random_state=ctx.seed,
                    )
                    mi_parts.append(np.asarray(mi_cat, dtype=float))
            else:
                if float(np.std(ctx.y)) > 1e-10:
                    mi_cat = mutual_info_regression(
                        codes, ctx.y,
                        discrete_features=True, random_state=ctx.seed,
                    )
                    mi_parts.append(np.asarray(mi_cat, dtype=float))
        except Exception:
            pass

    if not mi_parts:
        return np.empty(0, dtype=float)
    return np.concatenate(mi_parts)
