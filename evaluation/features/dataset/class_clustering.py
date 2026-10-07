"""Cluster-validity features for classification datasets.

pymfe's ``clustering`` group asks: *if we treat the true class labels as a
clustering of the feature space, how well-formed are those clusters?*  A
dataset whose classes sit in compact, well-separated blobs is far easier — and
its predictive uncertainty far more trustworthy — than one whose classes are
smeared together.  No existing group reports this directly: ``class_local``
only looks at the immediate k-NN neighbourhood, not global cluster geometry.

All indices are computed on the standardised numeric matrix ``ctx.X_sc`` with
the integer labels ``ctx.y_int`` used as the cluster assignment.

Features
--------
silhouette
    Mean silhouette coefficient in ``[-1, 1]``.  High → classes are compact and
    well separated; near 0 → overlapping; negative → points are on average
    closer to another class.
davies_bouldin
    Davies-Bouldin index, ``>= 0``.  *Lower* is better-separated.
calinski_harabasz
    Calinski-Harabasz index (between/within dispersion ratio).  *Higher* is
    better-separated.  Note: it grows with the sample size, so compare it only
    within a similarly sized benchmark or after a log transform.
centroid_radius_sep_ratio
    Mean within-class radius divided by mean between-centroid distance.
    Scale-robust separation summary — *lower* means tighter, further-apart
    classes.  Distinct from ``n2_intra_extra_ratio`` (Ho & Basu N2), which uses
    per-point nearest-neighbour distances rather than class centroids.
*_perm_ratio
    Observed clustering score divided by a label-permutation baseline that
    keeps the class sizes fixed.  ``silhouette_perm_ratio`` uses ``score + 1``
    before division because silhouette can be negative.

No regression analogue: these indices require a discrete partition of the
data, which only the class labels provide.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .. import ClassificationDatasetFeatureContext, DatasetFeatureContext, DatasetFeatureGroup

_MIN_SAMPLES: int = 10
_PERMUTATION_REPEATS: int = 5
_SILHOUETTE_PERM_SAMPLE_SIZE: int = 1_000


class ClusteringStructure(DatasetFeatureGroup):
    """Cluster-validity indices with the class labels as the partition."""

    name = "clustering_structure"
    feature_names = (
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "centroid_radius_sep_ratio",
        "silhouette_perm_ratio",
        "davies_bouldin_perm_ratio",
        "calinski_harabasz_perm_ratio",
        "centroid_radius_sep_perm_ratio",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        assert isinstance(ctx, ClassificationDatasetFeatureContext)
        nan_out = {name: float("nan") for name in self.feature_names}

        X = ctx.X_sc
        y = ctx.y_int
        n = X.shape[0]
        # Need >= 2 classes, enough rows, and at least one non-singleton class
        # so the cluster scatter is defined.
        if (
            ctx.d_num == 0
            or n < _MIN_SAMPLES
            or ctx.n_classes < 2
            or ctx.n_classes > n - 1
        ):
            return dict(nan_out)

        out: dict[str, float] = {}
        try:
            out["silhouette"] = float(silhouette_score(X, y))
            out["davies_bouldin"] = float(davies_bouldin_score(X, y))
            ch = float(calinski_harabasz_score(X, y))
            out["calinski_harabasz"] = ch
        except Exception:
            return dict(nan_out)

        out["centroid_radius_sep_ratio"] = _centroid_radius_sep_ratio(X, y, ctx.classes)
        out.update(_permutation_ratios(X, y, ctx.classes, out, seed=ctx.seed))
        return out


def _centroid_radius_sep_ratio(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Mean within-class radius / mean between-centroid distance."""
    centroids = []
    intra_radii = []
    for c in classes:
        pts = X[y == c]
        if pts.shape[0] == 0:
            continue
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        intra_radii.append(float(np.linalg.norm(pts - centroid, axis=1).mean()))

    if len(centroids) < 2:
        return float("nan")

    centroids = np.asarray(centroids)
    pair_dists = [
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    ]
    inter = float(np.mean(pair_dists))
    intra = float(np.mean(intra_radii))
    return intra / max(inter, 1e-10)


def _permutation_ratios(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    observed: dict[str, float],
    *,
    seed: int,
) -> dict[str, float]:
    """Observed clustering scores divided by same-class-size label permutations."""
    rng = np.random.default_rng(seed)
    perm_values: dict[str, list[float]] = {
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
        "centroid_radius_sep_ratio": [],
    }

    for _ in range(_PERMUTATION_REPEATS):
        y_perm = rng.permutation(y)
        try:
            perm_values["silhouette"].append(_silhouette_for_perm(X, y_perm, rng))
            perm_values["davies_bouldin"].append(float(davies_bouldin_score(X, y_perm)))
            perm_values["calinski_harabasz"].append(float(calinski_harabasz_score(X, y_perm)))
            perm_values["centroid_radius_sep_ratio"].append(
                _centroid_radius_sep_ratio(X, y_perm, classes),
            )
        except Exception:
            continue

    baselines = {
        name: _finite_mean(vals)
        for name, vals in perm_values.items()
    }
    return {
        "silhouette_perm_ratio": _shifted_silhouette_ratio(
            observed["silhouette"],
            baselines["silhouette"],
        ),
        "davies_bouldin_perm_ratio": _safe_ratio(
            observed["davies_bouldin"],
            baselines["davies_bouldin"],
        ),
        "calinski_harabasz_perm_ratio": _safe_ratio(
            observed["calinski_harabasz"],
            baselines["calinski_harabasz"],
        ),
        "centroid_radius_sep_perm_ratio": _safe_ratio(
            observed["centroid_radius_sep_ratio"],
            baselines["centroid_radius_sep_ratio"],
        ),
    }


def _silhouette_for_perm(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> float:
    n = X.shape[0]
    if n > _SILHOUETTE_PERM_SAMPLE_SIZE:
        return float(silhouette_score(
            X,
            y,
            sample_size=_SILHOUETTE_PERM_SAMPLE_SIZE,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        ))
    return float(silhouette_score(X, y))


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _safe_ratio(observed: float, baseline: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(baseline) or abs(baseline) <= 1e-10:
        return float("nan")
    return float(observed / baseline)


def _shifted_silhouette_ratio(observed: float, baseline: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(baseline):
        return float("nan")
    denom = baseline + 1.0
    if denom <= 1e-10:
        return float("nan")
    return float((observed + 1.0) / denom)
