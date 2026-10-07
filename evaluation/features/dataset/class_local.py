"""Local-structure & class-overlap features for classification datasets.

Two families of features:

* **k-NN neighbourhood** (``knn_disagreement_frac``, ``boundary_frac``,
  ``wg_dist``, ``wg_dist_null_ratio``) — analogous to
  ``HeteroscedasticityLocal`` for regression.
* **Ho & Basu data-complexity measures** (``n2_intra_extra_ratio``,
  ``f2_overlap_mean``, ``f3_max_efficiency``) — class-overlap descriptors that
  characterise how separable the classes are, and therefore how much
  *aleatoric* uncertainty is irreducible.

Features
--------
knn_disagreement_frac
    Mean fraction of k nearest neighbours with a *different* label.  High →
    classes overlap heavily / noisy labels.
boundary_frac
    Fraction of points with *at least one* differently-labelled k-NN.
wg_dist
    Mean k-NN distance in standardised feature space (geometry only).
wg_dist_null_ratio
    ``wg_dist`` divided by a column-permutation null baseline that preserves
    each feature's marginal distribution while breaking multivariate local
    geometry.
n2_intra_extra_ratio (Ho & Basu **N2**)
    ``Σ dist(x, nearest same-class point) / Σ dist(x, nearest other-class
    point)``.  > 1 → points sit closer to other classes than to their own
    (heavy overlap); ≪ 1 → tight, well-separated classes.
f2_overlap_mean (Ho & Basu **F2**)
    Per-feature overlap of class quantile ranges, averaged over features and
    over class pairs.  High → no feature axis cleanly separates the classes.
f3_max_efficiency (derived from Ho & Basu **F3**)
    ``1 - min_feature(fraction of points inside the class-range overlap)``,
    averaged over class pairs.  High → at least one single feature separates
    the classes well; → 0 → every feature is ambiguous on its own.

N2 is computed on the standardised matrix ``X_sc`` (consistent with the k-NN
features); F2 / F3 are scale-free ratios computed on the raw numeric matrix
``X_num_clean`` using 5%/95% quantile ranges instead of min/max ranges so
extreme-sample expansion with ``n`` has less influence.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .. import ClassificationDatasetFeatureContext, DatasetFeatureContext, DatasetFeatureGroup
from ._utils import knn_distance_null_ratio

_MIN_SAMPLES: int = 10
_RANGE_Q_LOW: float = 0.05
_RANGE_Q_HIGH: float = 0.95


class ClassLocalStructure(DatasetFeatureGroup):
    """k-NN label mixing and Ho & Basu class-overlap complexity measures."""

    name = "class_local_structure"
    feature_names = (
        "knn_disagreement_frac",
        "boundary_frac",
        "wg_dist",
        "wg_dist_null_ratio",
        "n2_intra_extra_ratio",
        "f2_overlap_mean",
        "f3_max_efficiency",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        assert isinstance(ctx, ClassificationDatasetFeatureContext)
        if ctx.X_sc.shape[0] < _MIN_SAMPLES:
            return {name: float("nan") for name in self.feature_names}

        X_sc  = ctx.X_sc
        y_int = ctx.y_int

        # ── k-NN neighbourhood ────────────────────────────────────────────
        k_nb = min(5, X_sc.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k_nb + 1).fit(X_sc)
        dists, idx_nn = nn.kneighbors(X_sc)

        neighbour_labels = y_int[idx_nn[:, 1:]]               # (n, k)
        self_labels      = y_int[:, np.newaxis]               # (n, 1)
        different        = neighbour_labels != self_labels    # (n, k) bool

        out: dict[str, float] = {
            "knn_disagreement_frac": float(different.mean()),
            "boundary_frac":         float(different.any(axis=1).mean()),
            "wg_dist":               float(dists[:, 1:].mean()),
        }
        out["wg_dist_null_ratio"] = knn_distance_null_ratio(
            X_sc,
            out["wg_dist"],
            seed=ctx.seed,
            n_neighbors=5,
        )

        # ── Ho & Basu class-overlap measures ──────────────────────────────
        if ctx.n_classes < 2:
            out["n2_intra_extra_ratio"] = float("nan")
            out["f2_overlap_mean"]      = float("nan")
            out["f3_max_efficiency"]    = float("nan")
            return out

        out["n2_intra_extra_ratio"] = _n2_intra_extra(X_sc, y_int, ctx.classes)

        if ctx.d_num > 0:
            X_raw = ctx.X_num_clean.to_numpy(dtype=float)
            out["f2_overlap_mean"]   = _f2_overlap(X_raw, y_int, ctx.classes)
            out["f3_max_efficiency"] = _f3_efficiency(X_raw, y_int, ctx.classes)
        else:
            out["f2_overlap_mean"]   = float("nan")
            out["f3_max_efficiency"] = float("nan")

        return out


def _n2_intra_extra(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Ho & Basu N2: Σ intra-class NN dist / Σ extra-class NN dist."""
    n = X.shape[0]
    intra = np.full(n, np.nan)
    extra = np.full(n, np.nan)

    for c in classes:
        mask = y == c
        idx_c = np.where(mask)[0]
        pts_c = X[mask]
        pts_other = X[~mask]

        if pts_c.shape[0] >= 2:
            nn_c = NearestNeighbors(n_neighbors=2).fit(pts_c)
            d, _ = nn_c.kneighbors(pts_c)
            intra[idx_c] = d[:, 1]                 # nearest excluding self
        if pts_other.shape[0] >= 1 and pts_c.shape[0] >= 1:
            nn_o = NearestNeighbors(n_neighbors=1).fit(pts_other)
            d, _ = nn_o.kneighbors(pts_c)
            extra[idx_c] = d[:, 0]

    paired = np.isfinite(intra) & np.isfinite(extra)
    if not paired.any():
        return float("nan")
    extra_sum = float(extra[paired].sum())
    if extra_sum <= 1e-10:
        return float("nan")
    return float(intra[paired].sum() / extra_sum)


def _f2_overlap(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Ho & Basu F2: per-feature class quantile-range overlap, averaged."""
    bounds: dict = {}
    for c in classes:
        pts = X[y == c]
        if pts.shape[0] > 0:
            bounds[c] = (
                np.quantile(pts, _RANGE_Q_LOW, axis=0),
                np.quantile(pts, _RANGE_Q_HIGH, axis=0),
            )

    present = list(bounds)
    if len(present) < 2:
        return float("nan")

    pair_vals: list[float] = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            min_a, max_a = bounds[present[i]]
            min_b, max_b = bounds[present[j]]
            overlap = np.maximum(0.0, np.minimum(max_a, max_b)
                                      - np.maximum(min_a, min_b))
            span = np.maximum(max_a, max_b) - np.minimum(min_a, min_b)
            pair_vals.append(float(np.mean(overlap / np.maximum(span, 1e-10))))

    return float(np.mean(pair_vals)) if pair_vals else float("nan")


def _f3_efficiency(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> float:
    """1 - Ho & Basu F3: best single-feature quantile-range efficiency."""
    cls_pts = {c: X[y == c] for c in classes}
    present = [c for c in classes if cls_pts[c].shape[0] > 0]
    if len(present) < 2:
        return float("nan")

    pair_min: list[float] = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            Xa, Xb = cls_pts[present[i]], cls_pts[present[j]]
            lo = np.maximum(
                np.quantile(Xa, _RANGE_Q_LOW, axis=0),
                np.quantile(Xb, _RANGE_Q_LOW, axis=0),
            )
            hi = np.minimum(
                np.quantile(Xa, _RANGE_Q_HIGH, axis=0),
                np.quantile(Xb, _RANGE_Q_HIGH, axis=0),
            )
            allv = np.concatenate([Xa, Xb], axis=0)           # (na+nb, d)
            in_overlap = (allv >= lo) & (allv <= hi)          # (na+nb, d)
            frac = in_overlap.mean(axis=0)                    # (d,)
            frac = np.where(hi < lo, 0.0, frac)               # no-overlap → 0
            pair_min.append(float(frac.min()))

    return 1.0 - float(np.mean(pair_min))
