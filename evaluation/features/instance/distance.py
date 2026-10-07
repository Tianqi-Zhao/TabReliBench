"""Distance / outlier features (test point x vs context D)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .. import InstanceFeatureContext, InstanceFeatureGroup

_log = logging.getLogger(__name__)


def _median_within_dist(X_ctx_sc: np.ndarray, seed: int = 0) -> float:
    """Median nearest-neighbour distance within a subsample of the context."""
    n_sub = min(len(X_ctx_sc), 500)
    rng = np.random.default_rng(seed)
    sub = X_ctx_sc[rng.choice(len(X_ctx_sc), n_sub, replace=False)]
    nn = NearestNeighbors(n_neighbors=2).fit(sub)
    d, _ = nn.kneighbors(sub)
    return max(float(np.median(d[:, 1])), 1e-10)


class DistanceOutlier(InstanceFeatureGroup):
    name = "distance_outlier"
    feature_names = (
        "knn_dist_k1",  "knn_dist_k1_norm",
        "knn_dist_k5",  "knn_dist_k5_norm",
        "knn_dist_k10", "knn_dist_k10_norm",
        "local_y_std",  "local_y_range",
        "ctx_density",  "mahal_dist",
        "isolation_score", "lof_score",
    )

    def __init__(self, k_list: tuple[int, ...] = (1, 5, 10)) -> None:
        self.k_list = k_list

    def compute(self, ctx: InstanceFeatureContext) -> pd.DataFrame:
        X_ctx_sc = ctx.X_ctx_sc
        X_te_sc  = ctx.X_te_sc
        y_ctx    = ctx.y_ctx
        n_test = len(X_te_sc)
        n_ctx  = len(X_ctx_sc)
        feats: dict[str, np.ndarray] = {}

        if X_ctx_sc.shape[1] == 0:
            return pd.DataFrame(
                {name: np.full(n_test, np.nan) for name in self.feature_names}
            )

        k_max = min(max(self.k_list), n_ctx)
        nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto").fit(X_ctx_sc)
        dists, nn_idx = nn.kneighbors(X_te_sc)

        ref_dist = _median_within_dist(X_ctx_sc)

        for k in self.k_list:
            k_eff = min(k, k_max)
            raw = dists[:, :k_eff].mean(axis=1)
            feats[f"knn_dist_k{k}"]      = raw
            feats[f"knn_dist_k{k}_norm"] = raw / ref_dist

        k_y = min(5, k_max)
        y_nbr = y_ctx[nn_idx[:, :k_y]]
        feats["local_y_std"]   = y_nbr.std(axis=1)
        feats["local_y_range"] = y_nbr.max(axis=1) - y_nbr.min(axis=1)

        counts = (dists[:, 0:1] <= ref_dist).sum(axis=1).astype(float)
        feats["ctx_density"] = counts / max(n_ctx, 1)

        try:
            if X_ctx_sc.shape[1] >= 2 and n_ctx > X_ctx_sc.shape[1] * 2:
                mcd = MinCovDet(support_fraction=0.75, random_state=0).fit(X_ctx_sc)
                feats["mahal_dist"] = np.sqrt(np.maximum(mcd.mahalanobis(X_te_sc), 0))
            else:
                feats["mahal_dist"] = np.full(n_test, np.nan)
        except Exception as e:
            _log.warning(
                "distance_outlier: mahal_dist failed (n_ctx=%d, n_feat=%d), "
                "filling with NaN: %s",
                n_ctx,
                X_ctx_sc.shape[1],
                e,
                exc_info=True,
            )
            feats["mahal_dist"] = np.full(n_test, np.nan)

        try:
            iso = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1)
            iso.fit(X_ctx_sc)
            feats["isolation_score"] = -iso.score_samples(X_te_sc)
        except Exception as e:
            _log.warning(
                "distance_outlier: isolation_score failed (n_ctx=%d, n_feat=%d), "
                "filling with NaN: %s",
                n_ctx,
                X_ctx_sc.shape[1],
                e,
                exc_info=True,
            )
            feats["isolation_score"] = np.full(n_test, np.nan)

        try:
            k_lof = min(20, n_ctx - 1)
            lof = LocalOutlierFactor(n_neighbors=k_lof, novelty=True)
            lof.fit(X_ctx_sc)
            feats["lof_score"] = -lof.score_samples(X_te_sc)
        except Exception as e:
            _log.warning(
                "distance_outlier: lof_score failed (n_ctx=%d, n_feat=%d), "
                "filling with NaN: %s",
                n_ctx,
                X_ctx_sc.shape[1],
                e,
                exc_info=True,
            )
            feats["lof_score"] = np.full(n_test, np.nan)

        return pd.DataFrame(feats)
