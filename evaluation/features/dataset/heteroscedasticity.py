"""Heteroscedasticity & local-structure features."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .. import DatasetFeatureContext, DatasetFeatureGroup
from ._utils import knn_distance_null_ratio


class HeteroscedasticityLocal(DatasetFeatureGroup):
    name = "heteroscedasticity_local"
    feature_names = (
        "heteroscedasticity_ratio",
        "conceptvar",
        "wg_dist",
        "wg_dist_null_ratio",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        out: dict[str, float] = {}

        edges = np.percentile(ctx.y, np.linspace(0, 100, 6))[1:-1]
        bin_labels = np.digitize(ctx.y, edges)
        bin_stds = [
            ctx.y[bin_labels == b].std()
            for b in range(5) if (bin_labels == b).sum() > 5
        ]
        out["heteroscedasticity_ratio"] = (
            max(bin_stds) / max(min(bin_stds), 1e-10)
            if len(bin_stds) >= 2 else float("nan")
        )

        if ctx.X_sc.shape[0] >= 10:
            k_nb = min(5, ctx.X_sc.shape[0] - 1)
            nn = NearestNeighbors(n_neighbors=k_nb + 1).fit(ctx.X_sc)
            dists, idx_nn = nn.kneighbors(ctx.X_sc)
            local_vars = np.array(
                [ctx.y_clean[idx_nn[i, 1:]].var() for i in range(len(ctx.y_clean))]
            )
            out["conceptvar"] = float(
                local_vars.mean() / max(float(ctx.y_clean.var()), 1e-10)
            )
            out["wg_dist"] = float(dists[:, 1:].mean())
            out["wg_dist_null_ratio"] = knn_distance_null_ratio(
                ctx.X_sc,
                out["wg_dist"],
                seed=ctx.seed,
                n_neighbors=5,
            )
        else:
            out["conceptvar"] = float("nan")
            out["wg_dist"]    = float("nan")
            out["wg_dist_null_ratio"] = float("nan")
        return out
