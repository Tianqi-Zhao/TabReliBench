"""Inter-feature correlation structure."""
from __future__ import annotations

import numpy as np

from .. import DatasetFeatureContext, DatasetFeatureGroup


class FeatureImportanceStructure(DatasetFeatureGroup):
    name = "feature_importance_structure"
    feature_names = ("mean_inter_feature_corr", "max_inter_feature_corr")

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        if ctx.d_num > 1 and ctx.n_clean >= 4:
            x = ctx.X_num_clean.values
            keep = np.std(x, axis=0) > 1e-10
            if int(keep.sum()) > 1:
                cc = np.abs(np.corrcoef(x[:, keep].T))
                triu = cc[np.triu_indices(int(keep.sum()), k=1)]
                triu = triu[np.isfinite(triu)]
                if len(triu) > 0:
                    return {
                        "mean_inter_feature_corr": float(triu.mean()),
                        "max_inter_feature_corr":  float(triu.max()),
                    }
        return {
            "mean_inter_feature_corr": float("nan"),
            "max_inter_feature_corr":  float("nan"),
        }
