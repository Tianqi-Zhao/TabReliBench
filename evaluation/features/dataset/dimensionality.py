"""Dimensionality & capacity: sizes, PCA effective dim, NaN coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .. import DatasetFeatureContext, DatasetFeatureGroup


class DimensionalityCapacity(DatasetFeatureGroup):
    name = "dimensionality_capacity"
    feature_names = (
        "n_train", "n_features", "dim_ratio", "effective_dim_ratio",
        "condition_number", "discrete_frac", "binary_frac",
        "nan_frac_overall", "nan_col_frac",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        out: dict[str, float] = {}
        out["n_train"]    = ctx.n
        out["n_features"] = ctx.d
        out["dim_ratio"]  = ctx.d / ctx.n

        if ctx.d_num > 0 and ctx.n_clean >= 4:
            pca = PCA().fit(ctx.X_sc)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_pcs_95 = int(np.searchsorted(cumvar, 0.95)) + 1
            out["effective_dim_ratio"] = n_pcs_95 / ctx.d_num
            sv = pca.singular_values_
            out["condition_number"] = float(sv[0] / max(sv[-1], 1e-10))
        else:
            out["effective_dim_ratio"] = float("nan")
            out["condition_number"]    = float("nan")

        out["discrete_frac"] = float(np.mean([
            0.0 if pd.api.types.is_numeric_dtype(ctx.X_df[c]) else 1.0
            for c in ctx.X_df.columns
        ]))
        out["binary_frac"] = float(np.mean([
            1.0 if ctx.X_df[c].dropna().nunique() == 2 else 0.0
            for c in ctx.X_df.columns
        ]))
        out["nan_frac_overall"] = float(ctx.X_df.isna().mean().mean())
        out["nan_col_frac"]     = float(ctx.X_df.isna().any().mean())
        return out
