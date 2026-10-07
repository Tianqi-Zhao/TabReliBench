"""Context (D) statistics: size, label distribution, sharpness."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .. import InstanceFeatureContext, InstanceFeatureGroup
from ...ppd import quantile_grid_integral


class ContextStats(InstanceFeatureGroup):
    name = "context_stats"
    feature_names = (
        "n_context", "log_n_context",
        "y_ctx_mean", "y_ctx_std", "y_ctx_skew", "y_ctx_kurtosis",
        "pred_mean_standardized", "abs_pred_mean_std",
        "pi_width_norm",
    )

    def compute(self, ctx: InstanceFeatureContext) -> pd.DataFrame:
        ppd_grid = ctx.ppd
        ppd = np.asarray(ppd_grid.ppd, dtype=np.float64)
        q_grid = np.asarray(ppd_grid.levels, dtype=np.float64)
        ppd_mean = quantile_grid_integral(ppd, q_grid)
        pi_width = np.asarray(ctx.intervals["width"], dtype=np.float64)

        n_test = len(ppd_mean)
        y_ctx = ctx.y_ctx
        y_mean = float(np.mean(y_ctx))
        y_std  = max(float(np.std(y_ctx)), 1e-10)

        return pd.DataFrame({
            "n_context":              np.full(n_test, float(ctx.n_context)),
            "log_n_context":          np.full(n_test, float(np.log1p(ctx.n_context))),
            "y_ctx_mean":             np.full(n_test, y_mean),
            "y_ctx_std":              np.full(n_test, y_std),
            "y_ctx_skew":             np.full(n_test, float(sp_stats.skew(y_ctx))),
            "y_ctx_kurtosis":         np.full(n_test, float(sp_stats.kurtosis(y_ctx))),
            "pred_mean_standardized": (ppd_mean - y_mean) / y_std,
            "abs_pred_mean_std":      np.abs((ppd_mean - y_mean) / y_std),
            "pi_width_norm":          pi_width / y_std,
        })
