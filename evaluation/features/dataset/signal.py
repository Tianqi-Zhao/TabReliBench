"""Signal quality: correlations, mutual information, near-constant features."""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from .. import DatasetFeatureContext, DatasetFeatureGroup
from ._utils import all_columns_target_mi, gini, shannon_entropy


class SignalQuality(DatasetFeatureGroup):
    name = "signal_quality"
    feature_names = (
        "max_abs_corr", "mean_abs_corr", "nonlinearity_gap",
        "mi_mean", "mi_max", "mi_entropy", "mi_entropy_norm",
        "gini_mi", "ns_ratio", "near_constant_frac",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        out: dict[str, float] = {}

        abs_pearson:  list[float] = []
        abs_spearman: list[float] = []
        y_varies = ctx.n_clean >= 2 and float(np.std(ctx.y_clean)) > 1e-10

        if ctx.d_num > 0 and ctx.n_clean >= 4 and y_varies:
            for col in ctx.X_num_clean.columns:
                xj = ctx.X_num_clean[col].values
                if xj.std() > 1e-10:
                    abs_pearson.append(abs(sp_stats.pearsonr(xj, ctx.y_clean)[0]))
                    abs_spearman.append(abs(sp_stats.spearmanr(xj, ctx.y_clean)[0]))

        if abs_pearson:
            out["max_abs_corr"]  = float(max(abs_pearson))
            out["mean_abs_corr"] = float(np.mean(abs_pearson))
            out["nonlinearity_gap"] = float(np.mean(abs_spearman) - np.mean(abs_pearson))
        else:
            out["max_abs_corr"]     = float("nan")
            out["mean_abs_corr"]    = float("nan")
            out["nonlinearity_gap"] = float("nan")

        mi_vals_arr = all_columns_target_mi(ctx)
        if mi_vals_arr.size > 0:
            mi_p = mi_vals_arr / (mi_vals_arr.sum() + 1e-10)
            mi_ent_raw = shannon_entropy(mi_p[mi_p > 0])
            out["mi_mean"]    = float(mi_vals_arr.mean())
            out["mi_max"]     = float(mi_vals_arr.max())
            out["mi_entropy"] = mi_ent_raw
            out["gini_mi"]    = gini(mi_vals_arr)
            out["ns_ratio"]   = float(
                np.mean(mi_vals_arr < 0.05 * max(mi_vals_arr.max(), 1e-10))
            )
            d_total = mi_vals_arr.size
            log_d = float(np.log(d_total)) if d_total > 1 else 0.0
            out["mi_entropy_norm"] = float(mi_ent_raw / log_d) if log_d > 1e-10 else float("nan")
        else:
            for k in ("mi_mean", "mi_max", "mi_entropy", "mi_entropy_norm",
                       "gini_mi", "ns_ratio"):
                out[k] = float("nan")

        if ctx.d_num > 0:
            nc = []
            for col in ctx.X_num.columns:
                xj = ctx.X_num[col].dropna().values
                rng_j = xj.max() - xj.min() if len(xj) > 1 else 0.0
                nc.append(
                    float(xj.std() / max(rng_j, 1e-10) < 0.01)
                    if len(xj) > 1 else 1.0
                )
            out["near_constant_frac"] = float(np.mean(nc))
        else:
            out["near_constant_frac"] = float("nan")
        return out
