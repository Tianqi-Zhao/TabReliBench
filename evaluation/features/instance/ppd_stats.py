"""PPD summary statistics + alpha-specific interval features.

The interval-related arrays (``pi_lower / pi_upper / pi_width``) are NOT
re-derived from the PPD here - they come directly from the metrics PKL via
``ctx.intervals``. Only the alpha-independent shape statistics (mean,
median, IQR, skewness, kurtosis, entropy, ...) are computed from the PPD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import InstanceFeatureContext, InstanceFeatureGroup
from ...ppd import quantile_grid_integral


class PPDStats(InstanceFeatureGroup):
    name = "ppd_stats"
    feature_names = (
        # alpha-independent
        "ppd_mean", "ppd_std", "ppd_median", "ppd_iqr",
        "ppd_skew_bowley", "ppd_skew_pearson",
        "ppd_skew_moment", "ppd_kurtosis_excess",
        "ppd_kurtosis_moors", "ppd_bimodality",
        "ppd_entropy", "ppd_tail_ratio", "ppd_cv",
        # alpha-specific (passthrough from intervals + asymmetry derived here)
        "pi_lower", "pi_upper", "pi_width", "pi_asymmetry",
    )

    def compute(self, ctx: InstanceFeatureContext) -> pd.DataFrame:
        ppd_grid = ctx.ppd
        ppd = np.asarray(ppd_grid.ppd, dtype=np.float64)
        q_grid = np.asarray(ppd_grid.levels, dtype=np.float64)
        feats: dict[str, np.ndarray] = {}

        feats["ppd_mean"]    = quantile_grid_integral(ppd, q_grid)
        ppd_mean_sq          = quantile_grid_integral(ppd ** 2, q_grid)
        feats["ppd_std"]     = np.sqrt(np.clip(ppd_mean_sq - feats["ppd_mean"] ** 2, 0, None))
        feats["ppd_median"]  = ppd_grid.quantile_at(0.50)

        q25 = ppd_grid.quantile_at(0.25)
        q75 = ppd_grid.quantile_at(0.75)
        feats["ppd_iqr"] = q75 - q25

        iqr_safe = np.where(feats["ppd_iqr"] > 1e-10, feats["ppd_iqr"], np.nan)
        feats["ppd_skew_bowley"] = (q75 + q25 - 2 * feats["ppd_median"]) / iqr_safe

        std_safe = np.where(feats["ppd_std"] > 1e-10, feats["ppd_std"], np.nan)
        feats["ppd_skew_pearson"] = (feats["ppd_mean"] - feats["ppd_median"]) / std_safe

        z = (ppd - feats["ppd_mean"][:, None]) / np.where(
            feats["ppd_std"][:, None] > 1e-10, feats["ppd_std"][:, None], 1.0
        )
        feats["ppd_skew_moment"]     = quantile_grid_integral(z ** 3, q_grid)
        feats["ppd_kurtosis_excess"] = quantile_grid_integral(z ** 4, q_grid) - 3.0

        q12 = ppd_grid.quantile_at(0.125)
        q37 = ppd_grid.quantile_at(0.375)
        q62 = ppd_grid.quantile_at(0.625)
        q87 = ppd_grid.quantile_at(0.875)
        feats["ppd_kurtosis_moors"] = (q87 - q62 + q37 - q12) / iqr_safe

        sk = feats["ppd_skew_moment"]
        ku = feats["ppd_kurtosis_excess"]
        feats["ppd_bimodality"] = (sk ** 2 + 1) / np.maximum(ku + 3, 1e-10)

        # Differential entropy from quantile spacings.
        dp = np.diff(q_grid)
        dq = np.diff(ppd, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dp > 0, dq / dp[None, :], 1.0)
            ratio = np.maximum(ratio, 1e-30)
            feats["ppd_entropy"] = np.sum(dp[None, :] * np.log(ratio), axis=1)

        q05 = ppd_grid.quantile_at(0.05)
        q95 = ppd_grid.quantile_at(0.95)
        feats["ppd_tail_ratio"] = (q95 - q05) / iqr_safe

        feats["ppd_cv"] = feats["ppd_std"] / np.where(
            np.abs(feats["ppd_mean"]) > 1e-10, np.abs(feats["ppd_mean"]), np.nan
        )

        # Alpha-specific: passthrough from metrics PKL.
        iv = ctx.intervals
        pi_lower = np.asarray(iv["lower"], dtype=np.float64)
        pi_upper = np.asarray(iv["upper"], dtype=np.float64)
        pi_width = np.asarray(iv["width"], dtype=np.float64)
        feats["pi_lower"] = pi_lower
        feats["pi_upper"] = pi_upper
        feats["pi_width"] = pi_width
        midpoint = (pi_lower + pi_upper) / 2.0
        feats["pi_asymmetry"] = np.where(
            pi_width > 1e-10,
            (feats["ppd_median"] - midpoint) / pi_width,
            0.0,
        )
        return pd.DataFrame(feats)
