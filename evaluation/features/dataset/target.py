"""Target-distribution features: shape, tails, normality, entropy."""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from .. import DatasetFeatureContext, DatasetFeatureGroup
from ._utils import shannon_entropy


class TargetDistribution(DatasetFeatureGroup):
    name = "target_distribution"
    feature_names = (
        "y_skew", "y_kurtosis", "y_cv", "y_bimodality",
        "y_ks_normal", "outlier_frac_2sigma", "outlier_frac_3sigma", "y_entropy",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        y = ctx.y
        y_mean = ctx.y_mean
        y_std  = ctx.y_std
        z = ctx.z

        out: dict[str, float] = {}
        if float(np.std(y)) <= 1e-10:
            return {name: float("nan") for name in self.feature_names}

        out["y_skew"]     = float(sp_stats.skew(y))
        out["y_kurtosis"] = float(sp_stats.kurtosis(y))
        out["y_cv"] = (float(y_std / abs(y_mean))
                       if abs(y_mean) > 1e-10 else float("nan"))
        out["y_bimodality"] = ((out["y_skew"] ** 2 + 1.0)
                               / max(out["y_kurtosis"] + 3.0, 1e-10))
        out["y_ks_normal"] = float(sp_stats.kstest(z, "norm").statistic)
        out["outlier_frac_2sigma"] = float(np.mean(np.abs(z) > 2))
        out["outlier_frac_3sigma"] = float(np.mean(np.abs(z) > 3))

        bin_edges = np.linspace(y.min(), y.max() + 1e-10, 11)
        bin_counts = np.histogram(y, bins=bin_edges)[0]
        bin_probs = bin_counts / max(bin_counts.sum(), 1)
        out["y_entropy"] = shannon_entropy(bin_probs)
        return out
