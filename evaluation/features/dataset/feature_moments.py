"""Per-feature distributional moments of the numeric design matrix X.

The other shared groups describe ``X`` only through linear correlation
(``correlation.py``) or PCA structure (``dimensionality.py``).  None of them
looks at the *shape* of each individual numeric feature's marginal
distribution.  This group fills that gap: it computes per-column statistics on
the numeric columns and aggregates them across columns.

Why this matters here
---------------------
TabPFN-style models apply distribution-sensitive preprocessing (power
transforms, outlier clipping, per-feature scaling).  Heavy tails, strong skew,
spike-at-a-value sparsity, and wildly heterogeneous feature scales all change
how well that preprocessing behaves — and therefore how reliable the model's
uncertainty estimates are.  Capturing the marginal shape of ``X`` makes those
dataset properties visible to the meta-analysis.

``feat_bimodality_mean`` additionally gives a (univariate) hint of multimodal /
cluster structure in the features that is *task-agnostic* — unlike the
classification-only ``clustering_structure`` group, it is also defined for
regression datasets.

Conventions
-----------
* Operates on ``ctx.X_num`` — numeric columns, already filtered for
  high-missing columns by the extractor.
* Each column is reduced with ``dropna()`` *independently*, so a NaN in one
  column never discards rows from another.
* A column contributes to the aggregates only if it has >= 4 non-null values
  and non-zero standard deviation; constant / tiny columns are skipped.
* If no numeric column qualifies, every feature is ``nan``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from .. import DatasetFeatureContext, DatasetFeatureGroup

_MIN_COL_OBS: int = 4
_OUTLIER_SIGMA: float = 3.0


class FeatureMoments(DatasetFeatureGroup):
    """Marginal-distribution shape of the numeric feature columns."""

    name = "feature_moments"
    feature_names = (
        "feat_skew_mean",
        "feat_skew_max",
        "feat_kurtosis_mean",
        "feat_kurtosis_max",
        "feat_outlier_frac_mean",
        "feat_bimodality_mean",
        "feat_sparsity_mean",
        "feat_mad_std_ratio_mean",
        "feat_scale_dispersion",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        nan_out = {name: float("nan") for name in self.feature_names}
        if ctx.d_num == 0:
            return dict(nan_out)

        abs_skew:    list[float] = []
        kurtosis:    list[float] = []
        outlier_fr:  list[float] = []
        bimodality:  list[float] = []
        sparsity:    list[float] = []
        mad_std:     list[float] = []
        col_stds:    list[float] = []

        for col in ctx.X_num.columns:
            v = ctx.X_num[col].dropna().to_numpy(dtype=float)
            if v.size < _MIN_COL_OBS:
                continue
            std = float(v.std())
            if std <= 1e-10:
                continue

            mean = float(v.mean())
            sk = float(sp_stats.skew(v))
            ku = float(sp_stats.kurtosis(v))            # excess kurtosis
            abs_skew.append(abs(sk))
            kurtosis.append(ku)

            z = (v - mean) / std
            outlier_fr.append(float(np.mean(np.abs(z) > _OUTLIER_SIGMA)))

            # Sarle's bimodality coefficient on excess kurtosis; > 5/9 hints
            # at a bimodal / multimodal (cluster-like) marginal.
            bimodality.append((sk ** 2 + 1.0) / max(ku + 3.0, 1e-10))

            # Sparsity proxy: mass concentrated on the single most frequent
            # value (captures spike-at-zero and heavily discretised columns).
            _, counts = np.unique(v, return_counts=True)
            sparsity.append(float(counts.max()) / v.size)

            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            mad_std.append(mad / std)                   # ~0.674 for a Gaussian

            col_stds.append(std)

        if not abs_skew:
            return dict(nan_out)

        out: dict[str, float] = {
            "feat_skew_mean":          float(np.mean(abs_skew)),
            "feat_skew_max":           float(np.max(abs_skew)),
            "feat_kurtosis_mean":      float(np.mean(kurtosis)),
            "feat_kurtosis_max":       float(np.max(kurtosis)),
            "feat_outlier_frac_mean":  float(np.mean(outlier_fr)),
            "feat_bimodality_mean":    float(np.mean(bimodality)),
            "feat_sparsity_mean":      float(np.mean(sparsity)),
            "feat_mad_std_ratio_mean": float(np.mean(mad_std)),
        }

        # Heterogeneity of raw feature scales: CV of the per-column std devs.
        # Needs >= 2 qualifying columns to be meaningful.
        if len(col_stds) >= 2:
            stds = np.asarray(col_stds, dtype=float)
            out["feat_scale_dispersion"] = float(stds.std() / max(stds.mean(), 1e-10))
        else:
            out["feat_scale_dispersion"] = float("nan")

        return out
