"""Instance-level meta-feature extraction.

Four groups, each a small module behind one ``InstanceFeatureExtractor``.

Group → module:

  distance_outlier   distance.py
  ppd_stats          ppd_stats.py
  context_stats      context.py
  input_quality      input_quality.py

The extractor consumes the per-instance ``IntervalArrays`` written by
:class:`evaluation.metrics.MetricsCalculator` (read from the metrics PKL by
:class:`evaluation.pipelines.InstanceFeaturePipeline`). It does NOT re-derive
``lower / upper / covered / width / winkler`` from the PPD - those live in
exactly one place.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import InstanceFeatureContext, InstanceFeatureGroup
from ...ppd import PPDQuantileGrid
from .context import ContextStats
from .distance import DistanceOutlier
from .input_quality import InputQuality
from .ppd_stats import PPDStats

warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_INSTANCE_GROUPS: tuple[InstanceFeatureGroup, ...] = (
    DistanceOutlier(),
    PPDStats(),
    ContextStats(),
    InputQuality(),
)


class InstanceFeatureExtractor:
    """Run every configured group and concat their DataFrames column-wise."""

    def __init__(
        self,
        groups: Sequence[InstanceFeatureGroup] = DEFAULT_INSTANCE_GROUPS,
    ) -> None:
        self.groups = list(groups)

    @property
    def feature_groups(self) -> dict[str, list[str]]:
        return {g.name: list(g.feature_names) for g in self.groups}

    def compute(
        self,
        *,
        X_ctx: pd.DataFrame,
        y_ctx: np.ndarray,
        X_test: pd.DataFrame,
        ppd: PPDQuantileGrid,
        n_context: int,
        intervals: dict,
    ) -> pd.DataFrame:
        """Return a (n_test, n_features) DataFrame.

        Parameters
        ----------
        intervals : dict
            ``IntervalArrays``-shaped dict pulled from a metrics PKL:
            keys ``lower``, ``upper``, ``width``, ``covered``, ``winkler``.
        """
        X_ctx_sc, X_te_sc = _to_numeric(X_ctx, X_test)
        ctx = InstanceFeatureContext(
            X_ctx=X_ctx, y_ctx=y_ctx, X_test=X_test,
            n_context=n_context,
            X_ctx_sc=X_ctx_sc, X_te_sc=X_te_sc,
            ppd=ppd, intervals=intervals,
        )

        frames: list[pd.DataFrame] = []
        for g in self.groups:
            frames.append(g.compute(ctx))
        return pd.concat(frames, axis=1)


def _to_numeric(X_ctx: pd.DataFrame, X_test: pd.DataFrame):
    """Return standardised numeric-only arrays (NaN filled by ctx mean).

    Numeric columns that are entirely NaN in the context split are dropped:
    their column mean is NaN, so ``fillna`` cannot fix them and scaling would
    propagate NaNs.

    When no numeric columns exist (or none remain after dropping), returns
    zero-column arrays so that downstream distance features gracefully fall
    back to NaN.
    """
    X_ctx_num = X_ctx.select_dtypes(include="number").dropna(axis=1, how="all")
    if X_ctx_num.shape[1] == 0:
        return (
            np.empty((len(X_ctx), 0), dtype=np.float64),
            np.empty((len(X_test), 0), dtype=np.float64),
        )
    col_means = X_ctx_num.mean()
    X_ctx_num_filled = X_ctx_num.fillna(col_means).values.astype(np.float64)
    X_te_num = (
        X_test.select_dtypes(include="number")
        .reindex(columns=X_ctx_num.columns)
        .fillna(col_means)
        .values.astype(np.float64)
    )
    scaler = StandardScaler().fit(X_ctx_num_filled)
    return scaler.transform(X_ctx_num_filled), scaler.transform(X_te_num)
