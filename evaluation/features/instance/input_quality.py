"""Input quality: NaN counts on the test point itself."""
from __future__ import annotations

import pandas as pd

from .. import InstanceFeatureContext, InstanceFeatureGroup


class InputQuality(InstanceFeatureGroup):
    name = "input_quality"
    feature_names = ("nan_count_x", "nan_frac_x")

    def compute(self, ctx: InstanceFeatureContext) -> pd.DataFrame:
        nan_count = ctx.X_test.isna().sum(axis=1).values.astype(float)
        return pd.DataFrame({
            "nan_count_x": nan_count,
            "nan_frac_x":  nan_count / max(ctx.X_test.shape[1], 1),
        })
