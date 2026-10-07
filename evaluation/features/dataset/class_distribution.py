"""Class-distribution features for classification datasets."""
from __future__ import annotations

import numpy as np

from .. import ClassificationDatasetFeatureContext, DatasetFeatureContext, DatasetFeatureGroup
from ._utils import shannon_entropy


class ClassDistribution(DatasetFeatureGroup):
    """Target-class statistics.

    Analogous to ``TargetDistribution`` for regression.  All features are
    derived from the class frequencies of ``ctx.y`` (the full training sample,
    not the NaN-dropped subset) so we get the true class distribution.
    """

    name = "class_distribution"
    feature_names = (
        "n_classes",
        "class_entropy",
        "class_entropy_norm",
        "majority_class_frac",
        "minority_class_frac",
        "imbalance_ratio",
        "class_gini",
        "class_gini_norm",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        # Use full y (not y_clean) for class distribution — consistent with the
        # ctx.y-based design; ctx.y_int in ClassificationDatasetFeatureContext
        # is derived from y_clean for numeric-feature-aligned groups.
        y = ctx.y.astype(int) if not np.issubdtype(ctx.y.dtype, np.integer) else ctx.y
        classes, counts = np.unique(y, return_counts=True)
        k = int(len(classes))
        probs = counts / counts.sum()

        p_max = float(probs.max())
        p_min = float(probs.min())

        ent = float(shannon_entropy(probs))
        log_k = float(np.log(k)) if k > 1 else 0.0
        gini_raw = float(1.0 - np.sum(probs ** 2))
        gini_max = 1.0 - 1.0 / k if k > 1 else 0.0
        return {
            "n_classes":           float(k),
            "class_entropy":       ent,
            "class_entropy_norm":  ent / log_k if log_k > 1e-10 else float("nan"),
            "majority_class_frac": p_max,
            "minority_class_frac": p_min,
            "imbalance_ratio":     p_max / max(p_min, 1e-10),
            "class_gini":          gini_raw,
            "class_gini_norm":     gini_raw / gini_max if gini_max > 1e-10 else float("nan"),
        }
