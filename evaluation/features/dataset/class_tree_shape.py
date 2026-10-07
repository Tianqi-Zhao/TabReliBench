"""Decision-tree *shape* descriptors (classification).

Classification analogue of :class:`~evaluation.features.dataset.tree_shape.TreeShape`.
Fits one ``DecisionTreeClassifier`` with the same sample-size-scaled leaf
budget (``max_leaf_nodes = max(5, int(sqrt(n)))``) and reports the identical
set of shape descriptors via the shared :func:`tree_shape_features` helper.
See ``tree_shape.py`` for the rationale and the per-feature meaning.
"""
from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from .. import ClassificationDatasetFeatureContext, DatasetFeatureContext, DatasetFeatureGroup
from .tree_shape import (
    TREE_SHAPE_FEATURE_NAMES,
    resolve_leaf_budget,
    tree_shape_features,
)

_MIN_SAMPLES: int = 10


class ClassTreeShape(DatasetFeatureGroup):
    """Shape of a single sample-size-budgeted classification CART."""

    name = "class_tree_shape"
    feature_names = TREE_SHAPE_FEATURE_NAMES

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        assert isinstance(ctx, ClassificationDatasetFeatureContext)
        nan_out = {name: float("nan") for name in self.feature_names}
        if (
            ctx.d_num == 0
            or ctx.X_sc.shape[0] < _MIN_SAMPLES
            or ctx.n_classes < 2
        ):
            return dict(nan_out)

        try:
            tree = DecisionTreeClassifier(
                max_leaf_nodes=resolve_leaf_budget(ctx.X_sc.shape[0]),
                random_state=ctx.seed,
            ).fit(ctx.X_sc, ctx.y_int)
        except Exception:
            return dict(nan_out)

        return tree_shape_features(tree, ctx.d_num)
