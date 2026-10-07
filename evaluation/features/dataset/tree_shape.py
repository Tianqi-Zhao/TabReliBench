"""Decision-tree *shape* descriptors (regression).

``complexity.py`` already fits CARTs, but only to answer "how deep must a tree
be to reach CV R² >= 0.8?".  The tree it keeps is depth-capped exactly at that
crossing point (or, when the target is never met, the depth-30 fully grown
tree).  Describing *that* tree's shape conflates genuine data complexity with
the arbitrary 0.8 threshold, and the not-met fallback puts it in a completely
different regime — so the descriptors would not be comparable across datasets.

This group instead fits **one** CART with a fixed, dataset-independent growth
rule: ``max_leaf_nodes = max(5, int(sqrt(n)))`` — the same sample-size-scaled
budget that ``hs_tree_complexity`` uses.  Every dataset therefore gets a tree
grown under the same rule, so the shape descriptors mean the same thing
everywhere.

Features (all read off the single fitted tree)
----------------------------------------------
tree_depth
    Depth of the deepest leaf.
tree_imbalance
    ``1 - H(leaf-reach probabilities) / log(n_leaves)``.  0 = every leaf
    receives an equal share of the training mass; → 1 = a few leaves dominate
    (a lopsided partition, i.e. the signal lives in a small region).
tree_leaf_depth_cv
    Coefficient of variation of leaf depths — how uneven the branches are.
tree_var_importance_gini
    Gini concentration of the impurity-based feature importances.  High = the
    tree leans on a few features.
tree_var_importance_top1
    Largest single-feature importance.
tree_feature_used_frac
    Fraction of numeric features that appear in at least one split.

The classification analogue is :class:`ClassTreeShape` in
``class_tree_shape.py`` (it shares the same feature names and the
:func:`tree_shape_features` helper).
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .. import DatasetFeatureContext, DatasetFeatureGroup
from ._utils import gini, shannon_entropy

_MIN_SAMPLES: int = 10

TREE_SHAPE_FEATURE_NAMES: tuple[str, ...] = (
    "tree_depth",
    "tree_imbalance",
    "tree_leaf_depth_cv",
    "tree_var_importance_gini",
    "tree_var_importance_top1",
    "tree_feature_used_frac",
)


def resolve_leaf_budget(n: int) -> int:
    """Sample-size-scaled leaf budget, matching ``hs_tree_complexity``."""
    return max(5, int(np.sqrt(n)))


def tree_shape_features(estimator, d_num: int) -> dict[str, float]:
    """Shape descriptors of a fitted sklearn decision tree.

    Works for both ``DecisionTreeRegressor`` and ``DecisionTreeClassifier``;
    only the tree topology and ``feature_importances_`` are read.
    """
    t = estimator.tree_
    children_left = t.children_left
    n_samples = t.n_node_samples.astype(float)
    is_leaf = children_left == -1

    # Node depths via iterative DFS from the root.
    depth = np.zeros(t.node_count, dtype=int)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        if children_left[node] != -1:
            stack.append((children_left[node], d + 1))
            stack.append((t.children_right[node], d + 1))

    leaf_depths = depth[is_leaf].astype(float)
    n_leaves = int(is_leaf.sum())

    tree_depth = float(leaf_depths.max())

    mean_ld = float(leaf_depths.mean())
    leaf_depth_cv = float(leaf_depths.std() / mean_ld) if mean_ld > 0 else 0.0

    if n_leaves > 1:
        p = n_samples[is_leaf] / max(n_samples[0], 1e-10)
        h = shannon_entropy(p)
        imbalance = max(0.0, float(1.0 - h / np.log(n_leaves)))
    else:
        imbalance = 0.0

    imp = np.asarray(estimator.feature_importances_, dtype=float)
    vi_gini = gini(imp)
    vi_top1 = float(imp.max()) if imp.size > 0 else 0.0

    used = np.unique(t.feature[~is_leaf])
    used = used[used >= 0]
    feature_used_frac = (
        float(used.size / d_num) if d_num > 0 else float("nan")
    )

    return {
        "tree_depth":               tree_depth,
        "tree_imbalance":           imbalance,
        "tree_leaf_depth_cv":       leaf_depth_cv,
        "tree_var_importance_gini": vi_gini,
        "tree_var_importance_top1": vi_top1,
        "tree_feature_used_frac":   feature_used_frac,
    }


class TreeShape(DatasetFeatureGroup):
    """Shape of a single sample-size-budgeted CART fitted on the dataset."""

    name = "tree_shape"
    feature_names = TREE_SHAPE_FEATURE_NAMES

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        nan_out = {name: float("nan") for name in self.feature_names}
        if ctx.d_num == 0 or ctx.X_sc.shape[0] < _MIN_SAMPLES:
            return dict(nan_out)

        try:
            tree = DecisionTreeRegressor(
                max_leaf_nodes=resolve_leaf_budget(ctx.X_sc.shape[0]),
                random_state=ctx.seed,
            ).fit(ctx.X_sc, ctx.y_clean)
        except Exception:
            return dict(nan_out)

        return tree_shape_features(tree, ctx.d_num)
