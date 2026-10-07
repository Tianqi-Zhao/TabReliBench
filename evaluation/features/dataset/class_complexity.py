"""Complexity / non-linearity features for classification datasets."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from .. import ClassificationDatasetFeatureContext, DatasetFeatureContext, DatasetFeatureGroup


class ClassificationComplexity(DatasetFeatureGroup):
    """Linear vs non-linear predictability; CART depth to reach target accuracy.

    Analogous to ``ComplexityNonlinearity`` for regression.  Scoring is
    ``balanced_accuracy`` so class imbalance does not distort the estimates.
    """

    name = "classification_complexity"
    feature_names = (
        "linear_acc",
        "nonlinear_acc",
        "complexity_ratio",
        "linear_acc_norm",
        "nonlinear_acc_norm",
        "complexity_ratio_norm",
        "tree_target_met",
        "tree_min_depth",
        "tree_terminal_nodes",
        "tree_achieved_acc",
        "tree_achieved_acc_norm",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        assert isinstance(ctx, ClassificationDatasetFeatureContext)
        out: dict[str, float] = {}

        y_int  = ctx.y_int
        n_cls  = ctx.n_classes

        lin_acc = gbt_acc = float("nan")
        if ctx.X_sc.shape[0] >= 20 and n_cls >= 2:
            cv = min(5, len(y_int) // 10) if len(y_int) >= 50 else 3
            try:
                lin_acc = float(np.mean(cross_val_score(
                    LogisticRegression(
                        max_iter=500, random_state=ctx.seed, solver="saga",
                    ),
                    ctx.X_sc, y_int, cv=cv, scoring="balanced_accuracy",
                )))
                gbt_acc = float(np.mean(cross_val_score(
                    GradientBoostingClassifier(
                        n_estimators=50, max_depth=3, random_state=ctx.seed,
                    ),
                    ctx.X_sc, y_int, cv=cv, scoring="balanced_accuracy",
                )))
            except Exception:
                pass

        out["linear_acc"]    = lin_acc
        out["nonlinear_acc"] = gbt_acc
        out["complexity_ratio"] = (
            gbt_acc / max(lin_acc, 0.01)
            if np.isfinite(lin_acc) and np.isfinite(gbt_acc)
            else float("nan")
        )
        # Chance-adjusted: (acc - 1/K) / (1 - 1/K), comparable across K.
        chance = 1.0 / max(n_cls, 2)
        denom = 1.0 - chance
        if np.isfinite(lin_acc):
            lin_norm = (lin_acc - chance) / denom
            out["linear_acc_norm"] = float(lin_norm)
        else:
            lin_norm = float("nan")
            out["linear_acc_norm"] = float("nan")
        if np.isfinite(gbt_acc):
            gbt_norm = (gbt_acc - chance) / denom
            out["nonlinear_acc_norm"] = float(gbt_norm)
        else:
            gbt_norm = float("nan")
            out["nonlinear_acc_norm"] = float("nan")
        out["complexity_ratio_norm"] = (
            gbt_norm / max(lin_norm, 0.01)
            if np.isfinite(lin_norm) and np.isfinite(gbt_norm)
            else float("nan")
        )

        if ctx.X_sc.shape[0] >= 10 and n_cls >= 2:
            out.update(_tree_depth_for_target_acc(ctx.X_sc, y_int, seed=ctx.seed))
        else:
            out.update({
                "tree_target_met":     float("nan"),
                "tree_min_depth":      float("nan"),
                "tree_terminal_nodes": float("nan"),
                "tree_achieved_acc":   float("nan"),
            })

        # Chance-adjusted version of tree_achieved_acc (balanced accuracy with
        # 1/K chance baseline), comparable across K.
        tree_acc = out["tree_achieved_acc"]
        if np.isfinite(tree_acc) and n_cls >= 2:
            out["tree_achieved_acc_norm"] = float((tree_acc - chance) / denom)
        else:
            out["tree_achieved_acc_norm"] = float("nan")

        return out


def _tree_depth_for_target_acc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    target_acc: float = 0.8,
    max_allowed_depth: int = 30,
    cv_folds: int = 5,
) -> dict[str, float]:
    """Minimum CART depth to reach ``target_acc`` balanced accuracy OOB."""
    n_splits = min(cv_folds, len(y) // 2)
    if n_splits < 2:
        return {
            "tree_target_met":     float("nan"),
            "tree_min_depth":      float("nan"),
            "tree_terminal_nodes": float("nan"),
            "tree_achieved_acc":   float("nan"),
        }

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    achieved_acc = float("nan")

    for depth in range(1, max_allowed_depth + 1):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        try:
            scores = cross_val_score(tree, X, y, cv=cv, scoring="balanced_accuracy")
        except Exception:
            break
        achieved_acc = float(np.mean(scores))
        if achieved_acc >= target_acc:
            tree.fit(X, y)
            return {
                "tree_target_met":     1.0,
                "tree_min_depth":      float(depth),
                "tree_terminal_nodes": float(tree.get_n_leaves()),
                "tree_achieved_acc":   achieved_acc,
            }

    tree = DecisionTreeClassifier(max_depth=max_allowed_depth, random_state=seed)
    try:
        tree.fit(X, y)
        n_leaves = float(tree.get_n_leaves())
    except Exception:
        n_leaves = float("nan")
    return {
        "tree_target_met":     0.0,
        "tree_min_depth":      float(max_allowed_depth),
        "tree_terminal_nodes": n_leaves,
        "tree_achieved_acc":   achieved_acc,
    }
