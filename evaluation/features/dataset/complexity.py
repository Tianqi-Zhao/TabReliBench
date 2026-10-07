"""Complexity & non-linearity: linear vs GBT R², bounded tree depth."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from .. import DatasetFeatureContext, DatasetFeatureGroup


class ComplexityNonlinearity(DatasetFeatureGroup):
    name = "complexity_nonlinearity"
    feature_names = (
        "linear_r2", "nonlinear_r2", "complexity_ratio", "snr",
        "tree_target_met", "tree_min_depth",
        "tree_terminal_nodes", "tree_achieved_r2",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        out: dict[str, float] = {}

        # Constant y makes R² undefined (SS_tot = 0).  sklearn will silently
        # return NaN or 0 from r2_score in that case — short-circuit so the
        # caller sees an honest NaN instead of garbage downstream values.
        y_varies = (
            ctx.X_sc.shape[0] >= 2 and float(np.std(ctx.y_clean)) > 1e-10
        )

        lin_r2 = gbt_r2 = float("nan")
        if ctx.X_sc.shape[0] >= 20 and y_varies:
            y_c = ctx.y_clean - ctx.y_clean.mean()
            cv = min(5, len(y_c) // 10) if len(y_c) >= 50 else 3
            lin_r2 = float(np.mean(cross_val_score(
                Ridge(alpha=1.0), ctx.X_sc, y_c, cv=cv, scoring="r2",
            )))
            gbt_r2 = float(np.mean(cross_val_score(
                GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                          random_state=ctx.seed),
                ctx.X_sc, y_c, cv=cv, scoring="r2",
            )))

        out["linear_r2"]    = lin_r2
        out["nonlinear_r2"] = gbt_r2
        out["complexity_ratio"] = (
            gbt_r2 / max(lin_r2, 0.01) if np.isfinite(lin_r2) else float("nan")
        )
        r2_best = max(gbt_r2, 0.0) if np.isfinite(gbt_r2) else float("nan")
        out["snr"] = (
            r2_best / max(1.0 - r2_best, 0.01)
            if np.isfinite(r2_best) else float("nan")
        )

        if ctx.X_sc.shape[0] >= 10 and y_varies:
            out.update(_tree_complexity_for_target_r2(
                ctx.X_sc,
                ctx.y_clean,
                seed=ctx.seed,
            ))
        else:
            out.update({
                "tree_target_met": float("nan"),
                "tree_min_depth": float("nan"),
                "tree_terminal_nodes": float("nan"),
                "tree_achieved_r2": float("nan"),
            })
        return out


def _tree_complexity_for_target_r2(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    target_r2: float = 0.8,
    max_allowed_depth: int = 30,
    cv_folds: int = 5,
) -> dict[str, float]:
    """CART depth and diagnostics needed to reach target out-of-sample R²."""
    n_samples = X.shape[0]
    n_splits = min(cv_folds, n_samples // 2)
    if n_splits < 2:
        return {
            "tree_target_met": float("nan"),
            "tree_min_depth": float("nan"),
            "tree_terminal_nodes": float("nan"),
            "tree_achieved_r2": float("nan"),
        }

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    achieved_r2 = float("nan")

    for depth in range(1, max_allowed_depth + 1):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=seed)
        r2_scores = cross_val_score(tree, X, y, cv=cv, scoring="r2")
        achieved_r2 = float(np.mean(r2_scores))

        if achieved_r2 >= target_r2:
            tree.fit(X, y)
            return {
                "tree_target_met": 1.0,
                "tree_min_depth": float(depth),
                "tree_terminal_nodes": float(tree.get_n_leaves()),
                "tree_achieved_r2": achieved_r2,
            }

    tree = DecisionTreeRegressor(
        max_depth=max_allowed_depth,
        random_state=seed,
    ).fit(X, y)
    return {
        "tree_target_met": 0.0,
        "tree_min_depth": float(max_allowed_depth),
        "tree_terminal_nodes": float(tree.get_n_leaves()),
        "tree_achieved_r2": achieved_r2,
    }
