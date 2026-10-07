"""HS Tree degrees-of-freedom complexity metric.

Hierarchical Shrinkage (HS) Trees (Agarwal et al. 2022) are equivalent to
ridge regression on a local decision stump basis.  For internal node j the
basis function is:

    ψ_j(x) = (N_L · 1{x goes right} − N_R · 1{x goes left}) / √(N_L · N_R)

active only within node j's region.  This basis is orthogonal (the centering
kills cross-terms for all ancestor/descendant pairs), so B^T B = diag(N_j),
and the effective degrees of freedom reduces to a sum over internal nodes:

    DOF = Σ_j  N_j / (N_j + λ)

where N_j = training samples in node j and λ = CV-optimal HS shrinkage.

Design choices
--------------
max_leaf_nodes='auto':
    Sets max_leaf_nodes = max(5, int(sqrt(n))) so tree capacity scales with
    the sample size.  Equivalently, each leaf holds on average sqrt(n) samples
    at maximum depth.  This prevents the tree budget from being fixed while
    datasets of very different sizes are compared.

n-scaled λ grid:
    The regularisation grid uses λ = n × r for r in {0, 1e-3, 0.01, 0.1, 0.5,
    1, 10}.  This is important: with a fixed grid like [0..500], large datasets
    (n ≫ 500) always end up with λ/n ≈ 0, so even pure-noise splits at the root
    (N_root ≈ n) contribute N_root/(N_root+500) ≈ 1 to DOF.  Scaling the grid
    by n ensures the metric is comparable across dataset sizes.

hs_dof_fraction:
    DOF / (K-1) where K = tree leaf count.  Answers "what fraction of the
    tree's internal nodes are statistically justified?", removing the dependence
    on tree budget size.  Ranges in [0, 1].

Compared with the CART-based features in complexity.py
-------------------------------------------------------
- DOF is continuous, not a thresholded integer.
- λ is self-calibrated by CV; no target R² hyperparameter needed.
- Small-sample leaves are soft-shrunk toward zero, so noisy splits do not
  inflate complexity the way raw leaf counts do.
- hs_dof_fraction is sample-size-invariant: pure noise stays near 0 regardless
  of n; a clean tree-representable signal approaches 1 as n → ∞.

Known limitation
----------------
Greedy CART fails to find pure interaction functions efficiently (the "XOR
failure mode"): for a target like sign(x₀)·sign(x₁)·sign(x₂)·sign(x₃) every
single-feature split looks equally marginal, so the tree exhausts its budget
on suboptimal splits.  HS DOF then gives a misleadingly low score for such
targets.  This is a property of the tree basis, not the HS regularisation.
"""
from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from imodels import HSTreeRegressor, HSTreeRegressorCV
from sklearn.model_selection import KFold, cross_val_score

from .. import DatasetFeatureContext, DatasetFeatureGroup

# λ/n ratios explored during CV — scaling by n ensures comparable regularisation
# strength across datasets of very different sizes.
_REG_PARAM_RATIOS: list[float] = [0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0]


def _n_scaled_reg_params(n: int) -> list[float]:
    return [r * n for r in _REG_PARAM_RATIOS]


def _resolve_max_leaf_nodes(spec: Union[int, str], n: int) -> int:
    if spec == "auto":
        return max(5, int(np.sqrt(n)))
    return int(spec)


def _warn_hs_skip(feature_group: str, reason: str) -> None:
    warnings.warn(
        f"{feature_group}: {reason}; returning NaN features.",
        RuntimeWarning,
        stacklevel=3,
    )


class HSTreeComplexity(DatasetFeatureGroup):
    """Effective degrees of freedom from a CV-optimal HS Tree.

    Parameters
    ----------
    max_leaf_nodes:
        Tree budget.  ``'auto'`` (default) sets ``max(5, int(sqrt(n)))``
        at compute time so capacity scales with sample size.  Pass an integer
        to fix it across all datasets.
    """

    name = "hs_tree_complexity"
    feature_names = (
        "hs_dof",           # Σ N_j/(N_j+λ) over internal nodes
        "hs_dof_fraction",  # hs_dof / (K-1): fraction of tree capacity supported
        "hs_lambda_ratio",  # CV-optimal λ/n (scale-invariant regularisation strength)
        "hs_r2",            # cross-validated R² of the HS Tree at λ_opt
        "hs_n_leaves",      # leaf count K (max possible DOF + 1)
    )

    def __init__(self, max_leaf_nodes: Union[int, str] = "auto") -> None:
        self.max_leaf_nodes = max_leaf_nodes

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        out: dict[str, float] = {k: float("nan") for k in self.feature_names}

        X, y = ctx.X_sc, ctx.y_clean
        n, d = X.shape[0], X.shape[1]
        if d == 0:
            _warn_hs_skip(self.name, "no numeric features (d=0)")
            return out
        if n < 20:
            _warn_hs_skip(self.name, f"too few samples (n={n} < 20)")
            return out
        if float(np.std(y)) <= 1e-10:
            # Constant y: HS-Tree CV is meaningless (no variance to explain,
            # R² is undefined, λ-selection has nothing to optimise).
            _warn_hs_skip(self.name, "constant y (std <= 1e-10)")
            return out

        y_c = y - y.mean()
        cv_folds = min(5, n // 10) if n >= 50 else 3
        if cv_folds < 2:
            _warn_hs_skip(
                self.name,
                f"insufficient data for cross-validation (cv_folds={cv_folds} < 2)",
            )
            return out

        max_ln = _resolve_max_leaf_nodes(self.max_leaf_nodes, n)
        reg_params = _n_scaled_reg_params(n)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_model = HSTreeRegressorCV(
                    max_leaf_nodes=max_ln,
                    reg_param_list=reg_params,
                    cv=cv_folds,
                )
                cv_model.fit(X, y_c)

            lam = float(cv_model.reg_param)
            tree = cv_model.estimator_  # sklearn DecisionTreeRegressor

            # DOF = Σ N_j / (N_j + λ) over internal nodes
            internal_mask = tree.tree_.children_left != -1
            n_node = tree.tree_.n_node_samples[internal_mask].astype(float)
            dof = float(np.sum(n_node / (n_node + lam))) if n_node.size > 0 else 0.0

            n_leaves = float(tree.get_n_leaves())
            k_minus_1 = max(1.0, n_leaves - 1.0)

            # hs_r2: CV R² of HS Tree at the optimal λ
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=ctx.seed)
            hs_fixed = HSTreeRegressor(
                reg_param=lam,
                max_leaf_nodes=max_ln,
                random_state=ctx.seed,
            )
            r2_scores = cross_val_score(hs_fixed, X, y_c, cv=kf, scoring="r2")
            r2 = float(np.mean(r2_scores))

        except Exception as exc:
            _warn_hs_skip(
                self.name,
                f"computation failed ({type(exc).__name__}: {exc})",
            )
            return out

        out["hs_dof"] = dof
        out["hs_dof_fraction"] = dof / k_minus_1
        out["hs_lambda_ratio"] = lam / n   # λ/n: scale-invariant
        out["hs_r2"] = r2
        out["hs_n_leaves"] = n_leaves
        return out
