"""HS Tree degrees-of-freedom complexity metric — classification variant.

Classification counterpart of :mod:`hs_tree_complexity`.  Uses
``imodels.HSTreeClassifierCV`` to pick a CV-optimal shrinkage λ for a
CART classifier and summarises the resulting tree with the same
degrees-of-freedom scores as the regression module, plus a
multi-class parameter count::

    clf_hs_dof       = Σ_j  N_j / (N_j + λ)     (effective unshrunk splits)
    clf_hs_param_dof = (C − 1) · clf_hs_dof      (heuristic parameter DOF)

The hierarchical-shrinkage update inside imodels is identical for
regressors and classifiers — for node *j* with parent sample count
``N_parent``, the shrinkage factor is ``1 / (1 + λ/N_parent)`` regardless
of task (see ``imodels/tree/hierarchical_shrinkage.py``).  Only the
quantity being shrunk differs: leaf means for regression, leaf
probabilities for classification.

``clf_hs_dof`` counts **splits** (one shrinkage factor per internal node,
same as regression ``hs_dof``).  ``clf_hs_param_dof`` multiplies by
``C − 1`` because each node's class-probability vector has ``C − 1``
independent degrees of freedom.  ``clf_hs_dof_fraction`` is unchanged
(the ``C − 1`` cancels in numerator and denominator).

Predictive quality at λ_opt is reported as:

* ``clf_hs_log_loss`` — cross-validated log-loss (lower is better);
* ``clf_hs_score_prior = 1 − clf_hs_log_loss / H(Y)`` — baseline is the
  **empirical class prior** (entropy of ``ctx.class_counts``); 0 means
  no gain over predicting training frequencies, recommended for
  meta-feature / calibration analysis.

Key implementation differences vs ``HSTreeComplexity``
------------------------------------------------------
* Labels are passed as integer class IDs (``ctx.y_int``) **without
  centring**.  Regression centres with ``y - y.mean()``; here the
  classifier handles encoding internally.
* λ is chosen by a single :class:`~sklearn.model_selection.StratifiedKFold`
  pass that also yields ``clf_hs_log_loss`` for free at the chosen λ.
* HS shrinkage is implemented **natively** by
  :func:`_hs_shrink_classifier_in_place` rather than via
  :class:`imodels.HSTreeClassifier`.  Reason: in scikit-learn ≥ 1.4
  ``tree_.value`` already stores normalised class probabilities (row sum
  = 1), but imodels assumes raw counts and divides by ``n_samples``
  again, so the shrunk leaf values end up on the wrong scale and
  ``predict_proba`` returns rows summing to ``~1/n_samples``.  The
  native implementation reads ``tree_.value`` as-is (re-normalising per
  row to be version-independent), runs the standard node-based
  ``1 / (1 + λ/N_parent)`` recursion, and writes the shrunk
  probabilities back.
* Stratification plus ``min(class_counts) >= cv_folds`` up front keeps CV
  well-defined; per-fold log-loss still maps test labels through
  ``base.classes_`` (training-fold classes only) so non-contiguous global
  labels and rare fold edge cases do not index ``predict_proba`` by raw ``y``.
* The dataset must have ``n_classes >= 2``.

Geometric caveat
----------------
Strict equivalence of HS-Tree to ridge regression on an orthogonal stump
basis (Agarwal et al. 2022) holds only under squared loss.  Under
cross-entropy the basis is no longer orthogonal, so ``clf_hs_dof`` /
``clf_hs_param_dof`` should be read as complexity scores rather than
strict Bayesian degrees-of-freedom.  ``clf_hs_param_dof`` is a
heuristic (orthogonal-basis intuition); relative ranking across datasets
remains informative.

The XOR/interaction failure mode flagged in ``HSTreeComplexity`` applies
equally here: greedy CART splits are myopic, so pure high-order
interaction targets will yield a misleadingly low DOF score.
"""
from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .. import (
    ClassificationDatasetFeatureContext,
    DatasetFeatureContext,
    DatasetFeatureGroup,
)
from ._utils import shannon_entropy
from .hs_tree_complexity import (
    _n_scaled_reg_params,
    _resolve_max_leaf_nodes,
    _warn_hs_skip,
)


def _hs_shrink_classifier_in_place(tree, lam: float) -> None:
    """Apply node-based HS shrinkage to a sklearn classifier ``tree_`` in place.

    Walks the tree in DFS pre-order (parents before children — sklearn's
    storage convention) and overwrites ``tree.value`` with::

        v_root^s = v_root
        v_j^s    = v_{parent(j)}^s + (v_j - v_{parent(j)}) / (1 + λ/N_{parent(j)})

    where ``v_j`` is the empirical class-probability vector at node *j*
    (row-normalised from ``tree.value[j]`` so it is correct for any
    sklearn version) and ``N_{parent(j)}`` is the parent's training
    sample count.
    """
    n_nodes = tree.node_count
    val = np.asarray(tree.value, dtype=np.float64).copy()  # (n_nodes, 1, K)
    sums = val.sum(axis=2, keepdims=True)
    sums[sums == 0.0] = 1.0
    val /= sums                                            # row-normalised probs

    children_left = tree.children_left
    children_right = tree.children_right
    parent = np.full(n_nodes, -1, dtype=np.int64)
    for i in range(n_nodes):
        if children_left[i] != -1:
            parent[children_left[i]] = i
            parent[children_right[i]] = i

    n_samp = tree.n_node_samples.astype(np.float64)
    shrunk = np.empty_like(val)
    shrunk[0] = val[0]
    for i in range(1, n_nodes):
        p = parent[i]
        shrunk[i] = shrunk[p] + (val[i] - val[p]) / (1.0 + lam / n_samp[p])

    tree.value[:] = shrunk


def _fold_log_loss_from_proba(
    proba: np.ndarray,
    y_te: np.ndarray,
    classes: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean negative log-likelihood on test rows whose label is in ``classes``.

    ``predict_proba`` columns follow sorted ``classes`` (sklearn convention),
    not raw integer label values.  Rows with labels absent from the training
    fold are excluded; returns NaN if no row is evaluable.
    """
    classes = np.asarray(classes)
    y_te = np.asarray(y_te)
    col = np.searchsorted(classes, y_te)
    valid = (col < len(classes)) & (classes[col] == y_te)
    if not np.any(valid):
        return float("nan")
    idx = np.arange(len(y_te))[valid]
    p_true = np.clip(proba[idx, col[valid]], eps, 1.0)
    return float(-np.nanmean(np.log(p_true)))


class HSTreeClassificationComplexity(DatasetFeatureGroup):
    """Effective degrees of freedom from a CV-optimal HS Tree (classification).

    Parameters
    ----------
    max_leaf_nodes:
        Tree budget.  ``'auto'`` (default) sets ``max(5, int(sqrt(n)))``
        at compute time so capacity scales with sample size.  Pass an
        integer to fix it across all datasets.
    """

    name = "hs_tree_classification_complexity"
    feature_names = (
        "clf_hs_dof",            # Σ N_j/(N_j+λ): effective unshrunk splits
        "clf_hs_param_dof",      # (C−1)·clf_hs_dof: multi-class parameter DOF
        "clf_hs_dof_fraction",   # clf_hs_dof / (K_leaves - 1)
        "clf_hs_lambda_ratio",   # CV-optimal λ/n (scale-invariant)
        "clf_hs_log_loss",       # CV log-loss at λ_opt (lower = better)
        "clf_hs_score_prior",    # 1 − log_loss / H(Y): prior baseline
        "clf_hs_n_leaves",       # leaf count K_leaves
    )

    def __init__(self, max_leaf_nodes: Union[int, str] = "auto") -> None:
        self.max_leaf_nodes = max_leaf_nodes

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        assert isinstance(ctx, ClassificationDatasetFeatureContext)
        out: dict[str, float] = {k: float("nan") for k in self.feature_names}

        X, y = ctx.X_sc, ctx.y_int
        n, d = X.shape[0], X.shape[1]
        K = int(ctx.n_classes)
        if d == 0:
            _warn_hs_skip(self.name, "no numeric features (d=0)")
            return out
        if n < 20:
            _warn_hs_skip(self.name, f"too few samples (n={n} < 20)")
            return out
        if K < 2:
            _warn_hs_skip(self.name, f"single class (K={K} < 2)")
            return out

        cv_folds = min(5, n // 10) if n >= 50 else 3
        if cv_folds < 2:
            _warn_hs_skip(
                self.name,
                f"insufficient data for cross-validation (cv_folds={cv_folds} < 2)",
            )
            return out
        # Stratification (and imodels' internal KFold) need every class to
        # have at least `cv_folds` samples so the per-fold log-loss is
        # well-defined.
        min_class = int(ctx.class_counts.min())
        if min_class < cv_folds:
            _warn_hs_skip(
                self.name,
                "minority class too small for stratified CV "
                f"(min_class_count={min_class} < cv_folds={cv_folds})",
            )
            return out

        max_ln = _resolve_max_leaf_nodes(self.max_leaf_nodes, n)
        reg_params = _n_scaled_reg_params(n)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skf = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=ctx.seed,
                )
                # fold_losses[lam_idx, fold_idx] = CV log-loss for that λ on
                # that fold.  One CV pass yields both λ-selection and
                # clf_hs_log_loss at the chosen λ.
                fold_losses = np.full((len(reg_params), cv_folds), np.nan)
                for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
                    X_tr, X_te = X[tr], X[te]
                    y_tr, y_te = y[tr], y[te]

                    base = DecisionTreeClassifier(
                        max_leaf_nodes=max_ln, random_state=ctx.seed,
                    )
                    base.fit(X_tr, y_tr)
                    # Snapshot the unshrunk leaf probabilities; we restore
                    # before applying each λ so successive shrinks all see
                    # the same baseline tree.
                    original_value = np.asarray(base.tree_.value).copy()

                    for lam_idx, lam_candidate in enumerate(reg_params):
                        base.tree_.value[:] = original_value
                        _hs_shrink_classifier_in_place(
                            base.tree_, lam_candidate,
                        )
                        proba = base.predict_proba(X_te)
                        fold_losses[lam_idx, fold_idx] = (
                            _fold_log_loss_from_proba(
                                proba, y_te, base.classes_,
                            )
                        )

                mean_losses = np.nanmean(fold_losses, axis=1)
                min_valid_folds = max(1, (cv_folds + 1) // 2)
                n_valid_folds = np.sum(np.isfinite(fold_losses), axis=1)
                mean_losses[n_valid_folds < min_valid_folds] = np.nan
                if not np.any(np.isfinite(mean_losses)):
                    _warn_hs_skip(
                        self.name,
                        "CV log-loss undefined "
                        f"(<{min_valid_folds} valid folds for every λ)",
                    )
                    return out

                best_idx = int(np.nanargmin(mean_losses))
                lam = float(reg_params[best_idx])
                ll = float(mean_losses[best_idx])

                # Final fit on full data: only the structure
                # (n_node_samples, leaf count) feeds the DOF formula, so
                # shrinkage is unnecessary here.
                final = DecisionTreeClassifier(
                    max_leaf_nodes=max_ln, random_state=ctx.seed,
                )
                final.fit(X, y)
                tree = final

            # DOF = Σ N_j / (N_j + λ) over internal nodes
            internal_mask = tree.tree_.children_left != -1
            n_node = tree.tree_.n_node_samples[internal_mask].astype(float)
            dof = float(np.sum(n_node / (n_node + lam))) if n_node.size > 0 else 0.0

            n_leaves = float(tree.get_n_leaves())
            k_minus_1 = max(1.0, n_leaves - 1.0)

            class_probs = np.asarray(ctx.class_counts, dtype=float)
            class_probs /= class_probs.sum()
            label_entropy = shannon_entropy(class_probs)

            score_prior = (
                1.0 - ll / label_entropy
                if label_entropy > 1e-12
                else float("nan")
            )
        except Exception as exc:
            _warn_hs_skip(
                self.name,
                f"computation failed ({type(exc).__name__}: {exc})",
            )
            return out

        out["clf_hs_dof"] = dof
        out["clf_hs_param_dof"] = dof * float(K - 1)
        out["clf_hs_dof_fraction"] = dof / k_minus_1
        out["clf_hs_lambda_ratio"] = lam / n
        out["clf_hs_log_loss"] = ll
        out["clf_hs_score_prior"] = score_prior
        out["clf_hs_n_leaves"] = n_leaves
        return out
