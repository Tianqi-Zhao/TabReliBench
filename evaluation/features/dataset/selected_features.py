"""Selected dataset-level meta-features and response categories.

The selected lists keep:

* features that are approximately independent of n/k/d,
* normalized or null-adjusted replacements for features that carry n/k/d, and
* explicit n/k/d channels.

Excluded features are left as comments next to the relevant group so plotting
code can import only the selected tuples below.

``context_to_feature_ratio`` is injected by the evaluation table from
``n_context / n_features``. It is selected instead of the extractor's
``dim_ratio`` because the former has the more natural "available context per
feature" direction for downstream plots.

Response categories group evaluation responses into *prediction*,
*calibration*, and *prediction + calibration* buckets.  These are used by the
feature-selection pipeline (and potentially other analyses) to intersect
top-k features within each category independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.metrics.directions import metric_category


REGRESSION_SELECTED_DATASET_FEATURES: tuple[str, ...] = (
    # target_distribution: all selected.
    "y_skew",
    "y_kurtosis",
    "y_cv",
    "y_bimodality",
    "y_ks_normal",
    "outlier_frac_2sigma",
    "outlier_frac_3sigma",
    "y_entropy",

    # signal_quality.
    "max_abs_corr",
    "mean_abs_corr",
    "nonlinearity_gap",
    "mi_mean",
    "mi_max",
    # Excluded: mi_entropy -> use mi_entropy_norm.
    "mi_entropy_norm",
    "gini_mi",
    "ns_ratio",
    "near_constant_frac",

    # complexity_nonlinearity.
    "linear_r2",
    "nonlinear_r2",
    "complexity_ratio",
    "snr",
    "tree_target_met",
    # Excluded: tree_min_depth, tree_terminal_nodes.
    "tree_achieved_r2",

    # tree_shape.
    # Excluded: tree_depth.
    "tree_imbalance",
    "tree_leaf_depth_cv",
    "tree_var_importance_gini",
    "tree_var_importance_top1",
    "tree_feature_used_frac",

    # hs_tree_complexity.
    # Excluded: hs_dof, hs_n_leaves -> use hs_dof_fraction.
    "hs_dof_fraction",
    "hs_lambda_ratio",
    "hs_r2",

    # heteroscedasticity_local.
    "heteroscedasticity_ratio",
    "conceptvar",
    # Excluded: wg_dist -> use wg_dist_null_ratio.
    "wg_dist_null_ratio",

    # dimensionality_capacity.
    "n_train",
    "n_features",
    "context_to_feature_ratio",
    "effective_dim_ratio",
    # Excluded: dim_ratio -> use context_to_feature_ratio.
    # Excluded: condition_number.
    "discrete_frac",
    "binary_frac",
    "nan_frac_overall",
    # Excluded: nan_col_frac.

    # feature_importance_structure.
    "mean_inter_feature_corr",
    "max_inter_feature_corr",

    # feature_moments: all selected.
    "feat_skew_mean",
    "feat_skew_max",
    "feat_kurtosis_mean",
    "feat_kurtosis_max",
    "feat_outlier_frac_mean",
    "feat_bimodality_mean",
    "feat_sparsity_mean",
    "feat_mad_std_ratio_mean",
    "feat_scale_dispersion",

    # categorical_features: all selected.
    "cat_cardinality_mean",
    "cat_cardinality_max",
    "cat_cardinality_var",
    "cat_id_like_frac",
    "cat_norm_entropy_mean",
    "cat_mode_frac_mean",
    "cat_rare_category_frac_mean",
)


CLASSIFICATION_SELECTED_DATASET_FEATURES: tuple[str, ...] = (
    # class_distribution.
    "n_classes",
    # Excluded: class_entropy -> use class_entropy_norm.
    "class_entropy_norm",
    "majority_class_frac",
    "minority_class_frac",
    "imbalance_ratio",
    # Excluded: class_gini -> use class_gini_norm.
    "class_gini_norm",

    # class_signal_quality.
    # Excluded: mi_mean, mi_max -> use normalized MI by H(Y).
    "mi_mean_norm",
    "mi_max_norm",
    # Excluded: mi_entropy -> use mi_entropy_norm.
    "mi_entropy_norm",
    "gini_mi",
    "ns_ratio",
    # Excluded: anova_f_mean, anova_f_max -> use eta-squared effect sizes.
    "anova_eta2_mean",
    "anova_eta2_max",
    "near_constant_frac",

    # classification_complexity.
    # Excluded: linear_acc, nonlinear_acc, complexity_ratio.
    "linear_acc_norm",
    "nonlinear_acc_norm",
    "complexity_ratio_norm",
    "tree_target_met",
    # Excluded: tree_min_depth, tree_terminal_nodes, tree_achieved_acc.
    "tree_achieved_acc_norm",

    # class_tree_shape.
    # Excluded: tree_depth.
    "tree_imbalance",
    "tree_leaf_depth_cv",
    "tree_var_importance_gini",
    "tree_var_importance_top1",
    "tree_feature_used_frac",

    # hs_tree_classification_complexity.
    # Excluded: clf_hs_dof, clf_hs_param_dof, clf_hs_n_leaves.
    "clf_hs_dof_fraction",
    "clf_hs_lambda_ratio",
    # Excluded: clf_hs_log_loss -> use prior-baseline score.
    "clf_hs_score_prior",

    # class_local_structure.
    "knn_disagreement_frac",
    "boundary_frac",
    # Excluded: wg_dist -> use wg_dist_null_ratio.
    "wg_dist_null_ratio",
    "n2_intra_extra_ratio",
    "f2_overlap_mean",
    "f3_max_efficiency",

    # clustering_structure.
    # Excluded: silhouette, davies_bouldin, calinski_harabasz,
    # centroid_radius_sep_ratio -> use permutation ratios.
    "silhouette_perm_ratio",
    "davies_bouldin_perm_ratio",
    "calinski_harabasz_perm_ratio",
    "centroid_radius_sep_perm_ratio",

    # dimensionality_capacity.
    "n_train",
    "n_features",
    "context_to_feature_ratio",
    "effective_dim_ratio",
    # Excluded: dim_ratio -> use context_to_feature_ratio.
    # Excluded: condition_number.
    "discrete_frac",
    "binary_frac",
    "nan_frac_overall",
    # Excluded: nan_col_frac.

    # feature_importance_structure.
    "mean_inter_feature_corr",
    "max_inter_feature_corr",

    # feature_moments: all selected.
    "feat_skew_mean",
    "feat_skew_max",
    "feat_kurtosis_mean",
    "feat_kurtosis_max",
    "feat_outlier_frac_mean",
    "feat_bimodality_mean",
    "feat_sparsity_mean",
    "feat_mad_std_ratio_mean",
    "feat_scale_dispersion",

    # categorical_features: all selected.
    "cat_cardinality_mean",
    "cat_cardinality_max",
    "cat_cardinality_var",
    "cat_id_like_frac",
    "cat_norm_entropy_mean",
    "cat_mode_frac_mean",
    "cat_rare_category_frac_mean",
)


SELECTED_DATASET_FEATURES_BY_TASK: dict[str, tuple[str, ...]] = {
    "regression": REGRESSION_SELECTED_DATASET_FEATURES,
    "classification": CLASSIFICATION_SELECTED_DATASET_FEATURES,
}


# -----------------------------------------------------------------------------
# Dependency-based feature groups
# -----------------------------------------------------------------------------
#
# These groups answer a different question from the extractor-module groups
# above: which observed variables are required to compute a meta-feature?
#
# * x_only:       computable from the design matrix X (including its shape).
# * y_only:       computable from the target/labels Y without looking at X.
# * xy_relation:  requires both X and Y.  This includes descriptors of a
#                 supervised fitted model (for example CART shape), because
#                 changing Y can change the fitted model and hence the feature.
#
# The three groups are an exact, disjoint partition of each task's selected
# feature list.  ``heteroscedasticity_ratio`` is y_only under the current
# implementation: it bins Y by Y quantiles and compares within-bin Y standard
# deviations; it does not condition on X.

X_ONLY_SELECTED_DATASET_FEATURES: tuple[str, ...] = (
    # Dimensionality and capacity.
    "n_train",
    "n_features",
    "context_to_feature_ratio",
    "effective_dim_ratio",
    "discrete_frac",
    "binary_frac",
    "nan_frac_overall",

    # Numeric structure and marginal shape.
    "near_constant_frac",
    "mean_inter_feature_corr",
    "max_inter_feature_corr",
    "wg_dist_null_ratio",
    "feat_skew_mean",
    "feat_skew_max",
    "feat_kurtosis_mean",
    "feat_kurtosis_max",
    "feat_outlier_frac_mean",
    "feat_bimodality_mean",
    "feat_sparsity_mean",
    "feat_mad_std_ratio_mean",
    "feat_scale_dispersion",

    # Categorical-column distribution.
    "cat_cardinality_mean",
    "cat_cardinality_max",
    "cat_cardinality_var",
    "cat_id_like_frac",
    "cat_norm_entropy_mean",
    "cat_mode_frac_mean",
    "cat_rare_category_frac_mean",
)


REGRESSION_Y_ONLY_SELECTED_DATASET_FEATURES: tuple[str, ...] = (
    "y_skew",
    "y_kurtosis",
    "y_cv",
    "y_bimodality",
    "y_ks_normal",
    "outlier_frac_2sigma",
    "outlier_frac_3sigma",
    "y_entropy",
    "heteroscedasticity_ratio",
)


REGRESSION_XY_RELATION_SELECTED_DATASET_FEATURES: tuple[str, ...] = (
    # Marginal association and mutual information.
    "max_abs_corr",
    "mean_abs_corr",
    "nonlinearity_gap",
    "mi_mean",
    "mi_max",
    "mi_entropy_norm",
    "gini_mi",
    "ns_ratio",

    # Supervised predictive and structural complexity.
    "linear_r2",
    "nonlinear_r2",
    "complexity_ratio",
    "snr",
    "tree_target_met",
    "tree_achieved_r2",
    "tree_imbalance",
    "tree_leaf_depth_cv",
    "tree_var_importance_gini",
    "tree_var_importance_top1",
    "tree_feature_used_frac",
    "hs_dof_fraction",
    "hs_lambda_ratio",
    "hs_r2",

    # Y variation inside X-defined neighborhoods.
    "conceptvar",
)


CLASSIFICATION_Y_ONLY_SELECTED_DATASET_FEATURES: tuple[str, ...] = (
    "n_classes",
    "class_entropy_norm",
    "majority_class_frac",
    "minority_class_frac",
    "imbalance_ratio",
    "class_gini_norm",
)


CLASSIFICATION_XY_RELATION_SELECTED_DATASET_FEATURES: tuple[str, ...] = (
    # Feature-label association.
    "mi_mean_norm",
    "mi_max_norm",
    "mi_entropy_norm",
    "gini_mi",
    "ns_ratio",
    "anova_eta2_mean",
    "anova_eta2_max",

    # Supervised predictive and structural complexity.
    "linear_acc_norm",
    "nonlinear_acc_norm",
    "complexity_ratio_norm",
    "tree_target_met",
    "tree_achieved_acc_norm",
    "tree_imbalance",
    "tree_leaf_depth_cv",
    "tree_var_importance_gini",
    "tree_var_importance_top1",
    "tree_feature_used_frac",
    "clf_hs_dof_fraction",
    "clf_hs_lambda_ratio",
    "clf_hs_score_prior",

    # Label-conditioned local geometry, overlap, and clustering.
    "knn_disagreement_frac",
    "boundary_frac",
    "n2_intra_extra_ratio",
    "f2_overlap_mean",
    "f3_max_efficiency",
    "silhouette_perm_ratio",
    "davies_bouldin_perm_ratio",
    "calinski_harabasz_perm_ratio",
    "centroid_radius_sep_perm_ratio",
)


DATASET_FEATURE_GROUPS_BY_TASK: dict[str, dict[str, tuple[str, ...]]] = {
    "regression": {
        "x_only": X_ONLY_SELECTED_DATASET_FEATURES,
        "y_only": REGRESSION_Y_ONLY_SELECTED_DATASET_FEATURES,
        "xy_relation": REGRESSION_XY_RELATION_SELECTED_DATASET_FEATURES,
    },
    "classification": {
        "x_only": X_ONLY_SELECTED_DATASET_FEATURES,
        "y_only": CLASSIFICATION_Y_ONLY_SELECTED_DATASET_FEATURES,
        "xy_relation": CLASSIFICATION_XY_RELATION_SELECTED_DATASET_FEATURES,
    },
}


def validate_dataset_feature_groups(task: str) -> None:
    """Raise when dependency groups are not an exact selected-feature partition."""
    if task not in SELECTED_DATASET_FEATURES_BY_TASK:
        raise ValueError(f"unknown task: {task!r}")
    selected = SELECTED_DATASET_FEATURES_BY_TASK[task]
    groups = DATASET_FEATURE_GROUPS_BY_TASK[task]
    flat = [feature for features in groups.values() for feature in features]
    duplicates = sorted({feature for feature in flat if flat.count(feature) > 1})
    missing = sorted(set(selected) - set(flat))
    extra = sorted(set(flat) - set(selected))
    if duplicates or missing or extra or len(flat) != len(selected):
        raise ValueError(
            f"invalid {task} dependency partition: duplicates={duplicates}, "
            f"missing={missing}, extra={extra}"
        )


for _task in SELECTED_DATASET_FEATURES_BY_TASK:
    validate_dataset_feature_groups(_task)


# ─────────────────────────────────────────────────────────────────────────────
# Response categories
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResponseCategory:
    """Group of evaluation responses that belong to the same semantic category.

    Parameters
    ----------
    name:
        Human-readable category label (e.g. ``"prediction"``,
        ``"calibration"``, ``"pred_cal"``).
    responses:
        Response column names that belong to this category.
    alpha_dependent:
        Mapping from response name to the list of alpha values it requires.
        Responses not present in this dict are treated as alpha-free.
    """
    name: str
    responses: list[str]
    alpha_dependent: dict[str, list[float]] = field(default_factory=dict)


# The response categories below are *derived* from the canonical
# ``RESPONSE_CATEGORY_METRICS`` map in ``evaluation.metrics.directions`` so the
# "which category does a metric belong to" knowledge lives in exactly one place.
# Here we only declare (a) the curated honest-ranking *subset* of responses that
# feature selection runs on, per (task, kind), and (b) the per-response alpha
# requirements; the grouping into categories is looked up via
# :func:`metric_category`.  The legacy ``pred_cal`` name (≡ proper_score) is kept
# because it is written into ``feature_selection_by_category.csv``.

# Curated subset feature selection runs on, in display order.  abs vs rel differ
# only by dropping the bidirectional signed coverage deviations from rel.
FEATSEL_RESPONSES: dict[tuple[str, str], list[str]] = {
    ("regression", "abs"): [
        "r2", "pit_ks_stat", "cov_dev_signed", "cov_abs_dev",
        "avg_width_norm", "wsc_dev_signed", "wsc_abs_dev", "crps_mean",
    ],
    ("regression", "rel"): [
        "r2", "pit_ks_stat", "cov_abs_dev", "avg_width_norm",
        "wsc_abs_dev", "crps_mean",
    ],
    # Classification has no abs/rel split for response categories.
    ("classification", "abs"): [
        "accuracy", "confidence_ece_em", "classwise_ece_em", "cmce_em",
        "brier_score",
    ],
    ("classification", "rel"): [
        "accuracy", "confidence_ece_em", "classwise_ece_em", "cmce_em",
        "brier_score",
    ],
}

# Per-response miscoverage levels required by alpha-dependent responses.
FEATSEL_ALPHA_DEPENDENT: dict[str, list[float]] = {
    "cov_dev_signed": [0.05, 0.2],
    "cov_abs_dev":    [0.05, 0.2],
    "avg_width_norm": [0.05, 0.2],
    "wsc_dev_signed": [0.05, 0.2],
    "wsc_abs_dev":    [0.05, 0.2],
}

# Legacy display name written into feature_selection_by_category.csv.
_FEATSEL_CATEGORY_NAME: dict[str, str] = {"proper_score": "pred_cal"}

# Human-readable labels for figures/tables: the *stored* category name
# (as written into feature_selection_by_category.csv) → display label.
# ``pred_cal`` (≡ proper_score) reads as "proper score" in plots.
FEATSEL_CATEGORY_DISPLAY: dict[str, str] = {"pred_cal": "proper score"}


def featsel_category_label(category: str) -> str:
    """Human-readable label for a feature-selection category name.

    Single source of truth shared by plotting code so the ``pred_cal`` →
    ``"proper score"`` relabelling lives in one place.
    """
    return FEATSEL_CATEGORY_DISPLAY.get(category, category)


def _build_response_categories(task: str, kind: str) -> list[ResponseCategory]:
    """Group the curated ``FEATSEL_RESPONSES`` subset into ResponseCategory
    objects using the canonical category map.  Both responses *and* categories
    keep their first-appearance order in ``FEATSEL_RESPONSES`` (dicts preserve
    insertion order), and the per-response alpha metadata is attached."""
    by_cat: dict[str, list[str]] = {}
    for resp in FEATSEL_RESPONSES[(task, kind)]:  # insertion order preserved
        by_cat.setdefault(metric_category(resp), []).append(resp)

    categories: list[ResponseCategory] = []
    for cat, members in by_cat.items():
        alpha_dep = {
            resp: FEATSEL_ALPHA_DEPENDENT[resp]
            for resp in members
            if resp in FEATSEL_ALPHA_DEPENDENT
        }
        categories.append(ResponseCategory(
            _FEATSEL_CATEGORY_NAME.get(cat, cat), members, alpha_dep,
        ))
    return categories


# Convenience lookup by (task, kind).  kind is "abs" or "rel".
RESPONSE_CATEGORIES_BY_TASK_KIND: dict[
    tuple[str, str], list[ResponseCategory]
] = {
    (task, kind): _build_response_categories(task, kind)
    for task in ("regression", "classification")
    for kind in ("abs", "rel")
}
