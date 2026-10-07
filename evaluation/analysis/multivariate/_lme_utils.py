"""Shared helpers for linear mixed-effects analyzers.

Both :class:`evaluation.analysis.multivariate.lme.LMEAnalyzer` (per-model
fits) and :class:`evaluation.analysis.multivariate.cross_model_lme.\
CrossModelLMEAnalyzer` (pooled-across-models fit with interactions) need
the same low-level utilities:

* sample-size gates that respect dataset-level clustering
  (``_min_lme_groups`` / ``_insufficient_lme_data``),
* QR-pivot collinearity drop on the standardised design,
* Nakagawa & Schielzeth pseudo-R² for Gaussian LMMs,
* z-scoring of dataset-level meta-features with constant-column drop,
* block-aware missing-value imputation that preserves structural NaN
  signals as indicator columns (``_impute_with_block_indicators``).

Keeping them in this module avoids cross-subpackage imports and gives
all multivariate analyzers access to a single coherent preprocessing
pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg as sp_linalg


# ─────────────────────────────────────────────────────────────────────────────
# Sample-size gates
# ─────────────────────────────────────────────────────────────────────────────

_MIN_LME_GROUPS = 30
_MIN_LME_GROUPS_PER_FEATURE = 2
_MIN_LME_REPS_PER_GROUP = 2


def _min_lme_groups(n_features: int) -> int:
    """Minimum distinct ``dataset_id`` groups for a multivariate LME fit."""
    return max(_MIN_LME_GROUPS, _MIN_LME_GROUPS_PER_FEATURE * n_features)


def _insufficient_lme_data(n_obs: int, n_groups: int, n_features: int) -> bool:
    """Return True when the slice lacks data for fixed or random effects.

    Fixed effects on dataset-level meta-features are identified mainly from
    ``n_groups``, not seed-inflated ``n_obs``.  We still require at least
    ``_MIN_LME_REPS_PER_GROUP`` rows per group on average so the random
    intercept and residual variance stay separable.
    """
    if n_groups < _min_lme_groups(n_features):
        return True
    return n_obs < n_groups * _MIN_LME_REPS_PER_GROUP


# ─────────────────────────────────────────────────────────────────────────────
# Design-matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _drop_collinear_columns(
    X: np.ndarray, names: list[str], tol: float = 1e-8,
) -> tuple[list[int], list[str]]:
    """Return ``(keep_indices, keep_names)`` after a rank-revealing QR.

    Uses QR with column pivoting; columns whose pivoted-R diagonal entry
    falls below ``tol * max(|diag(R)|)`` are dropped to fix
    "Singular matrix" failures from near-collinear standardised features.
    """
    if X.shape[1] == 0:
        return [], []
    _, R, piv = sp_linalg.qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0 or diag.max() == 0:
        return [], []
    rank = int((diag > tol * diag.max()).sum())
    keep_idx = sorted(piv[:rank].tolist())
    return keep_idx, [names[i] for i in keep_idx]


def _zscore_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Z-score ``feature_cols`` of ``df``, dropping constant columns first.

    Returns
    -------
    Xz_df
        DataFrame of standardised features (mean 0, std 1) indexed like
        ``df`` and ordered by the *kept* (non-constant) columns.
    mean, std
        ``pd.Series`` of the original-scale mean / std used for
        standardisation, indexed by kept feature names.
    dropped
        List of feature names dropped because they had ~zero variance.
    """
    feat_mat = df[feature_cols].values.astype(float)
    std0 = feat_mat.std(axis=0)
    keep_var = [f for f, k in zip(feature_cols, std0 > 1e-10) if k]
    feat_mat = df[keep_var].values.astype(float)

    mean = feat_mat.mean(axis=0)
    std  = feat_mat.std(axis=0)
    Xz   = (feat_mat - mean) / std

    Xz_df = pd.DataFrame(Xz, columns=keep_var, index=df.index)
    dropped = [f for f in feature_cols if f not in keep_var]
    return (
        Xz_df,
        pd.Series(mean, index=keep_var),
        pd.Series(std,  index=keep_var),
        dropped,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-R² (Nakagawa & Schielzeth)
# ─────────────────────────────────────────────────────────────────────────────

def _pseudo_r2(res, X_design: np.ndarray) -> tuple[float, float]:
    """Nakagawa & Schielzeth marginal & conditional R² for a Gaussian LMM."""
    fe_pred = X_design @ res.fe_params.values
    var_fe  = float(np.var(fe_pred, ddof=0))
    var_re  = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] >= 1 else 0.0
    var_e   = float(res.scale)
    denom   = var_fe + var_re + var_e
    if denom <= 0:
        return float("nan"), float("nan")
    return var_fe / denom, (var_fe + var_re) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Block-aware missing-value imputation
# ─────────────────────────────────────────────────────────────────────────────

# Hard-coded structurally-missing feature blocks.
# Each entry is (indicator_column_name, feature_name_prefix).
# cat_*  — only present for datasets that contain categorical columns.
# clf_hs_* — only present for classification datasets where stratified CV
#            succeeded (requires minority class >= cv_folds samples).
_FEATURE_BLOCKS: tuple[tuple[str, str], ...] = (
    ("has_cat",    "cat_"),
    ("has_clf_hs", "clf_hs_"),
)


def _impute_with_block_indicators(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    blocks: tuple[tuple[str, str], ...] = _FEATURE_BLOCKS,
) -> tuple[pd.DataFrame, list[str]]:
    """Impute missing meta-features using a block-aware B+C strategy.

    **Structurally-missing blocks** (groups of features that are either all
    present or all absent for a given dataset, such as ``cat_*``) are
    handled with a conditional z-score approach:

    1. For each ``(indicator_name, prefix)`` block whose features appear in
       ``feature_cols``:

       * Compute ``present = ~isna.any(axis=1)`` — the subset of rows where
         the entire block is observed.
       * If ``present`` is constant (all True or all False), no indicator
         signal exists; fill NaN with the column mean and continue.
       * Otherwise: fill NaN in each block feature with its within-``present``
         mean so that the absent rows sit at the block's centroid.  Then add
         a binary ``indicator_name`` column (1 = present, 0 = absent).

    After the fill, ``_zscore_features`` will standardise all columns.
    Because absent rows are filled with the within-present mean, their
    z-score becomes exactly 0 (the centroid) — they contribute nothing to
    the fitted slope for those features.  The indicator column captures the
    "has this feature block" signal as an explicit fixed effect.

    **Scattered NaN** (individual features missing for a small fraction of
    datasets, not part of any block) are filled with the global column mean.
    No indicator is added for these — the missingness is too infrequent to
    warrant an extra degree of freedom.

    Parameters
    ----------
    df:
        Input DataFrame.  Must contain all columns in ``feature_cols``.
    feature_cols:
        Ordered list of feature column names to process.
    blocks:
        Sequence of ``(indicator_name, feature_name_prefix)`` pairs
        identifying structurally-missing groups.

    Returns
    -------
    df_out:
        Copy of ``df`` with imputed feature columns and any surviving
        indicator columns appended.
    new_feature_cols:
        Updated feature column list: original order preserved, indicator
        columns for non-trivial blocks appended at the end.
    """
    df_out = df.copy()
    new_cols: list[str] = []
    block_feature_set: set[str] = set()

    for indicator_name, prefix in blocks:
        block_feats = [f for f in feature_cols if f.startswith(prefix)]
        if not block_feats:
            continue
        block_feature_set.update(block_feats)

        # present=True when every feature in the block is non-NaN for that row
        present = ~df_out[block_feats].isna().any(axis=1)

        if present.all() or (~present).all():
            # No variation in presence → indicator would be constant (no
            # information); just fill any NaN with the column mean.
            for f in block_feats:
                col_mean = float(df_out[f].mean()) if df_out[f].notna().any() else 0.0
                df_out[f] = df_out[f].fillna(col_mean)
            continue

        # Fill each block feature with its within-present-subset mean so that
        # absent rows land at the centroid and z-score to 0 afterwards.
        sub_present = df_out.loc[present, block_feats]
        within_means = sub_present.mean(axis=0)
        for f in block_feats:
            df_out[f] = df_out[f].fillna(float(within_means[f]))

        # Add the binary indicator column (0/1 float).
        df_out[indicator_name] = present.astype(float)
        new_cols.append(indicator_name)

    # Scattered NaN: global mean fill, no indicator.
    for f in feature_cols:
        if f in block_feature_set:
            continue
        if df_out[f].isna().any():
            col_mean = float(df_out[f].mean())
            df_out[f] = df_out[f].fillna(col_mean if np.isfinite(col_mean) else 0.0)

    new_feature_cols = list(feature_cols) + new_cols
    return df_out, new_feature_cols
