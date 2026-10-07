"""Shared SHAP scalar helpers for RF analysis and downstream plots."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

# Minimum finite (feature, SHAP) pairs for a per-seed direction estimate.
_MIN_SHAP_DIR_OBS: int = 3


def shap_dir_spearman(
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    *,
    min_obs: int = _MIN_SHAP_DIR_OBS,
) -> float:
    """Spearman rho between one feature's values and its SHAP (one seed)."""
    m = np.isfinite(feature_values) & np.isfinite(shap_values)
    xv = feature_values[m].astype(float)
    sv = shap_values[m].astype(float)
    if xv.size < min_obs or np.std(xv) == 0 or np.std(sv) == 0:
        return float("nan")
    c = spearmanr(xv, sv).correlation
    return float(c) if c is not None and np.isfinite(c) else float("nan")


def shap_dir_vector(
    feature_values: np.ndarray,
    shap_oof: np.ndarray,
    *,
    min_obs: int = _MIN_SHAP_DIR_OBS,
) -> np.ndarray:
    """Per-feature ``shap_dir`` for one seed's OOF matrices."""
    p = shap_oof.shape[1]
    out = np.full(p, np.nan)
    for jx in range(p):
        rho = shap_dir_spearman(
            feature_values[:, jx],
            shap_oof[:, jx],
            min_obs=min_obs,
        )
        if np.isfinite(rho):
            out[jx] = rho
    return out


def shap_mean_abs_vector(shap_oof: np.ndarray) -> np.ndarray:
    """Per-feature ``shap_mean_abs`` for one seed's OOF SHAP matrix."""
    return np.nanmean(np.abs(shap_oof), axis=0)


def mean_shap_dir_across_seeds(
    per_seed: list[tuple[np.ndarray, np.ndarray]],
    feature_index: int,
) -> float:
    """Cross-seed mean of per-seed Spearman — matches aggregated ``shap_dir``."""
    rhos = [
        shap_dir_spearman(X[:, feature_index], S[:, feature_index])
        for X, S in per_seed
    ]
    finite = [r for r in rhos if np.isfinite(r)]
    return float(np.mean(finite)) if finite else float("nan")


def mean_shap_mean_abs_across_seeds(
    per_seed: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Cross-seed mean of per-seed ``mean|SHAP|`` — matches ``shap_mean_abs``."""
    if not per_seed:
        return np.array([], dtype=float)
    per_seed_imp = np.stack(
        [shap_mean_abs_vector(S) for _, S in per_seed],
        axis=0,
    )
    return np.nanmean(per_seed_imp, axis=0)
