"""XGBoost-backed feature importance, per seed.

Renamed from ``analysis/importance.py``. Consumes seed-level tables
(``"long_abs"`` / ``"long_rel"``): for each ``(model, response)`` we run
the full K-fold importance pipeline **once per seed**, then aggregate
the per-feature importance scalars (gain, permutation importance, SHAP
mean / mean-abs / std) across seeds with
:class:`MeanSDAggregator` (default).

Defaults to :class:`~sklearn.model_selection.GroupKFold` over
``dataset_id`` within each seed slice — a per-seed slice has one row
per dataset, so held-out folds are always new datasets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from .._aggregation import MeanSDAggregator, SeedAggregator
from ..base import (
    _alpha_to_cell,
    alpha_col_for_html,
    DatasetAnalyzer,
    FeatureImportance,
    group_keys_by_ratio,
    InputKind,
    render_feature_pivot_html,
    render_fit_quality_html,
)
from ..tables import FitQualityTable
from ._shap_utils import shap_dir_vector, shap_mean_abs_vector


# ─────────────────────────────────────────────────────────────────────────────
# Shared K-fold importance helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_xgb(
    task: Literal["regression", "classification"],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    random_state: int,
) -> object:
    """Construct the XGBoost estimator — the single source of hyper-parameters
    for both the K-fold importance loop and the per-seed full-data refit.

    XGBoost handles NaN natively (a learned default split direction per node), so
    callers pass the raw feature matrix through; no imputation is required.
    """
    common = dict(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        tree_method="hist",
        random_state=int(random_state),
        n_jobs=-1,
        importance_type="gain",
    )
    if task == "regression":
        return xgb.XGBRegressor(**common)
    return xgb.XGBClassifier(**common, eval_metric="logloss")


def _xgb_cv_importance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: Literal["regression", "classification"],
    cv_splitter,
    groups: Optional[np.ndarray] = None,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    n_perm_repeats: int = 5,
    random_state: int = 0,
    max_samples_shap: Optional[int] = None,
    compute_shap_interactions: bool = True,
    compute_permutation: bool = True,
    compute_shap: bool = True,
) -> dict:
    """K-fold loop with XGBoost; per-fold gain, optional perm, and OOF SHAP."""
    if task not in {"regression", "classification"}:
        raise ValueError(
            f"task must be 'regression' or 'classification', got {task!r}"
        )
    n, p = X.shape

    scoring = "r2" if task == "regression" else "roc_auc"

    shap_oof = np.full((n, p), np.nan, dtype=np.float32)
    expected_oof = np.full(n, np.nan, dtype=np.float32)
    shap_inter_oof: Optional[np.ndarray] = (
        np.full((n, p, p), np.nan, dtype=np.float32)
        if (compute_shap and compute_shap_interactions) else None
    )
    sub_rng = np.random.default_rng(random_state)

    gain_per_fold:     list[np.ndarray] = []
    perm_per_fold:     list[np.ndarray] = []
    cv_scores:         list[float] = []
    train_scores:      list[float] = []
    train_accuracy_scores: list[float] = []
    expected_per_fold: list[float] = []

    splits = (cv_splitter.split(X, y, groups=groups)
              if groups is not None
              else cv_splitter.split(X, y))

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if task == "classification" and (
            len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2
        ):
            cv_scores.append(float("nan"))
            train_scores.append(float("nan"))
            train_accuracy_scores.append(float("nan"))
            gain_per_fold.append(np.full(p, np.nan))
            perm_per_fold.append(np.full(p, np.nan))
            continue

        model = _make_xgb(
            task, n_estimators, learning_rate, max_depth, random_state,
        )
        model.fit(X_tr, y_tr)

        gain_per_fold.append(
            np.asarray(model.feature_importances_, dtype=float)
        )

        if task == "regression":
            cv_scores.append(float(model.score(X_te, y_te)))
            train_scores.append(float(model.score(X_tr, y_tr)))
            train_accuracy_scores.append(float("nan"))
        else:
            try:
                cv_scores.append(float(roc_auc_score(
                    y_te, model.predict_proba(X_te)[:, 1],
                )))
            except ValueError:
                cv_scores.append(float("nan"))
            try:
                train_scores.append(float(roc_auc_score(
                    y_tr, model.predict_proba(X_tr)[:, 1],
                )))
            except ValueError:
                train_scores.append(float("nan"))
            try:
                train_accuracy_scores.append(float(model.score(X_tr, y_tr)))
            except ValueError:
                train_accuracy_scores.append(float("nan"))

        if compute_permutation:
            perm = permutation_importance(
                model, X_te, y_te,
                n_repeats=n_perm_repeats,
                random_state=random_state,
                n_jobs=-1,
                scoring=scoring,
            )
            perm_per_fold.append(perm.importances_mean)
        else:
            perm_per_fold.append(np.full(p, np.nan))

        if compute_shap:
            if (max_samples_shap is not None
                    and len(test_idx) > max_samples_shap):
                sel = sub_rng.choice(
                    len(test_idx), size=int(max_samples_shap), replace=False,
                )
                X_for_shap = X_te[sel]
                test_idx_used = test_idx[sel]
            else:
                X_for_shap = X_te
                test_idx_used = test_idx

            booster = model.get_booster()
            dshap = xgb.DMatrix(X_for_shap)
            contribs = np.asarray(
                booster.predict(dshap, pred_contribs=True)
            )
            if contribs.ndim == 3:
                contribs = contribs[:, 0, :]
            shap_oof[test_idx_used] = contribs[:, :-1].astype(np.float32)
            expected_oof[test_idx_used] = contribs[:, -1].astype(np.float32)
            expected_per_fold.append(float(contribs[0, -1]))

            if shap_inter_oof is not None:
                inter = np.asarray(
                    booster.predict(dshap, pred_interactions=True)
                )
                if inter.ndim == 4:
                    inter = inter[:, 0, :, :]
                shap_inter_oof[test_idx_used] = (
                    inter[:, :-1, :-1].astype(np.float32)
                )

    return {
        "gain_per_fold":     np.stack(gain_per_fold) if gain_per_fold else
                              np.empty((0, p)),
        "perm_per_fold":     np.stack(perm_per_fold) if perm_per_fold else
                              np.empty((0, p)),
        "cv_scores":         np.asarray(cv_scores, dtype=float),
        "train_scores":      np.asarray(train_scores, dtype=float),
        "train_accuracy_scores": np.asarray(train_accuracy_scores, dtype=float),
        "shap_oof":          shap_oof,
        "expected_oof":      expected_oof,
        "expected_per_fold": expected_per_fold,
        "shap_inter_oof":    shap_inter_oof,
        "n_splits":          len(gain_per_fold),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed RF result + cross-seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SeedRFFit:
    """Output of one seed's K-fold importance pipeline."""

    seed:              object
    importances:       pd.DataFrame
    shap_values:       Optional[np.ndarray]
    shap_interactions: Optional[np.ndarray]
    expected_value:    float
    expected_values:   Optional[np.ndarray]
    n_obs:             int
    cv_r2_mean:        float
    cv_r2_std:         float
    cv_n_splits:       int
    train_r2:          float
    feature_order:     list[str]
    n_clipped:         int = 0
    winsorize_lo:      float = float("nan")
    winsorize_hi:      float = float("nan")
    feature_values:    Optional[np.ndarray] = None  # raw X (NaN), aligned to shap_values
    # No-CV model fit on this seed's full data, for re-prediction.
    # Populated when ``save_models=True`` (the default).
    full_model:        Optional[object] = None


@dataclass
class RFResult:
    """Cross-seed aggregated RF importance for one (model, response).

    ``importances`` carries the cross-seed mean + SD + CI for each
    headline scalar (perm, gain, SHAP mean / abs / std).
    ``per_seed`` is the long per-seed DataFrame (one row per
    (seed, feature)). ``seed_fits`` keeps the raw per-seed
    :class:`_SeedRFFit` objects (with SHAP matrices) so :meth:`save`
    can dump per-seed npz artefacts.
    """

    importances:    pd.DataFrame
    per_seed:       pd.DataFrame
    seed_fits:      list[_SeedRFFit]
    n_obs:          int
    cv_r2_mean:     float
    cv_r2_std:      float
    cv_n_splits:    int
    train_r2:       float
    feature_order:  list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Tree-ensemble importance (XGBoost-backed)
# ─────────────────────────────────────────────────────────────────────────────

class RFImportanceAnalyzer(DatasetAnalyzer, FeatureImportance):
    """XGBoost gain + held-out permutation + OOF SHAP, **per seed**.

    Within each seed slice the analyzer runs a single K-fold loop
    (GroupKFold over ``dataset_id`` when possible), then aggregates the
    headline per-feature scalars across seeds with the configured
    aggregator. Per-seed SHAP matrices are kept and written to
    ``details/`` as separate npz files (one per seed) so downstream
    diagnostics can inspect seed stability of feature attribution.

    Missing values are passed through to XGBoost natively (it learns a default
    split direction per node); the analyzer does **no** imputation, so feature
    importances are attributed to the real features (no indicator columns).

    Output written by :meth:`save`:

    * ``summary.csv`` — long ``(ratio, alpha, model, response, feature,
      rf_perm_importance_mean, rf_perm_importance_se,
      rf_perm_importance_ci_low, rf_perm_importance_ci_high,
      rf_impurity_importance_mean, shap_mean_abs, shap_mean, shap_std,
      cv_r2_mean, cv_r2_std, train_r2, n_obs, cv_n_splits, n_seeds)``.
      Old column names (``rf_perm_importance_mean`` etc.) carry the
      *aggregated* cross-seed values so heatmap / topk plot scripts
      keep working.
    * ``summary_per_seed.csv`` — long: one row per (ratio, alpha, model,
      response, feature, seed) with each seed's raw importance scalars.
    * ``summary_ratio_<r>.html`` — feature × (response, model) pivot of
      aggregated ``rf_perm_importance_mean`` (signed; RdBu_r).
    * ``details/<model>_<resp>_ratio_<r>_<alpha>_seed_<s>_shap.npz`` —
      OOF SHAP matrix for one (model, response, ratio, alpha, seed).
    * ``models/<model>_<resp>_ratio_<r>_<alpha>_seed_<s>.ubj`` (+ ``.json``
      sidecar) — by default one full-data XGBoost per (model, response, ratio,
      alpha, seed) for later re-prediction; pass ``save_models=False`` to skip.
    """

    name: ClassVar[str] = "rf"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs", "long_rel")

    def __init__(
        self,
        n_estimators: int = 200,
        n_perm_repeats: int = 5,
        n_splits: int = 5,
        random_state: int = 0,
        max_samples_shap: Optional[int] = None,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        compute_shap_interactions: bool = True,
        compute_permutation: bool = True,
        compute_shap: bool = True,
        min_obs: int = 15,
        aggregator: Optional[SeedAggregator] = None,
        winsorize_quantiles: Optional[tuple[float, float]] = None,
        task: str = "regression",
        save_models: bool = True,
    ) -> None:
        if task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification'; got {task!r}"
            )
        self.task                      = task
        # When True (default), fit one no-CV XGBoost per seed and persist under
        # ``models/`` as ``<tag>_seed_<s>.ubj`` (+ JSON sidecar).
        self.save_models               = bool(save_models)
        self.n_estimators              = int(n_estimators)
        self.n_perm_repeats            = int(n_perm_repeats)
        self.n_splits                  = int(n_splits)
        self.random_state              = int(random_state)
        self.max_samples_shap          = max_samples_shap
        self.learning_rate             = float(learning_rate)
        self.max_depth                 = int(max_depth)
        self.compute_permutation       = bool(compute_permutation)
        self.compute_shap              = bool(compute_shap)
        self.compute_shap_interactions = (
            bool(compute_shap_interactions) and self.compute_shap
        )
        self.min_obs                   = int(min_obs)
        self.aggregator = aggregator if aggregator is not None else MeanSDAggregator()
        # Optional (lo, hi) quantiles for winsorizing the response before each
        # per-seed fit. ``None`` (default) leaves the target untouched — the
        # standard behavior. Set e.g. (0.05, 0.95) for a robust variant that
        # caps heavy-tailed pair-delta outliers (see ModelPairComparisonPipeline).
        if winsorize_quantiles is not None:
            lo_q, hi_q = winsorize_quantiles
            if not (0.0 <= float(lo_q) < float(hi_q) <= 1.0):
                raise ValueError(
                    "winsorize_quantiles must be (lo, hi) with 0 <= lo < hi <= 1; "
                    f"got {winsorize_quantiles!r}"
                )
            winsorize_quantiles = (float(lo_q), float(hi_q))
        self.winsorize_quantiles = winsorize_quantiles

    # ── single-seed fit (the old fit() body, returns _SeedRFFit) ────────────

    def _fit_one_seed(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str,
        seed: object,
    ) -> Optional[_SeedRFFit]:
        keep_cols = list(feature_cols) + [response]
        if "dataset_id" in df.columns:
            keep_cols = keep_cols + ["dataset_id"]
        sub = df[keep_cols].replace([np.inf, -np.inf], np.nan).copy()
        sub = sub.dropna(subset=[response])
        if len(sub) < self.min_obs:
            return None

        # XGBoost handles NaN natively (a learned default split direction per
        # node), so the raw feature matrix is passed through — no imputation and
        # no indicator columns.  Feature order is exactly ``feature_cols``.
        X = sub[feature_cols].values.astype(np.float32)
        y = sub[response].values.astype(float)
        n_clipped = 0
        win_lo = win_hi = float("nan")
        if (self.winsorize_quantiles is not None
                and self.task == "regression" and len(y) > 0):
            # Cap heavy-tailed response outliers within this seed slice (across
            # datasets) before the K-fold. Applied once up front so fit,
            # permutation scoring (y_te), and SHAP all see the same clipped y.
            win_lo, win_hi = (float(v) for v in np.nanquantile(y, self.winsorize_quantiles))
            n_clipped = int(np.count_nonzero((y < win_lo) | (y > win_hi)))
            y = np.clip(y, win_lo, win_hi)
            print(
                f"    {self._log_tag()}| seed={seed} winsorize {self.winsorize_quantiles} "
                f"clipped {n_clipped}/{len(y)} "
                f"({100.0 * n_clipped / max(len(y), 1):.1f}%) "
                f"to [{win_lo:.4g}, {win_hi:.4g}]"
            )
        n = len(y)
        groups = (
            sub["dataset_id"].to_numpy() if "dataset_id" in sub.columns else None
        )

        if groups is not None and len(np.unique(groups)) >= 2:
            n_splits = max(2, min(self.n_splits, len(np.unique(groups))))
            splitter = GroupKFold(n_splits=n_splits)
            split_groups = groups
        else:
            n_splits = max(2, min(self.n_splits, n))
            splitter = KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state,
            )
            split_groups = None

        out = _xgb_cv_importance(
            X, y,
            task=self.task,
            cv_splitter=splitter,
            groups=split_groups,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_perm_repeats=self.n_perm_repeats,
            random_state=self.random_state,
            max_samples_shap=self.max_samples_shap,
            compute_shap_interactions=self.compute_shap_interactions,
            compute_permutation=self.compute_permutation,
            compute_shap=self.compute_shap,
        )
        if out["n_splits"] == 0:
            return None

        gain_arr     = out["gain_per_fold"]
        perm_arr     = out["perm_per_fold"]
        cv_r2s       = out["cv_scores"]
        train_r2s    = out["train_scores"]
        shap_oof     = out["shap_oof"]
        expected_oof = out["expected_oof"]

        imp_data: dict = {
            "feature":                     list(feature_cols),
            "rf_impurity_importance_mean": np.nanmean(gain_arr, axis=0),
            "rf_impurity_importance_std":  np.nanstd(gain_arr, axis=0),
            "rf_perm_importance_mean":     (
                np.nanmean(perm_arr, axis=0)
                if self.compute_permutation
                else np.full(len(feature_cols), np.nan)
            ),
            "rf_perm_importance_std":      (
                np.nanstd(perm_arr, axis=0)
                if self.compute_permutation
                else np.full(len(feature_cols), np.nan)
            ),
        }

        row_has_shap = ~np.all(np.isnan(shap_oof), axis=1)
        expected_value = float("nan")
        if row_has_shap.any():
            imp_data["shap_mean_abs"] = shap_mean_abs_vector(shap_oof)
            imp_data["shap_mean"]     = np.nanmean(shap_oof, axis=0)
            imp_data["shap_std"]      = np.nanstd(shap_oof, axis=0)
            # Direction: Spearman corr between each feature's value and its SHAP
            # value across rows. >0 => higher feature value pushes the prediction
            # toward model_b winning (model-conditional effect direction).
            imp_data["shap_dir"] = shap_dir_vector(X, shap_oof)
            if out["expected_per_fold"]:
                expected_value = float(np.mean(out["expected_per_fold"]))
        else:
            imp_data["shap_mean_abs"] = np.full(len(feature_cols), np.nan)
            imp_data["shap_mean"]     = np.full(len(feature_cols), np.nan)
            imp_data["shap_std"]      = np.full(len(feature_cols), np.nan)
            imp_data["shap_dir"]      = np.full(len(feature_cols), np.nan)

        importances = pd.DataFrame(imp_data)

        # Optional no-CV model on this seed's full data, for re-prediction.
        # Reuses the (X, y) already prepared above — no duplicated data-prep, and
        # the estimator construction is shared with the CV loop via _make_xgb.
        full_model = None
        if self.save_models and not (
            self.task == "classification" and len(np.unique(y)) < 2
        ):
            full_model = _make_xgb(
                self.task, self.n_estimators, self.learning_rate,
                self.max_depth, self.random_state,
            ).fit(X, y)

        return _SeedRFFit(
            seed=seed,
            importances=importances,
            shap_values=shap_oof if self.compute_shap else None,
            shap_interactions=out["shap_inter_oof"],
            expected_value=expected_value,
            expected_values=expected_oof,
            n_obs=int(n),
            cv_r2_mean=float(np.nanmean(cv_r2s)),
            cv_r2_std=float(np.nanstd(cv_r2s)),
            cv_n_splits=int(out["n_splits"]),
            train_r2=float(np.nanmean(train_r2s)),
            feature_order=list(feature_cols),
            n_clipped=int(n_clipped),
            winsorize_lo=float(win_lo),
            winsorize_hi=float(win_hi),
            feature_values=X,
            full_model=full_model,
        )

    # ── per-seed aggregation ────────────────────────────────────────────────

    _SCALAR_COLS: tuple[str, ...] = (
        "rf_impurity_importance_mean",
        "rf_perm_importance_mean",
        "shap_mean_abs",
        "shap_mean",
        "shap_std",
        "shap_dir",
    )

    def _aggregate(
        self, per_seed: pd.DataFrame, feature_order: list[str],
    ) -> pd.DataFrame:
        """Aggregate per-seed per-feature scalars across seeds.

        For each scalar in ``_SCALAR_COLS`` we keep the headline column
        name pointing at the cross-seed mean (so plots don't change) and
        add ``<col>_se / _ci_low / _ci_high`` for the headline
        ``rf_perm_importance_mean``.
        """
        rows: list[dict] = []
        for feature in feature_order:
            grp = per_seed[per_seed["feature"] == feature]
            row: dict = {"feature": feature}
            n_seeds_seen = 0
            for col in self._SCALAR_COLS:
                vals = grp[col].dropna().tolist() if col in grp.columns else []
                per_list = [{"estimate": float(v)} for v in vals if np.isfinite(v)]
                agg = self.aggregator.combine(per_list)
                row[col] = agg["mean"]
                # Persist SE/CI/between_var for *every* scalar so downstream
                # plots can render uniform error bars across importance flavors
                # (perm / impurity gain / shap_mean_abs / shap_mean / shap_std).
                # For the two ``*_importance_mean`` columns we strip the
                # ``_mean`` suffix before attaching so the back-compat name
                # ``rf_perm_importance_se`` (already consumed by
                # importance_long / plot_forest) is preserved.
                base = col[:-len("_mean")] if col.endswith("_importance_mean") else col
                row[f"{base}_se"]          = agg["se"]
                row[f"{base}_ci_low"]      = agg["ci_low"]
                row[f"{base}_ci_high"]     = agg["ci_high"]
                row[f"{base}_between_var"] = agg["between_var"]
                n_seeds_seen = max(n_seeds_seen, agg["n_seeds"])
            row["n_seeds"] = int(n_seeds_seen)
            rows.append(row)
        return (
            pd.DataFrame(rows)
            .sort_values("rf_perm_importance_mean", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    def fit_per_model_response(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], RFResult]:
        out: dict[tuple[str, str], RFResult] = {}
        if "seed" in df.columns:
            seeds = sorted(s for s in df["seed"].dropna().unique())
        else:
            seeds = [None]

        for model in models:
            sub_m = df[df["model"] == model]
            for resp in responses:
                print(f"  {self._log_tag()} | model={model:10s} response={resp}")
                seed_fits: list[_SeedRFFit] = []
                for seed in seeds:
                    if seed is None:
                        sub_s = sub_m
                    else:
                        sub_s = sub_m[sub_m["seed"] == seed]
                    fit_k = self._fit_one_seed(sub_s, feature_cols, resp, seed)
                    if fit_k is None:
                        continue
                    seed_fits.append(fit_k)
                if not seed_fits:
                    print("    skipped (insufficient data)")
                    continue

                # Use the first seed's feature order; all seeds share the same
                # feature set (it is exactly ``feature_cols``).
                feature_order = seed_fits[0].feature_order

                # Build per-seed long DataFrame.
                per_seed_rows: list[dict] = []
                for fk in seed_fits:
                    for _, c in fk.importances.iterrows():
                        per_seed_rows.append({
                            "seed":                        fk.seed,
                            "feature":                     c["feature"],
                            "rf_impurity_importance_mean": float(c.get("rf_impurity_importance_mean", float("nan"))),
                            "rf_impurity_importance_std":  float(c.get("rf_impurity_importance_std", float("nan"))),
                            "rf_perm_importance_mean":     float(c.get("rf_perm_importance_mean", float("nan"))),
                            "rf_perm_importance_std":      float(c.get("rf_perm_importance_std", float("nan"))),
                            "shap_mean_abs":               float(c.get("shap_mean_abs", float("nan"))),
                            "shap_mean":                   float(c.get("shap_mean", float("nan"))),
                            "shap_std":                    float(c.get("shap_std", float("nan"))),
                            "shap_dir":                    float(c.get("shap_dir", float("nan"))),
                            "cv_r2_mean":                  fk.cv_r2_mean,
                            "cv_r2_std":                   fk.cv_r2_std,
                            "train_r2":                    fk.train_r2,
                            "n_obs":                       int(fk.n_obs),
                            "cv_n_splits":                 int(fk.cv_n_splits),
                            "n_clipped":                   int(fk.n_clipped),
                            "frac_clipped":                (fk.n_clipped / fk.n_obs) if fk.n_obs else float("nan"),
                            "winsorize_lo":                float(fk.winsorize_lo),
                            "winsorize_hi":                float(fk.winsorize_hi),
                        })
                per_seed = pd.DataFrame(per_seed_rows)
                aggregated = self._aggregate(per_seed, feature_order)
                if aggregated.empty:
                    print("    skipped (no feature after aggregation)")
                    continue

                out[(model, resp)] = RFResult(
                    importances=aggregated,
                    per_seed=per_seed,
                    seed_fits=seed_fits,
                    n_obs=int(max(fk.n_obs for fk in seed_fits)),
                    cv_r2_mean=float(np.mean([fk.cv_r2_mean for fk in seed_fits])),
                    cv_r2_std=float(np.std([fk.cv_r2_mean for fk in seed_fits], ddof=0)),
                    cv_n_splits=int(seed_fits[0].cv_n_splits),
                    train_r2=float(np.mean([fk.train_r2 for fk in seed_fits])),
                    feature_order=feature_order,
                )
        return out

    def _log_tag(self) -> str:
        return "RF  "

    # ── DatasetAnalyzer interface ────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], RFResult]:
        return self.fit_per_model_response(df, feature_cols, models, responses)

    def importance_long(
        self,
        results: dict[tuple[str, str], RFResult],
    ) -> pd.DataFrame:
        """Long-format rows with cross-seed *aggregated* importance scalars.

        Carries mean + SE/CI/between_var for every scalar so downstream plots
        can render uniform error bars across all importance flavors.
        """
        # Mean column → CI base name (strip ``_mean`` for impurity / perm; SHAP
        # cols keep their full name as the base). Mirrors the convention used
        # by ``_aggregate``.
        ci_base: dict[str, str] = {
            col: (col[:-len("_mean")] if col.endswith("_importance_mean") else col)
            for col in self._SCALAR_COLS
        }

        rows: list[dict] = []
        for (model, resp), result in results.items():
            # Winsorize bookkeeping is per-seed-fit (same across features); collapse
            # to per-seed then sum/average so summary.csv shows how much was clipped.
            ps = result.per_seed
            if "n_clipped" in ps.columns and "seed" in ps.columns:
                ps_seed = ps.drop_duplicates(subset=["seed"])
                n_clipped_total = int(ps_seed["n_clipped"].sum())
                frac_clipped_mean = float(ps_seed["frac_clipped"].mean())
            else:
                n_clipped_total, frac_clipped_mean = 0, float("nan")
            for _, c in result.importances.iterrows():
                row: dict = {
                    "model":                       model,
                    "response":                    resp,
                    "feature":                     c["feature"],
                    "cv_r2_mean":                  float(result.cv_r2_mean),
                    "cv_r2_std":                   float(result.cv_r2_std),
                    "train_r2":                    float(result.train_r2),
                    "n_obs":                       int(result.n_obs),
                    "cv_n_splits":                 int(result.cv_n_splits),
                    "n_seeds":                     int(c.get("n_seeds", 0)),
                    "n_clipped_total":             n_clipped_total,
                    "frac_clipped_mean":           frac_clipped_mean,
                }
                # Mean + SE/CI for every aggregated scalar.
                for col, base in ci_base.items():
                    row[col]                  = float(c.get(col, float("nan")))
                    row[f"{base}_se"]          = float(c.get(f"{base}_se", float("nan")))
                    row[f"{base}_ci_low"]      = float(c.get(f"{base}_ci_low", float("nan")))
                    row[f"{base}_ci_high"]     = float(c.get(f"{base}_ci_high", float("nan")))
                    row[f"{base}_between_var"] = float(c.get(f"{base}_between_var", float("nan")))
                rows.append(row)
        return pd.DataFrame(rows)

    def per_seed_long(
        self,
        results: dict[tuple[str, str], RFResult],
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for (model, resp), result in results.items():
            for _, c in result.per_seed.iterrows():
                rows.append({
                    "model":                       model,
                    "response":                    resp,
                    "feature":                     c["feature"],
                    "seed":                        c["seed"],
                    "rf_impurity_importance_mean": float(c["rf_impurity_importance_mean"]),
                    "rf_impurity_importance_std":  float(c["rf_impurity_importance_std"]),
                    "rf_perm_importance_mean":     float(c["rf_perm_importance_mean"]),
                    "rf_perm_importance_std":      float(c["rf_perm_importance_std"]),
                    "shap_mean_abs":               float(c["shap_mean_abs"]),
                    "shap_mean":                   float(c["shap_mean"]),
                    "shap_std":                    float(c["shap_std"]),
                    "shap_dir":                    float(c.get("shap_dir", float("nan"))),
                    "cv_r2_mean":                  float(c["cv_r2_mean"]),
                    "cv_r2_std":                   float(c["cv_r2_std"]),
                    "train_r2":                    float(c["train_r2"]),
                    "n_obs":                       int(c["n_obs"]),
                    "cv_n_splits":                 int(c["cv_n_splits"]),
                    "n_clipped":                   int(c.get("n_clipped", 0)),
                    "frac_clipped":                float(c.get("frac_clipped", float("nan"))),
                    "winsorize_lo":                float(c.get("winsorize_lo", float("nan"))),
                    "winsorize_hi":                float(c.get("winsorize_hi", float("nan"))),
                })
        return pd.DataFrame(rows)

    def save(
        self,
        results_by_key: dict[
            tuple[float, "str | float | None"],
            dict[tuple[str, str], RFResult],
        ],
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        grouped = group_keys_by_ratio(results_by_key)

        # 1. Long aggregated + per-seed CSVs.
        long_frames: list[pd.DataFrame] = []
        per_seed_frames: list[pd.DataFrame] = []
        for ratio, pairs in grouped.items():
            for alpha_label, results in pairs:
                df_agg = self.importance_long(results)
                if not df_agg.empty:
                    df_agg = df_agg.copy()
                    df_agg.insert(0, "ratio", float(ratio))
                    df_agg.insert(1, "alpha", _alpha_to_cell(alpha_label))
                    long_frames.append(df_agg)
                df_ps = self.per_seed_long(results)
                if not df_ps.empty:
                    df_ps = df_ps.copy()
                    df_ps.insert(0, "ratio", float(ratio))
                    df_ps.insert(1, "alpha", _alpha_to_cell(alpha_label))
                    per_seed_frames.append(df_ps)
        if long_frames:
            summary_df = pd.concat(long_frames, ignore_index=True)
            summary_df.to_csv(out_dir / "summary.csv", index=False)
            # Compact fit-quality view: can the meta-features predict each metric?
            # (CV R² / train R² per response × model, grouped by category/scale.)
            fq = FitQualityTable().build(summary_df)
            if not fq.empty:
                fq.to_csv(out_dir / "fit_quality.csv")
                render_fit_quality_html(
                    fq, out_dir / "fit_quality.html",
                    caption=f"RF fit quality ({out_dir.name})",
                )
        if per_seed_frames:
            (
                pd.concat(per_seed_frames, ignore_index=True)
                .to_csv(out_dir / "summary_per_seed.csv", index=False)
            )

        # 2. Per-ratio HTML pivot of permutation importance.
        for ratio, pairs in grouped.items():
            sub_frames: list[pd.DataFrame] = []
            for alpha_label, results in pairs:
                df = self.importance_long(results)
                if df.empty:
                    continue
                df = df.copy()
                df["alpha_label"] = (
                    "alpha_free" if (alpha_label is None
                                     or alpha_label == "alpha_free")
                    else f"alpha={float(alpha_label)}"
                )
                sub_frames.append(df)
            if not sub_frames:
                continue
            combined = pd.concat(sub_frames, ignore_index=True)
            render_feature_pivot_html(
                combined,
                out_dir / f"summary_ratio_{ratio}.html",
                value_cols={
                    "perm":      "rf_perm_importance_mean",
                    "shap_abs":  "shap_mean_abs",
                    "shap_mean": "shap_mean",
                },
                diverging_by_label={
                    "perm":      True,    # signed: R² drop
                    "shap_abs":  False,   # non-negative magnitude
                    "shap_mean": True,    # signed direction
                },
                caption=(
                    f"RF importance (cross-seed mean) "
                    f"(ratio={ratio}; sub-columns: "
                    f"`perm` = held-out permutation importance "
                    f"(score_before − score_after, signed); "
                    f"`shap_abs` = mean |SHAP|, non-negative magnitude; "
                    f"`shap_mean` = mean signed SHAP, "
                    f"positive = feature pushes response up; "
                    f"rows = meta-feature, "
                    f"cols = alpha × response × importance × model)"
                ),
                alpha_col=alpha_col_for_html(pairs),
            )

        # 3. Bulky per-seed OOF SHAP arrays under details/.
        any_shap = any(
            fk.shap_values is not None
            for pairs in grouped.values()
            for _, results in pairs
            for r in results.values()
            for fk in r.seed_fits
        )
        if any_shap:
            details_dir = out_dir / "details"
            details_dir.mkdir(exist_ok=True)
            for ratio, pairs in grouped.items():
                for alpha_label, results in pairs:
                    alpha_tag = (
                        "alpha_free"
                        if alpha_label is None or alpha_label == "alpha_free"
                        else f"alpha_{float(alpha_label)}"
                    )
                    for (model, resp), result in results.items():
                        for fk in result.seed_fits:
                            if fk.shap_values is None:
                                continue
                            seed_tag = (
                                "seed_NA" if fk.seed is None
                                else f"seed_{fk.seed}"
                            )
                            tag = (
                                f"{model}_{resp}_ratio_{ratio}_"
                                f"{alpha_tag}_{seed_tag}_shap"
                            )
                            save_kwargs: dict = {
                                "shap_values":   fk.shap_values,
                                "expected_value": np.array(
                                    [fk.expected_value], dtype=np.float32,
                                ),
                                "features":      np.asarray(fk.feature_order),
                            }
                            if fk.feature_values is not None:
                                # Raw feature matrix (NaN-preserving) aligned to
                                # shap_values (for beeswarm / dependence plots).
                                save_kwargs["feature_values"] = np.asarray(
                                    fk.feature_values, dtype=np.float32,
                                )
                            if fk.expected_values is not None:
                                save_kwargs["expected_values"] = fk.expected_values
                            if fk.shap_interactions is not None:
                                save_kwargs["shap_interactions"] = fk.shap_interactions
                            np.savez_compressed(details_dir / f"{tag}.npz", **save_kwargs)

        # 4. Per-seed full-data models (skipped when ``save_models=False``).
        if self.save_models:
            self._save_full_models(grouped, out_dir)

    def _save_full_models(self, grouped: dict, out_dir: Path) -> None:
        """Persist each seed's ``_SeedRFFit.full_model`` as ``models/<tag>.ubj``
        plus a ``<tag>.json`` sidecar (task, feature order, hyper-parameters) so a
        saved model can be reloaded to predict the metric on new datasets.

        One model per (model, response, ratio, alpha, **seed**), mirroring the
        per-seed SHAP ``details/`` naming.  Reload::

            import json, xgboost as xgb
            meta = json.load(open("<tag>.json"))
            m = xgb.XGBRegressor(); m.load_model("<tag>.ubj")
            X = new_df[meta["feature_order"]].to_numpy("float32")  # NaN ok
            y_hat = m.predict(X)
        """
        models_dir = out_dir / "models"
        models_dir.mkdir(exist_ok=True)
        hyperparams = {
            "n_estimators":  self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth":     self.max_depth,
            "tree_method":   "hist",
            "random_state":  self.random_state,
            "compute_permutation": self.compute_permutation,
            "compute_shap":  self.compute_shap,
        }
        n_written = 0
        for ratio, pairs in grouped.items():
            for alpha_label, results in pairs:
                is_alpha_free = alpha_label is None or alpha_label == "alpha_free"
                alpha_tag = "alpha_free" if is_alpha_free else f"alpha_{float(alpha_label)}"
                alpha_value = None if is_alpha_free else float(alpha_label)
                for (model, resp), result in results.items():
                    for fk in result.seed_fits:
                        if fk.full_model is None:
                            continue
                        seed_tag = "seed_NA" if fk.seed is None else f"seed_{fk.seed}"
                        tag = f"{model}_{resp}_ratio_{ratio}_{alpha_tag}_{seed_tag}"
                        fk.full_model.save_model(str(models_dir / f"{tag}.ubj"))
                        sidecar = {
                            "analyzer":        self.name,
                            "task":            self.task,
                            "model":           model,
                            "response":        resp,
                            "ratio":           float(ratio),
                            "alpha":           alpha_value,
                            "seed":            (None if fk.seed is None else int(fk.seed)),
                            "feature_order":   list(fk.feature_order),
                            "n_obs":           int(fk.n_obs),
                            "nan_handling":    "native",
                            "hyperparams":     hyperparams,
                            "xgboost_version": xgb.__version__,
                        }
                        (models_dir / f"{tag}.json").write_text(
                            json.dumps(sidecar, indent=2), encoding="utf-8",
                        )
                        n_written += 1
        if n_written:
            print(
                f"  {self._log_tag()}| saved {n_written} full-data model(s) "
                f"-> {models_dir}"
            )
