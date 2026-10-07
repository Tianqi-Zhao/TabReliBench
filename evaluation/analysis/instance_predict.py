"""Instance-level analyzers.

Counterpart to :mod:`evaluation.analysis.multivariate` and
:mod:`evaluation.analysis.univariate`: both packages expose stateless
``fit_per_model_*`` analyzers that take a long DataFrame and return a result
object per ``(model, response)``. The instance-level versions differ only in
that the held-out CV fold is a **new dataset** (``GroupKFold`` by
``dataset_id``), so the reported scores measure *cross-dataset*
generalisation rather than within-dataset interpolation.

Three analyzers:

* :class:`InstanceCoverageAnalyzer` — XGBoost on the
  binary ``covered`` outcome. Headline = cross-dataset ROC-AUC.

* :class:`InstanceWidthAnalyzer` — XGBoost on a continuous outcome
  (e.g. ``pi_width_norm``). Headline = cross-dataset R².

* :class:`InstanceChatterjeeAnalyzer` — :func:`scipy.stats.chatterjeexi`
  on the (sub-sampled) instance-level long table. Auto-flips
  ``y_continuous`` to ``False`` for binary responses such as ``covered``.

All instance analyzers honour a ``max_per_group`` cap on rows per
(dataset, seed, ratio) so huge datasets don't dominate the loss, and
share the helpers :func:`_subsample_per_group` /
:func:`_prepare_design`.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .multivariate.rf import _xgb_cv_importance
from .univariate.chatterjee import ChatterjeeAnalyzer, ChatterjeeResult

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subsample_per_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    max_per_group: int,
    random_state: int,
) -> pd.DataFrame:
    """Cap each (group_cols) tuple at ``max_per_group`` rows (random sample)."""
    if max_per_group is None or max_per_group <= 0:
        return df
    parts: list[pd.DataFrame] = []
    for _, grp in df.groupby(list(group_cols), sort=False):
        if len(grp) > max_per_group:
            grp = grp.sample(n=max_per_group, random_state=random_state)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def _prepare_design(
    df: pd.DataFrame,
    feature_cols: list[str],
    response: str,
    group_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Drop NaN/inf rows, drop constant columns, return ``(X, y, groups, kept)``."""
    cols = list(feature_cols) + [response, group_col]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return (
            np.empty((0, 0)), np.empty(0), np.empty(0, dtype=object), []
        )

    feats = sub[feature_cols].values.astype(float)
    std0 = feats.std(axis=0)
    keep = [f for f, s in zip(feature_cols, std0 > 1e-10) if s]
    X = sub[keep].values.astype(float)
    y = sub[response].values
    groups = sub[group_col].values
    return X, y, groups, keep


# ─────────────────────────────────────────────────────────────────────────────
# Coverage classifier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceCoverageResult:
    """Self-contained output of one coverage classification fit.

    ``train_acc`` and ``train_auc`` are both in-sample diagnostics
    (the classifier is evaluated on the very rows it was trained on) and
    are kept only as sanity checks against ``cv_auc_mean``. Because the
    coverage problem is highly imbalanced (base rate ≈ 1−α), accuracy is
    barely informative — the AUC is the meaningful one.
    """

    importances:    pd.DataFrame   # feature, lr_coef_abs, optional shap_*, optional chatterjee_xi (pipeline)
    n_obs:          int
    n_groups:       int
    base_rate:      float          # P(covered) on the (sub-sampled) data
    cv_auc_mean:    float
    cv_auc_std:     float
    cv_aucs:        np.ndarray     # per-fold AUCs
    cv_n_splits:    int
    train_acc:      float          # fold-mean in-sample XGB accuracy
    train_auc:      float          # fold-mean in-sample XGB ROC-AUC
    kept_features:  list[str]
    dropped_features: list[str]
    fit:            Optional[object] = None
    shap_values:    Optional[np.ndarray] = None
    shap_interactions: Optional[np.ndarray] = None
    expected_value: float = float("nan")
    expected_values: Optional[np.ndarray] = None
    task:           str = "classification"

    @property
    def cv_score_mean(self) -> float:
        return self.cv_auc_mean

    @property
    def cv_score_std(self) -> float:
        return self.cv_auc_std

    @property
    def train_score(self) -> float:
        return self.train_auc

    @property
    def feature_order(self) -> list[str]:
        """Design-matrix / importance row order (alias of ``kept_features``; cf. ``RFResult.feature_order``)."""
        return self.kept_features


class InstanceCoverageAnalyzer:
    """Per-test-point coverage classifier.

    GroupKFold is by ``dataset_id`` so every held-out fold is a *new*
    dataset — the AUC therefore measures how features explain coverage
    transfer across datasets. ``max_per_group`` caps rows per
    (dataset_id, seed, ratio) to balance the loss across datasets.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_perm_repeats: int = 5,
        max_depth: Optional[int] = 4,
        max_per_group: int = 1000,
        n_splits: int = 5,
        random_state: int = 0,
        group_col: str = "dataset_id",
        learning_rate: float = 0.05,
        max_samples_shap: Optional[int] = None,
        compute_shap_interactions: bool = False,
        compute_shap: bool = False,
    ) -> None:
        self.n_estimators              = int(n_estimators)
        self.n_perm_repeats            = int(n_perm_repeats)
        self.max_depth = int(max_depth) if max_depth is not None else 4
        self.max_per_group             = int(max_per_group)
        self.n_splits                  = int(n_splits)
        self.random_state              = int(random_state)
        self.group_col                 = str(group_col)
        self.learning_rate             = float(learning_rate)
        self.max_samples_shap          = max_samples_shap
        self.compute_shap_interactions = bool(compute_shap_interactions)
        self.compute_shap              = bool(compute_shap)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str = "covered",
    ) -> Optional[InstanceCoverageResult]:
        # Sub-sample per (dataset, seed, ratio) so that very large datasets
        # don't dominate the loss function.
        sub_keys = [c for c in (self.group_col, "seed", "ratio") if c in df.columns]
        sub = _subsample_per_group(
            df, sub_keys, self.max_per_group, self.random_state,
        )

        X, y, groups, kept = _prepare_design(
            sub, feature_cols, response, self.group_col,
        )
        if X.shape[0] < 50 or len(np.unique(groups)) < 3:
            return None
        y_int = y.astype(int)
        if len(np.unique(y_int)) < 2:
            return None

        scaler = StandardScaler().fit(X)
        Xz = scaler.transform(X)

        # ── Logistic regression coefficients (standardised features) ──────
        lr = LogisticRegression(C=1.0, max_iter=3000,
                                random_state=self.random_state)
        lr.fit(Xz, y_int)
        lr_coef_abs = np.abs(lr.coef_[0])

        # ── XGBoost CV score + native OOF SHAP ───────────────────────────
        n_unique = int(np.unique(groups).size)
        n_splits = max(2, min(self.n_splits, n_unique))
        gkf = GroupKFold(n_splits=n_splits)
        out = _xgb_cv_importance(
            X.astype(np.float32), y_int,
            task="classification",
            cv_splitter=gkf,
            groups=groups,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_perm_repeats=self.n_perm_repeats,
            random_state=self.random_state,
            max_samples_shap=self.max_samples_shap,
            compute_shap_interactions=self.compute_shap_interactions,
            compute_permutation=False,
            compute_shap=self.compute_shap,
        )
        if out["n_splits"] == 0:
            return None

        aucs         = out["cv_scores"]
        train_aucs   = out["train_scores"]
        train_accs   = out.get("train_accuracy_scores", np.array([np.nan]))
        shap_oof     = out["shap_oof"]
        expected_oof = out["expected_oof"]
        train_acc = float(np.nanmean(train_accs))
        train_auc = float(np.nanmean(train_aucs))

        imp_data: dict = {
            "feature":     list(kept),
            "lr_coef_abs": lr_coef_abs,
        }

        row_has_shap = ~np.all(np.isnan(shap_oof), axis=1)
        expected_value = float("nan")
        if row_has_shap.any():
            imp_data["shap_mean_abs"] = np.nanmean(np.abs(shap_oof), axis=0)
            imp_data["shap_mean"]     = np.nanmean(shap_oof, axis=0)
            imp_data["shap_std"]      = np.nanstd(shap_oof, axis=0)
            if out["expected_per_fold"]:
                expected_value = float(np.mean(out["expected_per_fold"]))
        else:
            imp_data["shap_mean_abs"] = np.full(len(kept), np.nan)

        importances = pd.DataFrame(imp_data).reset_index(drop=True)

        return InstanceCoverageResult(
            importances=importances,
            n_obs=int(X.shape[0]),
            n_groups=n_unique,
            base_rate=float(np.mean(y_int)),
            cv_auc_mean=float(np.nanmean(aucs)),
            cv_auc_std=float(np.nanstd(aucs)),
            cv_aucs=np.asarray(aucs, dtype=float),
            cv_n_splits=int(n_splits),
            train_acc=train_acc,
            train_auc=train_auc,
            kept_features=kept,
            dropped_features=[f for f in feature_cols if f not in kept],
            fit=out["last_fit"],
            shap_values=shap_oof,
            shap_interactions=out["shap_inter_oof"],
            expected_value=expected_value,
            expected_values=expected_oof,
        )

    def fit_per_model(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        response: str = "covered",
    ) -> dict[str, InstanceCoverageResult]:
        out: dict[str, InstanceCoverageResult] = {}
        for model in models:
            sub = df[df["model"] == model]
            log.info(
                "  COV  | model=%-10s response=%-8s n_rows=%d",
                model, response, len(sub),
            )
            if sub.empty:
                continue
            result = self.fit(sub, feature_cols, response)
            if result is None:
                log.info("    skipped (insufficient rows / classes / groups)")
                continue
            out[model] = result
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Continuous-outcome regressor (e.g. pi_width_norm; Winkler analysis optional)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceRegressionResult:
    """Self-contained output of one continuous-outcome regression fit."""

    importances:    pd.DataFrame   # feature, ridge_coef_abs, optional shap_*, optional chatterjee_xi (pipeline)
    n_obs:          int
    n_groups:       int
    cv_r2_mean:     float
    cv_r2_std:      float
    cv_r2:          np.ndarray
    cv_n_splits:    int
    train_r2:       float
    kept_features:  list[str]
    dropped_features: list[str]
    fit:            Optional[object] = None
    shap_values:    Optional[np.ndarray] = None
    shap_interactions: Optional[np.ndarray] = None
    expected_value: float = float("nan")
    expected_values: Optional[np.ndarray] = None
    task:           str = "regression"
    base_rate:      Optional[float] = None

    @property
    def cv_score_mean(self) -> float:
        return self.cv_r2_mean

    @property
    def cv_score_std(self) -> float:
        return self.cv_r2_std

    @property
    def train_score(self) -> float:
        return self.train_r2

    @property
    def feature_order(self) -> list[str]:
        """Design-matrix / importance row order (alias of ``kept_features``; cf. ``RFResult.feature_order``)."""
        return self.kept_features


class InstanceWidthAnalyzer:
    """Continuous-outcome counterpart to :class:`InstanceCoverageAnalyzer`.

    The response column is chosen by the caller (e.g. ``pi_width_norm`` for
    normalised interval width).  ``winkler`` can be used if enabled in the
    pipeline defaults.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_perm_repeats: int = 5,
        max_depth: Optional[int] = 4,
        max_per_group: int = 1000,
        n_splits: int = 5,
        random_state: int = 0,
        group_col: str = "dataset_id",
        learning_rate: float = 0.05,
        max_samples_shap: Optional[int] = None,
        compute_shap_interactions: bool = False,
        compute_shap: bool = False,
    ) -> None:
        self.n_estimators              = int(n_estimators)
        self.n_perm_repeats            = int(n_perm_repeats)
        self.max_depth = int(max_depth) if max_depth is not None else 4
        self.max_per_group             = int(max_per_group)
        self.n_splits                  = int(n_splits)
        self.random_state              = int(random_state)
        self.group_col                 = str(group_col)
        self.learning_rate             = float(learning_rate)
        self.max_samples_shap          = max_samples_shap
        self.compute_shap_interactions = bool(compute_shap_interactions)
        self.compute_shap              = bool(compute_shap)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str,
    ) -> Optional[InstanceRegressionResult]:
        if response not in df.columns:
            log.warning(
                "response %r missing from instance DataFrame; skipping regression.",
                response,
            )
            return None

        sub_keys = [c for c in (self.group_col, "seed", "ratio") if c in df.columns]
        sub = _subsample_per_group(
            df, sub_keys, self.max_per_group, self.random_state,
        )

        X, y, groups, kept = _prepare_design(
            sub, feature_cols, response, self.group_col,
        )
        if X.shape[0] < 50 or len(np.unique(groups)) < 3:
            return None

        scaler = StandardScaler().fit(X)
        Xz = scaler.transform(X)
        y  = y.astype(float)

        # ── Standardised Ridge coefficients ───────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge = Ridge(alpha=1.0, random_state=self.random_state)
            ridge.fit(Xz, y)
        ridge_coef_abs = np.abs(ridge.coef_)

        # ── XGBoost CV score + native OOF SHAP ───────────────────────────
        n_unique = int(np.unique(groups).size)
        n_splits = max(2, min(self.n_splits, n_unique))
        gkf = GroupKFold(n_splits=n_splits)
        out = _xgb_cv_importance(
            X.astype(np.float32), y,
            task="regression",
            cv_splitter=gkf,
            groups=groups,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_perm_repeats=self.n_perm_repeats,
            random_state=self.random_state,
            max_samples_shap=self.max_samples_shap,
            compute_shap_interactions=self.compute_shap_interactions,
            compute_permutation=False,
            compute_shap=self.compute_shap,
        )
        if out["n_splits"] == 0:
            return None

        r2s          = out["cv_scores"]
        train_r2s    = out["train_scores"]
        shap_oof     = out["shap_oof"]
        expected_oof = out["expected_oof"]

        imp_data: dict = {
            "feature":        list(kept),
            "ridge_coef_abs": ridge_coef_abs,
        }

        row_has_shap = ~np.all(np.isnan(shap_oof), axis=1)
        expected_value = float("nan")
        if row_has_shap.any():
            imp_data["shap_mean_abs"] = np.nanmean(np.abs(shap_oof), axis=0)
            imp_data["shap_mean"]     = np.nanmean(shap_oof, axis=0)
            imp_data["shap_std"]      = np.nanstd(shap_oof, axis=0)
            if out["expected_per_fold"]:
                expected_value = float(np.mean(out["expected_per_fold"]))
        else:
            imp_data["shap_mean_abs"] = np.full(len(kept), np.nan)

        importances = pd.DataFrame(imp_data).reset_index(drop=True)

        return InstanceRegressionResult(
            importances=importances,
            n_obs=int(X.shape[0]),
            n_groups=n_unique,
            cv_r2_mean=float(np.nanmean(r2s)),
            cv_r2_std=float(np.nanstd(r2s)),
            cv_r2=np.asarray(r2s, dtype=float),
            cv_n_splits=int(n_splits),
            train_r2=float(np.nanmean(train_r2s)),
            kept_features=kept,
            dropped_features=[f for f in feature_cols if f not in kept],
            fit=out["last_fit"],
            shap_values=shap_oof,
            shap_interactions=out["shap_inter_oof"],
            expected_value=expected_value,
            expected_values=expected_oof,
        )

    def fit_per_model(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        response: str,
    ) -> dict[str, InstanceRegressionResult]:
        out: dict[str, InstanceRegressionResult] = {}
        for model in models:
            sub = df[df["model"] == model]
            log.info(
                "  REG  | model=%-10s response=%-14s n_rows=%d",
                model, response, len(sub),
            )
            if sub.empty:
                continue
            result = self.fit(sub, feature_cols, response)
            if result is None:
                log.info("    skipped (insufficient rows / groups)")
                continue
            out[model] = result
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Instance-level Chatterjee ξ
# ─────────────────────────────────────────────────────────────────────────────

class InstanceChatterjeeAnalyzer:
    """Thin wrapper around :class:`ChatterjeeAnalyzer` for instance data.

    Differences from the dataset-level analyzer:

    * Sub-samples each ``(dataset, seed, ratio)`` to at most
      ``max_per_group`` rows so a few huge datasets don't dominate ξ.
    * Auto-detects whether the response is binary (e.g. ``covered``) and
      flips ``y_continuous=False`` accordingly. Continuous responses
      (e.g. ``pi_width_norm``; ``winkler`` optional) keep the asymptotic
      distribution code path.
    """

    def __init__(
        self,
        method: str = "asymptotic",
        min_obs: int = 50,
        tie_jitter: float = 1e-10,
        random_state: int = 0,
        max_per_group: int = 1000,
        group_col: str = "dataset_id",
    ) -> None:
        self.method        = str(method)
        self.min_obs       = int(min_obs)
        self.tie_jitter    = float(tie_jitter)
        self.random_state  = int(random_state)
        self.max_per_group = int(max_per_group)
        self.group_col     = str(group_col)

    def _build_analyzer(self, y_continuous: bool) -> ChatterjeeAnalyzer:
        return ChatterjeeAnalyzer(
            y_continuous=y_continuous,
            method=self.method,
            min_obs=self.min_obs,
            tie_jitter=self.tie_jitter,
            random_state=self.random_state,
        )

    @staticmethod
    def _is_binary(y: pd.Series) -> bool:
        vals = pd.unique(y.dropna())
        if len(vals) > 2:
            return False
        try:
            return set(np.asarray(vals, dtype=int)).issubset({0, 1})
        except (ValueError, TypeError):
            return False

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        responses: list[str],
    ) -> Optional[ChatterjeeResult]:
        sub_keys = [c for c in (self.group_col, "seed", "ratio") if c in df.columns]
        sub = _subsample_per_group(
            df, sub_keys, self.max_per_group, self.random_state,
        )
        if sub.empty:
            return None

        # Group responses by y_continuous so we only build two analyzers.
        cont_resps:    list[str] = []
        binary_resps:  list[str] = []
        for r in responses:
            if r not in sub.columns:
                continue
            if self._is_binary(sub[r]):
                binary_resps.append(r)
            else:
                cont_resps.append(r)

        frames: list[pd.DataFrame] = []
        n_obs_seen = 0
        for y_cont, resps in ((True, cont_resps), (False, binary_resps)):
            if not resps:
                continue
            analyzer = self._build_analyzer(y_continuous=y_cont)
            for r in resps:
                res = analyzer.fit(sub, feature_cols, r)
                if res is None:
                    continue
                part = res.correlations.copy()
                part.insert(0, "response", r)
                frames.append(part)
                n_obs_seen = max(n_obs_seen, res.n_obs)

        if not frames:
            return None
        correlations = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["response", "xi"], ascending=[True, False])
            .reset_index(drop=True)
        )
        return ChatterjeeResult(correlations=correlations, n_obs=n_obs_seen)

    def fit_per_model(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[str, ChatterjeeResult]:
        out: dict[str, ChatterjeeResult] = {}
        for model in models:
            sub = df[df["model"] == model]
            log.info("  XI   | model=%-10s responses=%s", model, responses)
            result = self.fit(sub, feature_cols, responses)
            if result is None:
                log.info("    skipped (insufficient data)")
                continue
            out[model] = result
        return out
