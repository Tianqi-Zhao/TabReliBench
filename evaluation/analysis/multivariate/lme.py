"""Linear mixed-effects analyzer.

``response ~ standardized_features + (1 | dataset_id)``.

Implementation notes:

* Uses the **array** interface ``sm.MixedLM(endog, exog_df, groups)`` rather
  than the formula interface so that the resulting ``MixedLMResultsWrapper``
  pickles cleanly (the formula API embeds a patsy reference to the calling
  frame which breaks pickling).
* Drops linearly-dependent feature columns via QR with column pivoting,
  fixing "Singular matrix" failures observed when 41 standardised features
  contain near-collinear pairs.
* Reports Nakagawa & Schielzeth marginal / conditional R²; computes AIC /
  BIC manually because statsmodels returns NaN under REML.
* Skips fits when ``n_groups`` (distinct datasets) is too small for the
  kept feature count — seed replicates inflate ``len(sub)`` but do not
  add independent information for dataset-constant meta-features.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..base import (
    _alpha_to_cell,
    alpha_col_for_html,
    DatasetAnalyzer,
    group_keys_by_ratio,
    InputKind,
    render_feature_pivot_html,
    render_fit_quality_html,
)
from ..tables import FitQualityTable
from ._lme_utils import (
    _drop_collinear_columns,
    _impute_with_block_indicators,
    _insufficient_lme_data,
    _pseudo_r2,
    _zscore_features,
)


@dataclass
class LMEResult:
    """Self-contained result of one LME fit."""

    fit:               object
    coefs:             pd.DataFrame
    kept_features:     list[str]
    dropped_features:  list[str]
    feat_mean:         pd.Series
    feat_std:          pd.Series
    n_obs:             int
    n_groups:          int
    r2_marginal:       float
    r2_conditional:    float
    var_re:            float
    var_resid:         float
    aic:               float
    bic:               float
    loglik:            float
    converged:         bool


# ─────────────────────────────────────────────────────────────────────────────

class LMEAnalyzer(DatasetAnalyzer):
    """Fit per-(model, response) linear mixed-effects models.

    Consumes ``eval_long`` (one row per dataset × seed × ratio × model);
    the random intercept on ``dataset_id`` absorbs unobserved per-dataset
    baseline shifts in the response.  Fixed-effect coefficients estimate the
    partial association between each meta-feature and the response,
    identified primarily from **between-dataset** variation (meta-features
    are dataset-level; where a feature varies across seeds within a dataset,
    REML pools that within-dataset information — this is not a
    within-cluster fixed-effects slope).

    Output written by :meth:`save`:

    * ``summary.csv`` — one row per ``(ratio, model, response, feature)``
      with the standardised fixed-effect coefficient, its std-err, z,
      p-value, CI bounds, plus the (denormalised) per-(model, response)
      fit-level scalars: ``r2_marginal``, ``r2_conditional``,
      ``var_re``, ``var_resid``, ``aic``, ``bic``, ``loglik``,
      ``converged``, ``n_obs``, ``n_groups``.
    * ``summary_ratio_<r>.html`` — feature × (response, model) pivot of
      the standardised coefficient, colour-scaled with diverging RdBu.
    * ``details/<model>_<resp>_ratio_<r>.pkl`` — full pickled
      :class:`LMEResult` (with the statsmodels fit object); only useful
      if you need the raw ``MixedLMResults`` for diagnostics.
    """

    name: ClassVar[str] = "lme"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs",)

    def __init__(self, group_col: str = "dataset_id") -> None:
        self.group_col = group_col

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str,
    ) -> Optional[LMEResult]:
        sub, feat_mean, feat_std, dropped = _prepare_design(
            df, feature_cols, response, self.group_col,
        )
        kept = [c for c in sub.columns if c not in (response, self.group_col)]
        n_groups = int(sub[self.group_col].nunique())
        if _insufficient_lme_data(len(sub), n_groups, len(kept)):
            return None
        if dropped:
            print(f"    dropped {len(dropped)} collinear/constant feature(s): "
                  f"{dropped[:6]}{'...' if len(dropped) > 6 else ''}")

        # Array interface, NOT formula interface.
        exog_names = ["Intercept"] + kept
        exog_df = pd.DataFrame(
            np.column_stack([np.ones(len(sub)), sub[kept].values.astype(float)]),
            columns=exog_names, index=sub.index,
        )
        endog  = sub[response].values.astype(float)
        groups = sub[self.group_col].values

        md = sm.MixedLM(endog, exog_df, groups=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = md.fit(reml=True, method="lbfgs")
            except Exception:
                try:
                    res = md.fit(reml=True, method="bfgs")
                except Exception as exc:
                    print(f"  [LME-FAIL] {response}: {exc}")
                    return None

        fe_idx = pd.Index(exog_names)
        ci = res.conf_int()
        ci_fe = ci.loc[fe_idx] if set(fe_idx).issubset(ci.index) else ci.iloc[: len(fe_idx)]
        coefs = pd.DataFrame({
            "feature":  fe_idx,
            "coef":     res.fe_params.values,
            "std_err":  res.bse_fe.values,
            "z":        res.tvalues.iloc[: len(fe_idx)].values,
            "p":        res.pvalues.iloc[: len(fe_idx)].values,
            "ci_low":   ci_fe.iloc[:, 0].values,
            "ci_high":  ci_fe.iloc[:, 1].values,
        })

        r2_marg, r2_cond = _pseudo_r2(res, exog_df.values)
        n_params = int(len(res.fe_params) + res.cov_re.shape[0] + 1)
        aic = float(-2 * res.llf + 2 * n_params)
        bic = float(-2 * res.llf + np.log(len(sub)) * n_params)

        return LMEResult(
            fit=res, coefs=coefs,
            kept_features=kept, dropped_features=dropped,
            feat_mean=feat_mean, feat_std=feat_std,
            n_obs=int(len(sub)),
            n_groups=n_groups,
            r2_marginal=float(r2_marg),
            r2_conditional=float(r2_cond),
            var_re=float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] else 0.0,
            var_resid=float(res.scale),
            aic=aic, bic=bic,
            loglik=float(res.llf),
            converged=bool(res.converged),
        )

    def fit_per_model_response(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], LMEResult]:
        out: dict[tuple[str, str], LMEResult] = {}
        for model in models:
            sub = df[df["model"] == model]
            for resp in responses:
                print(f"  LME  | model={model:10s} response={resp}")
                result = self.fit(sub, feature_cols, resp)
                if result is None:
                    print("    skipped (insufficient data or fit failed)")
                    continue
                out[(model, resp)] = result
        return out

    # ── DatasetAnalyzer interface ────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[tuple[str, str], LMEResult]:
        """Run LME for one (ne, ratio) slice; returns
        ``dict[(model, response), LMEResult]``."""
        return self.fit_per_model_response(df, feature_cols, models, responses)

    def save(
        self,
        results_by_key: dict[
            tuple[float, "str | float | None"],
            dict[tuple[str, str], LMEResult],
        ],
        out_dir: Path,
    ) -> None:
        """Persist long summary, per-ratio HTML pivots, and the full
        pickled fit objects under ``details/``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        details_dir = out_dir / "details"
        grouped = group_keys_by_ratio(results_by_key)

        # ── 1. Long summary CSV ───────────────────────────────────────
        long_rows: list[dict] = []
        for ratio, pairs in grouped.items():
            for alpha_label, results in pairs:
                alpha_cell = _alpha_to_cell(alpha_label)
                for (model, resp), result in results.items():
                    for _, c in result.coefs.iterrows():
                        long_rows.append({
                            "ratio":          float(ratio),
                            "alpha":          alpha_cell,
                            "model":          model,
                            "response":       resp,
                            "feature":        c["feature"],
                            "coef":           float(c["coef"]),
                            "std_err":        float(c["std_err"]),
                            "z":              float(c["z"]),
                            "p":              float(c["p"]),
                            "ci_low":         float(c["ci_low"]),
                            "ci_high":        float(c["ci_high"]),
                            # Fit-level scalars repeated per feature row.
                            "n_obs":          result.n_obs,
                            "n_groups":       result.n_groups,
                            "r2_marginal":    result.r2_marginal,
                            "r2_conditional": result.r2_conditional,
                            "var_re":         result.var_re,
                            "var_resid":      result.var_resid,
                            "aic":            result.aic,
                            "bic":            result.bic,
                            "loglik":         result.loglik,
                            "converged":      result.converged,
                        })
        if long_rows:
            summary_df = pd.DataFrame(long_rows)
            summary_df.to_csv(out_dir / "summary.csv", index=False)
            # Compact fit-quality view: how much of each metric do the
            # meta-features explain?  (Nakagawa marginal / conditional R² per
            # response × model, grouped by category/scale.)
            fq = FitQualityTable(value_columns={
                "r2_marginal":    "r2_marginal",
                "r2_conditional": "r2_conditional",
            }).build(summary_df)
            if not fq.empty:
                fq.to_csv(out_dir / "fit_quality.csv")
                render_fit_quality_html(
                    fq, out_dir / "fit_quality.html",
                    caption=f"LME fit quality ({out_dir.name})",
                )

        # ── 2. Per-ratio HTML pivot ───────────────────────────────────
        for ratio, pairs in grouped.items():
            if not pairs:
                continue
            self._write_html_pivot(
                pairs, out_dir / f"summary_ratio_{ratio}.html", ratio=ratio,
            )

        # ── 3. Bulky details ──────────────────────────────────────────
        if any(results for pairs in grouped.values() for _, results in pairs):
            details_dir.mkdir(exist_ok=True)
            for ratio, pairs in grouped.items():
                for alpha_label, results in pairs:
                    alpha_tag = (
                        "alpha_free"
                        if alpha_label is None or alpha_label == "alpha_free"
                        else f"alpha_{float(alpha_label)}"
                    )
                    for (model, resp), result in results.items():
                        tag = f"{model}_{resp}_ratio_{ratio}_{alpha_tag}"
                        with open(details_dir / f"{tag}.pkl", "wb") as f:
                            pickle.dump(
                                {k: v for k, v in result.__dict__.items()}, f,
                            )

    @staticmethod
    def _write_html_pivot(
        pairs: list[tuple[object, dict[tuple[str, str], LMEResult]]],
        out_path: Path,
        *,
        ratio: float,
    ) -> None:
        """Write a feature × (alpha_label, response, model) pivot of
        standardised β. Excludes the intercept row (not a feature
        attribution). When no alpha-dependent slice is present, falls
        back to a 2-level (response, model) layout.
        """
        rows: list[dict] = []
        for alpha_label, results in pairs:
            label = (
                "alpha_free"
                if alpha_label is None or alpha_label == "alpha_free"
                else f"alpha={float(alpha_label)}"
            )
            for (model, resp), result in results.items():
                for _, c in result.coefs.iterrows():
                    if c["feature"] == "Intercept":
                        continue
                    rows.append({
                        "feature":     c["feature"],
                        "response":    resp,
                        "model":       model,
                        "coef":        float(c["coef"]),
                        "alpha_label": label,
                    })
        if not rows:
            return
        df = pd.DataFrame(rows)
        render_feature_pivot_html(
            df,
            out_path,
            value_col="coef",
            caption=(
                f"LME standardised fixed-effect coefficient β "
                f"(ratio={ratio}; rows = meta-feature, "
                f"cols = alpha × response × model)"
            ),
            cmap="RdBu_r",
            diverging=True,
            fmt="{:+.3f}",
            alpha_col=alpha_col_for_html(pairs),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Design preparation (per-model LME-specific)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_design(
    df: pd.DataFrame,
    feature_cols: list[str],
    outcome: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Build the per-model LME design: imputed + z-scored features + outcome + group.

    Missing meta-features are handled by
    :func:`evaluation.analysis.multivariate._lme_utils._impute_with_block_indicators`
    before z-scoring.  Only rows missing the outcome or group identifier are
    dropped; feature NaN values are imputed rather than dropped.

    Z-scoring and constant-column drop are delegated to
    :func:`evaluation.analysis.multivariate._lme_utils._zscore_features`;
    collinearity is then resolved on the standardised matrix via
    :func:`evaluation.analysis.multivariate._lme_utils._drop_collinear_columns`.
    """
    cols = feature_cols + [outcome, group_col]
    sub  = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    # Only drop rows where the outcome or grouping variable is missing.
    sub  = sub.dropna(subset=[outcome, group_col]).copy()

    sub, new_feature_cols = _impute_with_block_indicators(sub, feature_cols)

    Xz_df, mean, std, _const_dropped = _zscore_features(sub, new_feature_cols)

    keep_idx, kept_features = _drop_collinear_columns(
        Xz_df.values, list(Xz_df.columns),
    )
    Xz_df = Xz_df.iloc[:, keep_idx]
    mean  = mean.iloc[keep_idx]
    std   = std.iloc[keep_idx]

    sub_out = Xz_df.copy()
    sub_out[outcome]   = sub[outcome].values
    sub_out[group_col] = sub[group_col].values

    dropped = [f for f in new_feature_cols if f not in kept_features]
    return sub_out, mean, std, dropped
