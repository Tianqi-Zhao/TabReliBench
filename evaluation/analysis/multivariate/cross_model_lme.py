"""Cross-model pooled LME analyzer.

Fits a single linear mixed-effects model **per (ratio, response)** that
pools observations from every model and uses model fixed effects plus
model-by-feature interactions::

    Y_{d,m,s} = β₀ + Σⱼ βⱼ x_{d,j} + αₘ + Σⱼ γⱼ,ₘ x_{d,j} I(m)
                + u_d + ε_{d,m,s}

where ``d`` indexes dataset, ``m`` indexes model, ``s`` indexes seed.

Compared with :class:`evaluation.analysis.multivariate.lme.LMEAnalyzer`
— which fits one independent LME per ``(model, response)`` cell — the
cross-model fit:

* puts every model on a shared scale (intercepts and slopes are
  estimated against a single reference model),
* lets us read off γⱼ,ₘ directly as "how much does model m's slope on
  feature j differ from the reference?",
* and reports a per-model *effective slope* β_j + γ_{j,m} together with
  the SE derived from the joint fixed-effect covariance matrix via
  ``Var(β + γ) = Var(β) + Var(γ) + 2·Cov(β, γ)``.

Implementation mirrors the existing LME (array interface for
pickle-cleanliness, REML with lbfgs → bfgs fallback, pseudo-R² and
manual AIC/BIC).  Shared helpers live in
:mod:`evaluation.analysis.multivariate._lme_utils`.

Missing meta-features are handled with the block-aware B+C strategy from
:func:`evaluation.analysis.multivariate._lme_utils._impute_with_block_indicators`:
structurally-absent feature groups (``cat_*``, ``clf_hs_*``) are filled
with within-present means and represented by an indicator column; scattered
NaN values are filled with the global column mean.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from ..base import (
    _alpha_to_cell,
    alpha_col_for_html,
    DatasetAnalyzer,
    group_keys_by_ratio,
    InputKind,
    render_feature_pivot_html,
)
from ._lme_utils import (
    _drop_collinear_columns,
    _impute_with_block_indicators,
    _insufficient_lme_data,
    _pseudo_r2,
    _zscore_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossModelLMEResult:
    """Self-contained result of one pooled cross-model LME fit."""

    fit:               object
    ref_model:         str
    model_levels:      list[str]          # [ref, *non_ref] in design order
    coefs:             pd.DataFrame       # tagged: intercept/main/model_fe/interaction
    per_model_slopes:  pd.DataFrame       # model × feature with slope + SE + z + p
    interaction_long:  pd.DataFrame       # γ_{j,m} + SE + z + p; ref rows = (0, NaN)
    model_effects:     pd.DataFrame       # α_m per model: (model, alpha, std_err, z, p, ci_low, ci_high)
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
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class CrossModelLMEAnalyzer(DatasetAnalyzer):
    """Pooled cross-model linear mixed-effects analyzer.

    One fit per ``(ratio, response)``; models are pooled into the
    design via reference-coded dummies and feature × dummy interactions.

    Output written by :meth:`save`:

    * ``summary.csv`` — long ``(ratio, response, feature, model,
      effective_slope, slope_se, slope_z, slope_p, gamma, gamma_se,
      gamma_p, ref_model, n_obs, n_groups, r2_marginal,
      r2_conditional, var_re, var_resid, aic, bic, loglik,
      converged)``.
    * ``summary_ratio_<r>.html`` — feature × (response, model) pivot
      of the per-model **effective slope** β_j + γ_{j,m} with
      significance stars on ``slope_p`` (diverging RdBu).
    * ``interaction_ratio_<r>.html`` — feature × (response, model)
      pivot of the **interaction** γ_{j,m} alone (reference column
      blank), stars on ``gamma_p``.
    * ``details/<response>_ratio_<r>.pkl`` — pickled
      :class:`CrossModelLMEResult` (one file per response, since
      models are pooled into a single fit).
    """

    name: ClassVar[str] = "cross_model_lme"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs",)

    def __init__(
        self,
        group_col: str = "dataset_id",
        ref_model: Optional[str] = None,
    ) -> None:
        self.group_col = group_col
        self.ref_model = ref_model

    # ── Core fit ─────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        response: str,
        models: list[str],
    ) -> Optional[CrossModelLMEResult]:
        if not models:
            return None

        cols = list(feature_cols) + [response, self.group_col, "model"]
        sub = (
            df[df["model"].isin(models)][cols]
            .replace([np.inf, -np.inf], np.nan)
            .copy()
        )
        # Only drop rows where the response or group identifier is missing.
        sub = sub.dropna(subset=[response, self.group_col])
        if sub.empty:
            return None

        # Resolve reference model: prefer the user-supplied one if it
        # actually appears in this slice, else fall back to the first
        # model in the supplied list that is present.
        present_models = [m for m in models if m in set(sub["model"].unique())]
        if not present_models:
            return None
        if self.ref_model and self.ref_model in present_models:
            ref = self.ref_model
        else:
            ref = present_models[0]
        non_ref = [m for m in present_models if m != ref]
        model_levels = [ref, *sorted(non_ref)]

        # Impute missing meta-features using the block-aware B+C strategy.
        # Structurally-absent blocks (cat_*, clf_hs_*) are filled with their
        # within-present means and contribute an indicator column; scattered
        # NaN values are filled with the global column mean.
        sub, new_feature_cols = _impute_with_block_indicators(sub, feature_cols)

        # Z-score features, then drop collinear columns on the standardised
        # matrix (collinearity is most easily diagnosed there).
        Xz_df, feat_mean, feat_std, _ = _zscore_features(sub, new_feature_cols)
        keep_idx, kept = _drop_collinear_columns(
            Xz_df.values, list(Xz_df.columns),
        )
        Xz_df = Xz_df.iloc[:, keep_idx]
        feat_mean = feat_mean.iloc[keep_idx]
        feat_std  = feat_std.iloc[keep_idx]

        K = len(kept)
        n_groups = int(sub[self.group_col].nunique())

        # Gate on K (number of kept features), matching LMEAnalyzer.
        # The model dummies α_m and interactions γ_{j,m} are identified from
        # within-cluster variation (each dataset has M×seeds rows), so they
        # do not inflate the n_groups requirement.
        if _insufficient_lme_data(len(sub), n_groups, K):
            return None

        dropped = [f for f in new_feature_cols if f not in kept]
        if dropped:
            print(f"    dropped {len(dropped)} collinear/constant feature(s): "
                  f"{dropped[:6]}{'...' if len(dropped) > 6 else ''}")

        # ── Build the pooled design matrix ──────────────────────────────
        # Order: Intercept | features | model dummies | interactions
        intercept_col = np.ones(len(sub))
        feat_block    = Xz_df.values.astype(float)
        model_arr     = sub["model"].values

        dummy_cols  = []
        dummy_names = []
        for m in model_levels[1:]:  # skip reference
            d = (model_arr == m).astype(float)
            dummy_cols.append(d)
            dummy_names.append(self._model_term(m))
        dummy_block = (
            np.column_stack(dummy_cols) if dummy_cols
            else np.empty((len(sub), 0))
        )

        inter_cols  = []
        inter_names = []
        for j, feat in enumerate(kept):
            for k, m in enumerate(model_levels[1:]):
                inter_cols.append(feat_block[:, j] * dummy_block[:, k])
                inter_names.append(self._interaction_term(feat, m))
        inter_block = (
            np.column_stack(inter_cols) if inter_cols
            else np.empty((len(sub), 0))
        )

        exog_names = (
            ["Intercept"] + list(kept) + dummy_names + inter_names
        )
        exog = np.column_stack([intercept_col, feat_block, dummy_block, inter_block])
        exog_df = pd.DataFrame(exog, columns=exog_names, index=sub.index)
        endog   = sub[response].values.astype(float)
        groups  = sub[self.group_col].values

        # ── Fit ─────────────────────────────────────────────────────────
        md = sm.MixedLM(endog, exog_df, groups=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = md.fit(reml=True, method="lbfgs")
            except Exception:
                try:
                    res = md.fit(reml=True, method="bfgs")
                except Exception as exc:
                    print(f"  [CROSS-LME-FAIL] {response}: {exc}")
                    return None

        # ── Coefs DataFrame (tagged) ───────────────────────────────────
        fe_idx = pd.Index(exog_names)
        ci     = res.conf_int()
        ci_fe  = (
            ci.loc[fe_idx] if set(fe_idx).issubset(ci.index)
            else ci.iloc[: len(fe_idx)]
        )
        kinds = (
            ["intercept"]
            + ["main"] * len(kept)
            + ["model_fe"] * len(dummy_names)
            + ["interaction"] * len(inter_names)
        )
        coefs = pd.DataFrame({
            "param":   fe_idx,
            "kind":    kinds,
            "coef":    res.fe_params.values,
            "std_err": res.bse_fe.values,
            "z":       res.tvalues.iloc[: len(fe_idx)].values,
            "p":       res.pvalues.iloc[: len(fe_idx)].values,
            "ci_low":  ci_fe.iloc[:, 0].values,
            "ci_high": ci_fe.iloc[:, 1].values,
        })

        # ── Per-model slopes and interaction long table ────────────────
        per_model_slopes, interaction_long = self._derive_per_model_tables(
            res=res,
            exog_names=exog_names,
            kept=list(kept),
            model_levels=list(model_levels),
        )

        # ── Model fixed-effect intercept shifts α_m ──────────────────
        model_effects = self._derive_model_effects(
            coefs=coefs,
            ref_model=ref,
            model_levels=list(model_levels),
        )

        # ── Pseudo-R² + manual AIC / BIC (statsmodels NaN under REML) ──
        r2_marg, r2_cond = _pseudo_r2(res, exog_df.values)
        n_params = int(len(res.fe_params) + res.cov_re.shape[0] + 1)
        aic = float(-2 * res.llf + 2 * n_params)
        bic = float(-2 * res.llf + np.log(len(sub)) * n_params)

        return CrossModelLMEResult(
            fit=res,
            ref_model=ref,
            model_levels=list(model_levels),
            coefs=coefs,
            per_model_slopes=per_model_slopes,
            interaction_long=interaction_long,
            model_effects=model_effects,
            kept_features=list(kept),
            dropped_features=dropped,
            feat_mean=feat_mean,
            feat_std=feat_std,
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

    # ── Per-response loop ────────────────────────────────────────────────

    def fit_per_response(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[str, CrossModelLMEResult]:
        out: dict[str, CrossModelLMEResult] = {}
        for resp in responses:
            print(f"  CROSS-LME | response={resp}  (pooled over {len(models)} models)")
            result = self.fit(df, feature_cols, resp, models)
            if result is None:
                print("    skipped (insufficient data or fit failed)")
                continue
            out[resp] = result
        return out

    # ── DatasetAnalyzer interface ────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> dict[str, CrossModelLMEResult]:
        return self.fit_per_response(df, feature_cols, models, responses)

    def save(
        self,
        results_by_key: dict[
            tuple[float, "str | float | None"],
            dict[str, CrossModelLMEResult],
        ],
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        details_dir = out_dir / "details"
        grouped = group_keys_by_ratio(results_by_key)

        # ── 1. Long summary CSV ───────────────────────────────────────
        # NB: column name ``model_alpha`` (not ``alpha``) avoids a clash
        # with the new miscoverage ``alpha`` column added per row below.
        long_rows: list[dict] = []
        for ratio, pairs in grouped.items():
            for alpha_label, results in pairs:
                alpha_cell = _alpha_to_cell(alpha_label)
                for resp, result in results.items():
                    alpha_lookup = result.model_effects.set_index("model")
                    slopes = result.per_model_slopes.set_index(["model", "feature"])
                    inters = result.interaction_long.set_index(["model", "feature"])
                    for (model, feature), s_row in slopes.iterrows():
                        i_row = inters.loc[(model, feature)] if (model, feature) in inters.index else None
                        a_row = alpha_lookup.loc[model] if model in alpha_lookup.index else None
                        long_rows.append({
                            "ratio":             float(ratio),
                            "alpha":             alpha_cell,
                            "response":          resp,
                            "feature":           feature,
                            "model":             model,
                            "effective_slope":   float(s_row["slope"]),
                            "slope_se":          float(s_row["slope_se"]),
                            "slope_z":           float(s_row["slope_z"]),
                            "slope_p":           float(s_row["slope_p"]),
                            "gamma":             float(i_row["gamma"])    if i_row is not None else float("nan"),
                            "gamma_se":          float(i_row["std_err"])  if i_row is not None else float("nan"),
                            "gamma_p":           float(i_row["p"])        if i_row is not None else float("nan"),
                            # LME model intercept shift α_m (repeated per
                            # feature row). Renamed from ``alpha`` to
                            # avoid clash with the miscoverage alpha.
                            "model_alpha":       float(a_row["alpha"])    if a_row is not None else float("nan"),
                            "model_alpha_se":    float(a_row["std_err"]) if a_row is not None else float("nan"),
                            "model_alpha_p":     float(a_row["p"])        if a_row is not None else float("nan"),
                            "ref_model":         result.ref_model,
                            "n_obs":             result.n_obs,
                            "n_groups":          result.n_groups,
                            "r2_marginal":       result.r2_marginal,
                            "r2_conditional":    result.r2_conditional,
                            "var_re":            result.var_re,
                            "var_resid":         result.var_resid,
                            "aic":               result.aic,
                            "bic":               result.bic,
                            "loglik":            result.loglik,
                            "converged":         result.converged,
                        })
        if long_rows:
            pd.DataFrame(long_rows).to_csv(out_dir / "summary.csv", index=False)

        # ── 2. Per-ratio HTML pivots ──────────────────────────────────
        for ratio, pairs in grouped.items():
            if not pairs:
                continue
            slope_rows: list[dict] = []
            inter_rows: list[dict] = []
            for alpha_label, results in pairs:
                a_html = (
                    "alpha_free"
                    if alpha_label is None or alpha_label == "alpha_free"
                    else f"alpha={float(alpha_label)}"
                )
                for resp, result in results.items():
                    for _, r in result.per_model_slopes.iterrows():
                        slope_rows.append({
                            "feature":         r["feature"],
                            "response":        resp,
                            "model":           r["model"],
                            "effective_slope": float(r["slope"]),
                            "slope_p":         float(r["slope_p"]),
                            "alpha_label":     a_html,
                        })
                    for _, r in result.interaction_long.iterrows():
                        if r["model"] == result.ref_model:
                            continue  # ref column has γ=0 by construction
                        inter_rows.append({
                            "feature":     r["feature"],
                            "response":    resp,
                            "model":       r["model"],
                            "gamma":       float(r["gamma"]),
                            "gamma_p":     float(r["p"]),
                            "alpha_label": a_html,
                        })

            html_alpha_col = alpha_col_for_html(pairs)
            if slope_rows:
                render_feature_pivot_html(
                    pd.DataFrame(slope_rows),
                    out_dir / f"summary_ratio_{ratio}.html",
                    value_col="effective_slope",
                    p_col="slope_p",
                    caption=(
                        f"Cross-model pooled LME — effective slope "
                        f"β_j + γ_(j,m) (ratio={ratio}; rows = meta-feature, "
                        f"cols = alpha × response × model; stars on slope p)"
                    ),
                    cmap="RdBu_r",
                    diverging=True,
                    fmt="{:+.3f}",
                    alpha_col=html_alpha_col,
                )
            if inter_rows:
                render_feature_pivot_html(
                    pd.DataFrame(inter_rows),
                    out_dir / f"interaction_ratio_{ratio}.html",
                    value_col="gamma",
                    p_col="gamma_p",
                    caption=(
                        f"Cross-model pooled LME — interaction γ_(j,m) "
                        f"(deviation from reference model; ratio={ratio}; "
                        f"rows = meta-feature, cols = alpha × response × non-ref model)"
                    ),
                    cmap="RdBu_r",
                    diverging=True,
                    fmt="{:+.3f}",
                    alpha_col=html_alpha_col,
                )

        # ── 3. Model fixed-effect intercepts α_m (HTML only) ─────────
        #     One HTML per (ratio, alpha_label); α_m values already live
        #     in summary.csv (model_alpha / _se / _p).
        for ratio, pairs in grouped.items():
            if not pairs:
                continue
            for alpha_label, results in pairs:
                if not results:
                    continue
                a_tag = (
                    "alpha_free"
                    if alpha_label is None or alpha_label == "alpha_free"
                    else f"alpha_{float(alpha_label)}"
                )
                fname = (
                    f"model_effects_ratio_{ratio}.html"
                    if alpha_label is None
                    else f"model_effects_ratio_{ratio}_{a_tag}.html"
                )
                self._write_model_effects_html(
                    results, out_dir / fname, ratio=ratio,
                )

        # ── 4. Bulky details ──────────────────────────────────────────
        if any(results for pairs in grouped.values() for _, results in pairs):
            details_dir.mkdir(exist_ok=True)
            for ratio, pairs in grouped.items():
                for alpha_label, results in pairs:
                    a_tag = (
                        "alpha_free"
                        if alpha_label is None or alpha_label == "alpha_free"
                        else f"alpha_{float(alpha_label)}"
                    )
                    for resp, result in results.items():
                        tag = f"{resp}_ratio_{ratio}_{a_tag}"
                        with open(details_dir / f"{tag}.pkl", "wb") as f:
                            pickle.dump(
                                {k: v for k, v in result.__dict__.items()}, f,
                            )

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _model_term(model: str) -> str:
        """Design-matrix column name for a model fixed-effect dummy."""
        return f"model[T.{model}]"

    @staticmethod
    def _interaction_term(feature: str, model: str) -> str:
        """Design-matrix column name for a feature × model interaction."""
        return f"{feature}:model[T.{model}]"

    @staticmethod
    def _write_model_effects_html(
        results: dict[str, "CrossModelLMEResult"],
        out_path: Path,
        *,
        ratio: float,
    ) -> None:
        """Write a model × response pivot of α_m (model intercept shifts).

        Since α_m has no feature dimension, the standard
        ``render_feature_pivot_html`` helper (which expects a ``feature``
        axis) doesn't apply.  Instead, build a simple ``model × response``
        pivot with diverging colour scale and significance stars.
        """
        from ..base import _significance_suffix

        rows: list[dict] = []
        for resp, result in results.items():
            for _, r in result.model_effects.iterrows():
                rows.append({
                    "response": resp,
                    "model":    r["model"],
                    "alpha":    float(r["alpha"]),
                    "p":        float(r["p"]),
                })
        if not rows:
            return
        df = pd.DataFrame(rows)
        pivot = df.pivot_table(
            index="model", columns="response",
            values="alpha", aggfunc="first",
        ).sort_index(axis=1)
        pivot_p = df.pivot_table(
            index="model", columns="response",
            values="p", aggfunc="first",
        ).sort_index(axis=1)
        # Reindex to match pivot — the ref model's all-NaN p row may have
        # been dropped by pivot_table.
        pivot_p = pivot_p.reindex(index=pivot.index, columns=pivot.columns)
        if pivot.empty:
            return

        star_levels = ((0.01, "**"), (0.05, "*"))

        # Build text cells with significance stars.
        text = pd.DataFrame(
            index=pivot.index, columns=pivot.columns, dtype=object,
        )
        for idx in pivot.index:
            for col in pivot.columns:
                v = pivot.at[idx, col]
                p = pivot_p.at[idx, col]
                if pd.isna(v) or not np.isfinite(v):
                    text.at[idx, col] = "—"
                    continue
                stars = _significance_suffix(float(p), levels=star_levels) if np.isfinite(p) else ""
                text.at[idx, col] = f"{v:+.3f}{stars}"

        max_abs = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
        vmax = max(max_abs, 1e-6)
        styled = (
            pivot.style
            .background_gradient(cmap="RdBu_r", axis=None, vmin=-vmax, vmax=vmax)
            .format(lambda _v: "", na_rep="—")  # hide raw numbers
        )
        # Overlay text cells (with stars) on top of colour background.
        for idx in pivot.index:
            styled = styled.format(
                {
                    col: (lambda _v, idx=idx, col=col: text.at[idx, col])
                    for col in pivot.columns
                },
                subset=pd.IndexSlice[idx, :],
            )
        styled = styled.set_caption(
            f"Cross-model LME — model intercept shifts α_m relative to "
            f"reference (ratio={ratio}; rows = model, cols = response)"
        )
        out_path.write_text(styled.to_html(), encoding="utf-8")

    @classmethod
    def _derive_model_effects(
        cls,
        *,
        coefs: pd.DataFrame,
        ref_model: str,
        model_levels: list[str],
    ) -> pd.DataFrame:
        """Build a ``(model, alpha, std_err, z, p, ci_low, ci_high)`` table.

        ``alpha`` is the model fixed-effect intercept shift α_m relative
        to the reference model.  The reference row has α = 0 and
        NaN statistics.
        """
        rows: list[dict] = []
        # Reference model: α = 0 by construction.
        rows.append({
            "model":   ref_model,
            "alpha":   0.0,
            "std_err": float("nan"),
            "z":       float("nan"),
            "p":       float("nan"),
            "ci_low":  float("nan"),
            "ci_high": float("nan"),
        })
        # Non-reference models: read from the model_fe rows of coefs.
        model_fe = coefs[coefs["kind"] == "model_fe"]
        for m in model_levels[1:]:
            term = cls._model_term(m)
            mask = model_fe["param"] == term
            if mask.any():
                r = model_fe.loc[mask.idxmax()]
                rows.append({
                    "model":   m,
                    "alpha":   float(r["coef"]),
                    "std_err": float(r["std_err"]),
                    "z":       float(r["z"]),
                    "p":       float(r["p"]),
                    "ci_low":  float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                })
            else:
                rows.append({
                    "model":   m,
                    "alpha":   float("nan"),
                    "std_err": float("nan"),
                    "z":       float("nan"),
                    "p":       float("nan"),
                    "ci_low":  float("nan"),
                    "ci_high": float("nan"),
                })
        return pd.DataFrame(rows)

    @classmethod
    def _derive_per_model_tables(
        cls,
        *,
        res,
        exog_names: list[str],
        kept: list[str],
        model_levels: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(per_model_slopes, interaction_long)`` from the fit.

        ``per_model_slopes`` reports the **effective slope** β_j (ref)
        or β_j + γ_{j,m} (non-ref) for every (model, feature) pair,
        with SE/z/p derived from the joint fixed-effect covariance via
        ``Var(β + γ) = Var(β) + Var(γ) + 2·Cov(β, γ)``.

        ``interaction_long`` mirrors ``coefs`` for the interaction rows
        and inserts a zero-valued, NaN-p row for the reference model so
        downstream pivots have a complete (feature, model) grid.
        """
        fe = res.fe_params
        cov = res.cov_params()
        # Restrict cov to fixed-effect rows/cols only (statsmodels also
        # exposes random-effect variance components).
        cov_fe = cov.loc[exog_names, exog_names]
        ci = res.conf_int()
        ref = model_levels[0]

        slope_rows = []
        inter_rows = []
        for feat in kept:
            beta = float(fe[feat])
            var_beta = float(cov_fe.at[feat, feat])

            # Reference model: effective slope = β; γ row = (0, NaN p).
            se_ref = float(np.sqrt(max(var_beta, 0.0)))
            z_ref  = beta / se_ref if se_ref > 0 else float("nan")
            p_ref  = (
                float(2.0 * (1.0 - norm.cdf(abs(z_ref))))
                if np.isfinite(z_ref) else float("nan")
            )
            slope_rows.append({
                "model":    ref,
                "feature":  feat,
                "slope":    beta,
                "slope_se": se_ref,
                "slope_z":  z_ref,
                "slope_p":  p_ref,
            })
            inter_rows.append({
                "model":   ref,
                "feature": feat,
                "gamma":   0.0,
                "std_err": float("nan"),
                "z":       float("nan"),
                "p":       float("nan"),
                "ci_low":  float("nan"),
                "ci_high": float("nan"),
            })

            # Non-reference models: slope = β + γ; γ is the interaction
            # coefficient on (feature × model dummy).
            for m in model_levels[1:]:
                term = cls._interaction_term(feat, m)
                gamma     = float(fe[term])
                var_gamma = float(cov_fe.at[term, term])
                cov_bg    = float(cov_fe.at[feat, term])

                slope = beta + gamma
                var_slope = var_beta + var_gamma + 2.0 * cov_bg
                se_slope  = float(np.sqrt(max(var_slope, 0.0)))
                z_slope   = slope / se_slope if se_slope > 0 else float("nan")
                p_slope   = (
                    float(2.0 * (1.0 - norm.cdf(abs(z_slope))))
                    if np.isfinite(z_slope) else float("nan")
                )

                se_gamma = float(np.sqrt(max(var_gamma, 0.0)))
                z_gamma  = gamma / se_gamma if se_gamma > 0 else float("nan")
                p_gamma  = (
                    float(2.0 * (1.0 - norm.cdf(abs(z_gamma))))
                    if np.isfinite(z_gamma) else float("nan")
                )
                if term in ci.index:
                    ci_lo = float(ci.loc[term, 0])
                    ci_hi = float(ci.loc[term, 1])
                else:
                    z_crit = float(norm.ppf(0.975))
                    ci_lo = gamma - z_crit * se_gamma
                    ci_hi = gamma + z_crit * se_gamma

                slope_rows.append({
                    "model":    m,
                    "feature":  feat,
                    "slope":    slope,
                    "slope_se": se_slope,
                    "slope_z":  z_slope,
                    "slope_p":  p_slope,
                })
                inter_rows.append({
                    "model":   m,
                    "feature": feat,
                    "gamma":   gamma,
                    "std_err": se_gamma,
                    "z":       z_gamma,
                    "p":       p_gamma,
                    "ci_low":  ci_lo,
                    "ci_high": ci_hi,
                })

        return pd.DataFrame(slope_rows), pd.DataFrame(inter_rows)
