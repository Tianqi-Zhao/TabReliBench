"""Signal-quality features for classification datasets."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_selection import f_classif

from .. import ClassificationDatasetFeatureContext, DatasetFeatureContext, DatasetFeatureGroup
from ._utils import all_columns_target_mi, gini, shannon_entropy


class ClassSignalQuality(DatasetFeatureGroup):
    """Feature ↔ class-label association.

    Analogous to ``SignalQuality`` for regression.  MI is computed over
    **numeric + categorical** columns (categorical columns are integer-coded
    and passed with ``discrete_features=True`` so the encoding choice does
    not affect the result).  ANOVA F is numeric-only on purpose:
    F-on-ordinal-encoded categoricals would inject a spurious order.
    """

    name = "class_signal_quality"
    feature_names = (
        "mi_mean",
        "mi_max",
        "mi_entropy",
        "mi_mean_norm",
        "mi_max_norm",
        "mi_entropy_norm",
        "gini_mi",
        "ns_ratio",
        "anova_f_mean",
        "anova_f_max",
        "anova_eta2_mean",
        "anova_eta2_max",
        "near_constant_frac",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        assert isinstance(ctx, ClassificationDatasetFeatureContext)
        out: dict[str, float] = {}

        mi_vals = all_columns_target_mi(ctx)
        if mi_vals.size > 0:
            mi_p = mi_vals / (mi_vals.sum() + 1e-10)
            mi_mean_raw = float(mi_vals.mean())
            mi_max_raw  = float(mi_vals.max())
            mi_ent_raw  = float(shannon_entropy(mi_p[mi_p > 0]))
            out["mi_mean"]    = mi_mean_raw
            out["mi_max"]     = mi_max_raw
            out["mi_entropy"] = mi_ent_raw
            out["gini_mi"]    = float(gini(mi_vals))
            out["ns_ratio"]   = float(
                np.mean(mi_vals < 0.05 * max(mi_vals.max(), 1e-10))
            )
            # Normalized variants: MI / H(Y) so values are comparable across K.
            probs = ctx.class_counts / ctx.class_counts.sum()
            hy = float(shannon_entropy(probs))
            out["mi_mean_norm"] = mi_mean_raw / hy if hy > 1e-10 else float("nan")
            out["mi_max_norm"]  = mi_max_raw / hy if hy > 1e-10 else float("nan")
            # mi_entropy / log(d) so values are comparable across feature count.
            d_total = mi_vals.size
            log_d = float(np.log(d_total)) if d_total > 1 else 0.0
            out["mi_entropy_norm"] = mi_ent_raw / log_d if log_d > 1e-10 else float("nan")
        else:
            out.update({k: float("nan") for k in (
                "mi_mean", "mi_max", "mi_entropy",
                "mi_mean_norm", "mi_max_norm", "mi_entropy_norm",
                "gini_mi", "ns_ratio",
            )})

        # Filter near-constant columns up front: f_classif on a column with
        # near-zero within-class variance yields 0/0 → NaN (truly undefined)
        # or k/0 → inf (perfectly separating).  Replacing those with 0 — as
        # we used to via nan_to_num — is wrong in both directions: the first
        # case loses the "undefined" signal, the second flips a perfectly
        # separating feature into "no signal".  Align with the regression
        # Pearson path, which already filters std<=1e-10.
        if ctx.d_num > 0 and ctx.n_clean >= 4 and ctx.n_classes >= 2:
            x = ctx.X_num_clean.values
            keep = np.std(x, axis=0) > 1e-10
            if keep.any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    f_vals, _ = f_classif(x[:, keep], ctx.y_int)
                f_vals = f_vals[np.isfinite(f_vals)]
                if f_vals.size > 0:
                    out["anova_f_mean"] = float(f_vals.mean())
                    out["anova_f_max"]  = float(f_vals.max())
                    n = len(ctx.y_int)
                    df_between = ctx.n_classes - 1
                    df_within = n - ctx.n_classes
                    if df_between > 0 and df_within > 0:
                        eta2 = (
                            f_vals * df_between
                            / (f_vals * df_between + df_within)
                        )
                        out["anova_eta2_mean"] = float(eta2.mean())
                        out["anova_eta2_max"] = float(eta2.max())
                    else:
                        out["anova_eta2_mean"] = float("nan")
                        out["anova_eta2_max"] = float("nan")
                else:
                    out["anova_f_mean"] = float("nan")
                    out["anova_f_max"]  = float("nan")
                    out["anova_eta2_mean"] = float("nan")
                    out["anova_eta2_max"] = float("nan")
            else:
                out["anova_f_mean"] = float("nan")
                out["anova_f_max"]  = float("nan")
                out["anova_eta2_mean"] = float("nan")
                out["anova_eta2_max"] = float("nan")
        else:
            out["anova_f_mean"] = float("nan")
            out["anova_f_max"]  = float("nan")
            out["anova_eta2_mean"] = float("nan")
            out["anova_eta2_max"] = float("nan")

        if ctx.d_num > 0:
            nc = []
            for col in ctx.X_num.columns:
                xj = ctx.X_num[col].dropna().values
                rng_j = xj.max() - xj.min() if len(xj) > 1 else 0.0
                nc.append(
                    float(xj.std() / max(rng_j, 1e-10) < 0.01)
                    if len(xj) > 1 else 1.0
                )
            out["near_constant_frac"] = float(np.mean(nc))
        else:
            out["near_constant_frac"] = float("nan")

        return out
