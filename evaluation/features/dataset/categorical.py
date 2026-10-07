"""Categorical-column meta-features shared by regression and classification.

Every other ``DatasetFeatureGroup`` operates on the numeric matrix
``ctx.X_num`` / ``ctx.X_sc`` and silently discards non-numeric columns.
This group fills that hole by extracting:

* **Structure** — cardinality of finite-cardinality categorical columns
  (mean / max / variance), plus the fraction of all dataset features that are
  ID-like categorical columns.  Column counts and ``cat_frac`` are *not*
  duplicated here; use ``discrete_frac`` and ``n_features`` from
  ``dimensionality_capacity`` instead.
* **Distribution** — normalised entropy, mode dominance, and how much
  mass sits in rare categories (<1%) among finite-cardinality categorical
  columns.

Target-association MI is **not** computed here.  ``SignalQuality`` and
``ClassSignalQuality`` already include each categorical column in the
all-columns MI vector (via ``all_columns_target_mi``), so a separate
``cat_target_mi_*`` here would be redundant.

Missingness is covered only by the global ``nan_frac_overall`` /
``nan_col_frac`` in ``dimensionality_capacity`` (all columns, not cat-only).

Definition of "categorical" here is ``select_dtypes`` based: ``object``,
``string``, ``category``, and ``bool`` columns.  A categorical column is
treated as ID-like when almost every observed value is unique; such columns are
excluded from the cardinality/distribution summaries so their effect is
concentrated in ``cat_id_like_frac``.
"""
from __future__ import annotations

import numpy as np

from .. import (
    DatasetFeatureContext,
    DatasetFeatureGroup,
)
from ._utils import shannon_entropy

_ID_LIKE_UNIQUE_FRAC: float = 0.9
_ID_LIKE_MIN_OBS: int = 20
_RARE_CATEGORY_PROB: float = 0.01

_CATEGORICAL_DTYPES: tuple[str, ...] = ("object", "string", "category", "bool")


class CategoricalFeatures(DatasetFeatureGroup):
    """Cardinality and distribution stats for cat columns."""

    name = "categorical_features"
    feature_names = (
        "cat_cardinality_mean",
        "cat_cardinality_max",
        "cat_cardinality_var",
        "cat_id_like_frac",
        "cat_norm_entropy_mean",
        "cat_mode_frac_mean",
        "cat_rare_category_frac_mean",
    )

    def compute(self, ctx: DatasetFeatureContext) -> dict[str, float]:
        X_cat = ctx.X_df.select_dtypes(include=list(_CATEGORICAL_DTYPES))
        n_cat = int(X_cat.shape[1])

        if n_cat == 0:
            return {name: float("nan") for name in self.feature_names}

        finite_cardinalities: list[int] = []
        id_like_count = 0
        norm_entropies: list[float] = []
        mode_fracs:     list[float] = []
        rare_fracs:     list[float] = []

        for col_name in X_cat.columns:
            col = X_cat[col_name]
            counts = col.dropna().value_counts()
            n_obs = int(counts.sum())
            if n_obs == 0:
                # Empty after dropna(), or pandas ``category`` dtype listing
                # declared levels with zero counts (e.g. sparse OpenML flags).
                finite_cardinalities.append(0)
                norm_entropies.append(float("nan"))
                mode_fracs.append(float("nan"))
                rare_fracs.append(float("nan"))
                continue

            k = int(counts.size)
            if _is_id_like_cardinality(k, n_obs):
                id_like_count += 1
                continue

            finite_cardinalities.append(k)
            probs = (counts.to_numpy(dtype=float) / n_obs)
            mode_fracs.append(float(probs.max()))
            rare_fracs.append(float(probs[probs < _RARE_CATEGORY_PROB].sum()))

            if k > 1:
                norm_entropies.append(float(shannon_entropy(probs) / np.log(k)))
            else:
                norm_entropies.append(0.0)

        cards = np.asarray(finite_cardinalities, dtype=float)
        id_like_frac = float(id_like_count / ctx.d) if ctx.d > 0 else float("nan")
        if cards.size == 0:
            return {
                "cat_cardinality_mean": float("nan"),
                "cat_cardinality_max": float("nan"),
                "cat_cardinality_var": float("nan"),
                "cat_id_like_frac": id_like_frac,
                "cat_norm_entropy_mean": float("nan"),
                "cat_mode_frac_mean": float("nan"),
                "cat_rare_category_frac_mean": float("nan"),
            }

        return {
            "cat_cardinality_mean": float(cards.mean()),
            "cat_cardinality_max": float(cards.max()),
            "cat_cardinality_var": float(cards.var()),
            "cat_id_like_frac": id_like_frac,
            "cat_norm_entropy_mean": _nanmean(norm_entropies),
            "cat_mode_frac_mean": _nanmean(mode_fracs),
            "cat_rare_category_frac_mean": _nanmean(rare_fracs),
        }


def _is_id_like_cardinality(cardinality: int, n_obs: int) -> bool:
    if n_obs < _ID_LIKE_MIN_OBS:
        return False
    return (cardinality / n_obs) >= _ID_LIKE_UNIQUE_FRAC


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else float("nan")
