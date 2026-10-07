"""Long DataFrame builders for downstream analysis.

Three builders, layered:

* :class:`MetricsTable` walks an :class:`ArtifactStore`, reads every metrics
  PKL it knows about, and emits one row per
  ``(dataset_id, model, seed, ratio, alpha)``. No dataset features. Used
  for cross-model comparisons (``analyze_tabpfn_calibration``).

* :class:`EvalTable` builds on :class:`MetricsTable` for one alpha and joins
  cached dataset-level meta-features. Used for within-model
  feature-explains-error analyses (``dataset_level_analysis``).

* :class:`InstanceTable` walks every instance-features PKL written by
  :class:`InstanceFeaturePipeline` (feature columns only) and concatenates
  them into one long DataFrame, joining outcomes from the metrics PKL and
  identifiers from the filename. Used by ``instance_level_analysis``.

Single source of truth for "metrics PKL → long DataFrame" — both LME / RF
and the cross-model comparison go through the same loader.

Top-level helpers
-----------------
:func:`slice_table_by` filters a long table to rows where one column equals
*value* (``None`` matches NaN). Used by the dataset-level pipeline for
``ratio`` / ``n_estimators`` slicing.

:meth:`EvalTable.list_ne_values` and :meth:`EvalTable.slice_by_ne` enumerate
and slice the ``n_estimators`` dimension of an :class:`EvalTable` output.

:func:`build_long_table` sweeps a ``{n_estimators: root_path}`` mapping and
concatenates per-ne DataFrames into one long table suitable for multi-ne
analysis.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import ClassVar, Iterable, Optional

import numpy as np
import pandas as pd

from ..metrics import (
    AXIS_RELATIVE_METRICS,
    PAIR_DELTA_ABSOLUTE_METRICS,
    RESPONSE_COLS,
    metric_direction,
)
from ..metrics.directions import (
    CATEGORY_ORDER,
    RESPONSE_CATEGORY_METRICS,
    metric_category,
    metric_scale,
)
from ..spec import ExperimentSpec, TASK_CLASSIFICATION, TASK_REGRESSION
from ..store import ArtifactStore


_BASE_KEYS = (
    "n_total", "n_train", "n_test", "n_context", "n_features",
)

# Keys excluded from the automatic _scalarize_per_dataset pass.
# Only non-metric metadata that should not appear as a float column in
# the long table belongs here.  String values are already auto-skipped by
# _scalarize_per_dataset, so only integer metadata needs to be listed.
_METADATA_SKIP: frozenset[str] = frozenset({
    "worst_slab_feature_idx", "wsc_search_n", "wsc_eval_n",
})


def _scalarize_per_dataset(
    m: dict,
    skip: frozenset[str] = frozenset(),
) -> dict:
    """Promote every numeric scalar in a per_dataset dict into the long row.

    Mirrors the contract enforced by BaseMetricsCalculator._merge_into:
    any Metric subclass that writes a key into per_dataset automatically
    becomes a column in the long table — no need to modify tables.py.

    * Keys in ``skip`` are dropped (integer metadata that should not be
      treated as a continuous metric column).
    * Non-numeric values (str / None / bool) are dropped silently.
    * Numeric values are cast to float.
    """
    out: dict[str, float] = {}
    for k, v in m.items():
        if k in skip:
            continue
        if v is None or isinstance(v, (str, bool)):
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
    return out


def slice_table_by(
    df: pd.DataFrame,
    col: str,
    value,
    *,
    reset_index: bool = False,
) -> pd.DataFrame:
    """Return rows of *df* where ``df[col]`` equals *value*.

    If *value* is ``None``, matches NaN / missing entries in *col*.
    """
    if df.empty:
        return df.copy()
    if col not in df.columns:
        raise KeyError(f"slice_table_by: column {col!r} not in DataFrame")
    if value is None:
        mask = df[col].isna()
    else:
        mask = df[col] == value
    out = df.loc[mask]
    if reset_index:
        out = out.reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MetricsTable — loader, no features
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTable:
    """Build a long DataFrame from every metrics PKL in *store*.

    Output schema
    ~~~~~~~~~~~~~
    Regression — one row per (dataset, model, seed, ratio, alpha)::

        dataset_id, model, seed, ratio, n_estimators,
        alpha, nominal_coverage,
        n_total, n_train, n_test, n_context, n_features,
        <all numeric scalars from alpha_dependent[alpha]["per_dataset"]>,
        <all numeric scalars from alpha_free["per_dataset"]>,
        _path,

    Classification — one row per (dataset, model, seed, ratio); no alpha::

        dataset_id, model, seed, ratio, n_estimators,
        n_total, n_train, n_test, n_context, n_features,
        <all numeric scalars from alpha_free["per_dataset"]>,
        _path,

    Integer metadata keys listed in ``_METADATA_SKIP`` (e.g.
    ``worst_slab_feature_idx``) are excluded from the automatic pass.
    String values are always skipped.

    Every numeric scalar in ``per_dataset`` (both alpha-dependent and
    alpha-free) is automatically included via :func:`_scalarize_per_dataset`.
    Adding a new ``Metric`` subclass to the default list requires zero changes
    here.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def build(
        self,
        *,
        task: str = TASK_REGRESSION,
        n_estimators: Optional[int] = None,
        alphas: Optional[Iterable[float]] = None,
        spec_filter: Optional[Iterable[ExperimentSpec]] = None,
        models: Optional[Iterable[str]] = None,
        allowed_dataset_ids: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        keep_alphas = set(map(float, alphas)) if alphas is not None else None
        keep_ids = (
            set(map(int, allowed_dataset_ids))
            if allowed_dataset_ids is not None else None
        )

        rows: list[dict] = []
        # Numeric-scalar keys observed in any PKL's ``alpha_free.per_dataset``
        # slot.  Attached to the returned frame as ``df.attrs[...]`` so that
        # downstream consumers (per-dataset summary, plotting, etc.) can tell
        # alpha-free metrics from alpha-dependent ones without maintaining a
        # parallel hardcoded list — the PKL schema (see metrics/base.py) is
        # the single source of truth.
        alpha_free_keys: set[str] = set()

        for spec, model, r in self.store.iter_metrics(
            specs=spec_filter, models=models, task=task,
        ):
            if keep_ids is not None and int(spec.dataset_id) not in keep_ids:
                continue
            base = {
                "dataset_id":   spec.dataset_id,
                "model":        model,
                "seed":         spec.seed,
                "ratio":        spec.ratio,
                **{k: r.get(k) for k in _BASE_KEYS},
                "_path": str(self.store.metrics_path(spec, model, task=task)),
            }
            if n_estimators is not None:
                base["n_estimators"] = n_estimators

            af_scalars = _scalarize_per_dataset(
                r.get("alpha_free", {}).get("per_dataset", {})
            )
            alpha_free_keys.update(af_scalars.keys())

            if task == TASK_CLASSIFICATION:
                # Classification: all metrics live in alpha_free; no alpha
                # dimension — omit alpha/nominal_coverage entirely so that
                # downstream has_alpha checks (RelTable, _avg_key_cols_for)
                # correctly treat this as alpha-free.
                rows.append({**base, **af_scalars})

            else:
                # Point-only regressors contribute once to alpha-free tables.
                if not r.get("alpha_dependent") and r.get("distribution_status") == "not_requested":
                    rows.append({**base, "alpha": np.nan, "nominal_coverage": np.nan, **af_scalars})
                # Regression: alpha-free scalars broadcast to every alpha row.
                for alpha, slot in r.get("alpha_dependent", {}).items():
                    a = float(alpha)
                    if keep_alphas is not None and a not in keep_alphas:
                        continue
                    m = slot.get("per_dataset", {})
                    rows.append({
                        **base,
                        "alpha":            a,
                        "nominal_coverage": float(1.0 - a),
                        **_scalarize_per_dataset(m, skip=_METADATA_SKIP),
                        **af_scalars,
                    })

        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        out.attrs["alpha_free_metrics"] = frozenset(alpha_free_keys)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# EvalTable — MetricsTable + dataset features join (single alpha)
# ─────────────────────────────────────────────────────────────────────────────

class EvalTable:
    """Long DataFrame with one row per
    ``(dataset, model, seed, ratio[, alpha])``.

    Joins the :class:`MetricsTable` output with the cached dataset-level
    meta-features.

    For regression, ``alphas`` is required (a list of one or more
    miscoverage levels in (0, 1)) and the build is restricted to those
    levels.  When multiple alphas are passed, the long table has one
    row per ``(dataset, model, seed, ratio, alpha)``; alpha-free metric
    columns are broadcast across alpha rows (see
    :class:`MetricsTable`).  For classification, ``alphas`` is ignored
    (classification metrics live entirely in ``alpha_free``).
    """

    def __init__(
        self,
        store: ArtifactStore,
        alphas: Optional[list[float]] = None,
        *,
        task: str = TASK_REGRESSION,
    ) -> None:
        if task == TASK_REGRESSION:
            if not alphas:
                raise ValueError(
                    "alphas must be a non-empty list for regression."
                )
            bad = [a for a in alphas if not (0.0 < float(a) < 1.0)]
            if bad:
                raise ValueError(
                    f"every alpha must be in (0, 1); got {bad}"
                )
        elif task != TASK_CLASSIFICATION:
            raise ValueError(
                f"task must be {TASK_REGRESSION!r} or "
                f"{TASK_CLASSIFICATION!r}; got {task!r}"
            )
        self.store  = store
        self.alphas = [float(a) for a in alphas] if alphas else None
        self.task   = task

    def build(
        self,
        spec_filter: Optional[Iterable[ExperimentSpec]] = None,
        *,
        allowed_dataset_ids: Optional[Iterable[int]] = None,
    ) -> tuple[pd.DataFrame, int]:
        """Return ``(df, n_missing_features)``."""
        if self.task == TASK_REGRESSION:
            metrics_df = MetricsTable(self.store).build(
                task=self.task,
                alphas=self.alphas,
                spec_filter=spec_filter,
                allowed_dataset_ids=allowed_dataset_ids,
            )
        else:
            metrics_df = MetricsTable(self.store).build(
                task=self.task,
                spec_filter=spec_filter,
                allowed_dataset_ids=allowed_dataset_ids,
            )
        if metrics_df.empty:
            return metrics_df, 0

        feature_cache: dict[tuple[int, int], dict] = {}
        n_missing = 0

        feature_rows: list[dict] = []
        for _, row in metrics_df.iterrows():
            cache_key = (int(row["dataset_id"]), int(row["seed"]))
            if cache_key not in feature_cache:
                spec = ExperimentSpec(
                    cache_key[0], cache_key[1], float(row["ratio"]),
                )
                feats = self.store.load_dataset_features(spec)
                if feats is None:
                    n_missing += 1
                feature_cache[cache_key] = feats or {}
            feature_rows.append(feature_cache[cache_key])

        feats_df = pd.DataFrame(feature_rows).reset_index(drop=True)
        df = pd.concat(
            [metrics_df.reset_index(drop=True), feats_df],
            axis=1,
        )
        # Resolve duplicate columns (e.g. "n_features") in favour of metrics PKL.
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        n_features = df["n_features"].astype(float)
        n_ctx      = df["n_context"].astype(float)
        df["context_to_feature_ratio"] = np.where(
            n_features > 0, n_ctx / n_features, np.nan,
        )

        # Drop the metadata column that's only useful for debugging.
        df = df.drop(columns=["_path"], errors="ignore")

        # Preserve alpha_free_metrics marker set from the underlying
        # MetricsTable so AvgTable/RelTable and the pipeline can tell
        # which columns are alpha-independent.
        df.attrs = dict(metrics_df.attrs)
        return df, n_missing

    @staticmethod
    def list_ne_values(eval_long: pd.DataFrame) -> list:
        """Return the list of ``n_estimators`` values to iterate.

        Finite values come first (sorted ascending).  Adds ``None`` at
        the end if any rows have NaN / missing ``n_estimators``
        (legacy PKLs).  If the column is absent entirely, returns
        ``[None]`` so the pipeline still produces a single
        ``ne_unknown/<task>/`` subtree.
        """
        if "n_estimators" not in eval_long.columns:
            return [None]
        ne = eval_long["n_estimators"]
        unique_finite = sorted(ne.dropna().unique().tolist())
        out: list = list(unique_finite)
        if ne.isna().any():
            out.append(None)
        return out

    @staticmethod
    def slice_by_ne(eval_long: pd.DataFrame, ne) -> pd.DataFrame:
        """Return the rows of *eval_long* belonging to this ``n_estimators``."""
        if "n_estimators" not in eval_long.columns:
            return eval_long.copy()
        return slice_table_by(
            eval_long, "n_estimators", ne, reset_index=True,
        )

    # Sentinel used by :meth:`slice_by_alpha` to request the deduplicated
    # alpha-free row set (all alpha-broadcast columns are constant across
    # the alpha dimension, so picking any single alpha gives the right
    # ``(dataset, model, seed, ratio)`` population).
    ALPHA_FREE: ClassVar[str] = "alpha_free"

    @staticmethod
    def slice_by_alpha(
        df: pd.DataFrame,
        alpha,
        *,
        dedup_alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return the rows of *df* for the given alpha selector.

        Three modes, dispatched on the *alpha* argument:

        * ``alpha=<float>`` — filter rows to ``df["alpha"] == alpha``.
        * ``alpha="alpha_free"`` (== :attr:`EvalTable.ALPHA_FREE`) —
          return rows for ``dedup_alpha`` only.  Alpha-free metric
          columns are broadcast across alphas in :class:`MetricsTable`,
          so any single alpha's row set is the right deduplicated
          ``(dataset, model, seed, ratio)`` population.
        * ``alpha=None`` — no alpha dimension (classification).  Return
          *df* unchanged.

        When the input has no ``alpha`` column at all (e.g. classification
        eval tables), the dataframe is returned unchanged regardless of
        *alpha*.

        This method only filters rows.  Choosing which response *columns*
        to feed downstream (alpha-free vs alpha-dependent) is a separate
        analysis-orchestration concern handled by the caller.
        """
        if alpha is None or "alpha" not in df.columns:
            return df

        if alpha == EvalTable.ALPHA_FREE:
            if dedup_alpha is None:
                raise ValueError(
                    "slice_by_alpha: dedup_alpha must be provided when "
                    "alpha=EvalTable.ALPHA_FREE."
                )
            return df.loc[(df["alpha"] == float(dedup_alpha)) | df["alpha"].isna()]

        return df.loc[df["alpha"] == float(alpha)]


# ─────────────────────────────────────────────────────────────────────────────
# Seed-averaged + peer-relative z table builders
# ─────────────────────────────────────────────────────────────────────────────

_AVG_KEY_COLS: tuple[str, ...] = ("dataset_id", "model", "ratio")


def _avg_key_cols_for(df: pd.DataFrame) -> list[str]:
    """Extend the base key cols with ``alpha`` when present in *df*.

    Multi-alpha eval tables carry an ``alpha`` column (one row per
    ``(dataset, model, seed, ratio, alpha)``). Aggregation must happen
    *within* an alpha so alpha-dependent metric values stay distinct.
    """
    keys = list(_AVG_KEY_COLS)
    if "alpha" in df.columns:
        keys.append("alpha")
    return keys


class AvgTable:
    """Seed-averaged eval table — one row per
    ``(dataset, model, ratio[, alpha])``.

    The raw long table written by :class:`EvalTable` has one row per
    ``(dataset, seed, ratio, model[, alpha])``. Multiple seeds of the same
    dataset are **not** independent observations from the perspective of
    feature-importance / correlation analyzers: they share dataset
    identity, so treating them as iid inflates apparent sample size,
    biases CV scores upward (same ``dataset_id`` ends up in train and
    test of a non-grouped fold), and breaks the variance assumption of
    permutation p-values.

    This builder collapses the seed replicates: the response and feature
    columns are mean-aggregated across seeds (NaN-aware); other
    (non-averaged) columns keep the first-seed value as metadata; the
    ``seed`` column is dropped. When the input carries an ``alpha``
    column (multi-alpha mode), it is included as a grouping key so
    alpha-dependent metric values stay distinct per alpha.

    Configuration goes in ``__init__`` (response / feature column lists,
    stable across calls); the raw long DataFrame is passed to
    :meth:`build`. Mirrors the :class:`EvalTable` / :class:`MetricsTable`
    style.
    """

    def __init__(
        self,
        response_cols: list[str],
        feature_cols: list[str],
    ) -> None:
        self.response_cols = list(response_cols)
        self.feature_cols  = list(feature_cols)

    def build(self, eval_long: pd.DataFrame) -> pd.DataFrame:
        if eval_long.empty:
            out = eval_long.iloc[:0].copy()
            out.attrs = dict(eval_long.attrs)
            return out

        key_cols = _avg_key_cols_for(eval_long)
        missing = [c for c in key_cols if c not in eval_long.columns]
        if missing:
            raise KeyError(
                f"AvgTable.build: input is missing key columns {missing}"
            )

        avg_cols = [
            c for c in self.response_cols + self.feature_cols
            if c in eval_long.columns
        ]
        if not avg_cols:
            out = (
                eval_long[key_cols]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            out.attrs = dict(eval_long.attrs)
            return out

        grouped = eval_long.groupby(key_cols, dropna=False, as_index=False)
        avg_part = grouped[avg_cols].mean(numeric_only=True)
        other_cols = [
            c for c in eval_long.columns
            if c not in key_cols and c not in avg_cols and c != "seed"
        ]
        if other_cols:
            warnings.warn(
                "AvgTable.build: columns not in response_cols or feature_cols "
                f"are aggregated with first() across seeds, not mean(): "
                f"{other_cols}",
                UserWarning,
                stacklevel=2,
            )
            meta = grouped[other_cols].first()
            out = avg_part.merge(meta, on=key_cols, how="left")
        else:
            out = avg_part

        ordered = (
            key_cols
            + [c for c in self.response_cols if c in out.columns]
            + [c for c in self.feature_cols  if c in out.columns]
            + [c for c in out.columns
               if c not in key_cols
               and c not in self.response_cols
               and c not in self.feature_cols]
        )
        out = out[ordered].reset_index(drop=True)
        # groupby loses .attrs — re-attach so alpha_free_metrics survives.
        out.attrs = dict(eval_long.attrs)
        return out


class _PeerRelativeTable:
    """Base class for peer-relative z-scored table builders.

    Subclasses define the row identity (``required_cols`` /
    ``base_pivot_index``) and the response semantics
    (:meth:`_direction_for`). The shared :meth:`build` implementation handles
    peer filtering, per-peer z-scoring, alpha-aware grouping, and column drop
    for responses whose direction is ``0``.
    """

    required_cols: ClassVar[tuple[str, ...]]
    base_pivot_index: ClassVar[tuple[str, ...]]

    def __init__(
        self,
        response_cols: list[str],
        peer_models: list[str],
    ) -> None:
        self.response_cols = list(response_cols)
        self.peer_models = list(peer_models)

    @classmethod
    def _direction_for(cls, resp: str) -> int:
        """Return ``+1`` / ``-1`` for kept responses, ``0`` to drop."""
        raise NotImplementedError

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            out = df.iloc[:0].copy()
            out.attrs = dict(df.attrs)
            return out

        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"{type(self).__name__}.build: input is missing key "
                f"columns {missing}"
            )

        # Per-(dataset, ratio[, seed][, alpha]) peer z-score: include alpha in
        # the pivot index so alpha-dependent metric values stay separated.
        pivot_index = list(self.base_pivot_index)
        merge_keys = [*pivot_index, "model"]
        if "alpha" in df.columns:
            pivot_index.append("alpha")
            merge_keys.append("alpha")

        peer_set = set(self.peer_models)
        out = df.loc[df["model"].isin(peer_set)].reset_index(drop=True)
        if out.empty:
            out.attrs = dict(df.attrs)
            return out

        for resp in self.response_cols:
            if resp not in out.columns:
                continue
            direction = int(type(self)._direction_for(resp))
            if direction == 0:
                out = out.drop(columns=[resp])
                continue

            pivot = out.pivot_table(
                index=pivot_index,
                columns="model",
                values=resp,
                aggfunc="first",
            )
            peers_present = [
                m for m in self.peer_models if m in pivot.columns
            ]
            if len(peers_present) < 2:
                out[resp] = np.nan
                continue
            sub = pivot[peers_present]
            mu = sub.mean(axis=1)
            sd = sub.std(axis=1, ddof=0).replace(0.0, np.nan)
            z = sub.sub(mu, axis=0).div(sd, axis=0) * direction

            z_long = z.stack(future_stack=True).rename(resp).reset_index()
            out = out.drop(columns=[resp]).merge(
                z_long, on=merge_keys, how="left",
            )

        out.attrs = dict(df.attrs)
        return out


class _AxisRelativeSemantics:
    """Response semantics for axis-inclusive peer z-scores."""

    @classmethod
    def _direction_for(cls, resp: str) -> int:
        d = metric_direction(resp)
        if d != 0:
            return d
        if resp in AXIS_RELATIVE_METRICS:
            return -1
        return 0


class _BiasRelativeSemantics:
    """Response semantics for raw peer z-scores without sign flipping."""

    @classmethod
    def _direction_for(cls, resp: str) -> int:
        return +1


class RelTable(_AxisRelativeSemantics, _PeerRelativeTable):
    """Axis-inclusive peer-relative z-scored eval table.

    For each ``(dataset_id, ratio, response)`` we stack the values from
    ``peer_models`` into a row, compute

        z = (v - mean_across_peers) / std_across_peers

    and multiply by the response's relative-analysis direction. Proper /
    monotone metrics use :func:`evaluation.metrics.metric_direction`; single
    axis lower-better metrics are also included with a lower-better sign flip.
    Positive values therefore mean better for proper / monotone metrics, and
    smaller / more favourable on the named axis for axis-only metrics. Signed
    bias metrics remain excluded because their optimum is at 0.

    Features pass through unchanged: they depend on ``(dataset, ratio)``,
    not on the model — there is no peer dimension to standardize
    against. Rows for models not in ``peer_models`` are filtered out.
    """

    required_cols: ClassVar[tuple[str, ...]] = _AVG_KEY_COLS
    base_pivot_index: ClassVar[tuple[str, ...]] = ("dataset_id", "ratio")


class RelBiasTable(_BiasRelativeSemantics, RelTable):
    """Raw peer-relative z-scored eval table.

    Includes configured responses without a quality / axis sign flip. Positive
    values mean the raw response is larger than peers. For signed coverage
    deviations this means more over-covering / less under-covering; for
    lower-better metrics it means worse on that raw metric.
    """


class RelEvalTable(RelTable):
    """Axis-inclusive seed-level peer-relative z-scored eval table.

    The seed-aware counterpart of :class:`RelTable`. Built directly from
    the long ``eval_long`` table — *not* from the seed-averaged
    ``eval_avg`` — so the peer reference distribution for each row only
    contains values from the **same seed**. This avoids contaminating
    the reference std with peers' own seed-to-seed noise (which would
    shrink z and underestimate significance).

    Output schema: one row per ``(dataset_id, seed, ratio[, alpha], model)``
    with the response columns replaced by ``z = (v - μ_peers_same_seed)
    / σ_peers_same_seed * metric_direction(resp)``. Features pass
    through unchanged.

    Used by the per-seed two-stage analyzers (Chatterjee, Spearman, RF,
    Univariate RF) under ``input_kinds = ("long_abs", "long_rel")``.
    """

    required_cols: ClassVar[tuple[str, ...]] = ("dataset_id", "seed", "ratio")
    base_pivot_index: ClassVar[tuple[str, ...]] = (
        "dataset_id", "seed", "ratio",
    )


class RelBiasEvalTable(_BiasRelativeSemantics, RelEvalTable):
    """Raw seed-level peer-relative z-scored eval table."""


# ─────────────────────────────────────────────────────────────────────────────
# PairDeltaTable — per-seed signed delta between two models
# ─────────────────────────────────────────────────────────────────────────────

class PairDeltaTable:
    """Per-seed eval table whose response is the signed pair delta between two models.

    For each ``(dataset_id, seed, ratio[, alpha])`` slice with both models present (matched
    on the same seed), every configured response is replaced by a signed delta whose form
    depends on the metric:

        relative (default):  delta = sign * (M_b - M_a) / M_a
        absolute:            delta = sign * (M_b - M_a)

    The *absolute* (raw difference) form is used for metrics in
    :data:`evaluation.metrics.PAIR_DELTA_ABSOLUTE_METRICS` — scale-free, bounded accuracy
    metrics (``r2``, ``accuracy``, …) where the relative form's ``/ M_a`` blows up whenever
    the baseline ``M_a`` approaches 0 (tiny-scale target or near-perfect baseline). All
    other metrics keep the relative form.

    ``sign(<metric>) = metric_direction(<metric>)`` (``+1`` for higher-better, ``-1`` for
    lower-better) so that positive values consistently mean *model_b is better*.
    Responses whose direction is ``0`` (bidirectional or unregistered) are skipped.

    The output ``model`` column is stamped with a single pseudo name so downstream
    analyzers (RF, LME, …) iterate it as a single model via their ``models=[...]`` arg.
    Meta-feature columns pass through unchanged (they depend on ``(dataset, ratio[, alpha])``,
    not on the model — we copy them from ``model_a``'s rows at the matching index).

    Row identity matches :class:`EvalTable` / :class:`RelEvalTable`: per-seed rows, **no
    seed pre-aggregation** — :class:`RFImportanceAnalyzer` fits one model per seed and
    aggregates importance across seeds via :class:`MeanSDAggregator`.
    """

    def __init__(
        self,
        model_a: str,
        model_b: str,
        response_cols: list[str],
        feature_cols: Optional[list[str]] = None,
        pseudo_model_name: Optional[str] = None,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.response_cols = list(response_cols)
        self.feature_cols = (
            list(feature_cols) if feature_cols is not None else None
        )
        self.pseudo_model_name = (
            pseudo_model_name if pseudo_model_name is not None
            else f"{model_b}_minus_{model_a}"
        )

    def build(self, eval_long: pd.DataFrame) -> pd.DataFrame:
        if eval_long.empty:
            out = eval_long.iloc[:0].copy()
            out.attrs = dict(eval_long.attrs)
            return out

        required = ("dataset_id", "seed", "ratio", "model")
        missing = [c for c in required if c not in eval_long.columns]
        if missing:
            raise KeyError(
                f"PairDeltaTable.build: input is missing column(s) {missing}"
            )

        sub = eval_long.loc[
            eval_long["model"].isin([self.model_a, self.model_b])
        ]
        if sub.empty:
            out = sub.iloc[:0].copy()
            out.attrs = dict(eval_long.attrs)
            return out

        pivot_index = ["dataset_id", "seed", "ratio"]
        if "alpha" in sub.columns:
            pivot_index.append("alpha")

        delta_frames: list[pd.Series] = []
        kept_metrics: list[str] = []
        for resp in self.response_cols:
            if resp not in sub.columns:
                continue
            direction = metric_direction(resp)
            if direction == 0:
                # Bidirectional / unregistered → no sign convention available.
                continue
            pivot = sub.pivot_table(
                index=pivot_index,
                columns="model",
                values=resp,
                aggfunc="first",
            )
            if (self.model_a not in pivot.columns
                    or self.model_b not in pivot.columns):
                continue
            a = pivot[self.model_a]
            b = pivot[self.model_b]
            with np.errstate(divide="ignore", invalid="ignore"):
                if resp in PAIR_DELTA_ABSOLUTE_METRICS:
                    # Scale-free, bounded metric → raw difference. Avoids the
                    # division blow-up that ``/ M_a`` causes when the baseline
                    # value approaches 0.
                    delta = float(direction) * (b - a)
                else:
                    delta = float(direction) * (b - a) / a
            delta = delta.where(np.isfinite(delta))
            delta_frames.append(delta.rename(resp))
            kept_metrics.append(resp)

        if not delta_frames:
            out = sub.iloc[:0].copy()
            out.attrs = dict(eval_long.attrs)
            return out

        delta_df = pd.concat(delta_frames, axis=1).reset_index()
        delta_df = delta_df.dropna(subset=kept_metrics, how="all")
        if delta_df.empty:
            out = sub.iloc[:0].copy()
            out.attrs = dict(eval_long.attrs)
            return out

        feat_cols = self.feature_cols
        if feat_cols is None:
            metric_set = set(self.response_cols)
            base_cols = (
                set(pivot_index)
                | {"model", "seed", "_path", "n_estimators",
                   "nominal_coverage"}
                | metric_set
            )
            feat_cols = [c for c in eval_long.columns if c not in base_cols]
        feat_cols = [c for c in feat_cols if c in sub.columns]
        if feat_cols:
            feat_sub = (
                sub.loc[sub["model"] == self.model_a, pivot_index + feat_cols]
                .drop_duplicates(subset=pivot_index, keep="first")
            )
            delta_df = delta_df.merge(feat_sub, on=pivot_index, how="left")

        delta_df["model"] = self.pseudo_model_name
        ordered = (
            pivot_index
            + ["model"]
            + kept_metrics
            + [c for c in feat_cols if c in delta_df.columns]
        )
        delta_df = delta_df[ordered].reset_index(drop=True)
        delta_df.attrs = dict(eval_long.attrs)
        return delta_df


# ─────────────────────────────────────────────────────────────────────────────
# FitQualityTable — "can the meta-features predict each metric?" summary
# ─────────────────────────────────────────────────────────────────────────────

def _ordered_categorical(values, order: Iterable[str]) -> pd.Categorical:
    """Ordered Categorical: the known *order* first (those that occur), then any
    unexpected values in first-seen order."""
    seen = list(dict.fromkeys(values))
    present = [c for c in order if c in set(seen)]
    present += [v for v in seen if v not in present]
    return pd.Categorical(values, categories=present, ordered=True)


class FitQualityTable:
    """Pivot an analyzer's per-fit goodness-of-fit scalars into a compact,
    category-grouped table.

    A dataset-level analyzer (RF, LME, …) fits one model per
    ``(model, response, ratio, alpha)`` — X = dataset meta-features, y = the
    *metric value* across datasets — and records fit-quality scalars that are
    constant across the per-feature rows of that fit (RF: ``cv_r2_mean`` /
    ``train_r2``; LME: ``r2_marginal`` / ``r2_conditional``).  This builder
    de-duplicates those scalars and lays them out as:

      * rows: a MultiIndex ``(category, scale, response, alpha)`` — ``ratio`` is
        prepended only when more than one ratio is present;
      * columns: a MultiIndex ``(model, metric)``.

    Categories come from :func:`evaluation.metrics.metric_category` (intrinsic to
    the metric name — no task needed: the analyzer always *regresses a continuous
    metric* on the features).  Ordering: categories by :data:`CATEGORY_ORDER`,
    ``scale`` as ``scale_free`` then ``scale_dependent``, responses by their order
    in the canonical map, α ascending with alpha-free first, models sorted,
    metrics in ``value_columns`` order.

    Parameters
    ----------
    value_columns:
        Mapping ``{output_label: source_column}`` selecting which fit-quality
        scalars to pivot.  Defaults to RF's ``{"cv_r2": "cv_r2_mean", "train_r2":
        "train_r2"}``; LME passes ``{"r2_marginal": "r2_marginal",
        "r2_conditional": "r2_conditional"}``.
    """

    DEFAULT_VALUE_COLUMNS: ClassVar[dict[str, str]] = {
        "cv_r2": "cv_r2_mean",
        "train_r2": "train_r2",
    }
    _ID_COLS: ClassVar[tuple[str, ...]] = ("ratio", "alpha", "model", "response")
    _SCALE_ORDER: ClassVar[tuple[str, ...]] = ("scale_free", "scale_dependent")

    def __init__(self, value_columns: Optional[dict[str, str]] = None) -> None:
        self.value_columns = dict(value_columns or self.DEFAULT_VALUE_COLUMNS)

    def build(self, summary_long: pd.DataFrame) -> pd.DataFrame:
        sources = list(self.value_columns.values())
        required = set(self._ID_COLS) | set(sources)
        if (summary_long is None or getattr(summary_long, "empty", True)
                or not required.issubset(summary_long.columns)):
            return pd.DataFrame()

        fit = summary_long.drop_duplicates(list(self._ID_COLS))
        long = fit.melt(
            id_vars=list(self._ID_COLS),
            value_vars=sources,
            var_name="metric", value_name="value",
        )
        long["metric"] = long["metric"].map(
            {v: k for k, v in self.value_columns.items()}
        )
        long["category"] = [metric_category(r) for r in long["response"]]
        long["scale"] = [metric_scale(r) for r in long["response"]]
        long["alpha"] = long["alpha"].map(
            lambda a: "" if pd.isna(a) else f"{float(a):g}"
        )

        # Order every grouping key via an ordered Categorical so pivot_table lays
        # the table out directly — no manual rank columns / reindex.
        resp_order = [
            m
            for task_map in RESPONSE_CATEGORY_METRICS.values()
            for cat in CATEGORY_ORDER
            for m in task_map.get(cat, ())
        ]
        alpha_order = [""] + sorted(
            (a for a in set(long["alpha"]) if a), key=float
        )
        long["category"] = _ordered_categorical(long["category"], CATEGORY_ORDER)
        long["scale"] = _ordered_categorical(long["scale"], self._SCALE_ORDER)
        long["response"] = _ordered_categorical(long["response"], resp_order)
        long["alpha"] = pd.Categorical(
            long["alpha"], categories=alpha_order, ordered=True,
        )
        long["metric"] = pd.Categorical(
            long["metric"], categories=list(self.value_columns), ordered=True,
        )
        long["model"] = _ordered_categorical(
            long["model"], sorted(set(long["model"])),
        )

        row_levels = ["category", "scale", "response", "alpha"]
        if long["ratio"].nunique() > 1:
            row_levels = ["ratio"] + row_levels

        return long.pivot_table(
            index=row_levels, columns=["model", "metric"],
            values="value", aggfunc="first", observed=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# InstanceTable — per-test-point long DataFrame
# ─────────────────────────────────────────────────────────────────────────────

# Outcome columns that should never be used as predictors in downstream
# instance-level analyses. Used by InstanceAnalysisPipeline._resolve_feature_cols.
INSTANCE_OUTCOME_COLS: frozenset[str] = frozenset({
    "covered", "y_test", "winkler", "miscovered",
})

# Identifier columns added by InstanceTable at load time (from spec + params).
INSTANCE_ID_COLS: frozenset[str] = frozenset({
    "dataset_id", "seed", "ratio", "alpha", "model",
})

class InstanceTable:
    """Concat every instance-features PKL into one long DataFrame.

    Instance-features PKLs store *only* feature columns (40 columns from the
    four feature groups). Identifiers (``dataset_id / seed / ratio / model``)
    are recovered from the on-disk filename (via ``ExperimentSpec``), and
    outcome arrays (``covered / y_test / winkler``) are read straight from
    the corresponding metrics PKL — never recomputed here.

    This mirrors the dataset-level design where ``features_cache/*.pkl``
    holds only features and ``metrics/*.pkl`` holds only metrics.
    """

    def __init__(self, store: ArtifactStore, alpha: float) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.store = store
        self.alpha = float(alpha)

    def build(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        spec_filter: Optional[Iterable[ExperimentSpec]] = None,
        allowed_dataset_ids: Optional[Iterable[int]] = None,
    ) -> tuple[pd.DataFrame, int]:
        """Return ``(df, n_pkls_used)``.

        ``df`` is empty (length 0) when no usable PKL is found.
        """
        keep_models = set(models) if models is not None else None
        keep_specs  = set(spec_filter) if spec_filter is not None else None
        keep_ids    = set(map(int, allowed_dataset_ids)) if allowed_dataset_ids is not None else None

        frames: list[pd.DataFrame] = []
        n_pkls = 0
        for spec, model, df in self.store.iter_instance_features():
            if df is None or len(df) == 0:
                continue
            if keep_models is not None and model not in keep_models:
                continue
            if keep_specs is not None and spec not in keep_specs:
                continue
            if keep_ids is not None and int(spec.dataset_id) not in keep_ids:
                continue

            df = df.copy()

            # Identifiers — from filename / spec (single source of truth).
            df["dataset_id"] = int(spec.dataset_id)
            df["seed"]       = int(spec.seed)
            df["ratio"]      = float(spec.ratio)
            df["model"]      = model
            df["alpha"]      = self.alpha

            # Outcomes — from the metrics PKL (single source of truth).
            try:
                mts = self.store.load_metrics(spec, model)
            except FileNotFoundError:
                # No metrics PKL → skip this entry.
                continue
            iv = mts.get("alpha_dependent", {}).get(self.alpha, {}).get("per_instance")
            if iv is None:
                continue
            y_test  = np.asarray(mts["y_test"],  dtype=np.float64)
            covered = np.asarray(iv["covered"],  dtype=np.int8)
            winkler = np.asarray(iv["winkler"],  dtype=np.float64)

            # alpha-free per-instance arrays (may be absent for legacy PKLs)
            af_pi = mts.get("alpha_free", {}).get("per_instance", {})

            del mts  # free the (large) dict early

            if len(covered) != len(df):
                # Length mismatch between features and metrics → skip.
                continue

            df["covered"]    = covered.astype(int)
            df["y_test"]     = y_test
            df["miscovered"] = 1 - df["covered"]
            # Winkler interval score is the single source of truth in the
            # metrics PKL (see evaluation.metrics.IntervalArrays). Do NOT
            # recompute it here from pi_lower / pi_upper — same formula in
            # two places is exactly what we're trying to avoid.
            df["winkler"]    = winkler

            # Optional alpha-free per-instance columns (CRPS, PIT)
            for col, arr in af_pi.items():
                arr = np.asarray(arr)
                if arr.shape == (len(df),):
                    df[col] = arr

            frames.append(df)
            n_pkls += 1

        if not frames:
            return pd.DataFrame(), 0
        return pd.concat(frames, ignore_index=True, sort=False), n_pkls


# ─────────────────────────────────────────────────────────────────────────────
# build_long_table — multi-ne sweep helper
# ─────────────────────────────────────────────────────────────────────────────

def build_long_table(
    roots: dict[int, "str | Path"],
    *,
    task: str,
    models: Optional[Iterable[str]] = None,
    alphas: Optional[Iterable[float]] = None,
    allowed_dataset_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Build a long DataFrame sweeping across multiple ``n_estimators`` runs.

    Parameters
    ----------
    roots:
        Mapping ``{n_estimators: result_root_path}``.  Each path is the base
        directory passed to :class:`ArtifactStore`.
    task:
        ``'regression'`` or ``'classification'``.
    models:
        Optional whitelist of model names.
    alphas:
        Optional whitelist of alpha levels (regression only).
    allowed_dataset_ids:
        Optional whitelist of dataset IDs.

    Returns
    -------
    pd.DataFrame — all rows from all ne-roots concatenated; the
    ``n_estimators`` column identifies which root each row came from.
    """
    frames: list[pd.DataFrame] = []
    alpha_free_union: set[str] = set()
    for ne, root in roots.items():
        store = ArtifactStore(Path(root))
        df = MetricsTable(store).build(
            task=task,
            n_estimators=int(ne),
            models=models,
            alphas=alphas,
            allowed_dataset_ids=allowed_dataset_ids,
        )
        if not df.empty:
            alpha_free_union.update(df.attrs.get("alpha_free_metrics", ()))
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # pd.concat keeps only the first frame's attrs; replace with the union so
    # all alpha-free keys seen across n_estimators / shards are represented.
    out.attrs["alpha_free_metrics"] = frozenset(alpha_free_union)
    return out
