"""Base classes for analyzers registered with the dataset-level pipeline.

The pipeline (:class:`evaluation.pipelines.DatasetAnalysisPipeline`)
iterates a list of :class:`DatasetAnalyzer` instances and calls
``run`` → ``save`` uniformly.  Subclasses declare their identifier
(``name``) and which eval tables they consume (``input_kinds``); the
pipeline takes care of slicing inputs by ``ne`` / ``ratio`` / ``kind``
and providing each analyzer with a pre-namespaced ``out_dir``.

Per-feature importance reporting is **optional**: analyzers that
produce per-(model, response, feature) scores mix in the
:class:`FeatureImportance` interface and implement ``importance_long``.
The pipeline itself does **not** aggregate across analyzers — each
analyzer owns its own output schema and writes its own summary files
under its own subtree.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal, Mapping, Optional

import numpy as np
import pandas as pd


InputKind = Literal[
    "long_abs",
    "long_rel",
    "long_rel_bias",
    "avg",
    "rel",
    "rel_bias",
]
"""Which eval table an analyzer consumes.

* ``"long_abs"`` — :class:`evaluation.analysis.EvalTable` output
  (one row per dataset × seed × ratio × model).  Used by analyzers
  whose statistical model needs the un-aggregated replicate structure,
  including LME (built-in random-effect over seed) and the per-seed
  two-stage analyzers (Chatterjee / Spearman / RF / Univariate RF).
* ``"long_rel"`` — :class:`evaluation.analysis.RelEvalTable` output
  (peer-relative z-scored responses, computed per (dataset, seed) so the
  reference contains only same-seed peers). Includes proper / monotone
  metrics plus single-axis lower-better metrics.
* ``"long_rel_bias"`` — :class:`evaluation.analysis.RelBiasEvalTable`
  output. Includes configured responses as raw peer z-scores; positive means
  the raw response is larger than peers, not better.
* ``"avg"`` — :class:`evaluation.analysis.AvgTable` output
  (seed-averaged; one row per dataset × ratio × model).  Legacy path
  retained for backwards compatibility.
* ``"rel"`` — :class:`evaluation.analysis.RelTable` output
  (peer-relative z-scored responses, computed from ``eval_avg``). Includes
  proper / monotone metrics plus single-axis lower-better metrics.
* ``"rel_bias"`` — :class:`evaluation.analysis.RelBiasTable` output.
"""


class DatasetAnalyzer(ABC):
    """Common interface for every analyzer plugged into
    :class:`~evaluation.pipelines.DatasetAnalysisPipeline`.

    Minimal contract — only :meth:`run` and :meth:`save`.  Per-feature
    importance reporting is an *optional* capability provided via the
    :class:`FeatureImportance` mixin so analyzers whose goal isn't
    feature importance (e.g. variance partitioning, model comparison)
    don't have to fake it.

    Class attributes
    ----------------
    name
        Identifier used as the analyzer's subdirectory name under the
        per-(ne, task) output root (e.g. ``"lme"``, ``"spearman"``).
    input_kinds
        Which eval tables this analyzer consumes.  ``("long_abs",)`` for
        LME-style (random-effect handles seed natively);
        ``("long_abs", "long_rel")`` for the per-seed two-stage importance
        analyzers (Chatterjee / Spearman / RF / Univariate RF).
        Legacy ``("avg", "rel")`` is still recognised for backwards
        compatibility but no longer used by default.
        Subclasses set this as a class attribute.

    Pipeline invocation
    -------------------
    The pipeline calls :meth:`run` **once per (ne, kind, ratio, alpha)
    slice** and collects the results into
    ``{(ratio, alpha_label): results}`` for each ``(analyzer, kind)``
    cell.  ``alpha_label`` is either the literal string
    ``"alpha_free"`` (for the slice running alpha-free responses on a
    deduplicated alpha row), a ``float`` (alpha-dependent responses for
    that alpha), or ``None`` for classification (no alpha dimension).

    When the loop is done, the pipeline calls :meth:`save` **once per
    (analyzer, kind)** with the whole ``results_by_key`` dict.  The
    analyzer is then responsible for writing whatever output files it
    wants under the pre-namespaced ``out_dir`` — typically
    ``summary.csv`` + ``summary_ratio_<r>.html`` + an optional
    ``details/`` subfolder for bulky binaries.
    """

    name: ClassVar[str]
    input_kinds: ClassVar[tuple[InputKind, ...]]

    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        models: list[str],
        responses: list[str],
    ) -> object:
        """Run the analysis on one input slice.

        Parameters
        ----------
        df
            The (ne, kind, ratio)-sliced eval table — already filtered
            to the right rows by the pipeline.
        feature_cols
            Dataset-level meta-feature column names available in
            ``df``.
        models
            Model names to analyze (rows of ``df`` are filtered to
            these inside the analyzer's own loop).
        responses
            Response column names to analyze.

        Returns
        -------
        Analyzer-specific results object.  The pipeline treats it as
        opaque — only :meth:`save` and (optionally)
        :meth:`FeatureImportance.importance_long` need to know its
        shape.
        """

    @abstractmethod
    def save(
        self,
        results_by_key: "dict[tuple[float, str | float | None], object]",
        out_dir: Path,
    ) -> None:
        """Persist this analyzer's output for **all (ratio, alpha)
        slices** under ``out_dir`` (already pre-namespaced as
        ``…/<analyzer.name>/[<kind>/]``).

        Key format: ``(ratio, alpha_label)`` where ``alpha_label`` is
        the literal string ``"alpha_free"`` for the dedicated alpha-free
        slice (regression), a ``float`` for one alpha-dependent slice,
        or ``None`` for classification (no alpha dimension).

        Conventional layout the analyzer is expected to write:

        * ``out_dir/summary.csv`` — single long-format CSV covering
          every (ratio, alpha) combination, with ``ratio`` and ``alpha``
          columns (``alpha`` is NaN for alpha-free rows or for
          classification).  Canonical machine-readable storage.
        * ``out_dir/summary_ratio_<r>.html`` — per-ratio human-readable
          pivot rendered with
          :meth:`pandas.io.formats.style.Styler.to_html`.  For
          regression with multiple alphas, columns are a 3-level
          MultiIndex ``(alpha_label, response, model)`` rendered via
          :func:`render_feature_pivot_html` with ``alpha_col=...``.
          For classification (single alpha_label=None), columns stay
          2-level ``(response, model)``.
        * ``out_dir/details/<...>.{npz, json, …}`` — optional bulky
          per-(model, response, ratio, alpha) artefacts that don't fit
          the long CSV (RF OOF SHAP matrices, LME fit objects).

        No return value; the pipeline doesn't aggregate anything across
        analyzers.  Each analyzer fully owns its output schema.
        """


class FeatureImportance(ABC):
    """Mixin for analyzers that produce per-(model, response, feature)
    importance scores.

    The mixin's role is to expose a standardised long-format method
    that downstream consumers (plotting scripts, ad-hoc analyses,
    cross-analyzer comparison utilities) can call uniformly via
    ``isinstance(analyzer, FeatureImportance)``.

    The pipeline does **not** aggregate :meth:`importance_long` across
    analyzers — each analyzer is free to use its own output in its
    :meth:`~DatasetAnalyzer.save` body however it likes (typically to
    build the long ``summary.csv``).
    """

    @abstractmethod
    def importance_long(self, results: object) -> pd.DataFrame:
        """Return per-feature long-format rows for one (ne, kind, ratio)
        slice's worth of results.

        The DataFrame **must** include the columns ``model``,
        ``response``, ``feature``; any other columns are
        analyzer-specific.  The pipeline does not add a ``ratio``
        column — that's :meth:`~DatasetAnalyzer.save`'s job when it
        stacks across ratios.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Shared key helpers — used by every analyzer's ``save``
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_to_cell(alpha_label) -> float:
    """Convert an analyzer-save key's alpha part to a CSV-friendly float.

    ``"alpha_free"`` / ``None`` → NaN.  Floats pass through.  Strings of
    the form ``"alpha=0.05"`` are unwrapped to ``0.05``.
    """
    if alpha_label is None:
        return float("nan")
    if isinstance(alpha_label, str):
        if alpha_label == _ALPHA_FREE_LABEL:
            return float("nan")
        if alpha_label.startswith("alpha="):
            try:
                return float(alpha_label.split("=", 1)[1])
            except ValueError:
                return float("nan")
        return float("nan")
    try:
        return float(alpha_label)
    except (TypeError, ValueError):
        return float("nan")


def group_keys_by_ratio(
    results_by_key: "dict[tuple[float, object], object]",
) -> "dict[float, list[tuple[object, object]]]":
    """Group ``{(ratio, alpha_label): result}`` → ``{ratio: [(alpha, result), …]}``.

    Order within each ratio: ``"alpha_free"`` / ``None`` first, then
    ascending numeric alpha.
    """
    out: dict[float, list[tuple[object, object]]] = {}
    for (ratio, alpha_label), result in results_by_key.items():
        out.setdefault(float(ratio), []).append((alpha_label, result))
    def _sort_key(item: tuple[object, object]) -> tuple[int, float]:
        label = item[0]
        if label is None or label == _ALPHA_FREE_LABEL:
            return (0, 0.0)
        try:
            return (1, float(label))
        except (TypeError, ValueError):
            return (2, 0.0)
    for ratio in list(out.keys()):
        out[ratio] = sorted(out[ratio], key=_sort_key)
    return out


def alpha_col_for_html(
    pairs: "list[tuple[object, object]]",
) -> Optional[str]:
    """Return ``"alpha_label"`` when at least one key carries a float alpha,
    or ``None`` (use the 2-level pivot) when every key is alpha-free or None
    (classification).
    """
    for label, _ in pairs:
        if label is None or label == _ALPHA_FREE_LABEL:
            continue
        return "alpha_label"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTML-pivot helper (used by every analyzer's ``save``)
# ─────────────────────────────────────────────────────────────────────────────

def _significance_suffix(
    p: float,
    *,
    levels: tuple[tuple[float, str], ...],
) -> str:
    """Return the star suffix for the smallest threshold *p* falls under."""
    if not np.isfinite(p):
        return ""
    for threshold, stars in levels:
        if p < threshold:
            return stars
    return ""


def _pivot_cell_labels_with_stars(
    pivot: pd.DataFrame,
    pivot_p: pd.DataFrame,
    *,
    fmt: str,
    levels: tuple[tuple[float, str], ...],
    na_rep: str = "—",
) -> pd.DataFrame:
    """Build ``feature × (response, model)`` display strings ``fmt(r) + stars(p)``."""
    text = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=object)
    for idx in pivot.index:
        for col in pivot.columns:
            r = pivot.at[idx, col]
            if pd.isna(r) or not np.isfinite(r):
                text.at[idx, col] = na_rep
                continue
            p = pivot_p.at[idx, col]
            stars = _significance_suffix(float(p), levels=levels)
            text.at[idx, col] = fmt.format(float(r)) + stars
    return text


_ALPHA_FREE_LABEL = "alpha_free"


def _alpha_group_label(val) -> str:
    """Top-level column label used for the (alpha_label, response, model) MultiIndex.

    NaN / missing values become the literal ``"alpha_free"`` group;
    finite floats are rendered as ``f"alpha={value}"``.
    """
    if val is None:
        return _ALPHA_FREE_LABEL
    try:
        if pd.isna(val):
            return _ALPHA_FREE_LABEL
    except (TypeError, ValueError):
        pass
    if isinstance(val, str):
        return val if val == _ALPHA_FREE_LABEL else f"alpha={val}"
    return f"alpha={val}"


def _alpha_group_sort_key(label: str) -> tuple[int, float]:
    """Sort key: alpha_free first, then ascending numeric alpha."""
    if label == _ALPHA_FREE_LABEL:
        return (0, 0.0)
    try:
        return (1, float(label.split("=", 1)[1]))
    except (IndexError, ValueError):
        return (2, 0.0)


def render_feature_pivot_html(
    long_df: pd.DataFrame,
    out_path: Path,
    *,
    value_col: Optional[str] = None,
    value_cols: Optional[Mapping[str, str]] = None,
    caption: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    diverging: bool = True,
    fmt: str = "{:+.3f}",
    p_col: Optional[str] = None,
    star_levels: tuple[tuple[float, str], ...] = ((0.01, "**"), (0.05, "*")),
    alpha_col: Optional[str] = None,
    diverging_by_label: Optional[Mapping[str, bool]] = None,
) -> None:
    """Render a ``feature × (response, model)`` pivot of *value_col* to
    ``out_path`` as an HTML file using
    :meth:`pandas.io.formats.style.Styler.to_html`.

    Expected columns in ``long_df``: ``feature``, ``response``,
    ``model``, plus ``value_col`` (or every column listed in
    ``value_cols``).  Other columns are ignored.

    Parameters
    ----------
    value_col
        Which column to pivot into cells (e.g. ``"r"`` for Spearman,
        ``"xi"`` for Chatterjee, ``"coef"`` for LME). Mutually exclusive
        with ``value_cols``.
    value_cols
        Optional ``{display_label: column_name}`` mapping for rendering
        **multiple importance flavors as sub-columns** under each
        ``response``. When set, the pivot column index gains a
        ``metric`` level — final column structure is
        ``[(alpha_label,) response, metric, model]``. Each metric block
        is coloured on its own ``vmin/vmax`` (so blocks with very
        different magnitudes — e.g. permutation importance vs
        |SHAP| — stay readable). ``p_col`` is ignored in multi-metric
        mode.
    caption
        Caption rendered above the table.
    cmap
        Matplotlib colormap name passed to
        :meth:`pandas.io.formats.style.Styler.background_gradient`.
    vmin, vmax
        Colour-scale limits.  If ``None`` and ``diverging`` is true,
        they default to ``±max(|value|)`` so 0 stays at the neutral
        mid-point of a divergent palette; if ``diverging`` is false,
        they default to ``(0, max(value))``. In multi-metric mode the
        per-block scale uses the same rule.
    diverging
        Controls the auto-derived vmin/vmax above.  Set ``False`` for
        sequential metrics like Chatterjee ξ (non-negative) or
        Univariate CV-R².
    diverging_by_label
        Optional per-metric override of ``diverging`` (only used in
        multi-metric mode). Example:
        ``{"perm": True, "shap_abs": False}`` keeps permutation
        importance centered at 0 while |SHAP| uses a sequential scale.
    fmt
        Cell number format string.
    p_col
        Optional column with p-values aligned to ``value_col``.  When set,
        cell text becomes ``fmt(value) + stars(p)`` where ``star_levels``
        maps thresholds to suffixes (checked from smallest threshold
        first).  Background colour still uses the numeric ``value_col``.
        **Ignored in multi-metric mode.**
    star_levels
        ``((threshold, suffix), …)`` pairs, e.g.
        ``((0.01, \"**\"), (0.05, \"*\"))``.
    alpha_col
        Optional column name carrying an alpha label per row. When set,
        the pivot's columns gain an extra leading level ``alpha_label``
        where the top level is ``"alpha_free"`` for NaN/missing values
        and ``f"alpha={v}"`` otherwise. Top-level sort order:
        ``alpha_free`` first, then ascending alpha.
    """
    if long_df.empty:
        return

    # ── Resolve single- vs multi-metric mode ─────────────────────────────
    if value_cols is None and value_col is None:
        raise ValueError(
            "render_feature_pivot_html: provide either value_col or value_cols"
        )
    if value_cols is not None and value_col is not None:
        raise ValueError(
            "render_feature_pivot_html: pass value_col OR value_cols, not both"
        )
    if value_cols is None:
        value_cols = {value_col: value_col}
    metric_labels = list(value_cols.keys())
    multi_metric = len(metric_labels) > 1

    # ── Validate columns ─────────────────────────────────────────────────
    required = ["feature", "response", "model"] + list(value_cols.values())
    if p_col is not None and not multi_metric:
        required.append(p_col)
    if alpha_col is not None:
        required.append(alpha_col)
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        raise KeyError(
            f"render_feature_pivot_html: missing required columns {missing}"
        )

    # ── Reshape into pivot ───────────────────────────────────────────────
    if multi_metric:
        # Melt each value column into a single "_value" column with a
        # "_metric" label so the pivot gains a metric level.
        pieces: list[pd.DataFrame] = []
        keep_base = ["feature", "response", "model"]
        if alpha_col is not None:
            keep_base.append(alpha_col)
        for label, col in value_cols.items():
            sub = long_df[keep_base].copy()
            sub["_value"] = long_df[col].values
            sub["_metric"] = label
            pieces.append(sub)
        work_df = pd.concat(pieces, ignore_index=True)
        actual_value_col = "_value"
        metric_col_name = "_metric"
    else:
        work_df = long_df.copy()
        actual_value_col = list(value_cols.values())[0]
        metric_col_name = None

    if alpha_col is not None:
        work_df["_alpha_label"] = work_df[alpha_col].map(_alpha_group_label)
        pivot_cols = ["_alpha_label", "response"]
    else:
        pivot_cols = ["response"]
    if metric_col_name is not None:
        pivot_cols.append(metric_col_name)
    pivot_cols.append("model")

    pivot = work_df.pivot_table(
        index="feature",
        columns=pivot_cols,
        values=actual_value_col,
        aggfunc="first",
    )
    if pivot.empty:
        return

    # ── Sort column MultiIndex ───────────────────────────────────────────
    metric_rank = {m: i for i, m in enumerate(metric_labels)}
    has_alpha = alpha_col is not None
    has_metric = metric_col_name is not None

    def _col_sort_key(t):
        if has_alpha and has_metric:
            # (alpha_label, response, metric, model)
            return (
                _alpha_group_sort_key(t[0]),
                t[1], metric_rank.get(t[2], len(metric_labels)), t[3],
            )
        if has_alpha:
            return (_alpha_group_sort_key(t[0]), t[1], t[2])
        if has_metric:
            return (t[0], metric_rank.get(t[1], len(metric_labels)), t[2])
        return t

    cols_sorted = sorted(pivot.columns.tolist(), key=_col_sort_key)
    new_names = list(pivot.columns.names)
    # Prettify internal level labels.
    new_names = [
        "alpha" if n == "_alpha_label"
        else "importance" if n == "_metric"
        else n
        for n in new_names
    ]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(
        cols_sorted, names=new_names,
    ))

    # ── Coloring ─────────────────────────────────────────────────────────
    if not multi_metric:
        if vmin is None or vmax is None:
            arr = pivot.to_numpy(dtype=float)
            if not np.isfinite(arr).any():
                return
            if diverging:
                max_abs = float(np.nanmax(np.abs(arr)))
                vmax_use = vmax if vmax is not None else max(max_abs, 1e-6)
                vmin_use = vmin if vmin is not None else -vmax_use
            else:
                vmax_use = vmax if vmax is not None else float(np.nanmax(arr))
                vmin_use = vmin if vmin is not None else float(np.nanmin(arr))
        else:
            vmin_use, vmax_use = vmin, vmax

        styled = pivot.style.background_gradient(
            cmap=cmap, axis=None, vmin=vmin_use, vmax=vmax_use,
        )
    else:
        # Multi-metric: color each metric block on its own scale.
        styled = pivot.style
        metric_level_idx = 2 if has_alpha else 1
        for label in metric_labels:
            sub_cols = [
                c for c in pivot.columns if c[metric_level_idx] == label
            ]
            if not sub_cols:
                continue
            sub_arr = pivot[sub_cols].to_numpy(dtype=float)
            if not np.isfinite(sub_arr).any():
                continue
            label_diverging = (
                diverging_by_label.get(label, diverging)
                if diverging_by_label is not None else diverging
            )
            if label_diverging:
                max_abs = float(np.nanmax(np.abs(sub_arr)))
                _vmax = max(max_abs, 1e-6)
                _vmin = -_vmax
            else:
                _vmax = float(np.nanmax(sub_arr))
                _vmin = float(np.nanmin(sub_arr))
                if _vmax == _vmin:
                    _vmax = _vmin + 1e-6
            styled = styled.background_gradient(
                cmap=cmap, axis=None,
                vmin=_vmin, vmax=_vmax,
                subset=pd.IndexSlice[:, sub_cols],
            )

    # ── Cell text ────────────────────────────────────────────────────────
    if p_col is not None and not multi_metric:
        pivot_p = work_df.pivot_table(
            index="feature",
            columns=pivot_cols,
            values=p_col,
            aggfunc="first",
        )
        pivot_p = pivot_p.reindex(index=pivot.index, columns=pivot.columns)
        text = _pivot_cell_labels_with_stars(
            pivot, pivot_p, fmt=fmt, levels=star_levels,
        )
        for idx in pivot.index:
            styled = styled.format(
                {
                    col: (lambda _v, idx=idx, col=col: text.at[idx, col])
                    for col in pivot.columns
                },
                subset=pd.IndexSlice[idx, :],
            )
    else:
        styled = styled.format(fmt, na_rep="—")
    if caption:
        styled = styled.set_caption(caption)
    out_path.write_text(styled.to_html(), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Fit-quality summary renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_fit_quality_html(
    pivot: pd.DataFrame,
    out_path: Path,
    *,
    caption: str = "",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Render a fit-quality pivot to a heat-mapped HTML table.

    The pivot is built by
    :class:`evaluation.analysis.tables.FitQualityTable`.  R² higher = better →
    sequential green-to-red scale (default ``[0, 1]``) so negative cells (e.g.
    scale-dependent abs metrics) clamp to red and read as "unpredictable" at a
    glance.
    """
    if pivot is None or pivot.empty:
        return
    styled = pivot.style.background_gradient(
        cmap="RdYlGn", axis=None, vmin=vmin, vmax=vmax,
    ).format("{:.3f}", na_rep="—")
    if caption:
        styled = styled.set_caption(caption)
    Path(out_path).write_text(styled.to_html(), encoding="utf-8")
