"""Per-model feature preprocessing.

The four supported tabular-foundation-model families have *different*
expectations about the ``X`` they receive in ``fit`` / ``predict``:

* **TabICL v1 / v2**
    Has its own ``TransformToNumerical`` step (OrdinalEncoder for
    categorical/object/boolean columns, SimpleImputer for numeric columns)
    inside :class:`tabicl._sklearn.preprocessing`. We can hand it the raw
    DataFrame as-is.

* **TabPFN v2 / v2.5**
    Internally calls
    :func:`tabpfn.preprocessing.clean.fix_dtypes` which (a) runs
    ``convert_dtypes()`` on the input DataFrame and then (b) coerces every
    column it sees as ``"number"`` to ``float64``. When a column is
    *object-typed* but happens to look numeric, pandas will infer it as
    ``Int64Dtype`` / ``Float64Dtype`` and the subsequent ``astype("float64")``
    raises ``Cannot cast object dtype to float64``. Stamping non-numeric
    columns as ``category`` dtype before handing the frame off makes
    TabPFN take the ordinal-encoder branch instead, which is robust to
    mixed-type values. The category vocabulary is the union of the
    context+test categories so unseen test labels do not become ``NaN``
    after the dtype switch.

* **Mitra**
    :meth:`MitraClassifier.fit` immediately calls ``X = X.values``. With a
    DataFrame containing ``object`` / ``category`` columns, that yields an
    ``object`` ndarray. The internal preprocessor in
    ``autogluon.tabular.models.mitra._internal.data.preprocessor.Preprocessor``
    then attempts arithmetic (``compute_pre_nan_mean`` / feature-count
    scaling), which surfaces as
    ``TypeError: unsupported operand type(s) for /: 'str' and 'int'`` or
    ``can only concatenate str (not "NoneType") to str``. Mitra *does*
    impute NaN internally (``impute_nan_features_with_mean``), so missing
    values are fine — but every column needs to be float.

* **TabDPT**
    :meth:`tabdpt._estimator.fit` requires ``X`` to be a 2-D numpy array
    and pipes it through ``SimpleImputer(strategy="mean")``, which raises
    ``Cannot use mean strategy with non-numeric data: could not convert
    string to float: '...'`` whenever any column is non-numeric.

This module centralises the divergent fixes behind a single
:class:`FeaturePreprocessor` object that fits on the *context* set
(ensuring test data sees the same vocabulary), then transforms the test
set with the same encoder. Strategy selection is driven by the model
name registered in :data:`_MODEL_STRATEGY`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Strategy registry
# ─────────────────────────────────────────────────────────────────────────────

# Strategy keys:
#   "passthrough" – Has its own categorical-aware preprocessing; pass the
#                   DataFrame through unchanged.
#   "passthrough_fill_ctx_nan"
#                 – TabICL family (v1.1 classifier: "tabicl";
#                   v2 classifier + v2 regressor: "tabiclv2").
#                   Same as ``passthrough`` but first fills
#                   NaN with ``0.0`` in columns that are all-NaN on the
#                   *context* split (the same fill is also applied to those
#                   columns in X_test, so any NaN positions there become
#                   0.0 while existing real values are kept). This works
#                   around a tabicl internal alignment bug: its
#                   ``feature_mask`` is computed on the *un-encoded* X_test
#                   (shape ``n_features``), while
#                   ``UniqueFeatureFilter.features_to_keep_`` is fit on the
#                   *encoded* X_ctx after ``SimpleImputer`` drops all-NaN
#                   columns (shape ``n_features - k``). When X_test happens
#                   to contain an all-NaN column (typical for sparse
#                   features with adversarial random splits),
#                   ``EnsembleGenerator.transform`` does
#                   ``feature_mask[features_to_keep_]`` and raises
#                   ``IndexError: boolean index did not match indexed array
#                   along axis 0; size of axis is N but size of
#                   corresponding boolean axis is N-k``. Pre-filling the
#                   all-NaN-on-ctx columns has two effects that together
#                   defuse the bug:
#                   (a) X_ctx's column is no longer all-NaN, so tabicl's
#                       ``SimpleImputer`` does not drop it and
#                       ``features_to_keep_`` has the original
#                       ``n_features`` length, matching ``feature_mask``;
#                   (b) X_test's column is no longer all-NaN, so
#                       ``feature_mask`` is all-False on every column and
#                       short-circuits to ``None`` via the
#                       ``not np.any(feature_mask)`` check, skipping the
#                       buggy code path entirely.
#                   The column is still effectively ignored by tabicl
#                   downstream (``UniqueFeatureFilter`` drops it because
#                   the X_ctx column is now constant ``0.0``), so the
#                   final model input is equivalent to dropping the
#                   column - but the dataframe schema (column names /
#                   order / count) is preserved, which keeps PKL
#                   metadata aligned with ``DatasetLoader.load_for_spec``
#                   output. Also silences the downstream "Skipping
#                   features without any observed values" /
#                   "All-NaN slice encountered" warnings.
#                   Then :func:`_fill_test_all_nan_non_numeric_columns`
#                   float-encodes non-numeric columns that are all-NaN on
#                   ``X_test`` but observed on ``X_ctx`` (anneal ``bc`` on
#                   some seeds), avoiding sklearn ``OrdinalEncoder`` calling
#                   ``np.isnan`` on string ``categories_``.
#   "tabpfn"      – TabPFN family. Stamp object/string/boolean columns as
#                   pandas ``category`` dtype so TabPFN's internal cleaner
#                   ordinal-encodes them instead of trying to ``astype('float64')``.
#                   Additionally, the targeted patch
#                   :func:`_patch_tabpfn_ctx_all_nan_strings` pre-encodes
#                   ctx-all-NaN columns whose test contains non-numeric
#                   strings (e.g. OpenML dataset 2 "anneal" with ``bc`` /
#                   ``exptl`` only observed as ``'Y'`` in the test split).
#                   The patch is a no-op for every (dataset, seed) that
#                   previously produced a TabPFN pkl, so existing pkls
#                   are byte-identical; only previously-crashing combos
#                   gain a new pkl.
#   "numeric_df"  – Models that internally treat ``X`` as a numeric ndarray
#                   (Mitra, TabDPT). Ordinal-encode every non-numeric column
#                   on the *context* vocabulary; force the rest to float64.
_MODEL_STRATEGY: dict[str, str] = {
    "tabpfnv3.5": "passthrough",
    "causilo": "passthrough",
    "limix2": "passthrough",
    "tabfm": "passthrough",
    "tabpfnv3":     "tabpfn",
    "tabpfnv2.5":   "tabpfn",
    "tabpfnv2":     "tabpfn",
    "tabicl":       "passthrough_fill_ctx_nan",
    "tabiclv2":     "passthrough_fill_ctx_nan",
    "mitra":        "numeric_df",
    "tabdpt":       "numeric_df",
    "tabdpt1.3":    "numeric_array",
}

STRATEGIES: tuple[str, ...] = (
    "passthrough",
    "passthrough_fill_ctx_nan",
    "tabpfn",
    "numeric_df",
)


def strategy_for(model_name: str) -> str:
    """Return the preprocessing-strategy name for ``model_name``.

    Raises :class:`KeyError` if ``model_name`` is not registered. Adding
    a new model should be a one-line change to :data:`_MODEL_STRATEGY`.
    """
    try:
        return _MODEL_STRATEGY[model_name]
    except KeyError as exc:
        raise KeyError(
            f"FeaturePreprocessor: no preprocessing strategy registered "
            f"for model {model_name!r}. Known models: "
            f"{sorted(_MODEL_STRATEGY)}."
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_non_numeric_column(s: pd.Series) -> bool:
    """``True`` iff ``s`` is object/string/bool/category dtype.

    We treat pandas ``boolean`` / ``category`` as non-numeric so they get
    consistent ordinal-encoded codes regardless of how the OpenML loader
    happened to spell the dtype. ``pd.api.types.is_numeric_dtype`` returns
    ``False`` for ``CategoricalDtype`` and string/object dtypes already,
    so this is mostly a thin wrapper that also folds plain ``bool``
    columns into the categorical bucket.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(s):
        return True
    return not pd.api.types.is_numeric_dtype(s)


def _non_numeric_columns(*frames: pd.DataFrame) -> list[str]:
    """Names of columns that look non-numeric in *any* of ``frames``.

    Using the union ensures a column counted as numeric in the context
    set but object-typed in the test set still gets the right treatment.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for f in frames:
        for c in f.columns:
            if c in seen_set:
                continue
            if any(_is_non_numeric_column(g[c]) for g in frames if c in g.columns):
                seen.append(c)
                seen_set.add(c)
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ─────────────────────────────────────────────────────────────────────────────

def _float64_test_col_ctx_all_nan(raw_test: pd.Series) -> np.ndarray:
    """Coerce one ctx-all-NaN column's test split to ``float64``.

    * NaN → ``0.0`` (reserved for missing; ctx is filled the same way).
    * Numeric non-NaN → keep the numeric value.
    * Each distinct non-numeric string → ``1.0``, ``2.0``, ``3.0``, … in
      lexicographic order of the string labels (deterministic across runs).

    Used by the TabICL ``passthrough_fill_ctx_nan`` path only. TabPFN's
    minimal patch uses a single ``-1.0`` sentinel instead (see
    :func:`_patch_tabpfn_ctx_all_nan_strings`).
    """
    num_test = pd.to_numeric(raw_test, errors="coerce")
    out = num_test.fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    non_numeric_mask = (raw_test.notna() & num_test.isna()).to_numpy()
    if not non_numeric_mask.any():
        return out
    labels = raw_test.loc[non_numeric_mask].astype(str)
    for i, label in enumerate(sorted(labels.unique())):
        hit = non_numeric_mask & (raw_test.astype(str).to_numpy() == label)
        out[hit] = float(i + 1)
    return out


def _fill_ctx_all_nan_columns(
    X_ctx: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fill NaN with ``0.0`` in columns that are all-NaN on ``X_ctx``.

    For each ctx-all-NaN column:

    * ``X_ctx[c]`` (by construction all NaN) becomes a constant ``0.0``
      ``float64`` column.
    * ``X_test[c]`` is converted to ``float64`` via
      :func:`_float64_test_col_ctx_all_nan`: numeric-coercible cells keep
      their value, NaN → ``0.0``, and each distinct non-numeric string
      (e.g. ``'Y'``, ``'N'``) gets its own code ``1.0``, ``2.0``, …
      (sorted lexicographically). This deliberately differs from TabDPT's
      ``unknown_value=-1`` collapse — TabICL never saw these labels on
      ctx, but preserving per-label codes avoids conflating different
      test-only categories. Without encoding, OpenML categorical
      columns whose context split happens to be 100% missing (e.g.
      ``bc`` / ``exptl`` on dataset 2 "anneal") would either crash here
      (the legacy ``.astype(np.float64)`` raises ``Cannot cast object
      dtype to float64`` on a ``category`` dtype whose categories
      contain a non-numeric string) or be silently coerced to ``NaN`` by
      ``pd.to_numeric``, which is then filled to ``0.0`` and conflates
      "observed non-numeric label only seen at test time" with "missing
      value".

    The fill works around a tabicl internal alignment bug: its
    ``feature_mask`` is computed on the un-encoded ``X_test``
    (shape ``n_features``), while
    ``UniqueFeatureFilter.features_to_keep_`` is fit on the encoded
    ``X_ctx`` after ``SimpleImputer`` drops all-NaN columns
    (shape ``n_features - k``). When ``X_test`` happens to contain an
    all-NaN column (typical for sparse features with adversarial
    random splits), ``EnsembleGenerator.transform`` does
    ``feature_mask[features_to_keep_]`` and raises
    ``IndexError: boolean index did not match indexed array...``.
    Pre-filling defuses the bug:

    (a) ``X_ctx[c]`` is no longer all-NaN, so tabicl's ``SimpleImputer``
        does not drop it and ``features_to_keep_`` keeps the original
        ``n_features`` length, matching ``feature_mask``;
    (b) ``X_test[c]`` is no longer all-NaN, so ``feature_mask`` is all-
        False on every column and short-circuits to ``None`` via the
        ``not np.any(feature_mask)`` check, skipping the buggy path.

    Returns ``(X_ctx_out, X_test_out, filled_columns)``. If no columns
    are all-NaN on the context split, both inputs are returned
    unchanged (no copy) and ``filled_columns`` is empty.
    """
    nan_mask = X_ctx.isna().all(axis=0)
    filled = [c for c, is_nan in nan_mask.items() if bool(is_nan)]
    if not filled:
        return X_ctx, X_test, []
    X_ctx_out = X_ctx.copy()
    X_test_out = X_test.copy()
    for c in filled:
        # ctx is all NaN by construction; ``pd.to_numeric`` is used
        # instead of ``.astype(np.float64)`` because the latter fails on
        # ``category`` dtypes whose categories include a non-numeric
        # string (e.g. ``categories=['Y']`` with all values NaN).
        X_ctx_out[c] = (
            pd.to_numeric(X_ctx_out[c], errors="coerce")
            .astype(np.float64)
            .fillna(0.0)
        )
        X_test_out[c] = _float64_test_col_ctx_all_nan(X_test_out[c])
    return X_ctx_out, X_test_out, filled


def _float64_encode_ctx_non_numeric(raw_ctx: pd.Series) -> np.ndarray:
    """Map one non-numeric context column to ``float64`` (NaN → ``0.0``)."""
    num = pd.to_numeric(raw_ctx, errors="coerce")
    if raw_ctx.notna().any() and int(num.notna().sum()) == int(raw_ctx.notna().sum()):
        return num.fillna(0.0).to_numpy(dtype=np.float64)
    out = np.zeros(len(raw_ctx), dtype=np.float64)
    labels = raw_ctx.dropna().astype(str)
    for i, label in enumerate(sorted(labels.unique())):
        hit = raw_ctx.notna() & (raw_ctx.astype(str) == label)
        out[hit.to_numpy()] = float(i + 1)
    return out


def _fill_test_all_nan_non_numeric_columns(
    X_ctx: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Float-encode non-numeric columns that are entirely missing on ``X_test``.

    Complements :func:`_fill_ctx_all_nan_columns` for OpenML "anneal"
    (dataset 2) seeds where sparse flags (e.g. ``bc``) appear in the
    context split but not in the test split. TabICL's
    ``OrdinalEncoder.transform`` then hits ``np.isnan`` on string
    ``categories_`` inside ``sklearn.utils._encode._check_unknown``.

    Skips columns already converted to numeric by the ctx-all-NaN pass.
    """
    filled: list[str] = []
    for c in X_ctx.columns:
        if not X_test[c].isna().all():
            continue
        if X_ctx[c].isna().all():
            continue
        if not (
            _is_non_numeric_column(X_ctx[c])
            or _is_non_numeric_column(X_test[c])
        ):
            continue
        filled.append(c)
    if not filled:
        return X_ctx, X_test, []
    X_ctx_out = X_ctx.copy()
    X_test_out = X_test.copy()
    for c in filled:
        X_ctx_out[c] = _float64_encode_ctx_non_numeric(X_ctx_out[c])
        X_test_out[c] = pd.Series(
            np.zeros(len(X_test_out), dtype=np.float64),
            index=X_test_out.index,
        )
    return X_ctx_out, X_test_out, filled


def _patch_tabpfn_ctx_all_nan_strings(
    X_ctx: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Pre-encode ctx-all-NaN columns whose ``X_test`` contains non-numeric strings.

    Targeted, minimal patch for the OpenML "anneal" (dataset 2) failure
    mode: a column is 100% NaN on the context split but has a handful of
    non-numeric string values (e.g. ``'Y'``) on the test split. TabPFN
    classifies such a column as ``CONSTANT`` modality at fit time
    (``nunique(dropna=False) == 1``), so its ``OrdinalEncoder``
    (selected via ``dtype_include=['category', 'string']``) does **not**
    cover the column — it falls through the ``remainder`` (identity)
    transformer. At predict time the un-encoded ``'Y'`` reaches the
    final ``X_encoded.astype(np.float64)`` in
    ``process_text_na_dataframe`` and raises
    ``could not convert string to float: 'Y'``.

    The patch only fires for columns where **all three** conditions hold:

    1. ``X_ctx[c].isna().all()`` (the modality detection treats it as
       CONSTANT and skips encoding);
    2. ``X_test[c]`` is not all-NaN;
    3. ``X_test[c]`` contains at least one value that ``pd.to_numeric``
       cannot coerce (i.e. a genuine non-numeric string).

    For columns matching all three, we replace the column with a
    ``float64`` representation:

    * ``X_ctx[c]`` → all NaN (it already is, just float64-typed);
    * ``X_test[c]`` → numeric-coercible cells keep their numeric value,
      NaN cells stay NaN, **non-numeric strings become ``-1.0``**.

    All other columns are untouched and TabPFN sees byte-identical input
    to before — so any (dataset, seed, model) combination that
    previously produced a pkl still produces the same pkl. Only the
    previously-crashing combinations gain a new pkl.

    Returns ``(X_ctx_out, X_test_out, patched_columns)``.
    """
    nan_mask = X_ctx.isna().all(axis=0)
    candidates = [c for c, is_nan in nan_mask.items() if bool(is_nan)]
    if not candidates:
        return X_ctx, X_test, []

    patched: list[str] = []
    X_ctx_out: Optional[pd.DataFrame] = None
    X_test_out: Optional[pd.DataFrame] = None
    for c in candidates:
        raw_test = X_test[c]
        if raw_test.isna().all():
            continue  # ctx-all-NaN ∧ test-all-NaN: original TabPFN succeeds.
        num_test = pd.to_numeric(raw_test, errors="coerce")
        non_numeric_mask = (raw_test.notna() & num_test.isna()).to_numpy()
        if not non_numeric_mask.any():
            continue  # test values are all numeric (or NaN): original TabPFN succeeds.

        if X_ctx_out is None:
            X_ctx_out = X_ctx.copy()
            X_test_out = X_test.copy()
        X_ctx_out[c] = pd.Series(
            np.full(len(X_ctx_out), np.nan, dtype=np.float64),
            index=X_ctx_out.index,
        )
        X_test_out[c] = np.where(
            non_numeric_mask,
            -1.0,
            num_test.to_numpy(),
        ).astype(np.float64)
        patched.append(c)

    if X_ctx_out is None:
        return X_ctx, X_test, []
    return X_ctx_out, X_test_out, patched  # type: ignore[return-value]


def _to_tabpfn(
    X_ctx: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Stamp non-numeric columns as ``category`` dtype with a shared vocabulary.

    This puts TabPFN's :func:`fix_dtypes` on the OrdinalEncoder branch for
    those columns and avoids the ``astype('float64')`` failure on object
    columns that ``convert_dtypes()`` mis-classifies as numeric.

    All category values are normalised to **string** dtype so that the
    resulting :class:`~pandas.CategoricalDtype` never has ``bool`` or
    ``int`` categories.  A bool-dtype ``CategoricalDtype`` makes
    ``pd.api.types.is_bool_dtype(cat_dtype)`` return ``True``, which
    triggers ``sklearn >= 1.7`` ``check_array``'s early-conversion path
    (``_pandas_dtype_needs_early_conversion``).  That path calls
    ``df.astype(None)`` — equivalent to ``df.astype(float64)`` — and
    immediately raises ``Cannot cast object dtype to float64`` when any
    string-category column is present.

    Before stamping categories, ctx-all-NaN columns whose test contains
    non-numeric strings are pre-encoded to ``float64`` via
    :func:`_patch_tabpfn_ctx_all_nan_strings` (see that function's
    docstring for the exact trigger conditions). That patch is a no-op
    for every (dataset, seed) combination that previously produced a
    TabPFN pkl, so existing pkls remain byte-identical.

    Returns ``(X_ctx_out, X_test_out, patched_columns)``, where
    ``patched_columns`` lists the column names rewritten by the ctx-all-
    NaN-string patch (empty when the patch does not fire).
    """
    X_ctx, X_test, patched = _patch_tabpfn_ctx_all_nan_strings(X_ctx, X_test)

    cat_cols = _non_numeric_columns(X_ctx, X_test)
    if not cat_cols:
        return X_ctx, X_test, patched

    X_ctx_out = X_ctx.copy()
    X_test_out = X_test.copy()
    for c in cat_cols:
        combined = pd.concat([X_ctx_out[c], X_test_out[c]], axis=0, ignore_index=True)
        raw = combined.astype("object")
        cats = pd.Index(
            pd.Series(raw.dropna().unique()).astype(str).values
        )
        for df_out in (X_ctx_out, X_test_out):
            obj = df_out[c].astype("object")
            str_vals = obj.where(obj.isna(), other=obj.astype(str))
            df_out[c] = pd.Categorical(str_vals, categories=cats)
    return X_ctx_out, X_test_out, patched


def _to_numeric(
    X_ctx: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[OrdinalEncoder], list[str]]:
    """Ordinal-encode non-numeric columns + force numeric columns to float64.

    NaN is preserved (Mitra / TabDPT impute NaN internally). Test rows with
    a category that was unseen in the context set get the sentinel ``-1.0``
    (rather than NaN), matching scikit-learn's ``handle_unknown=
    "use_encoded_value"`` convention.
    """
    cat_cols = _non_numeric_columns(X_ctx, X_test)

    encoder: Optional[OrdinalEncoder] = None
    if cat_cols:
        encoder = OrdinalEncoder(
            dtype=np.float64,
            handle_unknown="use_encoded_value",
            # sklearn requires an int (or np.nan); after dtype=float64 the
            # encoded array still ends up as -1.0 in float space.
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        # Use object dtype so the encoder sees raw labels (not category codes).
        encoder.fit(X_ctx[cat_cols].astype("object"))

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if encoder is not None and cat_cols:
            encoded = encoder.transform(out[cat_cols].astype("object"))
            for i, c in enumerate(cat_cols):
                out[c] = encoded[:, i]
        # Numeric columns: coerce to float64 so the eventual ``.values`` is
        # a plain float ndarray (avoids pandas Int64Dtype contaminating
        # ``X.values`` with an object dtype).
        num_cols = [c for c in out.columns if c not in cat_cols]
        if num_cols:
            out[num_cols] = out[num_cols].apply(
                pd.to_numeric, errors="coerce",
            ).astype(np.float64)
        return out

    return _apply(X_ctx), _apply(X_test), encoder, cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeaturePreprocessor:
    """Per-model feature preprocessor.

    Fit-on-context, transform-on-test by construction (so unseen test
    categories get a known unknown code rather than silently becoming a
    new vocabulary entry).

    Parameters
    ----------
    model_name
        Name registered in :data:`_MODEL_STRATEGY` (e.g. ``"tabpfnv2"``,
        ``"mitra"``).

    Examples
    --------
    >>> pre = FeaturePreprocessor("mitra")
    >>> X_ctx_p, X_test_p = pre.fit_transform(X_ctx, X_test)
    >>> pre.strategy
    'numeric_df'
    >>> pre.encoded_columns_  # set after fit_transform when applicable
    ['Education', 'Marital_Status']
    """

    model_name: str
    strategy: Optional[str] = None
    encoded_columns_: list[str] = field(init=False, default_factory=list)
    filled_columns_: list[str] = field(init=False, default_factory=list)
    patched_columns_: list[str] = field(init=False, default_factory=list)
    _encoder: Optional[OrdinalEncoder] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.strategy is None:
            self.strategy = strategy_for(self.model_name)

    def fit_transform(
        self,
        X_ctx: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
        """Fit on ``X_ctx``, transform both context and test, return both.

        For most models the returned objects are pandas DataFrames.
        For TabDPT, which requires a plain 2-D numpy array, both arrays
        are converted via ``.to_numpy()`` here so the runner needs no
        model-specific logic.
        """
        if not isinstance(X_ctx, pd.DataFrame):
            raise TypeError(
                f"FeaturePreprocessor expects a pandas DataFrame for "
                f"X_ctx; got {type(X_ctx).__name__}."
            )
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(
                f"FeaturePreprocessor expects a pandas DataFrame for "
                f"X_test; got {type(X_test).__name__}."
            )
        if list(X_ctx.columns) != list(X_test.columns):
            raise ValueError(
                "FeaturePreprocessor: X_ctx and X_test must have identical "
                "column names in identical order."
            )

        if self.strategy == "passthrough":
            return X_ctx, X_test
        if self.strategy == "passthrough_fill_ctx_nan":
            X_ctx_out, X_test_out, filled_ctx = _fill_ctx_all_nan_columns(
                X_ctx, X_test,
            )
            X_ctx_out, X_test_out, filled_test = (
                _fill_test_all_nan_non_numeric_columns(X_ctx_out, X_test_out)
            )
            self.filled_columns_ = filled_ctx + [
                c for c in filled_test if c not in filled_ctx
            ]
            return X_ctx_out, X_test_out
        if self.strategy == "tabpfn":
            X_ctx_out, X_test_out, patched = _to_tabpfn(X_ctx, X_test)
            self.patched_columns_ = patched
            return X_ctx_out, X_test_out
        if self.strategy in {"numeric_df", "numeric_array"}:
            X_ctx_out, X_test_out, encoder, cols = _to_numeric(X_ctx, X_test)
            self._encoder = encoder
            self.encoded_columns_ = cols
            if self.strategy == "numeric_array" or self.model_name == "tabdpt":
                return X_ctx_out.to_numpy(), X_test_out.to_numpy()
            return X_ctx_out, X_test_out
        raise AssertionError(f"unhandled strategy: {self.strategy!r}")
