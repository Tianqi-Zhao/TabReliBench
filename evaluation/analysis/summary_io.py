"""Load and filter numerical analysis summaries."""
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from evaluation.features.dataset.selected_features import SELECTED_DATASET_FEATURES_BY_TASK
DEFAULT_VALUE_COLS = ("xi", "r", "coef", "effective_slope", "rf_perm_importance_mean")

def load_summary(
    path: str | Path,
    *,
    value_col_candidates: Iterable[str] = DEFAULT_VALUE_COLS,
    p_candidates: Iterable[str] = ("p", "p_value", "slope_p"),
    task: str | None = None,
) -> pd.DataFrame:
    """Read an analyzer ``summary.csv`` and normalise it.

    Adds (or copies into) a unified ``value`` column and a unified ``p``
    column. The original columns are left untouched. Trims to rows where
    ``feature`` and ``response`` are non-null. When ``task`` is provided,
    restricts rows to the selected dataset-level features for that task.
    """
    df = pd.read_csv(path)

    # Locate value column.
    value_col = next((c for c in value_col_candidates if c in df.columns), None)
    if value_col is None:
        raise KeyError(
            f"load_summary({path}): none of {tuple(value_col_candidates)} "
            f"found in columns {list(df.columns)}"
        )
    if "value" not in df.columns:
        df["value"] = df[value_col]

    # Locate p column.
    if "p" not in df.columns:
        p_col = next((c for c in p_candidates if c in df.columns), None)
        if p_col is not None and p_col != "p":
            df["p"] = df[p_col]

    if "feature" in df.columns:
        df = df.dropna(subset=["feature"])
    if "response" in df.columns:
        df = df.dropna(subset=["response"])

    df = df.reset_index(drop=True)
    if task is not None:
        df = filter_selected_dataset_features(df, task=task, source=path)
    return df

def filter_selected_dataset_features(
    df: pd.DataFrame,
    *,
    task: str,
    source: str | Path | None = None,
) -> pd.DataFrame:
    """Restrict a dataset-level summary table to task-selected features.

    Missing selected features are reported but tolerated so old or
    task-specific summaries can still be plotted.
    """
    if task not in SELECTED_DATASET_FEATURES_BY_TASK:
        raise ValueError(
            f"task must be one of {tuple(SELECTED_DATASET_FEATURES_BY_TASK)}; "
            f"got {task!r}"
        )
    if "feature" not in df.columns:
        raise KeyError("filter_selected_dataset_features: missing 'feature' column")

    selected = SELECTED_DATASET_FEATURES_BY_TASK[task]
    selected_set = set(selected)
    source_label = str(source) if source is not None else "<summary>"

    present = set(df["feature"].dropna().astype(str).unique())
    non_selected = sorted(present - selected_set)
    missing_selected = [f for f in selected if f not in present]

    if non_selected:
        print(
            f"[selected-features] {source_label}: filtering out "
            f"{len(non_selected)} non-selected features: "
            f"{', '.join(non_selected)}"
        )
    if missing_selected:
        print(
            f"[selected-features] {source_label}: "
            f"{len(missing_selected)} selected features missing from summary: "
            f"{', '.join(missing_selected)}"
        )

    return df[df["feature"].isin(selected_set)].reset_index(drop=True)

def filter_ratio(df: pd.DataFrame, ratio: object) -> pd.DataFrame:
    """Return rows matching ``ratio``. Handles both numeric and string ratios."""
    if "ratio" not in df.columns:
        return df
    col = df["ratio"]
    try:
        col_f = pd.to_numeric(col, errors="coerce")
        r_f = float(ratio)
        if not np.isnan(r_f):
            return df[np.isclose(col_f, r_f)]
    except (TypeError, ValueError):
        pass
    return df[col.astype(str) == str(ratio)]
