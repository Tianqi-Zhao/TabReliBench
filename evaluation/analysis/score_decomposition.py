"""Descriptive, equal-weight dataset × model score decomposition.

Uses a complete panel of paired seeds. Fractions describe the observed panel,
not population variance components or noise-adjusted interaction reliability.
Seed identifiers must refer to the same evaluation split across models.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import ClassVar, Literal, Optional

import numpy as np
import pandas as pd

from ..metrics.directions import SCALE_DEPENDENT_METRICS
from .base import DatasetAnalyzer, InputKind, _alpha_to_cell
from .tables import AvgTable

# Brier and log loss are dimensionless, although the ranking registry groups
# them with scale-dependent metrics for other analysis purposes.
_TARGET_SCALE_METRICS = SCALE_DEPENDENT_METRICS - {"brier_score", "log_loss"}
_KEYS = ["dataset_id", "seed", "model"]


@dataclass
class ScoreDecomposition:
    """One complete dataset × model matrix and its orthogonal components."""

    grand_mean: float
    total_sum_sq: float
    components: pd.DataFrame
    dataset_effects: pd.Series
    model_effects: pd.Series
    interactions: pd.DataFrame


@dataclass
class ScoreDecompositionResult:
    """One response, with raw paired gaps and descriptive split diagnostics."""

    averaged: ScoreDecomposition
    per_split: pd.DataFrame
    paired_gaps: pd.DataFrame
    gap_summary: pd.DataFrame
    diagnostics: pd.DataFrame
    n_seeds: int
    scale_column: Optional[str]
    n_estimators: Optional[float]


class ScoreDecompositionAnalyzer(DatasetAnalyzer):
    """Decompose split-averaged scores and assess split sensitivity.

    ``models`` optionally fixes a model panel independently of the pipeline's
    available model list. ``seeds`` fixes the expected splits; when omitted,
    the union of observed seeds in the selected model panel is required.
    ``missing='drop_dataset'`` removes whole incomplete datasets, never cells.
    ``scale_columns`` maps responses to documented training-only scale columns;
    normalization precedes averaging and must be identical across models within
    each dataset/seed. This class checks consistency, not scale provenance.
    """

    name: ClassVar[str] = "score_decomposition"
    input_kinds: ClassVar[tuple[InputKind, ...]] = ("long_abs",)

    def __init__(
        self,
        *,
        models: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        missing: Literal["error", "drop_dataset"] = "error",
        scale_columns: Optional[dict[str, str]] = None,
    ) -> None:
        if missing not in ("error", "drop_dataset"):
            raise ValueError("missing must be 'error' or 'drop_dataset'")
        for label, values in (("models", models), ("seeds", seeds)):
            if values is not None and (not values or len(set(values)) != len(values)):
                raise ValueError(f"{label} must be non-empty and unique")
        self.models = list(models) if models is not None else None
        self.seeds = list(seeds) if seeds is not None else None
        self.missing = missing
        self.scale_columns = dict(scale_columns or {})

    @staticmethod
    def _decompose(scores: pd.DataFrame) -> ScoreDecomposition:
        values = scores.to_numpy(dtype=float)
        if min(values.shape) < 2 or not np.isfinite(values).all():
            raise ValueError("Decomposition requires a finite matrix with >=2 datasets and models")
        if not scores.index.is_unique or not scores.columns.is_unique:
            raise ValueError("Dataset and model labels must be unique")
        n_datasets, n_models = values.shape
        # Center around an observed value first: constant decimal-valued
        # matrices must have exactly zero variation despite mean roundoff.
        shifted = values - values[0, 0]
        centered = shifted - shifted.mean()
        grand_mean = float(values[0, 0] + shifted.mean())
        dataset = centered.mean(axis=1)
        model = centered.mean(axis=0)
        interaction = centered - dataset[:, None] - model[None, :]
        sum_sq = np.array([
            n_models * np.square(dataset).sum(),
            n_datasets * np.square(model).sum(),
            np.square(interaction).sum(),
        ])
        total = float(np.square(centered).sum())
        fractions = sum_sq / total if total > 0 else np.full(3, np.nan)
        return ScoreDecomposition(
            grand_mean=grand_mean,
            total_sum_sq=total,
            components=pd.DataFrame({
                "component": ["dataset", "model", "interaction"],
                "sum_sq": sum_sq,
                "fraction": fractions,
            }),
            dataset_effects=pd.Series(dataset, index=scores.index, name="effect"),
            model_effects=pd.Series(model, index=scores.columns, name="effect"),
            interactions=pd.DataFrame(interaction, index=scores.index, columns=scores.columns),
        )

    def _prepare_panel(self, df, models, response):
        required = _KEYS + [response]
        absent = [c for c in required if c not in df.columns]
        if absent:
            raise ValueError(f"Missing input columns: {absent}")
        sub = df.loc[df.model.isin(models)].copy()
        if sub.empty:
            raise ValueError(f"{response}: selected model panel is empty")
        if sub[_KEYS].isna().any().any():
            raise ValueError("Dataset, model and seed identifiers must not be missing")
        # The caller supplies one configuration slice. Do not silently average
        # configurations that AvgTable does not include in its grouping keys.
        for col in ("ratio", "alpha", "n_estimators"):
            if col in sub and sub[col].nunique(dropna=False) > 1:
                raise ValueError(f"Analyze one {col} at a time")
        if sub.duplicated(_KEYS).any():
            raise ValueError(f"{response}: duplicate dataset/model/seed records")
        seeds = self.seeds if self.seeds is not None else sorted(sub.seed.unique())
        datasets = sorted(sub.dataset_id.unique())
        sub = sub.loc[sub.seed.isin(seeds)]
        index = pd.MultiIndex.from_product([datasets, seeds], names=["dataset_id", "seed"])
        panel = sub.pivot(index=["dataset_id", "seed"], columns="model", values=response)
        panel = panel.reindex(index=index, columns=models).astype(float)
        scale_col = self.scale_columns.get(response)
        if response in _TARGET_SCALE_METRICS and scale_col is None:
            raise ValueError(f"{response}: supply a training-only scale via scale_columns")
        if scale_col is not None:
            if scale_col not in sub:
                raise ValueError(f"{response}: missing scale column {scale_col!r}")
            scale = sub.pivot(index=["dataset_id", "seed"], columns="model", values=scale_col)
            scale = scale.reindex(index=index, columns=models).astype(float)
            # NaN scales are incomplete observations; contradictory non-null
            # scales are a configuration error, not a reason to drop a dataset.
            if (scale.nunique(axis=1, dropna=True) > 1).any():
                raise ValueError(f"{scale_col}: scale must be identical across models per split")
            scale = scale.where(np.isfinite(scale) & (scale > 0))
            panel = panel / scale
        panel = panel.where(np.isfinite(panel))
        missing_cells = panel.isna().sum(axis=1).groupby(level="dataset_id").sum()
        diagnostics = missing_cells.rename("missing_cells").reset_index()
        diagnostics["kept"] = diagnostics.missing_cells.eq(0)
        diagnostics["reason"] = np.where(diagnostics.kept, "complete", "incomplete_or_invalid")
        excluded = diagnostics.loc[~diagnostics.kept, "dataset_id"].tolist()
        if excluded and self.missing == "error":
            raise ValueError(f"{response}: incomplete panel for datasets {excluded}; "
                             "use missing='drop_dataset' to exclude whole datasets")
        panel = panel.loc[~panel.index.get_level_values("dataset_id").isin(excluded)]
        if panel.index.get_level_values("dataset_id").nunique() < 2 or len(models) < 2:
            raise ValueError(f"{response}: need >=2 complete datasets and >=2 models")
        return panel, diagnostics

    def _split_diagnostics(self, panel):
        per_split = []
        for seed, block in panel.groupby(level="seed", sort=True):
            decomposition = self._decompose(block.droplevel("seed"))
            per_split.append(decomposition.components.assign(
                seed=seed, total_sum_sq=decomposition.total_sum_sq,
            ))
        gaps = []
        for model_a, model_b in combinations(panel.columns, 2):
            gap = (panel[model_b] - panel[model_a]).rename("gap").reset_index()
            gaps.append(gap.assign(model_a=model_a, model_b=model_b))
        paired = pd.concat(gaps, ignore_index=True)
        keys = ["dataset_id", "model_a", "model_b"]
        summary = paired.groupby(keys, as_index=False).agg(
            mean_gap=("gap", "mean"), std_gap=("gap", "std"), n_seeds=("gap", "size"),
            positive_fraction=("gap", lambda x: float((x > 0).mean())),
            negative_fraction=("gap", lambda x: float((x < 0).mean())),
            tie_fraction=("gap", lambda x: float((x == 0).mean())),
        )
        summary["direction_agreement"] = np.where(
            summary.mean_gap > 0, summary.positive_fraction,
            np.where(summary.mean_gap < 0, summary.negative_fraction, np.nan),
        )
        return pd.concat(per_split, ignore_index=True), paired, summary

    def run(self, df, *, feature_cols, models, responses):
        selected = self.models if self.models is not None else list(models)
        if len(set(selected)) != len(selected):
            raise ValueError("models must be unique")
        results = {}
        for response in responses:
            panel, diagnostics = self._prepare_panel(df, selected, response)
            # Reuse the established averaging builder after enforcing complete,
            # matched splits. A constant ratio suffices within this input slice.
            long = panel.rename_axis(columns="model").stack().rename(response).reset_index()
            long["ratio"] = 1.0
            averaged = AvgTable([response], []).build(long)
            scores = averaged.pivot(index="dataset_id", columns="model", values=response)
            scores = scores.reindex(columns=selected)
            per_split, paired, gap_summary = self._split_diagnostics(panel)
            results[response] = ScoreDecompositionResult(
                averaged=self._decompose(scores), per_split=per_split,
                paired_gaps=paired, gap_summary=gap_summary, diagnostics=diagnostics,
                n_seeds=panel.index.get_level_values("seed").nunique(),
                scale_column=self.scale_columns.get(response),
                n_estimators=(df.loc[df.model.isin(selected), "n_estimators"].iloc[0]
                              if "n_estimators" in df else None),
            )
        return results

    def save(self, results_by_key, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tables: dict[str, list[pd.DataFrame]] = {}
        for (ratio, alpha), results in results_by_key.items():
            for response, result in results.items():
                dec = result.averaged
                metadata = dict(ratio=ratio, alpha=_alpha_to_cell(alpha), response=response,
                                n_estimators=result.n_estimators,
                                scale_column=result.scale_column or "",
                                normalization="divide_by_training_scale" if result.scale_column else "raw")
                frames = {
                    "summary": dec.components.assign(
                        grand_mean=dec.grand_mean, total_sum_sq=dec.total_sum_sq,
                        n_datasets=len(dec.dataset_effects), n_models=len(dec.model_effects),
                        n_seeds=result.n_seeds,
                        status="zero_total_variation" if dec.total_sum_sq == 0 else "ok",
                    ),
                    "dataset_effects": dec.dataset_effects.rename_axis("dataset_id").reset_index(),
                    "model_effects": dec.model_effects.rename_axis("model").reset_index(),
                    "interactions": dec.interactions.rename_axis(index="dataset_id", columns="model")
                        .stack().rename("interaction").reset_index(),
                    "per_split": result.per_split,
                    "paired_gaps": result.paired_gaps,
                    "gap_summary": result.gap_summary,
                    "diagnostics": result.diagnostics,
                }
                for name, frame in frames.items():
                    tables.setdefault(name, []).append(frame.assign(**metadata))
        for name, frames in tables.items():
            pd.concat(frames, ignore_index=True).to_csv(out_dir / f"{name}.csv", index=False)
