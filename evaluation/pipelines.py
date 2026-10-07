"""End-to-end pipeline orchestrators.

Each pipeline encapsulates one stage of the calibration-analysis workflow.
CLI scripts (``evaluate_vanilla_tabpfn.py`` / ``compute_metrics.py`` / ...)
are thin wrappers that argparse → instantiate → ``.run()``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import hashlib
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .analysis import (
    AvgTable,
    ChatterjeeAnalyzer,
    ChatterjeeResult,
    DatasetAnalyzer,
    EvalTable,
    FeatureImportance,
    INSTANCE_ID_COLS,
    INSTANCE_OUTCOME_COLS,
    InstanceChatterjeeAnalyzer,
    InstanceCoverageAnalyzer,
    InstanceCoverageResult,
    InstanceRegressionResult,
    InstanceTable,
    InstanceWidthAnalyzer,
    LMEAnalyzer,
    LMEResult,
    MetricsTable,
    ModelComparator,
    PairDeltaTable,
    RelBiasEvalTable,
    RelBiasTable,
    RelEvalTable,
    RelTable,
    RFImportanceAnalyzer,
    RFResult,
    slice_table_by,
    SpearmanAnalyzer,
    SpearmanResult,
    SummaryReporter,
    UnivariateRFAnalyzer,
    UnivariateResult,
)
from .data import DatasetLoader
from .sampling import (
    LEGACY, SamplingProtocolError, check_legacy_feature_cache,
    claim_classification_output, claim_feature_cache,
)
from .features.dataset import (
    DEFAULT_CLASSIFICATION_DATASET_GROUPS,
    DEFAULT_DATASET_GROUPS,
    TASK_TO_EXTRACTOR_CLS,
)
from .features.dataset.selected_features import (
    RESPONSE_CATEGORIES_BY_TASK_KIND,
    ResponseCategory,
    SELECTED_DATASET_FEATURES_BY_TASK,
)
from .features.instance import DEFAULT_INSTANCE_GROUPS, InstanceFeatureExtractor
from .metrics import (
    CLASSIFICATION_RESPONSE_COLS,
    RESPONSE_COLS,
    BaseMetricsCalculator,
    ClassificationMetricsCalculator,
    RegressionMetricsCalculator,
)
from .models import (
    DEFAULT_QUANTILE_GRID,
    ClassificationModelRunner,
    ModelConfig,
    ModelRegistry,
    RegressionModelRunner,
    checkpoint_manifest_for_config,
    prediction_configuration,
    prediction_contract,
    make_runner,
)
from .ppd import PPDQuantileGrid
from .spec import TASK_CLASSIFICATION, TASK_REGRESSION, ExperimentSpec
from .store import ArtifactStore

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PredictionPipeline
# ─────────────────────────────────────────────────────────────────────────────

class PredictionPipeline:
    """Run every model on one ``ExperimentSpec`` and persist predictions.

    Task-aware: loading uses ``spec.task`` (use ``'auto'`` to detect from
    OpenML target dtype); after load the spec is re-stamped with the
    resolved concrete task for artefact paths. Records are written to
    ``predictions/<task>/...`` via :class:`ArtifactStore`.

    Regression record keys:
        task, model, dataset_id, seed, ratio, n_*, feature_names,
        quantile_levels, ppd_quantiles, point_pred, y_test, X_test.

    Classification record keys:
        task, model, dataset_id, seed, ratio, n_*, feature_names,
        proba, classes_, y_test, X_test.
    """

    def __init__(
        self,
        store: ArtifactStore,
        loader: DatasetLoader,
        models: Optional[dict[str, ModelConfig]] = None,
        q_grid: np.ndarray = DEFAULT_QUANTILE_GRID,
        *,
        classification_models: Optional[dict[str, ModelConfig]] = None,
        n_estimators: Optional[int] = 8,
    ) -> None:
        """Build a task-aware pipeline.

        ``models`` overrides the **regression** registry (back-compat
        with pre-classification call sites). Pass
        ``classification_models`` to override the classification
        registry.

        ``n_estimators`` controls the ensemble size used by every model
        in the registry. The default (8) matches TabPFN / TabICL /
        TabDPT defaults; adapters explicitly map it to native ensemble settings.
        Pass ``None`` to keep each library's own default.

        Both registries are resolved **lazily** the first time the
        relevant task is actually run, so a regression-only run never
        pays the cost of importing the classification packages (and a
        classification-only run won't fail to start just because the
        regression checkpoints aren't on disk yet).
        """
        self.store = store
        self.loader = loader
        self._regression_override = models
        self._classification_override = classification_models
        self._regression_cache: Optional[dict[str, ModelConfig]] = None
        self._classification_cache: Optional[dict[str, ModelConfig]] = None
        self.q_grid = np.asarray(q_grid, dtype=np.float64)
        self.n_estimators = (
            int(n_estimators) if n_estimators is not None else None
        )

    @property
    def regression_models(self) -> dict[str, ModelConfig]:
        if self._regression_cache is None:
            self._regression_cache = (
                self._regression_override
                if self._regression_override is not None
                else ModelRegistry.default_regression()
            )
        return self._regression_cache

    @property
    def classification_models(self) -> dict[str, ModelConfig]:
        if self._classification_cache is None:
            self._classification_cache = (
                self._classification_override
                if self._classification_override is not None
                else ModelRegistry.default_classification()
            )
        return self._classification_cache

    # Back-compat: legacy code may read ``self.models`` (regression).
    @property
    def models(self) -> dict[str, ModelConfig]:
        return self.regression_models

    def _registry_for(self, task: str) -> dict[str, ModelConfig]:
        if task == TASK_REGRESSION:
            return self.regression_models
        if task == TASK_CLASSIFICATION:
            return self.classification_models
        raise ValueError(f"Unsupported task: {task!r}")

    def run(
        self,
        spec: ExperimentSpec,
        *,
        only_models: Optional[Iterable[str]] = None,
    ) -> dict[str, dict]:
        """Fit each model and save its predictions PKL. Return ``{model: record}``.

        The loader receives ``spec.task`` (``'auto'`` triggers dtype-based
        detection in ``DatasetLoader``). ``ExperimentSpec`` defaults this
        field to ``'regression'`` when omitted (legacy).
        """
        load_task = spec.task or "auto"
        protocol = getattr(self.loader, "classification_sampling", LEGACY)
        if load_task == TASK_CLASSIFICATION:
            claim_classification_output(self.store.root, protocol, getattr(self.loader, "max_n", 10_000))
        try:
            split = self.loader.load_for_spec(spec, task=load_task)
        except SamplingProtocolError:
            raise
        except Exception as exc:
            log.warning("Skipping %s: failed to load (%s: %s)",
                        spec, type(exc).__name__, exc)
            return {}

        resolved_task = split.task  # always concrete after load_for_spec
        if load_task == "auto" and resolved_task == TASK_CLASSIFICATION:
            claim_classification_output(self.store.root, protocol, getattr(self.loader, "max_n", 10_000))
        # Re-stamp spec with the concrete task so save paths land in the
        # correct subdir even when the caller passed task='auto'.
        spec = ExperimentSpec(
            spec.dataset_id, spec.seed, spec.ratio, resolved_task,
        )

        log.info(
            "OK %s task=%s  n=%d  d=%d  train=%d  test=%d  context=%d",
            spec, resolved_task,
            len(split.y_train) + len(split.y_test), split.X_train.shape[1],
            len(split.y_train), len(split.y_test), len(split.y_ctx),
        )

        # Identify the actual context/query data, including row order and labels.
        # No additional split or sampling is performed here.
        split_hash = hashlib.sha256()
        for values in (split.X_ctx, split.y_ctx, split.X_test, split.y_test):
            frame = values if isinstance(values, pd.DataFrame) else pd.Series(values)
            split_hash.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        split_hash.update(repr(list(split.X_ctx.columns)).encode())
        split_fingerprint = split_hash.hexdigest()

        registry = self._registry_for(resolved_task)
        keep = set(only_models) if only_models is not None else None
        if keep is not None and keep - registry.keys():
            raise ValueError(f"Unknown {resolved_task} models: {sorted(keep - registry.keys())}")

        out: dict[str, dict] = {}
        for name, cfg in registry.items():
            if keep is not None and name not in keep:
                continue
            prediction_path = self.store.predictions_path(
                spec, name, task=resolved_task,
            )
            configuration, fingerprint = prediction_configuration(
                cfg, spec.seed, self.n_estimators, self.q_grid,
            )
            configuration["split_fingerprint"] = split_fingerprint
            fingerprint = hashlib.sha256((fingerprint + split_fingerprint).encode()).hexdigest()
            cache_matches = False
            if prediction_path.exists():
                cached = self.store.load_prediction(spec, name, task=resolved_task)
                previous = cached.get("configuration_fingerprint")
                # Preserve old artifacts for existing TFMs. New models always
                # require explicit provenance before a prediction can be reused.
                cache_matches = previous == fingerprint or (
                    previous is None and name not in {"tabpfnv3.5", "causilo", "limix2", "tabfm", "tabdpt1.3"}
                )
                if not cache_matches:
                    log.info("Inference configuration changed; recomputing %s", name)
            model_complete = self.store.model_exists(
                spec, name, task=resolved_task,
            )
            if cache_matches and model_complete:
                log.info("Cached prediction and checkpoint manifest; skipping %s", name)
                continue
            if cache_matches and not model_complete:
                manifest = {
                    **checkpoint_manifest_for_config(
                        cfg, spec.seed, n_estimators=self.n_estimators,
                    ),
                    "dataset_id": spec.dataset_id,
                    "ratio": spec.ratio,
                }
                if cached.get("configuration_fingerprint"):
                    manifest.update(configuration_fingerprint=fingerprint, model_metadata=configuration)
                model_saved = self.store.save_model(
                    spec, name, manifest, task=resolved_task,
                )
                log.info(
                    "Backfilled checkpoint manifest without TFM inference → %s",
                    model_saved,
                )
                continue
            log.info(
                "Fitting %s (ratio=%s, n_estimators=%s) ...",
                name, spec.ratio, self.n_estimators,
            )
            try:
                runner = make_runner(
                    cfg, seed=spec.seed, n_estimators=self.n_estimators,
                )
                if resolved_task == TASK_REGRESSION:
                    assert isinstance(runner, RegressionModelRunner)
                    ppd, point_pred = runner.fit_predict(
                        split.X_ctx, split.y_ctx, split.X_test, self.q_grid,
                    )
                else:
                    assert isinstance(runner, ClassificationModelRunner)
                    # Use the dataset-level encoder cardinality so that
                    # context subsets which happen to miss a class still
                    # produce predictions on the full global class space.
                    n_classes_global = (
                        int(split.classes_.size)
                        if split.classes_ is not None
                        else None
                    )
                    proba, classes_ = runner.fit_predict(
                        split.X_ctx, split.y_ctx, split.X_test,
                        n_classes_global=n_classes_global,
                    )
            except Exception as exc:
                if prediction_path.exists() and not cache_matches:
                    raise RuntimeError(
                        f"Failed to refresh {name}; existing prediction belongs to a different "
                        "configuration. Use a separate output directory for the new run."
                    ) from exc
                log.warning("Skipping %s on %s: %s: %s",
                            name, spec, type(exc).__name__, exc)
                continue

            base = {
                "configuration_fingerprint": fingerprint,
                "model_metadata": configuration,
                **prediction_contract(cfg),
                "task":           resolved_task,
                "model":          name,
                "dataset_id":     spec.dataset_id,
                "seed":           spec.seed,
                "ratio":          spec.ratio,
                "n_total":        len(split.y_train) + len(split.y_test),
                "n_train":        len(split.y_train),
                "n_test":         len(split.y_test),
                "n_context":      len(split.y_ctx),
                "n_features":     split.X_train.shape[1],
                "feature_names":  split.feature_names,
                "y_test":         split.y_test,
                "X_test":         split.X_test,
                "sampling_metadata": deepcopy(getattr(split, "sampling_metadata", {})),
            }
            if resolved_task == TASK_REGRESSION:
                record = {**base, "point_pred": point_pred.astype(np.float32)}
                if ppd is not None:
                    record.update(quantile_levels=self.q_grid, ppd_quantiles=ppd.astype(np.float32))
                shape_log = f"output={record['output_kind']}, point_pred shape={point_pred.shape}"

            else:
                record = {
                    **base,
                    "proba":     proba.astype(np.float32),
                    "classes_":  classes_,
                }
                shape_log = (
                    f"proba shape={proba.shape}, classes={list(classes_)}"
                )

            model_artifact = {
                **runner.checkpoint_manifest(),
                "configuration_fingerprint": fingerprint,
                "model_metadata": configuration,
                "dataset_id": spec.dataset_id,
                "ratio": spec.ratio,
                "sampling_metadata": deepcopy(getattr(split, "sampling_metadata", {})),
            }
            model_saved = self.store.save_model(
                spec, name, model_artifact, task=resolved_task,
            )
            log.info("Saved checkpoint manifest → %s", model_saved)
            record["model_artifact_path"] = str(model_saved)
            saved = self.store.save_prediction(
                spec, name, record, task=resolved_task,
            )
            log.info("Saved predictions → %s  (%s)", saved, shape_log)
            out[name] = record
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MetricsPipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricsRunResult:
    """Outcome of one centralized metrics computation/save operation."""

    status: str
    seconds: float
    metrics: Optional[dict] = None


class MetricsPipeline:
    """Compute and persist metrics for every prediction PKL in the store.

    Task-aware: dispatches on each record's ``task`` field (defaulting
    to ``'regression'`` for legacy records that lack it). Pass
    ``classification_calculator`` to override the default
    :class:`ClassificationMetricsCalculator` (e.g. custom metric list or
    ``n_bins``).
    """

    def __init__(
        self,
        store: ArtifactStore,
        alphas: list[float],
        *,
        overwrite: bool = False,
        regression_calculator: Optional[BaseMetricsCalculator] = None,
        classification_calculator: Optional[BaseMetricsCalculator] = None,
    ) -> None:
        self.store = store
        self.alphas = list(map(float, alphas))
        self.overwrite = overwrite
        self.regression_calc = (
            regression_calculator
            if regression_calculator is not None
            else RegressionMetricsCalculator(self.alphas)
        )
        self.classification_calc = (
            classification_calculator
            if classification_calculator is not None
            else ClassificationMetricsCalculator()
        )

    # Back-compat: ``self.calc`` historically meant the regression calc.
    @property
    def calc(self) -> BaseMetricsCalculator:
        return self.regression_calc

    def _calc_for(self, task: str) -> BaseMetricsCalculator:
        if task == TASK_REGRESSION:
            return self.regression_calc
        if task == TASK_CLASSIFICATION:
            return self.classification_calc
        raise ValueError(f"Unsupported task on record: {task!r}")

    def is_cached(
        self, spec: ExperimentSpec, model: str, *, task: str,
    ) -> bool:
        """Return whether this pipeline should reuse an existing artifact."""
        return (
            not self.overwrite
            and self.store.metrics_exist(spec, model, task=task)
        )

    def compute_and_save(
        self,
        record: dict,
        spec: ExperimentSpec,
        model: str,
        *,
        task: Optional[str] = None,
    ) -> MetricsRunResult:
        """Compute, enrich, and persist metrics through one canonical path.

        Prediction provenance used by downstream analysis is copied into the
        metrics artifact. Callers therefore only need to load or construct a
        prediction record; they must not reimplement task dispatch, metadata
        propagation, timing, cache handling, or metrics persistence.
        """
        record_task = record.get(
            "task", task or spec.task or TASK_REGRESSION,
        )
        cache_matches = self.is_cached(spec, model, task=record_task)
        if cache_matches and record.get("configuration_fingerprint"):
            previous = self.store.load_metrics(spec, model, task=record_task)
            cache_matches = (previous.get("configuration_fingerprint") == record["configuration_fingerprint"]
                             and previous.get("requested_alphas") == self.alphas)
        if cache_matches:
            log.info(
                "Skip (exists): %s",
                self.store.metrics_path(
                    spec, model, task=record_task,
                ).name,
            )
            return MetricsRunResult(status="cached", seconds=0.0)

        start = time.perf_counter()
        calculator = self._calc_for(record_task)
        metrics = calculator.compute_for_record(record)
        metrics["requested_alphas"] = self.alphas
        seconds = time.perf_counter() - start

        for key in (
            "configuration_fingerprint",
            "output_kind",
            "regression_metrics",
            "model_artifact_path",
            "baseline_config",
            "model_metadata",
            "sampling_metadata",
        ):
            if key in record:
                metrics[key] = deepcopy(record[key])

        source_timings = record.get("timing_seconds", {})
        if not isinstance(source_timings, dict):
            raise TypeError("record['timing_seconds'] must be a dict")
        metrics["timing_seconds"] = {
            **deepcopy(source_timings),
            "metrics": seconds,
        }

        path = self.store.save_metrics(
            spec, model, metrics, task=record_task,
        )
        log.info("Saved metrics → %s", path)
        return MetricsRunResult(
            status="written", seconds=seconds, metrics=metrics,
        )

    def run_one(
        self, spec: ExperimentSpec, model: str,
        task: Optional[str] = None,
    ) -> bool:
        t = task or spec.task or TASK_REGRESSION
        try:
            record = self.store.load_prediction(spec, model, task=t)
        except Exception as exc:
            log.warning("Could not load prediction for %s/%s (%s: %s)",
                        spec, model, type(exc).__name__, exc)
            return False
        try:
            self.compute_and_save(record, spec, model, task=t)
        except Exception as exc:
            record_task = record.get("task", t)
            log.warning("Failed metrics on %s/%s task=%s (%s: %s)",
                        spec, model, record_task, type(exc).__name__, exc)
            return False
        return True

    def run_all(self) -> tuple[int, int]:
        """Iterate over every prediction file the store knows about
        (both regression and classification subdirectories)."""
        n_ok = n_total = 0
        for spec, model, _ in self.store.iter_predictions():
            n_total += 1
            if self.run_one(spec, model, task=spec.task):
                n_ok += 1
        log.info("Metrics done: %d / %d files processed.", n_ok, n_total)
        return n_ok, n_total


# ─────────────────────────────────────────────────────────────────────────────
# DatasetFeaturePipeline
# ─────────────────────────────────────────────────────────────────────────────

class DatasetFeaturePipeline:
    """Compute and cache dataset-level meta-features.

    Features depend on ``(dataset_id, seed)`` but not on ``ratio`` (ratio
    only controls how many context examples the model sees). One PKL is
    written per ``(dataset_id, seed)``; downstream joins use the same
    features regardless of ratio.

    Both task-specific extractors are instantiated at construction time so
    they are ready to use immediately in :meth:`run`.  The correct one is
    selected per call via the ``task`` argument to :meth:`run` (override) or
    the task auto-detected from the dataset (fallback).
    """

    def __init__(
        self,
        store: ArtifactStore,
        loader: DatasetLoader,
        *,
        max_samples: int = 10_000,
    ) -> None:
        self.store      = store
        self.loader     = loader
        self._extractors = {
            t: cls(max_samples=max_samples)
            for t, cls in TASK_TO_EXTRACTOR_CLS.items()
        }

    def run(
        self,
        spec: ExperimentSpec,
        *,
        task: Optional[str] = None,
        skip_existing: bool = True,
    ) -> Optional[dict]:
        """Compute features for ``spec`` and write one PKL per (dataset_id, seed).

        Parameters
        ----------
        task:
            Force a specific task (``'regression'`` or ``'classification'``).
            When ``None`` (default) the task is auto-detected from the dataset
            via :meth:`DatasetLoader.load_raw`.
        """
        cache_spec = ExperimentSpec(spec.dataset_id, spec.seed, 1.0)
        protocol = getattr(self.loader, "classification_sampling", LEGACY)
        cache_path = self.store.dataset_features_path(cache_spec)
        if protocol == LEGACY:
            check_legacy_feature_cache(cache_path)
        if protocol == LEGACY and skip_existing and self.store.load_dataset_features(cache_spec) is not None:
            log.info("[%s seed=%d] cached - skip", spec.dataset_id, spec.seed)
            return None

        try:
            if protocol != LEGACY:
                # Use exactly the loader path used by foundation models and
                # baselines, including a forced classification target type.
                split = self.loader.load_for_spec(cache_spec, task=task or "auto")
                resolved_task = split.task
                X_train, y_train = split.X_train, split.y_train
                metadata = {
                    **split.sampling_metadata,
                    "task": resolved_task,
                    "sampling_protocol": protocol if resolved_task == TASK_CLASSIFICATION else LEGACY,
                    "max_n": self.loader.max_n,
                    "features_max_samples": self._extractors[resolved_task].max_samples,
                }
                claim_feature_cache(cache_path, metadata)
                if skip_existing and self.store.load_dataset_features(cache_spec) is not None:
                    log.info("[%s seed=%d] cached - skip", spec.dataset_id, spec.seed)
                    return None
            else:
                X_df, y, _, detected_task, _classes = self.loader.load_raw(
                    spec.dataset_id, spec.seed,
                )
                X_train, y_train, _, _ = self.loader.train_test_split(X_df, y, spec.seed)
                resolved_task = task if task is not None else detected_task
            feats = self._extractors[resolved_task].compute(X_train, y_train, spec.seed)
        except SamplingProtocolError:
            raise
        except Exception as exc:
            if protocol != LEGACY:
                raise
            log.warning("[%s seed=%d] FAILED - %s",
                        spec.dataset_id, spec.seed, exc)
            return None

        path = self.store.save_dataset_features(cache_spec, feats)
        log.info("Saved dataset features → %s", path)
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# InstanceFeaturePipeline
# ─────────────────────────────────────────────────────────────────────────────

class InstanceFeaturePipeline:
    """Compute per-instance meta-features for one ``ExperimentSpec``.

    Hard dependency on metrics PKL: the per-instance ``lower / upper /
    width / covered / winkler`` arrays are pulled from
    :class:`evaluation.metrics.MetricsCalculator`'s output, never recomputed
    from the PPD here.

    Pipeline order: ``PredictionPipeline → MetricsPipeline → InstanceFeaturePipeline``.
    Missing metrics PKL → :class:`FileNotFoundError`.
    """

    def __init__(
        self,
        store: ArtifactStore,
        loader: DatasetLoader,
        alpha: float,
        extractor: Optional[InstanceFeatureExtractor] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.store     = store
        self.loader    = loader
        self.alpha     = float(alpha)
        self.extractor = extractor or InstanceFeatureExtractor()

    def run(self, spec: ExperimentSpec) -> dict[str, "object"]:
        models = self.store.list_predictions(spec)
        if not models:
            raise FileNotFoundError(
                f"No prediction files found for {spec} under {self.store.root}"
            )

        log.info("Loading dataset for %s ...", spec)
        split = self.loader.load_for_spec(spec)

        out: dict[str, object] = {}
        for model in models:
            pred = self.store.load_prediction(spec, model)
            mts  = self.store.load_metrics(spec, model)
            ad = mts.get("alpha_dependent", {})
            if self.alpha not in ad:
                raise KeyError(
                    f"Metrics PKL for {spec}/{model} does not contain alpha "
                    f"{self.alpha} in 'alpha_dependent'. Available: "
                    f"{sorted(ad)}"
                )
            iv = ad[self.alpha]["per_instance"]
            ppd = PPDQuantileGrid.from_record(pred)

            df = self.extractor.compute(
                X_ctx=split.X_ctx, y_ctx=split.y_ctx,
                X_test=pred["X_test"],
                ppd=ppd,
                n_context=int(pred["n_context"]),
                intervals=iv,
            )
            # PKL stores ONLY feature columns. Identifiers (dataset_id,
            # seed, ratio, model) are encoded in the filename; outcomes
            # (covered, y_test, winkler) live in the metrics PKL.
            # InstanceTable.build() re-assembles everything at load time.

            path = self.store.save_instance_features(spec, model, df)
            log.info("Saved instance features → %s  (%d rows × %d cols)",
                     path, len(df), df.shape[1])
            out[model] = df
        return out


# ─────────────────────────────────────────────────────────────────────────────
# DatasetAnalysisPipeline
# ─────────────────────────────────────────────────────────────────────────────

# Features that are mathematically duplicated by another column in the eval
# table and should be dropped before any LME / RF fit (otherwise importance is
# split between the duplicates and inflates the "explained" feature count).
#
# Each entry is paired with the column it duplicates / is subsumed by:
#
#   dim_ratio            ≈ inverse of context_to_feature_ratio (when ratio≈1)
#                          and is a deterministic function of n_train,
#                          n_features.
_REDUNDANT_DATASET_FEATURES: frozenset[str] = frozenset({
    "dim_ratio"
})

def _all_dataset_feature_cols(task: str) -> list[str]:
    """Flat list of every dataset-level feature name in registration order.

    Drops :data:`_REDUNDANT_DATASET_FEATURES` so that downstream LME / RF
    analyses don't receive columns that are exact duplicates / deterministic
    transforms of other columns already in the eval table. To re-enable any
    of them, simply remove the entry from ``_REDUNDANT_DATASET_FEATURES``.
    Uses task-specific feature groups so classification-only meta-features
    are eligible for dataset-level analyses.
    """
    if task == TASK_REGRESSION:
        groups = DEFAULT_DATASET_GROUPS
    elif task == TASK_CLASSIFICATION:
        groups = DEFAULT_CLASSIFICATION_DATASET_GROUPS
    else:
        raise ValueError(
            f"task must be {TASK_REGRESSION!r} or "
            f"{TASK_CLASSIFICATION!r}; got {task!r}",
        )

    cols: list[str] = []
    for grp in groups:
        cols.extend(f for f in grp.feature_names
                    if f not in _REDUNDANT_DATASET_FEATURES)
    # ``context_to_feature_ratio`` is ratio-dependent (= n_context / n_features),
    # so it is not produced by any feature group's ``compute()`` — the feature
    # cache stays ratio-independent. ``EvalTable.build()`` injects it per row
    # from the metrics PKL's ``n_context`` / ``n_features`` fields.
    if "context_to_feature_ratio" not in _REDUNDANT_DATASET_FEATURES:
        cols.append("context_to_feature_ratio")
    return cols


class DatasetAnalysisPipeline:
    """Dataset-level analysis driven by a list of registered analyzers.

    Pipeline outline
    ----------------
    1. Build the long eval table from the metrics store
       (:class:`evaluation.analysis.EvalTable`).
    2. Build seed-averaged (``eval_avg``), axis-inclusive peer-relative
       (``eval_rel``), and raw peer-relative
       (``eval_rel_bias``) tables.
    3. Iterate the registered analyzers.  Each analyzer's
       ``input_kinds`` class attribute determines which of
       ``("long_abs", "long_rel", "avg", "rel")`` tables it consumes; the pipeline runs
       it on every (ratio) slice, then hands the
       ``{ratio: results}`` dict to the analyzer's ``save`` method
       which writes ``summary.csv`` + per-ratio HTML pivots + optional
       ``details/*`` under a pre-namespaced subdirectory.

    Output directory layout
    -----------------------
    ``output_dir`` is the final write target (assembled by the caller,
    typically a shell wrapper).  Example::

        <results_dir>/<task>/dataset_level_analysis/
          eval_long.csv  eval_avg.csv  eval_rel.csv
          eval_rel_bias.csv
          <analyzer.name>/ ...

    Task-aware: pass ``task=TASK_CLASSIFICATION`` to run on
    classification metrics PKLs.  When ``responses`` is omitted, the
    default is :data:`evaluation.metrics.RESPONSE_COLS` for regression
    and :data:`evaluation.metrics.CLASSIFICATION_RESPONSE_COLS` for
    classification.
    """

    def __init__(
        self,
        store: ArtifactStore,
        alphas: Optional[list[float]],
        output_dir: str | Path,
        *,
        task: str = TASK_REGRESSION,
        responses: Optional[list[str]] = None,
        peer_models: Optional[Iterable[str]] = None,
        analyzers: Optional[list[DatasetAnalyzer]] = None,
        allowed_dataset_ids: Optional[Iterable[int]] = None,
    ) -> None:
        if task == TASK_REGRESSION:
            if not alphas:
                raise ValueError(
                    "alphas must be a non-empty list for regression."
                )
            bad = [a for a in alphas if not (0.0 < float(a) < 1.0)]
            if bad:
                raise ValueError(
                    f"every alpha must lie in (0, 1); got {bad}"
                )
        elif task != TASK_CLASSIFICATION:
            raise ValueError(
                f"task must be {TASK_REGRESSION!r} or "
                f"{TASK_CLASSIFICATION!r}; got {task!r}"
            )
        self.store      = store
        self.task       = task
        # Regression: sorted list of miscoverage levels. Classification:
        # None (alpha dimension does not apply).
        self.alphas = (
            sorted({float(a) for a in alphas})
            if (task == TASK_REGRESSION and alphas)
            else None
        )
        self.output_dir = Path(output_dir)
        if responses is None:
            responses = (
                list(RESPONSE_COLS) if task == TASK_REGRESSION
                else list(CLASSIFICATION_RESPONSE_COLS)
            )
        self.responses  = list(responses)
        self.peer_models = (
            list(peer_models) if peer_models is not None else None
        )
        self.analyzers = (
            list(analyzers) if analyzers is not None
            else self._default_analyzers()
        )
        self.allowed_dataset_ids = (
            set(map(int, allowed_dataset_ids))
            if allowed_dataset_ids is not None else None
        )

    @staticmethod
    def _default_analyzers() -> list[DatasetAnalyzer]:
        """Default analyzer set when caller doesn't pass ``analyzers=``."""
        return [LMEAnalyzer(), SpearmanAnalyzer(), ChatterjeeAnalyzer()]

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self) -> dict:
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Build the long eval table ────────────────────────────────────
        log.info(
            "Building eval table (task=%s alphas=%s) ...",
            self.task, self.alphas,
        )
        eval_long, n_missing = EvalTable(
            self.store, self.alphas, task=self.task,
        ).build(allowed_dataset_ids=self.allowed_dataset_ids)
        if eval_long.empty:
            log.error("No eval rows built — check task / alpha / store paths.")
            return {"eval_long": eval_long}
        log.info(
            "Built eval_long: %d rows, %d datasets, %d missing feature caches",
            len(eval_long), eval_long["dataset_id"].nunique(), n_missing,
        )

        models = sorted(eval_long["model"].unique())
        seeds  = sorted(eval_long["seed"].unique())
        # Scope predictors to the manually curated dimensionless / scale-free
        # feature set (selected_features.py), matching the pair-delta
        # comparison pipeline.  Excluded raw / dimensional features (e.g.
        # mi_entropy, tree_depth, hs_dof, condition_number, wg_dist) are kept
        # out of the joint fits so they don't dilute importance via collinearity
        # with their normalized replacements.
        feature_cols = [
            f for f in SELECTED_DATASET_FEATURES_BY_TASK[self.task]
            if f in eval_long.columns
        ]
        responses = [r for r in self.responses if r in eval_long.columns]
        if not responses:
            log.error(
                "None of the configured responses %s present in eval_long.",
                self.responses,
            )
            return {"eval_long": eval_long}

        # Partition responses into alpha-free vs alpha-dependent.
        alpha_free_set = set(
            eval_long.attrs.get("alpha_free_metrics", frozenset())
        )
        alpha_free_responses = [r for r in responses if r in alpha_free_set]
        alpha_dep_responses  = [r for r in responses if r not in alpha_free_set]
        log.info(
            "Responses: %d alpha-free (%s) + %d alpha-dependent (%s)",
            len(alpha_free_responses), alpha_free_responses,
            len(alpha_dep_responses), alpha_dep_responses,
        )

        peer_models = (
            self.peer_models if self.peer_models is not None else models
        )

        log.info(
            "Models=%s seeds=%s features=%d alphas=%s",
            models, seeds, len(feature_cols), self.alphas,
        )

        task_dir = out_dir
        task_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "── %s ── %d rows × %d datasets",
            task_dir,
            len(eval_long), eval_long["dataset_id"].nunique(),
        )

        eval_avg = AvgTable(
            response_cols=responses, feature_cols=feature_cols,
        ).build(eval_long)
        eval_rel = RelTable(
            response_cols=responses, peer_models=peer_models,
        ).build(eval_avg)
        eval_rel_bias = RelBiasTable(
            response_cols=responses, peer_models=peer_models,
        ).build(eval_avg)
        eval_rel_long = RelEvalTable(
            response_cols=responses, peer_models=peer_models,
        ).build(eval_long)
        eval_rel_long_bias = RelBiasEvalTable(
            response_cols=responses, peer_models=peer_models,
        ).build(eval_long)

        eval_long.to_csv(task_dir / "eval_long.csv", index=False)
        eval_avg.to_csv(task_dir / "eval_avg.csv", index=False)
        eval_rel.to_csv(task_dir / "eval_rel.csv", index=False)
        eval_rel_bias.to_csv(task_dir / "eval_rel_bias.csv", index=False)
        eval_rel_long.to_csv(task_dir / "eval_rel_long.csv", index=False)
        eval_rel_long_bias.to_csv(
            task_dir / "eval_rel_long_bias.csv", index=False,
        )

        ratios = sorted(eval_long["ratio"].unique())
        rel_responses = [r for r in responses if r in eval_rel.columns]

        # Alpha labels to iterate per (ratio): "alpha_free" + one float
        # per configured alpha (regression). Classification yields a
        # single None label (no alpha dimension).
        if self.task == TASK_REGRESSION:
            alpha_labels: list[object] = (
                [EvalTable.ALPHA_FREE] + list(self.alphas)
            )
            dedup_alpha = float(self.alphas[0])
        else:
            alpha_labels = [None]
            dedup_alpha = None

        rel_long_responses = [r for r in responses if r in eval_rel_long.columns]
        rel_bias_responses = [
            r for r in responses if r in eval_rel_bias.columns
        ]
        rel_long_bias_responses = [
            r for r in responses if r in eval_rel_long_bias.columns
        ]

        # ── 2. Loop over analyzers × kinds ────────────────────────────────
        analyzer_outputs: dict[tuple[str, str], dict] = {}
        for analyzer in self.analyzers:
            for kind in analyzer.input_kinds:
                leaf_dir = task_dir / analyzer.name
                if kind in ("avg", "long_abs"):
                    leaf_dir = leaf_dir / "abs"
                elif kind in ("rel", "long_rel"):
                    leaf_dir = leaf_dir / "rel"
                elif kind in ("rel_bias", "long_rel_bias"):
                    leaf_dir = leaf_dir / "rel_bias"
                leaf_dir.mkdir(parents=True, exist_ok=True)

                # Source table for this (analyzer, kind).
                full_table = self._table_for_kind(
                    eval_long,
                    eval_avg,
                    eval_rel,
                    eval_rel_long,
                    eval_rel_bias,
                    eval_rel_long_bias,
                    kind,
                )
                if kind == "rel":
                    resps_full = rel_responses
                elif kind == "long_rel":
                    resps_full = rel_long_responses
                elif kind == "rel_bias":
                    resps_full = rel_bias_responses
                elif kind == "long_rel_bias":
                    resps_full = rel_long_bias_responses
                else:
                    resps_full = responses
                if full_table.empty or not resps_full:
                    continue

                # ── 3. Inner loop over ratios × alpha_labels ──────────────
                results_by_key: dict[
                    tuple[float, "object"], object,
                ] = {}
                for ratio in ratios:
                    ratio_slice = slice_table_by(
                        full_table, "ratio", ratio,
                    )
                    if ratio_slice.empty:
                        continue
                    for alpha_label in alpha_labels:
                        resps_for_slice = self._responses_for_alpha(
                            alpha_label,
                            alpha_free_responses=alpha_free_responses,
                            alpha_dep_responses=alpha_dep_responses,
                            full_responses=resps_full,
                        )
                        if not resps_for_slice:
                            continue
                        df = EvalTable.slice_by_alpha(
                            ratio_slice,
                            alpha_label,
                            dedup_alpha=dedup_alpha,
                        )
                        if df.empty:
                            continue
                        log.info(
                            "[%s/%s ratio=%s alpha=%s] "
                            "running on %d rows × %d responses",
                            analyzer.name, kind,
                            ratio, alpha_label,
                            len(df), len(resps_for_slice),
                        )
                        results_by_key[(float(ratio), alpha_label)] = (
                            analyzer.run(
                                df,
                                feature_cols=feature_cols,
                                models=models,
                                responses=resps_for_slice,
                            )
                        )

                if results_by_key:
                    analyzer.save(results_by_key, leaf_dir)
                    analyzer_outputs[
                        (analyzer.name, kind)
                    ] = results_by_key

        return {
            "eval_long":         eval_long,
            "n_missing":         n_missing,
            "task_dir":          task_dir,
            "eval_avg":          eval_avg,
            "eval_rel":          eval_rel,
            "eval_rel_bias":     eval_rel_bias,
            "eval_rel_long":     eval_rel_long,
            "eval_rel_long_bias": eval_rel_long_bias,
            "analyzer_outputs":  analyzer_outputs,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _table_for_kind(
        eval_long:     pd.DataFrame,
        eval_avg:      pd.DataFrame,
        eval_rel:      pd.DataFrame,
        eval_rel_long: pd.DataFrame,
        eval_rel_bias: pd.DataFrame,
        eval_rel_long_bias: pd.DataFrame,
        kind: str,
    ) -> pd.DataFrame:
        """Pick the right table for the analyzer's input kind."""
        if kind == "long_abs":
            return eval_long
        if kind == "long_rel":
            return eval_rel_long
        if kind == "long_rel_bias":
            return eval_rel_long_bias
        if kind == "avg":
            return eval_avg
        if kind == "rel":
            return eval_rel
        if kind == "rel_bias":
            return eval_rel_bias
        raise ValueError(f"unknown input kind: {kind!r}")

    def _responses_for_alpha(
        self,
        alpha_label: object,
        *,
        alpha_free_responses: list[str],
        alpha_dep_responses: list[str],
        full_responses: list[str],
    ) -> list[str]:
        """Pick the response columns applicable to one alpha slice.

        * ``alpha_label is None`` (classification, no alpha dimension) →
          return ``full_responses``.
        * ``alpha_label == "alpha_free"`` → ``alpha_free_responses``.
        * ``alpha_label = <float>`` → ``alpha_dep_responses``.

        Row filtering is :meth:`EvalTable.slice_by_alpha`'s job; this
        function only decides which response columns to feed the
        analyzer.
        """
        if alpha_label is None:
            return list(full_responses)
        full_set = set(full_responses)
        if alpha_label == EvalTable.ALPHA_FREE:
            return [r for r in alpha_free_responses if r in full_set]
        return [r for r in alpha_dep_responses if r in full_set]


# ─────────────────────────────────────────────────────────────────────────────
# FeatureSelectionPipeline
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field as dc_field, replace as dc_replace


@dataclass
class AnalyzerSpec:
    """Describes how to read and rank features from one analyzer's summary.csv.

    Parameters
    ----------
    name:
        Subdirectory name under the analysis root (e.g. ``"spearman"``).
    value_col:
        Column in ``summary.csv`` to rank features by.
    rank_by_abs:
        If ``True``, rank by ``|value_col|`` (useful for signed metrics
        like Spearman *r* or LME ``coef``).
    p_col:
        Optional p-value column name for future significance filtering.
    kinds:
        Which subdirectories (``"abs"``, ``"rel"``) to look for.
    threshold:
        Minimum importance for ``selection="threshold"`` mode, compared
        against the ranking value (``|value_col|`` when ``rank_by_abs``,
        the raw value otherwise — i.e. |r| for Spearman, raw ξ for
        Chatterjee).  ``None`` marks an unbounded metric (e.g. LME
        coefficients on ℝ, whose scale depends on the response): such
        analyzers are skipped entirely in threshold mode.
    """
    name: str
    value_col: str
    rank_by_abs: bool = True
    p_col: str | None = None
    kinds: tuple[str, ...] = ("abs", "rel")
    threshold: float | None = None


DEFAULT_ANALYZER_SPECS: list[AnalyzerSpec] = [
    # Thresholds are a-priori conventions, not tuned to any dataset:
    # spearman 0.30 = Cohen's "medium" monotone association;
    # chatterjee 0.05 = Gaussian-equivalent of r=0.3 via
    # ξ(ρ) = (3/π)·arcsin((1+ρ²)/2) − 1/2 (r=0.5 ↔ ξ≈0.145 for a
    # stricter cut).
    AnalyzerSpec("spearman",        "r",                True,  "p",       ("abs", "rel"), threshold=0.30),
    AnalyzerSpec("chatterjee",      "xi",               False, "p",       ("abs", "rel"), threshold=0.05),
    AnalyzerSpec("lme",             "coef",             True,  "p",       ("abs",)),
    AnalyzerSpec("cross_model_lme", "effective_slope",   True,  "slope_p", ("abs",)),
]

# Pseudo-model key used internally (and in output files) for scope="union",
# where per-model selections are merged before the category intersection.
UNION_MODEL_KEY = "(union)"


class FeatureSelectionPipeline:
    """Select important dataset-level meta-features from analyzer outputs.

    Reads existing ``summary.csv`` files produced by
    :class:`DatasetAnalysisPipeline` and applies a two-stage filter **per
    ratio** (ratios are never mixed):

    1. **Per (ratio, analyzer, kind, response[, alpha])**: select features
       per model — top-*k* ranked by the analyzer's metric
       (``selection="topk"``), all features with importance above the
       analyzer's threshold and ``p <= p_max`` (``selection="threshold"``),
       or the threshold survivors capped at the top *k*
       (``selection="threshold_topk"``).
       With ``scope="union"`` the per-model sets are then unioned across
       models; with ``scope="per_model"`` they are kept separate.  For
       alpha-dependent responses, union across alphas.
    2. **Per (ratio, analyzer, kind, category[, model])**: intersect the
       per-response feature sets within each response category (prediction
       / calibration / pred+cal) — once per model in per-model scope.

    The result is one feature set per ``(ratio, analyzer, kind, category)``
    triple (per model in per-model scope).  No cross-category or
    cross-analyzer consensus is computed; downstream users decide how to
    combine.

    In threshold mode, analyzers whose metric is unbounded on ℝ
    (``threshold=None`` in their spec: lme, cross_model_lme) are skipped —
    a fixed importance cutoff is meaningless for response-scale-dependent
    coefficients.

    Parameters
    ----------
    analysis_dir:
        Root directory containing analyzer subdirectories
        (e.g. ``eval_results/dataset_level_analysis_reg``).
    task:
        ``"regression"`` or ``"classification"``.
    ratio:
        When set, restrict selection to this ratio.  When ``None`` (default),
        auto-detect every ratio present in the summary tables and run
        selection independently for each.
    categories_abs:
        Response categories for abs mode.  Defaults to task-specific
        constants from ``selected_features.py``.
    categories_rel:
        Response categories for rel mode.  Defaults to task-specific
        constants from ``selected_features.py``.
    analyzer_specs:
        Which analyzers to read and how to rank their features.
        Defaults to :data:`DEFAULT_ANALYZER_SPECS`.
    selected_features:
        Feature scope (only these features are considered).  Defaults to
        :data:`SELECTED_DATASET_FEATURES_BY_TASK[task]`.
    k:
        Number of top features to keep per model (``selection="topk"``).
    scope:
        ``"both"`` (default): emit the union result *and* every per-model
        result in a single run — the union matrix
        (``summary_{kind}_ratio_{r}.csv``), one per-model matrix each
        (``..._model_{m}.csv``), and a combined ``by_category.csv`` with
        both ``(union)`` and per-model rows.  ``"union"``: merge per-model
        selections across models before the category intersection only.
        ``"per_model"``: keep each model's selections separate, yielding
        one feature set per (model, category).
    selection:
        ``"topk"`` (default): top-*k* per model.  ``"threshold"``: keep
        features with ranking value >= the analyzer's ``threshold`` and
        ``p <= p_max``; analyzers with ``threshold=None`` are skipped.
        ``"threshold_topk"``: apply the same threshold + p gate, then cap
        each model×response set at the top ``k`` by ranking value (keep
        all when fewer than ``k`` survive) — a relevance floor plus a
        parsimony cap.  Like ``"threshold"`` it skips ``threshold=None``
        analyzers.
    thresholds:
        Optional per-analyzer threshold overrides, e.g.
        ``{"spearman": 0.5, "chatterjee": 0.15}``.  Ignored (with a
        warning) for analyzers whose spec has ``threshold=None``.
    p_max:
        p-value gate for threshold mode (default 0.05).  ``None`` disables
        the p filter.
    output_dir:
        Where to write result CSVs.  Defaults to
        ``<analysis_dir>/feature_selection/``.  All files are written directly
        under ``output_dir``.  ``feature_selection_detail.csv`` and
        ``feature_selection_by_category.csv`` contain all ratios (with a
        ``ratio`` column).  The per-kind summary matrices are named
        ``summary_{kind}_ratio_{r}.csv`` so different ratios never overwrite
        each other; per-model scope appends ``_model_{m}``.
    """

    def __init__(
        self,
        analysis_dir: str | Path,
        task: str,
        *,
        ratio: float | None = None,
        categories_abs: list[ResponseCategory] | None = None,
        categories_rel: list[ResponseCategory] | None = None,
        analyzer_specs: list[AnalyzerSpec] | None = None,
        selected_features: tuple[str, ...] | None = None,
        k: int = 10,
        scope: str = "both",
        selection: str = "topk",
        thresholds: dict[str, float] | None = None,
        p_max: float | None = 0.05,
        output_dir: str | Path | None = None,
    ) -> None:
        self.analysis_dir = Path(analysis_dir)
        self.task = task
        self.ratio = float(ratio) if ratio is not None else None
        self.categories_abs = (
            categories_abs
            if categories_abs is not None
            else RESPONSE_CATEGORIES_BY_TASK_KIND[(task, "abs")]
        )
        self.categories_rel = (
            categories_rel
            if categories_rel is not None
            else RESPONSE_CATEGORIES_BY_TASK_KIND[(task, "rel")]
        )
        self.analyzer_specs = list(analyzer_specs or DEFAULT_ANALYZER_SPECS)
        self.selected_features = set(
            selected_features
            if selected_features is not None
            else SELECTED_DATASET_FEATURES_BY_TASK[task]
        )
        self.k = k
        if scope not in ("union", "per_model", "both"):
            raise ValueError(
                "scope must be 'union', 'per_model', or 'both', got "
                f"{scope!r}"
            )
        if selection not in ("topk", "threshold", "threshold_topk"):
            raise ValueError(
                "selection must be 'topk', 'threshold', or 'threshold_topk', "
                f"got {selection!r}"
            )
        self.scope = scope
        self.selection = selection
        self.p_max = p_max
        if thresholds:
            by_name = {s.name: i for i, s in enumerate(self.analyzer_specs)}
            for name, thr in thresholds.items():
                i = by_name.get(name)
                if i is None:
                    log.warning(
                        "Threshold override for unknown analyzer %r — "
                        "ignored.", name,
                    )
                elif self.analyzer_specs[i].threshold is None:
                    log.warning(
                        "Analyzer %r has an unbounded metric; threshold "
                        "override ignored (skipped in threshold mode).", name,
                    )
                else:
                    # replace() keeps the shared DEFAULT_ANALYZER_SPECS
                    # instances untouched.
                    self.analyzer_specs[i] = dc_replace(
                        self.analyzer_specs[i], threshold=float(thr),
                    )
        self.output_dir = (
            Path(output_dir) if output_dir is not None
            else self.analysis_dir / "feature_selection"
        )

    # ── Public entry point ──────────────────────────────────────────────────

    def run(self) -> dict:
        """Run the full pipeline.

        Returns
        -------
        dict
            ``results_by_ratio``: ``{ratio: {(analyzer, kind, category):
            {"by_model": {model: {"per_response": {resp: set},
            "intersection": set}}}}}`` — the model key is
            :data:`UNION_MODEL_KEY` in union scope, where the top-level
            ``"per_response"`` / ``"intersection"`` aliases are also kept
            for backward compatibility.  Plus ``detail_rows`` (all ratios,
            each row tagged with ``ratio``).

        Output files (all written directly under ``output_dir``):

        * ``feature_selection_detail.csv`` — one file, all ratios
          (distinguished by the ``ratio`` column).
        * ``feature_selection_by_category.csv`` — one file, all ratios
          (distinguished by the ``ratio`` column); has a ``model`` column
          (``"(union)"`` and/or per-model names; default ``scope="both"``
          includes both).
        * ``summary_{kind}_ratio_{r}.csv`` / ``.html`` — one per
          (kind, ratio); per-model results append ``_model_{m}``.
        """
        ratios = self._collect_ratios()
        if not ratios:
            log.warning(
                "No summary.csv files found under %s — nothing to select.",
                self.analysis_dir,
            )
            return {"results_by_ratio": {}, "detail_rows": []}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_by_ratio: dict[object, dict[tuple[str, str, str], dict]] = {}
        all_detail_rows: list[dict] = []

        for i, ratio in enumerate(ratios, 1):
            ratio_label = ratio if ratio is not None else "all"
            log.info(
                "Feature selection for ratio=%s (%d / %d) ...",
                ratio_label, i, len(ratios),
            )
            results, detail_rows = self._run_for_ratio(ratio)
            if not results and not detail_rows:
                log.warning(
                    "No selectable rows for ratio=%s under %s",
                    ratio_label, self.analysis_dir,
                )
                continue

            self._save_kind_summaries(
                results, out_dir=self.output_dir, ratio=ratio,
            )
            results_by_ratio[ratio] = results
            all_detail_rows.extend(detail_rows)

        # detail + by_category: one file each, all ratios merged.
        all_by_cat_rows = self._build_by_category_rows(results_by_ratio)
        self._save_merged(all_detail_rows, all_by_cat_rows)

        return {
            "results_by_ratio": results_by_ratio,
            "detail_rows": all_detail_rows,
        }

    def _run_for_ratio(
        self,
        ratio: object,
    ) -> tuple[dict[tuple[str, str, str], dict], list[dict]]:
        """Run selection for one ratio slice."""
        from .analysis.summary_io import filter_ratio

        results: dict[tuple[str, str, str], dict] = {}
        detail_rows: list[dict] = []

        for spec in self.analyzer_specs:
            if (
                self.selection in ("threshold", "threshold_topk")
                and spec.threshold is None
            ):
                log.info(
                    "selection=%r: skipping %s (unbounded metric).",
                    self.selection, spec.name,
                )
                continue
            for kind in spec.kinds:
                summary_path = (
                    self.analysis_dir / spec.name / kind / "summary.csv"
                )
                if not summary_path.is_file():
                    log.warning(
                        "Skipping %s/%s — summary.csv not found at %s",
                        spec.name, kind, summary_path,
                    )
                    continue

                df = self._load_summary(summary_path, spec)
                if ratio is not None:
                    df = filter_ratio(df, float(ratio))
                    if "ratio" in df.columns:
                        uniq = pd.to_numeric(
                            df["ratio"], errors="coerce",
                        ).dropna().unique()
                        if len(uniq) > 1:
                            log.warning(
                                "After ratio=%s filter, %s/%s still has "
                                "multiple ratios %s — skipping slice.",
                                ratio, spec.name, kind,
                                sorted(map(float, uniq)),
                            )
                            continue
                if df.empty:
                    log.warning(
                        "Skipping %s/%s ratio=%s — no rows after scoping.",
                        spec.name, kind, ratio,
                    )
                    continue

                categories = (
                    self.categories_abs if kind == "abs"
                    else self.categories_rel
                )
                for category in categories:
                    cat_result, cat_detail = self._select_for_category(
                        df, category, spec, kind, ratio=ratio,
                    )
                    results[(spec.name, kind, category.name)] = cat_result
                    detail_rows.extend(cat_detail)

        return results, detail_rows

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load_summary(
        self,
        path: Path,
        spec: AnalyzerSpec,
    ) -> pd.DataFrame:
        """Read a summary.csv, normalise columns, scope to selected features."""
        # Import lazily to avoid circular imports during package initialization.
        from .analysis.summary_io import load_summary

        df = load_summary(path)

        # Scope to selected features; drop LME Intercept.
        if "feature" in df.columns:
            df = df[
                df["feature"].isin(self.selected_features)
                & (df["feature"] != "Intercept")
            ]

        return df.reset_index(drop=True)

    def _collect_ratios(self) -> list[object]:
        """Return the ratio slices to process (never mixed in one pass)."""
        if self.ratio is not None:
            return [float(self.ratio)]

        ratio_set: set[float] = set()
        saw_ratio_col = False

        for spec in self.analyzer_specs:
            for kind in spec.kinds:
                summary_path = (
                    self.analysis_dir / spec.name / kind / "summary.csv"
                )
                if not summary_path.is_file():
                    continue
                df = self._load_summary(summary_path, spec)
                if df.empty or "ratio" not in df.columns:
                    continue
                saw_ratio_col = True
                for r in pd.to_numeric(df["ratio"], errors="coerce").dropna():
                    ratio_set.add(float(r))

        if not saw_ratio_col:
            return [None]
        return sorted(ratio_set)

    @staticmethod
    def _summary_kind_basename(kind: str, ratio: object) -> str:
        """``summary_abs`` or ``summary_abs_ratio_1.0`` (matches analyzer naming)."""
        if ratio is None:
            return f"summary_{kind}"
        return f"summary_{kind}_ratio_{ratio}"

    def _build_by_category_rows(
        self,
        results_by_ratio: dict[object, dict[tuple[str, str, str], dict]],
    ) -> list[dict]:
        """Assemble by-category summary rows across all ratios.

        One row per (ratio, analyzer, kind, category, model, feature);
        ``model`` is :data:`UNION_MODEL_KEY` in union scope, so per-model
        scope multiplies the row count by the number of models.
        """
        rows: list[dict] = []
        for ratio, results in sorted(
            results_by_ratio.items(),
            key=lambda kv: (kv[0] is None, kv[0]),
        ):
            for (analyzer, kind, category), data in sorted(results.items()):
                for model, mdata in sorted(data["by_model"].items()):
                    all_feats: set[str] = set()
                    for feats in mdata["per_response"].values():
                        all_feats |= feats
                    for feat in sorted(all_feats):
                        present_in = [
                            resp
                            for resp, feats in mdata["per_response"].items()
                            if feat in feats
                        ]
                        row: dict = {
                            "analyzer": analyzer,
                            "kind": kind,
                            "category": category,
                            "model": model,
                            "feature": feat,
                            "n_responses_present": len(present_in),
                            "n_responses_total": len(mdata["per_response"]),
                            "responses_present": ",".join(sorted(present_in)),
                            "in_intersection": feat in mdata["intersection"],
                        }
                        if ratio is not None:
                            row = {"ratio": float(ratio), **row}
                        rows.append(row)
        return rows

    def _save_merged(
        self,
        detail_rows: list[dict],
        by_cat_rows: list[dict],
    ) -> None:
        """Write the two merged (all-ratio) CSVs to ``output_dir``."""
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        if detail_rows:
            detail_df = pd.DataFrame(detail_rows)
            detail_df[self._csv_columns(detail_df)].to_csv(
                out / "feature_selection_detail.csv", index=False,
            )
            log.info(
                "Wrote %s/feature_selection_detail.csv (%d rows)",
                out, len(detail_df),
            )

        if by_cat_rows:
            cat_df = pd.DataFrame(by_cat_rows)
            cat_df[self._csv_columns(cat_df)].to_csv(
                out / "feature_selection_by_category.csv", index=False,
            )
            log.info(
                "Wrote %s/feature_selection_by_category.csv (%d rows)",
                out, len(cat_df),
            )

    @staticmethod
    def _csv_columns(df: pd.DataFrame) -> list[str]:
        """Column order = dict insertion order when rows were built."""
        return list(df.columns)

    def _apply_selection(
        self,
        m_sub: pd.DataFrame,
        spec: AnalyzerSpec,
    ) -> pd.DataFrame:
        """Select rows from one model's slice (sorted by ``_rank_val`` desc).

        ``"topk"`` keeps the top ``self.k`` rows; ``"threshold"`` keeps rows
        with ``_rank_val >= spec.threshold`` and ``p <= self.p_max`` (NaN
        values fail both comparisons and are dropped); ``"threshold_topk"``
        applies the same threshold + p gate then caps the survivors at the
        top ``self.k``.  When the ``p`` column is missing the p gate is
        skipped with a warning.
        """
        if self.selection in ("threshold", "threshold_topk"):
            keep = m_sub["_rank_val"] >= spec.threshold
            if self.p_max is not None:
                if "p" in m_sub.columns:
                    keep &= m_sub["p"] <= self.p_max
                else:
                    log.warning(
                        "%s: no 'p' column — threshold selection without "
                        "p-value gate.", spec.name,
                    )
            selected = m_sub[keep]
            # Hybrid: cap the threshold-passing set at the top k.  m_sub is
            # pre-sorted by _rank_val desc and the boolean mask preserves
            # order, so head(k) keeps the strongest k; fewer than k passing
            # means all are kept.
            if self.selection == "threshold_topk":
                selected = selected.head(self.k)
            return selected
        return m_sub.head(self.k)

    def _select_for_response(
        self,
        df: pd.DataFrame,
        response: str,
        alphas: list[float] | None,
        spec: AnalyzerSpec,
    ) -> tuple[dict[str, set[str]], list[dict]]:
        """Per-model selection for one response (unioned across alphas).

        Returns ``({model: feature_set}, detail_rows)``.  The caller unions
        across models (``scope="union"``) or keeps them separate
        (``scope="per_model"``).  Every model present in a slice gets an
        entry, even when no feature survives selection — an empty set must
        count as "selected nothing", not as a missing response.
        """
        value_col = "value"  # normalised by load_summary
        by_model: dict[str, set[str]] = {}
        detail_rows: list[dict] = []

        # Determine alpha slices to process.
        if alphas:
            alpha_slices = alphas
        else:
            alpha_slices = [None]

        for alpha in alpha_slices:
            sub = df[df["response"] == response].copy()
            if alpha is not None:
                sub = sub[
                    pd.to_numeric(sub["alpha"], errors="coerce")
                    .sub(alpha).abs().lt(1e-9)
                ]
            else:
                # alpha-free: keep rows where alpha is NaN or absent.
                if "alpha" in sub.columns:
                    sub = sub[sub["alpha"].isna()]

            if sub.empty:
                continue

            # Rank per model.
            if spec.rank_by_abs:
                sub = sub.assign(_rank_val=sub[value_col].abs())
            else:
                sub = sub.assign(_rank_val=sub[value_col])

            for model in sorted(sub["model"].dropna().unique()):
                by_model.setdefault(model, set())
                m_sub = sub[sub["model"] == model].copy()
                m_sub = m_sub.sort_values("_rank_val", ascending=False)
                selected = self._apply_selection(m_sub, spec)

                for rank_i, (_, row) in enumerate(selected.iterrows(), 1):
                    feat = row["feature"]
                    by_model[model].add(feat)
                    detail_rows.append({
                        "model": model,
                        "feature": feat,
                        "rank": rank_i,
                        "value": row.get(value_col, float("nan")),
                        "p": row.get("p", float("nan")),
                        "alpha": alpha,
                    })

        return by_model, detail_rows

    def _select_for_category(
        self,
        df: pd.DataFrame,
        category: ResponseCategory,
        spec: AnalyzerSpec,
        kind: str,
        *,
        ratio: object = None,
    ) -> tuple[dict, list[dict]]:
        """Process one (analyzer, kind, category) triple.

        Returns ``(result_dict, detail_rows)`` where ``result_dict`` has a
        ``"by_model"`` key mapping each model (or :data:`UNION_MODEL_KEY`
        in union scope) to ``{"per_response": {resp: set},
        "intersection": set}``.  In union scope the top-level
        ``"per_response"`` / ``"intersection"`` aliases are kept for
        backward compatibility.  A model's intersection runs over the
        responses where that model had data.
        """
        per_response_by_model: dict[str, dict[str, set[str]]] = {}
        all_detail: list[dict] = []

        for response in category.responses:
            # Check if this response exists in the data.
            if response not in df["response"].unique():
                log.info(
                    "Response %r not found in data for %s/%s; skipping.",
                    response, spec.name, kind,
                )
                continue

            alphas = category.alpha_dependent.get(response)
            by_model, detail = self._select_for_response(
                df, response, alphas, spec,
            )
            if self.scope in ("union", "both"):
                feats = (
                    set().union(*by_model.values()) if by_model else set()
                )
                per_response_by_model.setdefault(
                    UNION_MODEL_KEY, {},
                )[response] = feats
            if self.scope in ("per_model", "both"):
                for model, feats in by_model.items():
                    per_response_by_model.setdefault(
                        model, {},
                    )[response] = feats

            # Assemble detail rows in pipeline loop / field order.
            for partial in detail:
                entry: dict = {
                    "analyzer": spec.name,
                    "kind": kind,
                    "category": category.name,
                    "response": response,
                    "alpha": partial["alpha"],
                    "model": partial["model"],
                    "feature": partial["feature"],
                    "rank": partial["rank"],
                    "value": partial["value"],
                    "p": partial["p"],
                }
                if ratio is not None:
                    entry = {"ratio": float(ratio), **entry}
                all_detail.append(entry)

        # Intersection across all responses in this category, per model.
        by_model_result = {
            model: {
                "per_response": per_resp,
                "intersection": (
                    set.intersection(*per_resp.values())
                    if per_resp else set()
                ),
            }
            for model, per_resp in per_response_by_model.items()
        }

        result: dict = {"by_model": by_model_result}
        if self.scope in ("union", "both"):
            union_data = by_model_result.get(
                UNION_MODEL_KEY, {"per_response": {}, "intersection": set()},
            )
            result["per_response"] = union_data["per_response"]
            result["intersection"] = union_data["intersection"]
        return result, all_detail

    def _save_kind_summaries(
        self,
        results: dict[tuple[str, str, str], dict],
        *,
        out_dir: Path | None = None,
        ratio: object = None,
    ) -> None:
        """Write ``summary_abs_ratio_<r>.csv`` / ``summary_rel_ratio_<r>.csv``.

        Each file has a two-level column header::

            analyzer   | spearman              | chatterjee            | ...
            category   | prediction | cal | .. | prediction | cal | .. |
            -----------|------------|-----|----|-----------  |-----|-----|
            feature_0  |     X      |  X  |    |     X       |     |    |
            feature_1  |            |  X  |    |     X       |  X  |    |

        Rows are the union of all intersection features across
        (analyzer, category) for that kind.  An ``X`` marks membership.
        In per-model scope one file per model is written, suffixed
        ``_model_{m}``; an (analyzer, category) column is omitted for
        models absent from that slice (no data), while an empty column
        means an empty intersection.
        """
        out_dir = out_dir if out_dir is not None else self.output_dir
        kinds_seen: set[str] = set()
        for (_, kind, _) in results:
            kinds_seen.add(kind)

        models = sorted({
            m for data in results.values() for m in data["by_model"]
        })
        for model in models:
            for kind in sorted(kinds_seen):
                # Collect (analyzer, category) → sorted intersection list.
                columns: list[tuple[str, str]] = []
                col_features: dict[tuple[str, str], list[str]] = {}
                for (analyzer, k, category), data in sorted(results.items()):
                    if k != kind or model not in data["by_model"]:
                        continue
                    key = (analyzer, category)
                    columns.append(key)
                    col_features[key] = sorted(
                        data["by_model"][model]["intersection"]
                    )

                if not columns:
                    continue

                # Row index: union of all intersection features, sorted.
                all_feats = sorted(
                    set().union(*(set(v) for v in col_features.values()))
                )
                if not all_feats:
                    log.warning(
                        "summary_%s.csv (model=%s): no features survived "
                        "any intersection — skipping.",
                        kind, model,
                    )
                    continue

                # Build DataFrame with MultiIndex columns.
                mi = pd.MultiIndex.from_tuples(
                    columns, names=["analyzer", "category"],
                )
                df = pd.DataFrame(index=all_feats, columns=mi)
                df.index.name = "feature"

                for key, feats in col_features.items():
                    feat_set = set(feats)
                    for f in all_feats:
                        df.loc[f, key] = "X" if f in feat_set else ""

                base = self._summary_kind_basename(kind, ratio)
                if model != UNION_MODEL_KEY:
                    base += f"_model_{self._sanitize_filename(model)}"
                csv_path = out_dir / f"{base}.csv"
                df.to_csv(csv_path)
                log.info(
                    "Wrote %s (%d features × %d columns)",
                    csv_path, len(all_feats), len(columns),
                )

                # HTML — styled for easy browsing.
                html_path = out_dir / f"{base}.html"
                self._write_summary_html(
                    df, kind, html_path, ratio=ratio,
                    model=None if model == UNION_MODEL_KEY else model,
                )
                log.info("Wrote %s", html_path.name)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Make a model name safe for filenames (spaces/slashes → '_')."""
        return re.sub(r"[^\w.\-]+", "_", name)

    @staticmethod
    def _write_summary_html(
        df: pd.DataFrame,
        kind: str,
        path: Path,
        *,
        ratio: object = None,
        model: str | None = None,
    ) -> None:
        """Render the summary DataFrame as styled HTML via pandas Styler.

        Same approach as :func:`render_feature_pivot_html` in
        ``evaluation/analysis/base.py``: build a ``Styler`` from a pivot,
        apply background colour for selected cells, and call
        :meth:`pandas.io.formats.style.Styler.to_html`.  This produces
        markup that matches the look of the other ``summary*.html``
        pivots written by the analyzers.
        """
        n_feats, n_cols = df.shape
        counts = (df == "X").sum(axis=1)
        display = df.copy()
        display["count"] = counts.astype(str) + f"/{n_cols}"

        # Style: green background for "X" cells, neutral otherwise.
        def _color_x(v: object) -> str:
            return (
                "background-color: #d1fae5; color: #065f46; font-weight: 600;"
                if v == "X" else ""
            )

        styled = (
            display.style
            .map(_color_x, subset=df.columns)
            .set_caption(
                f"Feature selection summary — <code>{kind}</code>"
                + (f", ratio={ratio}" if ratio is not None else "")
                + (f", model={model}" if model is not None else "")
                + f" ({n_feats} features × {n_cols} (analyzer, category) "
                "columns). Cells marked <b>X</b> survived the "
                "within-category intersection for that analyzer."
            )
            .set_table_styles([
                {"selector": "th, td",
                 "props": "border: 1px solid #ccc; padding: 4px 10px; "
                          "text-align: center; font-size: 13px;"},
                {"selector": "th",
                 "props": "background: #f5f5f5;"},
                {"selector": "th.col_heading.level0",
                 "props": "background: #e0e7ff;"},
                {"selector": "th.col_heading.level1",
                 "props": "background: #f0f0f0; font-weight: normal;"},
                {"selector": "th.row_heading",
                 "props": "text-align: left; font-weight: 600;"},
                {"selector": "caption",
                 "props": "caption-side: top; text-align: left; "
                          "padding: 8px 0; font-size: 14px;"},
                {"selector": "table",
                 "props": "border-collapse: collapse; "
                          "font-family: -apple-system, BlinkMacSystemFont, "
                          "'Segoe UI', Roboto, Helvetica, Arial, sans-serif;"},
                {"selector": "tbody tr:hover",
                 "props": "background: #fafafa;"},
            ])
        )
        path.write_text(styled.to_html(), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# InstanceAnalysisPipeline
# ─────────────────────────────────────────────────────────────────────────────

# Default scale-invariant feature subset used when the caller doesn't pass an
# explicit ``feature_cols``. Excludes raw-y-unit features (``ppd_mean``,
# ``pi_lower / pi_upper / pi_width``, ``ppd_std``, ``ppd_iqr``, ``ppd_median``,
# ``y_ctx_mean``, ``y_ctx_std``, ``local_y_*``, raw ``knn_dist_k*``) and the
# constant-within-dataset ``n_context`` so that GroupKFold AUC across datasets
# isn't dominated by per-dataset scale.
DEFAULT_INSTANCE_FEATURE_COLS: list[str] = [
    # distance / outlier (scale-invariant)
    "knn_dist_k1_norm", "knn_dist_k5_norm", "knn_dist_k10_norm",
    "ctx_density", "mahal_dist", "isolation_score", "lof_score",
    # PPD shape (dimensionless)
    "ppd_skew_bowley", "ppd_skew_pearson", "ppd_skew_moment",
    "ppd_kurtosis_excess", "ppd_kurtosis_moors",
    "ppd_bimodality", "ppd_entropy", "ppd_tail_ratio", "ppd_cv",
    "pi_asymmetry",
    # context-derived but standardised
    "log_n_context", "y_ctx_skew", "y_ctx_kurtosis",
    "pred_mean_standardized", "abs_pred_mean_std", "pi_width_norm",
    # input quality
    "nan_count_x", "nan_frac_x",
]


def _all_instance_feature_cols() -> list[str]:
    """Flat list of every instance-level feature name in registration order."""
    cols: list[str] = []
    for grp in DEFAULT_INSTANCE_GROUPS:
        cols.extend(grp.feature_names)
    return cols


# Continuous outcomes the regression analyzer knows how to handle by default.
# NOTE: ``pi_width_norm`` is also listed in ``DEFAULT_INSTANCE_FEATURE_COLS``;
# when it is used as the regression response, ``run()`` automatically excludes
# it from the predictor list (see ``feats_resp`` below) so the regressor never
# sees the leak.
# Winkler regression / importance analysis disabled; uncomment ``winkler`` below to restore.
DEFAULT_INSTANCE_REG_RESPONSES: list[str] = [
    # "winkler",
    "pi_width_norm",
]


class InstanceAnalysisPipeline:
    """Per-test-point coverage / interval analysis.

    Counterpart to :class:`DatasetAnalysisPipeline`: instead of explaining
    one calibration scalar per dataset with meta-features, this one
    explains *each test point's* coverage outcome with per-instance
    features (k-NN distance to context, PPD shape, NaN load, ...).

    Workflow::

        InstanceTable(store, alpha)
            → long DataFrame (n_test rows × instance feature cols)
        per-model:
            InstanceCoverageAnalyzer       → coverage AUC + importance
            InstanceWidthAnalyzer × N      → R² + importance for each
                                             continuous response

    Outputs (under ``output_dir``)::

        instance_long.parquet                    full long table (optional)
        instance_summary.csv                     n_obs, base_rate, AUC per
                                                 model
        coverage/<model>_importance.csv          LR + XGB SHAP importance
        coverage/<model>_cv_aucs.csv             per-fold AUC arrays
        regression/<model>_<response>_importance.csv  Ridge + XGB SHAP
        regression/<model>_<response>_cv_r2.csv
        summary.txt                              human-readable digest
    """

    def __init__(
        self,
        store: ArtifactStore,
        alpha: float,
        output_dir: str | Path,
        *,
        feature_cols: Optional[list[str]] = None,
        regression_responses: Optional[list[str]] = None,
        coverage_analyzer: Optional[InstanceCoverageAnalyzer] = None,
        width_analyzer: Optional[InstanceWidthAnalyzer] = None,
        chatterjee_analyzer: Optional[InstanceChatterjeeAnalyzer] = None,
        run_chatterjee: bool = True,
        save_long_table: bool = True,
        models: Optional[list[str]] = None,
        allowed_dataset_ids: Optional[Iterable[int]] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.store      = store
        self.alpha      = float(alpha)
        self.output_dir = Path(output_dir)
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.regression_responses = (
            list(regression_responses) if regression_responses is not None
            else list(DEFAULT_INSTANCE_REG_RESPONSES)
        )
        self.coverage_analyzer = coverage_analyzer or InstanceCoverageAnalyzer()
        self.width_analyzer    = width_analyzer    or InstanceWidthAnalyzer()
        self.chatterjee_analyzer = (
            chatterjee_analyzer or InstanceChatterjeeAnalyzer()
        )
        self.run_chatterjee    = bool(run_chatterjee)
        self.save_long_table   = bool(save_long_table)
        self.models            = list(models) if models is not None else None
        self.allowed_dataset_ids = (
            set(map(int, allowed_dataset_ids)) if allowed_dataset_ids is not None else None
        )

    def run(self) -> dict:
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Loading instance features (alpha=%s) ...", self.alpha)
        long_df, n_pkls = InstanceTable(self.store, self.alpha).build(
            models=self.models,
            allowed_dataset_ids=self.allowed_dataset_ids,
        )
        if long_df.empty:
            log.error("No instance feature rows loaded - check alpha / store paths.")
            return {"long": long_df}
        log.info(
            "Loaded %d test-point rows from %d PKLs (datasets=%d, models=%s)",
            len(long_df), n_pkls, long_df["dataset_id"].nunique(),
            sorted(long_df["model"].unique().tolist()),
        )

        if self.save_long_table:
            try:
                long_df.to_parquet(out_dir / "instance_long.parquet", index=False)
                log.info("Saved instance_long.parquet (%d rows)", len(long_df))
            except Exception as exc:
                log.warning("Could not write parquet (%s); falling back to csv.gz",
                            exc)
                long_df.to_csv(out_dir / "instance_long.csv.gz",
                               index=False, compression="gzip")

        feature_cols = self._resolve_feature_cols(long_df)
        models = sorted(long_df["model"].unique().tolist())
        log.info("Models=%s | features=%d (of %d available)",
                 models, len(feature_cols),
                 len(_all_instance_feature_cols()))

        # ── Coverage classifier per model ────────────────────────────────
        log.info("Fitting coverage classifier per model ...")
        cov_results = self.coverage_analyzer.fit_per_model(
            long_df, feature_cols, models, response="covered",
        )

        # ── Continuous-outcome regressors per (model, response) ──────────
        reg_results: dict[tuple[str, str], InstanceRegressionResult] = {}
        reg_summary_rows: list[dict] = []
        if self.regression_responses:
            log.info("Fitting interval-score regressors per model × response ...")
            for resp in self.regression_responses:
                if resp not in long_df.columns:
                    log.warning("Response %r not in instance table; skipping.", resp)
                    continue
                feats_resp = [f for f in feature_cols if f != resp]
                per_model = self.width_analyzer.fit_per_model(
                    long_df, feats_resp, models, response=resp,
                )
                for model, result in per_model.items():
                    reg_results[(model, resp)] = result
                    reg_summary_rows.append({
                        "model":      model,
                        "response":   resp,
                        "n_obs":      result.n_obs,
                        "n_groups":   result.n_groups,
                        "cv_r2_mean": result.cv_r2_mean,
                        "cv_r2_std":  result.cv_r2_std,
                        "cv_n_splits": result.cv_n_splits,
                        "train_r2":   result.train_r2,
                    })
        reg_summary = pd.DataFrame(reg_summary_rows)

        # ── Chatterjee ξ correlations per model ──────────────────────────
        chat_results: dict[str, ChatterjeeResult] = {}
        chat_summary = pd.DataFrame()
        if self.run_chatterjee:
            log.info("Computing Chatterjee ξ correlations per model ...")
            chat_responses = ["covered"] + [
                r for r in self.regression_responses if r in long_df.columns
            ]
            chat_results = self.chatterjee_analyzer.fit_per_model(
                long_df, feature_cols, models, chat_responses,
            )
            chat_summary = self._save_chatterjee_results(
                chat_results, out_dir / "chatterjee",
            )
            if not chat_summary.empty:
                chat_summary.to_csv(
                    out_dir / "chatterjee" / "chatterjee_summary.csv",
                    index=False,
                )

        # ── Merge ξ into each result's importances and sort ──────────────
        # (ξ > SHAP > linear coef; done here so CSVs already have the right order)
        xgb_results: dict[
            tuple[str, str],
            InstanceCoverageResult | InstanceRegressionResult,
        ] = {(m, "covered"): r for m, r in cov_results.items()}
        xgb_results.update(reg_results)
        for (model, resp), result in xgb_results.items():
            if model in chat_results:
                xi = (
                    chat_results[model].correlations
                    .loc[lambda d: d["response"] == resp, ["feature", "xi"]]
                    .rename(columns={"xi": "chatterjee_xi"})
                )
                result.importances = result.importances.merge(xi, on="feature", how="left")
            result.importances = self._sort_imp(result.importances)

        # ── Save per-model CSVs ───────────────────────────────────────────
        cov_summary = self._save_coverage_results(cov_results, out_dir / "coverage")
        if not cov_summary.empty:
            cov_summary.to_csv(out_dir / "coverage" / "coverage_summary.csv", index=False)

        if not reg_summary.empty:
            self._save_regression_results(reg_results, out_dir / "regression")
            reg_summary.to_csv(out_dir / "regression" / "regression_summary.csv", index=False)

        xgb_summary = self._save_xgb_results(xgb_results, out_dir / "xgb")
        if not xgb_summary.empty:
            xgb_summary.to_csv(out_dir / "xgb" / "xgb_summary.csv", index=False)

        # ── Combined importance summary ───────────────────────────────────
        importance_summary = self._build_importance_summary(
            xgb_results, chat_summary,
        )
        if not importance_summary.empty:
            importance_summary.to_csv(
                out_dir / "instance_importance_summary.csv", index=False,
            )

        # ── Instance-level summary report ────────────────────────────────
        rep = self._build_summary(
            long_df, models, feature_cols, cov_summary, reg_summary, n_pkls,
            xgb_summary=xgb_summary,
            chat_summary=chat_summary,
            importance_summary=importance_summary,
        )
        rep.write(out_dir / "summary.txt")
        log.info("Wrote summary.txt → %s", out_dir / "summary.txt")

        return {
            "long":               long_df,
            "feature_cols":       feature_cols,
            "coverage":           cov_results,
            "coverage_summary":   cov_summary,
            "regression":         reg_results,
            "regression_summary": reg_summary,
            "xgb_results":        xgb_results,
            "xgb_summary":        xgb_summary,
            "chatterjee_results": chat_results,
            "chatterjee_summary": chat_summary,
            "importance_summary": importance_summary,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return the feature column list, intersecting with what's in df."""
        _non_feature = INSTANCE_OUTCOME_COLS | INSTANCE_ID_COLS
        feature_pool = {c for c in df.columns if c not in _non_feature}

        requested = (list(self.feature_cols) if self.feature_cols is not None
                     else list(DEFAULT_INSTANCE_FEATURE_COLS))

        kept    = [c for c in requested if c in feature_pool]
        missing = [c for c in requested if c not in feature_pool]
        if missing:
            log.warning("Dropping %d feature(s) absent from instance table: %s",
                        len(missing), missing)
        if not kept:
            raise ValueError(
                "No usable feature columns found in the instance table. "
                "Pass --feature_cols explicitly."
            )
        return kept

    @staticmethod
    def _sort_imp(df: pd.DataFrame) -> pd.DataFrame:
        """Sort importance rows by best available metric: ξ > SHAP > linear coef."""
        for key in ("chatterjee_xi", "shap_mean_abs", "lr_coef_abs", "ridge_coef_abs"):
            if key in df.columns and df[key].notna().any():
                return df.sort_values(key, ascending=False, na_position="last").reset_index(drop=True)
        return df.reset_index(drop=True)

    @staticmethod
    def _save_coverage_results(
        results: dict[str, InstanceCoverageResult], out_dir: Path,
    ) -> pd.DataFrame:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for model, result in results.items():
            result.importances.to_csv(
                out_dir / f"{model}_importance.csv", index=False,
            )
            pd.DataFrame({
                "fold":  np.arange(len(result.cv_aucs)),
                "auc":   result.cv_aucs,
            }).to_csv(out_dir / f"{model}_cv_aucs.csv", index=False)
            rows.append({
                "model":         model,
                "n_obs":         result.n_obs,
                "n_groups":      result.n_groups,
                "base_rate":     result.base_rate,
                "cv_auc_mean":   result.cv_auc_mean,
                "cv_auc_std":    result.cv_auc_std,
                "cv_n_splits":   result.cv_n_splits,
                "train_acc":     result.train_acc,
                "train_auc":     result.train_auc,
                "n_features":    len(result.kept_features),
                "n_dropped":     len(result.dropped_features),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _save_regression_results(
        results: dict[tuple[str, str], InstanceRegressionResult], out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for (model, resp), result in results.items():
            tag = f"{model}_{resp}"
            result.importances.to_csv(
                out_dir / f"{tag}_importance.csv", index=False,
            )
            pd.DataFrame({
                "fold": np.arange(len(result.cv_r2)),
                "r2":   result.cv_r2,
            }).to_csv(out_dir / f"{tag}_cv_r2.csv", index=False)

    @staticmethod
    def _save_xgb_results(
        results: dict[
            tuple[str, str],
            InstanceCoverageResult | InstanceRegressionResult,
        ],
        out_dir: Path,
    ) -> pd.DataFrame:
        """Write per-(model, response) XGB importance CSV + OOF SHAP NPZ.

        Mirrors :meth:`DatasetAnalysisPipeline._save_rf_results` so the
        resulting files have the same on-disk layout as the dataset-level
        run; the only differences are (a) the headline score is named
        ``cv_score_*`` (R² for regression / AUC for classification) and
        (b) the ``base_rate`` column is non-NaN for classification rows.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for (model, resp), result in results.items():
            tag = f"{model}_{resp}"
            result.importances.to_csv(
                out_dir / f"{tag}_importance.csv", index=False,
            )
            if result.shap_values is not None:
                save_kwargs: dict = {
                    "shap_values":    result.shap_values,
                    "expected_value": np.array(
                        [result.expected_value], dtype=np.float32,
                    ),
                    "features": np.asarray(result.kept_features),
                }
                if result.expected_values is not None:
                    save_kwargs["expected_values"] = result.expected_values
                if result.shap_interactions is not None:
                    save_kwargs["shap_interactions"] = result.shap_interactions
                np.savez_compressed(
                    out_dir / f"{tag}_shap_values.npz", **save_kwargs,
                )
            rows.append({
                "model":         model,
                "response":      resp,
                "task":          result.task,
                "n_obs":         result.n_obs,
                "n_groups":      result.n_groups,
                "base_rate":     result.base_rate
                                  if result.base_rate is not None else np.nan,
                "cv_score_mean": result.cv_score_mean,
                "cv_score_std":  result.cv_score_std,
                "cv_n_splits":   result.cv_n_splits,
                "train_score":   result.train_score,
                "n_features":    len(result.kept_features),
                "n_dropped":     len(result.dropped_features),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _save_chatterjee_results(
        results: dict[str, ChatterjeeResult], out_dir: Path,
    ) -> pd.DataFrame:
        out_dir.mkdir(parents=True, exist_ok=True)
        long_rows: list[pd.DataFrame] = []
        for model, result in results.items():
            result.correlations.to_csv(
                out_dir / f"{model}_chatterjee.csv", index=False,
            )
            df = result.correlations.copy()
            df.insert(0, "model", model)
            long_rows.append(df)
        if not long_rows:
            return pd.DataFrame()
        return (
            pd.concat(long_rows, ignore_index=True)
            .sort_values(
                ["model", "response", "xi"],
                ascending=[True, True, False], na_position="last",
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _build_importance_summary(
        xgb_results: dict[
            tuple[str, str],
            InstanceCoverageResult | InstanceRegressionResult,
        ],
        chat_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """Concatenate per-result importances (already sorted by ξ > SHAP > coef)."""
        _keep = ["feature", "chatterjee_xi", "shap_mean_abs", "shap_mean",
                 "shap_std", "lr_coef_abs", "ridge_coef_abs"]
        rows: list[pd.DataFrame] = []
        for (model, resp), result in xgb_results.items():
            df = result.importances[[c for c in _keep if c in result.importances.columns]].copy()
            df.insert(0, "model", model)
            df.insert(1, "response", resp)
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _build_summary(
        self,
        long_df: pd.DataFrame,
        models: list[str],
        feature_cols: list[str],
        cov_summary: pd.DataFrame,
        reg_summary: pd.DataFrame,
        n_pkls: int,
        *,
        xgb_summary: pd.DataFrame = pd.DataFrame(),
        chat_summary: pd.DataFrame = pd.DataFrame(),
        importance_summary: pd.DataFrame = pd.DataFrame(),
    ) -> SummaryReporter:
        rep = SummaryReporter()
        rep.add_header("Instance-Level Calibration Analysis")
        rep.add_kv("Datasets", f"{long_df['dataset_id'].nunique()}")
        rep.add_kv("Models",   models)
        rep.add_kv("Alpha",    f"{self.alpha} (nominal {1 - self.alpha:.0%})")
        rep.add_kv("Rows",     f"{len(long_df):,} test points from {n_pkls} PKL(s)")
        rep.add_kv("Features", f"{len(feature_cols)} instance features")
        rep.add_blank()

        per_model = (
            long_df.groupby("model")
            .agg(n_rows=("covered", "size"),
                 base_rate=("covered", "mean"))
            .round(4)
        )
        rep.add_section("Coverage rate per model (raw, no subsampling)",
                        per_model.to_string())

        if not cov_summary.empty:
            cols = ["model", "n_obs", "n_groups", "base_rate",
                    "cv_auc_mean", "cv_auc_std", "cv_n_splits"]
            rep.add_section(
                "Cross-dataset coverage AUC (GroupKFold by dataset_id)",
                cov_summary[cols].round(4).to_string(index=False),
            )

            top_lines: list[str] = []
            for model, sub in cov_summary.groupby("model"):
                imp_path = (
                    self.output_dir / "coverage" / f"{model}_importance.csv"
                )
                if not imp_path.is_file():
                    continue
                imp = pd.read_csv(imp_path).head(10)
                sort_key = next(
                    (k for k in ("chatterjee_xi", "shap_mean_abs", "lr_coef_abs")
                     if k in imp.columns and imp[k].notna().any()),
                    "rank",
                )
                top_lines.append(f"\n  {model}  (top 10 by {sort_key}):")
                top_lines.append(
                    imp.round(4).to_string(index=False).replace("\n", "\n  ")
                )
            if top_lines:
                rep.add_section(
                    "Top features predicting coverage (per model)",
                    "\n".join(top_lines).lstrip("\n"),
                )

        if not reg_summary.empty:
            cols = ["model", "response", "n_obs", "n_groups",
                    "cv_r2_mean", "cv_r2_std", "cv_n_splits", "train_r2"]
            rep.add_section(
                "Cross-dataset interval-score R² (GroupKFold by dataset_id)",
                reg_summary[cols].round(4).to_string(index=False),
            )

        if not xgb_summary.empty:
            cols = [c for c in [
                "model", "response", "task", "n_obs", "n_groups", "base_rate",
                "cv_score_mean", "cv_score_std", "cv_n_splits", "train_score",
            ] if c in xgb_summary.columns]
            rep.add_section(
                "XGBoost held-out CV score per (model, response) — "
                "OOF-SHAP from the same K-fold loop "
                "(GroupKFold by dataset_id; cv_score = R² for regression "
                "rows, ROC-AUC for classification)",
                xgb_summary[cols].round(4).to_string(index=False),
            )

        if not chat_summary.empty:
            top_lines = []
            for (model, resp), grp in chat_summary.groupby(["model", "response"]):
                head = grp.head(10)[["feature", "xi", "p", "n"]].round(4)
                top_lines.append(
                    f"\n  {model}  /  {resp}  "
                    f"(top 10 by ξ(feature → {resp})):"
                )
                top_lines.append(head.to_string(index=False).replace("\n", "\n  "))
            rep.add_section(
                "Chatterjee ξ correlation per (feature, response)  "
                "(``covered`` runs with y_continuous=False)",
                "\n".join(top_lines).lstrip("\n"),
            )

        if not importance_summary.empty:
            metric_cols = [c for c in [
                "shap_mean_abs",
                "chatterjee_xi",
                "lr_coef_abs",
                "ridge_coef_abs",
            ] if c in importance_summary.columns]
            top_lines = []
            for (model, resp), grp in importance_summary.groupby(
                ["model", "response"]
            ):
                top_lines.append(
                    f"\n  {model}  /  {resp}  "
                    f"(top 10 features per importance metric):"
                )
                for metric in metric_cols:
                    sub = (
                        grp[["feature", metric]]
                        .dropna(subset=[metric])
                        .sort_values(metric, ascending=False)
                        .head(10)
                        .round(4)
                    )
                    if sub.empty:
                        continue
                    top_lines.append(f"\n    -- {metric} --")
                    top_lines.append(
                        sub.to_string(index=False).replace("\n", "\n    ")
                    )
            rep.add_section(
                "Combined feature importance per (model, response)  "
                "(XGB SHAP / linear coefficients / Chatterjee ξ "
                "— full long table in instance_importance_summary.csv)",
                "\n".join(top_lines).lstrip("\n"),
            )

        return rep


# ─────────────────────────────────────────────────────────────────────────────
# ModelComparisonPipeline
# ─────────────────────────────────────────────────────────────────────────────

class ModelComparisonPipeline:
    """Cross-model preliminary calibration analysis.

    Counterpart to :class:`DatasetAnalysisPipeline`: that one explains
    miscalibration **within** a model with meta-features, this one
    compares models **between** each other on the same metrics.

    Inputs: only metrics PKLs (no dataset-features cache needed).

    Outputs (under ``output_dir``):

      * ``summary.csv``           — long DataFrame, one row per
        (dataset, seed, ratio, model, alpha).
      * ``model_comparison.csv``  — mean / median / std per (model, alpha).
      * ``per_dataset.csv``       — mean per (dataset, model, alpha).
      * ``summary.txt``           — human-readable digest.
    """

    def __init__(
        self,
        store: ArtifactStore,
        output_dir: str | Path,
        *,
        focus_alpha: Optional[float] = None,
        worst_n: int = 15,
        extreme_n: int = 5,
        allowed_dataset_ids: Optional[Iterable[int]] = None,
    ) -> None:
        self.store        = store
        self.output_dir   = Path(output_dir)
        self.focus_alpha  = float(focus_alpha) if focus_alpha is not None else None
        self.worst_n      = int(worst_n)
        self.extreme_n    = int(extreme_n)
        self.allowed_dataset_ids = (
            set(map(int, allowed_dataset_ids)) if allowed_dataset_ids is not None else None
        )

    def run(self) -> dict:
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Loading metrics from %s ...", self.store.root)
        df = MetricsTable(self.store).build(
            alphas=[self.focus_alpha] if self.focus_alpha is not None else None,
            allowed_dataset_ids=self.allowed_dataset_ids,
        )
        if df.empty:
            log.error("No usable rows parsed from PKL files.")
            return {"df": df}

        # Back-compat alias used by old ``analyze_tabpfn_calibration``
        # consumers.
        df = df.assign(calibration_error=df["total_abs_dev"])

        alphas = sorted(df["alpha"].unique())
        focus_alpha = (self.focus_alpha if self.focus_alpha is not None
                       else float(alphas[0]))
        log.info("Loaded %d rows  (%d datasets, models=%s, alphas=%s)",
                 len(df), df["dataset_id"].nunique(),
                 sorted(df["model"].unique()), alphas)

        cmp = ModelComparator(df)

        # ── Flat tables ─────────────────────────────────────────────────────
        df.drop(columns=["_path"], errors="ignore").to_csv(
            out_dir / "summary.csv", index=False,
        )
        cmp.model_comparison().to_csv(out_dir / "model_comparison.csv")
        cmp.per_dataset_table().to_csv(out_dir / "per_dataset.csv", index=False)

        # ── Text summary ────────────────────────────────────────────────────
        rep = self._build_summary(df, cmp, focus_alpha, alphas)
        rep.write(out_dir / "summary.txt")
        log.info("Wrote %s/summary.txt", out_dir)

        return {
            "df":          df,
            "comparator":  cmp,
            "focus_alpha": focus_alpha,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        df: pd.DataFrame,
        cmp: ModelComparator,
        focus_alpha: float,
        alphas: list[float],
    ) -> SummaryReporter:
        rep = SummaryReporter()
        rep.add_header("TabPFN / TabICL Calibration — Preliminary Analysis")
        rep.add_kv("Datasets", df["dataset_id"].nunique())
        rep.add_kv("Models",   sorted(df["model"].unique().tolist()))
        rep.add_kv("Seeds",    sorted(df["seed"].unique().tolist()))
        rep.add_kv("Ratios",   sorted(df["ratio"].unique().tolist()))
        rep.add_kv("Alphas",   alphas)
        rep.add_kv("Focus",    f"{focus_alpha}  (nominal coverage {1-focus_alpha:.0%})")

        rep.add_section(
            "1. Mean calibration metrics per model",
            cmp.model_comparison_simple().to_string(),
        )
        rep.add_section(
            "1b. WSC availability per (model, alpha)",
            cmp.wsc_availability().to_string(),
        )
        rep.add_section(
            "2. Coverage bias (over- vs under-coverage frequency)",
            cmp.coverage_bias().to_string(index=False),
        )

        tally, _best = cmp.model_wins(focus_alpha)
        rep.add_section(
            f"3. Model wins on calibration_error  (alpha={focus_alpha})",
            tally.to_string(),
        )
        rep.add_section(
            f"4. Pairwise Wilcoxon tests  (alpha={focus_alpha})",
            cmp.all_pairwise_wilcoxon(focus_alpha).to_string(index=False),
        )
        rep.add_section(
            "5. Interval efficiency (mean & median across datasets)",
            cmp.interval_efficiency().to_string(index=False),
        )
        rep.add_section(
            f"6. Worst-calibrated (dataset, model) pairs  (alpha={focus_alpha})",
            cmp.worst_datasets(focus_alpha, n=self.worst_n).to_string(index=False),
        )

        ext_lines = []
        for model, payload in cmp.per_model_extremes(
            focus_alpha, n=self.extreme_n,
        ).items():
            ext_lines.append(f"\n  {model}:")
            ext_lines.append(f"    Best 5  datasets: {payload['best']}")
            ext_lines.append(f"    Worst 5 datasets: {payload['worst']}")
        rep.add_section(
            f"7. Per-model extremes on calibration_error  (alpha={focus_alpha})",
            "\n".join(ext_lines).lstrip("\n"),
        )

        rank_rows = cmp.ranking_agreement(focus_alpha)
        if rank_rows:
            rank_lines = [
                f"  {r['model_a']} vs {r['model_b']}: "
                f"ρ={r['spearman_rho']:.3f}  p={r['spearman_p']:.3f}  n={r['n']}"
                for r in rank_rows
            ]
            rep.add_section(
                "8. Cross-model ranking agreement (Spearman ρ on cal. error)",
                "\n".join(rank_lines),
            )
        return rep


# ─────────────────────────────────────────────────────────────────────────────
# ModelPairComparisonPipeline
# ─────────────────────────────────────────────────────────────────────────────

class ModelPairComparisonPipeline:
    """Meta-feature explanation of the per-dataset performance gap between two models.

    Builds the per-seed pair-delta table (signed (M_b - M_a) / M_a, sign so positive
    means M_b better) via :class:`PairDeltaTable`, then hands per-(ratio, alpha) slices
    to a :class:`DatasetAnalyzer` (default :class:`RFImportanceAnalyzer`).

    Compares **one or more** metrics in a single run (``metrics``). The default
    pairs a prediction metric with a calibration / proper-scoring metric so both
    appear side by side: classification ``["accuracy", "brier_score"]``,
    regression ``["r2", "crps_mean"]`` (:data:`DEFAULT_METRICS_BY_TASK`). Each
    metric becomes one delta column / one response row in the analyzer outputs.

    For regression, ``alphas`` controls which miscoverage levels are loaded from the
    store (same as :class:`DatasetAnalysisPipeline`). Alpha-free metrics (incl. the
    defaults) are analyzed once under the ``alpha_free`` label (deduplicated at
    ``alphas[0]``); alpha-dependent metrics get one analyzer pass per configured alpha.
    The analyzer fits per seed and aggregates importance across seeds using its
    configured :class:`SeedAggregator` (default ``MeanSDAggregator``: mean + SE +
    95 % CI + between-seed variance).

    Runs on **one task** at a time (run twice if you want both classification and
    regression). Predictor columns are scoped to
    :data:`SELECTED_DATASET_FEATURES_BY_TASK` (the manually curated lists in
    ``selected_features.py``), intersected with columns present in ``eval_long``.
    ``output_dir`` should be the existing per-task
    ``dataset_level_analysis/`` directory; the pipeline creates ``<subdir_name>/``
    inside it as a sibling of the existing analyzer outputs (``rf/``, ``lme/``, …).

    The analysis is run once per *variant*, each writing its own subdir so they
    can be compared side by side:

      * ``standard`` — continuous pair-delta, regression.
      * ``robust``   — continuous pair-delta winsorized at ``winsorize_quantiles``
        before each per-seed fit (tames heavy-tailed outliers), regression.
      * ``binary``   — response = ``1[delta > 0]`` (model_b wins), classification.
        In this subdir the ``cv_r2_mean`` column holds **ROC AUC**, and SHAP is on
        the log-odds (margin) scale.

    Outputs (under ``<output_dir>/<subdir_name>/``):

      * ``eval_pair_delta.csv``            — the per-seed pair-delta input table (shared);
        one delta column per metric.
      * ``<variant>/summary.csv``          — cross-seed aggregated importances + CV R²;
        one row per (metric, feature).
      * ``<variant>/summary_per_seed.csv`` — raw per-seed importance (stability inspection).
      * ``<variant>/summary_ratio_<r>.html`` — feature × response pivot of perm importance.
      * ``<variant>/details/*_shap.npz``   — per-seed OOF SHAP matrices.
    """

    # One prediction metric + one calibration / proper-scoring metric per
    # task, so a single run reports both side by side.
    #   r2 (not rmse): scale-free + bounded → raw-difference pair-delta, avoids
    #     the divide-by-near-zero-baseline blow-up. See PAIR_DELTA_ABSOLUTE_METRICS.
    #   crps_mean: proper score over the full PPD (scale-dependent → relative delta).
    #   brier_score: classification proper score (bounded → raw-difference delta).
    DEFAULT_METRICS_BY_TASK: dict = {
        TASK_CLASSIFICATION: ["accuracy", "brier_score"],
        TASK_REGRESSION:     ["r2", "crps_mean"],
    }

    def __init__(
        self,
        store: ArtifactStore,
        output_dir: str | Path,
        *,
        task: str,
        subdir_name: str = "comparison_pfn_icl",
        model_a: str = "tabiclv2",
        model_b: str = "tabpfnv3",
        metrics: Optional[list[str]] = None,
        alphas: Optional[list[float]] = None,
        analyzer: Optional[DatasetAnalyzer] = None,
        pseudo_model_name: Optional[str] = None,
        allowed_dataset_ids: Optional[Iterable[int]] = None,
        winsorize_quantiles: tuple[float, float] = (0.05, 0.95),
        variants: tuple[str, ...] = ("standard", "robust", "binary"),
    ) -> None:
        if task not in (TASK_CLASSIFICATION, TASK_REGRESSION):
            raise ValueError(
                f"task must be {TASK_CLASSIFICATION!r} or "
                f"{TASK_REGRESSION!r}; got {task!r}"
            )
        if task == TASK_REGRESSION:
            if alphas is None:
                alphas = [0.1]
            if not alphas:
                raise ValueError(
                    "alphas must be a non-empty list for regression."
                )
            bad = [a for a in alphas if not (0.0 < float(a) < 1.0)]
            if bad:
                raise ValueError(
                    f"every alpha must lie in (0, 1); got {bad}"
                )
        # Resolve the metric list: explicit ``metrics`` wins; else the task
        # defaults (prediction + calibration pair).
        resolved_metrics = (
            list(metrics) if metrics is not None
            else list(self.DEFAULT_METRICS_BY_TASK[task])
        )
        if not resolved_metrics:
            raise ValueError("metrics must contain at least one metric name.")
        self.store               = store
        self.output_dir          = Path(output_dir)
        self.task                = task
        self.subdir_name         = subdir_name
        self.model_a             = model_a
        self.model_b             = model_b
        self.metrics             = resolved_metrics
        self.alphas = (
            sorted({float(a) for a in alphas})
            if (task == TASK_REGRESSION and alphas)
            else None
        )
        self.analyzer            = (
            analyzer if analyzer is not None else RFImportanceAnalyzer()
        )
        self.pseudo_model_name   = pseudo_model_name
        self.allowed_dataset_ids = (
            set(map(int, allowed_dataset_ids))
            if allowed_dataset_ids is not None else None
        )
        self.winsorize_quantiles = winsorize_quantiles
        # Each variant gets its own output subdir:
        #   "standard" → continuous delta, regression, no winsorize (orig behavior)
        #   "robust"   → continuous delta, regression, winsorize at winsorize_quantiles
        #   "binary"   → response = 1[delta > 0] (model_b wins), classification (AUC)
        # ``robust``/``binary`` are skipped if the analyzer can't winsorize / classify.
        self.variants = tuple(variants)

    @staticmethod
    def _alpha_labels_for_task(
        task: str,
        alphas: Optional[list[float]],
    ) -> tuple[list[object], Optional[float]]:
        """Return (alpha_labels, dedup_alpha) for the task — mirrors DatasetAnalysisPipeline.

        Classification has no alpha dimension (one ``None`` label).
        Regression iterates ``alpha_free`` (deduped at ``alphas[0]``) plus one
        label per configured alpha; which responses apply to each label is
        decided by :meth:`_responses_for_label`.
        """
        if task == TASK_CLASSIFICATION:
            return [None], None
        assert alphas is not None
        return [EvalTable.ALPHA_FREE] + list(alphas), float(alphas[0])

    @staticmethod
    def _responses_for_label(
        alpha_label: object,
        *,
        af_metrics: list[str],
        ad_metrics: list[str],
    ) -> list[str]:
        """Pick the metric columns applicable to one alpha label.

        * ``None`` (classification) → every metric.
        * ``alpha_free`` → only alpha-free metrics.
        * ``<float>`` → only alpha-dependent metrics.
        """
        if alpha_label is None:
            return af_metrics + ad_metrics
        if alpha_label == EvalTable.ALPHA_FREE:
            return list(af_metrics)
        return list(ad_metrics)

    def run(self) -> dict:
        out_dir = self.output_dir / self.subdir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load eval_long for this task.
        log.info(
            "Loading eval_long for pair-delta comparison "
            "(task=%s, alphas=%s, model_a=%s, model_b=%s, metrics=%s) ...",
            self.task, self.alphas, self.model_a, self.model_b, self.metrics,
        )
        eval_long, n_missing = EvalTable(
            self.store, self.alphas, task=self.task,
        ).build(allowed_dataset_ids=self.allowed_dataset_ids)
        if eval_long.empty:
            log.error(
                "No eval rows built — check task / alpha / store paths."
            )
            return {"eval_long": eval_long}

        feature_cols = [
            f for f in SELECTED_DATASET_FEATURES_BY_TASK[self.task]
            if f in eval_long.columns
        ]
        present_metrics = [m for m in self.metrics if m in eval_long.columns]
        missing_metrics = [m for m in self.metrics if m not in eval_long.columns]
        if missing_metrics:
            log.warning(
                "Metric(s) %s not present in eval_long columns — dropping. "
                "Available: %s",
                missing_metrics, sorted(eval_long.columns),
            )
        if not present_metrics:
            log.error(
                "None of the requested metrics %s are present in eval_long.",
                self.metrics,
            )
            return {"eval_long": eval_long}
        log.info(
            "eval_long: %d rows, %d datasets, %d selected feature cols, "
            "%d missing feature caches; metrics=%s",
            len(eval_long), eval_long["dataset_id"].nunique(),
            len(feature_cols), n_missing, present_metrics,
        )

        # 2. Build the per-seed pair-delta table (one delta column per metric).
        pair_delta = PairDeltaTable(
            model_a=self.model_a,
            model_b=self.model_b,
            response_cols=present_metrics,
            feature_cols=feature_cols,
            pseudo_model_name=self.pseudo_model_name,
        ).build(eval_long)
        if pair_delta.empty:
            log.error(
                "Pair-delta table empty (no matched %s vs %s rows with finite %s).",
                self.model_a, self.model_b, present_metrics,
            )
            return {"eval_long": eval_long, "pair_delta": pair_delta}

        # PairDeltaTable drops direction-0 / absent metrics, so re-derive the
        # metric columns that actually survived into the table.
        kept_metrics = [m for m in present_metrics if m in pair_delta.columns]
        pseudo_model_name = pair_delta["model"].iloc[0]
        ratios = sorted(pair_delta["ratio"].unique())
        log.info(
            "pair-delta: %d rows × %d datasets × ratios=%s × metrics=%s  (pseudo_model=%s)",
            len(pair_delta), pair_delta["dataset_id"].nunique(),
            ratios, kept_metrics, pseudo_model_name,
        )
        pair_delta.to_csv(out_dir / "eval_pair_delta.csv", index=False)

        # Partition metrics into alpha-free vs alpha-dependent so each analyzer
        # pass gets only the responses valid for its alpha label.
        alpha_free_metrics = frozenset(
            eval_long.attrs.get("alpha_free_metrics", frozenset())
        )
        af_metrics = [m for m in kept_metrics if m in alpha_free_metrics]
        ad_metrics = [m for m in kept_metrics if m not in alpha_free_metrics]
        alpha_labels, dedup_alpha = self._alpha_labels_for_task(
            self.task, self.alphas,
        )
        if self.task == TASK_REGRESSION:
            log.info(
                "Alpha-free metrics %s → one pass under alpha_free (dedup at "
                "alpha=%s); alpha-dependent metrics %s → one pass per alpha=%s.",
                af_metrics, dedup_alpha, ad_metrics, self.alphas,
            )

        # 3. Per-(ratio, alpha) analyzer fits — once per variant, each into its
        #    own subdir (standard/, robust/, binary/).
        results_by_variant: dict[str, dict] = {}
        can_winsorize = hasattr(self.analyzer, "winsorize_quantiles")
        can_classify = hasattr(self.analyzer, "task")
        pair_delta_binary: Optional[pd.DataFrame] = None  # built lazily
        for variant in self.variants:
            if variant == "standard":
                pdt, variant_task, wq = pair_delta, "regression", None
            elif variant == "robust":
                if not can_winsorize:
                    log.warning(
                        "Analyzer %s does not support winsorize; skipping "
                        "'robust' variant.", type(self.analyzer).__name__,
                    )
                    continue
                pdt, variant_task, wq = pair_delta, "regression", self.winsorize_quantiles
            elif variant == "binary":
                if not can_classify:
                    log.warning(
                        "Analyzer %s does not support task switching; skipping "
                        "'binary' variant.", type(self.analyzer).__name__,
                    )
                    continue
                if pair_delta_binary is None:
                    # Response = 1[model_b wins] (delta is sign-aligned so >0 = b
                    # better). Binarize each metric column independently.
                    pair_delta_binary = pair_delta.copy()
                    for m in kept_metrics:
                        pair_delta_binary[m] = (
                            pair_delta_binary[m] > 0
                        ).astype(int)
                pdt, variant_task, wq = pair_delta_binary, "classification", None
            else:
                log.warning("Unknown pair-delta variant %r; skipping.", variant)
                continue

            # Switch the shared analyzer between sequential passes.
            if can_winsorize:
                self.analyzer.winsorize_quantiles = wq
            if can_classify:
                self.analyzer.task = variant_task

            results_by_key = self._fit_all_slices(
                pdt, ratios, alpha_labels, dedup_alpha,
                feature_cols, pseudo_model_name,
                af_metrics=af_metrics, ad_metrics=ad_metrics,
                variant=variant,
            )
            variant_dir = out_dir / variant
            if results_by_key:
                variant_dir.mkdir(parents=True, exist_ok=True)
                self.analyzer.save(results_by_key, variant_dir)
                log.info(
                    "Wrote pair-delta '%s' analysis to %s", variant, variant_dir,
                )
            else:
                log.warning(
                    "No analyzer results produced for pair-delta '%s' variant.",
                    variant,
                )
            results_by_variant[variant] = results_by_key

        return {
            "eval_long":          eval_long,
            "pair_delta":         pair_delta,
            "results_by_variant": results_by_variant,
            # Back-compat: surface the standard variant under the old key.
            "results_by_key":     results_by_variant.get("standard", {}),
            "out_dir":            out_dir,
            "pseudo_model_name":  pseudo_model_name,
            "feature_cols":       feature_cols,
            "alphas":             self.alphas,
        }

    def _fit_all_slices(
        self,
        pair_delta: pd.DataFrame,
        ratios: list,
        alpha_labels: list,
        dedup_alpha: Optional[float],
        feature_cols: list[str],
        pseudo_model_name: str,
        *,
        af_metrics: list[str],
        ad_metrics: list[str],
        variant: str,
    ) -> dict:
        """Run the analyzer over every (ratio, alpha) slice for one variant.

        Each alpha label only feeds the responses that apply to it (alpha-free
        metrics under ``alpha_free`` / classification's ``None``;
        alpha-dependent metrics under a concrete alpha).
        """
        results_by_key: dict = {}
        for ratio in ratios:
            ratio_slice = slice_table_by(pair_delta, "ratio", ratio)
            if ratio_slice.empty:
                continue
            for alpha_label in alpha_labels:
                responses = self._responses_for_label(
                    alpha_label, af_metrics=af_metrics, ad_metrics=ad_metrics,
                )
                if not responses:
                    continue
                df = EvalTable.slice_by_alpha(
                    ratio_slice,
                    alpha_label,
                    dedup_alpha=dedup_alpha,
                )
                if df.empty:
                    continue
                log.info(
                    "[pair-delta %s ratio=%s alpha=%s] running %s on %d rows × %d responses",
                    variant, ratio, alpha_label,
                    type(self.analyzer).__name__, len(df), len(responses),
                )
                results = self.analyzer.run(
                    df,
                    feature_cols=feature_cols,
                    models=[pseudo_model_name],
                    responses=responses,
                )
                if results:
                    results_by_key[(float(ratio), alpha_label)] = results
        return results_by_key
