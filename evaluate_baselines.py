"""Incrementally evaluate uncertainty baselines on one OpenML dataset.

This entry point is intentionally separate from ``evaluate_vanilla_tabpfn.py``.
It never constructs or runs TabPFN, TabICL, or TabDPT.  Point it at an
existing result root to add only these files::

    <output_dir>/<task>/predictions/..._<baseline>.pkl
    <output_dir>/<task>/metrics/..._<baseline>_metrics.pkl

Existing baseline predictions are skipped by default.  When a prediction
exists but its metrics file does not, metrics are computed from the cached
prediction without fitting the model again.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evaluation import (
    ArtifactStore,
    DatasetLoader,
    ExperimentSpec,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TASKS,
    make_quantile_grid,
)
from evaluation.baselines import (
    BARTBaseline,
    BaselinePrediction,
    RandomForestBaseline,
    RealMLPBaseline,
    RealMLPHPOBaseline,
    XGBOOST_TABARENA_ADAPTED_SEARCH_SPACE,
    XGBoostQuantileBaseline,
    XGBoostQuantileHPOBaseline,
)
from evaluation.pipelines import MetricsPipeline
from evaluation.sampling import classification_protocol, claim_classification_output


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("baseline-evaluation")

DEFAULT_ALPHAS = (0.05, 0.1, 0.15, 0.2)


@dataclass(frozen=True)
class BaselineRegistration:
    """Keep construction and metadata for one baseline in one registry entry."""

    build: Callable[[argparse.Namespace], Any]
    config: Callable[[argparse.Namespace], dict[str, Any]]
    implemented: bool = True


BASELINE_REGISTRY: dict[str, BaselineRegistration] = {
    "random_forest": BaselineRegistration(
        build=lambda args: RandomForestBaseline(
            seed=args.seed,
            n_estimators=args.rf_n_estimators,
            n_jobs=args.rf_n_jobs,
            max_features=args.rf_max_features,
        ),
        config=lambda args: {
            "implementation": "sklearn",
            "preprocessing_version": "rf_context_ordinal",
            "n_estimators": args.rf_n_estimators,
            "n_jobs": args.rf_n_jobs,
            "max_features": args.rf_max_features,
            "categorical_encoding": "ordinal",
            "categorical_fit_scope": "context_only",
            "categorical_handle_unknown": "use_encoded_value",
            "categorical_unknown_value": -1,
            "sparse_features": False,
            "numeric_imputation": "context_median_or_zero_if_all_missing",
            "numeric_normalization": "none",
            "regression_distribution": "empirical_predictions_across_trees",
        },
    ),
    "xgboost_quantile": BaselineRegistration(
        build=lambda args: XGBoostQuantileBaseline(
            seed=args.seed,
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            n_jobs=args.xgb_n_jobs,
            device=args.xgb_device,
            multi_strategy=args.xgb_multi_strategy,
        ),
        config=lambda args: {
            "implementation": "xgboost",
            "minimum_version": "2.0",
            "preprocessing_version": "rf_context_ordinal",
            "n_estimators": args.xgb_n_estimators,
            "learning_rate": args.xgb_learning_rate,
            "max_depth": args.xgb_max_depth,
            "n_jobs": args.xgb_n_jobs,
            "device": args.xgb_device,
            "tree_method": "hist",
            "multi_strategy": args.xgb_multi_strategy,
            "validation_fraction": 0.2,
            "checkpoint_selection": "best_validation_round",
            "prediction_source": "selected_validation_model",
            "full_context_refit": False,
            "validation_split": "shared_with_realmlp_td",
            "categorical_encoding": "ordinal",
            "categorical_fit_scope": "context_only",
            "categorical_handle_unknown": "use_encoded_value",
            "categorical_unknown_value": -1,
            "numeric_imputation": "context_median_or_zero_if_all_missing",
            "numeric_normalization": "none",
            "regression_objective": "reg:quantileerror",
            "regression_selection_metric": "mean_pinball",
            "regression_distribution": "direct_dense_conditional_quantiles",
            "regression_point_prediction": "quantile_grid_integral",
            "classification_selection_metric": "log_loss",
            "classification_output": "predict_proba",
        },
    ),
    "xgboost_quantile_hpo": BaselineRegistration(
        build=lambda args: XGBoostQuantileHPOBaseline(
            seed=args.seed,
            n_hyperopt_steps=args.xgb_hpo_steps,
            n_cv=args.xgb_hpo_n_cv,
            max_n_estimators=args.xgb_hpo_max_n_estimators,
            n_jobs=args.xgb_n_jobs,
            device=args.xgb_device,
            multi_strategy=args.xgb_multi_strategy,
        ),
        config=lambda args: {
            "implementation": "xgboost_random_search",
            "minimum_version": "2.0",
            "hpo_space_name": "tabarena_xgboost_adapted",
            "search_space": XGBOOST_TABARENA_ADAPTED_SEARCH_SPACE,
            "search_space_source": "TabArena Appendix C.2, Table C.3",
            "search_space_adaptations": {
                "learning_rate": (
                    "lower bound raised from 0.005 to 0.02 for the "
                    "RealMLP-aligned 256-stage budget"
                ),
                "max_cat_to_onehot": (
                    "not applicable after context-fitted ordinal encoding"
                ),
            },
            "budget_reference": "project_realmlp_hpo",
            "selection": "best_single_configuration",
            "prediction_source": "selected_configuration_cv_ensemble",
            "full_context_refit": False,
            "n_hyperopt_steps": args.xgb_hpo_steps,
            "n_cv": args.xgb_hpo_n_cv,
            "validation_scheme": (
                "holdout" if args.xgb_hpo_n_cv == 1 else "k_fold"
            ),
            "validation_fraction": (
                0.2 if args.xgb_hpo_n_cv == 1 else None
            ),
            "validation_split": "shared_with_realmlp_hpo",
            "max_n_estimators": args.xgb_hpo_max_n_estimators,
            "round_selection": "full_budget_then_best_validation_round",
            "n_jobs": args.xgb_n_jobs,
            "device": args.xgb_device,
            "tree_method": "hist",
            "multi_strategy": args.xgb_multi_strategy,
            "preprocessing_version": "rf_context_ordinal",
            "categorical_encoding": "ordinal",
            "categorical_fit_scope": "context_only_before_hpo",
            "categorical_handle_unknown": "use_encoded_value",
            "categorical_unknown_value": -1,
            "numeric_imputation": "context_median_or_zero_if_all_missing",
            "numeric_normalization": "none",
            "regression_objective": "reg:quantileerror",
            "regression_selection_metric": "mean_pinball",
            "regression_distribution": "direct_dense_conditional_quantiles",
            "regression_point_prediction": "quantile_grid_integral",
            "classification_selection_metric": "log_loss",
            "classification_output": "predict_proba",
        },
    ),
    "realmlp": BaselineRegistration(
        build=lambda args: RealMLPBaseline(
            seed=args.seed,
            device=args.realmlp_device,
            n_epochs=args.realmlp_epochs,
            n_cv=args.realmlp_n_cv,
            n_refit=args.realmlp_n_refit,
            n_ens=args.realmlp_n_ens,
            n_threads=args.realmlp_n_threads,
            verbosity=args.realmlp_verbosity,
        ),
        config=lambda args: {
            "implementation": "pytabkit.RealMLP_TD",
            "device": args.realmlp_device,
            "n_epochs": args.realmlp_epochs,
            "n_cv": args.realmlp_n_cv,
            "n_refit": args.realmlp_n_refit,
            "n_ens": args.realmlp_n_ens,
            "n_threads": args.realmlp_n_threads,
            "validation_scheme": (
                "holdout" if args.realmlp_n_cv == 1 else "k_fold"
            ),
            "validation_fraction": (
                0.2 if args.realmlp_n_cv == 1 else None
            ),
            "validation_split": (
                "shared_with_xgboost_default"
                if args.realmlp_n_cv == 1
                else "shared_k_fold_protocol"
            ),
            "prediction_source": (
                "selected_validation_model"
                if args.realmlp_n_cv == 1
                else "cv_ensemble"
            ),
            "full_context_refit": False,
            "classification_selection_metric": "cross_entropy",
            "classification_use_ls": False,
            "regression_objective": "multi_pinball",
            "regression_selection_metric": "multi_pinball",
        },
    ),
    "realmlp_hpo": BaselineRegistration(
        build=lambda args: RealMLPHPOBaseline(
            seed=args.seed,
            device=args.realmlp_device,
            n_epochs=args.realmlp_epochs,
            n_cv=args.realmlp_hpo_n_cv,
            n_refit=args.realmlp_hpo_n_refit,
            n_hyperopt_steps=args.realmlp_hpo_steps,
            n_threads=args.realmlp_n_threads,
            verbosity=args.realmlp_verbosity,
            tmp_root=args.realmlp_hpo_tmp_root,
            time_limit_s=args.realmlp_hpo_time_limit_s,
        ),
        config=lambda args: {
            "implementation": "pytabkit.RealMLP_HPO",
            "hpo_space_name": "default",
            "selection": "best_single_configuration",
            "prediction_source": "selected_configuration_cv_ensemble",
            "searched_hyperparameters": [
                "num_emb_type",
                "add_front_scale",
                "lr",
                "p_drop",
                "wd",
                "plr_sigma",
                "hidden_sizes",
                "act",
                "ls_eps (classification only)",
            ],
            "n_hyperopt_steps": args.realmlp_hpo_steps,
            "device": args.realmlp_device,
            "n_epochs_per_configuration": args.realmlp_epochs,
            "n_cv": args.realmlp_hpo_n_cv,
            "n_refit": args.realmlp_hpo_n_refit,
            "validation_scheme": (
                "holdout" if args.realmlp_hpo_n_cv == 1 else "k_fold"
            ),
            "validation_fraction": (
                0.2 if args.realmlp_hpo_n_cv == 1 else None
            ),
            "validation_split": "shared_with_xgboost_hpo",
            "n_threads": args.realmlp_n_threads,
            "tmp_root": args.realmlp_hpo_tmp_root,
            "time_limit_s": args.realmlp_hpo_time_limit_s,
            "classification_selection_metric": "cross_entropy",
            "classification_label_smoothing": "default_space_ls_eps_search",
            "regression_objective": "multi_pinball",
            "regression_selection_metric": "multi_pinball",
        },
    ),
    "bart": BaselineRegistration(
        build=lambda args: BARTBaseline(
            seed=args.seed,
            n_trees=args.bart_n_trees,
            n_draws=args.bart_n_draws,
            n_burn=args.bart_n_burn,
            n_chains=args.bart_n_chains,
            device=args.bart_device,
        ),
        config=lambda args: {
            "implementation": "bartz.Bart",
            "minimum_bartz_version": "0.12",
            "model_persistence": "embedded_bartz_dump",
            "preprocessing": "context_fitted_numeric_and_one_hot",
            "categorical_encoding": "one_hot_handle_unknown_ignore",
            "categorical_split_prior": "equal_mass_per_source_feature",
            "data_split_owner": "evaluation.DatasetLoader",
            "n_trees": args.bart_n_trees,
            "n_draws_total": args.bart_n_draws,
            "draw_semantics": "total_retained_across_chains",
            "n_burn_per_chain": args.bart_n_burn,
            "n_chains": args.bart_n_chains,
            "device": args.bart_device,
            "regression_distribution": "native_posterior_predictive_draws",
            "classification_model": (
                "binary_probit_or_normalized_multivariate_one_vs_rest_probit"
            ),
        },
    ),
}
BASELINE_NAMES = tuple(BASELINE_REGISTRY)
# The standard no-argument panel compares fixed and HPO variants of
# XGBoost-Quantile and RealMLP. RF and BART remain available as explicit
# opt-ins.
DEFAULT_BASELINE_NAMES = (
    "xgboost_quantile",
    "xgboost_quantile_hpo",
    "realmlp",
    "realmlp_hpo",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one timing record atomically for safe resumable jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _timing_path(
    timing_root: Path,
    spec: ExperimentSpec,
    model: str,
) -> Path:
    stem = spec.predictions_filename(model).removesuffix(".pkl")
    return timing_root / spec.task / f"{stem}.json"


def _baseline_config(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    try:
        registration = BASELINE_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported baseline: {model_name!r}") from exc
    return registration.config(args)


def _make_baseline(args: argparse.Namespace, model_name: str):
    try:
        registration = BASELINE_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported baseline: {model_name!r}") from exc
    return registration.build(args)


def _prediction_record(
    *,
    spec: ExperimentSpec,
    model_name: str,
    split,
    prediction: BaselinePrediction,
    q_grid: np.ndarray,
    config: dict[str, Any],
    data_load_seconds: float,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task": spec.task,
        "model": model_name,
        "dataset_id": spec.dataset_id,
        "seed": spec.seed,
        "ratio": spec.ratio,
        "n_total": len(split.y_train) + len(split.y_test),
        "n_train": len(split.y_train),
        "n_test": len(split.y_test),
        "n_context": len(split.y_ctx),
        "n_features": split.X_train.shape[1],
        "feature_names": split.feature_names,
        "y_test": split.y_test,
        "X_test": split.X_test,
        "sampling_metadata": getattr(split, "sampling_metadata", {}),
        "baseline_config": config,
        "timing_seconds": {
            "data_load_shared": float(data_load_seconds),
            **(prediction.timing_seconds or {}),
        },
    }
    if prediction.model_metadata:
        base["model_metadata"] = prediction.model_metadata
    if spec.task == TASK_REGRESSION:
        if prediction.ppd_quantiles is None or prediction.point_pred is None:
            raise ValueError(f"{model_name}: missing regression prediction output")
        return {
            **base,
            "quantile_levels": np.asarray(q_grid, dtype=np.float64),
            "ppd_quantiles": np.asarray(
                prediction.ppd_quantiles, dtype=np.float32
            ),
            "point_pred": np.asarray(prediction.point_pred, dtype=np.float32),
        }
    if prediction.proba is None or prediction.classes_ is None:
        raise ValueError(f"{model_name}: missing classification prediction output")
    return {
        **base,
        "proba": np.asarray(prediction.proba, dtype=np.float32),
        "classes_": np.asarray(prediction.classes_, dtype=np.int64),
    }


def _model_record(
    *,
    spec: ExperimentSpec,
    model_name: str,
    prediction: BaselinePrediction,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a self-describing fitted-model artifact."""
    if prediction.model_bundle is None:
        raise ValueError(f"{model_name}: fitted model bundle was not returned")
    record: dict[str, Any] = {
        "artifact_schema_version": 1,
        "artifact_type": "fitted_baseline_model",
        "task": spec.task,
        "model": model_name,
        "dataset_id": spec.dataset_id,
        "seed": spec.seed,
        "ratio": spec.ratio,
        "baseline_config": config,
        "model_metadata": prediction.model_metadata or {},
        "model_bundle": prediction.model_bundle,
    }
    if prediction.external_checkpoint_dir is not None:
        record["external_checkpoint_dir"] = prediction.external_checkpoint_dir
    return record


def run(args: argparse.Namespace) -> int:
    if any(not 0.0 < alpha < 1.0 for alpha in args.alphas):
        raise ValueError("--alphas must all lie in (0, 1)")

    store = ArtifactStore(args.output_dir)
    if args.task == TASK_CLASSIFICATION:
        claim_classification_output(args.output_dir, classification_protocol(), args.max_n)
    spec = ExperimentSpec(args.dataset_id, args.seed, args.ratio, args.task)
    timing_root = (
        Path(args.timings_dir)
        if args.timings_dir
        else Path(args.output_dir) / "baseline_timings"
    )
    q_grid = make_quantile_grid(args.quantile_step)
    metrics_pipeline = MetricsPipeline(
        store, list(args.alphas), overwrite=args.overwrite,
    )

    # Determine whether any fit is needed before downloading/loading OpenML.
    # A historical prediction without a model bundle is intentionally refit to
    # backfill the requested model artifact; the existing prediction/metrics
    # remain untouched unless --overwrite was explicitly requested.
    cached_records: dict[str, dict[str, Any]] = {}
    models_to_fit: list[str] = []
    for model_name in args.models:
        prediction_path = store.predictions_path(spec, model_name, task=args.task)
        if prediction_path.exists() and not args.overwrite:
            try:
                record = store.load_prediction(
                    spec, model_name, task=args.task
                )
            except Exception as exc:
                # A process killed while pickle.dump was running may leave a
                # truncated baseline file. Recompute that baseline only.
                log.warning(
                    "Unreadable cached baseline prediction %s (%s: %s); "
                    "refitting %s only.",
                    prediction_path,
                    type(exc).__name__,
                    exc,
                    model_name,
                )
                models_to_fit.append(model_name)
            else:
                cached_records[model_name] = record
                if store.model_exists(spec, model_name, task=args.task):
                    log.info(
                        "Cached prediction and model; will not refit %s: %s",
                        model_name,
                        prediction_path,
                    )
                else:
                    models_to_fit.append(model_name)
                    log.info(
                        "Cached prediction has no complete model artifact; "
                        "refitting %s to backfill the model only.",
                        model_name,
                    )
        else:
            models_to_fit.append(model_name)

    failures = 0
    split = None
    data_load_seconds = 0.0
    if models_to_fit:
        load_start = time.perf_counter()
        try:
            split = DatasetLoader(max_n=args.max_n).load_for_spec(
                spec, task=args.task
            )
        except Exception as exc:
            data_load_seconds = time.perf_counter() - load_start
            log.exception("Failed to load %s", spec)
            for model_name in models_to_fit:
                _atomic_json(
                    _timing_path(timing_root, spec, model_name),
                    {
                        "status": "failed",
                        "stage": "data_load",
                        "task": spec.task,
                        "dataset_id": spec.dataset_id,
                        "seed": spec.seed,
                        "ratio": spec.ratio,
                        "model": model_name,
                        "data_load_seconds": data_load_seconds,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            return 1
        data_load_seconds = time.perf_counter() - load_start
        log.info(
            "Loaded %s in %.3fs: n=%d d=%d context=%d test=%d",
            spec,
            data_load_seconds,
            len(split.y_train) + len(split.y_test),
            split.X_train.shape[1],
            len(split.y_ctx),
            len(split.y_test),
        )

    for model_name in args.models:
        config = _baseline_config(args, model_name)
        timing_path = _timing_path(timing_root, spec, model_name)
        if model_name in cached_records and model_name not in models_to_fit:
            record = cached_records[model_name]
            try:
                metrics_seconds = 0.0
                metrics_status = "disabled"
                if args.compute_metrics:
                    metrics_result = metrics_pipeline.compute_and_save(
                        record, spec, model_name, task=spec.task,
                    )
                    metrics_seconds = metrics_result.seconds
                    metrics_status = metrics_result.status
                # Preserve an existing successful timing file.  This makes a
                # no-op resumption truly non-destructive.
                if not timing_path.exists():
                    _atomic_json(
                        timing_path,
                        {
                            "status": "cached",
                            "task": spec.task,
                            "dataset_id": spec.dataset_id,
                            "seed": spec.seed,
                            "ratio": spec.ratio,
                            "model": model_name,
                            "metrics_status": metrics_status,
                            "metrics_seconds": metrics_seconds,
                            "baseline_config": record.get(
                                "baseline_config", config
                            ),
                        },
                    )
            except Exception as exc:
                failures += 1
                log.exception("Failed metrics for cached %s on %s", model_name, spec)
                _atomic_json(
                    timing_path,
                    {
                        "status": "failed",
                        "stage": "metrics",
                        "task": spec.task,
                        "dataset_id": spec.dataset_id,
                        "seed": spec.seed,
                        "ratio": spec.ratio,
                        "model": model_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            continue

        assert split is not None
        job_start = time.perf_counter()
        checkpoint_dir: Path | None = None
        checkpoint_committed = False
        previous_checkpoint_dir: str | None = None
        previous_model_path = store.models_path(spec, model_name, task=args.task)
        if previous_model_path.exists():
            try:
                previous_checkpoint_dir = store.load_model(
                    spec, model_name, task=args.task
                ).get("external_checkpoint_dir")
            except Exception:
                pass
        log.info("Fitting baseline %s only (no foundation models) ...", model_name)
        try:
            baseline = _make_baseline(args, model_name)
            set_checkpoint_dir = getattr(baseline, "set_checkpoint_dir", None)
            if callable(set_checkpoint_dir):
                checkpoint_dir = store.create_model_checkpoint_dir(
                    spec, model_name, task=args.task
                )
                set_checkpoint_dir(checkpoint_dir)
            if args.task == TASK_REGRESSION:
                prediction = baseline.fit_predict_regression(
                    split.X_ctx,
                    split.y_ctx,
                    split.X_test,
                    q_grid,
                )
            else:
                if split.classes_ is None:
                    raise ValueError("classification split has no global classes")
                prediction = baseline.fit_predict_classification(
                    split.X_ctx,
                    split.y_ctx,
                    split.X_test,
                    n_classes_global=int(split.classes_.size),
                )

            record = _prediction_record(
                spec=spec,
                model_name=model_name,
                split=split,
                prediction=prediction,
                q_grid=q_grid,
                config=config,
                data_load_seconds=data_load_seconds,
            )
            model_record = _model_record(
                spec=spec,
                model_name=model_name,
                prediction=prediction,
                config=config,
            )
            model_record["sampling_metadata"] = getattr(split, "sampling_metadata", {})
            model_save_start = time.perf_counter()
            model_path = store.save_model(
                spec, model_name, model_record, task=args.task
            )
            checkpoint_committed = True
            if (
                previous_checkpoint_dir is not None
                and previous_checkpoint_dir != prediction.external_checkpoint_dir
            ):
                try:
                    store.remove_model_checkpoint_dir(
                        previous_checkpoint_dir, task=args.task
                    )
                except Exception:
                    log.exception(
                        "Could not remove superseded checkpoint %s",
                        previous_checkpoint_dir,
                    )
            model_save_seconds = time.perf_counter() - model_save_start
            log.info("Saved fitted model → %s", model_path)

            save_start = time.perf_counter()
            if model_name in cached_records and not args.overwrite:
                prediction_path = store.predictions_path(
                    spec, model_name, task=args.task
                )
                record = cached_records[model_name]
            else:
                record["model_artifact_path"] = str(model_path)
                prediction_path = store.save_prediction(
                    spec, model_name, record, task=args.task
                )
            save_seconds = time.perf_counter() - save_start
            if model_name in cached_records and not args.overwrite:
                log.info("Preserved cached prediction → %s", prediction_path)
            else:
                log.info("Saved prediction → %s", prediction_path)

            metrics_seconds = 0.0
            metrics_status = "disabled"
            if args.compute_metrics:
                metrics_result = metrics_pipeline.compute_and_save(
                    record, spec, model_name, task=spec.task,
                )
                metrics_seconds = metrics_result.seconds
                metrics_status = metrics_result.status

            timing = dict(prediction.timing_seconds or {})
            total_seconds = time.perf_counter() - job_start
            _atomic_json(
                timing_path,
                {
                    "status": "success",
                    "task": spec.task,
                    "dataset_id": spec.dataset_id,
                    "seed": spec.seed,
                    "ratio": spec.ratio,
                    "model": model_name,
                    "n_total": record["n_total"],
                    "n_context": record["n_context"],
                    "n_test": record["n_test"],
                    "n_features": record["n_features"],
                    "data_load_seconds": data_load_seconds,
                    "preprocess_seconds": timing.get("preprocess", 0.0),
                    "fit_seconds": timing.get("fit", 0.0),
                    "predict_seconds": timing.get("predict", 0.0),
                    "prediction_save_seconds": save_seconds,
                    "model_save_seconds": model_save_seconds,
                    "model_path": str(model_path),
                    "metrics_seconds": metrics_seconds,
                    "metrics_status": metrics_status,
                    "model_total_seconds": timing.get("model_total", 0.0),
                    "job_excluding_shared_load_seconds": total_seconds,
                    "baseline_config": config,
                    "model_metadata": prediction.model_metadata or {},
                },
            )
            log.info(
                "Timing %s: fit=%.2fs predict=%.2fs metrics=%.2fs total=%.2fs",
                model_name,
                timing.get("fit", 0.0),
                timing.get("predict", 0.0),
                metrics_seconds,
                total_seconds,
            )
        except Exception as exc:
            if checkpoint_dir is not None and not checkpoint_committed:
                try:
                    store.remove_model_checkpoint_dir(
                        checkpoint_dir, task=args.task
                    )
                except Exception:
                    log.exception(
                        "Failed to clean incomplete checkpoint %s",
                        checkpoint_dir,
                    )
            failures += 1
            elapsed = time.perf_counter() - job_start
            log.exception("Failed %s on %s", model_name, spec)
            _atomic_json(
                timing_path,
                {
                    "status": "failed",
                    "stage": "fit_predict_or_metrics",
                    "task": spec.task,
                    "dataset_id": spec.dataset_id,
                    "seed": spec.seed,
                    "ratio": spec.ratio,
                    "model": model_name,
                    "data_load_seconds": data_load_seconds,
                    "elapsed_seconds": elapsed,
                    "baseline_config": config,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally add baseline predictions and metrics "
            "to an existing benchmark result root. Foundation models are never run."
        )
    )
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_NAMES,
        default=list(DEFAULT_BASELINE_NAMES),
    )
    parser.add_argument("--max_n", type=int, default=10_000)
    parser.add_argument("--quantile_step", type=float, default=0.005)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS)
    )
    parser.add_argument(
        "--compute_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute metrics immediately (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refit and overwrite existing baseline predictions and metrics.",
    )
    parser.add_argument(
        "--timings_dir",
        default=None,
        help="Default: <output_dir>/baseline_timings.",
    )

    rf = parser.add_argument_group("Random Forest")
    rf.add_argument("--rf_n_estimators", type=int, default=500)
    rf.add_argument("--rf_n_jobs", type=int, default=-1)
    rf.add_argument(
        "--rf_max_features",
        default=None,
        help="Optional sklearn max_features value, e.g. sqrt or log2.",
    )
    xgb = parser.add_argument_group("XGBoost Quantile")
    xgb.add_argument(
        "--xgb_n_estimators",
        type=int,
        default=256,
        help=(
            "Maximum boosting rounds before selecting the best validation "
            "checkpoint (default: 256, aligned with RealMLP-TD epochs)."
        ),
    )
    xgb.add_argument("--xgb_learning_rate", type=float, default=0.3)
    xgb.add_argument("--xgb_max_depth", type=int, default=6)
    xgb.add_argument("--xgb_n_jobs", type=int, default=-1)
    xgb.add_argument("--xgb_device", default="cpu")
    xgb.add_argument(
        "--xgb_multi_strategy",
        choices=("one_output_per_tree", "multi_output_tree"),
        default="one_output_per_tree",
        help=(
            "How XGBoost represents the multi-quantile output. "
            "one_output_per_tree is the conservative default."
        ),
    )
    xgb_hpo = parser.add_argument_group(
        "XGBoost Quantile TabArena-adapted HPO"
    )
    xgb_hpo.add_argument("--xgb_hpo_steps", type=int, default=50)
    xgb_hpo.add_argument(
        "--xgb_hpo_n_cv",
        type=int,
        default=1,
        help=(
            "Cross-validation folds used to score each random configuration; "
            "1 means one 80/20 holdout, K>1 means K-fold (default: 1)."
        ),
    )
    xgb_hpo.add_argument(
        "--xgb_hpo_max_n_estimators",
        type=int,
        default=256,
        help=(
            "Boosting rounds trained for every candidate before selecting "
            "its best validation round (default: 256, matching RealMLP-HPO)."
        ),
    )
    mlp = parser.add_argument_group("RealMLP-TD")
    mlp.add_argument("--realmlp_device", default="cuda")
    mlp.add_argument("--realmlp_epochs", type=int, default=256)
    mlp.add_argument("--realmlp_n_cv", type=int, default=1)
    mlp.add_argument("--realmlp_n_refit", type=int, default=0)
    mlp.add_argument("--realmlp_n_ens", type=int, default=1)
    mlp.add_argument("--realmlp_n_threads", type=int, default=None)
    mlp.add_argument("--realmlp_verbosity", type=int, default=0)

    mlp_hpo = parser.add_argument_group("RealMLP default-space HPO")
    mlp_hpo.add_argument("--realmlp_hpo_steps", type=int, default=50)
    mlp_hpo.add_argument(
        "--realmlp_hpo_n_cv",
        type=int,
        default=1,
        help=(
            "Cross-validation folds used to score each random configuration; "
            "1 means one 80/20 holdout, K>1 means K-fold (default: 1)."
        ),
    )
    mlp_hpo.add_argument("--realmlp_hpo_n_refit", type=int, default=0)
    mlp_hpo.add_argument(
        "--realmlp_hpo_tmp_root",
        default=None,
        help="Optional parent for temporary on-disk candidate models.",
    )
    mlp_hpo.add_argument(
        "--realmlp_hpo_time_limit_s",
        type=float,
        default=None,
        help="Optional PyTabKit HPO time limit in seconds per dataset.",
    )

    bart = parser.add_argument_group("BART (bartz)")
    bart.add_argument("--bart_n_trees", type=int, default=200)
    bart.add_argument(
        "--bart_n_draws",
        type=int,
        default=1_000,
        help="Total retained posterior draws across all chains.",
    )
    bart.add_argument(
        "--bart_n_burn",
        type=int,
        default=1_000,
        help="Burn-in iterations per chain.",
    )
    bart.add_argument("--bart_n_chains", type=int, default=4)
    bart.add_argument(
        "--bart_device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
