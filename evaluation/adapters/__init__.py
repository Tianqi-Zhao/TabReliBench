"""Lazy, inference-only adapters for native TFM APIs."""
from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path
from numbers import Integral
from typing import Any

import numpy as np


def resolve_class(target: Any) -> Any:
    """Resolve an optional dependency only when its model is selected."""
    if not isinstance(target, str):
        return target
    module, name = target.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Cannot load {target}; install its supported runtime "
            "(see README.md#installation)."
        ) from exc


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class NativeEstimatorAdapter:
    """Delegate sklearn-style inference without training checkpoint weights."""

    def fit(self, X, y):
        self.estimator.fit(X, y)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = np.asarray(self.estimator.classes_)
        return self

    def predict(self, X, **kwargs):
        return self.estimator.predict(X, **kwargs)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


class TabDPT13RegressorAdapter(NativeEstimatorAdapter):
    """Translate TabDPT 1.3 full distributions to the runner's quantile API.

    The official helper uses borders already restored to the target's units.
    Keep mean predictions on the native path: averaging member means is not
    generally equivalent to taking the mean of the averaged logits.
    """

    def __init__(self, model_weight_path: str, verbose: bool = False) -> None:
        from tabdpt import distribution_quantiles

        self._distribution_quantiles = distribution_quantiles
        self.estimator = resolve_class("tabdpt.TabDPTRegressor")(
            model_weight_path=model_weight_path, verbose=verbose,
        )

    def predict(self, X, output_type="mean", quantiles=None, **kwargs):
        if output_type == "quantiles":
            full = self.estimator.predict(X, output_type="full", **kwargs)
            return self._distribution_quantiles(full, quantiles)
        return self.estimator.predict(X, output_type=output_type, **kwargs)


class CausiloAdapter(NativeEstimatorAdapter):
    """Load an explicit local checkpoint into Causilo's per-estimator engine.

    Upstream has no public checkpoint-path argument. Keep the private engine
    bridge here; do not monkeypatch the package's global download function.
    """

    def __init__(
        self, model_path: str, task: str, random_state: int = 42, n_estimators: int = 8,
    ) -> None:
        from causilo.checkpoints import load_checkpoint
        from causilo.engine import Engine

        target = "CausiloClassifier" if task == "classification" else "CausiloRegressor"
        self.estimator = resolve_class(f"causilo.{target}")(
            n_estimators=n_estimators, random_state=random_state,
        )
        engine = Engine(task, self.estimator.device)
        model = load_checkpoint(Path(model_path).parent)
        if model.config.task != task:
            raise ValueError(f"Causilo checkpoint task {model.config.task!r} != {task!r}")
        engine.model = model.to(engine.device)
        self.estimator._engine = engine


class TabFMAdapter(NativeEstimatorAdapter):
    """Plain PyTorch TabFM ensemble, with learned weighting/calibration disabled."""

    def __init__(
        self, model_path: str, task: str, random_state: int = 42, n_estimators: int = 8,
    ) -> None:
        from tabfm import tabfm_v1_0_0_pytorch

        model = tabfm_v1_0_0_pytorch.load(
            model_type=task, checkpoint_path=str(Path(model_path).parent),
            device=_device(), use_cache=False,
        )
        target = "TabFMClassifier" if task == "classification" else "TabFMRegressor"
        kwargs = dict(model=model, n_estimators=n_estimators,
                      random_state=random_state, enable_nnls=False)
        if task == "classification":
            kwargs.update(binary_calibration_method=None, multiclass_calibration_method=None)
        self.estimator = resolve_class(f"tabfm.{target}")(**kwargs)


class LimiXEnsembleConfig:
    """Select native pipelines deterministically, independently of any dataset.

    Greedily cover unseen normalization, categorical-encoding and permutation
    settings; ties use the original index. When all members are requested,
    retain upstream order. Larger ensembles repeat complete native recipes,
    with any remainder selected by coverage. Upstream assigns preprocessing
    seeds and shuffle offsets by member index, not by recipe identity.
    """

    def __init__(self, task: str, n_estimators: int) -> None:
        path = Path(__file__).parent / "configs" / f"limix2_{task}.json"
        encoded = path.read_bytes()
        config = json.loads(encoded)
        pipelines = config["pipelines"]
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators < 1:
            raise ValueError("LimiX-2 n_estimators must be a positive integer")
        repeats, remainder = divmod(int(n_estimators), len(pipelines))
        tokens = [{(key, json.dumps(p.get(key), sort_keys=True)) for key in (
            "RebalanceFeatureDistribution", "CategoricalFeatureEncoder", "FeatureShuffler",
        )} for p in pipelines]
        selected, seen = [], set()
        while len(selected) < remainder:
            index = max((i for i in range(len(pipelines)) if i not in selected),
                        key=lambda i: (len(tokens[i] - seen), -i))
            selected.append(index)
            seen.update(tokens[index])
        self.indices = list(range(len(pipelines))) * repeats + sorted(selected)
        self.config = deepcopy(config)
        self.config["pipelines"] = [deepcopy(pipelines[i]) for i in self.indices]
        self.metadata = {
            "selection_policy": ("native_cycles_coverage_v2" if n_estimators > len(pipelines)
                                 else "transform_coverage_v1"),
            "source_sha256": hashlib.sha256(encoded).hexdigest(),
            "pipeline_indices": self.indices,
            "effective_n_estimators": len(self.indices),
            "inference_config": deepcopy(self.config),
        }


class LimiXAdapter:
    """Expose the LimiX-2 context/query API through the existing runner interface."""

    def __init__(
        self, model_path: str, task: str, random_state: int = 42, n_estimators: int = 8,
    ) -> None:
        import torch

        self.task = task
        ensemble = LimiXEnsembleConfig(task, n_estimators)
        self.metadata = ensemble.metadata
        predictor = resolve_class("inference.v2_0.predictor.LimiXPredictor")
        self.predictor = predictor(
            device=torch.device(_device()), model_path=model_path,
            inference_config=deepcopy(ensemble.config), seed=random_state,
        )
        if self.predictor.n_estimators != n_estimators:
            raise ValueError("LimiX-2 did not apply the requested pipeline count")

    def fit(self, X, y):
        self.X_context = X.copy()
        self.y_context = np.asarray(y).copy()
        if self.task == "classification":
            self.classes_ = np.unique(y)
            if not 2 <= len(self.classes_) <= 10:
                raise ValueError("LimiX-2 requires 2..10 context classes")
        return self

    def predict(self, X):
        return self.predictor.predict(
            self.X_context, self.y_context, X,
            task_type=self.task.capitalize(),
        )

    def predict_proba(self, X):
        return self.predict(X)
