"""Model registry + dense-quantile-grid (regression) / proba (classification) prediction.

A ``ModelRunner`` encapsulates the small but real API divergences between
the supported tabular-foundation-model families when asking for a
calibrated output.

Regression
----------
TabPFN-{v2, v2.5, v3} returns a list-of-arrays through
``predict(X, output_type='quantiles', quantiles=[...])``; TabICL v2
returns a single 2D array via
``predict(X, output_type='quantiles', alphas=[...])``.
:class:`RegressionModelRunner` normalises both to a uniform
``(n_test, n_levels)`` PPD array (monotone enforced) and additionally
returns the model's canonical mean point prediction
(``predict(X, output_type='mean')`` for TabPFN, the default for TabICL v2).

Classification
--------------
The six classifier targets exposed here all implement a sklearn-style
``predict_proba(X) -> (n_test, n_classes)``. Their constructor APIs do
*not* agree, so :class:`ModelConfig` carries flexible
``init_kwargs`` / ``checkpoint_kwarg`` / ``seed_kwarg`` slots that
:class:`ModelRunner` translates into the right call:

* TabPFN v2 / v2.5 / v3  (``tabpfn.TabPFNClassifier``)
    ``cls(model_path=<ckpt>, random_state=<seed>)``
* TabICL v1 / v2    (``tabicl.TabICLClassifier``)
    ``cls(model_path=<ckpt>, checkpoint_version=<ver>, random_state=<seed>)``
* TabDPT            (``tabdpt.TabDPTClassifier``)
    ``cls(model_weight_path=<ckpt>, verbose=False, ...)``  -- no seed kwarg

:class:`ClassificationModelRunner` accepts integer-encoded support labels
on the global ``0..K-1`` class space and may be passed an explicit
``n_classes_global`` to declare that range. When the context happens to
miss some global classes (small ``ratio`` on imbalanced data), the
runner internally re-labels ``y_ctx`` to ``0..K_ctx-1`` for the model
and zero-pads the returned probability matrix back to
``(n_test, n_classes_global)``; the missing-class columns stay at zero,
which makes test samples with those labels permanently misclassified
(``argmax`` cannot pick them) while keeping log-loss finite via the
usual ``np.clip`` floor in the metrics layer.

Optional model packages are imported only when constructing their runner.
"""
from __future__ import annotations

import os
import hashlib
import json
from importlib import metadata, util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .adapters import LimiXEnsembleConfig, resolve_class
from .preprocessing import FeaturePreprocessor
from .ppd import make_quantile_grid
from .spec import TASK_CLASSIFICATION, TASK_REGRESSION


# Default dense quantile grid used to approximate the PPD. Step 0.005 means
# every level needed for alpha in {0.01, 0.02, ..., 0.99} is on the grid
# exactly (no interpolation error) and 199 values keep on-disk size manageable.
DEFAULT_QUANTILE_GRID: np.ndarray = make_quantile_grid(0.005)
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get("TABPFN_CHECKPOINT_DIR", str(Path.home() / ".cache/tabbenchmark/checkpoints"))
)

# Keep downloaded assets with the configured checkpoints; respect user overrides.
os.environ.setdefault("HF_HOME", str(DEFAULT_CHECKPOINT_DIR / "hf_cache"))


def _checkpoint_path(filename: str) -> str:
    """Return the absolute path for a locally downloaded checkpoint."""
    return str(DEFAULT_CHECKPOINT_DIR / filename)


def _require_local_path(model_name: str, path: str, *, label: str) -> None:
    """Fail early instead of letting model packages fall back to downloads."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{model_name}: local {label} not found at {path!r}. "
            "Download it under DEFAULT_CHECKPOINT_DIR or update the model config."
        )


def _require_zero_based_class_subset(
    model_name: str,
    y: np.ndarray,
    n_classes_global: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Return ``(ctx_classes, n_classes_global)`` with a sanity check.

    The contract is one of:

    * ``n_classes_global`` is given (preferred): ``unique(y)`` must be a
      subset of ``{0, 1, ..., n_classes_global - 1}``. The ctx-labels are
      allowed to be a *strict* subset (i.e. some global classes may be
      missing from the context when ``ratio`` is small); the caller is
      expected to pad the predicted-proba matrix to ``n_classes_global``
      columns afterwards.
    * ``n_classes_global`` is omitted (legacy / direct-use): ``unique(y)``
      must be exactly ``0..K-1`` for some ``K``, and that ``K`` becomes the
      inferred ``n_classes_global``. This keeps the old behaviour for any
      caller that invokes :class:`ClassificationModelRunner` directly
      without going through the pipeline.
    """
    classes_ = np.unique(np.asarray(y))
    if classes_.size == 0:
        raise ValueError(f"{model_name}: empty support set (y_ctx is empty).")
    if not np.issubdtype(classes_.dtype, np.integer):
        # Float-encoded class labels would silently corrupt the index
        # arithmetic below; fail fast instead.
        raise ValueError(
            f"{model_name}: classification labels must be integer-valued; "
            f"got dtype {classes_.dtype}."
        )
    if classes_.min() < 0:
        raise ValueError(
            f"{model_name}: classification labels must be non-negative; "
            f"got min label {int(classes_.min())}."
        )

    if n_classes_global is None:
        expected = np.arange(classes_.shape[0])
        if not np.array_equal(classes_, expected):
            raise ValueError(
                f"{model_name}: when n_classes_global is not provided, "
                f"classification labels must be contiguous integers 0..K-1 "
                f"for K classes; got sorted labels {classes_.tolist()}."
            )
        return classes_, int(classes_.shape[0])

    n_classes_global = int(n_classes_global)
    if n_classes_global < classes_.shape[0]:
        raise ValueError(
            f"{model_name}: n_classes_global={n_classes_global} is smaller "
            f"than the number of distinct labels in y_ctx "
            f"({classes_.shape[0]}); cannot be a valid global class count."
        )
    if int(classes_.max()) >= n_classes_global:
        raise ValueError(
            f"{model_name}: y_ctx contains label {int(classes_.max())} which "
            f"is outside the global range 0..{n_classes_global - 1}."
        )
    return classes_, n_classes_global


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """Static information about a single model name.

    ``cls`` is invoked as ``cls(**init_kwargs, **checkpoint_kw, **seed_kw,
    **ensemble_kw)`` where:

    * ``checkpoint_kw = {checkpoint_kwarg: checkpoint}`` if both fields are
      truthy, else ``{}``;
    * ``seed_kw = {seed_kwarg: seed}`` if ``seed_kwarg`` is set, else ``{}``;
    * ``ensemble_kw = {ensemble_init_kwarg: n_estimators}`` if
      ``ensemble_init_kwarg`` is set and ``n_estimators`` is provided,
      else ``{}``.

    Ensemble-size configuration follows the per-model API:

    * TabPFN / TabICL accept the ensemble size as a constructor
      kwarg (``n_estimators``); set ``ensemble_init_kwarg="n_estimators"``
      and leave ``ensemble_predict_kwarg=None``.
    * TabDPT only takes the ensemble size at predict time
      (``predict_proba(..., n_ensembles=...)`` /
      ``predict(..., n_ensembles=...)``); set
      ``ensemble_init_kwarg=None`` and
      ``ensemble_predict_kwarg="n_ensembles"``.

    Models whose randomness is controlled at prediction time can set
    ``predict_seed_kwarg`` (TabDPT uses ``"seed"``).  The runner then passes
    the experiment seed to every prediction call instead of leaving the model
    library to generate a nondeterministic ensemble.

    For regressors, ``quantile_param`` and ``returns_2d`` describe the
    minor divergence between TabPFN's and TabICL's quantile output;
    classifiers leave both at their (unused) defaults.
    """
    name: str
    task: str
    cls: Any
    checkpoint: Optional[str] = None
    checkpoint_kwarg: Optional[str] = "model_path"
    seed_kwarg: Optional[str] = "random_state"
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    quantile_param: str = "quantiles"
    returns_2d: bool = False
    ensemble_init_kwarg: Optional[str] = "n_estimators"
    ensemble_predict_kwarg: Optional[str] = None
    predict_seed_kwarg: Optional[str] = None
    # Name of the method to call for probability prediction.
    # Most models use ``predict_proba``; TabDPT exposes the ensemble
    # variant as a *separate* method (``ensemble_predict_proba``) that
    # accepts ``n_ensembles`` — its plain ``predict_proba`` does not.
    proba_method_name: str = "predict_proba"
    regression_output: str = "quantiles"
    preprocessing: Optional[str] = None

    def __post_init__(self):
        if self.regression_output not in {"quantiles", "point"}:
            raise ValueError("regression_output must be quantiles or point")


# ─────────────────────────────────────────────────────────────────────────────
# Default registries
#
# Registry entries hold import paths; only selected models import dependencies.
# ─────────────────────────────────────────────────────────────────────────────

def _default_regression_models() -> dict[str, ModelConfig]:

    return {
        "tabdpt1.3": ModelConfig(
            name="tabdpt1.3", task=TASK_REGRESSION,
            cls="evaluation.adapters.TabDPT13RegressorAdapter",
            checkpoint=_checkpoint_path("tabdpt1_3.safetensors"),
            checkpoint_kwarg="model_weight_path", seed_kwarg=None,
            init_kwargs={"verbose": False}, preprocessing="numeric_array",
            ensemble_init_kwarg=None, ensemble_predict_kwarg="n_ensembles",
            predict_seed_kwarg="seed", proba_method_name="predict_proba",
        ),

        "tabpfnv3.5": ModelConfig(
            preprocessing="passthrough",
            name="tabpfnv3.5", task=TASK_REGRESSION, cls="tabpfn.TabPFNRegressor",
            checkpoint=_checkpoint_path("tabpfn-v3.5-20260909.safetensors"),
        ),
        "causilo": ModelConfig(
            preprocessing="passthrough",
            name="causilo", task=TASK_REGRESSION, cls="evaluation.adapters.CausiloAdapter",
            checkpoint=_checkpoint_path("causilo/regressor/model.safetensors"),
            init_kwargs={"task": "regression"},
            returns_2d=True,
        ),
        "limix2": ModelConfig(
            preprocessing="passthrough",
            name="limix2", task=TASK_REGRESSION, cls="evaluation.adapters.LimiXAdapter",
            checkpoint=_checkpoint_path("LimiX-2.ckpt"),
            init_kwargs={"task": "regression"},
            regression_output="point",
        ),
        "tabfm": ModelConfig(
            preprocessing="passthrough",
            name="tabfm", task=TASK_REGRESSION, cls="evaluation.adapters.TabFMAdapter",
            checkpoint=_checkpoint_path("tabfm/regression/model.safetensors"),
            init_kwargs={"task": "regression"},
            regression_output="point",
        ),

        "tabpfnv3": ModelConfig(
            name="tabpfnv3",
            task=TASK_REGRESSION,
            cls="tabpfn.TabPFNRegressor",
            checkpoint=_checkpoint_path("tabpfn-v3-regressor-v3_default.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={"ignore_pretraining_limits": True},
            quantile_param="quantiles",
            returns_2d=False,
        ),
        "tabpfnv2.5": ModelConfig(
            name="tabpfnv2.5",
            task=TASK_REGRESSION,
            cls="tabpfn.TabPFNRegressor",
            checkpoint=_checkpoint_path("tabpfn-v2.5-regressor-v2.5_default.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={"ignore_pretraining_limits": True},
            quantile_param="quantiles",
            returns_2d=False,
        ),
        "tabpfnv2": ModelConfig(
            name="tabpfnv2",
            task=TASK_REGRESSION,
            cls="tabpfn.TabPFNRegressor",
            checkpoint=_checkpoint_path("tabpfn-v2-regressor-v2_default.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={"ignore_pretraining_limits": True},
            quantile_param="quantiles",
            returns_2d=False,
        ),
        "tabiclv2": ModelConfig(
            name="tabiclv2",
            task=TASK_REGRESSION,
            cls="tabicl.TabICLRegressor",
            checkpoint=_checkpoint_path("tabicl-regressor-v2-20260212.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            quantile_param="alphas",
            returns_2d=True,
        ),
    }


def _default_classification_models() -> dict[str, ModelConfig]:
    """Classification models, with optional dependencies resolved at construction."""
    return {
        "tabdpt1.3": ModelConfig(
            name="tabdpt1.3", task=TASK_CLASSIFICATION,
            cls="tabdpt.TabDPTClassifier",
            checkpoint=_checkpoint_path("tabdpt1_3.safetensors"),
            checkpoint_kwarg="model_weight_path", seed_kwarg=None,
            init_kwargs={"verbose": False}, preprocessing="numeric_array",
            ensemble_init_kwarg=None, ensemble_predict_kwarg="n_ensembles",
            predict_seed_kwarg="seed", proba_method_name="ensemble_predict_proba",
        ),

        "tabpfnv3.5": ModelConfig(
            preprocessing="passthrough",
            name="tabpfnv3.5", task=TASK_CLASSIFICATION, cls="tabpfn.TabPFNClassifier",
            checkpoint=_checkpoint_path("tabpfn-v3.5-20260909.safetensors"),
        ),
        "causilo": ModelConfig(
            preprocessing="passthrough",
            name="causilo", task=TASK_CLASSIFICATION, cls="evaluation.adapters.CausiloAdapter",
            checkpoint=_checkpoint_path("causilo/classifier/model.safetensors"),
            init_kwargs={"task": "classification"},
            returns_2d=True,
        ),
        "limix2": ModelConfig(
            preprocessing="passthrough",
            name="limix2", task=TASK_CLASSIFICATION, cls="evaluation.adapters.LimiXAdapter",
            checkpoint=_checkpoint_path("LimiX-2.ckpt"),
            init_kwargs={"task": "classification"},
        ),
        "tabfm": ModelConfig(
            preprocessing="passthrough",
            name="tabfm", task=TASK_CLASSIFICATION, cls="evaluation.adapters.TabFMAdapter",
            checkpoint=_checkpoint_path("tabfm/classification/model.safetensors"),
            init_kwargs={"task": "classification"},
        ),

        "tabpfnv3": ModelConfig(
            name="tabpfnv3",
            task=TASK_CLASSIFICATION,
            cls="tabpfn.TabPFNClassifier",
            checkpoint=_checkpoint_path("tabpfn-v3-classifier-v3_default.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={"ignore_pretraining_limits": True},
        ),
        "tabpfnv2.5": ModelConfig(
            name="tabpfnv2.5",
            task=TASK_CLASSIFICATION,
            cls="tabpfn.TabPFNClassifier",
            checkpoint=_checkpoint_path("tabpfn-v2.5-classifier-v2.5_default.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={"ignore_pretraining_limits": True},
        ),
        "tabpfnv2": ModelConfig(
            name="tabpfnv2",
            task=TASK_CLASSIFICATION,
            cls="tabpfn.TabPFNClassifier",
            checkpoint=_checkpoint_path("tabpfn-v2-classifier-v2_default.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={"ignore_pretraining_limits": True},
        ),
        "tabicl": ModelConfig(
            name="tabicl",
            task=TASK_CLASSIFICATION,
            cls="tabicl.TabICLClassifier",
            checkpoint=_checkpoint_path("tabicl-classifier-v1.1-20250506.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={
                # TabICL validates this against its built-in checkpoint
                # version names even when model_path points at a local file.
                "checkpoint_version": "tabicl-classifier-v1.1-20250506.ckpt",
            },
        ),
        "tabiclv2": ModelConfig(
            name="tabiclv2",
            task=TASK_CLASSIFICATION,
            cls="tabicl.TabICLClassifier",
            checkpoint=_checkpoint_path("tabicl-classifier-v2-20260212.ckpt"),
            checkpoint_kwarg="model_path",
            seed_kwarg="random_state",
            init_kwargs={
                "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
            },
        ),
        "tabdpt": ModelConfig(
            name="tabdpt",
            task=TASK_CLASSIFICATION,
            cls="tabdpt.TabDPTClassifier",
            # TabDPTClassifier auto-downloads from HuggingFace when
            # ``model_weight_path=None``; pass the local checkpoint instead.
            checkpoint=_checkpoint_path("tabdpt1_1.safetensors"),
            checkpoint_kwarg="model_weight_path",
            # TabDPT has no random_state init kwarg (seeding is per
            # predict call); leave seed out of construction.
            seed_kwarg=None,
            init_kwargs={"verbose": False},
            # ``predict_proba`` does NOT accept n_ensembles; the proper
            # ensemble entry-point is ``ensemble_predict_proba(n_ensembles=...)``.
            ensemble_init_kwarg=None,
            ensemble_predict_kwarg="n_ensembles",
            predict_seed_kwarg="seed",
            proba_method_name="ensemble_predict_proba",
        ),
    }


class ModelRegistry:
    """Lazy holder for the default per-task ``{name: ModelConfig}`` mappings."""

    _regression_cache: Optional[dict[str, ModelConfig]] = None
    _classification_cache: Optional[dict[str, ModelConfig]] = None

    @classmethod
    def default_regression(cls) -> dict[str, ModelConfig]:
        if cls._regression_cache is None:
            cls._regression_cache = _default_regression_models()
        return cls._regression_cache

    @classmethod
    def default_classification(cls) -> dict[str, ModelConfig]:
        if cls._classification_cache is None:
            cls._classification_cache = _default_classification_models()
        return cls._classification_cache

    @classmethod
    def default(cls, task: str = TASK_REGRESSION) -> dict[str, ModelConfig]:
        """Return the default registry for ``task``."""
        if task == TASK_REGRESSION:
            return cls.default_regression()
        if task == TASK_CLASSIFICATION:
            return cls.default_classification()
        raise ValueError(
            f"task must be 'regression' or 'classification'; got {task!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def _build_init_kwargs(
    config: ModelConfig,
    seed: int,
    n_estimators: Optional[int] = None,
) -> dict[str, Any]:
    """Translate a ``ModelConfig`` + seed (+ optional ensemble size) into ctor kwargs."""
    kwargs: dict[str, Any] = dict(config.init_kwargs)
    if config.checkpoint_kwarg and config.checkpoint is not None:
        _require_local_path(config.name, config.checkpoint, label="checkpoint")
        kwargs[config.checkpoint_kwarg] = config.checkpoint
    if config.seed_kwarg:
        kwargs[config.seed_kwarg] = seed
    if config.ensemble_init_kwarg and n_estimators is not None:
        kwargs[config.ensemble_init_kwarg] = int(n_estimators)
    return kwargs


def checkpoint_manifest_for_config(
    config: ModelConfig,
    seed: int,
    n_estimators: Optional[int] = None,
) -> dict[str, Any]:
    """Describe shared TFM weights without constructing or fitting the model."""
    checkpoint = config.checkpoint
    checkpoint_info: dict[str, Any] | None = None
    if checkpoint is not None:
        path = Path(checkpoint).resolve()
        checkpoint_info = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "artifact_schema_version": 1,
        "artifact_type": "shared_pretrained_checkpoint_manifest",
        "model": config.name,
        "task": config.task,
        "model_class": (config.cls if isinstance(config.cls, str) else f"{config.cls.__module__}.{config.cls.__qualname__}"),
        "checkpoint": checkpoint_info,
        "checkpoint_kwarg": config.checkpoint_kwarg,
        "seed": seed,
        "seed_kwarg": config.seed_kwarg,
        "init_kwargs": dict(config.init_kwargs),
        "preprocessing": config.preprocessing,
        "quantile_param": config.quantile_param,
        "returns_2d": config.returns_2d,
        "proba_method_name": config.proba_method_name,
        "n_estimators": n_estimators,
        "ensemble_init_kwarg": config.ensemble_init_kwarg,
        "ensemble_predict_kwarg": config.ensemble_predict_kwarg,
        "predict_seed_kwarg": config.predict_seed_kwarg,
        "weights_are_shared_across_datasets": True,
        "per_dataset_trained_weights": False,
    }


def prediction_contract(config: ModelConfig) -> dict[str, Any]:
    """Describe requested outputs independently of the installed model runtime."""
    point_only = config.task == TASK_REGRESSION and config.regression_output == "point"
    return {
        "output_kind": ("point" if point_only else "quantiles")
        if config.task == TASK_REGRESSION else "probabilities",
        "regression_metrics": ["r2"] if point_only else "full",
        "distribution_status": "not_requested" if point_only else "requested",
    }


def prediction_configuration(config, seed, n_estimators, q_grid):
    """Fingerprint inference settings and local asset identities, without imports.

    Weight files are identified by path, size and nanosecond mtime; we do not
    reread multi-GB weights for every dataset. Small sidecar/config files are
    hashed. Moving or replacing assets therefore invalidates cached predictions.
    """
    manifest = checkpoint_manifest_for_config(config, seed, n_estimators)
    assets = []
    if config.checkpoint:
        path = Path(config.checkpoint)
        for asset in (path, path.with_name("config.json")):
            if asset.is_file():
                stat = asset.stat()
                item = {"path": str(asset.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                if asset.name == "config.json":
                    item["sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
                assets.append(item)
    versions = {}
    family = "tabpfn" if config.name.startswith("tabpfn") else (
        "tabicl" if config.name.startswith("tabicl") else (
            "tabdpt" if config.name.startswith("tabdpt") else config.name
        )
    )
    for package in ("numpy", "pandas", "scikit-learn", "torch", family):
        try:
            distribution = metadata.distribution(package)
            versions[package] = {"version": distribution.version}
            source = distribution.read_text("direct_url.json")
            if source:
                versions[package]["source"] = json.loads(source)
        except metadata.PackageNotFoundError:
            pass
    result = {
        "schema_version": 1, "manifest": manifest, "assets": assets,
        "package_versions": versions, "contract": prediction_contract(config),
        "quantile_levels": list(map(float, q_grid))
        if config.task == TASK_REGRESSION and config.regression_output == "quantiles" else None,
    }
    if config.name == "limix2":
        result["ensemble"] = LimiXEnsembleConfig(
            config.task, 8 if n_estimators is None else n_estimators,
        ).metadata
        source = util.find_spec("inference")
        if source and source.submodule_search_locations:
            root = Path(next(iter(source.submodule_search_locations))).parent
            digest = hashlib.sha256()
            for folder in ("inference", "model", "models", "utils"):
                for path in sorted((root / folder).rglob("*.py")):
                    digest.update(str(path.relative_to(root)).encode())
                    digest.update(path.read_bytes())
            result["limix_source_sha256"] = digest.hexdigest()
    encoded = json.dumps(result, sort_keys=True, default=str).encode()
    return result, hashlib.sha256(encoded).hexdigest()


class ModelRunner:
    """Abstract base: instantiate one model and produce its calibration output."""

    def __init__(
        self,
        config: ModelConfig,
        seed: int,
        n_estimators: Optional[int] = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.n_estimators = (
            int(n_estimators) if n_estimators is not None else None
        )
        if self.n_estimators is not None and self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        self._model = resolve_class(config.cls)(
            **_build_init_kwargs(config, seed, n_estimators=self.n_estimators)
        )

    def _predict_kwargs(self) -> dict[str, Any]:
        """Return model-specific predict-time ensemble and seed kwargs."""
        kwargs: dict[str, Any] = {}
        if (
            self.config.ensemble_predict_kwarg
            and self.n_estimators is not None
        ):
            kwargs[self.config.ensemble_predict_kwarg] = int(self.n_estimators)
        if self.config.predict_seed_kwarg:
            kwargs[self.config.predict_seed_kwarg] = int(self.seed)
        return kwargs

    def checkpoint_manifest(self) -> dict[str, Any]:
        """Describe the shared pretrained checkpoint used by this TFM run.

        TFMs do not train a new weight checkpoint for each OpenML dataset.
        Their reusable model artifact is therefore a manifest pointing to the
        shared pretrained weights plus the exact construction configuration.
        """
        manifest = checkpoint_manifest_for_config(
            self.config, self.seed, n_estimators=self.n_estimators,
        )
        manifest["adapter_metadata"] = getattr(self._model, "metadata", {})
        return manifest


class RegressionModelRunner(ModelRunner):
    """Fit one regressor and return its dense quantile grid + mean point pred."""

    def __init__(
        self,
        config: ModelConfig,
        seed: int,
        n_estimators: Optional[int] = None,
    ) -> None:
        if config.task != TASK_REGRESSION:
            raise ValueError(
                f"RegressionModelRunner expected a regression config; "
                f"got task={config.task!r} for {config.name!r}"
            )
        super().__init__(config, seed, n_estimators=n_estimators)

    def fit_predict(
        self,
        X_ctx: pd.DataFrame,
        y_ctx: np.ndarray,
        X_test: pd.DataFrame,
        q_grid: np.ndarray,
    ) -> tuple[Optional[np.ndarray], np.ndarray]:
        """Return ``(ppd, point_pred)``; point-only regressors return ``ppd=None``.

        ``ppd`` has shape ``(n_test, len(q_grid))`` (monotone enforced
        along the quantile axis); ``point_pred`` has shape ``(n_test,)``
        and equals ``predict(X_test, output_type='mean')``.
        """
        # Per-model preprocessing — see ``evaluation.preprocessing`` for
        # which strategy each model name maps to.
        pre = FeaturePreprocessor(self.config.name, strategy=self.config.preprocessing)
        X_ctx_in, X_test_in = pre.fit_transform(X_ctx, X_test)

        self._model.fit(X_ctx_in, y_ctx)
        if self.config.regression_output == "point":
            point = self._model.predict(X_test_in, **self._predict_kwargs())
            return None, self._validate_point(point, len(X_test))
        q_levels = list(map(float, q_grid))

        kwargs = {self.config.quantile_param: q_levels}
        kwargs.update(self._predict_kwargs())
        out = self._model.predict(X_test_in, output_type="quantiles", **kwargs)

        if self.config.returns_2d:
            arr = np.asarray(out, dtype=np.float64)
            n_test, n_q = len(X_test), len(q_levels)
            if arr.ndim != 2 or arr.shape != (n_test, n_q):
                raise ValueError(
                    f"{self.config.name}: expected 2D array of shape "
                    f"({n_test}, {n_q}), got ndim={arr.ndim} shape={arr.shape}"
                )
            ppd = arr
        else:
            ppd = np.stack(
                [np.asarray(q, dtype=np.float64).reshape(-1) for q in out],
                axis=1,
            )
        if ppd.shape != (len(X_test), len(q_grid)) or not np.isfinite(ppd).all():
            raise ValueError(f"{self.config.name}: invalid quantile predictions")
        ppd = np.maximum.accumulate(ppd, axis=1)

        # Canonical point prediction. Cannot be recovered exactly from
        # the saved quantile grid (different code path internally), so we
        # store it explicitly. ``predict(output_type='mean')`` is the
        # default for TabPFN; TabICL exposes the same string.
        predict_kwargs = self._predict_kwargs()
        try:
            point = self._model.predict(
                X_test_in, output_type="mean", **predict_kwargs,
            )
        except TypeError:
            # Older / sklearn-style regressors without output_type kw.
            point = self._model.predict(X_test_in, **predict_kwargs)
        return ppd, self._validate_point(point, len(X_test))

    def _validate_point(self, point, n_test):
        result = np.asarray(point, dtype=np.float64).reshape(-1)
        if result.shape != (n_test,) or not np.isfinite(result).all():
            raise ValueError(f"{self.config.name}: invalid point predictions, shape={result.shape}")
        return result


class ClassificationModelRunner(ModelRunner):
    """Fit one classifier and return its predicted-probability matrix."""

    def __init__(
        self,
        config: ModelConfig,
        seed: int,
        n_estimators: Optional[int] = None,
    ) -> None:
        if config.task != TASK_CLASSIFICATION:
            raise ValueError(
                f"ClassificationModelRunner expected a classification "
                f"config; got task={config.task!r} for {config.name!r}"
            )
        super().__init__(config, seed, n_estimators=n_estimators)

    def fit_predict(
        self,
        X_ctx: pd.DataFrame,
        y_ctx: np.ndarray,
        X_test: pd.DataFrame,
        *,
        n_classes_global: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(proba, classes_)`` on the *global* class space.

        ``proba`` has shape ``(n_test, n_classes_global)`` with rows
        summing to ~1.0 over the columns the model can actually predict;
        ``classes_`` is ``np.arange(n_classes_global)``.

        When ``y_ctx`` is missing some of the global classes (e.g. small
        ``ratio`` on an imbalanced dataset), the model is fit on a locally
        re-labeled view ``0..K_ctx-1`` (so per-model contracts like
        TabDPT's "labels are integer column indices" still hold), and the
        resulting ``(n_test, K_ctx)`` proba matrix is then *padded* with
        zero columns at the global indices that did not appear in ``y_ctx``.
        Those zero columns make any test sample with a missing label
        permanently misclassified (``argmax`` cannot pick them), and the
        downstream log-loss is kept finite by the usual ``np.clip`` floor
        in :class:`LogLossMetric`.

        ``n_classes_global`` defaults to ``None`` for backward compat: when
        unset the legacy "y_ctx must be exactly 0..K-1" contract is
        enforced and the inferred ``K`` is used as the global class count.
        """
        ctx_classes, n_classes_global = _require_zero_based_class_subset(
            self.config.name, y_ctx, n_classes_global,
        )

        # Locally re-label y_ctx to 0..K_ctx-1 so that the underlying
        # model always sees a contiguous label set, regardless of which
        # global classes happened to be sampled into the context.
        if ctx_classes.shape[0] == n_classes_global and np.array_equal(
            ctx_classes, np.arange(n_classes_global),
        ):
            y_ctx_local = np.asarray(y_ctx, dtype=np.int64)
        else:
            global_to_local = np.full(n_classes_global, -1, dtype=np.int64)
            global_to_local[ctx_classes] = np.arange(ctx_classes.shape[0])
            y_ctx_local = global_to_local[np.asarray(y_ctx, dtype=np.int64)]

        # Per-model preprocessing — see ``evaluation.preprocessing``.
        # The ``numeric_df`` strategy produces an all-float DataFrame for
        # some classifiers and a numpy ndarray for TabDPT (the conversion is done
        # inside FeaturePreprocessor, not here).
        pre = FeaturePreprocessor(self.config.name, strategy=self.config.preprocessing)
        X_ctx_in, X_test_in = pre.fit_transform(X_ctx, X_test)

        self._model.fit(X_ctx_in, y_ctx_local)
        proba_fn = getattr(self._model, self.config.proba_method_name)
        proba_local = np.asarray(
            proba_fn(X_test_in, **self._predict_kwargs()),
            dtype=np.float64,
        )
        if proba_local.ndim != 2 or proba_local.shape[0] != len(X_test):
            raise ValueError(
                f"{self.config.name}: predict_proba returned shape "
                f"{proba_local.shape}; expected "
                f"(n_test={len(X_test)}, n_classes)."
            )
        if proba_local.shape[1] != ctx_classes.shape[0]:
            raise ValueError(
                f"{self.config.name}: predict_proba returned "
                f"{proba_local.shape[1]} columns but y_ctx covers "
                f"{ctx_classes.shape[0]} distinct classes."
            )

        local_classes = np.asarray(getattr(self._model, "classes_", np.arange(len(ctx_classes))))
        if not np.array_equal(np.sort(local_classes), np.arange(len(ctx_classes))):
            raise ValueError(f"{self.config.name}: invalid probability class labels")
        proba_local = proba_local[:, np.argsort(local_classes)]
        if (not np.isfinite(proba_local).all() or (proba_local < 0).any()
                or not np.allclose(proba_local.sum(axis=1), 1, atol=1e-5)):
            raise ValueError(f"{self.config.name}: invalid probability values or row sums")

        # Pad to the global class space; columns for classes absent from
        # y_ctx stay at zero on purpose (see docstring).
        if ctx_classes.shape[0] == n_classes_global:
            proba = proba_local
        else:
            proba = np.zeros(
                (proba_local.shape[0], n_classes_global), dtype=np.float64,
            )
            proba[:, ctx_classes] = proba_local

        classes_ = np.arange(n_classes_global, dtype=np.int64)
        return proba, classes_


def make_runner(
    config: ModelConfig,
    seed: int,
    *,
    n_estimators: Optional[int] = None,
) -> ModelRunner:
    """Factory: pick the right runner subclass for ``config.task``.

    ``n_estimators`` controls the ensemble size and is routed to the right
    place per model: as a ctor kwarg for TabPFN / TabICL (via
    ``ensemble_init_kwarg``), or as a predict-time kwarg for TabDPT (via
    ``ensemble_predict_kwarg``). Pass ``None`` to keep each model's own
    library default.
    """
    if config.task == TASK_REGRESSION:
        return RegressionModelRunner(config, seed, n_estimators=n_estimators)
    if config.task == TASK_CLASSIFICATION:
        return ClassificationModelRunner(
            config, seed, n_estimators=n_estimators,
        )
    raise ValueError(
        f"Unsupported config.task={config.task!r} for {config.name!r}"
    )
