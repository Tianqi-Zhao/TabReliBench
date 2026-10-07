"""Filesystem layout + I/O for every artefact in the pipeline.

One class — ``ArtifactStore`` — owns all paths and pickle reads/writes. The
rest of the codebase never touches ``pickle`` or ``Path.glob`` directly.

Layout (defaults)::

    <root>/
        regression/                                          # task-first
            predictions/<spec>_<model>.pkl                   # PredictionPipeline
            metrics/<spec>_<model>_metrics.pkl               # MetricsPipeline
        classification/
            predictions/<spec>_<model>.pkl
            metrics/<spec>_<model>_metrics.pkl
        meta_features/<spec>_<model>_meta_features.pkl       # InstanceFeaturePipeline

    <dataset_features_dir>/<spec>.pkl                        # DatasetFeaturePipeline
        # default = repo-root/features_cache/

Back-compat (read-only): the previous ``predictions/<task>/`` and
``metrics/<task>/`` layouts and the even-older flat ``predictions/`` /
``metrics/`` directories (no task subfolder at all) are still searched when
loading, so historical ``results/seed*/...`` archives remain readable without
any migration step.  New writes always use the task-first layout above.

Multi-run aggregation (e.g. multi-seed analyses)::

    ArtifactStore.from_run_dirs([Path("results/seed0_ratio1.0"), ...])
"""
from __future__ import annotations

import os
import json
import pickle
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from .metrics import upgrade_legacy_schema
from .spec import (
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TASKS,
    ExperimentSpec,
)


def load_dataset_ids(path: str | Path) -> frozenset[int]:
    """Load a dataset-IDs text file and return the IDs as a frozenset.

    File format: one integer per line; blank lines and lines starting with
    ``#`` are ignored.  Raises :class:`FileNotFoundError` if *path* does not
    exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset_ids file not found: {p}")
    return frozenset(
        int(line.strip())
        for line in p.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _validate_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return task


class ArtifactStore:
    """Single source of truth for filesystem layout and pickle I/O."""

    def __init__(
        self,
        root: str | Path,
        *,
        predictions_subdir: str = "predictions",
        metrics_subdir: str = "metrics",
        models_subdir: str = "models",
        model_checkpoints_subdir: str = "model_checkpoints",
        dataset_features_dir: str | Path | None = None,
        instance_features_subdir: str = "meta_features",
    ) -> None:
        self.root = Path(root)
        self._predictions_subdir = predictions_subdir
        self._metrics_subdir     = metrics_subdir
        self._models_subdir      = models_subdir
        self._model_checkpoints_subdir = model_checkpoints_subdir
        # Kept for back-compat (external code may read them; compute_metrics.py
        # may assign to metrics_dir to redirect metric writes).
        self.predictions_dir = self.root / predictions_subdir
        self.metrics_dir     = self.root / metrics_subdir
        self.models_dir      = self.root / models_subdir
        self.instance_features_dir = self.root / instance_features_subdir

        # Dataset features are not tied to a single run; default to the
        # repo-level cache. Caller can override.
        if dataset_features_dir is None:
            self.dataset_features_dirs: list[Path] = [Path("features_cache")]
        else:
            self.dataset_features_dirs = [Path(dataset_features_dir)]

        # Multi-run case: when constructed via `from_run_dirs`, these are
        # the secondary roots whose predictions/metrics/instance-features
        # subdirs are also searched in iter_*.
        self._extra_roots: list[Path] = []

    # ── Constructors for multi-run aggregation ────────────────────────────

    @classmethod
    def from_run_dirs(
        cls,
        dirs: Sequence[str | Path],
        *,
        dataset_features_dir: str | Path | None = None,
        **kwargs,
    ) -> "ArtifactStore":
        """Build a store that reads predictions/metrics/instance-features
        from multiple run roots (one per seed, typically). Writes still go
        to the *first* root.
        """
        if not dirs:
            raise ValueError("from_run_dirs needs at least one directory")
        store = cls(dirs[0], dataset_features_dir=dataset_features_dir, **kwargs)
        store._extra_roots = [Path(d) for d in dirs[1:]]
        return store

    # ── Internal: enumerate read roots, both for typed & legacy layouts ──

    def _predictions_dir_for(self, task: str) -> Path:
        """Primary write target: ``<root>/<task>/predictions/``."""
        return self.root / _validate_task(task) / self._predictions_subdir

    def _metrics_dir_for(self, task: str) -> Path:
        """Primary write target: ``<root>/<task>/metrics/``.

        If ``self.metrics_dir`` has been overridden externally (e.g. by
        ``compute_metrics.py --output_dir``), the override is honoured and
        task is appended as a subdirectory (preserving old semantics).
        """
        _validate_task(task)
        default_metrics = self.root / self._metrics_subdir
        if self.metrics_dir != default_metrics:
            return self.metrics_dir / task
        return self.root / task / self._metrics_subdir

    def _models_dir_for(self, task: str) -> Path:
        """Primary fitted-model / checkpoint-manifest write target."""
        return self.root / _validate_task(task) / self._models_subdir

    def _model_checkpoints_dir_for(self, task: str) -> Path:
        """Root for libraries whose serialized estimator references files."""
        return self.root / _validate_task(task) / self._model_checkpoints_subdir

    @staticmethod
    def _atomic_pickle(path: Path, record: dict) -> None:
        """Write a pickle completely before exposing its final path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def _all_predictions_dirs(self, task: str) -> list[Path]:
        """All directories searched when *reading* predictions for ``task``.

        Only the current task-first layout is used:
          ``<root>/<task>/predictions/``
        Extra roots (multi-seed) are appended in the same order.
        """
        _validate_task(task)
        p = self._predictions_subdir
        dirs: list[Path] = [self.root / task / p]
        for er in self._extra_roots:
            dirs.append(er / task / p)
        return dirs

    def _all_metrics_dirs(self, task: str) -> list[Path]:
        """All directories searched when *reading* metrics for ``task``."""
        _validate_task(task)
        dirs: list[Path] = [self._metrics_dir_for(task)]
        for er in self._extra_roots:
            dirs.append(er / task / self._metrics_subdir)
        return dirs

    def _all_instance_dirs(self) -> list[Path]:
        return [self.instance_features_dir] + [
            d / self.instance_features_dir.name for d in self._extra_roots
        ]

    # ── Predictions ───────────────────────────────────────────────────────

    def predictions_path(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> Path:
        t = _validate_task(task or spec.task)
        return self._predictions_dir_for(t) / spec.predictions_filename(model)

    def save_prediction(
        self,
        spec: ExperimentSpec,
        model: str,
        record: dict,
        task: Optional[str] = None,
    ) -> Path:
        t = _validate_task(task or spec.task)
        out_dir = self._predictions_dir_for(t)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / spec.predictions_filename(model)
        with open(out, "wb") as f:
            pickle.dump(record, f)
        return out

    # ── Fitted models / checkpoint manifests ───────────────────────────────

    def models_path(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> Path:
        t = _validate_task(task or spec.task)
        return self._models_dir_for(t) / spec.model_filename(model)

    def model_exists(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> bool:
        """Return true only for a non-empty, dependency-complete model bundle."""
        path = self.models_path(spec, model, task=task)
        complete = path.with_suffix(".complete.json")
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or not complete.is_file()
        ):
            return False
        try:
            record = json.loads(complete.read_text())
        except Exception:
            return False
        if record.get("size_bytes") != path.stat().st_size:
            return False
        external = record.get("external_checkpoint_dir")
        if external is not None and not Path(external).exists():
            return False
        shared = record.get("shared_checkpoint_path")
        return shared is None or Path(shared).is_file()

    def save_model(
        self,
        spec: ExperimentSpec,
        model: str,
        record: dict,
        task: Optional[str] = None,
    ) -> Path:
        """Atomically save one fitted-model bundle or TFM checkpoint manifest."""
        out = self.models_path(spec, model, task=task)
        complete = out.with_suffix(".complete.json")
        complete.unlink(missing_ok=True)
        self._atomic_pickle(out, record)
        completion = {
            "artifact_path": str(out.resolve()),
            "size_bytes": out.stat().st_size,
            "external_checkpoint_dir": record.get("external_checkpoint_dir"),
            "shared_checkpoint_path": (
                (record.get("checkpoint") or {}).get("path")
            ),
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{complete.name}.", suffix=".tmp", dir=out.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(completion, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, complete)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        return out

    def load_model(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> dict:
        path = self.models_path(spec, model, task=task)
        with open(path, "rb") as f:
            return pickle.load(f)

    def create_model_checkpoint_dir(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> Path:
        """Create a unique persistent directory for an external checkpoint."""
        t = _validate_task(task or spec.task)
        stem = spec.model_filename(model).removesuffix("_model.pkl")
        root = self._model_checkpoints_dir_for(t) / stem
        root.mkdir(parents=True, exist_ok=True)
        out = root / uuid.uuid4().hex
        out.mkdir()
        return out

    def remove_model_checkpoint_dir(
        self, path: str | Path, task: str,
    ) -> None:
        """Remove one checkpoint dir, guarded to the store-owned task root."""
        target = Path(path).resolve()
        root = self._model_checkpoints_dir_for(task).resolve()
        if target == root or root not in target.parents:
            raise ValueError(f"Refusing to remove checkpoint outside {root}: {target}")
        if target.exists():
            shutil.rmtree(target)

    def load_prediction(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> dict:
        t = _validate_task(task or spec.task)
        for d in self._all_predictions_dirs(t):
            p = d / spec.predictions_filename(model)
            if p.exists():
                with open(p, "rb") as f:
                    return pickle.load(f)
        raise FileNotFoundError(
            f"No prediction file found for {spec} model={model!r} task={t!r} "
            f"in {[str(p) for p in self._all_predictions_dirs(t)]}"
        )

    def iter_predictions(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        specs: Optional[Iterable[ExperimentSpec]] = None,
        task: Optional[str] = None,
    ) -> Iterator[tuple[ExperimentSpec, str, dict]]:
        """Yield ``(spec, model, record)`` for every prediction PKL.

        ``task=None`` iterates both regression *and* classification
        directories (and the legacy flat ``predictions/`` for regression).
        """
        keep_models = set(models) if models is not None else None
        keep_specs  = set(specs)  if specs  is not None else None
        tasks = [_validate_task(task)] if task else list(TASKS)

        for t in tasks:
            for d in self._all_predictions_dirs(t):
                if not d.is_dir():
                    continue
                for path in sorted(d.glob("dataset_*_seed*_ratio*_*.pkl")):
                    if path.name.endswith("_metrics.pkl"):
                        continue
                    parsed = ExperimentSpec.parse_predictions_filename(
                        path.name, task=t,
                    )
                    if parsed is None:
                        continue
                    pspec, model = parsed
                    if keep_models is not None and model not in keep_models:
                        continue
                    if keep_specs is not None and pspec not in keep_specs:
                        continue
                    with open(path, "rb") as f:
                        yield pspec, model, pickle.load(f)

    def list_predictions(
        self, spec: ExperimentSpec, task: Optional[str] = None,
    ) -> list[str]:
        """Return model names that have a prediction file for *spec*.

        ``task=None`` falls back to ``spec.task``. Only the typed (and, for
        regression, legacy-flat) directories are searched.
        """
        t = _validate_task(task or spec.task)
        out: list[str] = []
        glob = (
            f"dataset_{spec.dataset_id}_seed{spec.seed}"
            f"_ratio{spec.ratio}_*.pkl"
        )
        for d in self._all_predictions_dirs(t):
            if not d.is_dir():
                continue
            for path in d.glob(glob):
                if path.name.endswith("_metrics.pkl"):
                    continue
                parsed = ExperimentSpec.parse_predictions_filename(
                    path.name, task=t,
                )
                if parsed is None:
                    continue
                pspec, model = parsed
                if (pspec.dataset_id, pspec.seed, pspec.ratio) == (
                    spec.dataset_id, spec.seed, spec.ratio
                ):
                    out.append(model)
        return sorted(set(out))

    # ── Metrics ───────────────────────────────────────────────────────────

    def metrics_path(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> Path:
        t = _validate_task(task or spec.task)
        return self._metrics_dir_for(t) / spec.metrics_filename(model)

    def save_metrics(
        self,
        spec: ExperimentSpec,
        model: str,
        record: dict,
        task: Optional[str] = None,
    ) -> Path:
        t = _validate_task(task or spec.task)
        out_dir = self._metrics_dir_for(t)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / spec.metrics_filename(model)
        with open(out, "wb") as f:
            pickle.dump(record, f)
        return out

    def metrics_exist(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> bool:
        t = _validate_task(task or spec.task)
        for d in self._all_metrics_dirs(t):
            if (d / spec.metrics_filename(model)).exists():
                return True
        return False

    def load_metrics(
        self, spec: ExperimentSpec, model: str, task: Optional[str] = None,
    ) -> dict:
        t = _validate_task(task or spec.task)
        for d in self._all_metrics_dirs(t):
            p = d / spec.metrics_filename(model)
            if p.exists():
                with open(p, "rb") as f:
                    d_loaded = pickle.load(f)
                return upgrade_legacy_schema(d_loaded)
        raise FileNotFoundError(
            f"No metrics file found for {spec} model={model!r} task={t!r} "
            f"in {[str(p) for p in self._all_metrics_dirs(t)]}"
        )

    def iter_metrics(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        specs: Optional[Iterable[ExperimentSpec]] = None,
        task: Optional[str] = None,
    ) -> Iterator[tuple[ExperimentSpec, str, dict]]:
        keep_models = set(models) if models is not None else None
        keep_specs  = set(specs)  if specs  is not None else None
        tasks = [_validate_task(task)] if task else list(TASKS)

        for t in tasks:
            for d in self._all_metrics_dirs(t):
                if not d.is_dir():
                    continue
                for path in sorted(
                    d.glob("dataset_*_seed*_ratio*_*_metrics.pkl"),
                ):
                    parsed = ExperimentSpec.parse_metrics_filename(
                        path.name, task=t,
                    )
                    if parsed is None:
                        continue
                    pspec, model = parsed
                    if keep_models is not None and model not in keep_models:
                        continue
                    if keep_specs is not None and pspec not in keep_specs:
                        continue
                    try:
                        with open(path, "rb") as f:
                            yield pspec, model, upgrade_legacy_schema(pickle.load(f))
                    except Exception as exc:
                        print(f"  [WARN] failed to load {path.name}: {exc}")

    # ── Dataset-level meta features (independent of run dir) ──────────────

    def dataset_features_path(self, spec: ExperimentSpec) -> Path:
        return self.dataset_features_dirs[0] / spec.dataset_features_filename()

    def save_dataset_features(self, spec: ExperimentSpec, feats: dict) -> Path:
        out_dir = self.dataset_features_dirs[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / spec.dataset_features_filename()
        with open(out, "wb") as f:
            pickle.dump(feats, f)
        return out

    def load_dataset_features(self, spec: ExperimentSpec) -> Optional[dict]:
        for d in self.dataset_features_dirs:
            p = d / spec.dataset_features_filename()
            if p.exists():
                with open(p, "rb") as f:
                    return pickle.load(f)
        return None

    def iter_dataset_features(self) -> Iterator[tuple[ExperimentSpec, dict]]:
        for d in self.dataset_features_dirs:
            if not d.is_dir():
                continue
            for path in sorted(d.glob("dataset_*_seed_*.pkl")):
                spec = ExperimentSpec.parse_dataset_features_filename(path.name)
                if spec is None:
                    continue
                with open(path, "rb") as f:
                    yield spec, pickle.load(f)

    # ── Instance-level meta features ──────────────────────────────────────

    def instance_features_path(self, spec: ExperimentSpec, model: str) -> Path:
        return self.instance_features_dir / spec.instance_features_filename(model)

    def save_instance_features(self, spec: ExperimentSpec, model: str, df) -> Path:
        self.instance_features_dir.mkdir(parents=True, exist_ok=True)
        out = self.instance_features_path(spec, model)
        with open(out, "wb") as f:
            pickle.dump(df, f)
        return out

    def load_instance_features(self, spec: ExperimentSpec, model: str):
        for d in self._all_instance_dirs():
            p = d / spec.instance_features_filename(model)
            if p.exists():
                with open(p, "rb") as f:
                    return pickle.load(f)
        return None

    def iter_instance_features(self) -> Iterator[tuple[ExperimentSpec, str, object]]:
        for d in self._all_instance_dirs():
            if not d.is_dir():
                continue
            for path in sorted(d.glob("dataset_*_seed*_ratio*_*_meta_features.pkl")):
                parsed = ExperimentSpec.parse_instance_features_filename(path.name)
                if parsed is None:
                    continue
                spec, model = parsed
                with open(path, "rb") as f:
                    yield spec, model, pickle.load(f)
