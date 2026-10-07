"""Experiment specification and filename conventions.

A single ``ExperimentSpec`` is a frozen ``(dataset_id, seed, ratio[, task])``
record that knows how to format itself into every PKL filename in the
project. All other modules import filenames *from this class* so the
conventions live in exactly one place.

The ``task`` discriminator (``'regression'`` or ``'classification'``)
controls which ``predictions/<task>/`` and ``metrics/<task>/`` subdirectory
the artefact lives in; **filenames are unchanged** and the field defaults
to ``'regression'`` so legacy callers / parsers keep working.

Note on naming asymmetry preserved here:

* predictions / metrics / instance features use ``seed{seed}_ratio{ratio}``
  (no underscore between key and value), matching the existing files in
  ``results/seed*/predictions/`` and ``results/seed*/metrics/``;
* dataset features use ``seed_{seed}`` (underscored, no ratio — features
  depend only on ``(dataset_id, seed)``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


_PRED_RE     = re.compile(r"^dataset_(\d+)_seed(\d+)_ratio([0-9.]+)_(.+)\.pkl$")
_METRICS_RE  = re.compile(r"^dataset_(\d+)_seed(\d+)_ratio([0-9.]+)_(.+)_metrics\.pkl$")
_MODEL_RE    = re.compile(r"^dataset_(\d+)_seed(\d+)_ratio([0-9.]+)_(.+)_model\.pkl$")
_DSFEAT_RE   = re.compile(r"^dataset_(\d+)_seed_(\d+)\.pkl$")
_INSFEAT_RE  = re.compile(r"^dataset_(\d+)_seed(\d+)_ratio([0-9.]+)_(.+)_meta_features\.pkl$")


# Allowed task discriminators. Used in both ``ExperimentSpec.task`` and in
# the on-disk subdirectory name ``predictions/<task>/`` / ``metrics/<task>/``.
TASK_REGRESSION:     str = "regression"
TASK_CLASSIFICATION: str = "classification"
TASKS: tuple[str, ...] = (TASK_REGRESSION, TASK_CLASSIFICATION)


@dataclass(frozen=True)
class ExperimentSpec:
    """Identifies one (dataset_id, seed, ratio[, task]) experiment.

    ``task`` is an optional discriminator (defaults to ``'regression'``);
    it does **not** appear in any filename — the task is encoded by the
    directory the artefact is written to (``predictions/<task>/...``).
    Adding it as a field nonetheless lets ``iter_*`` callers receive a
    fully-typed identifier without having to track task on the side.
    """

    dataset_id: int
    seed: int
    ratio: float
    task: str = TASK_REGRESSION

    # ── Filenames ─────────────────────────────────────────────────────────

    def predictions_filename(self, model: str) -> str:
        return f"dataset_{self.dataset_id}_seed{self.seed}_ratio{self.ratio}_{model}.pkl"

    def metrics_filename(self, model: str) -> str:
        return (f"dataset_{self.dataset_id}_seed{self.seed}_ratio{self.ratio}"
                f"_{model}_metrics.pkl")

    def model_filename(self, model: str) -> str:
        """Filename for a fitted-model bundle or shared-checkpoint manifest."""
        return (f"dataset_{self.dataset_id}_seed{self.seed}_ratio{self.ratio}"
                f"_{model}_model.pkl")

    def dataset_features_filename(self) -> str:
        return f"dataset_{self.dataset_id}_seed_{self.seed}.pkl"

    def instance_features_filename(self, model: str) -> str:
        return (f"dataset_{self.dataset_id}_seed{self.seed}_ratio{self.ratio}"
                f"_{model}_meta_features.pkl")

    # ── Inverse parsing — used by ArtifactStore.iter_* discovery ──────────

    @classmethod
    def parse_predictions_filename(
        cls, name: str, task: str = TASK_REGRESSION,
    ) -> Optional[tuple["ExperimentSpec", str]]:
        m = _PRED_RE.match(name)
        if not m:
            return None
        did, seed, ratio, model = m.groups()
        return cls(int(did), int(seed), float(ratio), task), model

    @classmethod
    def parse_metrics_filename(
        cls, name: str, task: str = TASK_REGRESSION,
    ) -> Optional[tuple["ExperimentSpec", str]]:
        m = _METRICS_RE.match(name)
        if not m:
            return None
        did, seed, ratio, model = m.groups()
        return cls(int(did), int(seed), float(ratio), task), model

    @classmethod
    def parse_model_filename(
        cls, name: str, task: str = TASK_REGRESSION,
    ) -> Optional[tuple["ExperimentSpec", str]]:
        m = _MODEL_RE.match(name)
        if not m:
            return None
        did, seed, ratio, model = m.groups()
        return cls(int(did), int(seed), float(ratio), task), model

    @classmethod
    def parse_dataset_features_filename(cls, name: str) -> Optional["ExperimentSpec"]:
        m = _DSFEAT_RE.match(name)
        if not m:
            return None
        did, seed = m.groups()
        return cls(int(did), int(seed), 1.0)

    @classmethod
    def parse_instance_features_filename(
        cls, name: str, task: str = TASK_REGRESSION,
    ) -> Optional[tuple["ExperimentSpec", str]]:
        m = _INSFEAT_RE.match(name)
        if not m:
            return None
        did, seed, ratio, model = m.groups()
        return cls(int(did), int(seed), float(ratio), task), model
