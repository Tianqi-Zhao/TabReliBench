"""Versioned classification sampling and guards against mixing cached splits.

Classification defaults to stratified_v1. Set
CLASSIFICATION_SAMPLING_PROTOCOL=legacy only to reproduce historical splits.
"""
from __future__ import annotations

import heapq
import json
import os
from pathlib import Path
import tempfile

import numpy as np

LEGACY = "legacy"
STRATIFIED = "stratified_v1"
PROTOCOL_ENV = "CLASSIFICATION_SAMPLING_PROTOCOL"


class SamplingProtocolError(ValueError):
    """The requested class coverage or artifact reuse is not possible."""


def classification_protocol(value: str | None = None) -> str:
    protocol = os.environ.get(PROTOCOL_ENV, STRATIFIED) if value is None else value
    if protocol not in (LEGACY, STRATIFIED):
        raise SamplingProtocolError(f"Unknown classification sampling protocol: {protocol!r}")
    return protocol


def stratified_indices(
    y: np.ndarray, size: int, rng: np.random.Generator,
    *, minimum: int, reserve: int = 0,
) -> np.ndarray:
    """Draw distinct rows with bounded, approximately proportional class quotas.

    ``minimum`` rows per class go into the draw; ``reserve`` remain outside it.
    Minima take priority over proportions and are recorded in split metadata.
    No class is dropped, and no row is duplicated to satisfy these constraints.
    """
    labels, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    lower = np.full(len(labels), minimum, dtype=np.int64)
    upper = counts - reserve
    if (size <= 0 or not len(labels) or np.any(upper < lower)
            or size < lower.sum() or size > upper.sum()):
        raise SamplingProtocolError(
            f"Cannot draw {size} rows from class counts {counts.tolist()} with "
            f"at least {minimum} per class and {reserve} reserved per class. "
            "Increase the sample/context budget or review dataset eligibility; "
            "rows will not be duplicated and classes will not be silently removed."
        )
    target = size * counts.astype(float) / counts.sum()
    allocation = np.clip(np.floor(target).astype(np.int64), lower, upper)
    delta = int(size - allocation.sum())
    direction = 1 if delta >= 0 else -1
    # Marginal squared-error cost gives proportional allocation subject to
    # the coverage bounds. Seeded tie-breaking avoids favoring low label IDs.
    ties = rng.random(len(labels))
    def eligible(i):
        return allocation[i] < upper[i] if direction == 1 else allocation[i] > lower[i]
    def cost(i):
        return 2 * direction * (allocation[i] - target[i]) + 1
    heap = [(cost(i), ties[i], i) for i in range(len(labels)) if eligible(i)]
    heapq.heapify(heap)
    for _ in range(abs(delta)):
        _, _, i = heapq.heappop(heap)
        allocation[i] += direction
        if eligible(i):
            heapq.heappush(heap, (cost(i), ties[i], i))
    # Sort once instead of scanning the full dataset separately for each class.
    groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
    selected = np.concatenate([
        rng.choice(group, size=int(n), replace=False)
        for group, n in zip(groups, allocation)
    ])
    return rng.permutation(selected)


def _claim_marker(path: Path, expected: dict, *, legacy_exists: bool) -> None:
    """Create a complete immutable marker atomically, including concurrent jobs."""
    if not path.exists():
        if legacy_exists and expected["sampling_protocol"] != LEGACY:
            raise SamplingProtocolError(
                f"Unversioned cached artifacts at {path.parent}; use a new result/cache "
                "directory for stratified_v1. Existing results will not be overwritten."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".sampling-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(expected, stream, sort_keys=True, indent=2)
                stream.write("\n")
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
    actual = json.loads(path.read_text())
    if actual != expected:
        raise SamplingProtocolError(
            f"Sampling protocol/configuration mismatch at {path}. "
            "Use a separate result/cache directory; cached experiments cannot be mixed."
        )


def claim_classification_output(root: Path | str, protocol: str, max_n: int) -> None:
    directory = Path(root) / "classification"
    _claim_marker(
        directory / "sampling_protocol.json",
        {"sampling_protocol": classification_protocol(protocol), "max_n": int(max_n)},
        legacy_exists=any(directory.glob("*/*.pkl")),
    )


def check_legacy_feature_cache(path: Path) -> None:
    """A legacy writer must also reject a cache created by the new protocol."""
    marker = path.with_suffix(".sampling.json")
    if marker.exists() and json.loads(marker.read_text())["sampling_protocol"] != LEGACY:
        raise SamplingProtocolError(f"Stratified feature cache at {path}; use a separate cache directory.")


def claim_feature_cache(path: Path, metadata: dict) -> None:
    _claim_marker(path.with_suffix(".sampling.json"), metadata, legacy_exists=path.exists())
