"""Compute calibration metrics from saved PPD / proba predictions.

Thin CLI wrapper around :class:`evaluation.pipelines.MetricsPipeline` /
:class:`evaluation.metrics.BaseMetricsCalculator`. All real logic lives
in the ``evaluation`` package.

Layout::

    <results_dir>/
        predictions/
            regression/dataset_<id>_seed<s>_ratio<r>_<model>.pkl     <- input
            classification/dataset_<id>_seed<s>_ratio<r>_<model>.pkl <- input
        metrics/
            regression/dataset_<id>_seed<s>_ratio<r>_<model>_metrics.pkl     <- output
            classification/dataset_<id>_seed<s>_ratio<r>_<model>_metrics.pkl <- output

Each *regression* metrics PKL uses the nested schema documented in
:mod:`evaluation.metrics` (``alpha_dependent`` / ``alpha_free``).

Each *classification* metrics PKL uses the same top-level keys; all
classification scalars and per-instance arrays live under
``alpha_free`` (``alpha_dependent`` is always ``{}``), computed by
:class:`ClassificationMetricsCalculator`.

Back-compat: the legacy *flat* layout (``predictions/*.pkl`` /
``metrics/*.pkl`` with no ``regression/`` subfolder) still loads as
regression input.

Usage::

    python compute_metrics.py --results_dir results/seed1_ratio1.0
    python compute_metrics.py --results_dir results/seed1_ratio1.0 --task regression
    python compute_metrics.py --predictions_pkl path/to/foo.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

from evaluation import (
    ArtifactStore,
    ExperimentSpec,
    TASK_REGRESSION,
    TASKS,
    load_dataset_ids,
)
from evaluation.pipelines import MetricsPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("metrics")


DEFAULT_ALPHAS = [0.05, 0.1, 0.15, 0.2]


def _process_one(
    pred_path: Path,
    pred_task: str,
    pipeline: MetricsPipeline,
) -> bool:
    parsed = ExperimentSpec.parse_predictions_filename(
        pred_path.name, task=pred_task,
    )
    if parsed is None:
        log.warning("Skip (cannot parse filename): %s", pred_path.name)
        return False
    spec, model = parsed

    try:
        with open(pred_path, "rb") as f:
            record = pickle.load(f)
    except Exception as exc:
        log.warning("Could not load %s (%s: %s)",
                    pred_path.name, type(exc).__name__, exc)
        return False

    record_task = record.get("task", pred_task)
    log.info(
        "Computing metrics for %s (task=%s) ...",
        pred_path.name, record_task,
    )
    try:
        result = pipeline.compute_and_save(
            record, spec, model, task=pred_task,
        )
    except NotImplementedError as exc:
        log.warning(
            "Skip %s: %s metrics not implemented (%s).",
            pred_path.name, record_task, exc,
        )
        return False
    except Exception as exc:
        log.warning("Failed metrics on %s (%s: %s)",
                    pred_path.name, type(exc).__name__, exc)
        return False

    if record_task == TASK_REGRESSION and result.metrics is not None:
        out = result.metrics
        af = out.get("alpha_free", {}).get("per_dataset", {})
        if out.get("distribution_status") == "not_requested":
            log.info("  point-only: r2=%.4f; distribution metrics not requested",
                     af.get("r2", float("nan")))
        elif af:
            log.info(
                "  alpha-free: crps=%.4f  pit_ks_stat=%.4f  "
                "pit_ece=%.4f  pit_hist_l1=%.4f",
                af.get("crps_mean", float("nan")),
                af.get("pit_ks_stat", float("nan")),
                af.get("pit_ece", float("nan")),
                af.get("pit_hist_l1", float("nan")),
            )
        for alpha, slot in out.get("alpha_dependent", {}).items():
            m = slot.get("per_dataset", {})
            wsc = m.get("worst_slab_coverage", float("nan"))
            log.info(
                "  alpha=%.2f  nominal=%.2f  marginal_cov=%.4f  avg_len=%.4f  "
                "worst_slab_cov=%s (feat=%s)  pinball_mean=%.4f",
                alpha, 1 - alpha,
                m.get("marginal_coverage", float("nan")),
                m.get("avg_length", float("nan")),
                "nan" if wsc != wsc else f"{wsc:.4f}",
                m.get("worst_slab_feature_name") or "<no_slab>",
                m.get("pinball_mean", float("nan")),
            )
    return True


def _collect_files(
    base: Path,
    requested_task: Optional[str],
    *,
    allowed_dataset_ids: Optional[frozenset[int]] = None,
) -> list[tuple[Path, str]]:
    """Return ``[(path, task), ...]`` for every prediction PKL to process.

    Searches the task-first layout:
      - ``<base>/<task>/predictions/*.pkl``

    ``requested_task='regression'`` skips classification; vice-versa.
    """
    out: list[tuple[Path, str]] = []
    for task in TASKS:
        if requested_task is not None and requested_task != task:
            continue
        d = base / task / "predictions"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("dataset_*_seed*_ratio*_*.pkl")):
            if p.name.endswith("_metrics.pkl"):
                continue
            if allowed_dataset_ids is not None:
                parsed = ExperimentSpec.parse_predictions_filename(
                    p.name, task=task,
                )
                if parsed is None:
                    continue
                spec, _ = parsed
                if spec.dataset_id not in allowed_dataset_ids:
                    continue
            out.append((p, task))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute calibration metrics from saved prediction PKLs. "
            "Auto-detects task from each record (or from the on-disk "
            "subdirectory it lives in)."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--results_dir", type=str,
        help="Base dir; predictions read from <results_dir>/predictions/"
             "[<task>/], metrics written to <results_dir>/metrics/<task>/. "
             "Processes every pkl.",
    )
    src.add_argument(
        "--predictions_pkl", type=str,
        help="Process a single predictions pkl. Metrics are written under a "
             "sibling 'metrics/<task>/' folder (or use --output_dir).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override the metrics output directory (still split by <task>/ "
             "underneath).",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
        help=f"Miscoverage levels for regression (default: {DEFAULT_ALPHAS}).",
    )
    parser.add_argument(
        "--task", choices=("auto",) + TASKS, default="auto",
        help="Restrict to one task. 'auto' (default) processes every PKL "
             "under both predictions/regression/ and predictions/"
             "classification/ (plus the legacy flat predictions/ as "
             "regression).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing metrics pkls (default: skip).",
    )
    parser.add_argument(
        "--dataset_ids_file", type=str, default=None,
        help="Only process pkls whose dataset_id is listed in this file.",
    )
    args = parser.parse_args()

    if any(not (0.0 < a < 1.0) for a in args.alphas):
        parser.error("--alphas must all lie in (0, 1)")

    requested_task: Optional[str] = (
        None if args.task == "auto" else args.task
    )
    allowed_ids: Optional[frozenset[int]] = (
        load_dataset_ids(args.dataset_ids_file)
        if args.dataset_ids_file else None
    )

    if args.results_dir:
        base = Path(args.results_dir)
        pairs = _collect_files(
            base, requested_task, allowed_dataset_ids=allowed_ids,
        )
        if not pairs:
            parser.error(
                f"No prediction pkls found under {base} "
                f"(searched <task>/predictions/, predictions/<task>/, predictions/)"
            )
        store = ArtifactStore(base)
        if args.output_dir:
            store.metrics_dir = Path(args.output_dir)
        log.info("Found %d prediction pkls under %s", len(pairs), base)
    else:
        pred_pkl = Path(args.predictions_pkl)
        if not pred_pkl.is_file():
            parser.error(f"Not a file: {pred_pkl}")
        # Infer task from the parent dir name when possible; else fall
        # back to the user's --task or 'regression'.
        parent = pred_pkl.parent.name
        grandparent = pred_pkl.parent.parent.name
        if parent == "predictions" and grandparent in TASKS:
            # New layout: <base>/<task>/predictions/foo.pkl
            inferred = grandparent
            base = pred_pkl.parent.parent.parent
        elif parent in TASKS:
            # Old typed layout: <base>/predictions/<task>/foo.pkl
            inferred = parent
            base = pred_pkl.parent.parent.parent
        else:
            # Legacy flat: <base>/predictions/foo.pkl
            inferred = requested_task or TASK_REGRESSION
            base = pred_pkl.parent.parent
        store = ArtifactStore(base)
        if args.output_dir:
            store.metrics_dir = Path(args.output_dir)
        pairs = [(pred_pkl, inferred)]

    pipeline = MetricsPipeline(
        store, args.alphas, overwrite=args.overwrite,
    )

    n_ok = 0
    for path, task in pairs:
        if _process_one(path, task, pipeline):
            n_ok += 1
    log.info("Done: %d / %d files processed.", n_ok, len(pairs))


if __name__ == "__main__":
    main()
