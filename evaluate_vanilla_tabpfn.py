"""Run pretrained tabular foundation models on an OpenML dataset and
save raw predictions.

Thin CLI wrapper around :class:`evaluation.pipelines.PredictionPipeline`.
All real logic (data loading, splits, model registry, prediction) lives in
the ``evaluation`` package.

    Output layout::

    <output_dir>/<task>/predictions/dataset_<id>_seed<s>_ratio<r>_<model>.pkl
    <output_dir>/<task>/models/dataset_<id>_seed<s>_ratio<r>_<model>_model.pkl

where ``<task>`` is ``regression`` or ``classification``. The task is
either auto-detected from the OpenML target dtype (``--task auto``,
default) or forced via ``--task regression`` / ``--task classification``.

Usage::

    # Regression (auto-detected from dataset 46934)
    python evaluate_vanilla_tabpfn.py \\
        --dataset_id 46934 --seed 1 --ratio 1.0 \\
        --output_dir results/seed1_ratio1.0

    # Classification (auto-detected from dataset 31)
    python evaluate_vanilla_tabpfn.py \\
        --dataset_id 31 --seed 1 --ratio 1.0 \\
        --output_dir results/seed1_ratio1.0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from evaluation import (
    ArtifactStore,
    DatasetLoader,
    ExperimentSpec,
    TASKS,
    make_quantile_grid,
)
from evaluation.pipelines import PredictionPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a tabular-foundation-model (TabPFN / TabICL / Mitra / "
            "TabDPT) on an OpenML dataset and save the predicted PPD "
            "(regression: dense quantile grid + mean point pred) or "
            "predicted class probabilities (classification). Metrics are "
            "computed separately by compute_metrics.py."
        )
    )
    parser.add_argument("--dataset_id", type=int, required=True,
                        help="OpenML dataset ID.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Fraction of training set to use as model context (default: 1.0).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base directory; predictions and checkpoint "
                             "manifests are written under <output_dir>/<task>/. "
                             "Defaults to ~/vanilla_tabpfn_results/")
    parser.add_argument("--quantile_step", type=float, default=0.005,
                        help="Step of the dense quantile grid for the regression "
                             "PPD (default 0.005 -> 199 levels). Ignored for "
                             "classification.")
    parser.add_argument(
        "--task", choices=("auto",) + TASKS, default="auto",
        help="Task type. 'auto' (default) detects from the OpenML target "
             "column dtype: numeric -> regression, otherwise -> "
             "classification. Pass an explicit value to override.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Optional whitelist of model names (e.g. tabpfnv2 tabiclv2). "
             "Defaults to all models in the relevant registry.",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=8,
        help="Ensemble size used by every model (default: 8). Routed as a "
             "constructor kwarg or native pipeline count, and as a "
             "predict-time kwarg (n_ensembles) for TabDPT.",
    )
    args = parser.parse_args()

    base = Path(args.output_dir) if args.output_dir \
        else Path.home() / "vanilla_tabpfn_results"
    store = ArtifactStore(base)
    loader = DatasetLoader()

    pipeline = PredictionPipeline(
        store=store,
        loader=loader,
        q_grid=make_quantile_grid(args.quantile_step),
        n_estimators=args.n_estimators,
    )
    pipeline.run(
        ExperimentSpec(args.dataset_id, args.seed, args.ratio, args.task),
        only_models=args.models,
    )


if __name__ == "__main__":
    main()
