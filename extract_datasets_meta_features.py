"""Compute and cache dataset-level meta-features.

Thin CLI wrapper around :class:`evaluation.pipelines.DatasetFeaturePipeline`.

Usage::

    python extract_datasets_meta_features.py --dataset_ids_file dataset_ids_regression.txt

"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from evaluation import ArtifactStore, DatasetLoader, ExperimentSpec, load_dataset_ids
from evaluation.pipelines import DatasetFeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and cache dataset-level meta-features on training splits.",
    )
    parser.add_argument(
        "--dataset_ids_file", type=str, required=True,
        help="Text file with one dataset ID per line.",
    )
    parser.add_argument(
        "--task", choices=("auto", "regression", "classification"), default="auto",
        help="Feature groups to use: auto-detect per dataset (default), or force one task.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--features_cache_dir", type=str, default="features_cache")
    parser.add_argument("--max_samples", type=int, default=10_000)
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even when a cache PKL already exists.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.features_cache_dir)
    dataset_ids = sorted(load_dataset_ids(args.dataset_ids_file))
    forced_task = None if args.task == "auto" else args.task
    seeds = sorted(args.seeds)

    print(f"{len(dataset_ids)} datasets | task={args.task} | seeds={seeds}")
    print(f"cache → {cache_dir}/")

    store = ArtifactStore(
        root=cache_dir.parent if cache_dir.parent != Path(".") else Path("."),
        dataset_features_dir=cache_dir,
    )
    pipeline = DatasetFeaturePipeline(
        store=store,
        loader=DatasetLoader(),
        max_samples=args.max_samples,
    )

    n_ok = n_skip = n_fail = 0
    for did in dataset_ids:
        for seed in seeds:
            spec = ExperimentSpec(dataset_id=did, seed=seed, ratio=1.0)
            try:
                feats = pipeline.run(
                    spec, task=forced_task, skip_existing=not args.force,
                )
                if feats is None:
                    n_skip += 1
                else:
                    n_ok += 1
            except Exception as exc:
                logging.error("FAILED dataset=%d seed=%d: %s", did, seed, exc)
                n_fail += 1

    print(f"\nDone. computed={n_ok} skipped/cached={n_skip} failed={n_fail}")
    print(f"Cache directory: {cache_dir.resolve()}/")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
