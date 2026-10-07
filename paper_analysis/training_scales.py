"""Export per-dataset/per-seed training-response scales for normalized CRPS."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.data import DatasetLoader
from evaluation.spec import ExperimentSpec


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-ids', type=Path, default=ROOT/'dataset_ids_regression.txt')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from evaluation import load_dataset_ids
    dataset_ids = sorted(load_dataset_ids(args.dataset_ids))
    loader = DatasetLoader(max_n=10_000)
    rows = []
    for dataset_id in dataset_ids:
        for seed in range(5):
            spec = ExperimentSpec(int(dataset_id), seed, 1.0, "regression")
            split = loader.load_for_spec(spec, task="regression")
            scale = float(np.std(split.y_train, ddof=0))
            if not np.isfinite(scale) or scale < 0:
                raise ValueError(f"Invalid training-response scale for {dataset_id}, seed {seed}: {scale}")
            rows.append({
                "task": "regression",
                "dataset_id": int(dataset_id),
                "seed": seed,
                "n_train": int(len(split.y_train)),
                "training_response_std": scale,
            })
        print(f"dataset {dataset_id}: done", flush=True)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
