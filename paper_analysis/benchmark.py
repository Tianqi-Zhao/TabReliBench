"""Compute ranks, bootstrap intervals, score decompositions and split stability."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from .tables import BASE


def decomposition(values):
    center = values - values.mean()
    dataset = center.mean(axis=1, keepdims=True)
    model = center.mean(axis=0, keepdims=True)
    interaction = center - dataset - model
    total = np.square(center).sum()
    shares = np.array([values.shape[1]*np.square(dataset).sum(),
                       values.shape[0]*np.square(model).sum(), np.square(interaction).sum()])
    return shares / total if total else np.full(3, np.nan), interaction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.tables/'dataset_metrics.tsv', sep='\t')
    trials = pd.read_csv(args.tables/'trial_metrics.tsv', sep='\t')
    ranks, parts, stability, profiles, skipped = [], [], [], [], []
    for (task, metric), g in d.groupby(['task', 'metric']):
        w = g.pivot(index='dataset_id', columns='model', values='mean_across_five_splits')
        if not np.isfinite(w).all().all():
            skipped.append(dict(task=task, metric=metric, reason='nonfinite_or_incomplete_panel'))
            continue  # Undefined metrics are retained in input tables, never imputed.
        r = w.rank(axis=1, ascending=metric.split('.')[-1] not in {'accuracy', 'r2'})
        draw = np.random.default_rng(20261005).integers(len(w), size=(10000, len(w)))
        lo, hi = np.quantile(r.to_numpy()[draw].mean(axis=1), [.025, .975], axis=0)
        for j, model in enumerate(w):
            ranks.append(dict(task=task, metric=metric, model=model, mean_rank=r[model].mean(),
                              ci_lower=lo[j], ci_upper=hi[j], n_datasets=len(w)))
            profiles.append(dict(task=task, metric=metric, model=model, median=w[model].median(),
                                 q1=w[model].quantile(.25), q3=w[model].quantile(.75)))
        # Include raw CRPS sensitivity alongside normalized CRPS.
        if metric.split('.')[-1] not in {'accuracy','brier_score','confidence_ece_em',
                'classwise_ece_em','r2','crps_mean','crps_normalized','pit_ks_stat','cov_abs_dev','wsc_abs_dev','total_abs_dev'}:
            continue
        for panel in ('tfm', 'full'):
            models = [m for m in w if panel == 'full' or m not in BASE]
            if not models:
                continue
            shares, _ = decomposition(w[models].to_numpy())
            parts.append(dict(task=task, metric=metric, panel=panel, dataset=shares[0],
                              model=shares[1], interaction=shares[2], n_datasets=len(w), n_models=len(models)))
            matrices = [trials[trials.task.eq(task)&trials.metric.eq(metric)&trials.seed.eq(seed)]
                        .pivot(index='dataset_id', columns='model', values='value')
                        .reindex(index=w.index, columns=models) for seed in range(5)]
            common = np.logical_and.reduce([np.isfinite(v).all(axis=1) for v in matrices])
            if task == 'regression' and panel == 'full':
                incomplete = d.loc[d.task.eq(task) & d.n_seeds.lt(5), 'dataset_id']
                common &= ~w.index.isin(incomplete)
            if common.sum() < 2:
                continue
            residuals = [decomposition(v.loc[common].to_numpy())[1].ravel() for v in matrices]
            for a,b in combinations(range(5),2):
                stability.append(dict(task=task, metric=metric, panel=panel, seed_a=a,seed_b=b,
                    n_datasets=int(common.sum()),correlation=np.corrcoef(residuals[a],residuals[b])[0,1]))
    for name, rows in [('ranks',ranks),('decomposition',parts),('stability',stability),('profiles',profiles),('skipped_metrics',skipped)]:
        pd.DataFrame(rows).to_csv(args.output/f'{name}.csv',index=False)


if __name__ == '__main__':
    main()
