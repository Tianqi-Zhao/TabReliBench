"""Paired, dataset-equal tables; no filtering on direction of improvement."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def summarize(input_dir,output_dir):
    rows=[]
    for path in sorted(Path(input_dir).rglob('*_metrics.pkl')):
        with path.open('rb') as f:r=pickle.load(f)
        for alpha,slot in r['alpha_dependent'].items():
            for metric,value in slot['per_dataset'].items():
                if metric in ('marginal_coverage','cov_abs_dev','worst_slab_coverage','wsc_abs_dev','total_abs_dev','avg_width_norm','interval_score','pinball_mean','infinite_interval_rate','empty_set_rate'):
                    rows.append(dict(dataset_id=r['dataset_id'],seed=r['seed'],base_model=r['base_model'],method=r['method'],protocol=r['protocol'],alpha=alpha,metric=metric,value=value))
    df=pd.DataFrame(rows)
    if df.empty:raise ValueError('No conformal metrics found')
    keys=['dataset_id','seed','base_model','protocol','alpha','metric']
    if df.duplicated(keys+['method']).any():raise ValueError('Duplicate configuration: separate runs before aggregating')
    vanilla=df[df.method=='Vanilla'][keys+['value']].rename(columns={'value':'vanilla'})
    paired=df[df.method!='Vanilla'].merge(vanilla,on=keys,how='left',validate='many_to_one',indicator=True)
    if (paired['_merge']!='both').any():raise ValueError('Missing same-trial Vanilla')
    paired=paired.drop(columns='_merge');paired['delta']=paired.value-paired.vanilla
    paired['finite_pair']=np.isfinite(paired.value)&np.isfinite(paired.vanilla)
    paired['relative_delta']=np.where(paired.finite_pair & (paired.vanilla.abs()>1e-12),paired['delta']/paired.vanilla.abs(),np.nan)
    # Infinite/undefined values remain visible in raw tables and counts.
    group=['dataset_id','base_model','protocol','method','alpha','metric']
    # Undefined empty-set losses must propagate, never disappear through
    # pandas' default skip-NaN means. Counts retain the affected trials.
    strict_mean = lambda values: np.mean(values.to_numpy(dtype=float))
    means=paired.groupby(group,dropna=False).agg(value=('value',strict_mean),vanilla=('vanilla',strict_mean),delta=('delta',strict_mean),relative_delta=('relative_delta',strict_mean),n_pairs=('seed','size'),n_finite_pairs=('finite_pair','sum')).reset_index()
    means['negative_delta']=np.where(~np.isnan(means.delta),(means.delta<0).astype(float),np.nan)
    across=means.groupby(group[1:],dropna=False).agg(mean_delta=('delta',strict_mean),median_relative_delta=('relative_delta',lambda values: np.median(values.to_numpy(dtype=float))),negative_delta_fraction=('negative_delta',strict_mean),n_datasets=('dataset_id','nunique'),total_pairs=('n_pairs','sum'),finite_pairs=('n_finite_pairs','sum')).reset_index()
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    for filename,table in [('split_metrics.tsv',df),('paired_deltas.tsv',paired),('dataset_means.tsv',means),('dataset_equal_summary.tsv',across)]:
        path=output/filename
        if path.exists():raise FileExistsError(path)
        table.to_csv(path,sep='\t',index=False)
    return output
