"""Dataset-median ensemble trajectories, ranks and paired ensemble-size changes."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ENSEMBLES = [1, 4, 8, 16, 32]
MODELS = {
    "classification": [
        "tabpfnv3.5", "tabpfnv3", "tabpfnv2.5", "tabpfnv2", "tabiclv2",
        "tabicl", "tabdpt1.3", "tabdpt", "causilo", "limix2", "tabfm",
    ],
    "regression": [
        "tabpfnv3.5", "tabpfnv3", "tabpfnv2.5", "tabpfnv2", "tabiclv2",
        "tabdpt1.3", "causilo", "limix2", "tabfm",
    ],
}
REGRESSION_DISTRIBUTIONAL = [m for m in MODELS["regression"] if m not in {"limix2", "tabfm"}]
METRICS = {
    "classification": ["Accuracy", "Brier", "Confidence ECE", "Classwise ECE"],
    "regression": ["R2", "Normalized CRPS", "PIT-KS", "Coverage error", "WSC error", "Total error"],
}
HIGHER = {"Accuracy": True, "R2": True}
def bootstrap_mean(values, rng, draws=10000):
    values = np.asarray(values, float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(means, [.025, .975]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', type=Path, nargs='+', required=True, help='Five exported table directories, E=1/4/8/16/32')
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args();TABLES=args.output;TABLES.mkdir(parents=True,exist_ok=True)
    data=pd.concat([pd.read_csv(t/'trial_metrics.tsv',sep='\t') for t in args.tables],ignore_index=True)
    labels={'accuracy':'Accuracy','brier_score':'Brier','confidence_ece_em':'Confidence ECE','classwise_ece_em':'Classwise ECE','r2':'R2','crps_normalized':'Normalized CRPS','pit_ks_stat':'PIT-KS','cov_abs_dev':'Coverage error','wsc_abs_dev':'WSC error','total_abs_dev':'Total error','marginal_coverage':'Marginal coverage','worst_slab_coverage':'Worst-slab coverage','avg_width_norm':'Normalized width'}
    data=data[data.ensemble.gt(0)&(data.metric.str.startswith('alpha_free.')|data.metric.str.startswith('alpha_dependent.0.1.'))].copy()
    data['metric']=data.metric.str.split('.').str[-1].map(labels)
    data=data.dropna(subset=['metric']).rename(columns={'ensemble':'n_estimators'})
    assert set(data.n_estimators) == set(ENSEMBLES)
    assert not data.duplicated(["task", "dataset_id", "seed", "model", "n_estimators", "metric"]).any()
    assert np.isfinite(data.value).all()
    for task, models in MODELS.items():
        assert set(data.loc[data.task.eq(task), "model"]) == set(models)
        expected = 160 if task == "classification" else 81
        assert data.loc[data.task.eq(task), "dataset_id"].nunique() == expected

    means = (data.groupby(["task", "dataset_id", "model", "n_estimators", "metric"], as_index=False)
                 .agg(value=("value", "mean"), count=("value", "size")))
    assert means["count"].eq(5).all()
    means.drop(columns="count").to_csv(TABLES / "dataset_means.tsv.gz", sep="\t", index=False)

    # Match the manuscript: average seeds first, then take a dataset median.
    medians = means.groupby(["task", "metric", "model", "n_estimators"], as_index=False).agg(
        median_metric=("value", "median"), n_datasets=("dataset_id", "nunique"))
    medians["aggregation"] = "mean over five seeds within dataset, then equal-weight median over datasets"
    medians["normalization"] = medians.metric.map(lambda metric:
        "CRPS divided by population SD of y_train for the same dataset-seed split"
        if metric == "Normalized CRPS" else "none")

    rank_rows = []
    summary_rows = []
    transition_rows = []
    rng = np.random.default_rng(20261006)
    for task, metrics in METRICS.items():
        for metric in metrics:
            models = (MODELS[task] if task == "classification" or metric == "R2"
                      else REGRESSION_DISTRIBUTIONAL)
            configurations = pd.MultiIndex.from_product([models, ENSEMBLES], names=["model", "n_estimators"])
            d = means[means.task.eq(task) & means.metric.eq(metric)]
            wide = (d.pivot(index="dataset_id", columns=["model", "n_estimators"], values="value")
                     .reindex(columns=configurations))
            expected = 160 if task == "classification" else 81
            if len(wide) != expected or not np.isfinite(wide.to_numpy()).all():
                raise ValueError(f"Incomplete {task}/{metric} panel; export every ensemble with --training-scales")
            ranks = wide.rank(axis=1, ascending=not HIGHER.get(metric, False), method="average")
            average = ranks.mean()
            for (model, ensemble), value in average.items():
                rank_rows.append({"task": task, "metric": metric, "model": model,
                                  "n_estimators": ensemble, "mean_rank": value,
                                  "n_datasets": len(wide), "n_configurations": len(configurations)})
            normalized = (average - 1) / (len(configurations) - 1)
            by_e = normalized.groupby(level="n_estimators").mean().reindex(ENSEMBLES)
            for ensemble, value in by_e.items():
                summary_rows.append({"task": task, "metric": metric,
                                     "n_estimators": ensemble,
                                     "model_averaged_normalized_rank": value})

            cube = d.pivot(index=["dataset_id", "model"], columns="n_estimators", values="value")
            sign = 1 if HIGHER.get(metric, False) else -1
            for old, new in zip(ENSEMBLES[:-1], ENSEMBLES[1:]):
                improvements = sign * (cube[new] - cube[old])
                by_dataset = improvements.groupby(level="dataset_id").mean()
                mean_change, low, high = bootstrap_mean(by_dataset, rng)
                try:
                    pvalue = float(wilcoxon(by_dataset, alternative="two-sided").pvalue)
                except ValueError:
                    pvalue = 1.0
                transition_rows.append({
                    "task": task, "metric": metric, "from_e": old, "to_e": new,
                    "mean_signed_improvement": mean_change, "ci_low": low, "ci_high": high,
                    "dataset_improvement_fraction": float((by_dataset > 0).mean()),
                    "wilcoxon_p": pvalue, "n_datasets": len(by_dataset), "n_models": len(models),
                })

    medians.to_csv(TABLES / "ensemble_median_metrics.csv", index=False)
    ranks = pd.DataFrame(rank_rows)
    summaries = pd.DataFrame(summary_rows)
    transitions = pd.DataFrame(transition_rows)
    ranks.to_csv(TABLES / "ensemble_joint_ranks.csv", index=False)
    summaries.to_csv(TABLES / "ensemble_rank_summary.csv", index=False)
    transitions.to_csv(TABLES / "adjacent_ensemble_changes.csv", index=False)

    interval_rows = []
    for ensemble in ENSEMBLES:
        for model in REGRESSION_DISTRIBUTIONAL:
            for coverage in ["Marginal coverage", "Worst-slab coverage"]:
                c = means[(means.task == "regression") & (means.metric == coverage) &
                          (means.model == model) & (means.n_estimators == ensemble)].set_index("dataset_id").value
                w = means[(means.task == "regression") & (means.metric == "Normalized width") &
                          (means.model == model) & (means.n_estimators == ensemble)].set_index("dataset_id").value
                interval_rows.append({"n_estimators": ensemble, "model": model,
                                      "coverage_type": coverage, "coverage": c.median(),
                                      "width": w.median(), "n_datasets": len(c)})
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(TABLES / "ensemble_interval_medians.csv", index=False)

if __name__=='__main__':main()
