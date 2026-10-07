"""Predict shared shifts and model-specific departures on held-out datasets.

Inputs are saved benchmark metrics and training-only meta-features. Base models
are never refitted. Every preprocessing and tuning step uses training datasets.
"""
from pathlib import Path
import ast
import hashlib
import json
import os


import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE = {"realmlp", "realmlp_hpo", "xgboost_quantile", "xgboost_quantile_hpo", "bart"}
EXTRA_TREES_N_ESTIMATORS = int(os.environ.get("TABBENCHMARK_EXTRA_TREES_N_ESTIMATORS", "100"))
SPECS = {
    "classification": {"Accuracy": "accuracy", "Brier": "brier_score", "Confidence ECE": "confidence_ece_em", "Classwise ECE": "classwise_ece_em"},
    "regression": {"R2": "r2", "Normalized CRPS": "crps_normalized", "PIT-KS": "pit_ks_stat", "Total error": "alpha_dependent.0.1.per_dataset.total_abs_dev"},
}
# These are supervised fitted-model features, separated from direct X-y
# descriptors. The dependency partition is inherited from the extractor.
AUXILIARY = {
    "linear_acc_norm", "nonlinear_acc_norm", "complexity_ratio_norm",
    "linear_r2", "nonlinear_r2", "complexity_ratio", "snr",
    "tree_target_met", "tree_achieved_acc_norm", "tree_achieved_r2",
    "tree_imbalance", "tree_leaf_depth_cv", "tree_var_importance_gini",
    "tree_var_importance_top1", "tree_feature_used_frac",
    "clf_hs_dof_fraction", "clf_hs_lambda_ratio", "clf_hs_score_prior",
    "hs_dof_fraction", "hs_lambda_ratio", "hs_r2",
}


def feature_groups():
    """Read literal tuples without importing optional benchmark dependencies."""
    path = REPO / "evaluation/features/dataset/selected_features.py"
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    result = {}
    for task in SPECS:
        prefix = task.upper()
        selected = values[prefix + "_SELECTED_DATASET_FEATURES"]
        x = list(values["X_ONLY_SELECTED_DATASET_FEATURES"])
        y = list(values[prefix + "_Y_ONLY_SELECTED_DATASET_FEATURES"])
        xy = list(values[prefix + "_XY_RELATION_SELECTED_DATASET_FEATURES"])
        direct = [f for f in xy if f not in AUXILIARY]
        aux = [f for f in xy if f in AUXILIARY]
        assert set(x + y + direct + aux) == set(selected)
        assert len(x + y + direct + aux) == len(selected)
        result[task] = {"X": x, "+Y": x+y, "+XY": x+y+direct, "+Aux": x+y+direct+aux}
    return result


def targets(a, baseline):
    centered = a - baseline
    shared = centered.mean(axis=1)
    differential = centered - shared[:, None]
    return shared, differential


def transform(x, train, test):
    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    z = imp.fit_transform(x[train])
    t = imp.transform(x[test])
    # Rank scaling prevents unbounded extrapolation from extreme auxiliary
    # scores and ratios. Both the map and imputation use training data only.
    rank = QuantileTransformer(n_quantiles=min(100, len(train)), output_distribution="uniform", random_state=0)
    z = rank.fit_transform(z)
    t = rank.transform(t)
    scale = StandardScaler().fit(z)
    return scale.transform(z), scale.transform(t)


def choose_alphas(x, a, train, repeat):
    alphas = np.array([.1, 1., 10., 100., 1000.])
    losses = np.zeros((len(alphas), 2))
    for it, iv in KFold(3, shuffle=True, random_state=200+repeat).split(train):
        tr, va = train[it], train[iv]
        z, t = transform(x, tr, va)
        b = a[tr].mean(axis=0)
        q, h = targets(a[tr], b)
        qt, ht = targets(a[va], b)
        for j, alpha in enumerate(alphas):
            pred = Ridge(alpha=alpha).fit(z, np.column_stack([q, h])).predict(t)
            pred[:, 1:] -= pred[:, 1:].mean(axis=1, keepdims=True)
            losses[j] += [np.square(qt-pred[:, 0]).sum(), np.square(ht-pred[:, 1:]).mean(axis=1).sum()]
    return alphas[losses.argmin(axis=0)]


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    TABLES = args.output
    TABLES.mkdir(parents=True, exist_ok=True)
    metric_path = args.tables / "dataset_metrics.tsv"
    feature_path = args.tables / "dataset_feature_means.tsv"
    metrics = pd.read_csv(metric_path, sep="\t")
    features = pd.read_csv(feature_path, sep="\t")
    groups = feature_groups()
    rows, prediction_rows, tuning = [], [], []
    for task, specs in SPECS.items():
        f = features[features.task.eq(task)].pivot(index="dataset_id", columns="feature", values="mean_across_five_splits")
        f["context_to_feature_ratio"] = f.n_train / f.n_features.replace(0, np.nan)
        for metric, key in specs.items():
            key = key if key.startswith("alpha_") else "alpha_free.per_dataset." + key
            w = metrics[metrics.task.eq(task)&metrics.metric.eq(key)].pivot(index="dataset_id", columns="model", values="mean_across_five_splits").sort_index()
            for panel in ("tfm", "full"):
                models = [m for m in w.columns if panel == "full" or m not in BASE]
                a = w[models].to_numpy()
                assert np.isfinite(a).all()
                for learner in ("ridge", "extra_trees"):
                    group_names = list(groups[task]) if learner == "ridge" and panel == "tfm" else ["+Aux"]
                    for group in group_names:
                        x = f.loc[w.index, groups[task][group]].replace([np.inf, -np.inf], np.nan).to_numpy()
                        for repeat in range(5):
                            observed_q = np.full(len(a), np.nan)
                            observed_h = np.full_like(a, np.nan)
                            predicted_q = np.full(len(a), np.nan)
                            predicted_h = np.full_like(a, np.nan)
                            fold_ids = np.full(len(a), -1)
                            for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=4100+repeat).split(a)):
                                z, t = transform(x, train, test)
                                baseline = a[train].mean(axis=0)
                                q, h = targets(a[train], baseline)
                                qt, ht = targets(a[test], baseline)
                                if learner == "ridge":
                                    aq, ah = choose_alphas(x, a, train, repeat)
                                    pq = Ridge(alpha=aq).fit(z, q).predict(t)
                                    ph = Ridge(alpha=ah).fit(z, h).predict(t)
                                    tuning.append(dict(task=task, metric=metric, panel=panel, group=group, repeat=repeat, fold=fold, shared_alpha=aq, differential_alpha=ah))
                                else:
                                    params = dict(n_estimators=EXTRA_TREES_N_ESTIMATORS, min_samples_leaf=4, max_features=.7, random_state=8100+repeat, n_jobs=4)
                                    pq = ExtraTreesRegressor(**params).fit(z, q).predict(t)
                                    # Normalize output dimensions jointly to preserve their equal weights.
                                    ph = ExtraTreesRegressor(**params).fit(z, h).predict(t)
                                ph -= ph.mean(axis=1, keepdims=True)
                                observed_q[test], observed_h[test] = qt, ht
                                predicted_q[test], predicted_h[test] = pq, ph
                                fold_ids[test] = fold
                                np.testing.assert_allclose(np.mean((a[test]-(baseline+pq[:, None]+ph))**2, axis=1), (qt-pq)**2+np.mean((ht-ph)**2, axis=1), rtol=1e-9, atol=1e-12)
                            assert np.isfinite(predicted_q).all() and np.isfinite(predicted_h).all()
                            eq = (observed_q-predicted_q)**2
                            eh = np.mean((observed_h-predicted_h)**2, axis=1)
                            bq = observed_q**2
                            bh = np.mean(observed_h**2, axis=1)
                            record = dict(task=task, metric=metric, panel=panel, learner=learner, group=group, repeat=repeat)
                            for component, errors, zeros in [("shared",eq,bq),("differential",eh,bh)]:
                                rows.append(dict(**record, component=component, gain=1-errors.sum()/zeros.sum(), n_datasets=len(a)))
                            prediction_rows.extend(dict(**record, dataset_id=int(did), fold=int(fold_ids[i]), observed_shared=observed_q[i], predicted_shared=predicted_q[i], shared_loss=eq[i], shared_zero_loss=bq[i], differential_loss=eh[i], differential_zero_loss=bh[i]) for i,did in enumerate(w.index))
                            if group == "+Aux":
                                print(task, metric, panel, learner, repeat, "shared", round(1-eq.sum()/bq.sum(),3), "differential", round(1-eh.sum()/bh.sum(),3), flush=True)
    repeats = pd.DataFrame(rows)
    oof = pd.DataFrame(prediction_rows)
    repeats.to_csv(TABLES / "component_repeats.csv", index=False)
    oof.to_csv(TABLES / "component_oof.csv", index=False)
    pd.DataFrame(tuning).to_csv(TABLES / "component_tuning.csv", index=False)
    keys = ["task","metric","panel","learner","group","component"]
    summary = repeats.groupby(keys).agg(mean_gain=("gain","mean"), sd_gain=("gain","std"), n_datasets=("n_datasets","first")).reset_index()
    summary.to_csv(TABLES / "component_summary.csv", index=False)
    for key, g in oof.groupby(keys[:-1]+["repeat"]):
        for component in ("shared", "differential"):
            gain = 1-g[component+"_loss"].sum()/g[component+"_zero_loss"].sum()
            found = repeats
            for col, value in zip(keys[:-1]+["repeat"], key):
                found = found[found[col].eq(value)]
            np.testing.assert_allclose(gain, found[found.component.eq(component)].gain.iloc[0])
    validation = dict(status="passed", error_identity_verified=True, held_out_unit="dataset_id", outer_folds=5, repeats=5, preprocessing="fold-fitted median imputation, missingness indicators, empirical quantile transform and standardization", ridge_tuning="3-fold nested CV, separately for each component", features=groups, source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [metric_path,feature_path]}, zero_sum_predictions_verified=True)
    (TABLES / "component_validation.json").write_text(json.dumps(validation, indent=2)+"\n")


if __name__ == "__main__":
    main()
