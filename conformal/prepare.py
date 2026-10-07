"""Prepare frozen TFM predictions without using calibration/test labels to fit."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.data import DatasetLoader
from evaluation.spec import ExperimentSpec
from evaluation.models import ModelRegistry, RegressionModelRunner, DEFAULT_QUANTILE_GRID
from evaluation.store import ArtifactStore
from .pipeline import validate_bundle
from .features import distance_features


def prepare(dataset_id, seed, base_model, output, n_estimators=8, max_n=10000, reference=None, split_path=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if split_path is None:
        split=DatasetLoader(max_n=max_n).load_for_spec(ExperimentSpec(dataset_id,seed,1.0,task='regression'))
        # Preserve the benchmark test rows and order; reserve calibration from training only.
        order=np.random.default_rng(seed+2).permutation(len(split.y_train))
        cut=len(order)//2
        tr,cal=order[:cut],order[cut:]
        protocol='benchmark_test_fixed_train_half_cal_v1'
    else:
        import pickle
        from types import SimpleNamespace
        with Path(split_path).open('rb') as f:original=pickle.load(f)
        dataset_id,seed=int(original['dataset_id']),int(original['seed'])
        ntr=len(original['y_train']);ncal=len(original['y_cal'])
        tr,cal=np.arange(ntr),np.arange(ntr,ntr+ncal)
        split=SimpleNamespace(
            X_train=pd.concat([original['X_train_raw'],original['X_cal_raw']],ignore_index=True),
            y_train=np.concatenate([original['y_train'],original['y_cal']]),
            X_test=original['X_test'],y_test=original['y_test'])
        protocol=original['protocol']+'_refit_v1'
    if min(len(tr),len(cal),len(split.y_test))<2:raise ValueError('Split too small')
    X_train=split.X_train.iloc[tr].copy()
    queries=pd.concat([X_train,split.X_train.iloc[cal],split.X_test],ignore_index=True)
    if reference:
        import pickle
        with Path(reference).open('rb') as f:r=pickle.load(f)
        if int(r['dataset_id'])!=dataset_id or int(r['seed'])!=seed:
            raise ValueError('Reference identity mismatch')
        pd.testing.assert_frame_equal(split.X_test.reset_index(drop=True),r['X_test'].reset_index(drop=True))
        np.testing.assert_array_equal(split.y_test,r['y_test'])
    config=ModelRegistry.default_regression()[base_model]
    runner=RegressionModelRunner(config,seed,n_estimators=n_estimators)
    ppd,point=runner.fit_predict(X_train,split.y_train[tr],queries,DEFAULT_QUANTILE_GRID)
    features=distance_features(X_train,queries)
    lengths=[len(tr),len(cal),len(split.y_test)]
    ys=[split.y_train[tr],split.y_train[cal],split.y_test]
    # Row IDs are positions within the canonical train/test partitions, not feature hashes.
    ids=[[f'train:{i}' for i in tr],[f'train:{i}' for i in cal],[f'test:{i}' for i in range(len(split.y_test))]]
    b=dict(schema='conformal_input_v1',dataset_id=dataset_id,seed=seed,base_model=base_model,
           n_estimators=n_estimators,quantile_levels=DEFAULT_QUANTILE_GRID,
           X_test_raw=split.X_test,protocol=protocol,
           provenance={'max_n':max_n,'calibration_seed':seed+2,'base_train_fraction':len(tr)/(len(split.y_train)+len(split.y_test)),
                       'source_split':str(split_path) if split_path else None,
                       'test_reference':str(reference) if reference else None,
                       'checkpoint':runner.checkpoint_manifest(),
                       'distance_features':'training_only_numeric_standardize_onehot',
                       'score_features':'raw_dataframe_v1',
                       'ppd':'199 quantiles, constant tail extension; NORM std and PIT use grid approximation'})
    offset=0
    for name,n,y,row_ids in zip(('train','cal','test'),lengths,ys,ids):
        s=slice(offset,offset+n)
        b[name]=dict(X=features[s],X_raw=queries.iloc[s].reset_index(drop=True),
                     y=y,row_ids=row_ids,ppd_quantiles=ppd[s],point_pred=point[s])
        offset+=n
    validate_bundle(b);ArtifactStore._atomic_pickle(output,b)
    return output
