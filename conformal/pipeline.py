"""One validated input bundle -> predictions -> current benchmark metrics."""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
import time
import numpy as np
import pandas as pd
from evaluation.ppd import PPDQuantileGrid
from evaluation.store import ArtifactStore
from .models import CachedPredictor
from .allocation import usage
from .registry import METHODS, DEFAULT_METHODS, make_method, method_spec
from .metrics import ConformalMetricsCalculator


def load_bundle(path):
    with Path(path).open('rb') as f:
        bundle = pickle.load(f)
    validate_bundle(bundle)
    return bundle


def validate_bundle(b):
    if b.get('schema') != 'conformal_input_v1':
        raise ValueError('Expected conformal_input_v1, use prepare first')
    levels = np.asarray(b['quantile_levels'])
    if levels.ndim != 1 or len(levels)<2 or not np.all(np.diff(levels)>0) or not np.all((levels>0)&(levels<1)):
        raise ValueError('Invalid quantile levels')
    ids=[]
    raw_splits = [b[name].get('X_raw') for name in ('train', 'cal', 'test')]
    if any('X_raw' in b[name] for name in ('train', 'cal', 'test')):
        if not all(isinstance(raw, pd.DataFrame) for raw in raw_splits):
            raise ValueError('X_raw must be a DataFrame in every split')
        if not all(raw.columns.equals(raw_splits[0].columns) and raw.dtypes.equals(raw_splits[0].dtypes)
                   for raw in raw_splits):
            raise ValueError('X_raw columns and dtypes must match across splits')
    for name in ('train','cal','test'):
        s=b[name]; y=np.asarray(s['y']); X=np.asarray(s['X']); q=np.asarray(s['ppd_quantiles']); p=np.asarray(s['point_pred'])
        if y.ndim!=1 or len(y)<2 or X.ndim!=2 or len(X)!=len(y) or q.shape!=(len(y),len(levels)) or p.shape!=y.shape:
            raise ValueError(f'Invalid {name} shapes')
        if not all(np.isfinite(v).all() for v in (X,y,q,p)):
            raise ValueError(f'Nonfinite {name} input')
        if 'X_raw' in s and len(s['X_raw']) != len(y):
            raise ValueError(f'Raw {name} features not aligned')
        if np.any(np.diff(q,axis=1)<0):
            raise ValueError('Nonmonotone predictive grid')
        part=list(s['row_ids'])
        if len(part)!=len(y):raise ValueError('Missing row IDs')
        ids.extend(part)
    if len(set(ids))!=len(ids):raise ValueError('Training/calibration/test overlap')
    if len(b['X_test_raw'])!=len(b['test']['y']):raise ValueError('Raw test features not aligned')


class ConformalPipeline:
    def __init__(self, output, alphas=(.05,.1,.15,.2), methods=DEFAULT_METHODS, device='cpu'):
        self.output=Path(output);self.alphas=list(alphas);self.methods=list(methods);self.device=device
        if not self.methods or not set(methods)<=set(METHODS):raise ValueError('Unknown/empty method selection')
        self.metrics=ConformalMetricsCalculator(self.alphas)

    def run(self,bundle,*,clone_factory=None):
        validate_bundle(bundle)
        model=CachedPredictor(bundle,clone_factory);test=bundle['test']; written=[]
        if any(method_spec(name).needs_auxiliary_model for name in self.methods):
            if bundle['base_model'] == 'bart' and clone_factory is None:
                from .bart_score import score_config
                score_config(bundle)
            model.score_feature_values(model.query('train')[:0])
        identity=f"dataset_{bundle['dataset_id']}_seed{bundle['seed']}_{bundle['base_model']}"
        for name in self.methods:
            spec = method_spec(name)
            dest=self.output/'regression'/'predictions'/f'{identity}__{name}.pkl'
            metric_dest=self.output/'regression'/'metrics'/f'{identity}__{name}_metrics.pkl'
            if dest.exists() or metric_dest.exists():
                raise FileExistsError(f'Refusing to mix/overwrite an existing run: {dest}')
            start=time.perf_counter(); intervals={}
            method = None
            params_by_alpha = {}
            for alpha in self.alphas:
                if name == 'Vanilla':
                    lower,upper=PPDQuantileGrid(test['ppd_quantiles'],bundle['quantile_levels']).interval(alpha)
                else:
                    method=make_method(name,model,bundle,device=self.device,score_alpha=alpha)
                    method.fit()
                    params_by_alpha[alpha]=dict(method.params, score=method.score)
                    if spec.cqr is not None:
                        params_by_alpha[alpha].update(score_alpha=alpha,score_variant=spec.cqr.variant)
                    distance_args = dict(X_test_dist=test['X']) if spec.uses_distances else {}
                    _,lower,upper=method.predict(model.query('test'),alpha=alpha,**distance_args)
                    params_by_alpha[alpha]['threshold_diagnostics'] = dict(method.threshold_diagnostics)
                intervals[alpha]={'lower':np.asarray(lower),'upper':np.asarray(upper)}
                if spec.cqr is not None and not spec.cqr.nonnegative:
                    intervals[alpha]['empty_set'] = np.asarray(lower) > np.asarray(upper)
                    params_by_alpha[alpha]['empty_set_policy'] = 'exact_sublevel_set_v1'
            record={k:bundle[k] for k in ('dataset_id','seed','base_model','protocol','provenance')}
            if spec.needs_auxiliary_model:
                record['provenance'] = dict(
                    bundle['provenance'], score_features='raw_dataframe_v1',
                    score_preprocessing='feature_preprocessor_v1')
                if bundle['base_model'] == 'bart' and clone_factory is None:
                    from .bart_score import PREPROCESSING, score_config
                    record['provenance'].update(
                        score_preprocessing=PREPROCESSING,
                        auxiliary_model='bartz.Bart', auxiliary_config=score_config(bundle),
                        auxiliary_distribution='posterior_outcome_samples',
                        auxiliary_prediction_policy='frozen_reference_draws_v1')
            record.update(schema='conformal_prediction_v1',method=name,model=f"{bundle['base_model']}__{name}",
                          X_test=bundle['X_test_raw'],feature_names=list(bundle['X_test_raw'].columns),
                          y_test=test['y'],point_pred=test['point_pred'],intervals=intervals,
                          params=params_by_alpha, data_usage=usage(name,bundle),
                          elapsed_seconds=time.perf_counter()-start)
            # Save predictions before evaluation so a diagnostic failure does not lose expensive inference.
            ArtifactStore._atomic_pickle(dest,record)
            result=self.metrics.compute_for_record(record)
            ArtifactStore._atomic_pickle(metric_dest,result)
            written.append(str(dest));print(f'{identity} {name}: saved',flush=True)
        return written
