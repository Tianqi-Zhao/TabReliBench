"""Convert held-out benchmark predictions without fitting the base TFM."""
import hashlib
from pathlib import Path
import pickle
import numpy as np
from evaluation.store import ArtifactStore
from .features import distance_features
from .pipeline import validate_bundle

from .allocation import EQUAL_BUDGET_PROTOCOL

PROTOCOL = EQUAL_BUDGET_PROTOCOL
MODEL_ALIASES = {'tabicl': 'tabiclv2', 'tabiclv2': 'tabiclv2',
                 'tabpfnv2': 'tabpfnv2', 'tabpfnv2.5': 'tabpfnv2.5', 'tabpfnv3': 'tabpfnv3',
                 'tabpfnv3.5': 'tabpfnv3.5', 'causilo': 'causilo', 'tabdpt1.3': 'tabdpt1.3',
                 'bart': 'bart'}


def from_predictions(source, n_estimators=None):
    source = Path(source)
    payload = source.read_bytes()
    record = pickle.loads(payload)
    if record.get('task') != 'regression':
        raise ValueError('Expected regression predictions')
    base_model = MODEL_ALIASES[record['model']]
    # BART tree/draw counts belong to its saved baseline_config, not the
    # TFM ensemble setting. Both models use the same prediction-grid adapter.
    if base_model == 'bart':
        n_estimators = None
    elif n_estimators is None or n_estimators < 1:
        raise ValueError('TFM predictions require a positive n_estimators')
    seed, dataset_id = int(record['seed']), int(record['dataset_id'])
    X = record['X_test'].reset_index(drop=True)
    y = np.asarray(record['y_test'])
    n = len(y)
    # Depends only on dataset and seed, never on model, ensemble or outcomes.
    split_seed = np.random.SeedSequence([20260929, dataset_id, seed])
    order = np.random.default_rng(split_seed).permutation(n)
    quarter = n // 4
    indices = dict(train=order[:quarter], cal=order[quarter:2*quarter], test=order[2*quarter:])
    if quarter < 10:
        raise ValueError('Insufficient held-out rows for auxiliary fitting and calibration')
    features = distance_features(X.iloc[indices['train']], X)
    bundle = dict(schema='conformal_input_v1', dataset_id=dataset_id, seed=seed,
                  base_model=base_model, n_estimators=n_estimators,
                  quantile_levels=record['quantile_levels'], protocol=PROTOCOL,
                  X_test_raw=X.iloc[indices['test']].reset_index(drop=True),
                  provenance=dict(source_prediction=str(source.resolve()),
                      source_sha256=hashlib.sha256(payload).hexdigest(),
                      base_refitted=False, base_n_context=record.get('n_context'),
                      n_estimators=n_estimators,
                      base_config=record.get('baseline_config'),
                      split_seed_components=[20260929, dataset_id, seed],
                      train_role='auxiliary subset of original held-out predictions, not base training',
                      original_test_size=n, split_sizes={k:len(v) for k,v in indices.items()},
                      distance_features='auxiliary_only_numeric_standardize_onehot',
                      score_features='raw_dataframe_v1',
                      ppd='saved quantile grid; NORM std and PIT use grid approximation'))
    for name, idx in indices.items():
        bundle[name] = dict(X=features[idx], X_raw=X.iloc[idx].reset_index(drop=True), y=y[idx],
                           row_ids=[f'heldout:{int(i)}' for i in idx],
                           ppd_quantiles=np.asarray(record['ppd_quantiles'])[idx],
                           point_pred=np.asarray(record['point_pred'])[idx])
    validate_bundle(bundle)
    return bundle


def prepare_predictions(source, output, n_estimators=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    ArtifactStore._atomic_pickle(output, from_predictions(source, n_estimators))
    return output
