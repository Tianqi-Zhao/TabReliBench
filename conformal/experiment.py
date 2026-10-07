"""Run conformal methods on one prepared model/dataset/seed experiment."""
from pathlib import Path
import pickle
from evaluation.store import ArtifactStore
from .predictions import from_predictions
from .metrics import ConformalMetricsCalculator
from .pipeline import ConformalPipeline, load_bundle
from .registry import method_spec


class ConformalExperiment:
    def __init__(self, source, output, n_estimators=None):
        self.bundle = from_predictions(source, n_estimators)
        self._set_output(output)

    @classmethod
    def from_prepared(cls, source, output):
        experiment = cls.__new__(cls)
        experiment.bundle = load_bundle(source)
        experiment._set_output(output)
        return experiment

    def _set_output(self, output):
        self.output = Path(output)
        b = self.bundle
        self.identity = f"dataset_{b['dataset_id']}_seed{b['seed']}_{b['base_model']}"

    def run(self, method, alphas, *, device='cpu', resume=False):
        spec = method_spec(method)
        prediction = self.output/'regression/predictions'/f'{self.identity}__{method}.pkl'
        metrics = self.output/'regression/metrics'/f'{self.identity}__{method}_metrics.pkl'
        if resume and prediction.exists():
            with prediction.open('rb') as f:
                record = pickle.load(f)
            # Old PCP artifacts used a rewritten algorithm, not the pinned author code.
            if spec.algorithm == 'PCP':
                from ._vendor.pcp import IMPLEMENTATION
                if any(record.get('params', {}).get(a, {}).get('implementation')
                       != IMPLEMENTATION for a in alphas):
                    raise ValueError('PCP implementation changed; use a new output directory')
            if (record['provenance']['source_sha256'] != self.bundle['provenance']['source_sha256']
                    or set(record['intervals']) != set(alphas)
                    or record['protocol'] != self.bundle['protocol']):
                raise ValueError('Existing prediction belongs to a different run')
            if record.get('method') != method or record.get('base_model') != self.bundle['base_model']:
                raise ValueError('Existing prediction belongs to a different run')
            if spec.cqr is not None and not spec.cqr.nonnegative:
                if any(record.get('params', {}).get(a, {}).get('score_variant') != spec.cqr.variant
                       or record.get('params', {}).get(a, {}).get('empty_set_policy') != 'exact_sublevel_set_v1'
                       for a in alphas):
                    raise ValueError('CQR score variant changed; use a new output directory')
            if spec.needs_auxiliary_model and (
                    record['provenance'].get('score_features') != 'raw_dataframe_v1'
                    or self.bundle['provenance'].get('score_features') != 'raw_dataframe_v1'):
                raise ValueError('RCP score features changed; regenerate inputs and use a new output directory')
            if spec.needs_auxiliary_model:
                expected = 'feature_preprocessor_v1'
                if self.bundle['base_model'] == 'bart':
                    from .bart_score import PREPROCESSING, score_config
                    expected = PREPROCESSING
                    if record['provenance'].get('auxiliary_config') != score_config(self.bundle):
                        raise ValueError('BART auxiliary configuration changed; use a new output directory')
                if record['provenance'].get('score_preprocessing') != expected:
                    raise ValueError('RCP score preprocessing changed; use a new output directory')
            if not metrics.exists():
                ArtifactStore._atomic_pickle(metrics, ConformalMetricsCalculator(alphas).compute_for_record(record))
        else:
            ConformalPipeline(self.output, alphas, [method], device).run(self.bundle)
        return prediction
