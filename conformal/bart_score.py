"""BART auxiliary score regression using the frozen base run's configuration."""
import numpy as np


PREPROCESSING = 'bart_context_onehot_score_v1'


def score_config(bundle):
    config = bundle['provenance'].get('base_config') or {}
    required = ('n_trees', 'n_draws_total', 'n_burn_per_chain', 'n_chains')
    if any(key not in config for key in required):
        raise ValueError('BART auxiliary fitting requires saved baseline_config hyperparameters')
    return dict(n_trees=config['n_trees'], n_draws=config['n_draws_total'],
                n_burn=config['n_burn_per_chain'], n_chains=config['n_chains'],
                device=config.get('device', 'auto'))


class IndexedBARTScoreRegressor:
    """Fit on score-fit rows only; freeze outcome draws for every query row.

    Reference rows supply features only, never labels or preprocessing statistics.
    Freezing draws makes predictions independent of query order and batching.
    """

    def __init__(self, cache, X, y, seed):
        from evaluation.baselines import BARTBaseline, BARTFeaturePreprocessor
        self.cache = cache
        features = cache.score_feature_values(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(target) != len(features) or not np.isfinite(target).all():
            raise ValueError('Invalid BART auxiliary score targets')
        self.quantiles = {}
        # Constant score targets need no MCMC and otherwise have zero scale.
        self.constant = float(target[0]) if len(target) and np.ptp(target) == 0 else None
        if self.constant is not None:
            self.draws = None
            return
        baseline = BARTBaseline(seed=seed, **score_config(cache.bundle))
        prep = BARTFeaturePreprocessor()
        fitted, _ = prep.fit_transform(features, features)
        model = baseline._fit_regression_model(fitted, target, [])
        reference = prep.transform(cache._score_X)
        self.draws = np.asarray(baseline._sample_regression_posterior_predictive(model, reference))
        if self.draws.shape != (baseline.n_draws, len(reference)) or not np.isfinite(self.draws).all():
            raise ValueError('Invalid BART auxiliary posterior predictive draws')

    def predict_quantile(self, X, q):
        indices = self.cache._indices(X)
        if not 0 < q < 1:
            raise ValueError('Quantile must be strictly between zero and one')
        if self.constant is not None:
            return np.full(len(indices), self.constant)
        if q not in self.quantiles:
            self.quantiles[q] = np.quantile(self.draws, q, axis=0)
        return self.quantiles[q][indices]
