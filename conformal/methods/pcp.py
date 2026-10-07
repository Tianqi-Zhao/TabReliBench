"""Adapter for the pinned author implementation of Posterior Conformal Prediction.

Yao Zhang and Emmanuel J. Candès, arXiv:2409.19712.
The numerical algorithm is vendored verbatim, including its matrix mutation,
cluster-initialization objective and RNG reset behavior. See ../_vendor/pcp.
"""
from __future__ import annotations

from contextlib import contextmanager
import random
import threading

import numpy as np

from conformal._vendor.pcp import COMMIT, IMPLEMENTATION, utils
from conformal.methods.base import BaseConformalMethod, compute_scores, scores_to_intervals

_RNG_LOCK = threading.RLock()


@contextmanager
def _official_rng():
    """Preserve caller RNGs around upstream code; keep upstream internal behavior.

    PCP calls are serialized. Unrelated threads must not concurrently use the
    process-global RNGs while running PCP (the upstream API uses global state).
    """
    with _RNG_LOCK:
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


class PosteriorConformalPredictor(BaseConformalMethod):
    """Official absolute-residual PCP behind the benchmark fit/predict API.

    X_train/y_train are the auxiliary held-out split used for PCP tuning, not
    the data used to fit the base predictor. The benchmark supplies cached
    predictions for this split. No base-model refitting occurs here.

    Each fit creates one upstream PCP instance. Repeated predict calls retain
    its state, including upstream's cache mutations; for independent alphas,
    create fresh instances, as ConformalPipeline already does.
    """

    def __init__(self, base_model, X_cal, y_cal, X_train, y_train,
                 fold=20, grid=9, quantile_lo=10.0, quantile_hi=90.0,
                 m_min=5, random_state=None, score='absolute', score_alpha=0.1,
                 device=None, dtype=None):
        if score != 'absolute':
            raise ValueError("Official PCP adapter supports only score='absolute'")
        self.base_model = base_model
        self.X_cal, self.y_cal = np.asarray(X_cal), np.asarray(y_cal).reshape(-1)
        self.X_train, self.y_train = np.asarray(X_train), np.asarray(y_train).reshape(-1)
        self.fold, self.num_q = fold, grid
        self.quantile_lo, self.quantile_hi = quantile_lo, quantile_hi
        self.m_min, self.score, self.score_alpha = m_min, score, score_alpha
        # MT19937 state matches np.random.seed(seed), not default_rng(seed).
        self._state = (np.random.RandomState(random_state).get_state()
                       if random_state is not None else np.random.get_state())
        self._official = None
        self.threshold_diagnostics = {}
        self._init_backend(device, dtype)

    @classmethod
    def from_dataset(cls, base_model, dataset, **kwargs):
        return cls(base_model, dataset.val.X, dataset.val.y,
                   dataset.train.X, dataset.train.y, **kwargs)

    @property
    def name(self):
        return 'PCP(absolute)'

    @property
    def params(self):
        return dict(fold=self.fold, grid=self.num_q, quantile_lo=self.quantile_lo,
                    quantile_hi=self.quantile_hi, m_min=self.m_min, score=self.score,
                    implementation=IMPLEMENTATION, upstream_commit=COMMIT,
                    seed_compatibility='numpy_integer_to_python_int',
                    tuning_residuals='provided_auxiliary_split')

    def _features(self, X):
        return getattr(self.base_model, 'feature_values', np.asarray)(X)

    def fit(self, info=False):
        R_train = compute_scores(self.base_model, self.X_train, self.y_train, 'absolute')
        self._R_cal = compute_scores(self.base_model, self.X_cal, self.y_cal, 'absolute')
        with _official_rng():
            np.random.set_state(self._state)
            self._official = utils.PCP(fold=self.fold, grid=self.num_q,
                                       l=self.quantile_lo, u=self.quantile_hi,
                                       m_min=self.m_min)
            self._official.train(self._features(self.X_train), R_train, info=info)
        return self

    def predict(self, X_test, alpha=0.05, max_iter=10, tol=0.005):
        if not 0 < alpha < 1:
            raise ValueError('alpha must be in (0, 1).')
        if self._official is None:
            self.fit()
        X_test = np.asarray(X_test)
        y_pred = np.asarray(self.base_model.predict(X_test)).reshape(-1)
        with _official_rng():
            # R_test is used only for upstream's discarded coverage indicators.
            # finite=False prevents any test-residual-based interval replacement.
            thresholds, _ = self._official.calibrate(
                self._features(self.X_cal), self._R_cal, self._features(X_test),
                np.zeros(len(X_test)), alpha, finite=False, return_pi=False,
                max_iter=max_iter, tol=tol)
        self.thresholds_ = np.asarray(thresholds)
        return scores_to_intervals(
            self.base_model, X_test, self.thresholds_, 'absolute', y_pred=y_pred,
            threshold_diagnostics=self.threshold_diagnostics)
