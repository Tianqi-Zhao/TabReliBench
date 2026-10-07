"""Split Localized Conformal Prediction (SLCP).

Reference: Han, Tang, Ghosh, Liu — "Split Localized Conformal Prediction" (2022).
           https://arxiv.org/abs/2206.13092
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from conformal.methods.base import VALID_SCORES
from conformal.methods.slcp_base import BaseSLCP


class SLCP(BaseSLCP):
    """Split Localized Conformal Prediction for regression intervals.

    Constructs prediction intervals by leveraging a
    Nadaraya-Watson (NW) local approximation of the conditional score
    distribution (Algorithm 1, Han et al. 2022).

    The key differences from LCP:

    1. **Training-set weights**: The NW estimator uses the *training* set to
       compute kernel weights rather than the calibration set.  This avoids a
       circular dependency and preserves the split-conformal average coverage
       guarantee (Proposition 3.1).

    2. **Score normalization (studentization)**: Each calibration nonconformity
       score is normalized by subtracting a local quantile estimate obtained
       from the NW estimator.  Conformal correction is then applied to the
       normalized residuals using a standard (unweighted) empirical quantile.

    Parameters
    ----------
    base_model :
        Fitted model exposing ``predict(X)``.
    X_cal, y_cal : array-like
        Calibration features / targets.
    X_train, y_train : array-like
        Training features / targets (required for the NW estimator).
    score : {'absolute', 'variance_normalized', 'quantile', 'pit'}
        Nonconformity score function.
    score_alpha : float
        Quantile level for the ``'quantile'`` score.
    bandwidth : float or None
        Kernel bandwidth h.  When ``None`` (default) the bandwidth is chosen
        automatically:
        - If ``target_neff`` is ``None``: "median trick" — h = median of
          pairwise distances within the training set (paper default).
        - If ``target_neff`` is an integer: binary search targeting the
          given effective sample size (same as LCP auto-bandwidth).
    kernel : {'gaussian', 'box'}
        Kernel function K(·).
    X_cal_dist, X_train_dist : array-like or None
        Optional alternative feature matrices used *only* for distance
        computation (e.g. one-hot encoded categoricals).
    target_neff : int or None
        Target effective sample size for auto-bandwidth.
    """

    def __init__(self, base_model,
                 X_cal=None, y_cal=None,
                 X_train=None, y_train=None,
                 score='absolute', score_alpha=0.1,
                 bandwidth=None, kernel='gaussian',
                 X_cal_dist=None, X_train_dist=None,
                 target_neff=None,
                 device=None, dtype=None):
        if score not in VALID_SCORES:
            raise ValueError(f"score must be one of {VALID_SCORES}, got {score!r}")
        self.base_model   = base_model
        self.X_cal        = None if X_cal   is None else np.asarray(X_cal)
        self.y_cal        = None if y_cal   is None else np.asarray(y_cal).reshape(-1)
        self.X_train      = None if X_train is None else np.asarray(X_train)
        self.y_train      = None if y_train is None else np.asarray(y_train).reshape(-1)
        self.X_cal_dist   = None if X_cal_dist   is None else np.asarray(X_cal_dist)
        self.X_train_dist = None if X_train_dist is None else np.asarray(X_train_dist)
        self.score        = score
        self.score_alpha  = score_alpha
        self._auto_bandwidth = bandwidth is None
        self.bandwidth    = bandwidth
        self.kernel       = kernel
        self.target_neff  = target_neff
        self._fitted      = False
        self._init_backend(device, dtype)

    # ------------------------------------------------------------------
    # Factory (override to handle X_dist)
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(cls, base_model, dataset, **kwargs):
        return cls(
            base_model=base_model,
            X_cal=dataset.val.X,
            y_cal=dataset.val.y,
            X_train=dataset.train.X,
            y_train=dataset.train.y,
            X_cal_dist=getattr(dataset.val,   'X_dist', None),
            X_train_dist=getattr(dataset.train, 'X_dist', None),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        bw = 'auto' if self._auto_bandwidth else f'{self.bandwidth:.3f}'
        return f"SLCP(h={bw}, {self.kernel}, {self._score_tag})"

    @property
    def params(self) -> dict:
        d = {'bandwidth': self.bandwidth, 'kernel': self.kernel,
             'score': self.score, 'target_neff': self.target_neff}
        if self.score in ('quantile', 'quantile_standard'):
            d['score_alpha'] = self.score_alpha
        return d

    @property
    def auto_params(self) -> frozenset:
        return frozenset({'bandwidth'}) if self._auto_bandwidth else frozenset()

    # ------------------------------------------------------------------
    # Hook: _fit_impl
    # ------------------------------------------------------------------

    def _fit_impl(self, v_train):
        X_train_w = (self.X_train_dist if self.X_train_dist is not None
                     else self.X_train)
        X_cal_w   = (self.X_cal_dist   if self.X_cal_dist   is not None
                     else self.X_cal)

        self._v_train = self._t(v_train)

        # Bandwidth selection
        if self.bandwidth is None:
            sq_train = self._np(self._sq_dists(X_train_w, X_train_w, float64=True))
            np.fill_diagonal(sq_train, 0.0)

            if self.target_neff is not None:
                from conformal.methods.utils import auto_bandwidth
                max_h = float(np.sqrt(sq_train.max())) * 2 if sq_train.max() > 0 else 1.0
                if self.kernel == 'gaussian':
                    def kernel_fn(h):
                        log_K = -sq_train / (2 * h ** 2)
                        return np.exp(log_K - log_K.max(axis=1, keepdims=True))
                else:
                    def kernel_fn(h):
                        return (sq_train <= h ** 2).astype(float)
                self.bandwidth = auto_bandwidth(kernel_fn, max_h, self.target_neff)
            else:
                upper_tri = sq_train[np.triu_indices_from(sq_train, k=1)]
                med = float(np.median(np.sqrt(np.clip(upper_tri, 0, None))))
                self.bandwidth = med if med > 0 else 1.0

        # Cal-train kernel weight matrix (n_cal, n_train), unnormalized.
        sq_ct = self._sq_dists(X_cal_w, X_train_w)
        self._W_cal_train = self._kernel_weights(sq_ct)

    # ------------------------------------------------------------------
    # Hook: _local_quantiles_cal
    # ------------------------------------------------------------------

    def _local_quantiles_cal(self, q_level):
        return self._nw_quantile(self._W_cal_train, q_level)

    # ------------------------------------------------------------------
    # Hook: _local_quantiles_test
    # ------------------------------------------------------------------

    def _local_quantiles_test(self, X_test, q_level, **kwargs):
        X_test_dist = kwargs.get('X_test_dist', None)
        X_train_w = (self.X_train_dist if self.X_train_dist is not None
                     else self.X_train)
        X_test_w = (np.asarray(X_test_dist) if X_test_dist is not None
                    else X_test)

        sq_test = self._sq_dists(X_test_w, X_train_w)
        return self._nw_quantile(self._kernel_weights(sq_test), q_level)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kernel_weights(self, sq_dist):
        """Apply kernel to a squared-distance matrix.

        The Gaussian kernel uses a row-wise log-max-shift before exp to prevent
        underflow when distances are large relative to the bandwidth.  The shift
        cancels out after row-normalization and does not change the relative
        weights within each row.
        """
        h2 = self.bandwidth ** 2
        if self.use_torch:
            if self.kernel == 'gaussian':
                log_w = -sq_dist / (2.0 * h2)
                return torch.exp(log_w - log_w.max(dim=1, keepdim=True).values)
            if self.kernel == 'box':
                return (sq_dist <= h2).to(self.dtype)
        else:
            sq = self._np(sq_dist)
            if self.kernel == 'gaussian':
                log_w = -sq / (2.0 * h2)
                return np.exp(log_w - log_w.max(axis=1, keepdims=True))
            if self.kernel == 'box':
                return (sq <= h2).astype(float)
        raise ValueError(f"Unknown kernel: {self.kernel!r}")