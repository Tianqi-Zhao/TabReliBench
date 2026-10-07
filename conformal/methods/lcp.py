"""Local Conformal Prediction (LCP) with optional Torch backend."""

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from conformal.methods.wcp import CalWCP


class CalLCP(CalWCP):
    """Local Conformal Prediction for regression intervals.

    When *bandwidth* is ``None`` (the default), it is chosen automatically so
    that the effective sample size n_eff(h) ≈ *target_neff* (Equation 21).
    Per the paper, the expected values in Eq. 21 are estimated on the
    *training* set (``X_train``).  When ``X_train`` is not provided, the
    calibration set is used as a fallback.
    """

    def __init__(self, base_model, bandwidth=None, kernel="gaussian",
                 score="absolute", score_alpha=0.1, X_cal=None, y_cal=None,
                 X_train=None, target_neff=50,
                 X_cal_dist=None, X_train_dist=None,
                 device=None, dtype=None):
        super().__init__(base_model, X_cal=X_cal, y_cal=y_cal,
                         score=score, score_alpha=score_alpha,
                         X_cal_dist=X_cal_dist,
                         device=device, dtype=dtype)
        self._auto_bandwidth = bandwidth is None
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.X_train = None if X_train is None else np.asarray(X_train)
        self.X_train_dist = None if X_train_dist is None else np.asarray(X_train_dist)
        self.target_neff = target_neff

    @classmethod
    def from_dataset(cls, base_model, dataset, **kwargs):
        return cls(
            base_model=base_model,
            X_cal=dataset.val.X,
            y_cal=dataset.val.y,
            X_train=dataset.train.X,
            X_cal_dist=getattr(dataset.val, 'X_dist', None),
            X_train_dist=getattr(dataset.train, 'X_dist', None),
            **kwargs,
        )

    @property
    def name(self) -> str:
        if self._auto_bandwidth:
            return f"CalLCP(h=auto, {self._score_tag})"
        return f"CalLCP(h={self.bandwidth:.3f}, {self._score_tag})"

    @property
    def params(self) -> dict:
        d = {"bandwidth": self.bandwidth,
             "kernel": self.kernel, "score": self.score,
             "target_neff": self.target_neff}
        if self.score in ('quantile', 'quantile_standard'):
            d["score_alpha"] = self.score_alpha
        return d

    def fit(self):
        if self.bandwidth is None:
            from conformal.methods.utils import auto_bandwidth
            # Use one-hot features for distance-based bandwidth search
            X_ref = (self.X_train_dist if self.X_train_dist is not None
                     else self.X_train if self.X_train is not None
                     else self.X_cal_dist if self.X_cal_dist is not None
                     else self.X_cal)
            # Use float64 to avoid float32 GEMM catastrophic cancellation.
            sq_dists = self._np(self._sq_dists(X_ref, X_ref, float64=True))
            np.fill_diagonal(sq_dists, 0.0)
            max_h = float(np.sqrt(sq_dists.max())) * 2 if sq_dists.max() > 0 else 1.0
            if self.kernel == "gaussian":
                def kernel_fn(h):
                    log_K = -sq_dists / (2 * h ** 2)
                    return np.exp(log_K - log_K.max(axis=1, keepdims=True))
            else:  # box
                def kernel_fn(h):
                    return (sq_dists <= h ** 2).astype(float)
            self.bandwidth = auto_bandwidth(kernel_fn, max_h, self.target_neff)
        return super().fit()

    def _compute_weights(self, X_query, X_ref=None):
        if X_ref is None:
            X_ref = self.X_cal
        # _sq_dists returns backend-native (tensor on GPU, numpy on CPU).
        # torch.cdist is the key accelerated operation in the torch path.
        sq_dist = self._sq_dists(X_query, X_ref)
        if self.kernel == "gaussian":
            exp_fn = torch.exp if self.use_torch else np.exp
            return exp_fn(-sq_dist / (2 * self.bandwidth ** 2))
        elif self.kernel == "box":
            mask = sq_dist <= self.bandwidth ** 2
            return mask.to(self.dtype) if self.use_torch else mask.astype(float)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_self_weights(self, X_query):
        n = len(X_query)
        if self.use_torch:
            return torch.ones(n, device=self.device, dtype=self.dtype)
        return np.ones(n)
