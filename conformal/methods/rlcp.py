"""Randomized Local Conformal Prediction (RLCP) with optional Torch backend.

`RLCP` follows a compact object API:
- `fit()` computes calibration residual scores.
- `predict(X_test, alpha)` returns `(y_pred, lower, upper)`.

When PyTorch is available, heavy tensor operations run on GPU when possible;
otherwise computations fall back to NumPy on CPU.
"""

import numpy as np

from conformal.methods.base import (BaseConformalMethod, VALID_SCORES,
                           compute_scores, scores_to_intervals)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class RLCP(BaseConformalMethod):
    """Randomized Local Conformal Prediction for regression intervals.

    When *bandwidth* is ``None`` (the default), it is chosen automatically so
    that the RLCP-specific effective sample size ñ_eff(h) ≈ *target_neff*,
    using perturbed kernel weights (Equation 21, RLCP variant).
    """

    def __init__(self,
                 base_model,
                 bandwidth=None,
                 kernel="gaussian",
                 score='absolute',
                 score_alpha=0.1,
                 X_cal=None,
                 y_cal=None,
                 X_train=None,
                 random_state=0,
                 device=None,
                 dtype=None,
                 target_neff=50,
                 X_cal_dist=None,
                 X_train_dist=None):
        self.base_model = base_model
        if kernel not in ["gaussian", "box"]:
            raise ValueError(
                f"Unknown kernel: {kernel}. Use 'gaussian' or 'box'.")
        self.kernel = kernel
        if score not in VALID_SCORES:
            raise ValueError(
                f"score must be one of {VALID_SCORES}, got {score!r}")
        self.score = score
        self.score_alpha = score_alpha
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.X_train = None if X_train is None else np.asarray(X_train)
        self.X_cal_dist = None if X_cal_dist is None else np.asarray(X_cal_dist)
        self.X_train_dist = None if X_train_dist is None else np.asarray(X_train_dist)
        self._auto_bandwidth = bandwidth is None
        self.bandwidth = bandwidth
        self.target_neff = target_neff
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self._init_backend(device, dtype)
        if self.use_torch:
            self._torch_rng = torch.Generator(device=self.device)
            self._torch_rng.manual_seed(random_state)
        else:
            self._torch_rng = None

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
            return f"RLCP(h=auto, {self._score_tag})"
        return f"RLCP(h={self.bandwidth:.3f}, {self._score_tag})"

    @property
    def auto_params(self) -> frozenset[str]:
        return frozenset(["bandwidth"]) if self._auto_bandwidth else frozenset()

    @property
    def params(self) -> dict:
        d = {"bandwidth": self.bandwidth,
             "kernel": self.kernel, "target_neff": self.target_neff,
             "score": self.score}
        if self.score in ('quantile', 'quantile_standard'):
            d["score_alpha"] = self.score_alpha
        return d

    @staticmethod
    def _weighted_quantile_apply_torch(values, weights, quantile):
        sorted_values, order = torch.sort(values)
        sorted_weights = torch.index_select(weights, 1, order)
        sorted_weights = sorted_weights / torch.clamp(
            sorted_weights.sum(dim=1, keepdim=True), min=1e-12)
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        idx = torch.sum(cum_weights < quantile, dim=1)
        idx = torch.clamp(idx, max=sorted_values.shape[0] - 1)
        return sorted_values[idx]

    @staticmethod
    def _sample_uniform_ball_numpy(n, d, bandwidth, rng):
        z = rng.normal(size=(n, d))
        z_norm = np.linalg.norm(z, axis=1, keepdims=True)
        z_norm = np.clip(z_norm, 1e-12, None)
        u = rng.uniform(size=(n, 1))
        radii = bandwidth * (u**(1.0 / d))
        return radii * (z / z_norm)

    @staticmethod
    def _weighted_quantile_apply_numpy(values, weights, quantile):
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[:, sorted_idx]
        sorted_weights = sorted_weights / np.clip(
            sorted_weights.sum(axis=1, keepdims=True), 1e-12, None)
        cum_weights = np.cumsum(sorted_weights, axis=1)
        idx = np.sum(cum_weights < quantile, axis=1)
        idx = np.clip(idx, 0, sorted_values.shape[0] - 1)
        return sorted_values[idx]

    @property
    def calibration_scores(self):
        """Calibration scores as a numpy array (public API)."""
        return self._np(self._calibration_scores)

    def fit(self):
        """Compute and cache calibration nonconformity scores."""
        if self.bandwidth is None:
            from conformal.methods.utils import auto_bandwidth
            # Use one-hot features for distance-based bandwidth search
            X_d = (self.X_train_dist if self.X_train_dist is not None
                   else self.X_train if self.X_train is not None
                   else self.X_cal_dist if self.X_cal_dist is not None
                   else self.X_cal)
            n, d = X_d.shape

            sq_dists_train = self._np(self._sq_dists(X_d, X_d, float64=True))
            np.fill_diagonal(sq_dists_train, 0.0)
            max_h_val = float(np.sqrt(sq_dists_train.max()))
            max_h = max_h_val * 2 if max_h_val > 0 else 1.0

            if self.kernel == "gaussian":
                def kernel_fn(h):
                    rng = np.random.default_rng(self.random_state)
                    X_tilde = X_d + rng.normal(size=(n, d)) * h
                    sq_dists = self._np(self._sq_dists(X_d, X_tilde, float64=True))
                    log_K = -sq_dists / (2 * h ** 2)
                    return np.exp(log_K - log_K.max(axis=1, keepdims=True))
            else:  # box
                def kernel_fn(h):
                    rng = np.random.default_rng(self.random_state)
                    z = rng.normal(size=(n, d))
                    z_norm = np.clip(
                        np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
                    u = rng.uniform(size=(n, 1))
                    X_tilde = X_d + h * (u ** (1.0 / d)) * (z / z_norm)
                    sq_dists = self._np(self._sq_dists(X_d, X_tilde, float64=True))
                    return (sq_dists <= h ** 2).astype(float)
            self.bandwidth = auto_bandwidth(kernel_fn, max_h, self.target_neff)
        cal_scores = compute_scores(
            self.base_model, self.X_cal, self.y_cal,
            self.score, self.score_alpha)
        self._calibration_scores = self._t(cal_scores)
        # Store both ordinal (for TabPFN) and one-hot (for distance) cal features
        self._X_cal = self._t(self.X_cal)
        self._X_cal_dist = self._t(self.X_cal_dist) if self.X_cal_dist is not None else self._X_cal
        return

    def _compute_weights(self, X_test, X_test_dist=None):
        """Compute randomized local kernel weights for test points.

        Parameters
        ----------
        X_test : array
            Test features (unused when *X_test_dist* is provided — kept for
            API consistency).
        X_test_dist : array or None
            One-hot encoded test features for distance computation.  Falls
            back to ``X_test`` when ``None``.

        Returns ``(cal_weights, self_weights)`` where *cal_weights* has shape
        ``(n_test, n_cal)`` and *self_weights* has shape ``(n_test,)``.
        """
        X_d = X_test_dist if X_test_dist is not None else X_test
        n_test, d = np.asarray(X_d).shape

        if self.kernel == "gaussian":
            if self.use_torch:
                X_ = self._t(X_d)
                noise = torch.randn(n_test, d, generator=self._torch_rng,
                                    device=self.device, dtype=self.dtype)
                log_self = -torch.sum(noise ** 2, dim=-1) / 2
            else:
                X_ = np.asarray(X_d)
                noise = self.rng.normal(size=(n_test, d))
                log_self = -np.sum(noise ** 2, axis=-1) / 2

            sq_dist = self._sq_dists(X_ + noise * self.bandwidth, self._X_cal_dist)
            log_cal = -sq_dist / (2 * self.bandwidth ** 2)

            if self.use_torch:
                max_log = torch.maximum(log_cal.max(dim=1).values, log_self)
            else:
                max_log = np.maximum(log_cal.max(axis=1), log_self)
            log_cal = log_cal - max_log[:, None]
            log_self = log_self - max_log
            exp_fn = torch.exp if self.use_torch else np.exp
            return exp_fn(log_cal), exp_fn(log_self)

        elif self.kernel == "box":
            if self.use_torch:
                X_ = self._t(X_d)
                z = torch.randn(n_test, d, generator=self._torch_rng,
                                device=self.device, dtype=self.dtype)
                z_norm = torch.clamp(
                    torch.linalg.norm(z, dim=1, keepdim=True), min=1e-12)
                u = torch.rand(n_test, 1, generator=self._torch_rng,
                               device=self.device, dtype=self.dtype)
                X_ = X_ + self.bandwidth * torch.pow(u, 1.0 / d) * (z / z_norm)
                self_weights = torch.ones(n_test, device=self.device, dtype=self.dtype)
            else:
                X_ = np.asarray(X_d)
                X_ = X_ + self._sample_uniform_ball_numpy(
                    n_test, d, self.bandwidth, self.rng)
                self_weights = np.ones(n_test)

            sq_dist = self._sq_dists(X_, self._X_cal_dist)
            mask = sq_dist <= self.bandwidth ** 2
            weights = mask.to(self.dtype) if self.use_torch else mask.astype(float)
            return weights, self_weights

        raise ValueError(f"Unknown kernel: {self.kernel}")

    def predict(self, X_test, alpha=0.05, X_test_dist=None):
        """Predict point values and RLCP intervals.

        Parameters
        ----------
        X_test : array of shape (n_test, d)
            Test features (ordinal encoding for TabPFN).
        alpha : float
            Miscoverage level (e.g. 0.05 for 95% intervals).
        X_test_dist : array of shape (n_test, d') or None
            One-hot features for distance computation.

        Returns
        -------
        Tuple of NumPy arrays: ``(y_pred, lower, upper)``.
        """
        self.fit()
        weights, self_weights = self._compute_weights(X_test,
                                                       X_test_dist=X_test_dist)
        if self.use_torch:
            inf_score = torch.tensor([float("inf")],
                                     device=self.device,
                                     dtype=self.dtype)
            scores_augmented = torch.cat(
                [self._calibration_scores, inf_score])
            weights_augmented = torch.cat(
                [weights, self_weights.unsqueeze(1)], dim=1)
            score_threshold = self._weighted_quantile_apply_torch(
                scores_augmented, weights_augmented, 1 - alpha)
        else:
            scores_augmented = np.concatenate(
                [self._calibration_scores, [np.inf]])
            weights_augmented = np.hstack(
                [weights, self_weights[:, None]])
            score_threshold = self._weighted_quantile_apply_numpy(
                scores_augmented, weights_augmented, 1 - alpha)
        thresholds = self._np(score_threshold)
        return scores_to_intervals(
            self.base_model, X_test, thresholds,
            self.score, self.score_alpha,
            threshold_diagnostics=self.__dict__.setdefault("threshold_diagnostics", {}))
