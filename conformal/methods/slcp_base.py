"""BaseSLCP — abstract base for the SLCP family (Generalized SLCP).

Implements the common predict flow shared by SLCP, PDSLCP, and DRSLCP,
corresponding to Theorem 3.4 (Generalized SLCP) of Han et al. (2022).

All three methods follow the same four-step prediction procedure:

  1. Estimate local quantiles q(X_cal_i) for calibration points.
  2. Normalize calibration scores and compute conformal correction d.
  3. Estimate local quantiles q(X_test) for test points.
  4. Construct intervals via ``scores_to_intervals``.

Subclasses differ only in *how* the local conditional CDF F_hat(V|x) is
estimated and how quantiles are extracted from it:

  - SLCP:   Nadaraya-Watson weighted quantile + Euclidean kernel
  - PDSLCP: Nadaraya-Watson weighted quantile + Wasserstein kernel
  - DRSLCP: Doubly-robust CDF inversion  + Wasserstein kernel + score model CDF
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from conformal.methods.base import conformal_quantile

try:
    import torch
except ImportError:
    torch = None

from conformal.methods.base import (BaseConformalMethod, VALID_SCORES, SIGNED_SCORES,
                           compute_scores, scores_to_intervals,
                           compute_signed_scores, signed_scores_to_intervals)


class BaseSLCP(BaseConformalMethod):
    """Abstract base class for SLCP-family conformal methods.

    Subclasses must implement three hooks:

    * ``_fit_impl(v_train)``
    * ``_local_quantiles_cal(q_level)``
    * ``_local_quantiles_test(X_test, q_level)``

    Subclasses must set ``self.score`` and ``self.score_alpha`` in their
    ``__init__`` so that ``fit`` can compute general nonconformity scores.
    """

    @classmethod
    def from_dataset(cls, base_model, dataset, **kwargs):
        return cls(
            base_model=base_model,
            X_cal=dataset.val.X,
            y_cal=dataset.val.y,
            X_train=dataset.train.X,
            y_train=dataset.train.y,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self):
        """Compute nonconformity scores and delegate to subclass ``_fit_impl``."""
        if self.score in SIGNED_SCORES:
            v_train = compute_signed_scores(
                self.base_model, self.X_train, self.y_train, self.score)
            v_cal = compute_signed_scores(
                self.base_model, self.X_cal, self.y_cal, self.score)
        else:
            v_train = compute_scores(
                self.base_model, self.X_train, self.y_train,
                self.score, self.score_alpha)
            v_cal = compute_scores(
                self.base_model, self.X_cal, self.y_cal,
                self.score, self.score_alpha)
        self._v_cal = v_cal

        self._fit_impl(v_train)

        self._fitted = True
        return self

    def predict(self, X_test, alpha=0.05, X_test_dist=None, **kwargs):
        """Return ``(y_pred, lower, upper)`` prediction intervals.

        Parameters
        ----------
        X_test : array of shape (m, d)
        alpha : float
            Nominal miscoverage level; intervals target >= 1-alpha coverage.
        X_test_dist : array of shape (m, d') or None
            Optional one-hot encoded test features for distance-based methods
            (e.g. SLCP).  When provided, distances are computed in this space
            rather than the ordinal ``X_test`` space.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if not self._fitted:
            self.fit()

        X_test = np.asarray(X_test)
        kw = dict(X_test_dist=X_test_dist, **kwargs)
        if self.score in SIGNED_SCORES:
            return self._predict_signed(X_test, alpha, **kw)
        return self._predict_unsigned(X_test, alpha, **kw)

    # ------------------------------------------------------------------
    # Predict paths
    # ------------------------------------------------------------------

    def _predict_signed(self, X_test, alpha, **kwargs):
        """Asymmetric intervals via signed scores (absolute / variance_normalized)."""
        q_level = 1.0 - alpha / 2
        n_cal = len(self._v_cal)

        # Step 1 — local quantiles for calibration at both tails
        q_hi_cal = self._local_quantiles_cal(q_level)
        q_lo_cal = self._local_quantiles_cal(1 - q_level)

        # Step 2 — normalize and conformal correction (one per tail)
        v_norm_upper = self._np(self._v_cal) - self._np(q_hi_cal)
        v_norm_lower = self._np(q_lo_cal) - self._np(self._v_cal)

        q_outer_level = np.ceil(q_level * (n_cal + 1)) / n_cal
        q_hat_upper = conformal_quantile(v_norm_upper, alpha / 2)
        q_hat_lower = conformal_quantile(v_norm_lower, alpha / 2)

        # Step 3 — local quantiles for test at both tails
        q_hi_test = self._local_quantiles_test(X_test, q_level, **kwargs)
        q_lo_test = self._local_quantiles_test(X_test, 1 - q_level, **kwargs)

        # Step 4 — thresholds in signed-score space → intervals
        upper_thresh = self._np(q_hi_test) + q_hat_upper
        lower_thresh = self._np(q_lo_test) - q_hat_lower

        return signed_scores_to_intervals(
            self.base_model, X_test, lower_thresh, upper_thresh,
            self.score)

    def _predict_unsigned(self, X_test, alpha, **kwargs):
        """Symmetric-threshold inversion (CQR variants / PIT), allowing signed thresholds."""
        q_level = 1.0 - alpha
        n_cal = len(self._v_cal)

        # Step 1 — local quantiles for calibration points (subclass hook)
        q_cal = self._local_quantiles_cal(q_level)

        # Step 2 — normalize calibration scores and conformal correction
        v_norm = self._np(self._v_cal) - self._np(q_cal)

        q_outer_level = np.ceil(q_level * (n_cal + 1)) / n_cal
        q_hat = conformal_quantile(v_norm, alpha)

        # Step 3 — local quantiles for test points (subclass hook)
        q_test = self._local_quantiles_test(X_test, q_level, **kwargs)

        # Step 4 — thresholds → prediction intervals
        thresholds = self._np(q_test) + q_hat
        return scores_to_intervals(
            self.base_model, X_test, thresholds,
            self.score, self.score_alpha,
            threshold_diagnostics=self.__dict__.setdefault("threshold_diagnostics", {}))

    # ------------------------------------------------------------------
    # Shared NW helper (used by SLCP and PDSLCP)
    # ------------------------------------------------------------------

    def _nw_quantile(self, W, q_level):
        """Normalize *W* and compute NW quantile from training scores."""
        W_norm = self._normalize_rows(W)
        return self._weighted_quantile_rows(self._v_train, W_norm, q_level)

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit_impl(self, v_train: np.ndarray):
        """Subclass-specific fitting: bandwidth, kernel weights, precomputation.

        Called after calibration scores ``_v_cal`` have been stored.

        Parameters
        ----------
        v_train : (n_train,) nonconformity scores on training set
        """

    @abstractmethod
    def _local_quantiles_cal(
        self, q_level: float,
    ) -> np.ndarray:
        """Estimate local quantiles for calibration points.

        Parameters
        ----------
        q_level : float  (typically 1 - alpha)

        Returns
        -------
        q_cal : array of shape (n_cal,)
        """

    @abstractmethod
    def _local_quantiles_test(
        self, X_test: np.ndarray, q_level: float, **kwargs,
    ) -> np.ndarray:
        """Estimate local quantiles for test points.

        Parameters
        ----------
        X_test  : (m, d) test features
        q_level : float

        Returns
        -------
        q_test : array of shape (m,)
        """
