"""Weighted conformal prediction base class.

Subclasses implement ``_compute_weights(X_query)`` returning an **unnormalized**
kernel weight matrix H of shape ``(len(X_query), n_cal)``, where
``H[i, j] = H(X_query_i, X_cal_j)``.  The parent class handles score
computation, exact normalization, the efficient calLCP algorithm (Section 3.3),
and prediction-interval construction.

Algorithm summary (Section 3.3 — exact normalization)
------------------------------------------------------
Given calibration scores s_i = s(x_i, y_i) and a test point x_{n+1}, define
unnormalized quantities (computed offline / once per test point):

    A_i  = sum_{j : s_j < s_i} H(X_j, X_i)          (precomputed at fit time)
    S_i  = sum_{m=1}^n H(X_m, X_i) + H(X_{n+1}, X_i) (per test point)
    S_{n+1} = sum_{m=1}^n H(X_m, X_{n+1}) + H(X_{n+1}, X_{n+1})

The normalized transformed scores for g in (s_{(k)}, s_{(k+1)}) are:

    T^g_{n+1} = B_k / S_{n+1},  where B_k = sum_{j : s_j < s_{(k)}} H(X_j, X_{n+1})
    T^g_i     = A_i / S_i                  if s_i < g
              = (A_i + H(X_{n+1}, X_i)) / S_i   if s_i > g

We find the largest real g such that
    T^g_{n+1} <= Q_{1-alpha}(T^g_1, ..., T^g_n, T^g_{n+1})

by passing normalized quantities to ``solve_largest_x``:
    T_cal_base[k, j] = A_j / S_j^{(k)},
    T_cal_jump[k, j] = H(X_{n+1,k}, X_j) / S_j^{(k)},
    T_test_jump[k, j] = H(X_{n+1,k}, X_j) / S_{n+1}^{(k)}.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from conformal.methods.base import (BaseConformalMethod, VALID_SCORES,
                           compute_scores, scores_to_intervals)
from conformal.methods.utils import solve_largest_x




class CalWCP(BaseConformalMethod):
    """Template base class for weighted conformal regression.

    Parameters
    ----------
    base_model :
        Fitted probabilistic regression model.  The methods it must expose
        depend on the chosen ``score``; see ``fit`` docstring.
    X_cal : array of shape (n, d)
        Calibration features.
    y_cal : array of shape (n,)
        Calibration labels.
    score : {'absolute', 'variance_normalized', 'quantile', 'pit'}
        Nonconformity score type.
    """

    def __init__(self, base_model, X_cal=None, y_cal=None,
                 score='absolute', score_alpha=0.1,
                 X_cal_dist=None,
                 device=None, dtype=None):
        self.base_model = base_model
        self.X_cal = None if X_cal is None else np.asarray(X_cal)
        self.y_cal = None if y_cal is None else np.asarray(y_cal).reshape(-1)
        self.X_cal_dist = None if X_cal_dist is None else np.asarray(X_cal_dist)
        if score not in VALID_SCORES:
            raise ValueError(
                f"score must be one of {VALID_SCORES}, got {score!r}"
            )
        self.score = score
        self.score_alpha = score_alpha  # quantile levels for 'quantile' score
        self.s_sorted = None   # calibration scores, sorted ascending, shape (n,)
        self.A = None          # A_i values, shape (n,)
        self._init_backend(device, dtype)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self):
        """Compute calibration scores and precompute weight structure.

        The nonconformity score type is determined by ``self.score`` (set at
        construction time).  For the ``'quantile'`` score, quantile levels
        are set by ``self.score_alpha`` — a score-function hyperparameter
        independent of the coverage-level alpha passed to ``predict``.

        Returns
        -------
        self
        """
        cal_scores = compute_scores(
            self.base_model, self.X_cal, self.y_cal,
            self.score, self.score_alpha)

        # Sort calibration set by score so that _compute_weights consistently
        # uses the sorted order from this point on.
        order = np.argsort(cal_scores)
        self.X_cal = self.X_cal[order]
        if self.X_cal_dist is not None:
            self.X_cal_dist = self.X_cal_dist[order]
        self.s_sorted = cal_scores[order]

        # Unnormalized calibration weight matrix: W_cal[i, j] = H(X_i, X_j)
        # _compute_weights is private and returns tensor when use_torch=True.
        # Use one-hot features (X_cal_dist) for distance if available.
        X_cal_w = self.X_cal_dist if self.X_cal_dist is not None else self.X_cal
        w_cal = self._compute_weights(X_cal_w, X_ref=X_cal_w)  # (n, n)

        # A_i = sum_{j : s_j < s_i} H(X_j, X_i)
        # Symmetric kernels let us sum row i, but sorted position j < i
        # does not imply s_j < s_i: nonnegative CQR often has many zero ties.
        # Compare the original score precision before transferring the mask;
        # casting scores to the weight dtype could create additional ties.
        strict_lower = self.s_sorted[None, :] < self.s_sorted[:, None]
        if self.use_torch:
            mask = torch.as_tensor(strict_lower, device=w_cal.device)
            self.A = torch.where(mask, w_cal, 0).sum(dim=1)
        else:
            self.A = np.where(strict_lower, w_cal, 0).sum(axis=1)

        # Row sums of cal-cal weight matrix.  When a test point X_{n+1} arrives,
        # the normalization constant for calibration score i is
        #   S_i = sum_{m=1}^n H(X_m, X_i) + H(X_{n+1}, X_i)
        #       = _row_sum_cal[i]          + w_test[k, i]
        self._row_sum_cal = w_cal.sum(1)

        return self

    # ------------------------------------------------------------------
    # Core conformal algorithm  (Section 3.3)
    # ------------------------------------------------------------------

    def score_thresholds(self, X_test, alpha, X_test_dist=None):
        """Compute the conformal score threshold for each test point.

        For each test point k, finds the largest real threshold t such that

            T^t_{n+1} <= Q_{1-alpha}(T^t_1, ..., T^t_n, T^t_{n+1})

        using ``solve_largest_x`` with the exactly normalized identification

            a[k, j] = A_j / S_j^{(k)},
            B[k, j] = H(X_{n+1,k}, X_j) / S_j^{(k)},
            C[k, j] = H(X_{n+1,k}, X_j) / S_{n+1}^{(k)}.

        Parameters
        ----------
        X_test : array of shape (m, d)
        alpha : float
        X_test_dist : array of shape (m, d') or None
            One-hot features for distance computation.  When ``None``,
            ``X_test`` is used.

        Returns
        -------
        thresholds : array of shape (m,)
            Largest feasible score threshold per test point.
            ``np.inf``  — condition holds for all t (unbounded interval).
            ``np.nan``  — condition never holds (empty prediction set).
        """
        # _compute_weights / _compute_self_weights are private and return
        # tensor when use_torch=True, so all (m, n) arithmetic below runs on
        # the same backend.  Convert to numpy only at the boundary with the
        # sequential heap algorithm (solve_largest_x).
        X_test_w = X_test_dist if X_test_dist is not None else X_test
        X_cal_w = self.X_cal_dist if self.X_cal_dist is not None else self.X_cal
        w_test = self._compute_weights(X_test_w, X_ref=X_cal_w)  # (m, n)

        # --- Exact normalization (Section 3.3) ---
        # S_i^{(k)} = sum_{m=1}^n H(X_m, X_i) + H(X_{n+1,k}, X_i)
        S_cal = self._row_sum_cal[None, :] + w_test              # (m, n)

        # S_{n+1}^{(k)} = sum_{m=1}^n H(X_m, X_{n+1,k}) + H(X_{n+1,k}, X_{n+1,k})
        self_w = self._compute_self_weights(X_test_w)             # (m,)
        S_test = w_test.sum(1) + self_w                          # (m,)

        # Normalized quantities passed to the heap algorithm:
        # T_cal_base[k, j] = A_j / S_j^{(k)}                    (T^g_j when s_j < g)
        # T_cal_jump[k, j] = H(X_{n+1,k}, X_j) / S_j^{(k)}     (jump when g crosses s_j)
        # T_test_jump[k,j] = H(X_{n+1,k}, X_j) / S_{n+1}^{(k)} (increment in T^g_{n+1})
        T_cal_base  = self.A[None, :] / S_cal                   # (m, n)
        T_cal_jump  = w_test / S_cal                             # (m, n)
        T_test_jump = w_test / S_test[:, None]                   # (m, n)

        return solve_largest_x(
            self._np(T_cal_base), self._np(T_cal_jump),
            self.s_sorted,
            self._np(T_test_jump), alpha=alpha)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X_test, alpha=0.05, X_test_dist=None):
        """Return point predictions and conformal prediction intervals.

        Parameters
        ----------
        X_test : array of shape (m, d)
            Test features (ordinal encoding for TabPFN).
        alpha : float
            Nominal miscoverage level; intervals achieve >= 1-alpha coverage.
        X_test_dist : array of shape (m, d') or None
            One-hot features for distance-based weight computation.
            When ``None``, ``X_test`` is used for both distances and scores.

        Returns
        -------
        y_pred : array of shape (m,)
            Point predictions.
        lower : array of shape (m,)
            Lower bounds of the prediction intervals.
        upper : array of shape (m,)
            Upper bounds of the prediction intervals.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if self.s_sorted is None:
            self.fit()

        X_test = np.asarray(X_test)
        thresholds = self.score_thresholds(X_test, alpha,
                                           X_test_dist=X_test_dist)

        return scores_to_intervals(
            self.base_model, X_test, thresholds,
            self.score, self.score_alpha,
            threshold_diagnostics=self.__dict__.setdefault("threshold_diagnostics", {}))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_weights(self, X_query, X_ref=None):
        """Return a ``(len(X_query), len(X_ref))`` non-negative **unnormalized** weight matrix.

        ``W[i, j] = H(X_query_i, X_ref_j)`` is the raw kernel value between
        query point i and reference point j.

        This is a private method.  Implementations should return the
        backend-native type — ``torch.Tensor`` when ``use_torch`` is True,
        ``np.ndarray`` otherwise — so that the caller (``fit``,
        ``score_thresholds``) can chain further operations on the same device
        without unnecessary transfers.
        """
        raise NotImplementedError

    def _compute_self_weights(self, X_query):
        """Return H(x, x) for each point in X_query, shape ``(len(X_query),)``.

        Returns the backend-native type matching ``_compute_weights``.

        The default implementation computes the full ``(p, p)`` pairwise matrix
        via ``_compute_weights(X_query, X_ref=X_query)`` and extracts the
        diagonal — correct for any kernel but O(p²) in memory and time.

        Subclasses whose kernel satisfies H(x, x) = k(0) for a known constant
        ``k(0)`` should override this method with an O(p) implementation.
        """
        w = self._compute_weights(X_query, X_ref=X_query)
        if self.use_torch and torch is not None:
            return torch.diag(w)
        return np.diag(w)
