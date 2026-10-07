"""Rectified Conformal Prediction (RCP).

RCP transforms any base nonconformity score via a parametric family of
adjustment functions to improve conditional coverage while preserving
exact marginal coverage guarantees.

The score quantile estimator (QE) is fitted on the designated auxiliary
training split.  The full, disjoint calibration split is then used for the
conformal correction, matching the allocation used by the other learned
split-conformal methods in this benchmark.

Algorithm overview
------------------
Given calibration data (X_cal, y_cal) and a base regression model:

1. **fit()** — preparation phase
   a. Compute base nonconformity scores on the auxiliary training split.
   b. Fit a TabPFN model on (X_QE, scores_QE) to learn the conditional
      distribution of nonconformity scores given features.
   c. Compute nonconformity scores on the full calibration split.

2. **predict(alpha)** — inference phase
   a. Estimate conditional (1-alpha)-quantile τ̂(x) at each conformal-split
      point and at each test point via ``qe_tabpfn.predict_quantile``.
   b. Rectify calibration scores: Ṽ_k = f_{τ̂(X_k)}^{-1}(V_k)
      - 'difference': Ṽ_k = V_k − τ̂(X_k)
      - 'linear':     Ṽ_k = V_k / τ̂(X_k)
   c. Find standard conformal threshold q̂ = ceil((1-alpha)(n+1))-th smallest
      value of {Ṽ_1, …, Ṽ_n, +∞}.
   d. Compute per-test adjusted threshold: f_{τ̂(x_test)}(q̂)
      - 'difference': q̂ + τ̂(x_test)
      - 'linear':     q̂ * τ̂(x_test)
   e. Build prediction interval from adjusted threshold.

Reference
---------
"Rectifying Conformity Scores for Better Conditional Coverage"
Feldman, Bates, Tibshirani. ICML 2025. arXiv:2502.16336.
"""

from __future__ import annotations

import numpy as np
from conformal.methods.base import conformal_quantile

from conformal.methods.base import (BaseConformalMethod, VALID_SCORES,
                           compute_scores, scores_to_intervals)

_VALID_ADJUSTMENTS = ('difference', 'linear')


class RCP(BaseConformalMethod):
    """Rectified Conformal Prediction for regression intervals.

    Parameters
    ----------
    base_model :
        Fitted probabilistic regression model.  Required methods depend on
        ``score``; same contract as ``CalWCP``.
    X_cal : array of shape (n, d)
        Calibration features.
    y_cal : array of shape (n,)
        Calibration labels.
    X_train : array of shape (m, d)
        Auxiliary training features used to fit the score QE.
    y_train : array of shape (m,)
        Auxiliary training labels used to construct QE targets.
    score : {'absolute', 'variance_normalized', 'quantile', 'pit'}
        Base nonconformity score type.
    score_alpha : float
        Quantile level for the 'quantile' score (used as inner quantile level).
    adjustment : {'difference', 'linear'}
        Parametric adjustment function family.
        - 'difference': f_τ(v) = v + τ,  inverse f_τ^{-1}(v) = v − τ.
        - 'linear':     f_τ(v) = v * τ,  inverse f_τ^{-1}(v) = v / τ.
    random_state : int
        Seed forwarded to the auxiliary score model.
    """

    def __init__(
        self,
        base_model,
        X_cal,
        y_cal,
        X_train=None,
        y_train=None,
        score: str = 'absolute',
        score_alpha: float = 0.1,
        adjustment: str = 'difference',
        random_state: int = 0,
        device=None,
        dtype=None,
    ):
        if score not in VALID_SCORES:
            raise ValueError(f"score must be one of {VALID_SCORES}, got {score!r}")
        if adjustment not in _VALID_ADJUSTMENTS:
            raise ValueError(
                f"adjustment must be one of {_VALID_ADJUSTMENTS}, got {adjustment!r}"
            )
        if X_train is None or y_train is None:
            raise ValueError(
                "X_train and y_train are required to fit the RCP score model."
            )
        self.base_model = base_model
        self.X_cal = np.asarray(X_cal)
        self.y_cal = np.asarray(y_cal).reshape(-1)
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).reshape(-1)
        self.score = score
        self.score_alpha = score_alpha
        self.adjustment = adjustment
        self.random_state = random_state
        self._init_backend(device, dtype)

        # populated by fit():
        # ``_qe_model`` is a fresh wrapper of the same family as ``base_model``
        # (TabPFN / TabICL / ...), produced via ``base_model.clone_for(...)``.
        self._qe_model = None
        self._conf_X: np.ndarray | None = None
        self._conf_scores: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(cls, base_model, dataset, **kwargs):
        """Create an RCP instance from a dataset object.

        ``X_train`` / ``y_train`` are taken from ``dataset.train`` and used
        to fit the auxiliary score model.
        """
        return cls(
            base_model=base_model,
            X_cal=dataset.val.X,
            y_cal=dataset.val.y,
            X_train=dataset.train.X,
            y_train=dataset.train.y,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"RCP(adj={self.adjustment}, {self._score_tag})"

    @property
    def params(self) -> dict:
        d = {
            "score": self.score,
            "adjustment": self.adjustment,
        }
        if self.score in ('quantile', 'quantile_standard'):
            d["score_alpha"] = self.score_alpha
        return d

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> "RCP":
        """Fit the QE TabPFN and prepare the conformal calibration set.

        Fits the QE on auxiliary training scores and uses the full,
        disjoint calibration set for the conformal correction.

        Returns
        -------
        self
        """
        train_scores = compute_scores(
            self.base_model, self.X_train, self.y_train,
            self.score, self.score_alpha)
        self._conf_X = self.X_cal
        self._conf_scores = compute_scores(
            self.base_model, self.X_cal, self.y_cal,
            self.score, self.score_alpha)

        qe_fit_X, qe_fit_y = self.X_train, train_scores

        # Clone the same probabilistic family as base_model (TabPFN /
        # TabICL / ...) and refit it on the score targets.  This keeps
        # RCP wrapper-agnostic — no hardcoded TabPFN dependency here.
        self._qe_model = self.base_model.clone_for(
            qe_fit_X, qe_fit_y, random_state=self.random_state,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, X_test, alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return point predictions and RCP prediction intervals.

        Parameters
        ----------
        X_test : array of shape (m, d)
        alpha : float
            Nominal miscoverage level; intervals achieve >= 1-alpha coverage.

        Returns
        -------
        y_pred : array of shape (m,)
        lower : array of shape (m,)
        upper : array of shape (m,)
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if self._conf_scores is None:
            self.fit()

        X_test = np.asarray(X_test)

        tau_conf = self._qe_model.predict_quantile(self._conf_X, 1.0 - alpha)
        tau_test = self._qe_model.predict_quantile(X_test, 1.0 - alpha)

        rectified = self._rectify(self._conf_scores, tau_conf)
        n_conf = len(rectified)

        q_level = np.ceil((1.0 - alpha) * (n_conf + 1)) / n_conf
        q_hat = conformal_quantile(rectified, alpha)

        thresholds = self._forward(tau_test, q_hat)

        return scores_to_intervals(
            self.base_model, X_test, thresholds,
            self.score, self.score_alpha,
            threshold_diagnostics=self.__dict__.setdefault("threshold_diagnostics", {}))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rectify(self, scores: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Apply f_τ^{-1} to each score: Ṽ = f_{τ(x)}^{-1}(V).

        - 'difference': Ṽ = V − τ
        - 'linear':     Ṽ = V / τ  (τ clipped to avoid division by zero)
        """
        if self.adjustment == 'difference':
            return scores - tau
        else:  # linear
            return scores / np.clip(tau, 1e-8, None)

    def _forward(self, tau: np.ndarray, q_hat: float) -> np.ndarray:
        """Apply f_τ to the rectified threshold: adjusted = f_{τ(x)}(q̂).

        - 'difference': adjusted = q̂ + τ
        - 'linear':     adjusted = q̂ * τ
        """
        if self.adjustment == 'difference':
            return q_hat + tau
        else:  # linear
            return q_hat * tau
