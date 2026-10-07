"""Split conformal regression with exact finite-sample calibration."""

import numpy as np
from conformal.methods.base import conformal_quantile

from conformal.methods.base import (BaseConformalMethod, VALID_SCORES,
                          compute_scores, scores_to_intervals)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class ConformalPredictor(BaseConformalMethod):
    """Split/inductive conformal predictor for regression."""

    def __init__(self,
                 base_model,
                 X_cal=None,
                 y_cal=None,
                 score='absolute',
                 score_alpha=0.1,
                 device=None,
                 dtype=None):
        """
        Args:
            base_model: Any regressor with ``predict(X)``.
            X_cal: Calibration features.
            y_cal: Calibration targets.
            score: Nonconformity score type (see :data:`VALID_SCORES`).
            score_alpha: Quantile level for the ``'quantile'`` score.
            device: Torch device string (``"cuda"``, ``"cpu"``).
            dtype: Torch dtype used for tensor computations.
        """
        if score not in VALID_SCORES:
            raise ValueError(
                f"score must be one of {VALID_SCORES}, got {score!r}")
        self.base_model = base_model
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.score = score
        self.score_alpha = score_alpha
        self._init_backend(device, dtype)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"SCP({self._score_tag})"

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self):
        """Compute calibration nonconformity scores."""
        cal_scores = compute_scores(
            self.base_model, self.X_cal, self.y_cal,
            self.score, self.score_alpha)
        self._calibration_scores = self._t(cal_scores)
        return self

    @property
    def calibration_scores(self):
        """Calibration scores as a numpy array (public API)."""
        return self._np(self._calibration_scores)

    def predict(self, X_test, alpha=0.05):
        """Predict point values and conformal prediction intervals.

        Parameters
        ----------
        X_test : array-like
            Test features.
        alpha : float
            Nominal miscoverage level (intervals target ≥ 1 − alpha coverage).

        Returns
        -------
        ``(y_pred, lower, upper)`` — NumPy arrays of shape ``(n_test,)``.
        """
        if not hasattr(self, "_calibration_scores"):
            self.fit()

        n_cal = self._calibration_scores.shape[0]

        q_level = np.ceil((1.0 - alpha) * (n_cal + 1)) / n_cal
        q_hat = conformal_quantile(self._np(self._calibration_scores), alpha)

        return scores_to_intervals(
            self.base_model, X_test, q_hat,
            self.score, self.score_alpha,
            threshold_diagnostics=self.__dict__.setdefault("threshold_diagnostics", {}))

    def coverage_score(self, X_test, y_test, alpha=0.05):
        """Compute empirical coverage and average interval width."""
        _, lower, upper = self.predict(X_test, alpha)
        if self.use_torch:
            y_t = self._t(y_test).reshape(-1)
            lower_t = self._t(lower)
            upper_t = self._t(upper)
            in_interval = (y_t >= lower_t) & (y_t <= upper_t)
            coverage = torch.mean(in_interval.to(self.dtype))
            avg_width = torch.mean(upper_t - lower_t)
            return float(coverage.item()), float(avg_width.item())
        y_np = np.asarray(y_test).reshape(-1)
        in_interval = (y_np >= lower) & (y_np <= upper)
        return float(np.mean(in_interval)), float(np.mean(upper - lower))
