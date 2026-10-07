"""Unified abstract interface for all conformal prediction methods.

Every conformal method (SCP, CalLCP, CalPDLCP, RLCP, PCP, …) subclasses
``BaseConformalMethod`` and implements ``fit`` and ``predict``.  The
evaluation framework relies exclusively on this interface, so adding a new
method requires no changes to the runner, evaluator, or metrics.

The ``from_dataset`` classmethod constructs an instance from a
``BaseDataset``, automatically mapping calibration (and, for methods that
override it, training) splits.  This is what ``MethodConfig.create`` calls
internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from .cqr import CQR_SCORES

try:
    import torch
except ImportError:
    torch = None


class BaseConformalMethod(ABC):
    """Abstract base class for conformal prediction methods.

    Subclasses must implement:

    * ``fit()`` — prepare calibration data.
    * ``predict(X_test, alpha)`` — return ``(y_pred, lower, upper)``.

    Optional overrides:

    * ``from_dataset`` — default uses ``dataset.val`` as calibration data.
    * ``name`` — human-readable label; default is the class name.
    * ``params`` — dict of parameters recorded in evaluation results.
    """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(cls, base_model, dataset, **kwargs):
        """Construct from a ``BaseDataset``, using the val split as calibration.

        Override for methods needing additional data (e.g. training split).
        """
        return cls(
            base_model=base_model,
            X_cal=dataset.val.X,
            y_cal=dataset.val.y,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable method name (used as result-dict key)."""
        return self.__class__.__name__

    @property
    def _score_tag(self) -> str:
        """Return a short score suffix like ``'absolute'`` for use in names.

        Returns empty string when ``self.score`` is not set.
        """
        return getattr(self, "score", "")

    @property
    def params(self) -> dict:
        """Method-specific parameters recorded alongside results."""
        return {}

    @property
    def auto_params(self) -> frozenset[str]:
        """Keys in ``params`` whose values were chosen automatically.

        Override in subclasses when a hyper-parameter can be auto-selected
        (e.g. bandwidth).  The runner stores this in ``PredictionResult`` and
        ``ExperimentResult`` replaces those keys with ``\"auto\"`` so that
        trials with different fitted values are still grouped together.
        """
        return frozenset()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self) -> "BaseConformalMethod":
        """Prepare calibration data (score computation, weight matrices, etc.).

        This method is alpha-independent.  The miscoverage level ``alpha``
        is specified only at ``predict`` time, keeping the two concerns
        (calibration vs. coverage target) cleanly separated.

        For the *quantile* score in ``CalWCP``, quantile
        levels are set at construction via ``score_alpha``, separate from the
        coverage ``alpha`` in ``predict``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, X_test, alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(y_pred, lower, upper)`` prediction intervals.

        Parameters
        ----------
        X_test : array of shape (m, d)
        alpha : float
            Nominal miscoverage level; intervals target ≥ 1 − alpha coverage.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Backend support (numpy / torch)
    # ------------------------------------------------------------------

    def _init_backend(self, device=None, dtype=None):
        """Initialize the numpy/torch computation backend.

        Call from subclass ``__init__`` to enable GPU-accelerated operations
        via ``_t`` (to-tensor) and ``_np`` (to-numpy).

        When ``torch`` is installed and *device* is not explicitly
        ``"cpu"``, heavy array operations run on GPU when available.
        """
        self.use_torch = torch is not None
        if self.use_torch:
            self.dtype = dtype if dtype is not None else torch.float32
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
        else:
            self.device = None
            self.dtype = None

    def _t(self, x):
        """Convert *x* to the active backend tensor (torch or numpy)."""
        if getattr(self, 'use_torch', False):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=self.dtype)
            return torch.as_tensor(
                np.asarray(x), device=self.device, dtype=self.dtype)
        return np.asarray(x)

    @staticmethod
    def _np(x):
        """Convert *x* to a numpy array (no-op if already numpy)."""
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _sq_dists(self, A, B, float64=False):
        """Pairwise squared Euclidean distances between rows of A and B.

        Returns the backend-native type: ``torch.Tensor`` when ``use_torch``
        is True, ``np.ndarray`` otherwise.  Uses ``torch.cdist`` on GPU when
        available, falling back to a numpy broadcast.

        Parameters
        ----------
        float64 : bool
            When True, forces float64 precision regardless of the configured
            dtype.  Use this for bandwidth search: ``torch.cdist`` in float32
            uses a GEMM formula (|a|²+|b|²−2a·b) that causes catastrophic
            cancellation for nearly-identical rows, corrupting near-diagonal
            entries and making row sums zero.
        """
        if self.use_torch:
            if float64:
                A_t = torch.as_tensor(
                    np.asarray(A, dtype=np.float64),
                    dtype=torch.float64, device=self.device)
                B_t = torch.as_tensor(
                    np.asarray(B, dtype=np.float64),
                    dtype=torch.float64, device=self.device)
            else:
                A_t, B_t = self._t(A), self._t(B)
            return torch.cdist(A_t, B_t).pow(2)
        diff = np.asarray(A)[:, None, :] - np.asarray(B)[None, :, :]
        return np.sum(diff ** 2, axis=-1)

    def _normalize_rows(self, W):
        """Row-normalize weight matrix.  Zero-sum rows become uniform."""
        if self.use_torch:
            s = W.sum(dim=1, keepdim=True)
        else:
            s = W.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        return W / s

    def _weighted_quantile_rows(self, values, weight_matrix, q):
        """Vectorized weighted quantile for each row of *weight_matrix*.

        Parameters
        ----------
        values : (n,) ndarray or Tensor
        weight_matrix : (m, n) ndarray or Tensor  (rows sum to 1)
        q : float in (0, 1)

        Returns
        -------
        quantiles : (m,) ndarray or Tensor
        """
        if self.use_torch:
            sort_idx    = torch.argsort(values)
            sorted_vals = values[sort_idx]
            cum_w       = torch.cumsum(weight_matrix[:, sort_idx], dim=1)
            exceed      = cum_w >= q
            idx         = torch.argmax(exceed.long(), dim=1)
            idx         = torch.where(
                exceed.any(dim=1), idx,
                torch.full_like(idx, len(sorted_vals) - 1))
            return sorted_vals[idx]
        sort_idx    = np.argsort(values)
        sorted_vals = values[sort_idx]
        cum_w       = np.cumsum(weight_matrix[:, sort_idx], axis=1)
        exceed      = cum_w >= q
        idx         = np.argmax(exceed, axis=1)
        idx         = np.where(exceed.any(axis=1), idx, len(sorted_vals) - 1)
        return sorted_vals[idx]





# ---------------------------------------------------------------------------
# Shared score helpers
# ---------------------------------------------------------------------------

VALID_SCORES = ('absolute', 'variance_normalized', *CQR_SCORES, 'pit')
SIGNED_SCORES = ('absolute', 'variance_normalized')
_PIT_U_GRID = np.linspace(1e-3, 1.0 - 1e-3, 100)


def compute_signed_scores(base_model, X, y, score):
    """Compute signed nonconformity scores (can be negative).

    Used by SLCP-family methods for ``absolute`` and ``variance_normalized``
    scores to produce asymmetric prediction intervals.

    Parameters
    ----------
    base_model : fitted probabilistic model
    X : array of shape (n, d)
    y : array of shape (n,)
    score : one of :data:`SIGNED_SCORES`

    Returns
    -------
    scores : array of shape (n,)  — may be negative
    """
    y = np.asarray(y).reshape(-1)
    mu = np.asarray(base_model.predict(X)).reshape(-1)

    if score == 'absolute':
        return y - mu

    if score == 'variance_normalized':
        sigma = np.asarray(base_model.predict_std(X)).reshape(-1)
        return (y - mu) / np.clip(sigma, 1e-8, None)

    raise ValueError(f"score must be one of {SIGNED_SCORES}, got {score!r}")


def signed_scores_to_intervals(base_model, X_test, lower_thresholds,
                                upper_thresholds, score):
    """Convert signed-score thresholds to prediction intervals.

    Parameters
    ----------
    base_model : fitted probabilistic model
    X_test : array of shape (m, d)
    lower_thresholds : array of shape (m,)
        Thresholds in signed-score space for the lower bound.
    upper_thresholds : array of shape (m,)
        Thresholds in signed-score space for the upper bound.
    score : one of :data:`SIGNED_SCORES`

    Returns
    -------
    y_pred, lower, upper : arrays of shape (m,)
    """
    X_test = np.asarray(X_test)
    lower_thresholds = np.asarray(lower_thresholds, dtype=float).reshape(-1)
    upper_thresholds = np.asarray(upper_thresholds, dtype=float).reshape(-1)
    y_pred = np.asarray(base_model.predict(X_test)).reshape(-1)

    if score == 'absolute':
        return y_pred, y_pred + lower_thresholds, y_pred + upper_thresholds

    if score == 'variance_normalized':
        sigma = np.asarray(base_model.predict_std(X_test)).reshape(-1)
        return (y_pred,
                y_pred + lower_thresholds * sigma,
                y_pred + upper_thresholds * sigma)

    raise ValueError(f"score must be one of {SIGNED_SCORES}, got {score!r}")


def compute_scores(base_model, X, y, score, score_alpha=0.1):
    """Compute nonconformity scores for the given data.

    Parameters
    ----------
    base_model : fitted probabilistic model
    X : array of shape (n, d)
    y : array of shape (n,)
    score : one of :data:`VALID_SCORES`
    score_alpha : float
        Quantile level used only by the ``'quantile'`` score.

    Returns
    -------
    scores : array of shape (n,)
    """
    y = np.asarray(y).reshape(-1)

    if score == 'absolute':
        mu = np.asarray(base_model.predict(X)).reshape(-1)
        return np.abs(y - mu)

    if score == 'variance_normalized':
        mu = np.asarray(base_model.predict(X)).reshape(-1)
        sigma = np.asarray(base_model.predict_std(X)).reshape(-1)
        return np.abs(y - mu) / np.clip(sigma, 1e-8, None)

    if score in CQR_SCORES:
        return CQR_SCORES[score].compute(base_model, X, y, score_alpha)

    if score == 'pit':
        if hasattr(base_model, 'predict_cdf'):
            pit = np.asarray(base_model.predict_cdf(X, y)).reshape(-1)
        else:
            quant_matrix = np.stack([
                np.asarray(base_model.predict_quantile(X, u)).reshape(-1)
                for u in _PIT_U_GRID
            ])
            pit = np.array([
                np.interp(y[i], quant_matrix[:, i], _PIT_U_GRID)
                for i in range(len(y))
            ])
        return np.maximum(pit, 1 - pit)

    raise ValueError(f"score must be one of {VALID_SCORES}, got {score!r}")


def scores_to_intervals(base_model, X_test, thresholds, score,
                         score_alpha=0.1, y_pred=None, threshold_diagnostics=None):
    """Convert score thresholds to prediction intervals.

    For absolute, variance_normalized and nonnegative quantile scores,
    floor negative thresholds at zero. This conservatively enlarges an
    otherwise empty score sublevel set; it is an explicit nonempty-output
    convention, not exact inversion for negative thresholds. Signed-score
    conversion and PIT are separate and unchanged.
    Optional ``threshold_diagnostics`` is updated with pre-floor statistics.


    Parameters
    ----------
    base_model : fitted probabilistic model
    X_test : array of shape (m, d)
    thresholds : array of shape (m,) or scalar
    score : one of :data:`VALID_SCORES`
    score_alpha : float
        Quantile level used only by the ``'quantile'`` score.
    y_pred : array of shape (m,), optional
        Pre-computed point predictions.  When ``None``, ``base_model.predict``
        is called internally.

    Returns
    -------
    y_pred, lower, upper : arrays of shape (m,)
    """
    X_test = np.asarray(X_test)
    n_test = len(X_test)
    thresholds = np.broadcast_to(
        np.asarray(thresholds, dtype=float), (n_test,))

    if np.isnan(thresholds).any() or np.isneginf(thresholds).any():
        raise ValueError("NaN or negative-infinite score threshold")
    floor_zero = score in ('absolute', 'variance_normalized', 'quantile')
    negative = thresholds < 0
    if threshold_diagnostics is not None:
        threshold_diagnostics.clear()
        threshold_diagnostics.update(
            policy='nonnegative_threshold_v1' if floor_zero else 'unchanged',
            n_predictions=n_test,
            negative_threshold_count=int(negative.sum()),
            negative_threshold_fraction=float(negative.mean()) if n_test else 0.0,
            raw_threshold_min=float(thresholds.min()) if n_test else None,
            clipped_threshold_count=int(negative.sum()) if floor_zero else 0,
        )
    if floor_zero:
        thresholds = np.maximum(thresholds, 0.0)

    if y_pred is None:
        y_pred = np.asarray(base_model.predict(X_test)).reshape(-1)
    else:
        y_pred = np.asarray(y_pred).reshape(-1)

    if score == 'absolute':
        return y_pred, y_pred - thresholds, y_pred + thresholds

    if score == 'variance_normalized':
        sigma = np.asarray(base_model.predict_std(X_test)).reshape(-1)
        return y_pred, y_pred - thresholds * sigma, y_pred + thresholds * sigma

    if score in CQR_SCORES:
        lower, upper = CQR_SCORES[score].invert(base_model, X_test, thresholds, score_alpha)
        if threshold_diagnostics is not None:
            threshold_diagnostics['empty_set_count'] = int(np.sum(lower > upper))
        return y_pred, lower, upper

    if score == 'pit':
        quant_matrix = np.stack([
            np.asarray(base_model.predict_quantile(X_test, u)).reshape(-1)
            for u in _PIT_U_GRID
        ])
        lower = np.array([
            np.interp(1 - thresholds[k], _PIT_U_GRID, quant_matrix[:, k])
            for k in range(n_test)
        ])
        upper = np.array([
            np.interp(thresholds[k], _PIT_U_GRID, quant_matrix[:, k])
            for k in range(n_test)
        ])
        lower = np.where(thresholds >= 1, -np.inf, lower)
        upper = np.where(thresholds >= 1, np.inf, upper)
        return y_pred, lower, upper

    raise ValueError(f"score must be one of {VALID_SCORES}, got {score!r}")



def conformal_quantile(scores, alpha):
    """Exact ceil((n+1)(1-alpha))-th order statistic, with an infinity atom."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not 0 < alpha < 1 or len(values) == 0 or np.isnan(values).any():
        raise ValueError("Invalid calibration scores or alpha")
    k = int(np.ceil((len(values) + 1) * (1 - alpha)))
    return np.inf if k > len(values) else np.partition(values, k - 1)[k - 1]
