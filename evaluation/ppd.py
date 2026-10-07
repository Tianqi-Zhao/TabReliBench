"""Posterior predictive distribution helpers.

A ``PPDQuantileGrid`` wraps the ``(ppd_quantiles, quantile_levels)`` pair
that every prediction PKL stores. It centralises:

* monotonicity enforcement along the quantile axis
* linear interpolation to an arbitrary quantile level
* central-(1 - alpha) interval extraction
* integration over the full unit quantile range with constant tail extension

Before the refactor this logic was duplicated in ``compute_metrics.py``
(``quantile_at`` / ``interval_from_ppd``) and ``extract_instance_meta_features.py``
(per-row ``np.interp``). Both call sites now go through this class.
"""
from __future__ import annotations

import numpy as np


def make_quantile_grid(step: float) -> np.ndarray:
    """Build the canonical open-interval quantile grid for a step size.

    Prediction entry points must use this helper so foundation models and
    baselines cannot silently diverge through different rounding or validation
    rules.  Twelve decimal places preserve practical user-supplied steps while
    removing the usual floating-point noise from ``np.arange``.
    """
    step = float(step)
    if not np.isfinite(step) or not 0.0 < step < 0.5:
        raise ValueError("quantile step must lie in (0, 0.5)")
    grid = np.round(np.arange(step, 1.0, step), decimals=12)
    grid = np.unique(grid[(grid > 0.0) & (grid < 1.0)])
    if grid.size < 2:
        raise ValueError("quantile step produces fewer than two quantile levels")
    return grid.astype(np.float64, copy=False)


def quantile_grid_integral(
    values: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Integrate values sampled on a quantile grid over ``[0, 1]``.

    The benchmark grid normally excludes the endpoints. Values outside the
    supplied levels are therefore extended as constants at the first and last
    grid values, matching the baseline point-mean calculation. Integration
    uses ``np.trapezoid`` when available and falls back to ``np.trapz`` on
    NumPy versions before 2.0.

    The final axis of ``values`` must correspond to ``levels``. This allows the
    same helper to integrate the quantile function itself, its square, or
    standardized higher powers used by instance-level PPD summaries.
    """
    arr = np.asarray(values, dtype=np.float64)
    q = np.asarray(levels, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("levels must be a non-empty 1D array")
    if arr.ndim == 0 or arr.shape[-1] != q.size:
        raise ValueError(
            f"values shape {arr.shape} is incompatible with {q.size} levels"
        )
    if not np.all(np.isfinite(q)):
        raise ValueError("levels must be finite")
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("levels must lie in [0, 1]")
    if not np.all(np.diff(q) > 0.0):
        raise ValueError("levels must be strictly increasing")

    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(arr, q, axis=-1)
    else:  # pragma: no cover - exercised only on NumPy < 2
        integral = np.trapz(arr, q, axis=-1)
    return integral + q[0] * arr[..., 0] + (1.0 - q[-1]) * arr[..., -1]


class PPDQuantileGrid:
    """Per-test-row posterior predictive distribution stored on a fixed grid."""

    def __init__(
        self,
        ppd: np.ndarray,
        levels: np.ndarray,
        *,
        enforce_monotone: bool = True,
    ) -> None:
        ppd = np.asarray(ppd, dtype=np.float64)
        levels = np.asarray(levels, dtype=np.float64)
        if ppd.ndim != 2:
            raise ValueError(
                f"ppd must be 2D (n_test, n_levels); got ndim={ppd.ndim}")
        if ppd.shape[1] != len(levels):
            raise ValueError(
                f"ppd has {ppd.shape[1]} columns but levels has {len(levels)}")
        if not np.all(np.diff(levels) > 0):
            raise ValueError("quantile levels must be strictly increasing")
        if enforce_monotone:
            ppd = np.maximum.accumulate(ppd, axis=1)
        self.ppd = ppd
        self.levels = levels

    @classmethod
    def from_record(cls, record: dict, *, enforce_monotone: bool = True) -> "PPDQuantileGrid":
        return cls(
            record["ppd_quantiles"], record["quantile_levels"],
            enforce_monotone=enforce_monotone,
        )

    @property
    def n_test(self) -> int:
        return self.ppd.shape[0]

    @property
    def n_levels(self) -> int:
        return self.ppd.shape[1]

    def quantile_at(self, q: float) -> np.ndarray:
        """Return per-row q-quantile by linear interpolation on the grid."""
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}.")
        levels = self.levels
        ppd = self.ppd
        K = self.n_levels

        idx = int(np.searchsorted(levels, q))
        if idx < K and float(levels[idx]) == float(q):
            return ppd[:, idx].astype(np.float64)
        if idx == 0:
            return ppd[:, 0].astype(np.float64)
        if idx == K:
            return ppd[:, -1].astype(np.float64)
        q_lo = float(levels[idx - 1])
        q_hi = float(levels[idx])
        w = (q - q_lo) / (q_hi - q_lo)
        return ((1.0 - w) * ppd[:, idx - 1] + w * ppd[:, idx]).astype(np.float64)

    def interval(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` for the central (1 - alpha) interval."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        lower = self.quantile_at(alpha / 2.0)
        upper = self.quantile_at(1.0 - alpha / 2.0)
        return lower, upper

    def cdf_at(self, y: np.ndarray) -> np.ndarray:
        """Return per-row PIT values F_i(y_i) via linear interpolation.

        For each test point *i* the empirical CDF is approximated by
        linearly interpolating the quantile grid:

            F_i(y_i) = q  such that  F_i^{-1}(q) = y_i

        Implementation notes
        --------------------
        * Monotonicity is already guaranteed by the ``np.maximum.accumulate``
          applied at construction time.
        * ``np.interp`` uses left-extrapolation below the first grid point
          and right-extrapolation above the last one.  We therefore clip the
          output to ``(eps, 1 - eps)`` so that downstream ``log`` calls
          never see 0 or 1.
        * On plateau regions (consecutive identical quantile values) the
          standard ``np.interp`` returns the left edge of the plateau.
          We correct this to the **midpoint** of the corresponding level
          interval using a vectorised searchsorted approach.
        """
        y = np.asarray(y, dtype=np.float64)
        eps = np.finfo(np.float64).eps

        n_test = self.n_test
        pit = np.empty(n_test, dtype=np.float64)

        ppd    = self.ppd      # (n_test, n_levels)
        levels = self.levels   # (n_levels,)

        for i in range(n_test):
            row = ppd[i]
            yi  = y[i]

            # Standard linear interpolation: row is x-values, levels is y-values
            q_raw = float(np.interp(yi, row, levels))

            # Plateau correction: find the range of indices where row == row[j]
            # for the matched position, then average the corresponding levels.
            j_lo = int(np.searchsorted(row, yi, side="left"))
            j_hi = int(np.searchsorted(row, yi, side="right"))

            if j_hi > j_lo:
                # yi falls exactly on a plateau [row[j_lo] .. row[j_hi-1]]
                lev_lo = levels[j_lo]     if j_lo  < len(levels) else levels[-1]
                lev_hi = levels[j_hi - 1] if j_hi - 1 < len(levels) else levels[-1]
                q_raw  = float(0.5 * (lev_lo + lev_hi))

            pit[i] = q_raw

        return np.clip(pit, eps, 1.0 - eps)
