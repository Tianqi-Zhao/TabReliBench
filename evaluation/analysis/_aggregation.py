"""Cross-seed aggregators for two-stage (per-seed → combine) analyses.

Non-LME dataset-level analyzers (Chatterjee, Spearman, RF, etc.) fit one
model per seed and then aggregate. Each ``*Aggregator`` class encapsulates
a different aggregation strategy so analyzers can swap them via ``__init__``.

All aggregators implement the same ``combine(per_seed: list[dict]) -> dict``
interface. The returned dict always carries::

    mean, se, ci_low, ci_high, p, p_acat, t, df, n_seeds, within_var,
    between_var

with ``NaN`` for fields the strategy can't compute (e.g. ``MeanSDAggregator``
has no analytical p-value; ``FisherZAggregator`` leaves ``se`` as ``NaN``
because the Rubin SE lives in z-space while ``mean`` / CI are on r).

``p_acat`` is the Cauchy combination test (ACAT) p-value computed from the
per-seed p-values. It tests the intersection null "every seed's effect is
zero" and is robust to dependence between seeds. Present whenever per-seed
dicts carry a ``"p"`` key; ``NaN`` otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import stats as sp_stats


_NAN_RESULT: dict = {
    "mean":         float("nan"),
    "se":           float("nan"),
    "ci_low":       float("nan"),
    "ci_high":      float("nan"),
    "p":            float("nan"),
    "p_acat":       float("nan"),
    "t":            float("nan"),
    "df":           float("nan"),
    "n_seeds":      0,
    "within_var":   float("nan"),
    "between_var":  float("nan"),
}


def _empty_result() -> dict:
    return dict(_NAN_RESULT)


def _collect_valid_ps(per_seed: list[dict]) -> list[float]:
    """Return finite per-seed p-values, clipped away from 0 / 1."""
    ps: list[float] = []
    for r in per_seed:
        p = r.get("p")
        if p is None:
            continue
        p_f = float(p)
        if not np.isfinite(p_f):
            continue
        ps.append(max(min(p_f, 1.0 - 1e-15), 1e-15))
    return ps


def _acat_p(per_seed: list[dict]) -> tuple[float, float]:
    """Combine per-seed p-values via the Cauchy combination test (ACAT).

    Returns ``(p_combined, T)`` where ``T = mean_k tan((0.5 - p_k) * π)`` and
    ``p_combined = 0.5 - arctan(T) / π``. Both are ``NaN`` when no valid
    ``"p"`` keys are present in *per_seed*.
    """
    ps = _collect_valid_ps(per_seed)
    if not ps:
        return float("nan"), float("nan")
    p_arr = np.asarray(ps, dtype=float)
    T = float(np.mean(np.tan((0.5 - p_arr) * np.pi)))
    p_combined = float(0.5 - np.arctan(T) / np.pi)
    return p_combined, T


class SeedAggregator(Protocol):
    """Combine per-seed estimates into a single aggregated row.

    ``per_seed`` is a list of dicts. Each dict's required keys depend on
    the strategy (e.g. ``RubinAggregator`` needs ``estimate`` + ``se``;
    ``FisherZAggregator`` needs ``r`` + ``n``). Extra keys are ignored.
    """

    def combine(self, per_seed: list[dict]) -> dict: ...


# ─────────────────────────────────────────────────────────────────────────────
# Rubin's rules — pooled estimate + se accounting for within + between var
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RubinAggregator:
    """Rubin's rules combine.

    Requires ``per_seed`` entries with ``estimate`` and ``se`` (analytical
    standard error of that seed's fit). Suitable for analyzers whose
    per-fit estimator produces an asymptotic SE (e.g. Chatterjee ξ).

    Returns ``mean``, ``se = sqrt(T)`` where
    ``T = W + (1 + 1/K) * B``, ``df`` per Barnard–Rubin, and a two-sided
    t-test ``p`` value. ``within_var = W``, ``between_var = B``.
    """

    ci_level: float = 0.95

    def combine(self, per_seed: list[dict]) -> dict:
        estimates: list[float] = []
        ses: list[float] = []
        for r in per_seed:
            est = r.get("estimate")
            se = r.get("se")
            if est is None or se is None:
                continue
            est_f = float(est)
            se_f = float(se)
            if not (np.isfinite(est_f) and np.isfinite(se_f) and se_f > 0):
                continue
            estimates.append(est_f)
            ses.append(se_f)

        K = len(estimates)
        if K == 0:
            return _empty_result()

        est_arr = np.asarray(estimates, dtype=float)
        se_arr = np.asarray(ses, dtype=float)

        mean = float(est_arr.mean())
        W = float(np.mean(se_arr ** 2))
        B = float(est_arr.var(ddof=1)) if K > 1 else 0.0
        T = W + (1.0 + 1.0 / K) * B
        se = float(np.sqrt(T))

        # Barnard–Rubin degrees of freedom; with B=0 (single seed or
        # perfectly stable estimates) df → ∞ → use normal.
        if K > 1 and B > 0:
            df = (K - 1) * (1.0 + W / ((1.0 + 1.0 / K) * B)) ** 2
        else:
            df = float("inf")

        if se > 0:
            t = mean / se
            if np.isfinite(df):
                p = float(2.0 * sp_stats.t.sf(abs(t), df=df))
                t_crit = float(sp_stats.t.ppf(0.5 + self.ci_level / 2.0, df=df))
            else:
                p = float(2.0 * sp_stats.norm.sf(abs(t)))
                t_crit = float(sp_stats.norm.ppf(0.5 + self.ci_level / 2.0))
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
        else:
            t = float("nan")
            p = float("nan")
            ci_low = mean
            ci_high = mean

        return {
            "mean":        mean,
            "se":          se,
            "ci_low":      float(ci_low),
            "ci_high":     float(ci_high),
            "p":           float(p) if np.isfinite(p) else float("nan"),
            "p_acat":      _acat_p(per_seed)[0],
            "t":           float(t) if np.isfinite(t) else float("nan"),
            "df":          float(df) if np.isfinite(df) else float("inf"),
            "n_seeds":     K,
            "within_var":  W,
            "between_var": B,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fisher-z combine — Spearman / Pearson r aggregation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FisherZAggregator:
    """Combine Spearman/Pearson r across seeds via Fisher's z-transform.

    Requires ``per_seed`` entries with ``r`` and ``n`` (sample size used
    for that seed's correlation). Each seed's ``z_k = atanh(r_k)`` has
    analytical SE ``1/sqrt(n_k - 3)``; we apply Rubin in z-space and
    back-transform to r-space at the end.

    Output ``mean`` is the back-transformed r̄. ``ci_low/high`` are
    back-transformed from the z-space CI (so always lie in (-1, 1)).
    p-value and t are taken from the z-space t-test directly. ``se`` is
    left as ``NaN`` (Rubin SE is on z; use ``ci_low``/``ci_high`` on r).
    """

    ci_level: float = 0.95

    def combine(self, per_seed: list[dict]) -> dict:
        rs: list[float] = []
        ns: list[int] = []
        for row in per_seed:
            r = row.get("r")
            n = row.get("n")
            if r is None or n is None:
                continue
            r_f = float(r)
            n_i = int(n)
            if not np.isfinite(r_f) or n_i < 4:
                continue
            r_f = max(min(r_f, 1.0 - 1e-12), -1.0 + 1e-12)
            rs.append(r_f)
            ns.append(n_i)

        K = len(rs)
        if K == 0:
            return _empty_result()

        zs = np.arctanh(np.asarray(rs, dtype=float))
        se_zs = 1.0 / np.sqrt(np.asarray(ns, dtype=float) - 3.0)

        rubin = RubinAggregator(ci_level=self.ci_level).combine(
            [{"estimate": z, "se": s} for z, s in zip(zs, se_zs)]
        )
        if rubin["n_seeds"] == 0:
            return _empty_result()

        z_mean = rubin["mean"]
        r_mean = float(np.tanh(z_mean))
        ci_low_r = float(np.tanh(rubin["ci_low"]))
        ci_high_r = float(np.tanh(rubin["ci_high"]))

        return {
            "mean":        r_mean,
            "se":          float("nan"),
            "ci_low":      ci_low_r,
            "ci_high":     ci_high_r,
            "p":           rubin["p"],
            "p_acat":      _acat_p(per_seed)[0],
            "t":           rubin["t"],
            "df":          rubin["df"],
            "n_seeds":     rubin["n_seeds"],
            "within_var":  rubin["within_var"],
            "between_var": rubin["between_var"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mean + cross-seed SD — for estimators without an analytical per-fit SE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeanSDAggregator:
    """Aggregate by mean + cross-seed standard deviation.

    Requires ``per_seed`` entries with ``estimate`` (no SE needed). Uses
    the empirical cross-seed SD to compute ``se = SD/sqrt(K)`` and a
    t-CI with ``df = K - 1``. **No p-value is produced** (left as NaN);
    use ``ACATAggregator`` if a combined p is required.

    ``within_var`` is left NaN (no per-fit SE), ``between_var = SD²``.
    """

    ci_level: float = 0.95

    def combine(self, per_seed: list[dict]) -> dict:
        estimates: list[float] = []
        for r in per_seed:
            est = r.get("estimate")
            if est is None:
                continue
            est_f = float(est)
            if not np.isfinite(est_f):
                continue
            estimates.append(est_f)

        K = len(estimates)
        if K == 0:
            return _empty_result()

        est_arr = np.asarray(estimates, dtype=float)
        mean = float(est_arr.mean())

        if K > 1:
            sd = float(est_arr.std(ddof=1))
            se = sd / np.sqrt(K)
            df = K - 1
            t_crit = float(sp_stats.t.ppf(0.5 + self.ci_level / 2.0, df=df))
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
            between = sd * sd
        else:
            se = float("nan")
            df = 0
            ci_low = mean
            ci_high = mean
            between = 0.0

        return {
            "mean":        mean,
            "se":          float(se) if np.isfinite(se) else float("nan"),
            "ci_low":      float(ci_low),
            "ci_high":     float(ci_high),
            "p":           float("nan"),
            "p_acat":      _acat_p(per_seed)[0],
            "t":           float("nan"),
            "df":          float(df),
            "n_seeds":     K,
            "within_var":  float("nan"),
            "between_var": between,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Cauchy / ACAT — robust-to-dependence p-value combiner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ACATAggregator:
    """Cauchy combination test for p-values.

    Requires ``per_seed`` entries with ``p`` (and optionally ``estimate``
    just for the reported point estimate / mean). Robust to dependence
    between the seeds' p-values, which Fisher / Stouffer are not.

    The test statistic is ``T = mean_k tan((0.5 - p_k) * π)``, and
    ``p_combined = 0.5 - atan(T) / π``. We additionally pass through the
    mean of per-seed estimates (if provided) as a point summary, but do
    not produce SE / CI — ACAT only combines p-values.
    """

    def combine(self, per_seed: list[dict]) -> dict:
        p_combined, T = _acat_p(per_seed)
        if not np.isfinite(p_combined):
            return _empty_result()

        estimates: list[float] = []
        n_seeds = 0
        for r in per_seed:
            p = r.get("p")
            if p is not None and np.isfinite(float(p)):
                n_seeds += 1
            est = r.get("estimate")
            if est is not None and np.isfinite(float(est)):
                estimates.append(float(est))

        mean = float(np.mean(estimates)) if estimates else float("nan")

        return {
            "mean":        mean,
            "se":          float("nan"),
            "ci_low":      float("nan"),
            "ci_high":     float("nan"),
            "p":           p_combined,
            "p_acat":      p_combined,
            "t":           T,
            "df":          float("nan"),
            "n_seeds":     n_seeds,
            "within_var":  float("nan"),
            "between_var": float("nan"),
        }
