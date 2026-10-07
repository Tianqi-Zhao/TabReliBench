"""Marginal (per-feature) dataset-level analyzers.

Each analyzer scores one feature at a time against the response — adding
or removing other features does not change a given feature's result
(except Univariate RF's shared ``dropna`` across all listed columns).

Spearman r (direction) pairs with Chatterjee ξ (magnitude).  Univariate
RF is an optional CV-based predictive measure.
"""
from .chatterjee   import ChatterjeeAnalyzer, ChatterjeeResult
from .spearman     import MIN_SPEARMAN_N, SpearmanAnalyzer, SpearmanResult
from .univariate_rf import UnivariateRFAnalyzer, UnivariateResult

__all__ = [
    "ChatterjeeAnalyzer",
    "ChatterjeeResult",
    "MIN_SPEARMAN_N",
    "SpearmanAnalyzer",
    "SpearmanResult",
    "UnivariateRFAnalyzer",
    "UnivariateResult",
]
