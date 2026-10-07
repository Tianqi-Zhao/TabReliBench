"""Conditional (joint-feature) dataset-level analyzers.

LME and multivariate RF fit features jointly — coefficients / importance
scores depend on which other features are in the model.  The feature-group RF
analyzer compares three predefined predictor subsets while reusing the regular
RF fitting path.  The LME analyzers share the block-aware B+C imputation logic
from :mod:`evaluation.analysis.multivariate._lme_utils`; RF uses XGBoost's
native missing-value handling.
"""
from .cross_model_lme import CrossModelLMEAnalyzer, CrossModelLMEResult
from .feature_group_rf import FeatureGroupRFAnalyzer
from .lme import LMEAnalyzer, LMEResult
from .rf  import RFImportanceAnalyzer, RFResult

__all__ = [
    "CrossModelLMEAnalyzer",
    "CrossModelLMEResult",
    "FeatureGroupRFAnalyzer",
    "LMEAnalyzer",
    "LMEResult",
    "RFImportanceAnalyzer",
    "RFResult",
]
