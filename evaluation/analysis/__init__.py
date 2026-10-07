"""Statistical analyzers and table builders.

Two families:

* Dataset-level — every analyzer inherits :class:`DatasetAnalyzer`
  (common ``run`` + ``save`` interface) and is plugged into
  :class:`evaluation.pipelines.DatasetAnalysisPipeline` as part of a
  registered list.  Analyzers that produce per-feature scores also mix
  in :class:`FeatureImportance`.  Built-in members live under
  :mod:`evaluation.analysis.univariate` (Spearman, Chatterjee,
  Univariate RF) and :mod:`evaluation.analysis.multivariate` (LME,
  RF importance).

* Instance-level (one row per test point):
  :class:`InstanceCoverageAnalyzer`, :class:`InstanceWidthAnalyzer`.

Plus shared helpers: :class:`MetricsTable`, :class:`EvalTable`,
:class:`InstanceTable`, :class:`ModelComparator`,
:class:`SummaryReporter`, and the HTML pivot helper
:func:`render_feature_pivot_html` shared by analyzer ``save`` methods.
"""
from .base              import (
    _alpha_to_cell,
    alpha_col_for_html,
    DatasetAnalyzer,
    FeatureImportance,
    group_keys_by_ratio,
    InputKind,
    render_feature_pivot_html,
)
from .comparison        import ModelComparator, WilcoxonResult
from .score_decomposition import (
    ScoreDecomposition,
    ScoreDecompositionAnalyzer,
    ScoreDecompositionResult,
)
from .multivariate import (
    CrossModelLMEAnalyzer,
    CrossModelLMEResult,
    FeatureGroupRFAnalyzer,
    LMEAnalyzer,
    LMEResult,
    RFImportanceAnalyzer,
    RFResult,
)
from .univariate import (
    ChatterjeeAnalyzer,
    ChatterjeeResult,
    MIN_SPEARMAN_N,
    SpearmanAnalyzer,
    SpearmanResult,
    UnivariateRFAnalyzer,
    UnivariateResult,
)
from .instance_predict  import (
    InstanceChatterjeeAnalyzer,
    InstanceCoverageAnalyzer,
    InstanceCoverageResult,
    InstanceRegressionResult,
    InstanceWidthAnalyzer,
)
from .reporter          import SummaryReporter
from .tables            import (
    AvgTable,
    build_long_table,
    EvalTable,
    INSTANCE_ID_COLS,
    INSTANCE_OUTCOME_COLS,
    InstanceTable,
    MetricsTable,
    PairDeltaTable,
    RelBiasEvalTable,
    RelBiasTable,
    RelEvalTable,
    RelTable,
    slice_table_by,
)

__all__ = [
    "ScoreDecomposition",
    "ScoreDecompositionAnalyzer",
    "ScoreDecompositionResult",
    "alpha_col_for_html",
    "AvgTable",
    "build_long_table",
    "ChatterjeeAnalyzer",
    "ChatterjeeResult",
    "CrossModelLMEAnalyzer",
    "CrossModelLMEResult",
    "FeatureGroupRFAnalyzer",
    "DatasetAnalyzer",
    "EvalTable",
    "FeatureImportance",
    "group_keys_by_ratio",
    "INSTANCE_ID_COLS",
    "INSTANCE_OUTCOME_COLS",
    "InputKind",
    "InstanceChatterjeeAnalyzer",
    "InstanceCoverageAnalyzer",
    "InstanceCoverageResult",
    "InstanceRegressionResult",
    "InstanceTable",
    "InstanceWidthAnalyzer",
    "LMEAnalyzer",
    "LMEResult",
    "MIN_SPEARMAN_N",
    "MetricsTable",
    "ModelComparator",
    "PairDeltaTable",
    "RelBiasEvalTable",
    "RelBiasTable",
    "RelEvalTable",
    "RelTable",
    "render_feature_pivot_html",
    "RFImportanceAnalyzer",
    "RFResult",
    "SpearmanAnalyzer",
    "SpearmanResult",
    "SummaryReporter",
    "slice_table_by",
    "UnivariateRFAnalyzer",
    "UnivariateResult",
    "WilcoxonResult",
]
