"""Classification calibration metrics.

Concrete :class:`ClassificationMetric` subclasses
--------------------------------------------------

* :class:`TopLabelMetric`      — accuracy + confidence calibration
  (ECE / MCE) + per-row confidence / pred_label / hit_top1.

* :class:`ClasswiseECEMetric`  — classwise (one-vs-rest) ECE / MCE.

* :class:`CMCEMetric`          — Cumulative Mass Calibration Error +
  per-row rank_true / mass_at_true.

* :class:`BrierMetric`         — multiclass Brier score + per-row values.

* :class:`LogLossMetric`       — cross-entropy + per-row values +
  per-row true_prob.

Orchestrator
------------
:class:`ClassificationMetricsCalculator`
    Validates the prediction PKL, then iterates over a configurable list of
    :class:`ClassificationMetric` instances (default:
    :data:`DEFAULT_CLASSIFICATION_METRICS`) and merges their output into the
    ``alpha_free`` branch of the 2×2 nested schema.

``ConcreteClassificationMetricsCalculator`` is a deprecated alias of the
same class (for any code that still imports the old split name).

Private helpers
---------------
:func:`_label_to_index`, :func:`_equal_width_bins`,
:func:`_equal_mass_bins`, :func:`_binned_diff` — shared by multiple
metric classes.
"""
from __future__ import annotations

from typing import ClassVar, Optional, Sequence

import numpy as np

from ..spec import TASK_CLASSIFICATION
from .base import (
    BaseMetricsCalculator,
    ClassificationContext,
    ClassificationMetric,
    MetricOutput,
)

try:
    from sklearn.metrics import roc_auc_score, f1_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Private helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _label_to_index(y_test: np.ndarray, classes_: np.ndarray) -> np.ndarray:
    """Map arbitrary-dtype labels to integer indices into ``classes_``.

    Uses ``np.searchsorted`` so it works with any comparable dtype.
    The caller is responsible for ensuring every element of ``y_test``
    actually appears in ``classes_``.
    """
    classes_ = np.asarray(classes_)
    return np.searchsorted(classes_, y_test).astype(np.intp)


def _equal_width_bins(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each score in ``[0, 1]`` to an equal-width bin.

    Returns an integer array of shape ``(len(scores),)`` with values in
    ``[0, n_bins - 1]``.
    """
    edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.searchsorted(edges[1:], scores, side="left")
    return np.clip(bin_id, 0, n_bins - 1).astype(np.intp)


def _equal_mass_bins(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each score to an equal-frequency (quantile) bin.

    The bin boundaries are determined from ``scores`` itself, so every
    bin contains approximately ``len(scores) // n_bins`` points.
    Returns an integer array of shape ``(len(scores),)`` with values in
    ``[0, n_bins - 1]``.
    """
    edges  = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0]  = -np.inf
    edges[-1] =  np.inf
    bin_id = np.searchsorted(edges[1:], scores, side="left")
    return np.clip(bin_id, 0, n_bins - 1).astype(np.intp)


def _binned_diff(
    scores: np.ndarray,
    target: np.ndarray,
    bin_id: np.ndarray,
    n_bins: int,
    reduce: str,
) -> float:
    """Compute ECE or MCE from binned predictions.

    Parameters
    ----------
    scores:
        Predicted probability for the positive class per sample.
    target:
        Binary ground-truth label per sample (0 or 1).
    bin_id:
        Bin assignment for each sample (integer in ``[0, n_bins-1]``).
    n_bins:
        Total number of bins.
    reduce:
        ``'ece'`` → weighted average of absolute bin differences.
        ``'mce'`` → maximum absolute bin difference.

    Returns
    -------
    float  (ECE or MCE value in ``[0, 1]``).
    """
    n = len(scores)
    if n == 0:
        return float("nan")

    diffs: list[float] = []
    weights: list[float] = []

    for b in range(n_bins):
        mask = bin_id == b
        cnt  = int(mask.sum())
        if cnt == 0:
            continue
        mean_score  = float(scores[mask].mean())
        mean_target = float(target[mask].mean())
        diffs.append(abs(mean_target - mean_score))
        weights.append(cnt)

    if not diffs:
        return float("nan")

    if reduce == "mce":
        return float(max(diffs))

    # ECE: weighted average by bin population
    total = sum(weights)
    return float(sum(d * w for d, w in zip(diffs, weights)) / total)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete ClassificationMetric subclasses
# ─────────────────────────────────────────────────────────────────────────────

class TopLabelMetric(ClassificationMetric):
    """Accuracy + confidence calibration (ECE / MCE).

    Groups by ``max_j p_ij`` (the model's confidence) and compares it to
    the empirical top-1 accuracy within each bin.  Outputs both
    equal-width (``_ew``) and equal-mass (``_em``) variants.

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``accuracy``,
    ``confidence_ece_ew``, ``confidence_mce_ew``,
    ``confidence_ece_em``, ``confidence_mce_em``

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``confidence``, ``pred_label``, ``hit_top1``
    """
    name: ClassVar[str] = "top_label"

    def compute(self, ctx: ClassificationContext) -> MetricOutput:
        proba = ctx.proba                              # (N, K)
        y_idx = ctx.y_idx                             # (N,)  int

        pred_label = np.argmax(proba, axis=1)         # (N,)  int
        confidence = proba[np.arange(len(proba)), pred_label]  # (N,)
        hit_top1   = (pred_label == y_idx).astype(np.float64)  # (N,)

        acc = float(hit_top1.mean())

        ew = _equal_width_bins(confidence, ctx.n_bins)
        em = _equal_mass_bins(confidence,  ctx.n_bins)

        return MetricOutput(
            per_dataset={
                "accuracy":           acc,
                "confidence_ece_ew":  _binned_diff(confidence, hit_top1, ew, ctx.n_bins, "ece"),
                "confidence_mce_ew":  _binned_diff(confidence, hit_top1, ew, ctx.n_bins, "mce"),
                "confidence_ece_em":  _binned_diff(confidence, hit_top1, em, ctx.n_bins, "ece"),
                "confidence_mce_em":  _binned_diff(confidence, hit_top1, em, ctx.n_bins, "mce"),
            },
            per_instance={
                "confidence": confidence,
                "pred_label": pred_label,
                "hit_top1":   hit_top1,
            },
        )


class ClasswiseECEMetric(ClassificationMetric):
    """Classwise (one-vs-rest) ECE and MCE.

    For each class *j*, treats ``p_{·,j}`` as the predicted probability
    and ``𝟙[y = j]`` as the binary target.  Computes ``ECE_j`` and the
    per-bin gap for class *j*.  The reported scalars are:

    * ``classwise_ece_*`` = arithmetic mean of ``ECE_j`` over all *K* classes.
    * ``classwise_mce_*`` = max over all (bin, class) pairs.

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``classwise_ece_ew``, ``classwise_mce_ew``,
    ``classwise_ece_em``,  ``classwise_mce_em``

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    *(none — classwise quantities are inherently per-class, not per-instance)*
    """
    name: ClassVar[str] = "classwise_ece"

    def compute(self, ctx: ClassificationContext) -> MetricOutput:
        proba = ctx.proba   # (N, K)
        y_idx = ctx.y_idx   # (N,)
        K     = proba.shape[1]
        n_bins = ctx.n_bins

        ece_ew_list: list[float] = []
        mce_ew_vals: list[float] = []
        ece_em_list: list[float] = []
        mce_em_vals: list[float] = []

        for j in range(K):
            scores = proba[:, j]
            target = (y_idx == j).astype(np.float64)

            ew = _equal_width_bins(scores, n_bins)
            em = _equal_mass_bins(scores,  n_bins)

            ece_ew_list.append(_binned_diff(scores, target, ew, n_bins, "ece"))
            mce_ew_vals.append(_binned_diff(scores, target, ew, n_bins, "mce"))
            ece_em_list.append(_binned_diff(scores, target, em, n_bins, "ece"))
            mce_em_vals.append(_binned_diff(scores, target, em, n_bins, "mce"))

        def _safe_mean(vals: list[float]) -> float:
            finite = [v for v in vals if np.isfinite(v)]
            return float(np.mean(finite)) if finite else float("nan")

        def _safe_max(vals: list[float]) -> float:
            finite = [v for v in vals if np.isfinite(v)]
            return float(max(finite)) if finite else float("nan")

        return MetricOutput(
            per_dataset={
                "classwise_ece_ew": _safe_mean(ece_ew_list),
                "classwise_mce_ew": _safe_max(mce_ew_vals),
                "classwise_ece_em": _safe_mean(ece_em_list),
                "classwise_mce_em": _safe_max(mce_em_vals),
            },
            per_instance={},
        )


class CMCEMetric(ClassificationMetric):
    """Cumulative Mass Calibration Error (CMCE).

    For each instance *i*, the predicted probabilities are sorted
    descending to form nested sets of increasing cumulative mass.  For
    each prefix set of size *k* the cumulative mass is
    ``mass_{i,k} = Σ_{l≤k} sorted_p_{i,l}`` and the coverage indicator
    is ``cov_{i,k} = 𝟙[true label is in the top-k]``.

    All ``N·K`` (mass, cov) pairs are pooled and binned by mass.

        CMCE = (1 / N·K) · Σ_m |S_m| · |cov(S_m) − mass(S_m)|

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``cmce_ew``, ``cmce_em``

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``rank_true``     — 1-indexed rank of the true label (lower = better).
    ``mass_at_true``  — cumulative mass of the smallest nested set that
                        contains the true label.
    """
    name: ClassVar[str] = "cmce"

    def compute(self, ctx: ClassificationContext) -> MetricOutput:
        proba = ctx.proba   # (N, K)
        y_idx = ctx.y_idx   # (N,)
        N, K  = proba.shape
        n_bins = ctx.n_bins

        # Sort each row descending
        order      = np.argsort(-proba, axis=1)                     # (N, K)
        sorted_p   = np.take_along_axis(proba, order, axis=1)       # (N, K)
        mass       = np.cumsum(sorted_p, axis=1)                    # (N, K)

        # rank_true[i] = 0-indexed position of the true label after sorting
        rank_true  = (order == y_idx[:, None]).argmax(axis=1)       # (N,)
        mass_at_true = mass[np.arange(N), rank_true]                # (N,)

        # Coverage matrix: cov[i,k] = 1 iff true label rank <= k
        k_idx  = np.arange(K)                                       # (K,)
        cov    = (rank_true[:, None] <= k_idx[None, :]).astype(np.float64)  # (N, K)

        # Flatten for binning
        mass_flat = mass.ravel()   # (N*K,)
        cov_flat  = cov.ravel()    # (N*K,)

        ew = _equal_width_bins(mass_flat, n_bins)
        em = _equal_mass_bins(mass_flat,  n_bins)

        cmce_ew = _binned_diff(mass_flat, cov_flat, ew, n_bins, "ece")
        cmce_em = _binned_diff(mass_flat, cov_flat, em, n_bins, "ece")

        return MetricOutput(
            per_dataset={
                "cmce_ew": cmce_ew,
                "cmce_em": cmce_em,
            },
            per_instance={
                "rank_true":    (rank_true + 1).astype(np.int32),  # 1-indexed
                "mass_at_true": mass_at_true,
            },
        )


class BrierMetric(ClassificationMetric):
    """Multiclass Brier score.

    ``BS_i = Σ_j (p_{i,j} − 𝟙[y_i = j])²``

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``brier_score``  (mean over the test set)

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``brier_row``
    """
    name: ClassVar[str] = "brier"

    def compute(self, ctx: ClassificationContext) -> MetricOutput:
        proba = ctx.proba   # (N, K)
        y_idx = ctx.y_idx   # (N,)
        N, K  = proba.shape

        one_hot   = np.zeros_like(proba)
        one_hot[np.arange(N), y_idx] = 1.0

        brier_row = np.sum((proba - one_hot) ** 2, axis=1)   # (N,)

        return MetricOutput(
            per_dataset={"brier_score": float(brier_row.mean())},
            per_instance={"brier_row": brier_row},
        )


class LogLossMetric(ClassificationMetric):
    """Cross-entropy (log-loss).

    ``LL_i = −log(clip(p_{i, y_i}, eps, 1 − eps))``

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``log_loss``  (mean over the test set)

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    ``logloss_row``, ``true_prob``
    """
    name: ClassVar[str] = "log_loss"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = float(eps)

    def compute(self, ctx: ClassificationContext) -> MetricOutput:
        proba = ctx.proba   # (N, K)
        y_idx = ctx.y_idx   # (N,)

        true_prob   = proba[np.arange(len(proba)), y_idx]
        clipped     = np.clip(true_prob, self.eps, 1.0 - self.eps)
        logloss_row = -np.log(clipped)

        return MetricOutput(
            per_dataset={"log_loss": float(logloss_row.mean())},
            per_instance={
                "logloss_row": logloss_row,
                "true_prob":   true_prob,
            },
        )


class ClassificationAccuracyMetric(ClassificationMetric):
    """ROC-AUC (OvR macro) and macro-F1 for prediction accuracy.

    Requires scikit-learn.  Both metrics are skipped gracefully (NaN) when
    sklearn is not installed or when the problem has only one class.

    per_dataset keys
    ~~~~~~~~~~~~~~~~
    ``roc_auc_ovr_macro``  — OvR macro-averaged AUC; for binary problems this
                             is identical to the standard AUC.
    ``macro_f1``           — macro-averaged F1 score (predicted label = argmax).

    per_instance keys
    ~~~~~~~~~~~~~~~~~
    *(none — both scalars are inherently aggregate, not per-instance)*
    """
    name: ClassVar[str] = "cls_accuracy"

    def compute(self, ctx: ClassificationContext) -> MetricOutput:
        nan = float("nan")
        if not _SKLEARN_AVAILABLE:
            return MetricOutput(
                per_dataset={"roc_auc_ovr_macro": nan, "macro_f1": nan},
                per_instance={},
            )

        proba = ctx.proba   # (N, K)
        y_idx = ctx.y_idx   # (N,)  integer labels
        K     = proba.shape[1]

        # ROC-AUC (OvR macro)
        try:
            if K == 2:
                roc_auc = float(roc_auc_score(y_idx, proba[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(y_idx, proba, multi_class="ovr", average="macro")
                )
        except Exception:
            roc_auc = nan

        # Macro-F1 from hard predictions
        try:
            pred_label = np.argmax(proba, axis=1)
            macro_f1   = float(f1_score(y_idx, pred_label, average="macro", zero_division=0))
        except Exception:
            macro_f1 = nan

        return MetricOutput(
            per_dataset={"roc_auc_ovr_macro": roc_auc, "macro_f1": macro_f1},
            per_instance={},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Default metric list (the extension point)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLASSIFICATION_METRICS: list[ClassificationMetric] = [
    TopLabelMetric(),
    ClasswiseECEMetric(),
    CMCEMetric(),
    BrierMetric(),
    LogLossMetric(),
    ClassificationAccuracyMetric(),
]


# Default response columns used by LME / Spearman / Chatterjee / RF in the
# dataset-level analysis pipeline. Picked to cover prediction accuracy,
# proper scoring, and calibration without duplicating ew/em variants by
# default.
CLASSIFICATION_RESPONSE_COLS: list[str] = [
    "log_loss",
    "brier_score",
    "accuracy",
    "macro_f1",
    "roc_auc_ovr_macro",
    "confidence_ece_em",
    "classwise_ece_em",
    "cmce_em",
]


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationMetricsCalculator(BaseMetricsCalculator):
    """Compute classification calibration metrics for one prediction record.

    Validates the prediction-PKL record, then iterates over a list of
    :class:`ClassificationMetric` instances (default:
    :data:`DEFAULT_CLASSIFICATION_METRICS`) and merges their
    :class:`MetricOutput` into ``alpha_free`` (classification has no
    ``alpha_dependent`` branch — it is always an empty dict).

    Parameters
    ----------
    metrics:
        List of :class:`ClassificationMetric` instances to evaluate.
        Defaults to :data:`DEFAULT_CLASSIFICATION_METRICS`.
    n_bins:
        Number of bins for ECE / MCE / CMCE computation (default 15).
    """

    task: ClassVar[str] = TASK_CLASSIFICATION

    _REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "task", "proba", "classes_", "y_test",
    )

    def __init__(
        self,
        metrics: Optional[Sequence[ClassificationMetric]] = None,
        n_bins: int = 15,
    ) -> None:
        self.metrics = (
            list(metrics) if metrics is not None
            else list(DEFAULT_CLASSIFICATION_METRICS)
        )
        self.n_bins = int(n_bins)

    def _validate_record(self, record: dict) -> None:
        task = record.get("task", TASK_CLASSIFICATION)
        if task != TASK_CLASSIFICATION:
            raise ValueError(
                f"ClassificationMetricsCalculator: record has "
                f"task={task!r}; expected {TASK_CLASSIFICATION!r}."
            )
        missing = [k for k in self._REQUIRED_KEYS if k not in record]
        if missing:
            raise KeyError(
                f"Classification record missing required keys: {missing}. "
                f"Have: {sorted(record.keys())}"
            )
        proba    = np.asarray(record["proba"])
        classes_ = np.asarray(record["classes_"])
        y_test   = np.asarray(record["y_test"])
        if proba.ndim != 2:
            raise ValueError(
                f"proba must be 2D (n_test, n_classes); got "
                f"ndim={proba.ndim} shape={proba.shape}"
            )
        if proba.shape[1] != classes_.shape[0]:
            raise ValueError(
                f"proba has {proba.shape[1]} columns but classes_ has "
                f"{classes_.shape[0]} entries."
            )
        if proba.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"proba has {proba.shape[0]} rows but y_test has "
                f"{y_test.shape[0]} entries."
            )

    @staticmethod
    def _merge_into(
        dest_pd: dict,
        dest_pi: dict,
        output: MetricOutput,
        metric_name: str,
    ) -> None:
        for key in output.per_dataset:
            if key in dest_pd:
                raise KeyError(
                    f"Metric '{metric_name}' tried to write per_dataset key "
                    f"'{key}' which was already produced by a previous metric."
                )
            dest_pd[key] = output.per_dataset[key]

        for key in output.per_instance:
            if key in dest_pi:
                raise KeyError(
                    f"Metric '{metric_name}' tried to write per_instance key "
                    f"'{key}' which was already produced by a previous metric."
                )
            dest_pi[key] = output.per_instance[key]

    def compute_for_record(self, record: dict) -> dict:
        """Compute every registered metric for one prediction PKL record."""
        self._validate_record(record)
        proba    = np.asarray(record["proba"],    dtype=np.float64)
        classes_ = np.asarray(record["classes_"])
        y_test   = np.asarray(record["y_test"])
        y_idx    = _label_to_index(y_test, classes_)

        ctx = ClassificationContext(
            proba=proba, classes_=classes_,
            y_idx=y_idx, n_bins=self.n_bins,
        )

        af_pd: dict = {}
        af_pi: dict = {}
        for m in self.metrics:
            out = m.compute(ctx)
            self._merge_into(af_pd, af_pi, out, m.name)

        return {
            "task":            TASK_CLASSIFICATION,
            "model":           record.get("model"),
            "dataset_id":      record.get("dataset_id"),
            "seed":            record.get("seed"),
            "ratio":           record.get("ratio"),
            "n_total":         record.get("n_total"),
            "n_train":         record.get("n_train"),
            "n_test":          record.get("n_test"),
            "n_context":       record.get("n_context"),
            "n_features":      record.get("n_features"),
            "feature_names":   record.get("feature_names"),
            "classes_":        classes_,
            "y_test":          y_test,
            "alpha_dependent": {},
            "alpha_free": {
                "per_dataset":  af_pd,
                "per_instance": af_pi,
            },
        }


# Back-compat alias (deprecated name from the stub + concrete split).
ConcreteClassificationMetricsCalculator = ClassificationMetricsCalculator
