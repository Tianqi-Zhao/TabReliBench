"""Evaluate interval records using the benchmark's metric implementations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from evaluation.metrics.regression import RegressionMetricsCalculator, PointAccuracyMetric
from evaluation.metrics.base import RegressionContext


class IntervalView:
    """Expose only the two actual endpoints; never invent a full PPD."""
    def __init__(self, lower, upper, alpha):
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.alpha = alpha

    def interval(self, alpha):
        if alpha != self.alpha:
            raise ValueError('Interval alpha mismatch')
        return self.lower, self.upper

    def quantile_at(self, tau):
        if np.isclose(tau, self.alpha / 2, rtol=0, atol=1e-12):
            return self.lower
        if np.isclose(tau, 1 - self.alpha / 2, rtol=0, atol=1e-12):
            return self.upper
        raise ValueError('An interval does not identify a predictive distribution')


class PredictionSetView(IntervalView):
    """Exact standard-CQR sets, including explicitly marked empty sets.

    Reversed endpoints encode an empty set, never a negative width. Interval
    and endpoint losses are undefined there; do not score a fabricated point
    or average only the nonempty subset.
    """
    def __init__(self, slot, alpha, params):
        super().__init__(slot['lower'], slot['upper'], alpha)
        self.empty = self.lower > self.upper
        if 'empty_set' in slot:
            mask = np.asarray(slot['empty_set'])
            if (params.get('score_variant') != 'standard_cqr'
                    or params.get('empty_set_policy') != 'exact_sublevel_set_v1'
                    or mask.dtype != np.bool_ or mask.shape != self.lower.shape
                    or not np.array_equal(mask, self.empty)):
                raise ValueError('Invalid standard-CQR empty-set declaration')
        elif self.empty.any():
            raise ValueError('Crossed/empty intervals require explicit set-valued evaluation')

    def finalize_metrics(self, per_dataset, per_instance, y):
        e = slice(len(y)//2, None)
        width = np.maximum(self.upper - self.lower, 0.)
        per_dataset['empty_set_rate'] = float(self.empty[e].mean())
        per_dataset['infinite_interval_rate'] = float(np.isinf(width[e]).mean())
        if not self.empty.any():
            return
        per_instance['empty_set'] = self.empty
        per_instance['width'] = width
        per_dataset['avg_length'] = float(width[e].mean())
        per_dataset['avg_width_norm'] = float(width[e].mean() / max(float(np.std(y[e])), 1e-10))
        for key in ('winkler', 'pinball_lower_row', 'pinball_upper_row'):
            per_instance[key][self.empty] = np.nan
        if self.empty[e].any():
            for key in ('interval_score', 'pinball_lower', 'pinball_upper', 'pinball_mean'):
                per_dataset[key] = float('nan')
            per_dataset['interval_loss_status'] = 'undefined_empty_prediction_set'


class ConformalMetricsCalculator(RegressionMetricsCalculator):
    def compute_for_record(self, record):
        y = np.asarray(record['y_test'], dtype=float)
        X = record['X_test']
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=record.get('feature_names'))
        if len(X) != len(y) or len(y) < 2 or not np.isfinite(y).all():
            raise ValueError('Invalid or unaligned test data')
        names = list(X.columns)
        result = {k: record[k] for k in ('dataset_id','seed','ratio','model','base_model','method','protocol','provenance','data_usage') if k in record}
        result.update(task='regression', alphas=self.alphas, y_test=y, feature_names=names,
                      n_test=len(y), alpha_dependent={}, alpha_free={'per_dataset':{}, 'per_instance':{}},
                      unavailable_metrics={'crps_mean':'interval_only', 'pit':'interval_only'})
        for alpha in self.alphas:
            slot = record['intervals'][alpha]
            lower, upper = (np.asarray(slot[k], dtype=float) for k in ('lower','upper'))
            if lower.shape != y.shape or upper.shape != y.shape or np.isnan(lower).any() or np.isnan(upper).any():
                raise ValueError('Invalid interval arrays')
            view = PredictionSetView(slot, alpha, record.get('params', {}).get(alpha, {}))
            # Keep all existing alpha-dependent per-instance metrics, including pinball.
            ctx = RegressionContext(ppd=view, y_test=y, X_test=X, feature_names=names,
                                    alpha=alpha, eval_slice=slice(len(y)//2,None))
            pd_out, pi_out = self._run_primary_alpha(ctx, alpha)
            view.finalize_metrics(pd_out, pi_out, y)
            result['alpha_dependent'][alpha] = {'per_dataset':pd_out,'per_instance':pi_out}
        if record.get('point_pred') is not None:
            point = np.asarray(record['point_pred'], dtype=float)
            if point.shape != y.shape or not np.isfinite(point).all():
                raise ValueError('Invalid point predictions')
            ctx = RegressionContext(ppd=None, y_test=y, X_test=X, feature_names=names,
                                    alpha=None, eval_slice=slice(None), point_pred=point)
            out = PointAccuracyMetric().compute(ctx)
            result['alpha_free'] = {'per_dataset':out.per_dataset,'per_instance':out.per_instance}
        return result
