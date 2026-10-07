"""Frozen predictive grids and a same-family auxiliary score regressor."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from evaluation.ppd import PPDQuantileGrid, quantile_grid_integral

_MISSING = object()


def _value_key(value):
    if value is None or value is pd.NA:
        return ('na',)
    try:
        if pd.isna(value):
            return ('na',)
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return ('bool', bool(value))
    if isinstance(value, (int, np.integer)):
        return ('int', int(value))
    if isinstance(value, (float, np.floating)):
        return ('float', float(value))
    return ('str', str(value))


def _same_value(left, right):
    if left is None or right is None or left is pd.NA or right is pd.NA:
        return (left is None or left is pd.NA or (isinstance(left, float) and np.isnan(left))) and (
            right is None or right is pd.NA or (isinstance(right, float) and np.isnan(right)))
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return left == right


def _series_equal(left, right):
    if len(left) != len(right):
        return False
    if isinstance(left.dtype, pd.CategoricalDtype) or isinstance(right.dtype, pd.CategoricalDtype):
        return left.equals(right)
    if str(left.dtype) != str(right.dtype):
        return False
    return left.equals(right)


def _frame_equal(left, right):
    left = pd.DataFrame(left).reset_index(drop=True)
    right = pd.DataFrame(right).reset_index(drop=True)
    if list(left.columns) != list(right.columns):
        return False
    return all(_series_equal(left[c], right[c]) for c in left.columns)


def _frame_signature(frame):
    digest = hashlib.sha256()
    for name in frame.columns:
        column = frame[name]
        digest.update(str(name).encode())
        digest.update(str(column.dtype).encode())
        if isinstance(column.dtype, pd.CategoricalDtype):
            digest.update('\0'.join(map(str, column.cat.categories)).encode())
            digest.update(np.ascontiguousarray(column.cat.codes.to_numpy()).tobytes())
        else:
            digest.update(np.ascontiguousarray(column.to_numpy(copy=False)).tobytes())
    return digest.digest()


class _AuxiliaryPreprocessor:
    """Apply the base model's FeaturePreprocessor without refitting on every query.

    Context is the auxiliary fit rows. The category vocabulary and ctx-all-NaN
    patch are fixed from every cached raw row, so calibration and test queries
    share one fitted score model. A query the shared encoding cannot represent
    is returned with a local fit frame and the score model is refit for it.
    """

    def __init__(self, config, X_fit, reference):
        from evaluation.preprocessing import FeaturePreprocessor
        self.raw_fit = pd.DataFrame(X_fit).reset_index(drop=True)
        name = getattr(config, 'name', None)
        try:
            self.pre = FeaturePreprocessor(name) if name else None
        except KeyError:
            self.pre = None
        self.model_name = name
        self._model_takes_array = False
        if self.pre is None:
            self.fit_frame = self.raw_fit
            self._encoders = None
            return
        ref = self.raw_fit if reference is None else pd.DataFrame(reference).reset_index(drop=True)
        if list(ref.columns) != list(self.raw_fit.columns):
            raise ValueError('Auxiliary reference columns must match the fit rows')
        fit_frame, reference_frame = self.pre.fit_transform(self.raw_fit, ref)
        # TabDPT's numeric_array strategy returns a plain ndarray. Keep the
        # raw column labels for the encoder, and hand the array to the model.
        self._model_takes_array = isinstance(fit_frame, np.ndarray)
        self.fit_frame = self._label(fit_frame)
        reference_frame = self._label(reference_frame)
        self._encoders = [self._compile(c, ref, reference_frame) for c in self.raw_fit.columns]

    def _label(self, frame):
        if not isinstance(frame, np.ndarray):
            return frame
        columns = list(self.raw_fit.columns)
        if frame.ndim != 2 or frame.shape[1] != len(columns):
            raise ValueError('Auxiliary numeric array does not match the raw columns')
        return pd.DataFrame(frame, columns=columns)

    def model_input(self, frame):
        if self._model_takes_array:
            return np.ascontiguousarray(frame.to_numpy())
        return frame

    def _compile(self, column, raw_reference, reference_frame):
        encoded = self.fit_frame[column]
        if isinstance(encoded.dtype, pd.CategoricalDtype):
            categories = encoded.cat.categories
            known = set(map(str, categories))

            def stamp(values, categories=categories, known=known):
                obj = values.astype('object')
                labels = obj.where(obj.isna(), other=obj.astype(str))
                if labels.notna().any() and not set(labels.dropna().unique()).issubset(known):
                    return None
                return pd.Categorical(labels, categories=categories)
            return stamp
        raw_fit = self.raw_fit[column]
        raw_ref = raw_reference[column]
        if _series_equal(raw_fit, encoded) and _series_equal(raw_ref, reference_frame[column]):
            return lambda values: values
        mapping = {}
        for raw, enc in zip(raw_ref.tolist(), reference_frame[column].tolist()):
            key = _value_key(raw)
            previous = mapping.get(key, _MISSING)
            if previous is not _MISSING and not _same_value(previous, enc):
                return None
            mapping[key] = enc
        dtype = encoded.dtype

        def mapped(values, mapping=mapping, dtype=dtype):
            out = []
            for raw in values.tolist():
                key = _value_key(raw)
                if key not in mapping:
                    return None
                out.append(mapping[key])
            return pd.Series(out, index=values.index, dtype=dtype)
        return mapped

    def transform_query(self, X):
        query = pd.DataFrame(X).reset_index(drop=True)
        if self.pre is None:
            return query, None
        if list(query.columns) != list(self.raw_fit.columns):
            raise ValueError('Auxiliary query columns must match the fit rows')
        encoded = self._encode(query)
        if encoded is None or self._needs_local_fit(query, encoded):
            from evaluation.preprocessing import FeaturePreprocessor
            local_fit, local_query = FeaturePreprocessor(self.model_name).fit_transform(self.raw_fit, query)
            return self._label(local_query), self._label(local_fit)
        return encoded, None

    def _encode(self, query):
        if any(encoder is None for encoder in self._encoders):
            return None
        columns = {}
        for name, encoder in zip(query.columns, self._encoders):
            encoded = encoder(query[name])
            if encoded is None:
                return None
            columns[name] = encoded
        return pd.DataFrame(columns, index=query.index)

    def _needs_local_fit(self, query, encoded):
        from evaluation.preprocessing import _is_non_numeric_column
        if self.pre.strategy != 'passthrough_fill_ctx_nan':
            return False
        for name in query.columns:
            if _is_non_numeric_column(encoded[name]) and encoded[name].isna().all():
                return True
        return False

    def changes_inputs(self, queries):
        if self.pre is None:
            return False
        if not _frame_equal(self.raw_fit, self.fit_frame):
            return True
        for query in queries:
            encoded, local_fit = self.transform_query(query)
            if local_fit is not None or not _frame_equal(query, encoded):
                return True
        return False


class ScoreRegressor:
    def __init__(self, config, X, y, seed, n_estimators, reference=None):
        from evaluation.models import ModelRunner
        self.runner = ModelRunner(config, seed, n_estimators)
        self.y = np.asarray(y).reshape(-1)
        self._prep = _AuxiliaryPreprocessor(config, X, reference)
        self.runner._model.fit(self._prep.model_input(self._prep.fit_frame), self.y)
        self._fitted_sig = _frame_signature(self._prep.fit_frame)

    def _ensure_fit(self, frame):
        signature = _frame_signature(frame)
        if signature != self._fitted_sig:
            self.runner._model.fit(self._prep.model_input(frame), self.y)
            self._fitted_sig = signature

    def predict_quantile(self, X, q):
        frame, local_fit = self._prep.transform_query(X)
        self._ensure_fit(self._prep.fit_frame if local_fit is None else local_fit)
        r = self.runner
        values = r._model.predict(self._prep.model_input(frame), output_type='quantiles',
                    **{r.config.quantile_param:[float(q)]}, **r._predict_kwargs())
        return np.asarray(values).reshape(-1)


class CachedPredictor:
    def __init__(self, bundle, clone_factory=None):
        self.bundle = bundle
        self.levels = np.asarray(bundle['quantile_levels'])
        self.X = np.concatenate([bundle[s]['X'] for s in ('train','cal','test')])
        self.grid = np.concatenate([bundle[s]['ppd_quantiles'] for s in ('train','cal','test')])
        self.point = np.concatenate([bundle[s]['point_pred'] for s in ('train','cal','test')])
        self.clone_factory = clone_factory
        self.queries = {}
        offset = 0
        for name in ('train', 'cal', 'test'):
            n = len(bundle[name]['y'])
            self.queries[name] = np.arange(offset, offset+n, dtype=np.int64)[:, None]
            offset += n

    def query(self, split):
        """Opaque row keys, never used as geometry or auxiliary model features."""
        return self.queries[split]

    def _indices(self, X):
        keys = np.asarray(X)
        if keys.ndim != 2 or keys.shape[1] != 1 or not np.issubdtype(keys.dtype, np.integer):
            raise ValueError('Cached predictions require integer row keys from query()')
        indices = keys[:, 0]
        if np.any(indices < 0) or np.any(indices >= len(self.X)):
            raise ValueError('Cached prediction row key out of bounds')
        return indices

    def feature_values(self, X):
        return self.X[self._indices(X)]

    def score_feature_values(self, X):
        """Keep original columns and dtypes for the auxiliary tabular model."""
        if not all('X_raw' in self.bundle[s] for s in ('train', 'cal', 'test')):
            raise ValueError('RCP requires X_raw in every split; regenerate the input with prepare or prepare-predictions')
        if not hasattr(self, '_score_X'):
            self._score_X = pd.concat(
                [self.bundle[s]['X_raw'] for s in ('train', 'cal', 'test')], ignore_index=True)
        return self._score_X.iloc[self._indices(X)].reset_index(drop=True)

    def predict(self, X):
        return self.point[self._indices(X)]

    def predict_quantile(self, X, q):
        return PPDQuantileGrid(self.grid[self._indices(X)], self.levels).quantile_at(q)

    def predict_std(self, X):
        values = self.grid[self._indices(X)]
        mean = quantile_grid_integral(values, self.levels)
        return np.sqrt(np.maximum(quantile_grid_integral(values**2, self.levels)-mean**2, 1e-16))

    def predict_cdf(self, X, y):
        return PPDQuantileGrid(self.grid[self._indices(X)], self.levels).cdf_at(y)

    def clone_for(self, X, y, random_state=None):
        seed = self.bundle['seed'] if random_state is None else random_state
        if self.bundle['base_model'] == 'bart' and self.clone_factory is None:
            from .bart_score import IndexedBARTScoreRegressor
            return IndexedBARTScoreRegressor(self, X, y, seed)
        features = self.score_feature_values(X)
        if self.clone_factory is not None:
            regressor = self.clone_factory(features,y,seed)
        else:
            from evaluation.models import ModelRegistry
            config = ModelRegistry.default_regression()[self.bundle['base_model']]
            regressor = ScoreRegressor(
                config, features, y, seed, self.bundle['n_estimators'], reference=self._score_X)
        return IndexedScoreRegressor(regressor, self)


class IndexedScoreRegressor:
    """Resolve query keys before passing real features to an auxiliary model."""
    def __init__(self, regressor, cache):
        self.regressor, self.cache = regressor, cache

    def predict_quantile(self, X, q):
        return self.regressor.predict_quantile(self.cache.score_feature_values(X), q)
