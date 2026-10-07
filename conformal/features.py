"""Distance features fitted exclusively on the auxiliary/proper training rows."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def distance_features(training, queries):
    numeric = list(training.select_dtypes(include='number').columns)
    categorical = [c for c in training.columns if c not in numeric]

    def clean(frame):
        frame = frame.copy()
        for column in categorical:
            frame[column] = frame[column].astype('string').fillna('__MISSING__').astype(str)
        return frame

    transformers = []
    if numeric:
        transformers.append(('num', make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), numeric))
    if categorical:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical))
    encoder = ColumnTransformer(transformers)
    encoder.fit(clean(training))
    return np.asarray(encoder.transform(clean(queries)), dtype=float)
