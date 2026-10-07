"""CQR score strategies shared by global and localized calibrators."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CQRScore:
    nonnegative: bool

    @property
    def variant(self):
        return 'nonnegative_cqr' if self.nonnegative else 'standard_cqr'

    def bounds(self, model, X, alpha):
        return tuple(np.asarray(model.predict_quantile(X, q)).reshape(-1)
                     for q in (alpha / 2, 1 - alpha / 2))

    def compute(self, model, X, y, alpha):
        lower, upper = self.bounds(model, X, alpha)
        scores = np.maximum(lower - y, y - upper)
        return np.maximum(scores, 0.) if self.nonnegative else scores

    def invert(self, model, X, thresholds, alpha):
        lower, upper = self.bounds(model, X, alpha)
        # Standard CQR preserves negative thresholds and exact empty sets
        # (lower > upper). The evaluator handles these explicitly.
        return lower - thresholds, upper + thresholds


CQR_SCORES = {'quantile': CQRScore(nonnegative=True),
              'quantile_standard': CQRScore(nonnegative=False)}
