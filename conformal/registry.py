"""Calibration algorithms and explicit, independently named score variants."""
import numpy as np
from dataclasses import dataclass
from .methods.cqr import CQR_SCORES

DEFAULT_METHODS = ('Vanilla','SCP-ABS','SCP-NORM','SCP-CQR','SCP-PIT','CalLCP','RLCP','SLCP','PCP','RCP')


@dataclass(frozen=True)
class MethodSpec:
    """Separate the calibration algorithm from its score configuration."""
    algorithm: str
    score: str | None

    @property
    def needs_auxiliary_model(self):
        return self.algorithm == 'RCP'

    @property
    def uses_distances(self):
        return self.algorithm in ('CalLCP', 'RLCP', 'SLCP')

    @property
    def cqr(self):
        return CQR_SCORES.get(self.score)


_BASE_SCORES = {'Vanilla': None, 'SCP-ABS': 'absolute',
                'SCP-NORM': 'variance_normalized', 'SCP-PIT': 'pit', 'PCP': 'absolute'}
METHOD_SPECS = {name: MethodSpec(name, _BASE_SCORES.get(name, 'quantile'))
                for name in DEFAULT_METHODS}
STANDARD_METHODS = tuple(name + '-standard' for name, spec in METHOD_SPECS.items()
                         if spec.cqr is not None)
METHOD_SPECS.update({name: MethodSpec(name.removesuffix('-standard'), 'quantile_standard')
                     for name in STANDARD_METHODS})
METHODS = tuple(METHOD_SPECS)


def default_methods(base_model):
    return ('Vanilla', 'SCP-CQR', 'SCP-CQR-standard') if base_model == 'bart' else DEFAULT_METHODS


def method_spec(name):
    try:
        return METHOD_SPECS[name]
    except KeyError:
        raise ValueError(f'Unknown conformal method: {name}') from None


def make_method(name, model, bundle, *, device='cpu', score_alpha=0.1):
    from .allocation import uses_pool
    spec = method_spec(name)
    name = spec.algorithm
    if name == 'Vanilla':
        raise ValueError(name)
    from .methods.split_conformal import ConformalPredictor
    from .methods.lcp import CalLCP
    from .methods.rlcp import RLCP
    from .methods.slcp import SLCP
    from .methods.pcp import PosteriorConformalPredictor
    from .methods.rcp import RCP
    cal, train = bundle['cal'], bundle['train']
    cal_keys, cal_y, cal_X = model.query('cal'), cal['y'], cal['X']
    if uses_pool(name, bundle):
        cal_keys = np.concatenate([model.query('train'), cal_keys])
        cal_y = np.concatenate([train['y'], cal_y])
        cal_X = np.concatenate([train['X'], cal_X])
    common = dict(base_model=model, X_cal=cal_keys, y_cal=cal_y, device=device)
    train_args = dict(X_train=model.query('train'), y_train=train['y'])
    distances = dict(X_train_dist=train['X'], X_cal_dist=cal_X)
    bandwidth_keys = model.query('train')
    if uses_pool(name, bundle) and name in ('CalLCP','RLCP'):
        # Bandwidth sees development features only, never labels or test rows.
        bandwidth_keys = cal_keys
        distances['X_train_dist'] = cal_X
    if name.startswith('SCP-'):
        return ConformalPredictor(**common,score=spec.score,score_alpha=score_alpha)
    if name == 'CalLCP':
        return CalLCP(**common,X_train=bandwidth_keys,**distances,score=spec.score,score_alpha=score_alpha,target_neff=100)
    if name == 'RLCP':
        return RLCP(**common,X_train=bandwidth_keys,**distances,score=spec.score,score_alpha=score_alpha,target_neff=100,random_state=bundle['seed'])
    if name == 'SLCP':
        return SLCP(**common,**train_args,**distances,score=spec.score,score_alpha=score_alpha,kernel='gaussian')
    if name == 'PCP':
        return PosteriorConformalPredictor(**common,**train_args,random_state=bundle['seed'],score='absolute')
    return RCP(**common,**train_args,score=spec.score,score_alpha=score_alpha,
               adjustment='difference',random_state=bundle['seed'])
