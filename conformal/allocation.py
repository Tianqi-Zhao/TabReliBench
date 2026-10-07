"""Method-specific use of a common held-out development pool."""
import hashlib
import json
import numpy as np
from .registry import method_spec

EQUAL_BUDGET_PROTOCOL = 'cached_holdout_equal_budget50_test50_v1'
POOLED_CALIBRATION = {'SCP-ABS', 'SCP-NORM', 'SCP-CQR', 'SCP-PIT', 'CalLCP', 'RLCP'}


def uses_pool(name, bundle):
    return bundle['protocol'] == EQUAL_BUDGET_PROTOCOL and method_spec(name).algorithm in POOLED_CALIBRATION


def usage(name, bundle):
    name = method_spec(name).algorithm
    aux, cal = list(bundle['train']['row_ids']), list(bundle['cal']['row_ids'])
    available = aux + cal
    fit = []
    if name == 'Vanilla':
        cal = []
    elif uses_pool(name, bundle):
        cal = available
    if name in ('RCP', 'SLCP', 'PCP'):
        fit = aux
    def fingerprint(rows):
        return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return dict(available_label_count=len(available), auxiliary_fit_count=len(fit),
                calibration_count=len(cal), test_count=len(bundle['test']['y']),
                available_rows_sha256=fingerprint(available),
                auxiliary_fit_rows_sha256=fingerprint(fit), calibration_rows_sha256=fingerprint(cal),
                test_rows_sha256=fingerprint(list(bundle['test']['row_ids'])),
                bandwidth_rule=('pooled_features_only_target_neff100' if uses_pool(name,bundle)
                                and name in ('CalLCP','RLCP') else None))
