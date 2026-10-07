"""Unmodified upstream PCP with a Python integer-seed compatibility shim."""
import random
import numpy as np
from . import utils

COMMIT = "7a4b33ee852b95bd65f8fe5486b4d7e6ded24bad"
IMPLEMENTATION = "official_pcp_" + COMMIT


class _RandomCompatibility:
    # Python 3.11+ rejects numpy integer seeds. Preserve their numeric value
    # and the standard-library RNG/choices algorithm used by upstream.
    @staticmethod
    def seed(value=None, version=2):
        return random.seed(int(value) if isinstance(value, np.integer) else value,
                           version=version)

    def __getattr__(self, name):
        return getattr(random, name)


utils.random = _RandomCompatibility()
