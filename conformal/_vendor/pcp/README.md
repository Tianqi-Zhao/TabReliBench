# Official Posterior Conformal Prediction source

This directory contains code by Yao Zhang and Emmanuel J. Candès for [Posterior Conformal Prediction](https://arxiv.org/abs/2409.19712), from [yaozhang24/pcp](https://github.com/yaozhang24/pcp) at commit `7a4b33ee852b95bd65f8fe5486b4d7e6ded24bad`. `utils.py` is unmodified; [SOURCE.json](SOURCE.json) records its SHA-256.

The [benchmark adapter](../../methods/pcp.py) calls upstream `PCP.train` and `PCP.calibrate` with absolute residual scores. It preserves upstream numerical behavior, including matrix mutation, clustering, RNG resets and infinite thresholds. The benchmark supplies its own auxiliary/calibration split; it does not reproduce the upstream example's base-model training procedure.

Integration details:

- `__init__.py` converts NumPy integer seeds to Python integers for compatibility with `random.seed`, without changing their values.
- Calls use legacy NumPy MT19937 and restore the caller's NumPy/Python RNG states. Calls are serialized; unrelated threads must not consume these global RNGs concurrently.
- Test residuals passed upstream are dummy zeros used only for discarded coverage indicators. Coverage is evaluated separately from the resulting intervals.
- Each alpha starts a fresh official instance. Upstream prediction mutates internal matrices, so changing test-row order or chunking can change results.
- Saved parameters include the implementation identity and upstream commit. Resume rejects incompatible results.

See the [main README](../../../README.md#4-conformal-prediction) for the experiment commands.
