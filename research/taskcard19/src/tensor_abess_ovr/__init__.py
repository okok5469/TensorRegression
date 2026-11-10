"""Tensor-ABESS OvR-LS application; importing this module starts no numerical runtime."""

import os as _os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ[_thread_variable] = "1"
del _os, _thread_variable

__version__ = "0.1.0"
__all__ = ["TensorABESSOvR"]


def __getattr__(name):
    if name == "TensorABESSOvR":
        from .model import TensorABESSOvR
        return TensorABESSOvR
    raise AttributeError(name)
