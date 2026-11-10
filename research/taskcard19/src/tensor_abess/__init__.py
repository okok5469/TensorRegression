"""Tensor-ABESS project: vector and tensor best-subset selection."""

from tensor_abess.group import group_bess_fix
from tensor_abess.metrics import evaluate_support
from tensor_abess.tensor import tensor_abess, tensor_bcd
from tensor_abess.vector.abess import abess, bess_fix
from tensor_abess.vector.sic import compute_sic


def run_all_baselines(*args, **kwargs):
    """Lazily import optional baseline dependencies when requested."""
    from tensor_abess.baselines import run_all_baselines as _run_all_baselines

    return _run_all_baselines(*args, **kwargs)

__all__ = [
    "abess",
    "bess_fix",
    "compute_sic",
    "evaluate_support",
    "group_bess_fix",
    "run_all_baselines",
    "tensor_abess",
    "tensor_bcd",
]
