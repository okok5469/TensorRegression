"""Baseline methods for comparison."""

from tensor_abess.baselines.base import BaselineResult
from tensor_abess.baselines.cp import cp_regression
from tensor_abess.baselines.group_lasso import group_lasso_mode1
from tensor_abess.baselines.lasso import lasso_elementwise
from tensor_abess.baselines.tucker import tucker_regression

__all__ = [
    "BaselineResult",
    "cp_regression",
    "group_lasso_mode1",
    "lasso_elementwise",
    "tucker_regression",
]


def run_all_baselines(
    X,
    y,
    *,
    cp_rank: int = 3,
    tucker_ranks: tuple[int, int, int] = (3, 2, 2),
    random_state: int = 0,
) -> list[BaselineResult]:
    """Run all Phase-C baselines on the same data."""
    return [
        lasso_elementwise(X, y, random_state=random_state),
        group_lasso_mode1(X, y, random_state=random_state),
        cp_regression(X, y, rank=cp_rank, random_state=random_state),
        tucker_regression(X, y, ranks=tucker_ranks),
    ]
