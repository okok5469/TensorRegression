"""Revised Tensor-ABESS entrypoint following the 0720 specification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensor_abess.tensor.bcd import TensorBcdAdaptiveResult, tensor_bcd_adaptive
from tensor_abess.utils.init import TensorInitResult, tensor_init_glasso


@dataclass
class TensorAbessResult:
    """Output of revised Tensor-ABESS with GLasso initialization and adaptive BCD."""

    B: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    s1: int
    s2: int
    s3: int
    s_max: tuple[int, int, int]
    loss: float
    rss: float
    sic: float
    loss_history: list[float] = field(default_factory=list)
    sic_history: list[float] = field(default_factory=list)
    init_supports: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    init_result: TensorInitResult | None = None
    bcd_result: TensorBcdAdaptiveResult | None = None


def tensor_abess(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k_max: int | None = None,
    tau: float | None = None,
    eps: float = 1e-8,
    c: float = 2.0,
    criterion: str = "cv",
    n_lambdas: int = 20,
    cv_folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-6,
    max_cycles: int = 30,
    m_max: int = 15,
) -> TensorAbessResult:
    """
    Revised Tensor-ABESS:
    1) Tensor-Init-GLasso for warm supports and sparsity caps
    2) Adaptive BCD with GroupBESS.Adaptive on each mode
    3) Final OLS already embedded in the adaptive BCD result
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    init_res = tensor_init_glasso(
        X,
        y,
        c=c,
        criterion=criterion,
        n_lambdas=n_lambdas,
        cv_folds=cv_folds,
        max_iter=max_iter,
        tol=tol,
    )
    bcd_res = tensor_bcd_adaptive(
        X,
        y,
        s1_max=init_res.s1_max,
        s2_max=init_res.s2_max,
        s3_max=init_res.s3_max,
        S1_init=init_res.S1_init,
        S2_init=init_res.S2_init,
        S3_init=init_res.S3_init,
        max_cycles=max_cycles,
        m_max=m_max,
        k_max=k_max,
        tau=tau,
        eps=eps,
    )
    return TensorAbessResult(
        B=bcd_res.B,
        S1=bcd_res.S1,
        S2=bcd_res.S2,
        S3=bcd_res.S3,
        s1=bcd_res.s1,
        s2=bcd_res.s2,
        s3=bcd_res.s3,
        s_max=(init_res.s1_max, init_res.s2_max, init_res.s3_max),
        loss=bcd_res.loss,
        rss=bcd_res.rss,
        sic=bcd_res.sic,
        loss_history=bcd_res.loss_history,
        sic_history=bcd_res.sic_history,
        init_supports=(init_res.S1_init, init_res.S2_init, init_res.S3_init),
        init_result=init_res,
        bcd_result=bcd_res,
    )
