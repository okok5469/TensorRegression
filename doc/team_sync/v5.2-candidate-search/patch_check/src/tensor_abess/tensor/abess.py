"""Revised Tensor-ABESS entrypoint following the 0720 specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from tensor_abess.tensor.bcd import (
    TensorBcdAdaptiveResult,
    TensorBcdResult,
    tensor_bcd,
    tensor_bcd_adaptive,
)
from tensor_abess.tensor.sic import compute_tensor_sic_from_rss
from tensor_abess.utils.init import TensorInitResult, _mode_slice_scores, tensor_init_glasso


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
    bcd_result: TensorBcdAdaptiveResult | TensorBcdResult | None = None


def _marginal_top_supports(
    X: np.ndarray,
    y: np.ndarray,
    sparsity: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warm start fixed-size BCD with the strongest marginal slices."""
    supports = []
    for mode, s in enumerate(sparsity, start=1):
        scores = _mode_slice_scores(X, y, mode)
        supports.append(np.sort(np.argsort(-scores)[: int(s)]))
    return tuple(supports)  # type: ignore[return-value]


def tensor_abess(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_sparsity: tuple[int, int, int] | None = None,
    k_max: int | None = None,
    tau: float | None = None,
    eps: float = 1e-8,
    c: float = 2.0,
    criterion: str = "cv",
    n_lambdas: int = 20,
    cv_folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-6,
    adaptive_criterion: Literal["sic", "msic"] = "msic",
    msic_refine_alpha: float = 1.5,
    msic_refine_gamma: float = 1.0,
    msic_refine_one_se_scale: float = 1.0,
    max_cycles: int = 30,
    m_max: int = 15,
    cap_policy: Literal["legacy", "candidate"] = "legacy",
    grid_policy: Literal["legacy", "full", "coarse_to_fine"] = "legacy",
    candidate_budget: int = 2000,
    fixed_candidate_cache: dict | None = None,
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
        cap_policy=cap_policy,
    )
    if target_sparsity is not None:
        s1, s2, s3 = (int(s) for s in target_sparsity)
        S1_init, S2_init, S3_init = _marginal_top_supports(X, y, (s1, s2, s3))
        bcd_fixed = tensor_bcd(
            X,
            y,
            s1,
            s2,
            s3,
            max_cycles=max_cycles,
            m_max=m_max,
            k_max=k_max,
            S1_init=S1_init,
            S2_init=S2_init,
            S3_init=S3_init,
        )
        n, p1, p2, p3 = X.shape
        rss = float(2.0 * n * bcd_fixed.loss)
        sic = compute_tensor_sic_from_rss(rss, s1, s2, s3, n, p1, p2, p3)
        return TensorAbessResult(
            B=bcd_fixed.B,
            S1=bcd_fixed.S1,
            S2=bcd_fixed.S2,
            S3=bcd_fixed.S3,
            s1=s1,
            s2=s2,
            s3=s3,
            s_max=(init_res.s1_max, init_res.s2_max, init_res.s3_max),
            loss=bcd_fixed.loss,
            rss=rss,
            sic=sic,
            loss_history=bcd_fixed.loss_history,
            sic_history=[sic],
            init_supports=(S1_init, S2_init, S3_init),
            init_result=init_res,
            bcd_result=bcd_fixed,
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
        information_criterion=adaptive_criterion,
        msic_refine_alpha=msic_refine_alpha,
        msic_refine_gamma=msic_refine_gamma,
        msic_refine_one_se_scale=msic_refine_one_se_scale,
        cap_policy=cap_policy,
        grid_policy=grid_policy,
        candidate_budget=candidate_budget,
        fixed_candidate_cache=fixed_candidate_cache,
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
