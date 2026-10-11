"""Synthetic tensor-regression data generation for numerical simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tensor_abess.tensor.ops import tensor_inner

SupportShape = Literal["continuous", "discrete"]
TensorAbessMode = Literal["fixed", "adaptive"]


@dataclass(frozen=True)
class TensorSimulationConfig:
    """Configuration for one tensor-regression simulation setting."""

    n_train: int = 200
    n_test: int = 500
    dims: tuple[int, int, int] = (32, 20, 25)
    sparsity: tuple[int, int, int] = (4, 3, 5)
    rho: tuple[float, float, float] = (0.3, 0.3, 0.3)
    snr: float = 2.0
    support_shape: SupportShape = "continuous"
    sigma: float = 1.0
    k_max: int = 5
    max_cycles: int = 10
    n_lambdas: int = 20
    cv_folds: int = 5
    max_iter: int = 500
    tol: float = 1e-6
    abess_init_criterion: str = "bic"
    abess_init_c: float = 3.0
    tensor_abess_mode: TensorAbessMode = "fixed"
    msic_refine_alpha: float = 1.5
    msic_refine_gamma: float = 1.0
    msic_refine_one_se_scale: float = 1.0
    random_state: int = 2026
    test_batch_size: int = 100


@dataclass(frozen=True)
class TensorSimulationDataset:
    """One realized training dataset plus the population objects behind it."""

    config: TensorSimulationConfig
    X_train: np.ndarray
    y_train: np.ndarray
    B_true: np.ndarray
    supports_true: tuple[np.ndarray, np.ndarray, np.ndarray]
    covariances: tuple[np.ndarray, np.ndarray, np.ndarray]
    cholesky_factors: tuple[np.ndarray, np.ndarray, np.ndarray]
    signal_variance: float


def make_ar1_covariance(size: int, rho: float) -> np.ndarray:
    """AR(1) covariance with entry rho^|a-b|."""
    idx = np.arange(size)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _sample_contiguous_support(p: int, s: int, rng: np.random.Generator) -> np.ndarray:
    if not (1 <= s <= p):
        raise ValueError(f"need 1 <= s <= p, got s={s}, p={p}")
    start = int(rng.integers(0, p - s + 1))
    return np.arange(start, start + s, dtype=int)


def _sample_discrete_support(p: int, s: int, rng: np.random.Generator) -> np.ndarray:
    if not (1 <= s <= p):
        raise ValueError(f"need 1 <= s <= p, got s={s}, p={p}")
    return np.sort(rng.choice(p, size=s, replace=False).astype(int))


def generate_supports(
    dims: tuple[int, int, int],
    sparsity: tuple[int, int, int],
    support_shape: SupportShape,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate rectangular supports, either contiguous or arbitrary."""
    if len(dims) != len(sparsity):
        raise ValueError("dims and sparsity must have the same length")
    sampler = _sample_contiguous_support if support_shape == "continuous" else _sample_discrete_support
    return tuple(sampler(p, s, rng) for p, s in zip(dims, sparsity))


def generate_base_coefficient_tensor(
    dims: tuple[int, int, int],
    supports: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an unscaled coefficient tensor with uniform magnitudes and random signs."""
    p1, p2, p3 = dims
    S1, S2, S3 = supports
    B = np.zeros((p1, p2, p3), dtype=float)
    magnitudes = rng.uniform(0.8, 1.2, size=(len(S1), len(S2), len(S3)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(len(S1), len(S2), len(S3)))
    B[np.ix_(S1, S2, S3)] = magnitudes * signs
    return B


def apply_mode_cholesky_factors(
    E: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    L3: np.ndarray,
) -> np.ndarray:
    """Apply separable covariance factors to i.i.d. Gaussian tensor noise."""
    X = np.einsum("ab,nbcd->nacd", L1, E)
    X = np.einsum("bc,nacd->nabd", L2, X)
    X = np.einsum("cd,nabd->nabc", L3, X)
    return X


def sample_tensor_predictors(
    n_samples: int,
    cholesky_factors: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample tensor covariates under a separable Gaussian covariance."""
    L1, L2, L3 = cholesky_factors
    E = rng.normal(size=(n_samples, L1.shape[0], L2.shape[0], L3.shape[0]))
    return apply_mode_cholesky_factors(E, L1, L2, L3)


def tensor_signal_variance(
    B: np.ndarray,
    Sigma1: np.ndarray,
    Sigma2: np.ndarray,
    Sigma3: np.ndarray,
) -> float:
    """Compute Var(<X, B>) under vec(X) ~ N(0, Sigma3 ⊗ Sigma2 ⊗ Sigma1)."""
    tmp = np.einsum("ab,bcd->acd", Sigma1, B)
    tmp = np.einsum("bc,acd->abd", Sigma2, tmp)
    tmp = np.einsum("cd,abd->abc", Sigma3, tmp)
    return float(np.sum(B * tmp))


def scale_tensor_to_snr(
    B0: np.ndarray,
    covariances: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_snr: float,
    sigma: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Scale an unscaled tensor so the signal-to-noise ratio equals target_snr."""
    signal_variance0 = tensor_signal_variance(B0, *covariances)
    if signal_variance0 <= 0:
        raise ValueError("unscaled signal variance must be positive")
    scale = sigma * np.sqrt(target_snr / signal_variance0)
    B = scale * B0
    return B, tensor_signal_variance(B, *covariances)


def generate_simulation_dataset(
    config: TensorSimulationConfig,
    *,
    seed: int | None = None,
) -> TensorSimulationDataset:
    """Generate one training dataset and its underlying population quantities."""
    seed = config.random_state if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    p1, p2, p3 = config.dims

    covariances = (
        make_ar1_covariance(p1, config.rho[0]),
        make_ar1_covariance(p2, config.rho[1]),
        make_ar1_covariance(p3, config.rho[2]),
    )
    cholesky_factors = tuple(np.linalg.cholesky(Sigma) for Sigma in covariances)
    supports = generate_supports(config.dims, config.sparsity, config.support_shape, rng)
    B0 = generate_base_coefficient_tensor(config.dims, supports, rng)
    B_true, signal_variance = scale_tensor_to_snr(B0, covariances, config.snr, sigma=config.sigma)

    X_train = sample_tensor_predictors(config.n_train, cholesky_factors, rng)
    signal = tensor_inner(X_train, B_true)
    y_train = signal + rng.normal(0.0, config.sigma, size=config.n_train)

    return TensorSimulationDataset(
        config=config,
        X_train=X_train,
        y_train=y_train,
        B_true=B_true,
        supports_true=supports,
        covariances=covariances,
        cholesky_factors=cholesky_factors,
        signal_variance=signal_variance,
    )


def iter_test_batches(
    dataset: TensorSimulationDataset,
    *,
    n_samples: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
):
    """Yield independent test batches without storing the whole test tensor set."""
    config = dataset.config
    remaining = config.n_test if n_samples is None else int(n_samples)
    batch_size = config.test_batch_size if batch_size is None else int(batch_size)
    rng = np.random.default_rng(config.random_state + 10_000 if seed is None else int(seed))

    while remaining > 0:
        current = min(batch_size, remaining)
        X_batch = sample_tensor_predictors(current, dataset.cholesky_factors, rng)
        signal_batch = tensor_inner(X_batch, dataset.B_true)
        y_batch = signal_batch + rng.normal(0.0, config.sigma, size=current)
        yield X_batch, y_batch, signal_batch
        remaining -= current


def total_dimension(dims: tuple[int, int, int]) -> int:
    """Total number of tensor coefficients."""
    p1, p2, p3 = dims
    return int(p1 * p2 * p3)
