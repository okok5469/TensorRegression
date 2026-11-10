"""Tensor regression public API with lazy imports to avoid cycles."""

from __future__ import annotations

__all__ = [
    "TensorAbessResult",
    "TensorBcdAdaptiveResult",
    "TensorBcdResult",
    "build_mode_design",
    "build_mode_design_subset",
    "compute_mode_sic_from_rss",
    "compute_tensor_sic",
    "compute_tensor_sic_from_rss",
    "tensor_abess",
    "tensor_bcd",
    "tensor_bcd_adaptive",
    "tensor_inner",
    "tensor_loss",
]


def __getattr__(name: str):
    if name in {"TensorAbessResult", "tensor_abess"}:
        from tensor_abess.tensor.abess import TensorAbessResult, tensor_abess

        return {"TensorAbessResult": TensorAbessResult, "tensor_abess": tensor_abess}[name]
    if name in {"TensorBcdAdaptiveResult", "TensorBcdResult", "tensor_bcd", "tensor_bcd_adaptive"}:
        from tensor_abess.tensor.bcd import (
            TensorBcdAdaptiveResult,
            TensorBcdResult,
            tensor_bcd,
            tensor_bcd_adaptive,
        )

        return {
            "TensorBcdAdaptiveResult": TensorBcdAdaptiveResult,
            "TensorBcdResult": TensorBcdResult,
            "tensor_bcd": tensor_bcd,
            "tensor_bcd_adaptive": tensor_bcd_adaptive,
        }[name]
    if name in {"tensor_inner", "tensor_loss"}:
        from tensor_abess.tensor.ops import tensor_inner, tensor_loss

        return {"tensor_inner": tensor_inner, "tensor_loss": tensor_loss}[name]
    if name in {"build_mode_design", "build_mode_design_subset"}:
        from tensor_abess.tensor.reduction import build_mode_design, build_mode_design_subset

        return {
            "build_mode_design": build_mode_design,
            "build_mode_design_subset": build_mode_design_subset,
        }[name]
    if name in {"compute_mode_sic_from_rss", "compute_tensor_sic", "compute_tensor_sic_from_rss"}:
        from tensor_abess.tensor.sic import (
            compute_mode_sic_from_rss,
            compute_tensor_sic,
            compute_tensor_sic_from_rss,
        )

        return {
            "compute_mode_sic_from_rss": compute_mode_sic_from_rss,
            "compute_tensor_sic": compute_tensor_sic,
            "compute_tensor_sic_from_rss": compute_tensor_sic_from_rss,
        }[name]
    raise AttributeError(name)
