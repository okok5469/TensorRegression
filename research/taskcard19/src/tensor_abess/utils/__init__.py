"""Utility helpers with lazy exports to avoid circular imports."""

from __future__ import annotations

__all__ = [
    "TensorInitResult",
    "ValidationReport",
    "ValidationSuiteResult",
    "generate_validation_report",
    "initial_support",
    "run_tensor_abess_monte_carlo",
    "run_tensor_abess_validation_suite",
    "tensor_init_glasso",
]


def __getattr__(name: str):
    if name in {"TensorInitResult", "initial_support", "tensor_init_glasso"}:
        from tensor_abess.utils.init import TensorInitResult, initial_support, tensor_init_glasso

        return {
            "TensorInitResult": TensorInitResult,
            "initial_support": initial_support,
            "tensor_init_glasso": tensor_init_glasso,
        }[name]
    if name in {
        "ValidationReport",
        "ValidationSuiteResult",
        "generate_validation_report",
        "run_tensor_abess_monte_carlo",
        "run_tensor_abess_validation_suite",
    }:
        from tensor_abess.utils.validation import (
            ValidationReport,
            ValidationSuiteResult,
            generate_validation_report,
            run_tensor_abess_monte_carlo,
            run_tensor_abess_validation_suite,
        )

        return {
            "ValidationReport": ValidationReport,
            "ValidationSuiteResult": ValidationSuiteResult,
            "generate_validation_report": generate_validation_report,
            "run_tensor_abess_monte_carlo": run_tensor_abess_monte_carlo,
            "run_tensor_abess_validation_suite": run_tensor_abess_validation_suite,
        }[name]
    raise AttributeError(name)
