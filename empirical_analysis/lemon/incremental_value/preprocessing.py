"""Reuse the existing, explicit training-only nuisance and scaling contract."""
from ..confounds import FoldPreprocessor, fit_preprocessor, validate_covariates

__all__ = ['FoldPreprocessor', 'fit_preprocessor', 'validate_covariates']
