"""Baseline result container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    """Common output for baseline methods."""

    name: str
    B: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    loss: float
