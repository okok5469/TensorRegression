# tensor-sparse-regression

Tensor best-subset selection via splicing and BCD for sparse tensor regression (2026 innovation project).

## Overview

This repository implements **Tensor-ABESS**: an extension of the Adaptive Best Subset Selection (ABESS) algorithm ([Zhu et al., 2020](https://arxiv.org/abs/1911.03804)) to three-way tensor regression with **rectangular support**.

**Model:**

\[
y_i = \langle X_i, B \rangle + \varepsilon_i, \quad X_i, B \in \mathbb{R}^{p_1 \times p_2 \times p_3}
\]

The coefficient tensor \(B\) is nonzero only on \(S_1 \times S_2 \times S_3\).

**Algorithm stack:**

1. Vector ABESS (splicing + SIC)
2. Group splicing (fix two modes, select the third)
3. BCD outer loop over modes
4. Tensor SIC to select \((s_1, s_2, s_3)\)

## Quick start

```bash
conda env create -f environment.yml
conda activate tensor-abess
pip install -e .
```

```bash
# Vector ABESS vs official abess package
python examples/01_compare_abess.py

# Tensor BCD on synthetic 10×5×8 data
python examples/02_tensor_bcd_demo.py

# Phase C benchmark (fixed seed)
python scripts/run_phase_c_benchmark.py --seed 2026

# Run tests
pytest tests/ -v
```

## Project structure

```
src/tensor_abess/
├── vector/       # Vector ABESS (BESS.Fix + SIC)
├── group/        # Group-level splicing
├── tensor/       # BCD + tensor_abess + tensor SIC
├── baselines/    # LASSO, group LASSO, CP, Tucker
├── metrics/      # Support recovery metrics
└── utils/        # Synthetic data + initialization

tests/            # 18 unit tests
examples/         # Demo scripts
docs/             # Progress report (Chinese)
scripts/          # Reproducible benchmark
```

## Python API

```python
from tensor_abess import tensor_abess, tensor_bcd, evaluate_support
from tensor_abess.utils.data import make_tensor_data

data = make_tensor_data(n=200, p1=10, p2=5, p3=8, s1=3, s2=2, s3=2)
result = tensor_abess(data["X"], data["y"],
                      s1_grid=[2, 3, 4], s2_grid=[1, 2, 3], s3_grid=[1, 2, 3])
```

## Status

- [x] Phase B: vector ABESS + group splicing + minimal BCD
- [x] Phase C: full Tensor-ABESS + baselines + tests (18/18 passing)
- [ ] Phase D: large-scale simulation
- [ ] Phase E: EEG pilot study

See [docs/阶段BC代码线进展报告.md](docs/阶段BC代码线进展报告.md) for a detailed progress report.

## References

- Zhu, J., Wen, C., Zhu, J., Zhang, H., & Wang, X. (2020). A polynomial algorithm for best-subset selection problem. *PNAS*.
- [abess](https://github.com/abess-team/abess) — official ABESS library

## License

TBD.
