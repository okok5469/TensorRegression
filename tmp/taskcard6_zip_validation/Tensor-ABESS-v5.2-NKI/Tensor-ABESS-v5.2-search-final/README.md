# Tensor-ABESS v5.2-search-final

Complete approved D2 source release: all 32 Python core files, not a single-file patch.
The HCP and NKI ZIPs are byte-identical copies. No participant data, simulation
results, virtual environment, or research-only solver is included.

Runtime dependencies: NumPy and SciPy. Use the project's existing validated
environment; do not copy a virtual environment or silently upgrade libraries.
Reference cluster: Python 3.9.25, NumPy 2.0.2, SciPy 1.13.1.
The ZIP validation record reports the environment actually used for its checks.
This source package uses PYTHONPATH; it does not claim to be an installable wheel.

From this extracted directory, using your existing Python:

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
python -B scripts/check_version.py --import-check
python -B examples/smoke_test.py
python -B -m unittest discover -s tests -q
```

```bash
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B scripts/check_version.py --import-check
python -B examples/smoke_test.py
python -B -m unittest discover -s tests -q
```

Real-data interface (after project-specific input/preprocessing/CV audit):

```python
from tensor_abess.tensor.abess import tensor_abess
result = tensor_abess(X_train, y_train, target_sparsity=None,
                      **your_unchanged_training_configuration)
prediction = __import__('numpy').einsum('nijk,ijk->n', X_test, result.B)
```

Prediction above is for already prepared tensors/responses. Preserve your original
fold-specific confound adjustment, scaling, intercept and inverse transforms.
Do not substitute this line for the project's full preprocessing pipeline.

Keep HCP/NKI subjects, response, MRI order, confounds, validation splits and training
configuration unchanged. Refit preprocessing within training folds and rerun
Tensor-ABESS support search, model choice, OLS and prediction. Never reuse old
coefficients, supports, q or model caches under the new version identity.

Complete recovery improved in the prescribed simulations, while mean prediction
error did not improve overall. Better empirical predictions are not guaranteed.
Return version/input/config/split/environment identities, complete OOF predictions,
fold supports/q, training/test errors and all failure/completion records.

See INSTALL_AND_UPGRADE.md before replacing an existing package.
