#!/bin/bash -l
# Shared initialization; sourced by run_array.sh and merge.sh.
module load anaconda3 || exit $?
conda activate myenv || exit $?
set -euo pipefail
cd -- /public/home/xjh/xmx/tensor-sparse-regression-main
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c 'import sys, numpy, scipy, sklearn, joblib, threadpoolctl; from sklearn.linear_model import Lasso, Ridge; from sklearn.cross_decomposition import PLSRegression; print("Python",sys.version); print("NumPy",numpy.__version__,"SciPy",scipy.__version__,"scikit-learn",sklearn.__version__,flush=True); assert sklearn.__version__ == "1.5.2", "BLOCKED: scikit-learn 1.5.2 is required for saved-model replay; provide a matching isolated environment without modifying existing myenv"' || { printf '%s\n' 'BLOCKED: missing or incompatible model dependencies; scikit-learn must be 1.5.2. A matching isolated environment is required; no installation or upgrade was attempted.' >&2; exit 2; }
