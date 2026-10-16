#!/bin/bash
set -euo pipefail
task_root="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$task_root/src:$task_root"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$task_root/scripts/release_v52/release_control.py" preflight --root "$task_root"
