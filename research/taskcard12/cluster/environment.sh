#!/bin/bash -l
# Source this file; site initialization must precede nounset.
module load anaconda3 || exit $?
conda activate myenv || exit $?
set -euo pipefail
task_repo="${TASK12_ROOT:-/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle}"
cd -- "$task_repo"
export PYTHONPATH="$task_repo"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
task_output="$task_repo/numerical_simulation/outputs_msic_pair_task12"
task_logs="$task_repo/numerical_simulation/msic_pair_task12_logs"
task_protocol="$task_repo/doc/taskcard12/CLUSTER_PROTOCOL.json"
