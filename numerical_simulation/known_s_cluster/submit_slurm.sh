#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch [Slurm resource options] $0 OUTPUT_DIR" >&2
  exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?submit this script with sbatch --array}"

task_repo_root="${SLURM_SUBMIT_DIR:?run sbatch from the project root}"
cd -- "$task_repo_root"
test -f numerical_simulation/known_s_cluster/run.py
task_python="${KNOWN_S_PYTHON:-python}"
export PYTHONPATH="$task_repo_root/src"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$task_python" -m numerical_simulation.known_s_cluster.run run \
  --output-dir "$1" --task-id "$SLURM_ARRAY_TASK_ID" --resume
