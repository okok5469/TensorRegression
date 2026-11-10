#!/bin/bash -l
#SBATCH -J taskcard18
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh

module load anaconda3 || exit $?
conda activate myenv || exit $?
set -euo pipefail
task18_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Slurm executes a spool copy; submissions are bound to this payload's cwd.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  task18_root="${SLURM_SUBMIT_DIR:?missing payload submission directory}"
fi
[[ -f "$task18_root/research/taskcard18/cluster.py" ]]
cd -- "$task18_root"
export PYTHONPATH="$task18_root/runtime:$task18_root/src:$task18_root"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python -B -m research.taskcard18.cluster "$@"
