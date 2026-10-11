#!/bin/bash -l
#SBATCH -J known_s_full
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh
#SBATCH --array=0-699%64
#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/known_s_cluster_logs/%A_%a.out
#SBATCH --error=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/known_s_cluster_logs/%A_%a.err

module load anaconda3
source /public/home/xjh/xmx/.conda/etc/profile.d/conda.sh
conda activate myenv
set -euo pipefail

task_repo="/public/home/xjh/xmx/tensor-sparse-regression-main"
task_output="numerical_simulation/outputs_known_s_full_50"
task_logs="numerical_simulation/known_s_cluster_logs"

cd -- "$task_repo"
export PYTHONPATH="$task_repo/src"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -c 'import platform, numpy, scipy; expected=("3.9.25", "2.0.2", "1.13.1"); actual=(platform.python_version(), numpy.__version__, scipy.__version__); assert actual == expected, f"environment mismatch: expected={expected}, actual={actual}"; print("environment:", actual)'

case "${1:-run}" in
  prepare)
    mkdir -p -- "$task_logs"
    python -m numerical_simulation.known_s_cluster.run plan \
      --profile full \
      --n-reps 50 \
      --methods fixed adaptive oracle \
      --output-dir "$task_output"
    python -m numerical_simulation.known_s_cluster.run status \
      --output-dir "$task_output"
    ;;
  run)
    : "${SLURM_ARRAY_TASK_ID:?submit this mode with sbatch after prepare}"
    python -m numerical_simulation.known_s_cluster.run run \
      --output-dir "$task_output" \
      --task-id "$SLURM_ARRAY_TASK_ID" \
      --resume
    ;;
  merge)
    python -m numerical_simulation.known_s_cluster.run status \
      --output-dir "$task_output"
    python -m numerical_simulation.known_s_cluster.run merge \
      --output-dir "$task_output"
    ;;
  *)
    echo "Usage: bash run_known_s_full.sh prepare | sbatch run_known_s_full.sh | bash run_known_s_full.sh merge" >&2
    exit 2
    ;;
esac
