#!/bin/bash -l
#SBATCH -J v52_search_final_full
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh
#SBATCH --array=0-699%64
#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/v52_search_final_full_logs/%A_%a.out
#SBATCH --error=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/v52_search_final_full_logs/%A_%a.err
module load anaconda3 || exit $?
conda activate myenv || exit $?
set -euo pipefail
task_repo="${V52_REMOTE_ROOT:-/public/home/xjh/xmx/tensor-sparse-regression-main}"
task_output="numerical_simulation/outputs_v52_search_final_full"
task_logs="numerical_simulation/v52_search_final_full_logs"
cd -- "$task_repo"
export PYTHONPATH="$task_repo/src:$task_repo"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
task_control="scripts/release_v52/release_control.py"
if [[ "${1:-run}" == prepare ]]; then
  mkdir -p -- "$task_logs"
fi
python -B "$task_control" verify --root "$task_repo"
case "${1:-run}" in
  prepare)
    python -B -m numerical_simulation.known_s_cluster.run plan --profile full --n-reps 50 \
      --methods adaptive oracle lasso glasso --output-dir "$task_output"
    python -B "$task_control" plan --root "$task_repo"
    python -B -m numerical_simulation.known_s_cluster.run status --output-dir "$task_output"
    ;;
  run)
    : "${SLURM_JOB_ID:?submit through sbatch}"
    : "${SLURM_ARRAY_JOB_ID:?submit as an array}"
    : "${SLURM_ARRAY_TASK_ID:?missing array task ID}"
    : "${SLURM_ARRAY_TASK_COUNT:?missing array task count}"
    [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] && (( SLURM_ARRAY_TASK_ID < 700 ))
    if [[ "$SLURM_ARRAY_TASK_COUNT" == 1 && "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
      python -B "$task_control" plan --root "$task_repo"
    else
      python -B "$task_control" gate --root "$task_repo"
    fi
    mkdir -p -- "$task_output/slurm"
    task_log="$task_output/slurm/task_$(printf '%06d' "$SLURM_ARRAY_TASK_ID")_${SLURM_JOB_ID}_$(date +%s%N).log"
    python -B -m numerical_simulation.known_s_cluster.run run --output-dir "$task_output" \
      --task-id "$SLURM_ARRAY_TASK_ID" --resume | tee "$task_log"
    task_outcome=$(awk -F' ' '/^task_id=.* (completed|skipped)$/ {print $2}' "$task_log")
    [[ "$task_outcome" == completed || "$task_outcome" == skipped ]]
    python -B "$task_control" job-record --root "$task_repo" --log "$task_log" --outcome "$task_outcome"
    ;;
  status)
    python -B "$task_control" plan --root "$task_repo"
    python -B -m numerical_simulation.known_s_cluster.run status --output-dir "$task_output"
    ;;
  merge)
    python -B "$task_control" complete --root "$task_repo"
    python -B -m numerical_simulation.known_s_cluster.run merge --output-dir "$task_output"
    ;;
  *) echo "Usage: bash run_v52_search_final_full.sh prepare|status|merge; sbatch --array=0 run_v52_search_final_full.sh; full sbatch requires preflight_gate.sh" >&2; exit 2 ;;
esac
