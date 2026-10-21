#!/bin/bash -l
#SBATCH -J msic_pair_task12
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh
#SBATCH --chdir=/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle
#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/numerical_simulation/msic_pair_task12_logs/%A_%a.out
#SBATCH --error=/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle/numerical_simulation/msic_pair_task12_logs/%A_%a.err
task_repo="${TASK12_ROOT:-/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle}"
source "$task_repo/research/taskcard12/cluster/environment.sh"
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_ARRAY_JOB_ID:?Submit an array}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task ID}"
: "${SLURM_ARRAY_TASK_COUNT:?Missing array task count}"
task_phase="${1:?Pass smoke or formal}"
case "$task_phase" in smoke|formal) ;; *) echo 'Phase must be smoke or formal' >&2; exit 2 ;; esac
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || exit 2
trap 'task_code=$?; echo "END phase=$task_phase task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID exit=$task_code"; exit "$task_code"' EXIT
echo "START phase=$task_phase task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID host=$(hostname)"
python -u -B -m research.taskcard12.simulation worker --output "$task_output" --phase "$task_phase" --task-id "$SLURM_ARRAY_TASK_ID"
