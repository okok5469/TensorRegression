#!/bin/bash -l
#SBATCH --job-name=apple-task13
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh
#SBATCH --array=0-104%2
#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/logs/%A_%a.out
#SBATCH --error=/public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/logs/%A_%a.err

source /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/environment.sh
: "${SLURM_ARRAY_TASK_ID:?This script requires its Slurm array task index}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || { printf '%s\n' 'BLOCKED: invalid array index' >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID <= 104 )) || { printf '%s\n' 'BLOCKED: array index must be 0..104' >&2; exit 2; }
python -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage one --task-index "$SLURM_ARRAY_TASK_ID"
