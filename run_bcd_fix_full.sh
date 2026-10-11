#!/bin/bash -l
#SBATCH -J v51_bcd_full
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh
#SBATCH --array=0-699%64
#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/v51_bcd_full_logs/%A_%a.out
#SBATCH --error=/public/home/xjh/xmx/tensor-sparse-regression-main/numerical_simulation/v51_bcd_full_logs/%A_%a.err

module load anaconda3
source activate myenv
set -euo pipefail
task_repo="/public/home/xjh/xmx/tensor-sparse-regression-main"
task_output="numerical_simulation/outputs_v51_bcd_fix_full"
cd -- "$task_repo"
export PYTHONPATH="$task_repo/src"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c 'import platform,numpy,scipy; actual=(platform.python_version(),numpy.__version__,scipy.__version__); expected=("3.9.25","2.0.2","1.13.1"); assert actual==expected,(actual,expected); print("existing environment:",actual)'
python -c 'from pathlib import Path; import hashlib; p=Path("src/tensor_abess/tensor/bcd.py"); h=hashlib.sha256(p.read_text(encoding="utf-8-sig").replace("\r\n","\n").encode("utf-8")).hexdigest(); expected="4966b563a0c3d8c0b233141b1f1d136a0dce3b4af3c02331cfe191af6310816c"; assert h==expected,(h,expected); print("v5.1-bcd-fix verified")'
case "${1:-run}" in
  prepare)
    mkdir -p numerical_simulation/v51_bcd_full_logs
    python -m numerical_simulation.known_s_cluster.run plan \
      --profile full --n-reps 50 --methods adaptive oracle lasso glasso \
      --output-dir "$task_output"
    python -m numerical_simulation.known_s_cluster.run status --output-dir "$task_output"
    ;;
  run)
    : "${SLURM_ARRAY_TASK_ID:?submit with sbatch after prepare}"
    python -m numerical_simulation.known_s_cluster.run run \
      --output-dir "$task_output" --task-id "$SLURM_ARRAY_TASK_ID" --resume
    ;;
  merge)
    python -m numerical_simulation.known_s_cluster.run status --output-dir "$task_output"
    python -m numerical_simulation.known_s_cluster.run merge --output-dir "$task_output"
    ;;
  *) echo "Usage: bash run_bcd_fix_full.sh prepare | sbatch run_bcd_fix_full.sh | bash run_bcd_fix_full.sh merge" >&2; exit 2 ;;
esac
