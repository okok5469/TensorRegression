#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
action="${1:-prepare}"
dataset="${2:-lemon}"
case "$action" in prepare|run) ;; *) echo "Expected prepare or run" >&2; exit 2 ;; esac
case "$dataset" in lemon|hcp_ya|nki_rs) ;; *) echo "Unknown dataset" >&2; exit 2 ;; esac
"${PYTHON:-python}" -B -m research.taskcard12.empirical "$action" --config "research/taskcard12/empirical_configs/$dataset.json"
