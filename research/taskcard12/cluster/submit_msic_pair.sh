#!/bin/bash -l
task_repo="${TASK12_ROOT:-/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle}"
source "$task_repo/research/taskcard12/cluster/environment.sh"
task_run="$task_repo/research/taskcard12/cluster/run_msic_pair.sh"
case "${1:-status}" in
  prepare)
    mkdir -p -- "$task_logs"
    python -B -m research.taskcard12.simulation prepare --output "$task_output" --protocol "$task_protocol"
    ;;
  smoke)
    python -B -m research.taskcard12.simulation verify --protocol "$task_protocol"
    python -B -m research.taskcard12.simulation status --output "$task_output"
    mkdir -p -- "$task_logs"
    sbatch --parsable --array=0-2%3 "$task_run" smoke
    ;;
  gate)
    python -B -m research.taskcard12.simulation gate --output "$task_output"
    ;;
  formal)
    python -B -m research.taskcard12.simulation check-gate --output "$task_output"
    mkdir -p -- "$task_logs"
    sbatch --parsable --array=0-379%64 "$task_run" formal
    ;;
  retry)
    python -B -m research.taskcard12.simulation check-gate --output "$task_output"
    task_ids="${2:?Pass only verified missing task IDs from status}"
    [[ "$task_ids" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Expected comma-separated missing IDs' >&2; exit 2; }
    mkdir -p -- "$task_logs"
    sbatch --parsable --array="${task_ids}%64" "$task_run" formal
    ;;
  status)
    python -B -m research.taskcard12.simulation status --output "$task_output"
    ;;
  *) echo 'Usage: submit_msic_pair.sh prepare|smoke|gate|formal|status|retry IDS' >&2; exit 2 ;;
esac
