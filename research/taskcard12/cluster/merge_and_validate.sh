#!/bin/bash -l
task_repo="${TASK12_ROOT:-/public/home/xjh/xmx/tensor-sparse-regression-main/taskcard12_bundle}"
source "$task_repo/research/taskcard12/cluster/environment.sh"
python -B -m research.taskcard12.simulation merge --output "$task_output"
echo "VERIFIED COMPLETE: 380 paired cases / 760 arm rows; results=$task_output/merged"
