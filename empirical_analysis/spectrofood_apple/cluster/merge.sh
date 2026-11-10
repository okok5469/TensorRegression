#!/bin/bash -l
# Run only after all 105 array tasks have completed successfully.
source /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/environment.sh
python -m empirical_analysis.spectrofood_apple.run --config empirical_analysis/spectrofood_apple/config.yaml --stage collect
