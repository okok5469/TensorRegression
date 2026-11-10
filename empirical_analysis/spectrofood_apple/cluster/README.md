# Apple task card 13: cluster continuation

Status: `PENDING_CLUSTER_RESULTS`. The original seven-model smoke passed prediction replay, with a projected cost of approximately 100,498 seconds above the 7,200-second local budget. The final frozen smoke record contains the updated estimate after the documented Lasso convergence correction. The full design remains three repeats, five folds, and seven models.

Site settings come from `CLUSTER_GUIDE.md`: Slurm partition `CMO_HPC`, account `xjh`, project `/public/home/xjh/xmx/tensor-sparse-regression-main`, `module load anaconda3`, and `conda activate myenv`. The array has 105 tasks (0 through 104), concurrency 2, and one CPU per task. Four numerical thread counts are 1; requeue is disabled. Unconfirmed memory and time limits are left to site defaults.

## Upload and initialize

Upload `doc/taskcard13/cluster_handoff/apple_task13_cluster.zip` into the cluster project root. Archive paths are relative to that root. The archive contains frozen execution sources, release manifest, configuration, compact `data.npz`, sample mapping, contract, splits, and scripts. Raw MAT/CSV, the Windows runtime, old experiments, and the rest of the repository are unnecessary.

Extract while preserving relative paths. If existing files differ, stop and reconcile their identities; do not overwrite old experiment results or edit the contract to bypass checks.

```bash
unzip /public/home/xjh/xmx/tensor-sparse-regression-main/apple_task13_cluster.zip -d /public/home/xjh/xmx/tensor-sparse-regression-main
```

Create the log directory before submission: Slurm opens logs before executing the script.

```bash
mkdir -p /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/logs
```

Check existing `myenv` dependencies without installing or upgrading packages:

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/environment.sh
```

Missing or incompatible imports for NumPy, SciPy, scikit-learn, joblib, threadpoolctl, or model classes produce `BLOCKED` and stop execution. Scikit-learn must be exactly **1.5.2**, matching local saved-model replay; any other version stops execution and requires a matching isolated environment. The scripts never install, upgrade, or modify existing `myenv`. NumPy and SciPy retain their actual existing Linux versions, recorded per task. Windows-fitted models cannot replace complete Linux fits.

## Submit the full array

Use this single absolute command after successful environment checks. Preparation of this handoff did not submit any remote job.

```bash
sbatch /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/run_array.sh
```

`task-index // 7` selects the outer split; `task-index % 7` selects M0 through M6 in frozen order. Each task runs `--stage one`, including full inner tuning and final outer-training fitting. Data, source, configuration, and split fingerprints must match the contract. Only complete results with matching fingerprints and actual environments may be reused. Failures retain logs; do not change parameters, seeds, grids, or data to produce replacement results.

Logs: `empirical_analysis/spectrofood_apple/cluster/logs/<ArrayJobID>_<TaskID>.out` and `.err`.

Results: `empirical_analysis/spectrofood_apple/outputs/apple_v1/tasks/r<repeat>_f<fold>_M<model-number>/`.

An empty queue does not establish successful completion.

## Collect after all 105 tasks finish

After confirming successful completion of every task, run:

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/merge.sh
```

This script only invokes `--stage collect`. It validates 105 completion markers, artifact hashes, frozen identities, held-out IDs and responses, training-fold scalers, and independent prediction reconstruction. It requires 720 OOF records per model, 5,040 in total. It writes `oof_predictions.csv`, `fold_metrics.csv`, `abess_fold_diagnostics.csv`, `repeat_metrics.csv`, `summary_metrics.csv`, and `OOF_REPLAY_VALIDATION.json`. Missing or failed tasks prevent full acceptance. It does not generate figures or the final report.

Return the entire `empirical_analysis/spectrofood_apple/outputs/apple_v1/` directory, including all 105 models, scalers, completion markers, CV records, and diagnostics, plus this array's `cluster/logs/`. Summary tables alone are insufficient. Preserve relative paths. Local processing will verify results and generate figures and the final report. Until results are returned and accepted, status remains `PENDING_CLUSTER_RESULTS`.

Keep original Windows raw-data paths in the frozen configuration unchanged. Stages `one` and `collect` read compact cache files and never open raw MAT/CSV. Do not run `audit`, `prepare`, or `freeze` on Linux; do not edit Windows paths.

Local Bash syntax and static contract checks are preparation checks. They do not establish Slurm submission, successful cluster dependency checks, or completion of 105 formal tasks.
