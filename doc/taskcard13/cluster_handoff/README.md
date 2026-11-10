# Apple task card 13: minimal cluster upload

Upload `doc/taskcard13/cluster_handoff/apple_task13_cluster.zip` to `/public/home/xjh/xmx/tensor-sparse-regression-main/`. The archive contains only this experiment's frozen execution sources, manifest, configuration, compact data, sample mapping, contract, splits, and scripts. It excludes raw MAT/CSV, Windows runtime, and old results.

Extract while preserving archive-relative paths. Create `empirical_analysis/spectrofood_apple/cluster/logs/` before submission and check dependencies using existing `myenv`. Scikit-learn must be exactly **1.5.2** for local saved-model replay; another version produces `BLOCKED` and requires a matching isolated environment. No existing environment is installed into or upgraded. NumPy and SciPy retain their actual existing versions. Full commands and failure handling are in the archive's `empirical_analysis/spectrofood_apple/cluster/README.md`.

Submit with this single absolute command:

```bash
sbatch /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/run_array.sh
```

The array is 0 through 104 with concurrency 2. After all 105 tasks finish successfully, collect and verify outputs:

```bash
bash -l /public/home/xjh/xmx/tensor-sparse-regression-main/empirical_analysis/spectrofood_apple/cluster/merge.sh
```

Return the entire `empirical_analysis/spectrofood_apple/outputs/apple_v1/` directory and this array's logs. All model artifacts, scalers, completion markers, CV records, and diagnostics are required. Local processing generates final figures and report.

Status: `PENDING_CLUSTER_RESULTS`. Local preparation checks do not replace cluster dependency checks or full experiment acceptance. No remote job has been submitted. Keep frozen Windows raw-data paths unchanged; workers read only the compact cache.
