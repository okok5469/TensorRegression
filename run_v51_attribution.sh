#!/bin/bash -l
#SBATCH -J v51_attribution
#SBATCH -p CMO_HPC
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=xjh
#SBATCH --chdir=/public/home/xjh/xmx/tensor-sparse-regression-main
#SBATCH --output=/public/home/xjh/xmx/tensor-sparse-regression-main/v51-attribution-%j.out
#SBATCH --error=/public/home/xjh/xmx/tensor-sparse-regression-main/v51-attribution-%j.err

module load anaconda3 || exit $?
conda activate myenv || exit $?
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'Submit one job, without --array' >&2; exit 2; }
cd -- /public/home/xjh/xmx/tensor-sparse-regression-main
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Starting v5.1 attribution: job=$SLURM_JOB_ID host=$(hostname)"

python -u -B - <<'PY'
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def copy_plan(r, origin, target):
    content = origin.read_bytes()
    if target.exists():
        r.require(target.read_bytes() == content, 'Existing plan differs; refusing overwrite')
        return
    with target.open('xb') as handle:
        handle.write(content)


def completed_result(r, out, plan, task):
    task_id = task['task_id']
    path = out / 'tasks' / f'task_{task_id:06d}.json'
    npz = path.with_suffix('.npz')
    gate = out / 'gates' / path.name
    pilot = task_id in plan['gate_task_ids']
    expected = [path, npz] + ([gate] if pilot else [])
    if not any(p.exists() for p in expected):
        return False
    r.require(all(p.is_file() for p in expected),
              f'Partial task {task_id}; preserve and inspect: {[str(p) for p in expected if not p.is_file()]}')
    record = r.read(path)
    r.check_seal(record)
    identity = {'task_id': task_id, 'plan_id': plan['sha256'],
                'fit_version': plan['fit_version'],
                'core': plan['source_manifests']['v5.1-bcd-fix'],
                'environment': plan['expected_environment'],
                'input_fingerprints': task['expected_input_fingerprints'],
                'npz_sha256': r.sha(npz)}
    r.require(all(record.get(k) == v for k, v in identity.items()),
              f'Result identity or NPZ checksum differs: {path}')
    if pilot:
        saved_gate = r.read(gate)
        r.check_seal(saved_gate)
        r.require(saved_gate['task_id'] == task_id and saved_gate['plan_id'] == plan['sha256']
                  and saved_gate['result_sha256'] == record['sha256'], f'Pilot gate differs: {gate}')
    return True


def checked_preflight(r, out, plan):
    path = out / 'preflight_task_000000.json'
    if not path.exists():
        return False
    record = r.read(path)
    r.check_seal(record)
    task = next(t for t in plan['tasks'] if t['task_id'] == 0)
    expected = task['expected_input_fingerprints']
    identity = {'plan_id': plan['sha256'], 'task_id': 0, 'fit_executed': False,
                'expected': expected, 'actual': expected,
                'matches': dict.fromkeys(expected, True),
                'expected_environment': plan['expected_environment'],
                'actual_environment': plan['expected_environment'],
                'environment_matches': True, 'passed': True}
    r.require(all(record.get(k) == v for k, v in identity.items()),
              f'Existing preflight failed or differs; preserve and inspect: {path}')
    return True


def execute_workflow(r, repo, invoke):
    out = repo / 'numerical_simulation/outputs_v51_attribution/cluster_run'
    origin = repo / 'numerical_simulation/outputs_version_attribution/cluster_run/plan.json'
    copy_plan(r, origin, out / 'plan.json')
    log('Checking frozen plan, 96 core files and 500 historical task records; no fitting yet')
    plan = r.load_plan(out)
    r.require(not any((out / name).exists() for name in ('paired_three_versions.csv', 'integrity.json')),
              'Existing merge output; refusing to overwrite. Inspect the existing CSV/integrity files')
    complete = {t['task_id']: completed_result(r, out, plan, t) for t in plan['tasks']}
    if not checked_preflight(r, out, plan):
        log('Strict input/environment preflight START')
        invoke('preflight', out, 0, False)
        r.require(checked_preflight(r, out, plan), 'Preflight did not produce a valid record')
    log('Stored strict preflight verified; each new worker rechecks actual inputs/environment')
    by_id = {t['task_id']: t for t in plan['tasks']}
    ordered = plan['gate_task_ids'] + [t['task_id'] for t in plan['tasks']
                                      if t['task_id'] not in plan['gate_task_ids']]
    for index, task_id in enumerate(ordered, 1):
        pilot = task_id in plan['gate_task_ids']
        if index == len(plan['gate_task_ids']) + 1:
            r.require_gates(out, plan)
            log('ALL FIVE PILOT GATES PASSED; starting remaining 245 tasks')
        if complete[task_id]:
            log(f'{index}/{len(ordered)} task {task_id}: VERIFIED, skipped existing fit')
            continue
        log(f'{index}/{len(ordered)} task {task_id}: START' + (' (pilot)' if pilot else ''))
        invoke('worker', out, task_id, pilot)
        r.require(completed_result(r, out, plan, by_id[task_id]), f'Missing worker result: {task_id}')
        log(f'{index}/{len(ordered)} task {task_id}: PASS')
    log('250/250 task records verified; original merge START')
    invoke('merge', out, None, False)
    log(f'COMPLETE: 250/250; results: {out}; paired_three_versions.csv and integrity.json')


def main():
    import fcntl
    from numerical_simulation.version_attribution import run as r

    repo = Path.cwd()
    out = repo / 'numerical_simulation/outputs_v51_attribution/cluster_run'
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another launcher/worker holds this output lock; do not submit again')

        def invoke(command, output, task_id, pilot):
            args = [sys.executable, '-u', '-B', '-m', 'numerical_simulation.version_attribution.run',
                    command, '--output-dir', str(output)]
            if task_id is not None:
                args.extend(['--task-id', str(task_id)])
            if pilot:
                args.append('--pilot')
            subprocess.run(args, check=True, pass_fds=(lock.fileno(),))

        execute_workflow(r, repo, invoke)


if __name__ == '__main__':
    main()
PY
