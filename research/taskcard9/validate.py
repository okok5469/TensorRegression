"""Independent delivery checks on saved evidence; never generate data or fit."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (a, b)


def write(path, body):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(body, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def check_attribution(out):
    selected = rows(out / 'attribution/selected_models.csv')
    pairs = rows(out / 'attribution/paired_changes.csv')
    assert len(selected) == 750 and len(pairs) == 500
    key = {(r['task_id'], r['version']): r for r in selected}
    assert len(key) == 750
    assert Counter(r['version'] for r in selected) == {'v5':250, 'v51':250, 'v52':250}
    for pair in pairs:
        left = key[(pair['task_id'], pair['left_version'])]
        right = key[(pair['task_id'], pair['right_version'])]
        for name in ('pmse', 'rss', 'msic', 'q'):
            close(pair['delta_' + name], float(right[name]) - float(left[name]))
        assert (pair['B_exact'] == 'True') == (left['B_sha256'] == right['B_sha256'])
    groups = []
    for scenario, phase in (('n100','BCD'), ('n100','Grid'), ('n400','BCD'), ('n400','Grid')):
        group = [r for r in pairs if r['scenario'] == scenario and r['phase'] == phase]
        assert len(group) == 50
        values = [float(r['delta_pmse']) for r in group]
        groups.append({'scenario':scenario, 'phase':phase, 'n':len(group), 'mean_delta_pmse':mean(values),
                       'improved':sum(v < -1e-12 for v in values),
                       'equal':sum(abs(v) <= 1e-12 for v in values),
                       'worse':sum(v > 1e-12 for v in values)})
    assert [g['worse'] for g in groups] == [28,0,2,0]
    validation = read(out/'attribution/validation.json')
    assert validation['status'] == 'PASS' and validation['seed_families'] == 150
    return {'passed':True, 'models':750, 'phase_pairs':500, 'independently_aggregated':groups}


def check_pool(root, out):
    selected = rows(out/'pool_msic/four_rule_selected.csv')
    assert len(selected) == 864
    assert Counter(r['rule'] for r in selected) == dict.fromkeys(('A0','A1','A2','A3'),216)
    key = {(r['cohort'],int(r['task_id']),r['rule']):r for r in selected}
    assert len(key) == 864
    protocol = read(out/'PROTOCOL.json')
    replay = {}
    for request in protocol['test_replay_requests']:
        task, cohort = request['task_id'], request['cohort']
        body = read(out/f'pool_msic/replay_{cohort}_{task:03d}.json')
        summary = body['summary']
        assert summary['gates_passed'] and summary['new_fits'] == 0
        values = [r['pmse'] for r in body['candidates']]
        close(min(values), summary['pool_oracle_pmse'])
        close(values[summary['selected_index']] - min(values), summary['selection_gap'])
        replay[(cohort,task)] = {r['B_sha256']:r for r in body['candidates']}
    assert len(replay) == 12
    known_counts = Counter()
    # Independently verify every reused prediction refers to an already evaluated B.
    for cohort in ('development','confirmation'):
        for task in range(108):
            directory = root/'numerical_simulation/outputs_selection_research_task8'/cohort/f'task_{task:03d}'
            record = read(directory/'complete.json')
            pool = read(directory/'trace.json')['candidates']
            choices = read(directory/'selection.json')['choices']
            old = {pool[choice['selected_index']]['B_sha256']: metric
                   for choice, metric in zip(choices,record['metrics'])}
            for rule in ('A0','A1','A2','A3'):
                row = key[(cohort,task,rule)]
                candidate = pool[int(row['selected_index'])]
                assert row['B_sha256'] == candidate['B_sha256']
                assert json.loads(row['support']) == candidate['support']
                assert int(row['q']) == candidate['q']
                truth = record['supports_true']
                assert int(row['exact_recovery']) == int(candidate['support'] == truth)
                expected = replay.get((cohort,task),{}).get(row['B_sha256'],old.get(row['B_sha256']))
                if expected is None:
                    assert row['pmse'] == '' and row['signal_mse'] == ''
                    continue
                close(row['pmse'], expected['pmse'])
                close(row['signal_mse'], expected['signal_mse'])
                known_counts[rule] += 1
    assert dict(known_counts) == {'A0':216,'A1':141,'A2':123,'A3':216}
    statistics = []
    for rule in ('A0','A1','A2','A3'):
        group = [r for r in selected if r['rule'] == rule]
        statistics.append({'rule':rule, 'n':216, 'q_mean':mean(float(r['q']) for r in group),
                           'under_rate':mean(float(r['q_under']) for r in group),
                           'exact_count':sum(int(r['exact_recovery']) for r in group),
                           'pmse_available':known_counts[rule]})
    assert [r['exact_count'] for r in statistics] == [71,65,49,69]
    identities = read(out/'pool_msic/identity_audit.json')
    assert identities['passed'] and identities['exact_requests'] == 216
    assert identities['candidate_B_hash_finite_support_checks'] == 31968
    return {'passed':True,'rows':864,'replay_cases':12,'independently_aggregated':statistics,
            'all_known_PMSE_verified_by_candidate_identity':True}


def main():
    root = Path(__file__).resolve().parents[2]
    out = root/'doc/taskcard9'
    assert not (out/'FINAL_VALIDATION.json').exists(), 'Never overwrite delivery validation'
    assert not (out/'DELIVERY_INDEX.json').exists(), 'Never overwrite delivery index'
    protocol = read(out/'PROTOCOL.json')
    for rel,digest in protocol['source_sha256'].items():
        assert sha(root/'src'/rel) == digest
        assert sha(root/'doc/algorithm_versions/v5.2-search-final/src'/rel) == digest
    for rel,digest in protocol['dependencies'].items():
        assert sha(root/rel) == digest
    assert sha(out/'DESIGN.md') == protocol['task9_design_sha256']
    attribution = check_attribution(out)
    pool = check_pool(root,out)
    warm = read(out/'warm/gate_summary.json')
    assert warm['fixed_scope_correctness_passed'] and warm['distinct_top_level_checks'] == 25
    assert warm['full_adaptive_correctness'] == 'NOT_RUN'
    assert warm['full_pipeline_performance_allowed'] is False
    current_green = read(out/'warm/repro_entry_check/green.json')
    assert current_green['failures'] == current_green['errors'] == 0
    for rel,digest in current_green['source_and_test_sha256'].items():
        assert sha(root/rel) == digest
    # A separate runner verifies analysis math without any solver or new data.
    command = [sys.executable,'-B','-m','unittest','research.taskcard9.test_attribution','research.taskcard9.test_pool_msic']
    result = subprocess.run(command,cwd=root,capture_output=True,text=True,check=False,timeout=60)
    with (out/'ANALYSIS_TESTS.log').open('x',encoding='utf-8') as handle:
        handle.write(result.stdout + result.stderr)
    assert result.returncode == 0, 'See ANALYSIS_TESTS.log'
    protect = read(out/'PROTECTED_FINAL_CHECK.json')
    assert protect['passed'] and protect['checked'] == 5852
    original = read(out/'GIT_INITIAL.json')['entries']
    current = subprocess.run(['git','status','--porcelain','-z','--untracked-files=no'],cwd=root,capture_output=True,check=True).stdout.decode('utf-8').split('\0')
    assert current == original, 'Tracked status changed outside the recorded workflow modifications'
    required = ('EVIDENCE_ASSET_AUDIT.md','N100_REGRESSION_ATTRIBUTION.md','CANDIDATE_SELECTION_GAP.md',
                'MSIC_COMPONENT_ABLATION.md','PURE_WARM_DESIGN_AND_GATE.md','TASKCARD9_FINAL_DECISION.md')
    assert all((out/name).is_file() for name in required)
    planned_self = {out/'FINAL_VALIDATION.json',out/'DELIVERY_INDEX.json'}
    count = 0
    for path in out.glob('*.md'):
        body = path.read_text(encoding='utf-8')
        assert '\ufffd' not in body
        for target in re.findall(r'\]\(([^)]+)\)',body):
            if '://' in target or target.startswith('#'):
                continue
            resolved = (path.parent/target.split('#')[0].strip('<>')).resolve()
            assert resolved.is_file() or resolved in planned_self, (path.name,target)
            count += 1
    diff = subprocess.run(['git','diff','--check','--','doc/task_plan.md','doc/findings.md','doc/progress.md'],cwd=root,capture_output=True,text=True,check=False)
    assert diff.returncode == 0, diff.stdout + diff.stderr
    body = {'status':'PASS_COMPLETE_WITH_DECLARED_LIMITS','closed_client_date':'2026-10-18',
            'checked_utc':datetime.now(timezone.utc).isoformat(),'research_only':True,'release_approved':False,
            'formal_version':'v5.2-search-final','production_core_files_unchanged':32,'frozen_core_files_unchanged':32,
            'protected_historical_files_unchanged':5852,'tracked_status_matches_initial':True,
            'task7_attribution':attribution,'task8_saved_pool':pool,
            'warm_fixed_scope':warm,'analysis_math_test_returncode':result.returncode,
            'analysis_test_scope':'Descriptive summary, report isolation, score ties/tolerance and sparse predictions; solver gate is separate25',
            'required_reports':list(required),'markdown_relative_links_checked':count,
            'protocol_sha256':sha(out/'PROTOCOL.json'),'old_production_or_historical_mutations':0,
            'new_large_scale_simulations':0,'new_model_fits_in_pool_analysis':0,'new_LEMON_OOF_runs':0,
            'strictly_bounded_test_replay_cases':12,
            'not_run':['Task7 original Linux RSS/PMSE regeneration and all-pool oracle (assets missing)',
                       'Task8 new A1/A2 test metrics outside allowed12 or exact previously evaluated B identity',
                       'Pure Warm full adaptive/grid/MSIC integration','Warm performance and Warm x new MSIC','Next task card'],
            'scientific_limits':['250 labels have150 seed families; no cross-scenario IID inference',
                                 'Task8 development/confirmation data now seen; no new independent confirmation',
                                 'Pool oracle is a posthoc finite-test diagnostic, never used for selection or tuning',
                                 'Historical Task8 promotion gates remain failed; no rule promoted']}
    write(out/'FINAL_VALIDATION.json',body)
    paths = [p for directory in (out,root/'research/taskcard9') for p in directory.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'DELIVERY_INDEX.json']
    write(out/'DELIVERY_INDEX.json',{'scope':'Task9 deliverables and isolated research code; current hashes at closure; index excludes itself',
          'files':{p.relative_to(root).as_posix():{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)},
          'file_count':len(paths),'research_only':True,'release_approved':False})
    print(json.dumps({'status':body['status'],'protected':5852,'reports':6,'links':count,'indexed_files':len(paths)}))


if __name__ == '__main__':
    main()
