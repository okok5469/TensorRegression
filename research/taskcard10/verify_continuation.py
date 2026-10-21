"""Independent final audit of Task10's bounded continuation; no scientific fits."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from research.taskcard10 import verify as original


def file_manifest(folder):
    return {p.relative_to(folder).as_posix():original.sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def assert_copy_identity(source, destination, expected):
    assert file_manifest(Path(source))==expected
    assert file_manifest(Path(destination))==expected


def assert_protocol_amendment(old, new):
    before, after = copy.deepcopy(old),copy.deepcopy(new)
    assert after['formal']==dict(before['formal'],workers=2,wall_time_limit_seconds=28800,case_timeout_seconds=3600)
    after['formal'] = before['formal']
    continuation = after.pop('continuation')
    assert continuation['global_clock_reset_allowed'] is False
    assert before==after, 'Scientific source, method, thresholds, environment or requests changed'


def assert_clock_bounds(clock, event_epochs):
    assert clock['limit_seconds']==28800
    first, last = float(clock['started_epoch']),float(clock['started_epoch'])+28800
    assert all(first-1. <= float(event) <= last+2. for event in event_epochs), 'Shared eight-hour clock violated'


def _epoch(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def _write(path, body):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(body,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def verify_original_protection(root, task_report, raw):
    baseline = original.read(task_report/'PROTECTED_BASELINE.json')
    index_path = task_report/'DELIVERY_INDEX.json'
    index = original.read(index_path)
    assert len(baseline['files'])==5922 and index['file_count']==len(index['files'])==897
    expected = dict(baseline['files'])
    for name, record in index['files'].items():
        assert name not in expected or expected[name]==record
        expected[name] = record
    expected[index_path.relative_to(root).as_posix()] = {'size':index_path.stat().st_size,'sha256':original.sha(index_path)}
    sealed = original.read(raw/'PROTECTED_ORIGINALS.json')
    assert sealed['files']==expected and sealed['file_count']==len(expected)==6820
    assert sealed['baseline_sha256']==original.sha(task_report/'PROTECTED_BASELINE.json')
    assert sealed['delivery_index_sha256']==original.sha(index_path)
    for relative, record in expected.items():
        path = (root/relative).resolve()
        assert path.is_relative_to(root) and path.stat().st_size==record['size']
        assert original.sha(path)==record['sha256'], relative
    return {'passed':True,'historical_baseline_files':5922,'original_delivery_files':897,
            'original_delivery_index_preserved':True,'total_files_unchanged':6820}


def verify_preparation(root, task_report, raw, report):
    plan = original.read(raw/'plan.json')
    old = root/plan['continuation']['original_raw_directory']
    old_plan_path = old/'plan.json'
    old_plan = original.read(old_plan_path)
    assert plan['requests']==old_plan['requests'] and len(plan['requests'])==96
    assert original.sha(old_plan_path)==plan['continuation']['original_plan_sha256']
    assert original.sha(root/old_plan['protocol_path'])==old_plan['protocol_sha256']
    assert original.sha(root/plan['protocol_path'])==plan['protocol_sha256']
    assert (root/plan['protocol_path']).resolve()==(report/'PROTOCOL.json').resolve()
    assert_protocol_amendment(old_plan['protocol'],plan['protocol'])
    normalized = copy.deepcopy(plan)
    normalized.pop('continuation')
    for field in ('protocol','protocol_path','protocol_sha256'):
        normalized[field] = old_plan[field]
    assert normalized==old_plan, 'Plan amendment changed original request or historical provenance'
    provenance = plan['protocol']['continuation']
    assert provenance['original_plan_sha256']==original.sha(old_plan_path)
    assert provenance['original_protocol_sha256']==old_plan['protocol_sha256']
    assert provenance['reused_ids']==list(range(30)) and provenance['retry_ids']==list(range(30,48))
    assert provenance['amendment_sha256']==original.sha(report/'DESIGN_AMENDMENT.md')
    assert plan['protocol']['source_files']==old_plan['protocol']['source_files']
    assert plan['protocol']['source_identity']==old_plan['protocol']['source_identity']
    for relative,digest in plan['protocol']['source_identity'].items():
        assert original.sha(root/relative)==digest
    seal = original.read(raw/'CONTINUATION_SEAL.json')
    assert seal['reused_cases']==30 and seal['retry_ids']==list(range(30,48)) and seal['original_protected_files']==6820
    for relative,digest in seal['files'].items():
        assert original.sha(root/relative)==digest, relative
    for name in ('research/taskcard10/resume.py','research/taskcard10/test_resume.py'):
        assert name in seal['files']
    protection = verify_original_protection(root,task_report,raw)
    reuse = original.read(raw/'REUSE_MANIFEST.json')
    assert reuse['old_plan_sha256']==original.sha(old_plan_path)
    assert set(reuse['cases'])=={str(i) for i in range(30)}
    for text_id, row in reuse['cases'].items():
        relative = Path('development')/f'task_{int(text_id):03d}'
        assert row['relative_directory']==relative.as_posix()
        assert Path(row['source']).resolve()==(old/relative).resolve()
        assert_copy_identity(old/relative,raw/relative,row['files'])
        assert row['complete_sha256']==original.sha(old/relative/'complete.json')==original.sha(raw/relative/'complete.json')
    return plan,old,old_plan,protection


def verify_retry_inputs(old, raw, development):
    expected = {i for i in range(30,48) if (old/'development'/f'task_{i:03d}'/'input_identity.json').is_file()}
    path = raw/'RETRY_INPUT_PRECHECK.json'
    assert path.is_file(), 'Training-only retry precheck evidence missing'
    precheck = original.read(path)
    assert precheck['passed'] and precheck['training_only'] and precheck['search_fits']==0
    assert precheck['test_generated'] is False
    rows = {r['task_id']:r for r in precheck['cases']}
    assert len(rows)==len(precheck['cases']) and set(rows)==expected
    complete = {r['request']['task_id']:r for r in development}
    for task_id,row in rows.items():
        source = old/'development'/f'task_{task_id:03d}'/'input_identity.json'
        saved = original.read(source)
        assert row['old_input_identity_sha256']==original.sha(source)
        assert row['input_fingerprints']==saved['input_fingerprints']
        if task_id in complete:
            assert complete[task_id]['input_fingerprints']==saved['input_fingerprints']
    return {'passed':True,'old_partial_ids':sorted(expected),'precheck_training_only':True,
            'completed_retries_checked':sum(i in complete for i in expected)}


def _coverage(requests, records, cohort, raw, reused=()):
    stop_path = raw/('stopped_'+cohort+'.json')
    if not stop_path.exists():
        return original.assert_cohort_coverage(requests,records,cohort),None
    stop = original.read(stop_path)
    assert stop['status']!='COMPLETE'
    reused = set(reused)
    planned = {r['task_id'] for r in requests if r['cohort']==cohort}
    remaining = planned-reused
    assert 'completed' in stop, 'Stop lacks task partition; requires explicit precheck-failure diagnosis'
    done, failed, pending = set(stop['completed']),{r['task_id'] for r in stop['failed']},set(stop['not_started'])
    assert len(done)==len(stop['completed']) and len(failed)==len(stop['failed']) and len(pending)==len(stop['not_started'])
    assert not (done&failed or done&pending or failed&pending)
    assert done|failed|pending==remaining
    return original.assert_cohort_coverage(requests,records,cohort,reused|done),stop


def confirmation_execution_state(raw, completed_count):
    """A launched stage can stop before its first success; zero is not NOT_RUN."""
    raw = Path(raw)
    stop_path, not_run = raw/'stopped_confirmation.json',raw/'confirmation_NOT_RUN.json'
    if stop_path.exists():
        assert not not_run.exists(), 'Launched/stopped confirmation cannot also be NOT_RUN'
        assert original.read(stop_path)['status']!='COMPLETE' and completed_count<48
        return 'STOPPED_PARTIAL'
    if completed_count:
        assert completed_count==48 and (raw/'confirmation_gate.json').is_file()
        assert not not_run.exists()
        return 'COMPLETE'
    reason = original.read(not_run)
    assert reason['status']=='NOT_RUN' and reason['reason']
    assert not (raw/'confirmation').exists() and not (raw/'execution_confirmation').exists()
    return 'NOT_RUN'


def verify_scientific_gates(root, task_report, raw, plan, records, coverage, stopped, analysis):
    thresholds = original.read(task_report/'GATE_INTENT.json')['gates']
    protocol = plan['protocol']
    assert protocol['development_entry_gate']==protocol['promotion_gates']==thresholds and len(thresholds)==9
    lock_path = raw/'development_lock.json'
    confirm = {'status':'NOT_RUN','passed':None}
    if stopped['development'] is not None:
        assert not lock_path.exists() and not records['confirmation']
        dev = {'status':'NOT_EVALUABLE_INCOMPLETE','passed':None}
        assert not analysis.get('development_gate',{}).get('criteria')
    else:
        assert coverage['development']['completed']==48
        dev = original.recompute_gates(records['development'],thresholds)
        lock = original.read(lock_path)
        assert lock['plan_sha256']==original.sha(raw/'plan.json') and lock['completed_development_tasks']==48
        assert lock['reused_cases']==30 and lock['new_complete_cases']==18
        assert lock['reuse_manifest_sha256']==original.sha(raw/'REUSE_MANIFEST.json')
        assert lock['development_artifacts']==file_manifest(raw/'development')
        original.compare_gates(lock['entry_gate'],dev)
        original.compare_gates(original.read(raw/'development_gate_independent.json'),dev)
        original.compare_gates(analysis['development_gate'],dev)
        completion = original.read(raw/'development_completion.json')
        assert completion['status']=='COMPLETE' and completion['completed_development_tasks']==48
        assert completion['reused']==30 and completion['newly_completed']==18
        original.compare_gates(completion['entry_gate'],dev)
        reserve = original.read(raw/'confirmation_budget_gate.json')
        required = 1.5*sum(r['resources']['wall_seconds'] for r in records['development'])/2.+60.
        original.near(required,reserve['required_seconds'])
        assert reserve['scientific_gate_passed']==dev['passed']
        assert reserve['passed']==bool(dev['passed'] and reserve['remaining_seconds']>=required)
        assert completion['confirmation_entry']==reserve['passed']
    if records['confirmation'] or stopped['confirmation'] is not None:
        assert dev['passed'] and original.read(raw/'confirmation_budget_gate.json')['passed']
        assert all(r['development_lock_sha256']==original.sha(lock_path) for r in records['confirmation'])
        if stopped['confirmation'] is None:
            assert coverage['confirmation']['completed']==48
            confirm = original.recompute_gates(records['confirmation'],thresholds)
            original.compare_gates(original.read(raw/'confirmation_gate.json'),confirm)
            original.compare_gates(analysis['confirmation_gate'],confirm)
            original.compare_gates(original.read(raw/'completion.json')['promotion_gate'],confirm)
        else:
            confirm = {'status':'NOT_EVALUABLE_INCOMPLETE','passed':None}
            assert not (raw/'confirmation_gate.json').exists()
            assert not (raw/'confirmation_NOT_RUN.json').exists()
            assert analysis.get('confirmation_gate') is None or not analysis['confirmation_gate'].get('criteria')
    else:
        reason = original.read(raw/'confirmation_NOT_RUN.json')
        assert reason['status']=='NOT_RUN' and reason['reason']
        assert not (raw/'confirmation').exists() and not (raw/'confirmation_gate.json').exists()
        assert analysis.get('confirmation_gate') is None or analysis['confirmation_gate'].get('status')=='NOT_RUN'
    return {'development':dev,'confirmation':confirm,'thresholds_unchanged':True}


def verify_shared_clock(raw, records):
    clock = original.read(raw/'formal_budget.json')
    events = []
    for cohort in ('development','confirmation'):
        execution = raw/('execution_'+cohort)
        if not execution.exists(): continue
        started = original.read(execution/'started.json')
        assert started['workers']==2 and started['case_seconds']==3600
        expected_ids = list(range(30,48)) if cohort=='development' else list(range(48))
        assert started['task_ids']==expected_ids
        timestamp = _epoch(started['started_utc'])
        assert 0 < started['total_seconds'] <= 28800
        assert timestamp+started['total_seconds'] <= clock['started_epoch']+28802.
        events.append(timestamp)
        for record in records[cohort]:
            task_id = record['request']['task_id']
            if cohort=='development' and task_id<30: continue
            complete_path = execution/f'task_{task_id:03d}'/'complete.json'
            complete = original.read(complete_path)
            assert complete['task_id']==task_id and complete['returncode']==0 and 'reason' not in complete
            assert complete['elapsed_seconds']<=3602. and record['resources']['wall_seconds']<=3602.
            events.append(_epoch(record['completed_utc']))
    assert_clock_bounds(clock,events)
    if records['confirmation'] or (raw/'stopped_confirmation.json').exists():
        reserve = original.read(raw/'confirmation_budget_gate.json')
        started = original.read(raw/'execution_confirmation/started.json')
        left_at_start = clock['started_epoch']+28800-_epoch(started['started_utc'])
        assert left_at_start+2.>=reserve['required_seconds'], 'Confirmation launched below frozen reserve'
    return {'passed':True,'shared_limit_seconds':28800,'per_case_limit_seconds':3600,'workers':2,
            'started_epoch':clock['started_epoch'],'new_completion_and_stage_events_checked':len(events),
            'last_scientific_event_elapsed_seconds':max(events)-clock['started_epoch'] if events else None,
            'clock_reset_allowed':False}


def verify_mapped_external(root, task_report, raw, report, counts):
    base = report/'external_audit'
    scope = original.read(base/'SCOPE.json')
    old_scope = original.read(task_report/'external_audit/SCOPE.json')
    assert scope['case_limit']==old_scope['case_limit']==28 and len(scope['cases'])==28
    assert scope['script_sha256']==old_scope['script_sha256']==original.sha(root/'research/taskcard10/verify_external.py')
    assert scope['test_sha256']==old_scope['test_sha256']==original.sha(root/'research/taskcard10/test_verify_external.py')
    old_cases = {(c['cohort'],c['task_id']):c for c in old_scope['cases']}
    new_cases = {(c['cohort'],c['task_id']):c for c in scope['cases']}
    assert len(new_cases)==28 and set(new_cases)==set(old_cases)
    assert scope['mapping_amendment']['original_scope_sha256']==original.sha(task_report/'external_audit/SCOPE.json')
    for key, case in new_cases.items():
        cohort,task_id = key
        assert case['request']==old_cases[key]['request']
        if cohort=='pilot' or (cohort=='development' and task_id<30):
            assert case==old_cases[key]
        else:
            assert case['folder']==(raw/cohort/f'task_{task_id:03d}').relative_to(root).as_posix()
            assert case['plan']==(raw/'plan.json').relative_to(root).as_posix()
            assert case['plan_sha256']==original.sha(raw/'plan.json')
    assert set(scope['plan_sha256'])=={c['plan'] for c in scope['cases']}
    for relative,digest in scope['plan_sha256'].items():
        assert original.sha(root/relative)==digest
    pilot = original.read(task_report/'external_audit/pilot/SUMMARY.json')
    assert pilot['passed'] and pilot['cases_checked']==4 and not (base/'pilot').exists(), 'Pilot must be referenced, not rerun'
    result = {'pilot':{'status':'REUSED_PASS','cases_checked':4,'summary_sha256':original.sha(task_report/'external_audit/pilot/SUMMARY.json')}}
    for cohort in ('development','confirmation'):
        expected = [c for c in scope['cases'] if c['cohort']==cohort]
        path = base/cohort/'SUMMARY.json'
        if not path.exists():
            assert not (base/cohort).exists()
            reason = original.read(base/(cohort+'_NOT_RUN.json'))
            assert reason['status']=='NOT_RUN' and reason['reason']
            assert counts[cohort]<48, 'Complete planned scientific cohort requires its exact12 replay'
            result[cohort] = {'status':'NOT_RUN','planned_cases':12}
            continue
        assert counts[cohort]==48
        summary = original.read(path)
        assert summary['passed'] and summary['cohort']==cohort and summary['cases_checked']==12
        assert summary['task_ids']==[c['task_id'] for c in expected] and summary['selected_models_checked']==36
        assert summary['source_fingerprint']==scope['script_sha256'] and summary['scope_sha256']==original.sha(base/'SCOPE.json')
        assert summary['search_fits']==summary['selection_changes']==0
        assert {p.name for p in (base/cohort).glob('task_*.json')}=={f"task_{c['task_id']:03d}.json" for c in expected}
        rows = []
        for case in expected:
            row = original.read(base/cohort/f"task_{case['task_id']:03d}.json")
            folder = root/case['folder']; complete = original.read(folder/'complete.json')
            assert row['passed'] and row['request']==complete['request']==case['request']
            assert row['complete_sha256']==original.sha(folder/'complete.json')
            assert row['search_fits']==row['selection_changes']==0
            for field in ('source_and_artifact_hashes_exact','training_fingerprints_exact','truth_arrays_exact',
                          'saved_external_y_signal_exact','test_fingerprint_exact','selected_B_identities_exact'):
                assert row[field] is True
            assert row['test_fingerprint']==complete['test_fingerprint']
            for arm in ('A0','A1','A2'):
                for metric in ('pmse','signal_mse'):
                    original.near(row['metrics'][arm][metric],complete['metrics'][arm][metric])
            assert row['maximum_prediction_absolute_error']<=1e-8 and row['maximum_PMSE_signal_MSE_absolute_error']<=1e-8
            rows.append(row)
        original.near(summary['maximum_prediction_absolute_error'],max(r['maximum_prediction_absolute_error'] for r in rows))
        original.near(summary['maximum_PMSE_signal_MSE_absolute_error'],max(r['maximum_PMSE_signal_MSE_absolute_error'] for r in rows))
        result[cohort] = {'status':'PASS','cases_checked':12,'selected_models_checked':36,
                         'maximum_prediction_absolute_error':summary['maximum_prediction_absolute_error'],
                         'maximum_PMSE_signal_MSE_absolute_error':summary['maximum_PMSE_signal_MSE_absolute_error']}
    return {'original_scope_sha256':original.sha(task_report/'external_audit/SCOPE.json'),
            'mapped_scope_sha256':original.sha(base/'SCOPE.json'),'same28_request_ids':True,'cohorts':result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,default=Path('doc/taskcard10/results/continuation_01'))
    parser.add_argument('--report',type=Path,default=Path('doc/taskcard10/continuation_01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    task_report,raw,report = root/'doc/taskcard10',(root/args.raw).resolve(),(root/args.report).resolve()
    assert raw.is_relative_to(task_report) and report.is_relative_to(task_report)
    assert not (report/'FINAL_VALIDATION.json').exists() and not (report/'DELIVERY_INDEX.json').exists()
    plan,old,old_plan,protection = verify_preparation(root,task_report,raw,report)
    records,checks = {'development':[],'confirmation':[]},[]
    for cohort in records:
        for path in sorted((raw/cohort).glob('task_*/complete.json')):
            record = original.read(path); task_id = record['request']['task_id']
            reused = cohort=='development' and task_id<30
            source_folder = old/'development'/f'task_{task_id:03d}' if reused else path.parent
            source_plan = old_plan if reused else plan
            check = original.check_case(root,source_folder,source_plan,root/source_plan['protocol_path'])
            check['origin'] = 'reused_original' if reused else 'continuation_new'
            check['canonical_directory'] = path.parent.relative_to(root).as_posix()
            checks.append(check)
            record['_summary'] = original.read(path.parent/'summary.json')
            records[cohort].append(record)
    counts = {c:len(rows) for c,rows in records.items()}
    assert len({c['input_key'] for c in checks})==len(checks)
    coverage,stopped = {},{}
    coverage['development'],stopped['development'] = _coverage(plan['requests'],records['development'],'development',raw,range(30))
    confirmation_status = confirmation_execution_state(raw,counts['confirmation'])
    if confirmation_status!='NOT_RUN':
        coverage['confirmation'],stopped['confirmation'] = _coverage(plan['requests'],records['confirmation'],'confirmation',raw)
    else:
        coverage['confirmation'] = original.assert_cohort_coverage(plan['requests'],[],'confirmation',[])
        stopped['confirmation'] = None
    assert all(c['planned']==48 for c in coverage.values())
    retry = verify_retry_inputs(old,raw,records['development'])
    analysis = original.read(report/'analysis/summary.json')
    assert analysis['counts']==counts
    gates = verify_scientific_gates(root,task_report,raw,plan,records,coverage,stopped,analysis)
    statistics = original.verify_analysis_tables(report,records)
    timing = verify_shared_clock(raw,records)
    external = verify_mapped_external(root,task_report,raw,report,counts)
    reports = list(report.rglob('*.md'))
    assert (report/'TASKCARD10_FINAL_DECISION.md').is_file()
    links = 0
    for path in reports:
        text = path.read_text(encoding='utf-8')
        assert '\ufffd' not in text
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'): continue
            destination = (path.parent/target.split('#')[0].strip('<>')).resolve()
            assert destination.is_file() or destination in (report/'FINAL_VALIDATION.json',report/'DELIVERY_INDEX.json'), (path,target)
            links += 1
    complete_development = counts['development']==48 and stopped['development'] is None
    complete_confirmation = counts['confirmation']==48 and stopped['confirmation'] is None
    study_closed = complete_development and (complete_confirmation or
        (counts['confirmation']==0 and gates['development']['passed'] is False))
    body = {'status':'COMPLETE_RESEARCH_WITH_DECLARED_GATE_RESULT' if study_closed else 'COMPLETE_DIAGNOSIS_WITH_SCIENTIFIC_STAGE_STOPPED',
        'checked_utc':datetime.now(timezone.utc).isoformat(),'formal_version':'v5.2-search-final','research_only':True,'release_approved':False,
        'original_stopped_state_preserved':True,'original_protection':protection,
        'scientific_stage_complete_under_entry_rules':study_closed,
        'scientific_confirmation_status':confirmation_status,
        'counts':counts,'original_Pilot_referenced_not_recounted':4,'reused_development':30,
        'new_development':counts['development']-30,'request_coverage':coverage,'scientific_stops':stopped,
        'retry_input_identity':retry,'scientific_gates_independently_recomputed':gates,
        'shared_compute_budget':timing,'six_csv_independent_reconstruction':statistics,
        'external_replay':external,'case_checks':checks,
        'candidate_coefficient_hash_checks':sum(c['full_candidates']+c['inner_candidates'] for c in checks),
        'cache_folds_verified':sum(c['cache_folds'] for c in checks),'relative_links_checked':links,
        'mixed_concurrency':{'old_complete_cases_workers':4,'new_cases_workers':2,'uniform_speed_benchmark':False},
        'Q5_same_shape_multiple_support_source_ranking':'NOT_IDENTIFIABLE_UNRESOLVED',
        'limits':['No new scientific repetitions from copies or failed-request retries',
                  'Development remains seen; only untouched original confirmation requests can be independent',
                  'Original failed work and mixed concurrency must be reported separately from successful-case cost ratio',
                  'Q5 scope diagnosis does not establish same-shape candidate ranking benefit',
                  'Two replicates per cell do not establish high-power inferential noninferiority']}
    _write(report/'FINAL_VALIDATION.json',body)
    code_paths = [root/'research/taskcard10'/name for name in ('resume.py','test_resume.py','verify_continuation.py','test_verify_continuation.py','continuation_report.py')]
    paths = {p for base in (raw,report) for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='DELIVERY_INDEX.json'}
    paths.update(p for p in code_paths if p.is_file())
    _write(report/'DELIVERY_INDEX.json',{'scope':'Continuation only; original6820 separately protected, index excludes itself',
        'file_count':len(paths),'files':{p.relative_to(root).as_posix():{'size':p.stat().st_size,'sha256':original.sha(p)} for p in sorted(paths)}})
    print(json.dumps({'status':body['status'],'counts':counts,'files':len(paths)}))


if __name__ == '__main__':
    main()
