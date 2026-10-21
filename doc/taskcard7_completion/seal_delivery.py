"""Validate target reports and seal this completion delivery without touching raw results."""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_new(path, data):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(data,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n')


def main():
    repo=Path(__file__).resolve().parents[2]
    out=Path(__file__).resolve().parent
    baseline=read(out/'protected_baseline.json')
    for name,entry in baseline['files'].items():
        path=repo/name
        assert path.stat().st_size==entry['size'] and sha(path)==entry['sha256'],name
    cluster,numeric,analysis=(read(out/name) for name in ('cluster_audit.json','estimate_audit.json','analysis.json'))
    assert cluster['status']==numeric['status']=='PASS'
    assert cluster['record_seals']==cluster['existing_csv_exact_native_rows']==250
    assert numeric['saved_models_audited']==750
    assert analysis['task_labels']==analysis['unique_full_input_keys']==250
    assert analysis['seed_families']==150
    assert sha(repo/'numerical_simulation/outputs_v51_attribution/cluster_run/paired_three_versions.csv')==analysis['source_csv_sha256']
    status=read(repo/'numerical_simulation/outputs_v51_attribution/STATUS.json')
    assert status['full_attribution_complete'] and status['completed_v51_task_count']==250
    qa=read(out/'figures/QA_REPORT.json')
    assert all(record['exit_code']==0 for record in qa.values())
    assert read(out/'figures/VISUAL_REVIEW.json')['passed']
    for path in out.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    report_names=[name for name in baseline['mutable_targets'] if name.startswith('doc/') and
                  Path(name).name not in ('task_plan.md','findings.md','progress.md')]
    assert len(report_names)==10
    index_path=out/'DELIVERY_INDEX.json'
    assert not index_path.exists(), 'Existing delivery is immutable; use a new completion directory'
    links=[]
    for name in report_names+['doc/taskcard7_completion/REPRODUCIBILITY.md',
                              'numerical_simulation/outputs_v51_attribution/README.md']:
        path=repo/name
        text=path.read_text(encoding='utf-8')
        assert '\ufffd' not in text,name
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):
                continue
            dest=(path.parent/target.split('#')[0]).resolve()
            assert dest.is_file() or dest==index_path,(name,target)
            links.append((name,target))
    mutable=baseline['mutable_targets']
    check=subprocess.run(['git','diff','--check','--',*mutable,'doc/taskcard7_completion'],
                         capture_output=True,text=True,encoding='utf-8')
    assert check.returncode==0,check.stdout+check.stderr
    git=subprocess.run(['git','status','--short','--untracked-files=no','--',*mutable,'doc/taskcard7_completion'],
                       capture_output=True,text=True,encoding='utf-8',check=True)
    (out/'git_final.txt').write_text(git.stdout,encoding='utf-8')
    write_new(out/'FINAL_VALIDATION.json',{
        'passed':True,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'completed_tasks':250,'pilot_gates':5,'saved_models_independently_audited':750,
        'protected_files_unchanged':len(baseline['files']),'report_count':len(report_names),
        'local_links_checked':len(links),'scoped_git_diff_check_passed':True,
        'metrics_max_absolute_errors':numeric['max_absolute_errors'],
        'v51_v52_exact_models':numeric['v51_v52_exact'],
        'full_input_keys':250,'seed_testseed_families':150,'new_fits':0,'merge_rerun':False,
        'independent_RSS_PMSE_reconstruction':False,
        'explicit_root_audit_refactor_replay_matches_serialized_original':True,
        'old_indices_and_raw_results_edited':False,'production_or_algorithm_changed':False})
    files={p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files.update(repo/name for name in mutable)
    files.update(p for p in (repo/'numerical_simulation/outputs_v51_attribution/cluster_run').rglob('*') if p.is_file())
    entries={p.relative_to(repo).as_posix():{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}
    write_new(index_path,{'scope':'Task card7 completion: targets, preserved prior report bytes, all returned raw artifacts, new analysis/audits/figure/reproducibility; excludes this index and bytecode',
        'full_attribution_complete':True,'production_changed':False,'new_fits':0,
        'protected_baseline_file_count':len(baseline['files']),
        'report_paths':report_names,'file_count':len(entries),'files':entries})
    for name,entry in entries.items():
        assert sha(repo/name)==entry['sha256'],name
    for name,target in links:
        assert ((repo/name).parent/target.split('#')[0]).resolve().is_file()
    print('Sealed',len(report_names),'reports;',len(entries),'delivery files;',len(links),'valid links')


if __name__=='__main__':
    main()
