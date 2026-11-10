"""Build and verify a portable offline Task18 bundle, using only stdlib.

Wheel download/provenance verification is a separate explicit preparation step.
This builder never downloads, installs, fits models, or imports Linux wheels.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_payload_path(name):
    path = PurePosixPath(name)
    forbidden = {'.git', '.venv', '__pycache__', 'runtime', 'initial_training_diagnostics'}
    if ('\\' in name or path.is_absolute() or '..' in path.parts or
            any(':' in part for part in path.parts) or not path.parts or
            any(part in forbidden for part in path.parts) or
            str(path).startswith('numerical_simulation/outputs_taskcard18_closeout/') or
            path.suffix.lower() in {'.pyc', '.pyd', '.dll', '.exe'}):
        raise ValueError('Unsafe or nonportable payload path: '+name)
    return path


def write_archive(entries, target, prefix):
    for name in entries:
        validate_payload_path(name)
    hashes = {name: file_sha(path) for name,path in sorted(entries.items())}
    manifest = {'schema_version':1, 'files_sha256':hashes}
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name,path in sorted(entries.items()):
            info = zipfile.ZipInfo(prefix+'/'+name, date_time=(2026,10,26,0,0,0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.suffix == '.sh' else 0o644
            info.external_attr = (stat.S_IFREG|mode)<<16
            archive.writestr(info,path.read_bytes())
        archive.writestr(prefix+'/PAYLOAD_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
        archive.writestr(prefix+'/numerical_simulation/outputs_taskcard18_closeout/',b'')
    for name,path in entries.items():
        if file_sha(path)!=hashes[name]:
            raise ValueError('Source changed during bundle build: '+name)
    return manifest


def safe_extract(archive_path, destination, prefix):
    """Validate every member before writing; refuse paths outside the named root."""
    base = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        seen=set()
        for member in archive.infolist():
            path=PurePosixPath(member.filename)
            target=(base/Path(*path.parts)).resolve()
            mode=member.external_attr>>16
            if ('\\' in member.filename or path.is_absolute() or '..' in path.parts or
                    not path.parts or path.parts[0]!=prefix or any(':' in p for p in path.parts) or
                    not target.is_relative_to(base) or member.filename in seen or stat.S_ISLNK(mode)):
                raise ValueError('Unsafe ZIP member: '+member.filename)
            seen.add(member.filename)
        for member in archive.infolist():
            target=base/Path(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True,exist_ok=True)
                continue
            content=archive.read(member)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.read_bytes()!=content:
                    raise ValueError('Refusing to overwrite different verification input: '+str(target))
            else:
                with target.open('xb') as handle:handle.write(content)
    return base/prefix


def verify_payload(root):
    manifest=json.loads((root/'PAYLOAD_MANIFEST.json').read_text(encoding='utf-8'))
    total=0
    for name,expected in manifest['files_sha256'].items():
        validate_payload_path(name)
        path=root/name
        if file_sha(path)!=expected:
            raise ValueError('Payload SHA256 mismatch: '+name)
        total+=path.stat().st_size
    return {'file_count':len(manifest['files_sha256']),'uncompressed_bytes':total}


def verified_wheels(delivery):
    manifest=json.loads((delivery/'WHEELS_MANIFEST.json').read_text(encoding='utf-8'))
    expected={'scikit-learn':'1.5.2','joblib':'1.4.2','threadpoolctl':'3.5.0'}
    if manifest['platform']!='linux_x86_64' or manifest['python_tag']!='cp39':
        raise ValueError('Wrong wheel platform/Python target')
    if {item['distribution']:item['version'] for item in manifest['wheels']}!=expected or len(manifest['wheels'])!=3:
        raise ValueError('Exactly the three pinned offline dependencies are required')
    entries={}
    for item in manifest['wheels']:
        name=item['filename'];path=delivery/'wheels'/name
        if '/' in name or '\\' in name or file_sha(path)!=item['sha256'] or item['sha256']!=item['published_sha256']:
            raise ValueError('Wheel SHA256 differs from official PyPI metadata: '+name)
        if not item['source_url'].startswith('https://files.pythonhosted.org/') or not item['pypi_json_url'].startswith('https://pypi.org/pypi/'):
            raise ValueError('Wheel provenance is not official PyPI')
        with zipfile.ZipFile(path) as archive:
            if any(PurePosixPath(p).suffix.lower() in {'.pyd','.dll','.exe'} for p in archive.namelist()):
                raise ValueError('Windows binary found inside wheel: '+name)
            wheel_name=next(p for p in archive.namelist() if p.endswith('.dist-info/WHEEL'))
            tags=[line[5:] for line in archive.read(wheel_name).decode().splitlines() if line.startswith('Tag: ')]
        valid_tag='cp39-cp39-manylinux2014_x86_64' if item['distribution']=='scikit-learn' else 'py3-none-any'
        if valid_tag not in tags:
            raise ValueError('Wrong pinned wheel compatibility tag: '+name)
        entries['wheelhouse/'+name]=path
    entries['wheelhouse/manifest.json']=delivery/'WHEELS_MANIFEST.json'
    return entries


def collect_entries(root, delivery):
    historical=root/'numerical_simulation/outputs_v52_search_final_full'
    plan=json.loads((historical/'plan.json').read_text(encoding='utf-8'))
    if len(plan['source_fingerprints'])!=38:
        raise ValueError('Formal source closure must contain38 files')
    entries={}
    for name,expected in plan['source_fingerprints'].items():
        if file_sha(root/name)!=expected:raise ValueError('Frozen38 source conflict: '+name)
        entries[name]=root/name
    manifest_path=root/'doc/releases/v5.2-search-final/RELEASE_MANIFEST.json'
    release=json.loads(manifest_path.read_text(encoding='utf-8'))
    if len(release['source_sha256'])!=32 or any(file_sha(root/name)!=value for name,value in release['source_sha256'].items()):
        raise ValueError('Frozen production32 source conflict')
    for name in ['plan.json','raw_results.csv','summary_results.csv','integrity_report.json']:
        path=historical/name;entries[path.relative_to(root).as_posix()]=path
    for extension in ['json','npz']:
        files=sorted((historical/'tasks').glob('*.'+extension))
        if len(files)!=700:raise ValueError('Historical task inventory requires700 '+extension)
        for path in files:entries[path.relative_to(root).as_posix()]=path
    for name in ['doc/releases/v5.2-search-final/RELEASE_MANIFEST.json',
                 'doc/releases/v5.2-search-final/SOURCE_SHA256.txt',
                 'doc/taskcard10/continuation_01/FINAL_VALIDATION.json',
                 'doc/taskcard12/FINAL_VALIDATION.json','AGENTS.md']:
        entries[name]=root/name
    for path in (root/'research/taskcard18').rglob('*'):
        if path.is_file() and path.suffix in {'.py','.md','.sh'} and '__pycache__' not in path.parts:
            entries[path.relative_to(root).as_posix()]=path
    mutable={'gates.json','TASKCARD18_FINAL_REPORT.md','CHAPTER_FREEZE.md'}
    for folder in ['doc/taskcard18','paper']:
        for path in (root/folder).rglob('*'):
            relative=path.relative_to(root)
            if (not path.is_file() or 'cluster_delivery' in relative.parts or
                    'mplconfig' in relative.parts or '__pycache__' in relative.parts or
                    path.name in mutable or (path.suffix in {'.txt','.log'} and 'figure_qa' not in relative.parts)):
                continue
            entries[relative.as_posix()]=path
    shell=root/'run_taskcard18_cluster.sh'
    if not shell.is_file() or not (root/'research/taskcard18/cluster.py').is_file():
        raise ValueError('Cluster runner/SH closure is not ready')
    entries['run_taskcard18_cluster.sh']=shell
    entries.update(verified_wheels(delivery))
    scope=delivery/'CLUSTER_BUNDLE_SCOPE.md'
    scope.write_text('# Cluster bundle scope\n\nLocal diagnostic reviews, identity reports and frozen ABESS/Oracle figures are historical snapshots, not cluster completion evidence. No local fits, local RUN_IDENTITY, evaluations or runtime are shipped. Cluster outputs begin empty. The chapters remain DRAFT until the complete revised comparison is accepted.\n\nPAYLOAD_MANIFEST hashes immutable shipped files only. Mutable gates/final reports/CHAPTER_FREEZE/test logs are omitted so collect and resume can create their own outputs.\n',encoding='utf-8')
    entries['CLUSTER_BUNDLE_SCOPE.md']=scope
    for name in entries:validate_payload_path(name)
    return entries,plan


def verify_windows_import_closure(root, delivery):
    """Exercise extracted source using only existing Windows dependency folders.

    This deliberately does not load the bundled Linux wheels or establish HPC
    binary compatibility. -S prevents site/.pth files from restoring the repo.
    """
    original=Path(__file__).resolve().parents[2]
    dependencies=sorted({str(Path(importlib.util.find_spec(name).origin).parent.parent)
                         for name in ('numpy','scipy')})
    dependencies.append(str(original/'doc/taskcard13/runtime'))
    search=[str(root/'src'),str(root),*dependencies]
    script=r'''
import importlib, json
from pathlib import Path
import sys, unittest
root=Path.cwd().resolve()
plan=json.loads((root/'numerical_simulation/outputs_v52_search_final_full/plan.json').read_text(encoding='utf-8'))
modules=[]
for name in plan['source_fingerprints']:
    path=Path(name)
    if path.suffix!='.py': continue
    parts=list(path.with_suffix('').parts)
    if parts[0]=='src': parts=parts[1:]
    if parts[-1]=='__init__': parts=parts[:-1]
    modules.append('.'.join(parts))
modules+=['research.taskcard18.'+name for name in ('workflow','solvers','baselines','analysis','accept','identity','report','cluster','bootstrap_runtime')]
for name in sorted(set(modules)):
    module=importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError('Extracted-source import escaped mini-root: '+name)
tests=['research.taskcard18.test_baselines.ConvexSolverTests.test_lasso_matches_orthogonal_soft_threshold_solution',
       'research.taskcard18.test_baselines.ConvexSolverTests.test_group_lasso_matches_orthogonal_block_solution',
       'research.taskcard18.test_baselines.BaselineBoundaryTests.test_cv_uses_validation_sse_and_first_descending_lambda_on_ties',
       'research.taskcard18.test_accept','research.taskcard18.test_closeout',
       'research.taskcard18.test_identity','research.taskcard18.test_report']
suite=unittest.defaultTestLoader.loadTestsFromNames(tests)
result=unittest.TextTestRunner(verbosity=1).run(suite)
if not result.wasSuccessful(): raise SystemExit(1)
import numpy, scipy, sklearn
print(json.dumps({'status':'PASS_WINDOWS_DEPENDENCY_SOURCE_CLOSURE_ONLY',
                 'extracted_source_modules':len(set(modules)), 'focused_synthetic_tests':result.testsRun,
                 'module_origins':{name:str(Path(sys.modules[name].__file__).resolve()) for name in sorted(set(modules))},
                 'dependency_origins':{m.__name__:m.__file__ for m in (numpy,scipy,sklearn)},
                 'sys_path':sys.path,'linux_wheels_imported':False,'HPC_binary_compatibility_proven':False}))
'''
    result=subprocess.run([sys.executable,'-B','-S','-c',script],cwd=root,capture_output=True,
                          text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8',
                          PYTHONPATH=os.pathsep.join(search),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                          OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'))
    (delivery/'MINI_ROOT_IMPORT_VALIDATION.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise ValueError('Extracted mini-root import/focused checks failed; see MINI_ROOT_IMPORT_VALIDATION.txt')
    record=json.loads(result.stdout)
    record['explicit_pythonpath']=search
    (delivery/'MINI_ROOT_IMPORT_VALIDATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {key:record[key] for key in ('status','extracted_source_modules','focused_synthetic_tests','linux_wheels_imported','HPC_binary_compatibility_proven')}


def verify_bundle(archive, delivery, prefix, plan):
    with zipfile.ZipFile(archive) as handle:
        members=handle.infolist()
        archive_counts={'zip_entries':len(members),'zip_files':sum(not item.is_dir() for item in members),
                        'zip_uncompressed_bytes':sum(item.file_size for item in members)}
    root=safe_extract(archive,delivery/('verification_'+file_sha(archive)[:12]),prefix)
    counts=verify_payload(root)
    for name,expected in plan['source_fingerprints'].items():
        if file_sha(root/name)!=expected:raise ValueError('Extracted frozen source conflict: '+name)
    output=root/'numerical_simulation/outputs_taskcard18_closeout'
    if any(output.iterdir()) or (root/'runtime').exists():
        raise ValueError('New cluster output must be empty and runtime must not exist before bootstrap')
    python_files=list(root.rglob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path),feature_version=(3,9))
    result=subprocess.run([sys.executable,'-B','-S','-m','research.taskcard18.cluster','submit','--dry-run'],
                          cwd=root,capture_output=True,text=True,encoding='utf-8',
                          env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONPATH=''))
    (delivery/'EXTRACTED_SCHEDULER_DRY_RUN.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode or 'sbatch' not in result.stdout:
        raise ValueError('Extracted scheduler dry-run failed')
    verify_payload(root)
    if any(output.iterdir()) or (root/'runtime').exists():
        raise ValueError('Scheduler dry-run wrote runtime or output files')
    imports=verify_windows_import_closure(root,delivery)
    verify_payload(root)
    if any(output.iterdir()) or (root/'runtime').exists():
        raise ValueError('Mini-root import checks wrote runtime or formal output files')
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    shell_result={'status':'NOT_AVAILABLE','reason':'Existing Git Bash not found; no installation attempted'}
    if bash.is_file():
        check=subprocess.run([str(bash),'-n',(root/'run_taskcard18_cluster.sh').as_posix()],capture_output=True,text=True)
        if check.returncode:raise ValueError('Existing Git Bash syntax check failed: '+check.stderr)
        shell_result={'status':'PASS','shell':'existing Git Bash -n'}
    return {**counts,**archive_counts,'source38_verified':38,'production32_verified':32,'historical_json':700,'historical_npz':700,
            'python39_ast_files':len(python_files),'scheduler_dry_run':'PASS','mini_root_imports':imports,'bash_syntax':shell_result,
            'runtime_directory_absent':True,'new_output_directory_empty':True,'windows_runtime_or_local_fits_included':False,
            'linux_sklearn_imported_on_windows':False,'actual_HPC_execution':'NOT_RUN','verification_root':str(root)}


def main():
    root=Path(__file__).resolve().parents[2]
    delivery=root/'doc/taskcard18/cluster_delivery';delivery.mkdir(parents=True,exist_ok=True)
    prefix='taskcard18_cluster_20261026'
    entries,plan=collect_entries(root,delivery)
    temporary=delivery/(prefix+'.building.zip')
    write_archive(entries,temporary,prefix)
    result=verify_bundle(temporary,delivery,prefix,plan)
    target=delivery/(prefix+'.zip');temporary.replace(target)
    digest=file_sha(target)
    (delivery/(target.name+'.sha256')).write_text(digest+'  '+target.name+'\n',encoding='ascii')
    result.update({'schema_version':1,'status':'PASS_PORTABLE_OFFLINE_BUNDLE','zip':target.name,
                   'zip_sha256':digest,'zip_bytes':target.stat().st_size,'top_level':prefix+'/'})
    (delivery/'BUILD_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
