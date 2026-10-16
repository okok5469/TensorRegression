"""Inventory first; delete only an explicit, untracked, hash-locked bytecode plan.

No directory moves, recursive deletions, dependency removal, or historical output
changes are performed. Errors are retained in audit records, never hidden.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protected(path):
    parts = Path(path).parts
    return (any(part in {'.git', '.codex', '.agents', '.aws', '.venv', 'venv',
                         'test_tools', 'taskcard4_test_tools'} for part in parts)
            or path.startswith(('doc/algorithm_versions/', 'doc/team_sync/',
                                    'empirical_analysis/'))
            or any(part.startswith('outputs') for part in parts))


def classify(path, tracked, references):
    # These exact directories were inspected: explicit stopped records preserve
    # their diagnostic evidence. ARCHIVE means an index, never a move or delete.
    archive_prefixes = (
        'numerical_simulation/outputs_a2_msic_confirmation/',
        'numerical_simulation/outputs_candidate_search_study/development/',
        'numerical_simulation/outputs_candidate_search_study/confirmation/',
        'numerical_simulation/outputs_candidate_search_study/historical/',
        'numerical_simulation/outputs_support_search_stability/historical/')
    if path.startswith(archive_prefixes):
        return 'ARCHIVE', 'inspected stopped research; retained and indexed in place'
    if protected(path) or tracked or path in references:
        return 'KEEP', 'protected history/dependency, tracked file, or referenced path'
    if Path(path).suffix == '.pyc' and '__pycache__' in Path(path).parts:
        return 'DELETE', 'untracked reproducible bytecode; no exact path reference'
    return 'KEEP', 'default retain; ambiguity requires human review'


def apply_plan(root, plan, tracked, references):
    root = Path(root).resolve(strict=True)
    targets = []
    for item in plan:
        relative = item['path']
        source_path = root / relative
        target = source_path.resolve()
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or source_path.is_symlink()
                or target != source_path
                or not target.is_relative_to(root) or target.is_symlink()
                or not target.is_file()
                or classify(relative, relative in tracked, references)[0] != 'DELETE'
                or target.stat().st_size != item['size_bytes']
                or sha256(target) != item['sha256']):
            raise ValueError('Unsafe or changed deletion target: ' + relative)
        targets.append((target, item))
    # Validate the entire explicit plan before unlinking any one file.
    for target, _ in targets:
        target.unlink()
    return [item for _, item in targets]


def git_state(root):
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=root,
                             capture_output=True, check=True).stdout
    status = subprocess.run(['git', 'status', '--porcelain=v1', '-z',
                             '--untracked-files=all'], cwd=root, capture_output=True)
    return ({p.decode('utf-8') for p in tracked.split(b'\0') if p},
            {'returncode': status.returncode,
             'status': status.stdout.decode('utf-8', errors='replace').split('\0'),
             'warnings': status.stderr.decode('utf-8', errors='replace')})


def inventory(root):
    root = Path(root).resolve(strict=True)
    files, errors, pending = [], [], [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = Path(entry.path).relative_to(root).as_posix()
                    try:
                        if entry.is_symlink():
                            files.append({'path': relative, 'size_bytes': 0,
                                          'mtime_ns': 0, 'kind': 'symlink'})
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        else:
                            info = entry.stat(follow_symlinks=False)
                            files.append({'path': relative, 'size_bytes': info.st_size,
                                          'mtime_ns': info.st_mtime_ns, 'kind': 'file'})
                    except OSError as error:
                        errors.append({'path': relative, 'error': str(error)})
        except OSError as error:
            errors.append({'path': directory.relative_to(root).as_posix(),
                           'error': str(error)})
    return sorted(files, key=lambda item: item['path']), errors


def exact_references(root, files, candidates):
    found = set()
    for item in files:
        path = item['path']
        if path.startswith('doc/releases/v5.2-search-final/audit/') or path in {
                'doc/PROJECT_FILE_MANIFEST.csv', 'doc/PROJECT_DIRECTORY_AUDIT.md',
                'doc/PROJECT_CLEANUP_PLAN.md', 'doc/PROJECT_CLEANUP_REPORT.md'}:
            continue
        if protected(path) and any(part in {'.git', '.venv', 'venv', 'test_tools',
                                            'taskcard4_test_tools'} for part in Path(path).parts):
            continue
        if item['kind'] != 'file' or Path(path).suffix.lower() not in {
                '.py', '.md', '.sh', '.ps1', '.json', '.yaml', '.yml', '.toml',
                '.txt', '.csv', '.patch', '.html'}:
            continue
        text = (root / path).read_text(encoding='utf-8', errors='replace')
        found.update(candidate for candidate in candidates if candidate in text
                     or candidate.replace('/', '\\') in text)
    return found


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def snapshot(root, output):
    start = datetime.now(timezone.utc).isoformat()
    tracked, status = git_state(root)
    files, errors = inventory(root)
    print('Inventory: %d entries, %d access errors' % (len(files), len(errors)), flush=True)
    candidates = {item['path'] for item in files
                  if classify(item['path'], item['path'] in tracked, set())[0] == 'DELETE'}
    references = exact_references(root, files, candidates)
    plan, important = [], []
    for item in files:
        path = item['path']
        action, reason = classify(path, path in tracked, references)
        item.update(action=action, reason=reason, tracked=path in tracked, sha256='')
        valuable = (path.startswith(('src/', 'doc/algorithm_versions/', 'doc/team_sync/',
                                     'empirical_analysis/', 'numerical_simulation/'))
                    and not any(part in {'__pycache__', 'test_tools', 'taskcard4_test_tools'}
                                for part in Path(path).parts))
        if item['kind'] == 'file' and (valuable or action == 'DELETE'):
            try:
                item['sha256'] = sha256(root / path)
            except OSError as error:
                errors.append({'path': path, 'error': str(error)})
                item.update(action='KEEP', reason='hash failed; manual review')
        if item['sha256'] and valuable:
            important.append({'path': path, 'sha256': item['sha256'],
                              'size_bytes': item['size_bytes']})
        if item['action'] == 'DELETE':
            plan.append({'path': path, 'size_bytes': item['size_bytes'],
                         'sha256': item['sha256']})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'file_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'size_bytes', 'mtime_ns',
                                                    'kind', 'action', 'reason', 'tracked', 'sha256'])
        writer.writeheader()
        writer.writerows(files)
    write_json(output / 'git_status.json', status)
    write_json(output / 'important_sha256.json', important)
    write_json(output / 'delete_dry_run.json', plan)
    summary = {'started_utc': start, 'ended_utc': datetime.now(timezone.utc).isoformat(),
               'files': len(files), 'bytes': sum(item['size_bytes'] for item in files),
               'delete_count': len(plan), 'delete_bytes': sum(item['size_bytes'] for item in plan),
               'important_hash_count': len(important), 'references': sorted(references),
               'scan_errors': errors, 'complete': not errors}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply-plan', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if not args.output.resolve().is_relative_to(root):
        raise ValueError('Audit output must stay in repository')
    if args.apply_plan:
        tracked, _ = git_state(root)
        plan = json.loads(args.apply_plan.read_text(encoding='utf-8'))
        files, errors = inventory(root)
        if errors:
            raise ValueError('Incomplete scan blocks actual cleanup; inspect inventory errors')
        references = exact_references(root, files, {item['path'] for item in plan})
        # Generated audit records necessarily quote the explicit plan; do not
        # treat those records as independent reproducibility references.
        deletion = apply_plan(root, plan, tracked, references)
        write_json(args.output / 'deleted_files.json', deletion)
    print(json.dumps(snapshot(root, args.output), ensure_ascii=True))


if __name__ == '__main__':
    main()
