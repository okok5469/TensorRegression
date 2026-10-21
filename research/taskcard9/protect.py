"""Snapshot current historical bytes; never recreate old scientific results."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets(root, mutable):
    paths = set()
    for manifest in ('doc/taskcard7_completion/protected_baseline.json',
                     'doc/taskcard7_completion/DELIVERY_INDEX.json',
                     'doc/taskcard8/DELIVERY_INDEX.json'):
        body = json.loads((root / manifest).read_text(encoding='utf-8'))
        paths.update(body['files'])
        paths.add(manifest)
    # Current bytes include all authorized post-Task8 report updates.
    for directory in ('src', 'tests', 'scripts', 'Math', 'work_progress'):
        paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p.relative_to(root).as_posix() for p in (root / 'doc').glob('*.md'))
    paths.update(p.relative_to(root).as_posix() for p in (root / 'numerical_simulation').glob('outputs_v51_attribution/*') if p.is_file())
    return sorted(paths - mutable)


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / 'doc/taskcard9'
    mutable = {'doc/task_plan.md', 'doc/findings.md', 'doc/progress.md'}
    path = out / 'PROTECTED_BASELINE.json'
    if sys.argv[1] == 'snapshot':
        body = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Current bytes of historical protected unions plus scientific source/tests/scripts/docs/Math; workflow files alone mutable; no old-index current-hash assumption',
                'mutable': sorted(mutable),
                'files': {rel: {'sha256': sha(root / rel), 'size': (root / rel).stat().st_size} for rel in targets(root, mutable)}}
        with path.open('x', encoding='utf-8') as handle:
            json.dump(body, handle, indent=2, ensure_ascii=False)
        print('PROTECTED', len(body['files']))
        return
    body = json.loads(path.read_text(encoding='utf-8'))
    changed = [rel for rel, info in body['files'].items()
               if not (root / rel).is_file() or sha(root / rel) != info['sha256']]
    result = {'passed': not changed, 'checked': len(body['files']), 'changed': changed,
              'baseline_sha256': sha(path)}
    with (out / 'PROTECTED_FINAL_CHECK.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    if changed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
