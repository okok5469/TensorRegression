"""Task10 protection: snapshot current approved historical bytes, no fitting."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    out = root/'doc/taskcard10'
    path = out/'PROTECTED_BASELINE.json'
    if sys.argv[1] == 'snapshot':
        names = set()
        for manifest in ('doc/taskcard9/PROTECTED_BASELINE.json','doc/taskcard9/DELIVERY_INDEX.json'):
            names.update(json.loads((root/manifest).read_text(encoding='utf-8'))['files'])
            names.add(manifest)
        names.update(p.relative_to(root).as_posix() for p in (root/'doc').glob('*.md'))
        names -= {'doc/task_plan.md','doc/findings.md','doc/progress.md'}
        body = {'created_utc':datetime.now(timezone.utc).isoformat(),
                'scope':'Current approved historical bytes, all Task9 outputs and scientific source; only three doc workflow files mutable',
                'files':{name:{'size':(root/name).stat().st_size,'sha256':sha(root/name)} for name in sorted(names)}}
        with path.open('x',encoding='utf-8') as handle:
            json.dump(body,handle,indent=2,ensure_ascii=False)
        status = subprocess.run(['git','status','--porcelain','-z','--untracked-files=no'],capture_output=True,check=True).stdout.decode('utf-8').split('\0')
        with (out/'GIT_INITIAL.json').open('x',encoding='utf-8') as handle:
            json.dump(status,handle,indent=2)
        print('PROTECTED',len(body['files']))
        return
    body = json.loads(path.read_text(encoding='utf-8'))
    changed = [name for name,v in body['files'].items() if not (root/name).is_file() or sha(root/name) != v['sha256']]
    result = {'passed':not changed,'checked':len(body['files']),'changed':changed,'baseline_sha256':sha(path)}
    with (out/'PROTECTED_FINAL_CHECK.json').open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result))
    assert not changed


if __name__ == '__main__':
    main()
