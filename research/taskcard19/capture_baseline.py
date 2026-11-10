"""Read-only frozen-source and workspace baseline for task card 19."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def capture(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'doc/releases/v5.2-search-final/RELEASE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    expected = manifest['source_sha256']
    actual = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in expected}
    if len(expected) != 32 or actual != expected:
        raise RuntimeError('Frozen production source identity mismatch')
    git = subprocess.run(['git', 'status', '--porcelain=v1', '--untracked-files=no'], cwd=root, capture_output=True, check=True).stdout.decode('utf-8')
    protected = list(expected) + list(manifest.get('simulation_sha256', {}))
    protected += ['paper/sections/02_method.md', 'paper/sections/04_simulation.md', 'paper/appendix/simulation_protocol.md', 'doc/taskcard18/FINAL_VALIDATION.json']
    return {'source_sha256': actual, 'protected_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in protected}, 'git_tracked_status': git, 'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'root_workflow_files': [n for n in ('task_plan.md', 'progress.md', 'findings.md', 'task_issue.md') if (root/n).exists()]}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    result = capture(root)
    target = root / 'doc/taskcard19/BASELINE.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('32 frozen production sources match release; workspace baseline saved')
