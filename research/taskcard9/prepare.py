"""Freeze Task9 bounded diagnostic protocol before any new stream replay."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import scipy


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / 'doc/taskcard9'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    previous = root / 'numerical_simulation/outputs_selection_research_task8/plan.json'
    plan = json.loads(previous.read_text(encoding='utf-8'))
    ids = [0, 9, 15, 18, 24, 33, 72, 81, 87, 90, 96, 105]
    requests = [r for r in plan['requests'] if r['cohort'] == 'development' and r['task_id'] in ids]
    assert len(requests) == 12 and all(r['replicate'] == 0 for r in requests)
    assert [r['config']['n_train'] for r in requests] == [100] * 6 + [400] * 6
    for rel, expected in plan['dependencies'].items():
        assert sha(root / rel) == expected, rel
    source = root / 'doc/algorithm_versions/v5.2-search-final/src'
    actual = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*.py')}
    assert actual == plan['core_sha256'] and len(actual) == 32
    current = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
    assert all(current[k] == plan['environment'][k] for k in current)
    body = {'created_utc': datetime.now(timezone.utc).isoformat(), 'research_only': True,
            'release_approved': False, 'base_version': 'v5.2-search-final',
            'source_sha256': actual, 'dependencies': plan['dependencies'],
            'environment': current, 'historical_plan_sha256': sha(previous),
            'task9_design_sha256': sha(out / 'DESIGN.md'),
            'taskcard_sha256': sha(next(root.glob('*/*9.md'))),
            'test_replay_requests': requests, 'max_replay_cases': 12,
            'replay_wall_seconds_max': 1800, 'warm_wall_seconds_max': 900,
            'new_support_fits_in_pool_analysis': 0,
            'rules': [{'rule': rule, 'alpha': alpha, 'scale': scale, 'gamma': 1.0}
                      for rule, alpha, scale in [('A0',1.5,1.),('A1',2.,1.),('A2',1.5,0.),('A3',2.,0.)]],
            'selection_rule': 'first minimum; score<=min+scale*sqrt(q_min*log(n)); min(q,score,sum(sizes)); stable original order',
            'case_selection': 'development replicate0; each n100/400 crosses three SNR and two truth shapes, rho parity of SNR ordinal+shape ordinal; no outcome input',
            'gates_before_new_metrics': ['frozen source/dependencies/environment', 'train/truth/support exact fingerprints',
                                        'saved coefficient/support identities', 'test stream exact fingerprint', 'saved A0/A3 PMSE/signal-MSE numerical reproduction'],
            'inference_scope': 'Task8 data already seen; diagnostic only; no new independent confirmation',
            'warm_scope': 'Fixed-size Cold retained plus extra warm initial support using unchanged formal fixed solver; full adaptive integration/performance NOT_RUN'}
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as handle:
        json.dump(body, handle, ensure_ascii=False, indent=2)
    print('PROTOCOL', sha(out / 'PROTOCOL.json'), 'CASES', ids)


if __name__ == '__main__':
    main()
