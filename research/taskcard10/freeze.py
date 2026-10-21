"""Freeze bounded research identities; never creates data or launches fits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import scipy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('pilot','formal'),required=True)
    parser.add_argument('--replicates',type=int)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root/'doc/taskcard10'
    stage = args.stage
    source = root/'doc/algorithm_versions/v5.2-search-final/src'
    expected = json.loads((root/'doc/algorithm_versions/v5.2-search-final/VERSION.json').read_text(encoding='utf-8'))['source_sha256']
    assert len(expected) == 32
    for rel,digest in expected.items():
        assert sha(source/rel) == digest and sha(root/'src'/rel) == digest
    names = ('research/taskcard10/cv.py','research/taskcard10/experiment.py',
             'research/taskcard10/test_cv.py','research/taskcard10/test_experiment.py',
             'research/taskcard10/audit.py','research/taskcard10/test_audit.py',
             'research/taskcard10/gates.py','research/taskcard10/freeze.py',
             'scripts/taskcard7/lemon.py','numerical_simulation/version_validation/worker.py',
             'numerical_simulation/data_generation.py','numerical_simulation/metrics.py')
    assert all((root/name).is_file() for name in names), 'Finish implementation and correctness tests first'
    environment = {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                   'executable':sys.executable,'platform':platform.platform(),
                   'threads':{name:os.environ.get(name) for name in
                              ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    assert all(value == '1' for value in environment['threads'].values())
    intent = json.loads((out/'GATE_INTENT.json').read_text(encoding='utf-8'))
    gates = json.loads((out/'CORRECTNESS_GATES.json').read_text(encoding='utf-8'))
    assert gates['passed'] and gates['failures'] == gates['errors'] == 0
    for name,digest in gates['source_and_tests_sha256'].items():
        assert sha(root/name) == digest, 'Source changed after correctness gate: '+name
    body = {'created_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'research_only':True,'release_approved':False,
            'base_version':'v5.2-search-final','source_files':{name:sha(root/name) for name in names},
            'correctness_gate_sha256':sha(out/'CORRECTNESS_GATES.json'),
            'core_sha256':expected,'environment':environment,'gate_intent_sha256':sha(out/'GATE_INTENT.json'),
            'taskcard_sha256':sha(next(root.glob('*/*10.md'))),'design_sha256':sha(out/'DESIGN_AND_PROTOCOL.md'),
            'cv_folds':3,'preprocessing':'identity_per_fold_zero_mean_DGP',
            'capacity':'formal largest-cap shrink from n-only floor, common full/inner componentwise minimum; n100334/n400566',
            'candidate_program':'formal marginal initialized fixed BCD, full-training native fixed pool reuse; adaptive terminal excluded for A1/A2',
            'cv_tie':'absolute MSE tolerance1e-12, then q and stable lexicographic shape/config ID',
            'development_entry_gate':intent['gates'],'promotion_gates':intent['gates'],
            'pilot':{'workers':args.workers,'wall_time_limit_seconds':1800,'case_timeout_seconds':900,'replicates':1},
            'formal':{'workers':args.workers,'wall_time_limit_seconds':7200,'case_timeout_seconds':900,
                      'replicates':args.replicates},
            'source_identity':{**{'doc/algorithm_versions/v5.2-search-final/src/'+rel:digest
                                  for rel,digest in expected.items()},
                               **{name:sha(root/name) for name in names}},
            'seed_bases':{'pilot':3101810000,'development':3501810000,'confirmation':3901810000},
            'seed_offsets':{'test':50000000,'split':100000000},
            'no_independent_confirmation_reuse':'Pilot, development, all materialized historical train/test seed fields and full-input hashes excluded'}
    if stage == 'formal':
        assert args.replicates in (2,3), 'Insufficient budget for multiple repetitions is a declared stop, not silent single-repeat confirmation'
        pilot = out/'results/pilot/completion.json'
        assert pilot.is_file(), 'Pilot must finish before formal freeze'
        body['pilot_completion_sha256'] = sha(pilot)
    path = out/(stage.upper()+'_PROTOCOL.json')
    with path.open('x',encoding='utf-8') as handle:
        json.dump(body,handle,ensure_ascii=False,indent=2)
        handle.write('\n')
    print('FROZEN',stage,sha(path))


if __name__ == '__main__':
    main()
