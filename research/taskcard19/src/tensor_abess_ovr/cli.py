"""The portable, single-process task-card-19 command line."""
import os

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Tensor-ABESS OvR-LS 0.1.0: frozen regression scores for classification')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify-release')
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--output', required=True)
    for name in ('validate-data', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--data', required=True)
        command.add_argument('--config', required=True)
        if name == 'run':
            command.add_argument('--output', required=True)
    for name in ('validate-return', 'pack-return'):
        command = commands.add_parser(name)
        command.add_argument('--run', required=True)
        if name == 'pack-return':
            command.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'verify-release':
        from .identity import verify_release
        result = verify_release()
    elif args.command == 'smoke':
        from .smoke import run_smoke
        result = run_smoke(args.output)
    elif args.command == 'validate-data':
        from .data import load_dataset
        data = load_dataset(args.data, args.config)
        result = {'status': 'PASS', 'samples': len(data['sample_ids']), 'identity': data['identity']}
    elif args.command == 'run':
        from .runner import run
        result = run(args.data, args.config, args.output)
    elif args.command == 'validate-return':
        from .returns import validate_return
        result = validate_return(args.run)
    else:
        from .returns import pack_return
        result = pack_return(args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
