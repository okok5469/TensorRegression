"""Validate all 32 source files and, optionally, the actual import location."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def check(root, import_check=False, source=None):
    root = Path(root).resolve()
    source = Path(source).resolve() if source is not None else root / 'src/tensor_abess'
    expected = json.loads((root / 'RELEASE_MANIFEST.json').read_text())['source_sha256']
    actual = {'src/tensor_abess/' + p.relative_to(source).as_posix() for p in source.rglob('*.py')}
    if actual != set(expected) or len(actual) != 32:
        raise ValueError('Incomplete or additional core source files')
    for name, value in expected.items():
        path = source / Path(name).relative_to('src/tensor_abess')
        if hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise ValueError('Source checksum mismatch: ' + name)
    if import_check:
        sys.path.insert(0, str(source.parent))
        import tensor_abess
        from tensor_abess.tensor.abess import tensor_abess as entry
        import tensor_abess.tensor.bcd as bcd
        if Path(tensor_abess.__file__).resolve().parent != source:
            raise ValueError('Wrong tensor_abess import path')
        if Path(bcd.__file__).resolve() != source / 'tensor/bcd.py':
            raise ValueError('Wrong BCD import path')
        if len(bcd._candidate_size_triples(6, 6, 7, 280)) != 252:
            raise ValueError('Wrong A2 grid contract')
        assert callable(entry)
    return {'version': 'v5.2-search-final', 'verified_core_files': 32,
            'import_checked': import_check}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path, help='Target project tensor_abess directory; use the delivered manifest')
    parser.add_argument('--import-check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.import_check, args.source)))
