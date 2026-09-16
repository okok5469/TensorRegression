"""Small explicit persistence helpers; no module-level state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=str):
        digest.update(str(path.resolve()).encode('utf-8'))
        if not path.is_file():
            digest.update(b'MISSING')
            continue
        stat = path.stat()
        digest.update(f'{stat.st_size}:{stat.st_mtime_ns}'.encode('ascii'))
        if path.suffix.lower() != '.eeg':
            digest.update(path.read_bytes())
    return digest.hexdigest()
