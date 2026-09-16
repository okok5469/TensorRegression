"""Configuration with an explicit path anchor (the YAML file directory)."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or not isinstance(config.get('data'), dict):
        raise ValueError('Configuration must contain a data mapping')
    paths = {}
    for key, value in {**config['data'], 'output_dir': config['output_dir']}.items():
        if value is None:
            paths[key] = None
            continue
        candidate = Path(value).expanduser()
        paths[key] = (path.parent / candidate).resolve()
    if not paths.get('eeg_root') or not paths.get('tmt_csv'):
        raise ValueError('eeg_root and tmt_csv must be explicit paths')
    paths['config_file'] = path
    config['paths'] = paths
    return config
