"""Run from the repository root: python -m empirical_analysis.lemon."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run_stage, stage_names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='LEMON EEG pipeline with ordered data gates')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--stage', choices=(*stage_names(), 'all'), required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        stages = stage_names() if args.stage == 'all' else (args.stage,)
        for stage in stages:
            print(f'LEMON: {stage}', flush=True)
            run_stage(config, stage)
            print(f'Completed {stage}; outputs: {config["paths"]["output_dir"]}', flush=True)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
