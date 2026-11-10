"""Run a checked command with faithful combined output (PowerShell-safe)."""
import argparse
from pathlib import Path
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        result = subprocess.run(args.command, stdout=stream, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    print(path.read_text(encoding='utf-8')[-5000:])
    raise SystemExit(result.returncode)
