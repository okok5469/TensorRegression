#!/bin/bash
set -euo pipefail
task_root="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
python -B "$task_root/scripts/release_v52/release_control.py" inventory --root "$task_root"
