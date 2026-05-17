#!/usr/bin/env bash
set -euo pipefail

wheel="${1:-}"
python_bin="${PYTHON:-python}"

if [[ -n "$wheel" ]]; then
  "$python_bin" scripts/smoke_dist.py --python "$python_bin" "$wheel"
else
  "$python_bin" scripts/smoke_dist.py --python "$python_bin"
fi
