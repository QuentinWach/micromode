#!/usr/bin/env bash
set -euo pipefail

wheel="${1:-}"
if [[ -z "$wheel" ]]; then
  wheel="$(ls -t dist/micromode-*.whl | head -n 1)"
fi
python_bin="${PYTHON:-python}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

"$python_bin" -m venv "$tmpdir/venv"
"$tmpdir/venv/bin/python" -m pip install --upgrade pip
"$tmpdir/venv/bin/python" -m pip install "$wheel"
"$tmpdir/venv/bin/python" scripts/smoke_wheel.py
