"""Smoke-test a built wheel in a fresh virtual environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wheel",
        nargs="?",
        help="Wheel to install; defaults to the newest matching dist/micromode-*.whl",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", sys.executable),
        help="Python executable used to create the temporary virtual environment",
    )
    args = parser.parse_args()

    wheel = Path(args.wheel).resolve() if args.wheel else latest_wheel(python_tag(args.python))
    with tempfile.TemporaryDirectory() as tmp:
        venv_dir = Path(tmp) / "venv"
        run([args.python, "-m", "venv", str(venv_dir)])
        venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(venv_python), "-m", "pip", "install", str(wheel)])
        run([str(venv_python), str(ROOT / "scripts/smoke_wheel.py")])


def latest_wheel(required_tag: str) -> Path:
    wheels = sorted(
        (ROOT / "dist").glob(f"micromode-*-{required_tag}-*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not wheels:
        raise SystemExit(f"no {required_tag} wheel found in dist/")
    return wheels[0].resolve()


def python_tag(python: str) -> str:
    command = [
        python,
        "-c",
        "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
