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
    """Install a built wheel into a temporary environment and smoke-test it."""
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
    """Find the newest compatible wheel in dist."""
    wheels = sorted((ROOT / "dist").glob(f"micromode-*-{required_tag}-*.whl"), key=_mtime, reverse=True)
    if not wheels:
        wheels = sorted((ROOT / "dist").glob("micromode-*-py3-none-any.whl"), key=_mtime, reverse=True)
    if not wheels:
        raise SystemExit(f"no {required_tag} or py3-none-any wheel found in dist/")
    return wheels[0].resolve()


def _mtime(path: Path) -> float:
    """Return a path modification timestamp for sorting."""
    return path.stat().st_mtime


def python_tag(python: str) -> str:
    """Return the CPython wheel tag for an interpreter."""
    command = [
        python,
        "-c",
        "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def run(command: list[str]) -> None:
    """Run a subprocess command and fail on nonzero exit."""
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
