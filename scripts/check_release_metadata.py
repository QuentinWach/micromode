"""Check release metadata that is easy to forget before publishing."""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pyproject = load_toml("pyproject.toml")
    cargo = load_toml("Cargo.toml")
    project = pyproject["project"]
    package = cargo["package"]
    publish_workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    require(project["name"] == "micromode", "Python package name must be micromode")
    require(project["version"] != "0.0.0", "Python package version must not be 0.0.0")
    require(
        package["version"] == python_to_cargo_version(project["version"]),
        "Rust crate version must match the Python package version",
    )
    require(project["requires-python"] == ">=3.10,<3.14", "Python support must match the release wheel matrix")
    require(package["version"] != "0.0.0", "Rust crate version must not be 0.0.0")
    require(project["license"] == "Apache-2.0", "Python package license must be Apache-2.0")
    require(package["license"] == "Apache-2.0", "Rust crate license must be Apache-2.0")
    require((ROOT / "LICENSE").exists(), "LICENSE file is missing")
    require((ROOT / "CHANGELOG.md").exists(), "CHANGELOG.md is missing")
    require(project.get("authors"), "project.authors is missing")
    require(project.get("classifiers"), "project.classifiers is missing")
    require(project.get("urls"), "project.urls is missing")
    require(package.get("repository"), "Cargo repository is missing")
    require((ROOT / ".github/workflows/publish.yml").exists(), "publish workflow is missing")
    require((ROOT / ".github/workflows/tests.yml").exists(), "tests workflow is missing")
    require((ROOT / "scripts/smoke_wheel.py").exists(), "wheel smoke test is missing")
    for version in ("3.10", "3.11", "3.12", "3.13"):
        require(
            f'"{version}"' in publish_workflow,
            f"publish workflow is missing a Python {version} wheel build",
        )
    require("--compatibility pypi" in publish_workflow, "publish workflow must request PyPI-compatible wheels")
    require("--auditwheel repair" in publish_workflow, "Linux release wheels must be auditwheel-repaired")
    require("windows-latest" in publish_workflow, "publish workflow is missing Windows wheel builds")
    require(
        "--require-platform macosx manylinux win" in publish_workflow,
        "release artifact check must require macOS, manylinux, and Windows wheels",
    )

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    require(project["version"] in changelog, "Python version is not mentioned in CHANGELOG.md")
    require(
        re.fullmatch(r"\d+\.\d+\.\d+(a\d+|b\d+|rc\d+)?", project["version"]),
        f"Python version {project['version']!r} is not a normal PyPI release version",
    )
    require_tag_matches_version(project["version"])
    print(f"release metadata looks ready for micromode {project['version']}")


def load_toml(path: str) -> dict:
    with (ROOT / path).open("rb") as handle:
        return tomllib.load(handle)


def python_to_cargo_version(version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?", version)
    require(match is not None, f"Python version {version!r} is not a supported release version")
    base, phase, number = match.groups()
    if phase is None:
        return base
    phase_names = {"a": "alpha", "b": "beta", "rc": "rc"}
    return f"{base}-{phase_names[phase]}.{number}"


def require_tag_matches_version(version: str) -> None:
    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return

    tag = os.environ.get("GITHUB_REF_NAME", "")
    require(tag == f"v{version}", f"Git tag {tag!r} must match Python version {version!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


if __name__ == "__main__":
    main()
