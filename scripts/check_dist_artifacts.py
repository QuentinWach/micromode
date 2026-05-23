"""Validate release artifacts before publishing them to a package index."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

WHEEL_RE = re.compile(
    r"^micromode-(?P<version>[^-]+)-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^.]+(?:\.[^.]+)*)\.whl$"
)


def main() -> None:
    """Validate wheel and source-distribution artifact coverage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs="?", default="dist", help="Directory containing release artifacts")
    parser.add_argument(
        "--require-cpython",
        nargs="*",
        default=[],
        metavar="VERSION",
        help="Require at least one wheel for each CPython minor version, for example 3.10",
    )
    parser.add_argument(
        "--require-platform",
        nargs="*",
        default=[],
        metavar="PREFIX",
        help="Require at least one wheel platform tag with each prefix, for example macosx or manylinux",
    )
    parser.add_argument(
        "--allow-pure-python",
        action="store_true",
        help="Allow a py3-none-any wheel instead of platform-specific wheels.",
    )
    args = parser.parse_args()

    dist = Path(args.dist)
    wheels = sorted(dist.glob("micromode-*.whl"))
    sdists = sorted(dist.glob("micromode-*.tar.gz"))
    require(sdists, f"no source distribution found in {dist}")
    require(wheels, f"no wheels found in {dist}")

    platform_tags: set[str] = set()
    python_tags: set[str] = set()
    for wheel in wheels:
        match = WHEEL_RE.match(wheel.name)
        if match is None:
            raise SystemExit(f"unexpected wheel filename: {wheel.name}")
        platform = match["platform"]
        python_tag = match["python"]
        platform_tags.update(platform.split("."))
        if args.allow_pure_python and python_tag == "py3" and match["abi"] == "none" and platform == "any":
            python_tags.update(args.require_cpython)
            platform_tags.update(args.require_platform)
            continue
        if python_tag.startswith("cp") and python_tag[2:].isdigit():
            version_digits = python_tag[2:]
            python_tags.add(f"3.{version_digits[1:]}")
        require(
            not any(tag.startswith("linux_") for tag in platform.split(".")),
            f"{wheel.name} uses a native linux tag; PyPI requires manylinux or musllinux wheels",
        )

    missing = sorted(set(args.require_cpython) - python_tags)
    require(not missing, f"missing wheels for CPython: {', '.join(missing)}")
    missing_platforms = sorted(
        prefix for prefix in args.require_platform if not any(tag.startswith(prefix) for tag in platform_tags)
    )
    require(not missing_platforms, f"missing wheel platform families: {', '.join(missing_platforms)}")
    print(f"release artifacts look publishable: {len(sdists)} sdist(s), {len(wheels)} wheel(s)")


def require(condition: object, message: str) -> None:
    """Raise SystemExit with a message when a condition is false."""
    if not condition:
        raise SystemExit(message)


if __name__ == "__main__":
    main()
