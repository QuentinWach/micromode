# Release Checklist

MicroMode publishes as the `micromode` Python package.

## One-Time Setup

1. Confirm the package name is available on PyPI and TestPyPI.
2. Configure PyPI Trusted Publishing for:
   - PyPI project: `micromode`
   - Repository: `QuentinWach/micromode`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. Configure TestPyPI Trusted Publishing with the same repository and workflow,
   but environment `testpypi`.
4. In GitHub, create protected environments named `pypi` and `testpypi`.
   Require manual approval for `pypi`.
5. Configure Codecov for `QuentinWach/micromode` if the coverage badge should
   show uploaded coverage.

## Local Preflight

Run these before tagging:

```bash
uv sync --all-extras
env -u CONDA_PREFIX uv run maturin develop
uv run pytest --cov=micromode --cov-report=xml --cov-report=term-missing
cargo test
uv build
uv run twine check dist/*
./scripts/smoke_dist.sh
```

The smoke test installs the built wheel into a fresh virtual environment,
imports `micromode`, solves a tiny waveguide problem, checks power
normalization, and round-trips HDF5 output.

## TestPyPI Release

Use the `Release` workflow manually with:

```text
publish_target = testpypi
```

After it publishes, test from TestPyPI in a clean environment:

```bash
python -m venv /tmp/micromode-testpypi
/tmp/micromode-testpypi/bin/python -m pip install --upgrade pip
/tmp/micromode-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  micromode
/tmp/micromode-testpypi/bin/python scripts/smoke_wheel.py
```

## PyPI Release

1. Update `CHANGELOG.md`.
2. Commit the release changes.
3. Tag the release:

```bash
git tag v0.1.0a1
git push origin main --tags
```

The tag push runs `.github/workflows/publish.yml`, builds the distributions,
checks metadata, and publishes to PyPI using Trusted Publishing.

## Current Wheel Scope

The release workflow builds macOS and Linux wheels for Python 3.10 through
3.13. Windows wheels are intentionally not enabled yet because the sparse
backend depends on ARPACK/SuiteSparse-style native libraries. Add Windows only
after that native dependency story is reliable.
