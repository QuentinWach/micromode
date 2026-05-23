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
uv run pytest --cov=micromode --cov-report=xml --cov-report=term-missing
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
2. Confirm `pyproject.toml` and `CHANGELOG.md` both use the same new release
   version. PyPI filenames are immutable, so never reuse a version that has
   reached PyPI.
3. Commit the release changes.
4. Tag the release with the same version from `pyproject.toml`:

```bash
git tag v$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
git push origin main --tags
```

The tag push runs `.github/workflows/publish.yml`, builds the distributions,
checks metadata, and publishes to PyPI using Trusted Publishing.

## Current Wheel Scope

The release workflow builds a pure-Python wheel and source distribution for
Python 3.10 through 3.13.
