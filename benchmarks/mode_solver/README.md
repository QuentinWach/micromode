# Mode-Solver Reference Fixtures

This directory contains read-only helper utilities for committed mode-solver reference outputs.
The fixtures are static HDF5 result files plus compact JSON summaries and manifests.

Suites:

- `smoke`: small committed fixtures for regular checks.
- `extended`: additional heavier or specialized cases.

Inspect a suite:

```bash
uv run python benchmarks/compare_mode_solver_fixtures.py --suite smoke
```

Run the local raster solver against every fixture that has enough neutral metadata to reconstruct
the mode plane:

```bash
uv run python benchmarks/compare_mode_solver_fixtures.py --suite extended --run-local --report-json tmp/reference_fixture_validation_rust_sparse.json
```

Local fixture validation uses the Rust sparse backend by default. To run the same reconstructable
fixture recipes through the SciPy reference backend:

```bash
uv run --extra scipy python benchmarks/compare_mode_solver_fixtures.py \
  --suite extended \
  --run-local \
  --backend scipy_reference \
  --report-json tmp/reference_fixture_validation_scipy_reference.json
```

Local validation reports each case as `pass`, `fail`, or `unsupported`. `fail` means the local
solver ran but exceeded the fixture tolerance; `unsupported` means the case is explicitly classified
as outside the current production target, blocked by missing neutral raster metadata, or not yet
implemented in the fixture harness. Use both failure gates for CI-style checks:

```bash
uv run python benchmarks/compare_mode_solver_fixtures.py \
  --suite extended \
  --run-local \
  --fail-on-tolerance \
  --fail-on-production-gap
```

For reconstructable raster cases, the fixture field coordinates are treated as mode-plane grid
edges. Rectangular dielectric recipes are rasterized onto the same staggered diagonal epsilon
positions used by the mode solver; using one center-sampled scalar epsilon array for all components
validates the wrong Yee grid.

The package does not ship reference-data generation code or adapter dependencies. New reference
data should be generated outside the package and committed only as neutral result files.

Some reference cases intentionally cover behavior outside the first runtime target, such as PEC
boundary-condition fixtures. They are retained as future baselines, not as current support claims.

The extended suite includes baselines for sloped sidewalls, cylindrical rods, slot and rib
waveguides, cubic and polynomial interpolation, and non-default mode sorting metrics. Nonzero
angle-plus-bend transforms still need a validated tensorial reference run before a fixture is
committed.

## Backend Timing

Use the backend benchmark to track solve time and sparse-operator diagnostics across grid sizes:

```bash
uv run python benchmarks/micromode_backend_benchmark.py --grid 20x14 --grid 32x22
```

The benchmark uses MicroMode directly, writes JSON to `tmp/` by default, and records backend name,
operator size, nonzero count, residuals, elapsed time, and solved effective indices.
