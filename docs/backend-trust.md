# Solver Trust Model

MicroMode uses one solver implementation: a Python/SciPy sparse mode solver.
The finite-difference operators are assembled in inspectable Python code and
the eigenproblems are solved with SciPy/ARPACK.

## Supported Scope

The SciPy solver covers the core solve families:

- diagonal scalar or diagonal-anisotropic material grids,
- full tensor material grids,
- angle and bend transforms that route through the tensorial operator,
- PML stretching,
- one-dimensional slices, because they are padded into ordinary mode-plane
  grids before solver dispatch.

## What Is Tested

The test suite checks representative grids for:

- returned complex effective indices,
- sparse operator size and nonzero count,
- unit-power normalization diagnostics,
- Lorentz orthogonality diagnostics.

The covered cases include diagonal grids, PML, full-tensor/off-diagonal grids,
and transformed grids.

Run the focused solver checks with:

```bash
uv run pytest tests/test_micromode_api.py
```

## Reading The Code

The relevant files are:

- `python/micromode/scipy_reference.py`: Python/SciPy implementation of the
  diagonal and tensorial sparse paths.
- `python/micromode/raster.py`: public solve orchestration and `Result`
  wrapping.

The intended trust chain is: inspect the Python/SciPy implementation, then run
the fixture and API tests to verify behavior on representative mode-solver
cases.

## External References

MicroMode keeps Tidy3D-oriented examples and fixtures because Tidy3D is a
recognizable reference point for photonics users. See the Tidy3D example in
`examples/` and the committed mode-solver fixture harness for comparison
infrastructure.
