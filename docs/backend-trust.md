# Backend Trust Model

MicroMode has two backend roles:

- The Rust backend is the production solver. It is fast, portable, and does not
  require users to install ARPACK, SuiteSparse, BLAS/LAPACK bindings, or a
  Fortran toolchain.
- The SciPy reference backend is an optional audit path. It mirrors the sparse
  operators in Python and calls SciPy/ARPACK so the core numerical method can be
  inspected by users who are more comfortable reading Python.

The SciPy backend is selected explicitly:

```python
import micromode as mm

data = mm.solve_grid(
    eps_xx=eps,
    x_edges=x_edges,
    y_edges=y_edges,
    freqs=[freq],
    num_modes=2,
    target_neff=2.5,
    backend="scipy-reference",
)
```

Install the optional dependency with:

```bash
pip install "micromode[scipy]"
```

## Supported Scope

The reference backend covers the same core solve families as Rust:

- diagonal scalar or diagonal-anisotropic material grids,
- full tensor material grids,
- angle and bend transforms that route through the tensorial operator,
- PML stretching,
- one-dimensional slices, because they are padded into ordinary mode-plane
  grids before backend dispatch.

The remaining difference is operational rather than mathematical: the SciPy
backend depends on SciPy/ARPACK and is expected to be slower and less portable
than the Rust backend.

## What Is Compared

The test suite runs representative grids through both backends and checks:

- returned complex effective indices,
- sparse operator size and nonzero count,
- unit-power normalization diagnostics,
- Lorentz orthogonality diagnostics.

The covered cases include diagonal grids, PML, full-tensor/off-diagonal grids,
and transformed grids.

Run the focused cross-backend check with:

```bash
uv run --extra scipy pytest tests/test_micromode_api.py -k scipy_reference
```

Without the SciPy extra installed, the comparison test is skipped and the Rust
production tests still run.

## Reading The Code

The relevant files are:

- `python/micromode/scipy_reference.py`: readable Python/SciPy implementation of
  the diagonal and tensorial sparse paths.
- `python/micromode/raster.py`: public backend selection and `Result` wrapping.
- `src/operators.rs`: Rust Maxwell operator assembly.
- `src/eigensolve.rs`: Rust shift-invert Arnoldi and sparse LU path.
- `src/mode_solver.rs`: Rust field reconstruction, normalization, and Lorentz
  orthogonalization.

The intended trust chain is: inspect the Python reference, inspect the Rust
operator comments, then run the cross-backend test to verify that Rust and SciPy
agree on the supported cases.

## External References

MicroMode also keeps Tidy3D-oriented examples and fixtures because Tidy3D is a
recognizable reference point for photonics users. Tidy3D's
[public source docs](https://docs.flexcompute.com/projects/tidy3d/en/latest/_modules/tidy3d/components/mode/mode_solver.html)
show that its local mode solver import can fail when SciPy is unavailable, so
Tidy3D-style comparisons are useful as behavioral validation in addition to the
internal Rust-vs-SciPy checks. See the Tidy3D example in `examples/` and the
committed mode-solver fixture harness for existing comparison infrastructure.
