# Backend Trust Model

MicroMode has two backend roles:

- The Rust backend is the production solver. It is fast, portable, and does not
  require users to install ARPACK, SuiteSparse, BLAS/LAPACK bindings, or a
  Fortran toolchain.
- The SciPy reference backend is an optional audit path. It mirrors the diagonal
  sparse operator in Python and calls SciPy/ARPACK so the core numerical method
  can be inspected by users who are more comfortable reading Python.

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

The first reference backend intentionally covers only the smallest useful
surface:

- diagonal scalar or diagonal-anisotropic material grids,
- no angle or bend transform,
- no PML,
- the same reduced transverse eigenproblem used by the Rust diagonal backend.

Full tensor grids, transformed grids, and PML remain Rust-only for now. That
keeps the reference implementation short enough to audit.

## What Is Compared

The test suite runs the same diagonal grid through both backends and checks:

- returned complex effective indices,
- sparse operator size and nonzero count,
- unit-power normalization diagnostics,
- Lorentz orthogonality diagnostics.

Run the focused cross-backend check with:

```bash
uv run --extra scipy pytest tests/test_micromode_api.py -k scipy_reference
```

Without the SciPy extra installed, the comparison test is skipped and the Rust
production tests still run.

## Reading The Code

The relevant files are:

- `python/micromode/scipy_reference.py`: readable Python/SciPy implementation of
  the diagonal sparse path.
- `python/micromode/raster.py`: public backend selection and `Result` wrapping.
- `src/operators.rs`: Rust Maxwell operator assembly.
- `src/eigensolve.rs`: Rust shift-invert Arnoldi and sparse LU path.
- `src/mode_solver.rs`: Rust field reconstruction, normalization, and Lorentz
  orthogonalization.

The intended trust chain is: inspect the Python reference, inspect the Rust
operator comments, then run the cross-backend test to verify that Rust and SciPy
agree on the supported diagonal cases.
