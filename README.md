# micromode

A minimal FDFD electromagnetic mode solver for rasterized waveguide cross sections with a Rust core made to be a standard plugin for FDTD engines.

[![License](https://img.shields.io/github/license/QuentinWach/micromode)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/QuentinWach/micromode/tests.yml?branch=main&label=tests)](https://github.com/QuentinWach/micromode/actions/workflows/tests.yml)
![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)

```bash
pip install micromode
```

## Supported Platforms

The Python package uses the portable Rust sparse backend. It does not require
external native sparse-solver libraries at install time.

Published wheels are built for:

- Linux, CPython 3.10-3.13
- macOS Intel and Apple Silicon, CPython 3.10-3.13

Source builds should work on platforms with a supported Rust toolchain and
Python 3.10-3.13. Windows source builds use the same portable backend, but
Windows wheels are not release-tested yet.

## Source Builds

For the portable backend, install Rust and build normally:

```bash
pip install .
```

or, for local development:

```bash
uv sync --all-extras
uv run maturin develop
```

## Performance

MicroMode is designed to make high-performance mode solving available without
requiring users to install external solver stacks. The production backend is a
portable Rust sparse shift-invert eigensolver, so source installs and wheels do
not depend on ARPACK, UMFPACK, SuiteSparse, BLAS/LAPACK, or a Fortran compiler.
That matters for simulation workflows that need to run in CI, notebooks,
container images, FDTD plugins, and cross-platform design tools.

The native solver is not a dense fallback. It uses sparse finite-difference
operators throughout, applies AMD fill-reducing ordering before sparse LU
factorization, stores LU factors in a packed format for repeated triangular
solves, and runs an Arnoldi iteration targeted around the requested effective
index. The Arnoldi stage uses shift-invert, adaptive Ritz-pair checkpointing,
early stopping once requested modes are stable, and selective Ritz vector
reconstruction so work is spent on the modes that will actually be returned.

On the repository benchmark problem, the pure Rust backend solves larger grids
in the same performance class as the previous optional UMFPACK-backed path while
remaining much easier to install and distribute. For example, a release build on
an Apple Silicon development machine solves an `80x50` diagonal benchmark grid
in roughly `90 ms` for two modes with residuals around `1e-12`. Exact timings
depend on hardware and problem shape, but the important point is architectural:
MicroMode keeps the deployability of a pure Rust package without giving up the
sparse-solver performance expected for practical waveguide grids.

## Why Use It?

- Grid-first API: pass arrays directly, with no required geometry model.
- Fast, portable Rust sparse backend: one production solve path.
- Practical outputs: fields, `n_eff`, `k_eff`, mode area, polarization fractions,
  Lorentz overlaps, plotting, dataframe export, and HDF5 save/load.
- Tensor-aware: supports scalar, diagonal anisotropic, and full tensor material
  grids.
- Works for both 2D cross sections and 1D slices.

You give it a material grid. It returns guided modes: effective indices, six-component fields, polarization metrics, mode area, overlaps, diagnostics, plots, and HDF5 output. MicroMode is intentionally not a CAD or geometry package. It is the solver piece you use after geometry has already been rasterized onto a mode-plane grid.


## Quick Start

```python
import micromode as mm

wavelength_um = 1.55
freq = mm.C_0 / wavelength_um

# Arrays from your own rasterizer.
eps_xx, x_edges, y_edges = mode_plane_arrays(...)

materials = mm.Materials.from_diagonal(
    eps_xx=eps_xx,
    x_edges=x_edges,
    y_edges=y_edges,
)

data = mm.solve_modes(
    material_grid=materials,
    freqs=[freq],
    num_modes=2,
    target_neff=2.5,
)

print(data.n_eff.values)
data.plot_field("Ex", mode_index=0)
data.to_hdf5("modes.h5")
```
