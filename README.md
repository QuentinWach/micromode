# micromode

An electromagnetic mode solver using the FDFD method on a regular Yee-grid, written in native Rust.

[![License](https://img.shields.io/github/license/QuentinWach/micromode)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/QuentinWach/micromode/tests.yml?branch=main&label=tests)](https://github.com/QuentinWach/micromode/actions/workflows/tests.yml)
![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)

```bash
pip install micromode
```


## Why Use It?

- **Grid-first API**: pass arrays directly, with no required geometry model.
- **Fast**, portable Rust sparse backend: one production solve path.
- **Practical** outputs: fields, `n_eff`, `k_eff`, mode area, polarization fractions,
  Lorentz overlaps, plotting, dataframe export, and HDF5 save/load.
- **Tensor-aware**: supports scalar, diagonal anisotropic, and full tensor material
  grids.
- Works for both **2D cross sections and 1D slices**.

You give it a material grid. It returns guided modes: effective indices, six-component fields, polarization metrics, mode area, overlaps, diagnostics, plots, and HDF5 output. MicroMode is intentionally not a CAD or geometry package. It is the solver piece you use after geometry has already been rasterized onto a mode-plane grid.

_Micromode is the **default mode solver** in the [BEAMZ FDTD engine](https://github.com/beamzorg/beamz)._


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


## Physics Model

MicroMode solves the source-free frequency-domain Maxwell equations on a
rasterized Yee mode plane,

$$
\nabla\times\mathbf{E}=-i\omega\mu\mathbf{H},
\qquad
\nabla\times\mathbf{H}=i\omega\epsilon\mathbf{E},
$$

with modal fields

$$
\mathbf{E},\mathbf{H}\propto e^{i k_0 n_\mathrm{eff} z}.
$$

On diagonal material grids this becomes a transverse eigenproblem,

$$
A_\mathrm{diag}
\begin{bmatrix}E_x\\E_y\end{bmatrix}
=
-n_\mathrm{eff}^2
\begin{bmatrix}E_x\\E_y\end{bmatrix}
$$

while full tensor or transformed grids use a first-order tensorial form. The
detailed derivation is in [docs/physics-model.md](docs/physics-model.md), and
the public solver controls are summarized in
[docs/mode-solver-methods.md](docs/mode-solver-methods.md).


## High Performance

MicroMode is designed to make high-performance mode solving available without
requiring users to install external solver stacks. The production backend is a
**portable Rust [sparse](https://en.wikipedia.org/wiki/Sparse_matrix)
[shift-invert](https://en.wikipedia.org/wiki/Preconditioner#Spectral_transformation)
[eigensolver](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)**, so
source installs and wheels do **not** depend on
[ARPACK](https://en.wikipedia.org/wiki/ARPACK),
[UMFPACK](https://en.wikipedia.org/wiki/UMFPACK),
[SuiteSparse](https://en.wikipedia.org/wiki/SuiteSparse),
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)/
[LAPACK](https://en.wikipedia.org/wiki/LAPACK), or a Fortran compiler.
That matters for simulation workflows that need to run in CI, notebooks,
container images, FDTD plugins, and cross-platform design tools.

The native solver is **not a dense fallback**. It uses
[sparse](https://en.wikipedia.org/wiki/Sparse_matrix)
[finite-difference](https://en.wikipedia.org/wiki/Finite_difference_method)
operators throughout, applies
[AMD fill-reducing ordering](https://en.wikipedia.org/wiki/Minimum_degree_algorithm)
before sparse [LU factorization](https://en.wikipedia.org/wiki/LU_decomposition),
stores LU factors in a packed format for repeated triangular solves, and runs an
[Arnoldi iteration](https://en.wikipedia.org/wiki/Arnoldi_iteration) targeted
around the requested effective index. The Arnoldi stage uses
**shift-invert**, adaptive
[Ritz-pair](https://en.wikipedia.org/wiki/Ritz_method) checkpointing, early
stopping once requested modes are stable, and selective Ritz vector
reconstruction so work is spent on the modes that will actually be returned.

On the repository benchmark problem, the **pure Rust backend** solves larger grids
in the same performance class as the previous optional UMFPACK-backed path while
remaining much easier to install and distribute. For example, a release build on
an Apple Silicon development machine solves an `80x50` diagonal benchmark grid
in roughly **`90 ms` for two modes** with residuals around **`1e-12`**. Exact
timings depend on hardware and problem shape, but the important point is
architectural: MicroMode keeps the **deployability of a pure Rust package**
without giving up the sparse-solver performance expected for practical
waveguide grids.
