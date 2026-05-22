# micromode

An **electromagnetic mode solver** using the **[FDFD method](https://en.wikipedia.org/wiki/Finite-difference_frequency-domain_method)** on a **[rectilinear Yee-grid](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method)**, written in native **[Rust](https://rust-lang.org/)**.

```bash
pip install micromode
```

[![License](https://img.shields.io/github/license/QuentinWach/micromode)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/QuentinWach/micromode/tests.yml?branch=main&label=tests)](https://github.com/QuentinWach/micromode/actions/workflows/tests.yml)
![Coverage](docs/assets/coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/micromode)](https://pypi.org/project/micromode/)
![Status](https://img.shields.io/badge/status-alpha-orange)


## Why Use It?

- **Grid-first API**: pass arrays directly, with no required geometry model.
- **Fast**, portable Rust sparse backend: one production solve path.
- **Auditable** optional SciPy reference backend for diagonal-grid checks.
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

## Examples


### Tidy3D Waveguide
![Tidy3D modal monitor example](docs/assets/tidy3d_modal_modes.png)

The Tidy3D modal monitor example recreates the strip-waveguide setup from
Flexcompute's modal sources and monitors notebook. It solves the first three
x-propagating modes of a silicon waveguide on a silica substrate and plots
`|Ey|` and `|Ez|` on the same y-z mode plane. (See [Tidy3D, "Defining Mode Sources and Monitors"](https://www.flexcompute.com/tidy3d/examples/notebooks/ModalSourcesMonitors/).)

```bash
uv run --extra dev python examples/tidy3d_modal_sources_monitors.py
```

### Hybridization Sweep
![Hybridization sweep example](docs/assets/hybridization_sweep.png)

The SOI hybridization example sweeps the width of a 220 nm silicon ridge and
solves several modes at each step. It shows how nearby modes exchange character
as the geometry changes by plotting effective index and TE fraction across the
sweep, then rendering representative field profiles.

```bash
uv run --extra dev python examples/soi_hybridization_sweep.py
```


## Physics

MicroMode solves the source-free frequency-domain Maxwell equations on a rasterized Yee mode plane, $\nabla\times\mathbf{E}=-i\omega\mu\mathbf{H}, \; \nabla\times\mathbf{H}=i\omega\epsilon\mathbf{E},$ with modal fields $\mathbf{E},\mathbf{H}\propto e^{i k_0 n_\mathrm{eff} z}.$

On diagonal material grids this becomes a transverse eigenproblem,
while full tensor or transformed grids use a first-order tensorial form. The
detailed derivation is in [docs/physics-model.md](docs/physics-model.md), and
the public solver controls are summarized in [docs/mode-solver-methods.md](docs/mode-solver-methods.md).


## Solver

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

For users who want an executable Python reference, MicroMode also provides an
optional SciPy/ARPACK backend for the untransformed diagonal sparse path:

```bash
pip install "micromode[scipy]"
```

```python
data = mm.solve_modes(..., backend="scipy-reference")
```

This backend is intentionally slower and narrower than Rust. Its purpose is to
make the core diagonal eigenproblem easy to inspect in Python and to validate
that the production Rust backend returns the same effective indices and
normalization diagnostics on supported cases. See
[docs/backend-trust.md](docs/backend-trust.md).
