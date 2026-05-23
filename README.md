# micromode

An **electromagnetic mode solver** using the **[FDFD method](https://en.wikipedia.org/wiki/Finite-difference_frequency-domain_method)** on a **[rectilinear Yee-grid](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method)**.
MicroMode is a transparent, grid-first SciPy mode solver for rasterized
photonics workflows.

```bash
pip install micromode
```

[![License](https://img.shields.io/github/license/QuentinWach/micromode)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/QuentinWach/micromode/tests.yml?branch=main&label=tests)](https://github.com/QuentinWach/micromode/actions/workflows/tests.yml)
![Coverage](docs/assets/coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/micromode)](https://pypi.org/project/micromode/)
![Status](https://img.shields.io/badge/status-alpha-orange)


## Why Use It?

- **Grid-first API**: pass already-rasterized material tensors and grid edges
  directly, with no required CAD or geometry model.
- **Readable SciPy implementation**: sparse operators are assembled in Python
  and solved with SciPy/ARPACK, so the numerical path can be inspected by users
  who do not want to trust a custom native solver.
- **Small integration surface**: MicroMode is the solver piece you embed after
  geometry has already been rasterized by an FDTD, FEM, or custom photonics
  pipeline.
- **Tensor-aware**: supports scalar, diagonal anisotropic, and full tensor material
  grids, plus transformed angle and bend solves.
- **Practical outputs**: coordinate-aware fields, `n_eff`, `k_eff`, mode area,
  polarization fractions, Lorentz overlaps, plotting, dataframe export, and
  HDF5 save/load.
- Works for both **2D cross sections and 1D slices**.

MicroMode is intentionally not a full simulation platform or geometry package.
That is the point: you give it a mode-plane material grid, and it returns guided
modes with the diagnostics and result helpers needed by downstream simulation
workflows.

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

MicroMode assembles the finite-difference sparse operators in Python and solves
the shift-invert eigenproblems with
[SciPy/ARPACK](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html).
That makes the numerical method easier for academic users to inspect and
debug, while still using a trusted sparse eigensolver stack.

This means MicroMode is differentiated by workflow and inspectability rather
than by a custom solver engine: it is built for users who already have
permittivity on a Yee grid and want a focused, auditable mode-solver component
that fits cleanly into a larger photonics pipeline.
