# Material Grid Demos

Use the grid-first demos to see MicroMode solve directly from rasterized
material arrays. They do not use reference fixtures or call any external
solver. Each example constructs raw NumPy material arrays, wraps them in
`Materials`, and calls `solve_modes()`:

```bash
uv run --extra dev python examples/material_grid_demos.py
```

The script writes PNG files to `examples/material_grid_outputs/` by default.

Available cases:

- `strip_grid`: rectangular silicon strip encoded directly as pixels.
- `slot_grid`: two silicon rails separated by a low-index slot.
- `rib_grid`: slab plus ridge grid.
- `circular_rod_grid`: circular dielectric inclusion rasterized on the grid.
- `anisotropic_tensor_grid`: full tensor permittivity grid with off-diagonal terms.
- `angled_bent_grid`: diagonal grid solved through the Rust tensorial angle/bend transform path.

Render selected cases with:

```bash
uv run --extra dev python examples/material_grid_demos.py \
  --case anisotropic_tensor_grid \
  --case angled_bent_grid
```

Run the SOI hybridization sweep:

```bash
uv run --extra dev python examples/soi_hybridization_sweep.py
```

That example sweeps a 220 nm fully etched SOI ridge width, tracks modal branches by field overlap,
and writes the effective-index and TE/TM fraction plots to
`examples/soi_hybridization_outputs/`.

Run the README ridge-waveguide example:

```bash
uv run --extra dev python examples/ridge_waveguide_readme.py
```

That example rasterizes a 500 nm film, 400 nm ridge, 500 nm width waveguide
with angled sidewalls and writes plots to `examples/ridge_waveguide_outputs/`.
