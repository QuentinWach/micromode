"""Recreate the Tidy3D modal source/monitor mode plot with MicroMode.

Reference setup:
https://www.flexcompute.com/tidy3d/examples/notebooks/ModalSourcesMonitors/

The Tidy3D notebook solves modes of an x-propagating silicon strip waveguide on
a silica substrate. In MicroMode, this is represented by setting normal_axis=0:
the two material-grid coordinates are global y and z, and the returned global
field components are labeled Ey and Ez as in the Tidy3D plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import micromode as mm

matplotlib.use("Agg")

import matplotlib.pyplot as plt

WAVELENGTH_UM = 1.55
WG_HEIGHT_UM = 0.22
WG_WIDTH_UM = 0.45
N_SI = 3.48
N_SIO2 = 1.45
N_AIR = 1.0


def main() -> None:
    """Run the example script from parsed command-line options."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    materials, eps = tidy3d_waveguide_materials(y_step=args.y_step, z_step=args.z_step)
    data = mm.solve_modes(
        material_grid=materials,
        wavelength=WAVELENGTH_UM,
        num_modes=3,
        target_neff=2.35,
        components=("Ey", "Ez"),
        krylov_dim=args.krylov_dim,
    )

    print("n_eff:", np.round(data.n_eff.values[0], 6))
    output_path = args.output_dir / "tidy3d_modal_modes.png"
    plot_tidy3d_mode_fields(materials, eps, data, output_path)
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the example script."""
    parser = argparse.ArgumentParser(description="Recreate the Tidy3D modal sources/monitors mode plot.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tidy3d_modal_outputs",
    )
    parser.add_argument("--y-step", type=float, default=0.025, help="Mode-plane y grid step in microns.")
    parser.add_argument("--z-step", type=float, default=0.025, help="Mode-plane z grid step in microns.")
    parser.add_argument("--krylov-dim", type=int, default=64)
    return parser.parse_args()


def tidy3d_waveguide_materials(*, y_step: float, z_step: float) -> tuple[mm.Materials, np.ndarray]:
    """Rasterize the Tidy3D strip waveguide mode plane.

    The Tidy3D mode plane has size [0, 3, 2] in x, y, z. The substrate fills
    z < 0, and the silicon strip spans |y| <= 0.225 um and 0 <= z <= 0.22 um.
    """

    y_edges = centered_edges(width=3.0, step=y_step)
    z_edges = centered_edges(width=2.0, step=z_step)
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    _yy, zz = np.meshgrid(y, z, indexing="ij")

    eps = np.where(zz < 0.0, N_SIO2**2, N_AIR**2).astype(np.complex128)
    waveguide = (np.abs(_yy) <= 0.5 * WG_WIDTH_UM) & (zz >= 0.0) & (zz <= WG_HEIGHT_UM)
    eps[waveguide] = N_SI**2

    materials = mm.Materials.from_diagonal(
        eps_xx=eps,
        x_edges=y_edges,
        y_edges=z_edges,
        normal_axis=0,
    )
    return materials, eps


def centered_edges(*, width: float, step: float) -> np.ndarray:
    """Return evenly spaced cell edges centered on zero."""
    if step <= 0.0:
        raise ValueError("step must be positive")
    cells = round(width / step)
    if cells < 1:
        raise ValueError("width must be at least one step")
    return np.linspace(-0.5 * width, 0.5 * width, cells + 1)


def plot_tidy3d_mode_fields(materials: mm.Materials, eps: np.ndarray, data: mm.Result, path: Path) -> None:
    """Plot Tidy3D-style modal field panels."""
    y_edges = np.asarray(materials.grid.x_edges, dtype=float)
    z_edges = np.asarray(materials.grid.y_edges, dtype=float)
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    extent = (y_edges[0], y_edges[-1], z_edges[0], z_edges[-1])

    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.0), constrained_layout=True, sharex=True, sharey=True)
        for mode_index in range(3):
            for col, component in enumerate(("Ey", "Ez")):
                ax = axes[mode_index, col]
                values = field_abs(data, component=component, mode_index=mode_index)
                image = ax.imshow(
                    values.T,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    cmap="magma",
                    vmin=0.0,
                    vmax=float(np.nanpercentile(values, 99.5)),
                    interpolation="nearest",
                )
                draw_material_context(ax, y, z, eps)
                ax.set_title(f"{component}, mode_index={mode_index}")
                ax.set_xlabel("y (um)")
                ax.set_ylabel("z (um)")
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                cbar = fig.colorbar(image, ax=ax, extend="both", fraction=0.046, pad=0.04)
                cbar.set_label(f"|{component}|")
        save_figure(fig, path)
        plt.close(fig)


def field_abs(data: mm.Result, *, component: str, mode_index: int) -> np.ndarray:
    """Return the magnitude image for one field component and mode."""
    field = data.field_components[component].isel(f=0, mode_index=mode_index).squeeze(drop=True)
    return np.abs(np.asarray(field.transpose("y", "z").values))


def draw_material_context(ax, y: np.ndarray, z: np.ndarray, eps: np.ndarray) -> None:
    """Draw material outlines for the Tidy3D-style example."""
    ax.axhline(0.0, color="#111111", linewidth=0.85, alpha=0.35)
    silicon_level = 0.5 * (N_AIR**2 + N_SI**2)
    ax.contour(y, z, eps.real.T, levels=[silicon_level], colors="#2f2f2f", linewidths=1.0, alpha=0.75)


def publication_style() -> dict[str, object]:
    """Return matplotlib rcParams for generated example figures."""
    return {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }


def save_figure(fig, path: Path) -> None:
    """Write a figure to disk and close it."""
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
