"""Ridge waveguide example used by the README.

The geometry mirrors a common semi-vectorial ridge-waveguide demo, but the
implementation is MicroMode-native: the structure is rasterized into a material
grid first, then the solver receives only arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import micromode as mm


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve and plot a rasterized ridge waveguide.")
    parser.add_argument("--step", type=float, default=0.04, help="Grid step in microns. Use 0.02 for a finer run.")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/ridge_waveguide_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    materials, eps = ridge_waveguide_materials(step=args.step)
    data = mm.solve_modes(
        material_grid=materials,
        wavelength=1.55,
        num_modes=2,
        target_neff=2.4,
        krylov_dim=32,
    )

    print("n_eff:", np.round(data.n_eff.values[0], 6))
    plot_index(materials, eps, args.output_dir / "ridge_index.png")
    plot_modes(materials, eps, data, args.output_dir / "ridge_modes.png")
    plot_readme_figure(materials, eps, data, args.output_dir / "ridge_waveguide_readme.png")


def ridge_waveguide_materials(step: float = 0.04) -> tuple[mm.Materials, np.ndarray]:
    """Rasterize a rib/ridge waveguide into a diagonal permittivity grid."""

    wg_height = 0.4
    wg_width = 0.5
    sub_height = 0.5
    sub_width = 2.0
    clad_height = 0.5
    n_sub = 1.4
    n_wg = 3.0
    n_clad = 1.0
    film_thickness = 0.5
    sidewall_angle_deg = 75.0

    x_edges = centered_edges(width=sub_width, step=step)
    y_edges = positive_edges(height=sub_height + film_thickness + clad_height, step=step)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")

    eps = np.full(xx.shape, n_clad**2, dtype=np.complex128)
    eps[yy < sub_height] = n_sub**2

    slab_height = film_thickness - wg_height
    slab_top = sub_height + slab_height
    film_top = sub_height + film_thickness
    eps[(yy >= sub_height) & (yy < slab_top)] = n_wg**2

    ridge_layer = (yy >= slab_top) & (yy < film_top)
    sidewall_extra = wg_height / np.tan(np.deg2rad(sidewall_angle_deg))
    vertical_fraction = np.clip((yy - slab_top) / wg_height, 0.0, 1.0)
    half_width = 0.5 * wg_width + vertical_fraction * sidewall_extra
    eps[ridge_layer & (np.abs(xx) <= half_width)] = n_wg**2

    return mm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges), eps


def centered_edges(*, width: float, step: float) -> np.ndarray:
    cells = int(round(width / step))
    return np.linspace(-0.5 * width, 0.5 * width, cells + 1)


def positive_edges(*, height: float, step: float) -> np.ndarray:
    cells = int(round(height / step))
    return np.linspace(0.0, height, cells + 1)


def plot_index(materials: mm.Materials, eps: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    x_edges = np.asarray(materials.grid.x_edges, dtype=float)
    y_edges = np.asarray(materials.grid.y_edges, dtype=float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    image = ax.imshow(
        np.sqrt(eps.real).T,
        origin="lower",
        extent=(x.min(), x.max(), y.min(), y.max()),
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title("Ridge waveguide refractive index")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(image, ax=ax, label="n")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_modes(materials: mm.Materials, eps: np.ndarray, data: mm.Result, path: Path) -> None:
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    x_edges = np.asarray(materials.grid.x_edges, dtype=float)
    y_edges = np.asarray(materials.grid.y_edges, dtype=float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    for ax, component in zip(axes.ravel(), components):
        field = data.field_components[component].isel(f=0, mode_index=0)
        values = np.asarray(field.values).squeeze().real
        limit = max(float(np.nanmax(np.abs(values))), np.finfo(float).eps)
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=(
                x_edges[0],
                x_edges[-1],
                y_edges[0],
                y_edges[-1],
            ),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.contour(
            x,
            y,
            eps.real.T,
            levels=[2.0**2],
            colors="white",
            linewidths=0.8,
        )
        ax.set_title(component)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle(f"Fundamental ridge mode, n_eff={data.n_eff.values[0, 0]:.4f}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_readme_figure(materials: mm.Materials, eps: np.ndarray, data: mm.Result, path: Path) -> None:
    """Create a compact presentation figure for the README."""

    x_edges = np.asarray(materials.grid.x_edges, dtype=float)
    y_edges = np.asarray(materials.grid.y_edges, dtype=float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    panels = (
        ("mode 0 |E|", None, 0),
        ("mode 1 |E|", None, 1),
        ("mode 0 Ex", "Ex", 0),
        ("mode 0 Ey", "Ey", 0),
    )

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
        }
    ):
        fig = plt.figure(figsize=(12.5, 6.8), constrained_layout=True)
        grid = fig.add_gridspec(2, 4, width_ratios=(1.35, 1.0, 1.0, 1.0))
        ax_index = fig.add_subplot(grid[:, 0])
        axes = [
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[0, 3]),
            fig.add_subplot(grid[1, 1]),
        ]
        ax_text = fig.add_subplot(grid[1, 2:])

        index_image = ax_index.imshow(
            np.sqrt(eps.real).T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="mako" if "mako" in plt.colormaps() else "viridis",
            vmin=1.0,
            vmax=3.0,
        )
        draw_material_outline(ax_index, x, y, eps, color="white", linewidth=1.1)
        ax_index.set_title("Rasterized ridge")
        ax_index.set_xlabel("x (um)")
        ax_index.set_ylabel("y (um)")
        fig.colorbar(index_image, ax=ax_index, shrink=0.82, label="refractive index")

        for ax, (title, component, mode_index) in zip(axes, panels):
            values, cmap, vmin, vmax = readme_panel_values(data, component=component, mode_index=mode_index)
            image = ax.imshow(
                values.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            draw_material_outline(ax, x, y, eps, color="#111827", linewidth=0.9)
            ax.set_title(title)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            fig.colorbar(image, ax=ax, shrink=0.78)

        ax_text.axis("off")
        neff = data.n_eff.values[0]
        te = data.pol_fraction_waveguide["te"].values[0]
        mode_area = data.mode_area.values[0]
        lines = [
            "Ridge waveguide mode solve",
            "",
            "500 nm film, 400 nm ridge, 500 nm width",
            "75 deg sidewalls, n_core=3.0, n_sub=1.4",
            "",
            f"mode 0 n_eff = {neff[0]:.4f}",
            f"mode 1 n_eff = {neff[1]:.4f}",
            f"mode 0 wg TE fraction = {te[0]:.3f}",
            f"mode 0 area = {mode_area[0]:.3f} um^2",
        ]
        ax_text.text(
            0.03,
            0.94,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=11,
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.55,rounding_size=0.08",
                "facecolor": "#f1f5f9",
                "edgecolor": "#cbd5e1",
            },
        )
        fig.suptitle("MicroMode solves directly from a material grid", fontsize=16, weight="bold")
        fig.savefig(path, dpi=200)
        plt.close(fig)


def draw_material_outline(ax, x: np.ndarray, y: np.ndarray, eps: np.ndarray, **kwargs) -> None:
    if "color" in kwargs:
        kwargs["colors"] = kwargs.pop("color")
    if "linewidth" in kwargs:
        kwargs["linewidths"] = kwargs.pop("linewidth")
    ax.contour(x, y, eps.real.T, levels=[2.0**2], **kwargs)


def readme_panel_values(
    data: mm.Result,
    *,
    component: str | None,
    mode_index: int,
) -> tuple[np.ndarray, str, float, float]:
    if component is None:
        values = np.sqrt(
            sum(
                np.abs(np.asarray(data.field_components[name].isel(f=0, mode_index=mode_index).values).squeeze()) ** 2
                for name in ("Ex", "Ey", "Ez")
            )
        )
        scale = max(float(np.nanmax(values)), np.finfo(float).eps)
        return values / scale, "magma", 0.0, 1.0

    values = np.asarray(data.field_components[component].isel(f=0, mode_index=mode_index).values).squeeze().real
    scale = max(float(np.nanmax(np.abs(values))), np.finfo(float).eps)
    return values / scale, "RdBu_r", -1.0, 1.0


if __name__ == "__main__":
    main()
