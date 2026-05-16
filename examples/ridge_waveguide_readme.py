"""Ridge waveguide example used by the README.

The geometry is a compact SOI rib waveguide. It is MicroMode-native: the
structure is rasterized into a material grid first, then the solver receives
only arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import micromode as mm


WAVELENGTH_UM = 1.55
N_SI = 3.476
N_SIO2 = 1.444
N_AIR = 1.0


def publication_style() -> dict[str, object]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve and plot a rasterized ridge waveguide.")
    parser.add_argument("--step", type=float, default=0.02, help="Grid step in microns.")
    parser.add_argument("--subpixels", type=int, default=7, help="Subpixel samples per axis for material averaging.")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/ridge_waveguide_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    materials, eps = ridge_waveguide_materials(step=args.step, subpixels=args.subpixels)
    data = mm.solve_modes(
        material_grid=materials,
        wavelength=WAVELENGTH_UM,
        num_modes=2,
        target_neff=2.5,
        krylov_dim=48,
    )

    print("n_eff:", np.round(data.n_eff.values[0], 6))
    plot_index(materials, eps, args.output_dir / "ridge_index.png")
    plot_modes(materials, eps, data, args.output_dir / "ridge_modes.png")
    plot_readme_figure(materials, eps, data, args.output_dir / "ridge_waveguide_readme.png")


def ridge_waveguide_materials(step: float = 0.02, subpixels: int = 7) -> tuple[mm.Materials, np.ndarray]:
    """Rasterize an SOI rib waveguide with subpixel material averaging."""

    film_thickness = 0.22
    slab_height = 0.09
    top_width = 0.50
    sidewall_angle_deg = 82.0
    substrate_height = 1.0
    clad_height = 0.8
    domain_width = 3.0

    x_edges = centered_edges(width=domain_width, step=step)
    y_edges = offset_edges(lower=-substrate_height, upper=film_thickness + clad_height, step=step)
    sample_x, sample_y = subpixel_centers(x_edges, y_edges, subpixels=subpixels)

    eps_samples = np.full(sample_x.shape, N_AIR**2, dtype=np.complex128)
    eps_samples[sample_y < 0.0] = N_SIO2**2

    silicon = (sample_y >= 0.0) & (sample_y < slab_height)

    ridge_height = film_thickness - slab_height
    bottom_extra = ridge_height / np.tan(np.deg2rad(sidewall_angle_deg))
    vertical_fraction = np.clip((sample_y - slab_height) / ridge_height, 0.0, 1.0)
    half_width = 0.5 * top_width + (1.0 - vertical_fraction) * bottom_extra
    ridge = (sample_y >= slab_height) & (sample_y < film_thickness) & (np.abs(sample_x) <= half_width)
    silicon |= ridge
    eps_samples[silicon] = N_SI**2

    materials = mm.Materials.from_subpixel_diagonal(
        eps_xx=eps_samples,
        x_edges=x_edges,
        y_edges=y_edges,
        subpixel_shape=(subpixels, subpixels),
    )
    eps = np.asarray(materials.eps_tensor[0, 0], dtype=np.complex128)
    return materials, eps


def centered_edges(*, width: float, step: float) -> np.ndarray:
    cells = int(round(width / step))
    return np.linspace(-0.5 * width, 0.5 * width, cells + 1)


def offset_edges(*, lower: float, upper: float, step: float) -> np.ndarray:
    cells = int(round((upper - lower) / step))
    return np.linspace(lower, upper, cells + 1)


def subpixel_centers(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    subpixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    if subpixels <= 0:
        raise ValueError("subpixels must be positive")
    offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels
    x_samples = x_edges[:-1, None] + np.diff(x_edges)[:, None] * offsets[None, :]
    y_samples = y_edges[:-1, None] + np.diff(y_edges)[:, None] * offsets[None, :]
    sample_x = x_samples[:, None, :, None]
    sample_y = y_samples[None, :, None, :]
    shape = (len(x_edges) - 1, len(y_edges) - 1, subpixels, subpixels)
    return np.broadcast_to(sample_x, shape), np.broadcast_to(sample_y, shape)


def plot_index(materials: mm.Materials, eps: np.ndarray, path: Path) -> None:
    x_edges = np.asarray(materials.grid.x_edges, dtype=float)
    y_edges = np.asarray(materials.grid.y_edges, dtype=float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    with plt.rc_context(publication_style()):
        fig, ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
        image = ax.imshow(
            np.sqrt(eps.real).T,
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            aspect="equal",
            cmap="Greys",
            interpolation="bicubic",
            vmin=N_AIR,
            vmax=N_SI,
        )
        draw_material_outline(ax, x, y, eps, color="black", linewidth=1.3)
        draw_material_outline(ax, x, y, eps, color="white", linewidth=0.55)
        ax.set_title("SOI rib waveguide")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.28, 0.42)
        fig.colorbar(image, ax=ax, label="n", shrink=0.82)
        save_figure(fig, path)
        plt.close(fig)


def plot_modes(materials: mm.Materials, eps: np.ndarray, data: mm.Result, path: Path) -> None:
    components = ("Ex", "Ey", "Hx", "Hy")
    x_edges = np.asarray(materials.grid.x_edges, dtype=float)
    y_edges = np.asarray(materials.grid.y_edges, dtype=float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 2, figsize=(4.9, 2.65), constrained_layout=False, sharex=True, sharey=True)
        fig.subplots_adjust(left=0.02, right=0.995, bottom=0.03, top=0.89, wspace=0.06, hspace=-0.04)
        for ax, component in zip(axes.ravel(), components):
            field = data.field_components[component].isel(f=0, mode_index=0)
            values = normalize_signed(np.asarray(field.values).squeeze().real)
            ax.imshow(
                values.T,
                origin="lower",
                extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                aspect="equal",
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
                interpolation="bicubic",
            )
            draw_material_outline(ax, x, y, eps, color="#111827", linewidth=0.85)
            ax.set_title(component, pad=2)
            ax.set_xlim(-0.9, 0.9)
            ax.set_ylim(-0.28, 0.42)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.set_xlabel("")
            ax.set_ylabel("")
        save_figure(fig, path)
        plt.close(fig)


def plot_readme_figure(materials: mm.Materials, eps: np.ndarray, data: mm.Result, path: Path) -> None:
    """Create a compact presentation figure for the README."""

    x_edges = np.asarray(materials.grid.x_edges, dtype=float)
    y_edges = np.asarray(materials.grid.y_edges, dtype=float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    panels = (
        ("permittivity", "eps", 0),
        ("mode 0 |E|", None, 0),
        ("mode 0 Ex", "Ex", 0),
        ("mode 1 |E|", None, 1),
        ("mode 1 Ex", "Ex", 1),
    )

    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 1.95), constrained_layout=False, sharex=True, sharey=True)
        fig.subplots_adjust(left=0.035, right=0.995, bottom=0.12, top=0.82, wspace=0.10)
        for ax, (title, component, mode_index) in zip(axes, panels):
            values, cmap, vmin, vmax = readme_panel_values(eps, data, component=component, mode_index=mode_index)
            ax.imshow(
                values.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
            )
            if component == "eps":
                draw_material_outline(ax, x, y, eps, color="black", linewidth=1.2)
                draw_material_outline(ax, x, y, eps, color="white", linewidth=0.5)
            else:
                outline_color = "white" if cmap == "magma" else "#111827"
                draw_material_outline(ax, x, y, eps, color=outline_color, linewidth=0.9)
            ax.set_title(title)
            ax.set_xlim(-0.9, 0.9)
            ax.set_ylim(-0.28, 0.42)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            if component != "eps":
                ax.text(
                    0.06,
                    0.90,
                    f"n={data.n_eff.values[0, mode_index]:.4f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="white" if cmap == "magma" else "black",
                    alpha=0.5,
                    fontsize=8,
                )
        save_figure(fig, path)
        plt.close(fig)


def draw_material_outline(ax, x: np.ndarray, y: np.ndarray, eps: np.ndarray, **kwargs) -> None:
    if "color" in kwargs:
        kwargs["colors"] = kwargs.pop("color")
    if "linewidth" in kwargs:
        kwargs["linewidths"] = kwargs.pop("linewidth")
    levels = [0.5 * (N_AIR**2 + N_SI**2), 0.5 * (N_SIO2**2 + N_SI**2)]
    ax.contour(x, y, eps.real.T, levels=levels, **kwargs)


def readme_panel_values(
    eps: np.ndarray,
    data: mm.Result,
    *,
    component: str | None,
    mode_index: int,
) -> tuple[np.ndarray, str, float, float]:
    if component == "eps":
        return eps.real, "Greys", N_AIR**2, N_SI**2
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
    return normalize_signed(values), "RdBu_r", -1.0, 1.0


def normalize_signed(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = max(float(np.nanmax(np.abs(values))), np.finfo(float).eps)
    return values / scale


def save_figure(fig, path: Path) -> None:
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
