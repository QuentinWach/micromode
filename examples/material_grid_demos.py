from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

import micromode as sm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

FREQ_1550 = sm.C_0 / 1.55
SI_EPS = 3.48**2
SIO2_EPS = 1.44**2
AIR_EPS = 1.0


@dataclass(frozen=True)
class GridDemo:
    key: str
    title: str
    description: str
    make_material_grid: Callable[[], tuple[sm.Materials, np.ndarray]]
    target_neff: float
    num_modes: int = 2
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    bend_radius: float | None = None
    bend_axis: int = 0
    krylov_dim: int = 32


def main() -> None:
    args = parse_args()
    selected = select_demos(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for demo in selected:
        material_grid, eps_background = demo.make_material_grid()
        print(f"Solving {demo.key}: {demo.title}")
        data = sm.solve_modes(
            material_grid=material_grid,
            freqs=[FREQ_1550],
            num_modes=demo.num_modes,
            target_neff=demo.target_neff,
            angle_theta=demo.angle_theta,
            angle_phi=demo.angle_phi,
            bend_radius=demo.bend_radius,
            bend_axis=demo.bend_axis,
            krylov_dim=demo.krylov_dim,
        )
        output_path = args.output_dir / f"{demo.key}.png"
        plot_demo(demo, material_grid, eps_background, data, output_path)
        n_eff = np.asarray(data.n_complex.values)[0]
        n_eff_text = ", ".join(f"{value.real:.6f}{value.imag:+.1e}j" for value in n_eff)
        print(f"  n_eff: {n_eff_text}")
        print(f"  wrote {output_path}")

    if args.show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure Materials MicroMode demos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "material_grid_outputs",
        help="Directory where PNG figures are written.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case key to render. Can be passed more than once. Defaults to all cases.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures after writing PNG files.")
    return parser.parse_args()


def select_demos(case_keys: Sequence[str] | None) -> list[GridDemo]:
    demos = grid_demos()
    if not case_keys:
        return demos
    by_key = {demo.key: demo for demo in demos}
    missing = [key for key in case_keys if key not in by_key]
    if missing:
        known = ", ".join(sorted(by_key))
        raise SystemExit(f"Unknown case(s): {', '.join(missing)}. Known cases: {known}")
    return [by_key[key] for key in case_keys]


def grid_demos() -> list[GridDemo]:
    return [
        GridDemo(
            key="strip_grid",
            title="Rectangular Strip From Raw Grid",
            description="A silicon core drawn directly into a silica permittivity array.",
            make_material_grid=make_strip_grid,
            target_neff=2.5,
            num_modes=2,
        ),
        GridDemo(
            key="slot_grid",
            title="Slot Waveguide From Raw Grid",
            description="Two high-index rails with a low-index slot, no geometry objects.",
            make_material_grid=make_slot_grid,
            target_neff=2.4,
            num_modes=2,
        ),
        GridDemo(
            key="rib_grid",
            title="Rib Waveguide From Raw Grid",
            description="A slab plus ridge encoded as pixels in the material grid.",
            make_material_grid=make_rib_grid,
            target_neff=2.6,
            num_modes=2,
        ),
        GridDemo(
            key="circular_rod_grid",
            title="Circular Rod From Raw Grid",
            description="A circular dielectric inclusion rasterized directly onto the mode grid.",
            make_material_grid=make_circular_rod_grid,
            target_neff=1.7,
            num_modes=2,
        ),
        GridDemo(
            key="anisotropic_tensor_grid",
            title="Full Tensor Anisotropic Grid",
            description="Diagonal anisotropy plus off-diagonal epsilon terms solved by the sparse tensorial path.",
            make_material_grid=make_anisotropic_tensor_grid,
            target_neff=2.0,
            num_modes=1,
            krylov_dim=24,
        ),
        GridDemo(
            key="angled_bent_grid",
            title="Angled And Bent Grid Solve",
            description="A diagonal grid transformed through the tensorial angle/bend path.",
            make_material_grid=make_strip_grid,
            target_neff=2.5,
            num_modes=1,
            angle_theta=0.08,
            angle_phi=0.25,
            bend_radius=8.0,
            bend_axis=0,
            krylov_dim=24,
        ),
    ]


def make_strip_grid() -> tuple[sm.Materials, np.ndarray]:
    x_edges, y_edges, xx, yy = demo_grid(nx=42, ny=30)
    eps = np.full(xx.shape, SIO2_EPS, dtype=np.complex128)
    eps[(np.abs(xx) <= 0.25) & (np.abs(yy) <= 0.11)] = SI_EPS
    return sm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges), eps


def make_slot_grid() -> tuple[sm.Materials, np.ndarray]:
    x_edges, y_edges, xx, yy = demo_grid(nx=42, ny=30)
    eps = np.full(xx.shape, SIO2_EPS, dtype=np.complex128)
    rail = np.abs(yy) <= 0.11
    left = (xx >= -0.30) & (xx <= -0.06)
    right = (xx >= 0.06) & (xx <= 0.30)
    eps[rail & (left | right)] = SI_EPS
    return sm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges), eps


def make_rib_grid() -> tuple[sm.Materials, np.ndarray]:
    x_edges, y_edges, xx, yy = demo_grid(nx=46, ny=30)
    eps = np.full(xx.shape, SIO2_EPS, dtype=np.complex128)
    slab = (np.abs(xx) <= 0.72) & (yy >= -0.16) & (yy <= -0.05)
    ridge = (np.abs(xx) <= 0.28) & (yy >= -0.05) & (yy <= 0.18)
    eps[slab | ridge] = SI_EPS
    return sm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges), eps


def make_circular_rod_grid() -> tuple[sm.Materials, np.ndarray]:
    x_edges, y_edges, xx, yy = demo_grid(nx=42, ny=34, width=2.2, height=1.8)
    eps = np.full(xx.shape, AIR_EPS, dtype=np.complex128)
    eps[xx**2 + yy**2 <= 0.32**2] = 2.6**2
    return sm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges), eps


def make_anisotropic_tensor_grid() -> tuple[sm.Materials, np.ndarray]:
    x_edges, y_edges, xx, yy = demo_grid(nx=28, ny=22, width=2.0, height=1.4)
    core = (np.abs(xx) <= 0.32) & (np.abs(yy) <= 0.16)
    eps_xx = np.full(xx.shape, SIO2_EPS, dtype=np.complex128)
    eps_yy = np.full(xx.shape, SIO2_EPS, dtype=np.complex128)
    eps_zz = np.full(xx.shape, SIO2_EPS, dtype=np.complex128)
    eps_xz = np.zeros(xx.shape, dtype=np.complex128)
    eps_zx = np.zeros(xx.shape, dtype=np.complex128)
    eps_xx[core] = 2.45**2
    eps_yy[core] = 2.10**2
    eps_zz[core] = 1.92**2
    eps_xz[core] = 0.04
    eps_zx[core] = 0.04
    material_grid = sm.Materials.from_components(
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        eps_xz=eps_xz,
        eps_zx=eps_zx,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    return material_grid, eps_xx


def demo_grid(
    *,
    nx: int,
    ny: int,
    width: float = 2.4,
    height: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(-width / 2, width / 2, nx + 1)
    y_edges = np.linspace(-height / 2, height / 2, ny + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return x_edges, y_edges, xx, yy


def plot_demo(
    demo: GridDemo,
    material_grid: sm.Materials,
    eps_background: np.ndarray,
    data: sm.Result,
    output_path: Path,
) -> None:
    mode_count = int(data.n_complex.shape[1])
    columns = ("eps_xx", "|E|", "Re(Ex)", "Re(Ey)", "Re(Ez)")
    fig, axes = plt.subplots(
        mode_count,
        len(columns),
        figsize=(3.5 * len(columns), 3.2 * mode_count),
        squeeze=False,
        constrained_layout=True,
    )

    for mode_index in range(mode_count):
        dims, coords, magnitude = electric_magnitude_image(data, mode_index)
        images = {
            "eps_xx": np.asarray(eps_background.real, dtype=float),
            "|E|": magnitude,
            "Re(Ex)": component_image(data, "Ex", mode_index)[2].real,
            "Re(Ey)": component_image(data, "Ey", mode_index)[2].real,
            "Re(Ez)": component_image(data, "Ez", mode_index)[2].real,
        }
        n_eff = complex(np.asarray(data.n_complex.values)[0, mode_index])
        for column_index, column in enumerate(columns):
            ax = axes[mode_index, column_index]
            if column == "eps_xx":
                image = draw_image(ax, dims, coords, images[column], cmap="viridis", symmetric=False)
                plot_eps_contours(ax, coords, eps_background)
            elif column == "|E|":
                image = draw_image(ax, dims, coords, images[column], cmap="magma", symmetric=False)
                plot_eps_contours(ax, coords, eps_background)
            else:
                image = draw_image(ax, dims, coords, images[column], cmap="RdBu_r", symmetric=True)
                plot_eps_contours(ax, coords, eps_background)
            ax.set_title(f"mode {mode_index}, {column}\nn_eff={n_eff.real:.5f}{n_eff.imag:+.1e}j")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{demo.title}: {demo.description}", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def component_image(
    data: sm.Result, component: str, mode_index: int
) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray], np.ndarray]:
    field = data.field_components[component]
    image = field.isel(f=0, mode_index=mode_index).squeeze(drop=True)
    spatial_dims = tuple(dim for dim in ("x", "y", "z") if dim in image.dims and image.sizes[dim] > 1)
    if len(spatial_dims) != 2:
        raise ValueError(f"{component} does not reduce to a 2D image; got dims={image.dims}")
    image = image.transpose(*spatial_dims)
    values = np.nan_to_num(np.asarray(image.values), copy=False)
    coords = tuple(np.asarray(image.coords[dim].values, dtype=float) for dim in spatial_dims)
    return spatial_dims, coords, values


def electric_magnitude_image(
    data: sm.Result, mode_index: int
) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray], np.ndarray]:
    dims, coords, ex = component_image(data, "Ex", mode_index)
    magnitude_squared = np.abs(ex) ** 2
    for component in ("Ey", "Ez"):
        other_dims, other_coords, values = component_image(data, component, mode_index)
        coords_match = all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(coords, other_coords, strict=True))
        if other_dims != dims or not coords_match:
            raise ValueError("field components are not colocated on a common plotting grid")
        magnitude_squared += np.abs(values) ** 2
    return dims, coords, np.sqrt(magnitude_squared)


def draw_image(
    ax: Axes,
    dims: tuple[str, str],
    coords: tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    *,
    cmap: str,
    symmetric: bool,
):
    x, y = coords
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
    extent = (
        float(x.min() - dx / 2),
        float(x.max() + dx / 2),
        float(y.min() - dy / 2),
        float(y.max() + dy / 2),
    )
    plot_values = np.asarray(values, dtype=float)
    max_abs = float(np.nanmax(np.abs(plot_values))) if plot_values.size else 0.0
    if symmetric:
        limit = max(max_abs, np.finfo(float).eps)
        kwargs = {"vmin": -limit, "vmax": limit}
    else:
        kwargs = {
            "vmin": float(np.nanmin(plot_values)),
            "vmax": max(float(np.nanmax(plot_values)), np.finfo(float).eps),
        }
    image = ax.imshow(
        plot_values.T,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        **kwargs,
    )
    ax.set_xlabel(f"{dims[0]} (um)")
    ax.set_ylabel(f"{dims[1]} (um)")
    return image


def plot_eps_contours(ax: Axes, coords: tuple[np.ndarray, np.ndarray], eps: np.ndarray) -> None:
    values = np.asarray(eps.real, dtype=float)
    if np.nanmax(values) - np.nanmin(values) < 1e-12:
        return
    level = 0.5 * (float(np.nanmin(values)) + float(np.nanmax(values)))
    x, y = coords
    ax.contour(x, y, values.T, levels=[level], colors="black", linewidths=2.2, alpha=0.72)
    ax.contour(x, y, values.T, levels=[level], colors="white", linewidths=1.0, alpha=0.96)


if __name__ == "__main__":
    main()
