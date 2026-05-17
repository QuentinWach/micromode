from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

import micromode as mm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WAVELENGTH_UM = 1.55
FREQ_1550 = mm.C_0 / WAVELENGTH_UM
N_SI = 3.476
N_SIO2 = 1.444
N_AIR = 1.0
MODE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")


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
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    widths = np.arange(args.width_start, args.width_stop + 0.5 * args.width_step, args.width_step)
    x_edges = centered_edges(args.domain_width, args.x_step)
    y_edges = np.arange(-args.substrate_height, args.clad_height + args.y_step, args.y_step)

    raw_results = []
    eps_grids = []
    for width in widths:
        materials, eps = soi_ridge_materials(
            width=width,
            x_edges=x_edges,
            y_edges=y_edges,
            film_thickness=args.film_thickness,
        )
        print(f"Solving width={width:.3f} um on grid {materials.shape[0]}x{materials.shape[1]}")
        raw_results.append(
            mm.solve_modes(
                material_grid=materials,
                freqs=[FREQ_1550],
                num_modes=args.num_modes,
                target_neff=args.target_neff,
                krylov_dim=args.krylov_dim,
            )
        )
        eps_grids.append(eps)

    sorted_results = tuple(sort_result_by_neff(result) for result in raw_results)
    sweep = mm.Sweep(values=widths, results=sorted_results, parameter_name="width_um")
    summary = write_summary(args.output_dir / "summary.json", sweep, args)
    plot_sweep(args.output_dir / "hybridization_sweep.png", sweep)
    plot_profiles(args.output_dir / "hybridization_profiles.png", widths, sorted_results, eps_grids)
    print(f"Wrote {args.output_dir / 'hybridization_sweep.png'}")
    print(f"Wrote {args.output_dir / 'hybridization_profiles.png'}")
    print(f"Wrote {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MicroMode SOI mode-hybridization sweep.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "soi_hybridization_outputs",
    )
    parser.add_argument("--width-start", type=float, default=0.2)
    parser.add_argument("--width-stop", type=float, default=2.0)
    parser.add_argument("--width-step", type=float, default=0.025)
    parser.add_argument("--x-step", type=float, default=0.05)
    parser.add_argument("--y-step", type=float, default=0.04)
    parser.add_argument("--domain-width", type=float, default=4.0)
    parser.add_argument("--substrate-height", type=float, default=1.0)
    parser.add_argument("--clad-height", type=float, default=1.0)
    parser.add_argument("--film-thickness", type=float, default=0.22)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--target-neff", type=float, default=2.4)
    parser.add_argument("--krylov-dim", type=int, default=56)
    return parser.parse_args()


def centered_edges(width: float, step: float) -> np.ndarray:
    half_cells = int(np.ceil(width / (2 * step)))
    return np.arange(-half_cells, half_cells + 1, dtype=float) * step


def soi_ridge_materials(
    *,
    width: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    film_thickness: float,
) -> tuple[mm.Materials, np.ndarray]:
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    _xx, yy = np.meshgrid(x, y, indexing="ij")
    eps = np.where(yy < 0.0, N_SIO2**2, N_AIR**2).astype(np.complex128)
    silicon_fill = rectangle_fill_fraction(
        x_edges=x_edges,
        y_edges=y_edges,
        x_min=-0.5 * width,
        x_max=0.5 * width,
        y_min=0.0,
        y_max=film_thickness,
    )
    eps = eps * (1.0 - silicon_fill) + N_SI**2 * silicon_fill
    return mm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges), eps


def rectangle_fill_fraction(
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    x_overlap = interval_fill_fraction(x_edges, x_min, x_max)
    y_overlap = interval_fill_fraction(y_edges, y_min, y_max)
    return np.outer(x_overlap, y_overlap)


def interval_fill_fraction(edges: np.ndarray, lower: float, upper: float) -> np.ndarray:
    cell_lower = edges[:-1]
    cell_upper = edges[1:]
    overlap = np.clip(np.minimum(cell_upper, upper) - np.maximum(cell_lower, lower), 0.0, None)
    widths = np.maximum(cell_upper - cell_lower, np.finfo(float).eps)
    return overlap / widths


def write_summary(path: Path, sweep: mm.Sweep, args: argparse.Namespace) -> Path:
    payload = {
        "wavelength_um": WAVELENGTH_UM,
        "n_si": N_SI,
        "n_sio2": N_SIO2,
        "n_air": N_AIR,
        "x_step_um": args.x_step,
        "y_step_um": args.y_step,
        "width_um": sweep.values.tolist(),
        "n_eff": sweep.n_eff.tolist(),
        "pol_fraction": {key: value.tolist() for key, value in sweep.pol_fraction.items()},
        "pol_fraction_waveguide": {key: value.tolist() for key, value in sweep.pol_fraction_waveguide.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def sort_result_by_neff(result: mm.Result) -> mm.Result:
    """Order modes from highest to lowest effective index."""

    order = np.argsort(-np.asarray(result.n_eff.values)[0])
    mode_coord = np.arange(len(order))
    n_complex = result.n_complex.isel(mode_index=order).assign_coords(mode_index=mode_coord)
    field_components = {
        name: data_array.isel(mode_index=order).assign_coords(mode_index=mode_coord)
        for name, data_array in result.field_components.items()
    }
    n_group = None
    if result.n_group is not None:
        n_group = result.n_group.isel(mode_index=order).assign_coords(mode_index=mode_coord)
    dispersion = None
    if result.dispersion is not None:
        dispersion = result.dispersion.isel(mode_index=order).assign_coords(mode_index=mode_coord)
    return mm.Result(
        n_complex=n_complex,
        field_components=field_components,
        n_group=n_group,
        dispersion=dispersion,
        solver_info=result.solver_info,
    )


def plot_sweep(path: Path, sweep: mm.Sweep) -> None:
    pol = sweep.pol_fraction
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.0),
            constrained_layout=True,
            gridspec_kw={"width_ratios": (1.08, 1.0)},
        )
        colors = [MODE_COLORS[index % len(MODE_COLORS)] for index in range(sweep.num_modes)]
        plotted_modes = list(range(min(4, sweep.num_modes)))
        line_width = 2.025

        for mode_index in plotted_modes:
            x_smooth, y_smooth = smooth_line(sweep.values, sweep.n_eff[:, mode_index])
            axes[0].plot(
                x_smooth,
                y_smooth,
                color=colors[mode_index],
                linewidth=line_width,
                label=f"mode {mode_index}",
            )
        axes[0].set_xlabel("ridge width (um)")
        axes[0].set_ylabel("effective index")
        axes[0].grid(color="#d9dde3", linewidth=0.55)
        axes[0].legend(ncol=2, frameon=False, handlelength=1.8, columnspacing=1.1)

        highlighted = [index for index in (0, 1, 2, 3) if index < sweep.num_modes]
        if not highlighted:
            highlighted = [0]
        for mode_index in highlighted:
            x_smooth, y_smooth = smooth_line(sweep.values, pol["te"][:, mode_index], y_limits=(0.0, 1.0))
            axes[1].plot(
                x_smooth,
                y_smooth,
                linewidth=line_width,
                color=colors[mode_index],
                label=f"mode {mode_index}",
            )
        axes[1].set_xlabel("ridge width (um)")
        axes[1].set_ylabel("TE fraction")
        axes[1].set_ylim(-0.04, 1.04)
        axes[1].grid(color="#d9dde3", linewidth=0.55)

        for ax in axes:
            ax.set_xlim(float(sweep.values[0]), float(sweep.values[-1]))
            ax.margins(x=0)

        mode_handles = [Line2D([0], [0], color=colors[index], lw=2.4) for index in highlighted]
        axes[1].legend(
            mode_handles,
            [f"mode {index}" for index in highlighted],
            frameon=False,
            loc="lower right",
            handlelength=1.8,
        )

        save_figure(fig, path)
        plt.close(fig)


def smooth_line(
    x: np.ndarray,
    y: np.ndarray,
    *,
    samples_per_interval: int = 10,
    y_limits: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shape-preserving cubic interpolation for figure rendering."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) < 3:
        return x_values, y_values

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    unique = np.concatenate(([True], np.diff(x_values) > 0.0))
    x_values = x_values[unique]
    y_values = y_values[unique]
    if len(x_values) < 3:
        return x_values, y_values

    slopes = pchip_slopes(x_values, y_values)
    dense_count = max(300, samples_per_interval * (len(x_values) - 1) + 1)
    x_dense = np.linspace(float(x_values[0]), float(x_values[-1]), dense_count)
    y_dense = evaluate_hermite(x_values, y_values, slopes, x_dense)
    if y_limits is not None:
        y_dense = np.clip(y_dense, y_limits[0], y_limits[1])
    return x_dense, y_dense


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)

    for index in range(1, len(y) - 1):
        if delta[index - 1] == 0.0 or delta[index] == 0.0 or np.sign(delta[index - 1]) != np.sign(delta[index]):
            slopes[index] = 0.0
        else:
            weight_1 = 2.0 * h[index] + h[index - 1]
            weight_2 = h[index] + 2.0 * h[index - 1]
            slopes[index] = (weight_1 + weight_2) / (weight_1 / delta[index - 1] + weight_2 / delta[index])

    slopes[0] = pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def pchip_endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    if np.sign(slope) != np.sign(delta0):
        return 0.0
    if np.sign(delta0) != np.sign(delta1) and abs(slope) > abs(3.0 * delta0):
        return 3.0 * delta0
    return slope


def evaluate_hermite(x: np.ndarray, y: np.ndarray, slopes: np.ndarray, x_dense: np.ndarray) -> np.ndarray:
    interval_indices = np.searchsorted(x, x_dense, side="right") - 1
    interval_indices = np.clip(interval_indices, 0, len(x) - 2)
    x0 = x[interval_indices]
    x1 = x[interval_indices + 1]
    y0 = y[interval_indices]
    y1 = y[interval_indices + 1]
    m0 = slopes[interval_indices]
    m1 = slopes[interval_indices + 1]
    h = x1 - x0
    t = (x_dense - x0) / h
    h00 = (2.0 * t**3) - (3.0 * t**2) + 1.0
    h10 = (t**3) - (2.0 * t**2) + t
    h01 = (-2.0 * t**3) + (3.0 * t**2)
    h11 = (t**3) - (t**2)
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def plot_profiles(path: Path, widths: np.ndarray, results: tuple[mm.Result, ...], eps_grids: list[np.ndarray]) -> None:
    selected_indices = unique_nearest_indices(widths, [0.5, 1.0, 2.0])
    rows = len(selected_indices)
    columns = ("index", "mode0_abs", "mode0_ex", "mode1_abs", "mode1_ex")
    column_titles = ("permittivity", "mode 0 |E|", "mode 0 Ex", "mode 1 |E|", "mode 1 Ex")
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(
            rows,
            len(columns),
            figsize=(7.4, 0.80 * rows + 0.48),
            constrained_layout=False,
            sharex=True,
            sharey=True,
        )
        fig.subplots_adjust(left=0.085, right=0.995, bottom=0.02, top=0.90, wspace=0.12, hspace=0.02)
        if rows == 1:
            axes = axes[None, :]

        for row, step_index in enumerate(selected_indices):
            width = widths[step_index]
            result = results[step_index]
            eps = eps_grids[step_index]
            axes[row, 0].text(
                -0.18,
                0.5,
                f"{width:.2f} µm",
                transform=axes[row, 0].transAxes,
                ha="center",
                va="center",
                rotation=90,
                fontsize=9,
            )
            for column_index, column in enumerate(columns):
                ax = axes[row, column_index]
                if column == "index":
                    dims, coords, _ = component_image(result, "Ex", 0)
                    draw_image(ax, dims, coords, np.sqrt(eps.real), cmap="Greys", symmetric=False)
                else:
                    mode_index = 0 if column.startswith("mode0") else 1
                    if mode_index >= result.n_complex.sizes["mode_index"]:
                        ax.set_visible(False)
                        continue
                    dims, coords, values = electric_magnitude_image(result, mode_index)
                    if column.endswith("_ex"):
                        _, _, values = component_image(result, "Ex", mode_index)
                        values = normalize_signed(values.real)
                        draw_image(ax, dims, coords, values, cmap="RdBu_r", symmetric=True, interpolation="bicubic")
                        label_color = "black"
                    else:
                        values = normalize_positive(values)
                        draw_image(
                            ax,
                            dims,
                            coords,
                            values,
                            cmap="magma",
                            symmetric=False,
                            value_limits=(0.0, 1.0),
                            interpolation="bicubic",
                        )
                        label_color = "white"
                    ax.text(
                        0.06,
                        0.90,
                        f"n={result.n_eff.values[0, mode_index]:.4f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        color=label_color,
                        alpha=0.5,
                        fontsize=8,
                    )
                plot_eps_contours(ax, coords, eps)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                ax.set_xlabel("")
                ax.set_ylabel("")

        for column_index, title in enumerate(column_titles):
            axes[0, column_index].set_title(title, pad=6)
        save_figure(fig, path)
        plt.close(fig)


def unique_nearest_indices(values: np.ndarray, targets: list[float]) -> list[int]:
    indices = []
    for target in targets:
        index = int(np.argmin(np.abs(values - target)))
        if index not in indices:
            indices.append(index)
    return indices


def component_image(
    result: mm.Result,
    component: str,
    mode_index: int,
) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray], np.ndarray]:
    image = result.field_components[component].isel(f=0, mode_index=mode_index).squeeze(drop=True)
    spatial_dims = tuple(dim for dim in ("x", "y", "z") if dim in image.dims and image.sizes[dim] > 1)
    if len(spatial_dims) != 2:
        raise ValueError(f"{component} does not reduce to a 2D image")
    image = image.transpose(*spatial_dims)
    coords = tuple(np.asarray(image.coords[dim].values, dtype=float) for dim in spatial_dims)
    return spatial_dims, coords, np.nan_to_num(np.asarray(image.values), copy=False)


def electric_magnitude_image(
    result: mm.Result,
    mode_index: int,
) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray], np.ndarray]:
    dims, coords, ex = component_image(result, "Ex", mode_index)
    magnitude_squared = np.abs(ex) ** 2
    for component in ("Ey", "Ez"):
        other_dims, other_coords, values = component_image(result, component, mode_index)
        coords_match = all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(coords, other_coords, strict=True))
        if other_dims != dims or not coords_match:
            raise ValueError("field components are not colocated")
        magnitude_squared += np.abs(values) ** 2
    return dims, coords, np.sqrt(magnitude_squared)


def draw_image(
    ax,
    dims: tuple[str, str],
    coords: tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    *,
    cmap: str,
    symmetric: bool,
    value_limits: tuple[float, float] | None = None,
    interpolation: str = "nearest",
):
    x, y = coords
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
    extent = (float(x.min() - dx / 2), float(x.max() + dx / 2), float(y.min() - dy / 2), float(y.max() + dy / 2))
    plot_values = np.asarray(values, dtype=float)
    if value_limits is not None:
        kwargs = {"vmin": value_limits[0], "vmax": value_limits[1]}
    elif symmetric:
        limit = max(float(np.nanmax(np.abs(plot_values))), np.finfo(float).eps)
        kwargs = {"vmin": -limit, "vmax": limit}
    else:
        kwargs = {"vmin": float(np.nanmin(plot_values)), "vmax": float(np.nanmax(plot_values))}
    image = ax.imshow(
        plot_values.T,
        extent=extent,
        origin="lower",
        interpolation=interpolation,
        aspect="equal",
        cmap=cmap,
        **kwargs,
    )
    return image


def normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = max(float(np.nanmax(np.abs(values))), np.finfo(float).eps)
    return values / scale


def normalize_signed(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = max(float(np.nanmax(np.abs(values))), np.finfo(float).eps)
    return values / scale


def plot_eps_contours(ax, coords: tuple[np.ndarray, np.ndarray], eps: np.ndarray) -> None:
    values = np.asarray(eps.real, dtype=float)
    if np.nanmax(values) - np.nanmin(values) < 1e-12:
        return
    x, y = coords
    level = 0.5 * (float(np.nanmin(values)) + float(np.nanmax(values)))
    ax.contour(x, y, values.T, levels=[level], colors="black", linewidths=2.0, alpha=0.7)
    ax.contour(x, y, values.T, levels=[level], colors="white", linewidths=0.9, alpha=0.95)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
