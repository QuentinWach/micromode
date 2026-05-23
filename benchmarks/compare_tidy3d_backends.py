"""Benchmark MicroMode against equivalent Tidy3D mode-solver setups."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import micromode as mm


@dataclass(frozen=True)
class BenchmarkCase:
    """Configuration for one backend comparison problem."""

    case_id: str
    description: str
    ny: int
    nz: int
    num_modes: int = 2
    target_neff: float = 2.5
    krylov_dim: int = 64
    num_pml: tuple[int, int] = (0, 0)
    problem: str = "strip"


PRESETS = {
    "quick": (
        BenchmarkCase("strip_60x40", "SOI strip waveguide, coarse", 60, 40, krylov_dim=48),
        BenchmarkCase("strip_120x80", "SOI strip waveguide", 120, 80, krylov_dim=48),
        BenchmarkCase("strip_pml_120x80", "SOI strip waveguide with mode PML", 120, 80, krylov_dim=48, num_pml=(8, 8)),
        BenchmarkCase(
            "slot_120x80", "Silicon slot waveguide, fundamental", 120, 80, num_modes=1, krylov_dim=48, problem="slot"
        ),
    ),
    "large": (
        BenchmarkCase("strip_60x40", "SOI strip waveguide, coarse", 60, 40, krylov_dim=48),
        BenchmarkCase("strip_120x80", "SOI strip waveguide", 120, 80, krylov_dim=48),
        BenchmarkCase("strip_240x160", "SOI strip waveguide, large", 240, 160, krylov_dim=64),
        BenchmarkCase("strip_480x320", "SOI strip waveguide, very large", 480, 320, krylov_dim=64),
        BenchmarkCase(
            "slot_120x80", "Silicon slot waveguide, fundamental", 120, 80, num_modes=1, krylov_dim=48, problem="slot"
        ),
    ),
}

WIDTH_Y = 3.0
WIDTH_Z = 2.0
WAVELENGTH_UM = 1.55
N_SI = 3.48
N_SIO2 = 1.45
N_AIR = 1.0


def main() -> None:
    """Run selected backend-comparison cases and print a markdown table."""
    args = parse_args()
    cases = list(PRESETS[args.preset])
    rows = [run_case(case, profile_source=args.profile_source) for case in cases]
    print(markdown_table(rows))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    """Parse backend benchmark CLI options."""
    parser = argparse.ArgumentParser(description="Compare MicroMode SciPy and Tidy3D local solves.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument(
        "--profile-source",
        choices=("tidy3d", "analytic"),
        default="tidy3d",
        help="Use Tidy3D's exact local solver grid/epsilon profile, or MicroMode's independent analytic raster.",
    )
    parser.add_argument("--output", type=Path, default=Path("tmp/tidy3d_solver_benchmark.json"))
    return parser.parse_args()


def run_case(case: BenchmarkCase, *, profile_source: str) -> dict[str, object]:
    """Execute one benchmark case for MicroMode and Tidy3D."""
    tidy3d_solver = make_tidy3d_solver(case)
    materials = (
        micromode_materials_from_tidy3d_solver(tidy3d_solver, case)
        if profile_source == "tidy3d"
        else micromode_materials(case)
    )
    row: dict[str, object] = {
        "case_id": case.case_id,
        "description": case.description,
        "grid": f"{case.ny}x{case.nz}",
        "cells": case.ny * case.nz,
        "profile_source": profile_source,
    }
    scipy_seconds, scipy_neff = time_micromode(case, materials)
    tidy3d_seconds, tidy3d_neff = time_tidy3d(tidy3d_solver)

    row.update(
        {
            "scipy_seconds": scipy_seconds,
            "tidy3d_seconds": tidy3d_seconds,
            "scipy_n_eff": scipy_neff.tolist(),
            "tidy3d_n_eff": tidy3d_neff.tolist(),
            "scipy_tidy3d_max_abs_neff": max_abs_delta(scipy_neff, tidy3d_neff),
        }
    )
    print(
        f"{case.case_id}: scipy={scipy_seconds:.3f}s tidy3d={tidy3d_seconds:.3f}s "
        f"delta_tidy3d={row['scipy_tidy3d_max_abs_neff']:.3e}",
        flush=True,
    )
    return row


def time_micromode(case: BenchmarkCase, materials: mm.Materials) -> tuple[float, np.ndarray]:
    """Time the MicroMode solve for a prepared material grid."""
    start = time.perf_counter()
    data = mm.solve_modes(
        material_grid=materials,
        wavelength=WAVELENGTH_UM,
        num_modes=case.num_modes,
        target_neff=case.target_neff,
        krylov_dim=case.krylov_dim,
        pml=mm.PmlSpec(num_cells=case.num_pml),
    )
    return time.perf_counter() - start, np.asarray(data.n_eff.values[0], dtype=float)


def time_tidy3d(solver) -> tuple[float, np.ndarray]:
    """Time a Tidy3D mode solve."""
    start = time.perf_counter()
    data = solver.solve()
    return time.perf_counter() - start, np.asarray(data.n_eff.values[0], dtype=float)


def make_tidy3d_solver(case: BenchmarkCase):
    """Construct a Tidy3D mode solver for one benchmark case."""
    try:
        import tidy3d as td
        from tidy3d.plugins.mode import ModeSolver
    except ImportError as exc:  # pragma: no cover - benchmark dependency only.
        raise SystemExit("Install Tidy3D for this benchmark: uv run --with tidy3d ...") from exc

    structures = tidy3d_structures(td, case.problem)
    dl = min(WIDTH_Y / case.ny, WIDTH_Z / case.nz)
    freq = td.C_0 / WAVELENGTH_UM
    sim = td.Simulation(
        size=(0.1, WIDTH_Y, WIDTH_Z),
        grid_spec=td.GridSpec.uniform(dl=dl),
        run_time=1e-12,
        structures=structures,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PECBoundary()),
    )
    plane = td.Box(center=(0.0, 0.0, 0.0), size=(0.0, WIDTH_Y, WIDTH_Z))
    solver = ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=td.ModeSpec(num_modes=case.num_modes, target_neff=case.target_neff, num_pml=case.num_pml),
        freqs=[freq],
    )
    return solver


def micromode_materials_from_tidy3d_solver(solver, case: BenchmarkCase) -> mm.Materials:
    """Rasterize Tidy3D solver materials into a MicroMode grid."""
    eps = np.asarray(solver._solver_eps(tidy3d_frequency()), dtype=np.complex128)
    grid = solver._solver_grid
    return mm.Materials.from_components(
        x_edges=np.asarray(grid.boundaries.y, dtype=float),
        y_edges=np.asarray(grid.boundaries.z, dtype=float),
        normal_axis=2,
        eps_xx=eps[0],
        eps_xy=eps[1],
        eps_xz=eps[2],
        eps_yx=eps[3],
        eps_yy=eps[4],
        eps_yz=eps[5],
        eps_zx=eps[6],
        eps_zy=eps[7],
        eps_zz=eps[8],
    )


def tidy3d_frequency() -> float:
    """Return the benchmark frequency in Hz."""
    import tidy3d as td

    return float(td.C_0 / WAVELENGTH_UM)


def micromode_materials(case: BenchmarkCase) -> mm.Materials:
    """Build a direct MicroMode material grid for one benchmark case."""
    y_edges = np.linspace(-0.5 * WIDTH_Y, 0.5 * WIDTH_Y, case.ny + 1)
    z_edges = np.linspace(-0.5 * WIDTH_Z, 0.5 * WIDTH_Z, case.nz + 1)
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    yy, zz = np.meshgrid(y, z, indexing="ij")
    eps = np.where(zz < 0.0, N_SIO2**2, N_AIR**2).astype(np.complex128)
    if case.problem == "strip":
        eps[(np.abs(yy) <= 0.225) & (zz >= 0.0) & (zz <= 0.22)] = N_SI**2
    elif case.problem == "slot":
        left = (yy >= -0.29) & (yy <= -0.09) & (zz >= 0.0) & (zz <= 0.22)
        right = (yy >= 0.09) & (yy <= 0.29) & (zz >= 0.0) & (zz <= 0.22)
        eps[left | right] = N_SI**2
    else:
        raise ValueError(f"unknown problem: {case.problem}")
    return mm.Materials.from_diagonal(eps_xx=eps, x_edges=y_edges, y_edges=z_edges, normal_axis=0)


def tidy3d_structures(td, problem: str):
    """Return Tidy3D geometry structures for one benchmark problem."""
    structures = [
        td.Structure(
            geometry=td.Box(center=(0.0, 0.0, -0.5001), size=(td.inf, td.inf, 1.0002)),
            medium=td.Medium(permittivity=N_SIO2**2),
        )
    ]
    if problem == "strip":
        structures.append(
            td.Structure(
                geometry=td.Box(center=(0.0, 0.0, 0.11), size=(td.inf, 0.45, 0.22)),
                medium=td.Medium(permittivity=N_SI**2),
            )
        )
    elif problem == "slot":
        for y_center in (-0.19, 0.19):
            structures.append(
                td.Structure(
                    geometry=td.Box(center=(0.0, y_center, 0.11), size=(td.inf, 0.20, 0.22)),
                    medium=td.Medium(permittivity=N_SI**2),
                )
            )
    else:
        raise ValueError(f"unknown problem: {problem}")
    return structures


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    """Return the maximum absolute difference between sorted mode arrays."""
    count = min(left.size, right.size)
    if count == 0:
        return float("nan")
    return float(np.max(np.abs(left[:count] - right[:count])))


def markdown_table(rows: list[dict[str, object]]) -> str:
    """Format benchmark rows as a markdown table."""
    header = (
        "| Problem | Grid | MicroMode SciPy (s) | Tidy3D local (s) | max abs Δn_eff SciPy/Tidy3D |\n"
        "|---|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['description']} | {row['grid']} | {row['scipy_seconds']:.3f} | "
            f"{row['tidy3d_seconds']:.3f} | "
            f"{row['scipy_tidy3d_max_abs_neff']:.3e} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
