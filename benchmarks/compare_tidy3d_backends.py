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
        BenchmarkCase("slot_120x80", "Silicon slot waveguide", 120, 80, krylov_dim=48, problem="slot"),
    ),
    "large": (
        BenchmarkCase("strip_60x40", "SOI strip waveguide, coarse", 60, 40, krylov_dim=48),
        BenchmarkCase("strip_120x80", "SOI strip waveguide", 120, 80, krylov_dim=48),
        BenchmarkCase("strip_240x160", "SOI strip waveguide, large", 240, 160, krylov_dim=64),
        BenchmarkCase("strip_480x320", "SOI strip waveguide, very large", 480, 320, krylov_dim=64),
        BenchmarkCase("slot_120x80", "Silicon slot waveguide", 120, 80, krylov_dim=48, problem="slot"),
    ),
}

WIDTH_Y = 3.0
WIDTH_Z = 2.0
WAVELENGTH_UM = 1.55
N_SI = 3.48
N_SIO2 = 1.45
N_AIR = 1.0


def main() -> None:
    args = parse_args()
    cases = list(PRESETS[args.preset])
    rows = [run_case(case) for case in cases]
    print(markdown_table(rows))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MicroMode Rust, MicroMode SciPy, and Tidy3D local solves.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    parser.add_argument("--output", type=Path, default=Path("tmp/tidy3d_backend_benchmark.json"))
    return parser.parse_args()


def run_case(case: BenchmarkCase) -> dict[str, object]:
    materials = micromode_materials(case)
    row: dict[str, object] = {
        "case_id": case.case_id,
        "description": case.description,
        "grid": f"{case.ny}x{case.nz}",
        "cells": case.ny * case.nz,
    }
    rust_seconds, rust_neff = time_micromode(case, materials, backend="rust")
    scipy_seconds, scipy_neff = time_micromode(case, materials, backend="scipy-reference")
    tidy3d_seconds, tidy3d_neff = time_tidy3d(case)

    row.update(
        {
            "rust_seconds": rust_seconds,
            "scipy_seconds": scipy_seconds,
            "tidy3d_seconds": tidy3d_seconds,
            "rust_n_eff": rust_neff.tolist(),
            "scipy_n_eff": scipy_neff.tolist(),
            "tidy3d_n_eff": tidy3d_neff.tolist(),
            "rust_scipy_max_abs_neff": max_abs_delta(rust_neff, scipy_neff),
            "rust_tidy3d_max_abs_neff": max_abs_delta(rust_neff, tidy3d_neff),
        }
    )
    print(
        f"{case.case_id}: rust={rust_seconds:.3f}s scipy={scipy_seconds:.3f}s "
        f"tidy3d={tidy3d_seconds:.3f}s delta_tidy3d={row['rust_tidy3d_max_abs_neff']:.3e}",
        flush=True,
    )
    return row


def time_micromode(case: BenchmarkCase, materials: mm.Materials, *, backend: str) -> tuple[float, np.ndarray]:
    start = time.perf_counter()
    data = mm.solve_modes(
        material_grid=materials,
        wavelength=WAVELENGTH_UM,
        num_modes=case.num_modes,
        target_neff=case.target_neff,
        krylov_dim=case.krylov_dim,
        pml=mm.PmlSpec(num_cells=case.num_pml),
        backend=backend,
    )
    return time.perf_counter() - start, np.asarray(data.n_eff.values[0], dtype=float)


def time_tidy3d(case: BenchmarkCase) -> tuple[float, np.ndarray]:
    try:
        import tidy3d as td
        from tidy3d.plugins.mode import ModeSolver
    except ImportError as exc:  # pragma: no cover - benchmark dependency only.
        raise SystemExit("Install Tidy3D for this benchmark: uv run --with tidy3d --extra scipy ...") from exc

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
    start = time.perf_counter()
    data = solver.solve()
    return time.perf_counter() - start, np.asarray(data.n_eff.values[0], dtype=float)


def micromode_materials(case: BenchmarkCase) -> mm.Materials:
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
    count = min(left.size, right.size)
    if count == 0:
        return float("nan")
    return float(np.max(np.abs(left[:count] - right[:count])))


def markdown_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| Problem | Grid | Rust (s) | SciPy backend (s) | Tidy3D local (s) | "
        "max abs Δn_eff Rust/SciPy | max abs Δn_eff Rust/Tidy3D |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['description']} | {row['grid']} | {row['rust_seconds']:.3f} | "
            f"{row['scipy_seconds']:.3f} | {row['tidy3d_seconds']:.3f} | "
            f"{row['rust_scipy_max_abs_neff']:.3e} | {row['rust_tidy3d_max_abs_neff']:.3e} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
