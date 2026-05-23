from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import micromode as mm


def main() -> None:
    args = parse_args()
    rows = []
    grids = args.grid or ["20x14", "32x22", "48x32"]
    for nx, ny in parse_grid_sizes(grids):
        materials = strip_materials(nx=nx, ny=ny)
        for repeat in range(args.repeats):
            start = time.perf_counter()
            data = mm.solve_modes(
                material_grid=materials,
                freqs=[mm.C_0 / args.wavelength],
                num_modes=args.num_modes,
                target_neff=args.target_neff,
                krylov_dim=args.krylov_dim,
                pml=mm.PmlSpec(num_cells=args.num_pml),
            )
            elapsed = time.perf_counter() - start
            run_info = data.solver_info["runs"][0]
            row = {
                "nx": nx,
                "ny": ny,
                "cells": nx * ny,
                "repeat": repeat,
                "seconds": elapsed,
                "solver": run_info["backend"],
                "solver_kind": run_info["backend_kind"],
                "operator_size": run_info["operator_size"],
                "operator_nnz": run_info["operator_nnz"],
                "max_residual": float(np.max(run_info["residuals"])),
                "n_eff": [float(value.real) for value in np.asarray(data.n_complex.values)[0]],
            }
            rows.append(row)
            print(
                f"{nx:4d}x{ny:<4d} repeat={repeat} "
                f"{elapsed:7.3f}s solver={row['solver']} "
                f"max_res={row['max_residual']:.2e}"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MicroMode sparse solves over grid sizes.")
    parser.add_argument(
        "--grid",
        action="append",
        default=None,
        help="Grid size as NxM. Can be supplied more than once.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--target-neff", type=float, default=2.5)
    parser.add_argument("--krylov-dim", type=int, default=40)
    parser.add_argument("--wavelength", type=float, default=1.55)
    parser.add_argument("--num-pml", type=int, nargs=2, metavar=("NX", "NY"), default=(0, 0))
    parser.add_argument("--output", type=Path, default=Path("tmp/micromode_solver_benchmark.json"))
    return parser.parse_args()


def parse_grid_sizes(values: list[str]) -> list[tuple[int, int]]:
    sizes = []
    for value in values:
        left, sep, right = value.lower().partition("x")
        if not sep:
            raise SystemExit(f"grid size must be formatted as NxM, got {value!r}")
        sizes.append((int(left), int(right)))
    return sizes


def strip_materials(*, nx: int, ny: int) -> mm.Materials:
    x_edges = np.linspace(-1.2, 1.2, nx + 1)
    y_edges = np.linspace(-0.8, 0.8, ny + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    eps = np.full((nx, ny), 1.44**2, dtype=np.complex128)
    eps[(np.abs(xx) <= 0.25) & (np.abs(yy) <= 0.11)] = 3.48**2
    return mm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges)


if __name__ == "__main__":
    main()
