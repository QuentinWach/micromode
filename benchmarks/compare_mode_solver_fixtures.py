from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from mode_solver.fixtures import (
        DEFAULT_FIXTURE_ROOT,
        data_path,
        load_data_array,
        manifest_path,
        phase_aligned_relative_error,
        read_json,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as a package module.
    from benchmarks.mode_solver.fixtures import (
        DEFAULT_FIXTURE_ROOT,
        data_path,
        load_data_array,
        manifest_path,
        phase_aligned_relative_error,
        read_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect committed mode-solver reference fixtures.")
    parser.add_argument(
        "--suite",
        choices=("smoke", "extended"),
        default="smoke",
        help="Fixture suite to inspect.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Inspect one case id. Can be supplied more than once.",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=None,
        help="Fixture root. Defaults to fixtures/mode_solver/<suite>.",
    )
    parser.add_argument(
        "--run-local",
        action="store_true",
        help="Run local MicroMode solves for supported fixture cases and compare n_eff/fields.",
    )
    parser.add_argument(
        "--fail-on-tolerance",
        action="store_true",
        help="Exit nonzero when a supported local comparison exceeds fixture tolerances.",
    )
    parser.add_argument(
        "--fail-on-production-gap",
        action="store_true",
        help="Exit nonzero when a fixture marked as production support does not pass locally.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture_root = args.fixture_root or (DEFAULT_FIXTURE_ROOT / args.suite)
    manifest = read_json(manifest_path(fixture_root))
    entries = manifest["cases"]
    if args.case_ids:
        requested = set(args.case_ids)
        entries = [entry for entry in entries if entry["case_id"] in requested]
        missing = requested.difference(entry["case_id"] for entry in entries)
        if missing:
            raise SystemExit(f"Unknown case id(s): {', '.join(sorted(missing))}")

    report = {
        "fixture_root": str(fixture_root),
        "backend": "rust_sparse" if args.run_local else None,
        "cases": [],
        "summary": {"pass": 0, "fail": 0, "unsupported": 0, "not_run": 0},
    }
    failures = 0
    production_gaps = 0
    for index, entry in enumerate(entries, start=1):
        n_complex = load_data_array(data_path(fixture_root, entry["case_id"]), "n_complex")
        values = np.asarray(n_complex.values)
        real_min = float(np.min(values.real))
        real_max = float(np.max(values.real))
        imag_max = float(np.max(np.abs(values.imag)))
        print(
            f"[{index}/{len(entries)}] {entry['case_id']}: "
            f"shape={values.shape}, n_eff=[{real_min:.6g}, {real_max:.6g}], "
            f"max |k_eff|={imag_max:.3e}"
        )
        status = {"status": "not_run", "summary": "local solve not requested"}
        if args.run_local:
            status = _compare_local_case(fixture_root, entry)
            print(f"      local rust_sparse: {status['status']}: {status['summary']}")
            if status["failed"]:
                failures += 1
            if status.get("support") == "production" and status["status"] != "pass":
                production_gaps += 1
        report["summary"][status["status"]] += 1
        report["cases"].append({"case_id": entry["case_id"], **status})

    print(f"Inspected {len(entries)} fixture(s) from {fixture_root}")
    print(
        "Local validation: "
        + ", ".join(f"{key}={value}" for key, value in report["summary"].items() if value)
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.report_json}")
    if args.fail_on_tolerance and failures:
        raise SystemExit(f"{failures} local fixture comparison(s) exceeded tolerance")
    if args.fail_on_production_gap and production_gaps:
        raise SystemExit(f"{production_gaps} production fixture comparison(s) did not pass")


def _compare_local_case(root: Path, entry: dict) -> dict:
    case_id = entry["case_id"]
    try:
        import micromode as sm
    except ImportError as exc:
        return _status("fail", f"could not import micromode: {exc}")

    mode_data = data_path(root, case_id)
    ref_n = load_data_array(mode_data, "n_complex")
    try:
        ref_ex = load_data_array(mode_data, "Ex")
    except KeyError as exc:
        return _status("unsupported", f"missing Ex field ({exc})")

    if case_id not in _LOCAL_CASES:
        return _status("unsupported", "no local reconstruction recipe", support="metadata_missing")

    recipe = _LOCAL_CASES[case_id]
    support = recipe.get("support", "production")
    if recipe.get("unsupported"):
        return _status("unsupported", recipe["unsupported"], support=support)
    if tuple(recipe.get("num_pml", (0, 0))) != (0, 0):
        return _status("unsupported", "local Rust comparison does not support PML", support=support)

    tangent_dims = tuple(dim for dim in ("x", "y", "z") if dim in ref_ex.dims and ref_ex.sizes.get(dim, 0) > 1)
    if len(tangent_dims) != 2:
        return _status("unsupported", f"expected two raster dimensions, got {tangent_dims}", support=support)
    normal_dim = next(dim for dim in ("x", "y", "z") if dim not in tangent_dims)
    edges = _solver_edges_from_field_coords(
        tuple(np.asarray(ref_ex.coords[dim].values, dtype=float) for dim in tangent_dims),
        recipe,
    )
    normal_coord = float(np.asarray(ref_ex.coords[normal_dim].values, dtype=float).reshape(-1)[0])

    try:
        result = _solve_recipe(
            sm=sm,
            recipe=recipe,
            ref_n=ref_n,
            edges=edges,
            tangent_dims=tangent_dims,
            normal_dim=normal_dim,
            normal_coord=normal_coord,
        )
    except NotImplementedError as exc:
        return _status("unsupported", str(exc), support=support)
    actual_n = _reorder_modes(result.n_complex.values, recipe)
    n_error = float(np.max(np.abs(actual_n - ref_n.values)))
    tolerance = _n_tolerance(entry, recipe)
    failed = n_error > tolerance

    field_errors = []
    if "Ex" in result.field_components and result.field_components["Ex"].shape == ref_ex.shape:
        field_rel, overlap_error = phase_aligned_relative_error(
            ref_ex.values[..., 0],
            _reorder_field_modes(result.field_components["Ex"].values, recipe)[..., 0],
        )
        field_errors.append(f"Ex rel={field_rel:.3e}, overlap={overlap_error:.3e}")
    details = {
        "n_complex_max_abs_error": n_error,
        "n_complex_atol": tolerance,
        "support": support,
    }
    field_summary = "" if not field_errors else ", " + ", ".join(field_errors)
    return _status(
        "fail" if failed else "pass",
        f"n_complex max abs error={n_error:.3e} (tol={tolerance:.3e}){field_summary}",
        **details,
    )


_LOCAL_CASES = {
    "strip_z_scalar_single": {
        "support": "production",
        "boxes": [{"size": (1.2, 0.8, 100.0), "center": (0.0, 0.0, 0.0), "eps": 4.0}],
        "clad_eps": 1.0,
        "target_neff": 2.0,
    },
    "group_index_silicon_strip": {
        "support": "production",
        "boxes": [{"size": (0.5, 0.22, 100.0), "center": (0.0, 0.0, 0.0), "eps": 12.1104}],
        "clad_eps": 1.44**2,
        "target_neff": 3.0,
        "direction": "-",
    },
    "strip_y_scalar_double_3freq": {
        "support": "production",
        "boxes": [{"size": (1.5, 100.0, 1.0), "center": (0.0, 0.0, 0.0), "eps": 4.0}],
        "clad_eps": 1.0,
        "target_neff": 2.0,
        "direction": "-",
        "dmin_pmc": (False, True),
    },
    "strip_x_lossy_double": {
        "support": "production",
        "boxes": [{"size": (100.0, 1.2, 0.9), "center": (0.0, 0.0, 0.0), "eps": 4.0, "conductivity": 1e-4}],
        "clad_eps": 1.0,
        "target_neff": 2.0,
        "direction": "-",
        "solve_each_frequency": True,
    },
    "bend_y_radius": {
        "support": "production",
        "boxes": [{"size": (1.5, 100.0, 1.0), "center": (0.0, 0.0, 0.0), "eps": 4.0}],
        "clad_eps": 1.0,
        "target_neff": 2.0,
        "bend_radius": 5.0,
        "bend_axis": 1,
        "trim_edges": ((0, 0), (1, 1)),
    },
    "interp_linear_reduced": {
        "support": "production",
        "boxes": [{"size": (0.8, 0.8, 100.0), "center": (0.0, 0.0, 0.0), "eps": 4.0}],
        "clad_eps": 1.0,
        "target_neff": 2.0,
    },
    "slot_waveguide_silicon": {
        "support": "production",
        "boxes": [
            {"size": (0.24, 0.22, 100.0), "center": (-0.18, 0.0, 0.0), "eps": 3.48**2},
            {"size": (0.24, 0.22, 100.0), "center": (0.18, 0.0, 0.0), "eps": 3.48**2},
        ],
        "clad_eps": 1.44**2,
        "target_neff": 2.4,
    },
    "rib_waveguide_silicon": {
        "support": "production",
        "boxes": [
            {"size": (1.40, 0.10, 100.0), "center": (0.0, -0.08, 0.0), "eps": 3.48**2},
            {"size": (0.55, 0.22, 100.0), "center": (0.0, 0.08, 0.0), "eps": 3.48**2},
        ],
        "clad_eps": 1.44**2,
        "target_neff": 2.6,
    },
    "cylindrical_rod_air": {
        "support": "production",
        "circles": [{"radius": 0.35, "center": (0.0, 0.0, 0.0), "eps": 4.0}],
        "clad_eps": 1.0,
        "target_neff": 1.5,
    },
    "partial_fields_subset": {
        "support": "production",
        "clad_eps": 1.0,
        "target_neff": 1.0,
    },
    "sort_n_eff_ascending": {
        "support": "production",
        "clad_eps": 2.0,
        "target_neff": 3.48,
        "direction": "-",
        "dmin_pmc": (False, True),
        "trim_edges": ((1, 1), (0, 0)),
        "backend_tolerances": {"rust_sparse": 1e-5},
        "sort_order": "ascending",
        "krylov_dim": 64,
    },
    "pec_parallel_plate_y": {
        "support": "outside_80_target",
        "unsupported": "PEC boundary conditions are outside the local solver's 80% target",
    },
    "simulation_2d_te": {
        "support": "future_feature",
        "unsupported": "2D simulation reduction, TE filtering, and PML are outside this raster comparison",
    },
    "pml_cross_section_y": {
        "support": "future_fixture_harness",
        "unsupported": "PML comparison is not implemented for the local Rust fixture harness",
    },
    "custom_cartesian_left": {
        "support": "metadata_missing",
        "unsupported": "custom Cartesian media are outside the current local reconstruction harness",
    },
    "custom_cartesian_right": {
        "support": "metadata_missing",
        "unsupported": "custom Cartesian media are outside the current local reconstruction harness",
    },
    "polyslab_sloped_sidewall": {
        "support": "metadata_missing",
        "unsupported": "sloped PolySlab rasterization metadata is not encoded in the neutral fixture",
    },
    "interp_cubic_strip": {
        "support": "metadata_missing",
        "unsupported": "exact interpolation fixture recipe is not encoded in the neutral fixture",
    },
    "interp_poly_cheb_strip": {
        "support": "metadata_missing",
        "unsupported": "exact Chebyshev interpolation fixture recipe is not encoded in the neutral fixture",
    },
    "sort_mode_area_te_filter": {
        "support": "metadata_missing",
        "unsupported": "mode-area sorting and TE filtering fixture recipe is not encoded in the neutral fixture",
    },
}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _solver_edges_from_field_coords(edges: tuple[np.ndarray, np.ndarray], recipe: dict) -> tuple[np.ndarray, np.ndarray]:
    dmin_pmc = tuple(bool(value) for value in recipe.get("dmin_pmc", (False, False)))
    trim_edges = tuple(recipe.get("trim_edges", ((0, 0), (0, 0))))
    out = []
    for axis_edges, has_min_symmetry, (trim_start, trim_end) in zip(edges, dmin_pmc, trim_edges):
        if not has_min_symmetry:
            trimmed = axis_edges
            if trim_start or trim_end:
                end = None if trim_end == 0 else -int(trim_end)
                trimmed = trimmed[int(trim_start) : end]
            out.append(trimmed)
            continue
        trimmed = axis_edges[axis_edges >= -1e-12].copy()
        if trim_start or trim_end:
            end = None if trim_end == 0 else -int(trim_end)
            trimmed = trimmed[int(trim_start) : end]
        if trimmed.size == 0:
            raise ValueError("could not trim symmetric fixture coordinates to the positive half-plane")
        if abs(trimmed[0]) < 1e-12:
            trimmed[0] = 0.0
        out.append(trimmed)
    return tuple(out)  # type: ignore[return-value]


def _solve_recipe(
    *,
    sm,
    recipe: dict,
    ref_n,
    edges: tuple[np.ndarray, np.ndarray],
    tangent_dims: tuple[str, str],
    normal_dim: str,
    normal_coord: float,
):
    freqs = tuple(float(freq) for freq in ref_n.coords["f"].values)
    if recipe.get("solve_each_frequency"):
        rows = []
        field_rows = None
        first_result = None
        for freq in freqs:
            result = _solve_recipe_for_freq(
                sm=sm,
                recipe=recipe,
                freq=freq,
                num_modes=ref_n.shape[1],
                edges=edges,
                tangent_dims=tangent_dims,
                normal_dim=normal_dim,
                normal_coord=normal_coord,
            )
            if first_result is None:
                first_result = result
            rows.append(result.n_complex.values[0])
            if field_rows is None:
                field_rows = {name: [] for name in result.field_components}
            for name, data_array in result.field_components.items():
                field_rows[name].append(data_array.values[:, :, :, 0, :])
        if first_result is None:
            raise RuntimeError("no frequencies to solve")
        n_complex = np.asarray(rows, dtype=np.complex128)
        n_complex_array = xr.DataArray(
            n_complex,
            dims=("f", "mode_index"),
            coords={"f": np.asarray(freqs), "mode_index": np.arange(n_complex.shape[1])},
        )
        field_components = dict(first_result.field_components)
        if field_rows is not None:
            for name, rows_for_name in field_rows.items():
                values = np.stack(rows_for_name, axis=3)
                coords_for_field = dict(first_result.field_components[name].coords)
                coords_for_field["f"] = np.asarray(freqs)
                field_components[name] = xr.DataArray(
                    values,
                    dims=first_result.field_components[name].dims,
                    coords=coords_for_field,
                )
        return first_result.__class__(
            n_complex=n_complex_array,
            n_group=None,
            field_components=field_components,
        )
    return _solve_recipe_for_freq(
        sm=sm,
        recipe=recipe,
        freq=freqs,
        num_modes=ref_n.shape[1],
        edges=edges,
        tangent_dims=tangent_dims,
        normal_dim=normal_dim,
        normal_coord=normal_coord,
    )


def _solve_recipe_for_freq(
    *,
    sm,
    recipe: dict,
    freq,
    num_modes: int,
    edges: tuple[np.ndarray, np.ndarray],
    tangent_dims: tuple[str, str],
    normal_dim: str,
    normal_coord: float,
):
    freqs = (float(freq),) if np.isscalar(freq) else tuple(float(value) for value in freq)
    centers = tuple((axis_edges[:-1] + axis_edges[1:]) / 2 for axis_edges in edges)
    eps_xx, eps_yy, eps_zz = _eps_components_from_recipe(recipe, edges, centers, tangent_dims, freqs[0], sm)
    material_grid = sm.Materials.from_diagonal(
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        x_edges=edges[0],
        y_edges=edges[1],
        normal_axis=_AXIS_INDEX[normal_dim],
        normal_coordinate=normal_coord,
    )
    return sm.solve_modes(
        material_grid=material_grid,
        freqs=freqs,
        num_modes=num_modes,
        target_neff=recipe.get("target_neff"),
        direction=recipe.get("direction", "+"),
        pml=sm.PmlSpec(num_cells=tuple(recipe.get("num_pml", (0, 0)))),
        boundary=sm.BoundarySpec(
            low=tuple("pmc" if bool(value) else "pec" for value in recipe.get("dmin_pmc", (False, False)))
        ),
        bend_radius=recipe.get("bend_radius"),
        bend_axis=recipe.get("bend_axis", 0),
        krylov_dim=recipe.get("krylov_dim"),
    )


def _eps_components_from_recipe(
    recipe: dict,
    edges: tuple[np.ndarray, np.ndarray],
    coords: tuple[np.ndarray, np.ndarray],
    tangent_dims: tuple[str, str],
    freq: float,
    sm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if recipe.get("yee_staggered", True):
        return (
            _eps_from_recipe(recipe, (coords[0], edges[1][:-1]), tangent_dims, freq, sm),
            _eps_from_recipe(recipe, (edges[0][:-1], coords[1]), tangent_dims, freq, sm),
            _eps_from_recipe(recipe, (edges[0][:-1], edges[1][:-1]), tangent_dims, freq, sm),
        )
    eps = _eps_from_recipe(recipe, coords, tangent_dims, freq, sm)
    return eps, eps, eps


def _eps_from_recipe(
    recipe: dict,
    coords: tuple[np.ndarray, np.ndarray],
    tangent_dims: tuple[str, str],
    freq: float,
    sm,
) -> np.ndarray:
    grids = np.meshgrid(*coords, indexing="ij")
    eps = np.full(tuple(len(coord) for coord in coords), recipe.get("clad_eps", 1.0), dtype=np.complex128)
    for box in recipe.get("boxes", ()):
        mask = np.ones(eps.shape, dtype=bool)
        center = box.get("center", (0.0, 0.0, 0.0))
        size = box["size"]
        for grid, dim in zip(grids, tangent_dims):
            axis = _AXIS_INDEX[dim]
            mask &= np.abs(grid - center[axis]) <= abs(size[axis]) / 2
        eps_value = complex(box["eps"])
        conductivity = float(box.get("conductivity", 0.0) or 0.0)
        if conductivity:
            eps_value += 1j * conductivity / (2 * np.pi * freq * sm.EPSILON_0)
        eps[mask] = eps_value
    for circle in recipe.get("circles", ()):
        center = circle.get("center", (0.0, 0.0, 0.0))
        radius = float(circle["radius"])
        distance_sq = np.zeros(eps.shape, dtype=float)
        for grid, dim in zip(grids, tangent_dims):
            distance_sq += (grid - center[_AXIS_INDEX[dim]]) ** 2
        eps[distance_sq <= radius * radius] = complex(circle["eps"])
    return eps


def _reorder_modes(values: np.ndarray, recipe: dict) -> np.ndarray:
    if recipe.get("sort_order") != "ascending":
        return values
    order = np.argsort(values.real, axis=1)
    return np.take_along_axis(values, order, axis=1)


def _reorder_field_modes(values: np.ndarray, recipe: dict) -> np.ndarray:
    if recipe.get("sort_order") != "ascending":
        return values
    # Field reordering is only used for a coarse overlap diagnostic; n sorting is authoritative.
    return values[..., ::-1]


def _status(status: str, summary: str, **details) -> dict:
    return {"status": status, "failed": status == "fail", "summary": summary, **details}


def _n_tolerance(entry: dict, recipe: dict | None = None) -> float:
    if recipe is not None:
        backend_tolerance = recipe.get("backend_tolerances", {}).get("rust_sparse")
        if backend_tolerance is not None:
            return float(backend_tolerance)
    return float(
        entry.get("tolerances", {}).get(
            "n_complex_atol",
            entry.get("case", {}).get("tolerances", {}).get("n_complex_atol", 1e-6),
        )
    )


if __name__ == "__main__":
    main()
