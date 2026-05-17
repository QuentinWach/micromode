from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import micromode as mm


def _strip_grid(nx: int = 5, ny: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(-1.0, 1.0, nx + 1)
    y_edges = np.linspace(-0.8, 0.8, ny + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    eps = np.where((np.abs(xx) <= 0.35) & (np.abs(yy) <= 0.25), 3.4**2, 1.44**2)
    return eps, x_edges, y_edges


def test_grid_api_solves_with_rust_sparse_backend():
    eps, x_edges, y_edges = _strip_grid(6, 5)
    freq = mm.C_0 / 1.55

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[freq],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    assert data.n_complex.shape == (1, 2)
    assert data.n_complex.values[0, 0].real >= data.n_complex.values[0, 1].real
    assert data.field_components["Ex"].shape == (6, 5, 1, 1, 2)
    assert set(data.field_components) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    assert data.solver_info["runs"][0]["phase_convention"] == "dominant_e_real_positive"
    assert data.solver_info["runs"][0]["normalization"] == "lorentz_orthogonal_unit_transverse_power"
    np.testing.assert_allclose(data.solver_info["runs"][0]["power_norms"], np.ones(2), rtol=1e-10, atol=1e-10)
    assert data.solver_info["runs"][0]["lorentz_orthogonality_error"] < 1e-8
    assert abs(abs(data.overlap(mode_index=0, kind="power", normalize=False)) - 1.0) < 1e-10
    lorentz_matrix = data.overlap_matrix(kind="lorentz").values
    np.testing.assert_allclose(lorentz_matrix - np.diag(np.diag(lorentz_matrix)), 0.0, atol=1e-8)
    electric = np.stack(
        [data.field_components[component].isel(f=0, mode_index=0).values for component in ("Ex", "Ey", "Ez")]
    )
    anchor = electric.reshape(-1)[np.argmax(np.abs(electric.reshape(-1)))]
    assert anchor.real >= 0.0
    assert abs(anchor.imag) <= 1e-10 * max(abs(anchor), 1.0)


def test_materials_api_matches_component_api_for_diagonal_grid():
    eps, x_edges, y_edges = _strip_grid()
    freq = mm.C_0 / 1.55
    materials = mm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges)

    from_materials = mm.solve_modes(
        material_grid=materials,
        freqs=[freq],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )
    from_components = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[freq],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    np.testing.assert_allclose(from_materials.n_complex.values, from_components.n_complex.values)
    assert from_materials.field_components["Ex"].shape == (5, 4, 1, 1, 2)


def test_materials_api_accepts_full_tensor_grid():
    x_edges = np.linspace(-1.0, 1.0, 5)
    y_edges = np.linspace(-0.8, 0.8, 4)
    shape = (len(x_edges) - 1, len(y_edges) - 1)
    eps_xx = np.full(shape, 2.2**2)
    eps_yy = np.full(shape, 2.0**2)
    eps_zz = np.full(shape, 1.9**2)
    eps_xz = np.full(shape, 0.03)
    eps_zx = np.full(shape, 0.03)
    materials = mm.Materials.from_components(
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        eps_xz=eps_xz,
        eps_zx=eps_zx,
        x_edges=x_edges,
        y_edges=y_edges,
    )

    data = mm.solve_modes(
        material_grid=materials,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.0,
        krylov_dim=20,
    )

    assert not materials.is_diagonal
    assert data.n_complex.shape == (1, 1)
    assert np.isfinite(data.n_complex.values).all()
    assert data.field_components["Ez"].shape == (4, 3, 1, 1, 1)


def test_solver_specs_report_diagnostics_and_persist_to_hdf5(tmp_path):
    eps, x_edges, y_edges = _strip_grid(6, 5)
    pml = mm.PmlSpec(num_cells=(1, 1), sigma_max=1.6, kappa_min=1.0, kappa_max=2.2, order=2)
    boundary = mm.BoundarySpec(low=("pec", "pmc"))

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        pml=pml,
        boundary=boundary,
        krylov_dim=18,
    )

    assert data.solver_info["pml"]["num_cells"] == (1, 1)
    assert data.solver_info["pml"]["sigma_max"] == pytest.approx(1.6)
    assert data.solver_info["boundary"]["low"] == ("pec", "pmc")
    assert data.solver_info["backend"]
    assert data.solver_info["runs"][0]["operator_size"] > 0
    assert data.solver_info["runs"][0]["operator_nnz"] > 0
    assert len(data.solver_info["runs"][0]["residuals"]) == 1
    assert data.solver_info["runs"][0]["power_norms"][0] == pytest.approx(1.0)
    assert len(data.solver_info["runs"][0]["lorentz_norms"]) == 1

    loaded = mm.Result.from_hdf5(data.to_hdf5(tmp_path / "diagnostic_result.h5"))
    assert loaded.solver_info["pml"]["num_cells"] == [1, 1]
    assert loaded.solver_info["boundary"]["low"] == ["pec", "pmc"]
    assert loaded.solver_info["runs"][0]["operator_size"] == data.solver_info["runs"][0]["operator_size"]


def test_slice_api_solves_1d_x_slice_and_matches_single_cell_grid(tmp_path):
    x_edges = np.linspace(-1.0, 1.0, 9)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    eps = np.where(np.abs(x_centers) <= 0.3, 3.4**2, 1.44**2)
    y_edges = np.asarray([-0.125, 0.125])

    sliced = mm.solve_slice(
        eps_xx=eps,
        coord_edges=x_edges,
        axis="x",
        invariant_width=0.25,
        freqs=[mm.C_0 / 1.55],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )
    padded = mm.solve_grid(
        eps_xx=eps[:, None],
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    np.testing.assert_allclose(sliced.n_complex.values, padded.n_complex.values)
    assert sliced.field_components["Ex"].shape == (8, 1, 1, 1, 2)
    assert sliced.field_components["Ex"].attrs["normal_dim"] == "z"
    np.testing.assert_allclose(sliced.field_components["Ex"].coords["y_width"].values, [0.25])
    assert sliced.overlap(mode_index=0, kind="power") == pytest.approx(1.0 + 0.0j)

    path = sliced.to_hdf5(tmp_path / "slice_result.h5")
    loaded = mm.Result.from_hdf5(path)
    assert loaded.field_components["Ex"].attrs["normal_dim"] == "z"
    np.testing.assert_allclose(loaded.field_components["Ex"].coords["y_width"].values, [0.25])

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ax = sliced.plot_field("Ex", mode_index=0)
    assert ax.lines
    plt.close("all")


def test_materials_slice_api_supports_y_axis_and_tensor_components():
    y_edges = np.linspace(-0.8, 0.8, 7)
    shape = (len(y_edges) - 1,)
    eps_xx = np.full(shape, 2.2**2)
    eps_yy = np.full(shape, 2.0**2)
    eps_zz = np.full(shape, 1.9**2)
    eps_xz = np.full(shape, 0.02)
    materials = mm.Materials.from_slice(
        coord_edges=y_edges,
        axis="y",
        invariant_width=0.4,
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        eps_xz=eps_xz,
    )

    assert materials.shape == (1, 6)
    assert not materials.is_diagonal
    np.testing.assert_allclose(materials.eps_tensor[0, 2, 0, :], eps_xz)

    data = mm.solve_modes(
        material_grid=materials,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.0,
        krylov_dim=16,
    )

    assert data.field_components["Ey"].shape == (1, 6, 1, 1, 1)
    assert data.field_components["Ey"].attrs["normal_dim"] == "z"


def test_y_normal_solve_returns_global_component_names():
    z_edges = np.linspace(-1.0, 1.0, 9)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    eps = np.where(np.abs(z_centers) <= 0.3, 3.4**2, 1.44**2)

    data = mm.solve_slice(
        eps_xx=eps,
        coord_edges=z_edges,
        axis="y",
        invariant_width=0.5,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        normal_axis=1,
        components=("Ex", "Ez", "Hy", "Hz"),
        krylov_dim=16,
    )

    assert set(data.field_components) == {"Ex", "Ez", "Hy", "Hz"}
    assert data.field_components["Ex"].dims[:3] == ("x", "z", "y")
    assert data.field_components["Ex"].attrs["normal_dim"] == "y"
    assert np.linalg.norm(data.field_components["Ex"].values) > 0
    assert np.linalg.norm(data.field_components["Ez"].values) > 0


def test_y_normal_power_overlap_is_positive():
    z_edges = np.linspace(-1.0, 1.0, 9)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    eps = np.where(np.abs(z_centers) <= 0.3, 3.4**2, 1.44**2)

    data = mm.solve_slice(
        eps_xx=eps,
        coord_edges=z_edges,
        axis="y",
        invariant_width=0.5,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        normal_axis=1,
        krylov_dim=16,
    )

    power = data.overlap(mode_index=0, kind="power", normalize=False)
    assert power.real > 0
    assert abs(abs(power) - 1.0) < 1e-10


def test_materials_subpixel_averaging_helpers():
    x_edges = np.linspace(-1.0, 1.0, 3)
    y_edges = np.linspace(-0.5, 0.5, 3)
    samples = np.asarray(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
            [9.0, 11.0, 13.0, 15.0],
            [10.0, 12.0, 14.0, 16.0],
        ]
    )

    averaged = mm.Materials.average_subpixels(samples, shape=(2, 2), subpixel_shape=(2, 2))
    np.testing.assert_allclose(averaged, [[2.5, 6.5], [10.5, 14.5]])

    harmonic = mm.Materials.average_subpixels(
        samples,
        shape=(2, 2),
        subpixel_shape=(2, 2),
        method="harmonic",
    )
    assert np.all(harmonic.real < averaged.real)

    materials = mm.Materials.from_subpixel_diagonal(
        eps_xx=samples,
        x_edges=x_edges,
        y_edges=y_edges,
        subpixel_shape=(2, 2),
    )
    np.testing.assert_allclose(materials.diagonal_eps()[0], averaged)
    assert materials.shape == (2, 2)


def test_angle_and_bend_use_tensorial_rust_path():
    eps, x_edges, y_edges = _strip_grid()

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        angle_theta=0.08,
        angle_phi=0.25,
        bend_radius=8.0,
        bend_axis=0,
        krylov_dim=24,
    )

    assert data.n_complex.shape == (1, 1)
    assert np.isfinite(data.n_complex.values).all()
    assert np.isfinite(data.field_components["Hz"].values).all()


def test_full_tensor_grid_supports_angle_and_bend_transform():
    x_edges = np.linspace(-1.0, 1.0, 5)
    y_edges = np.linspace(-0.8, 0.8, 4)
    shape = (len(x_edges) - 1, len(y_edges) - 1)
    materials = mm.Materials.from_components(
        eps_xx=np.full(shape, 2.2**2),
        eps_yy=np.full(shape, 2.0**2),
        eps_zz=np.full(shape, 1.9**2),
        eps_xz=np.full(shape, 0.02),
        eps_zx=np.full(shape, 0.02),
        x_edges=x_edges,
        y_edges=y_edges,
    )

    data = mm.solve_modes(
        material_grid=materials,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.0,
        angle_theta=0.04,
        angle_phi=0.2,
        bend_radius=10.0,
        bend_axis=0,
        krylov_dim=18,
    )

    assert data.solver_info["runs"][0]["backend_kind"] == "tensorial_sparse"
    assert np.isfinite(data.n_complex.values).all()
    assert np.isfinite(data.field_components["Ex"].values).all()


def test_spec_can_drive_grid_solve_options():
    eps, x_edges, y_edges = _strip_grid()
    spec = mm.Spec(num_modes=1, target_neff=2.5)

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        spec=spec,
        krylov_dim=16,
    )

    assert data.n_complex.dtype == np.dtype("complex128")
    assert {field.dtype for field in data.field_components.values()} == {np.dtype("complex128")}


def test_result_metrics_io_plotting_and_overlap(tmp_path):
    eps, x_edges, y_edges = _strip_grid(6, 5)
    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    assert np.isfinite(data.k_eff.values).all()
    assert np.isfinite(data.pol_fraction["te"].values).all()
    assert np.isfinite(data.pol_fraction_waveguide["te"].values).all()
    assert np.isfinite(data.mode_area.values).all()
    frame = data.to_dataframe()
    assert {"n eff", "k eff", "mode area", "TE fraction", "wg TE fraction"} <= set(frame.columns)

    path = data.to_hdf5(tmp_path / "mode_result.h5")
    loaded = mm.Result.from_hdf5(path)
    np.testing.assert_allclose(loaded.n_complex.values, data.n_complex.values)
    assert set(loaded.field_components) == set(data.field_components)

    assert abs(data.overlap(mode_index=0, kind="power")) > 0
    matrix = data.overlap_matrix(kind="electric")
    assert matrix.shape == (2, 2)
    assert mm.overlap(data, mode_index=0, kind="electric") == pytest.approx(1.0 + 0.0j)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ax = data.plot_field("Ex", mode_index=0)
    assert ax.get_title()
    fig, axes = data.plot_field_components(components=("Ex", "Ey", "Ez"), mode_index=0)
    assert fig is not None
    assert axes.size >= 3
    plt.close("all")


def test_result_overlap_for_synthetic_orthogonal_modes():
    dims = ("x", "y", "z", "f", "mode_index")
    coords = {
        "x": np.asarray([-0.5, 0.5]),
        "y": np.asarray([-0.25, 0.25]),
        "z": np.asarray([0.0]),
        "f": np.asarray([mm.C_0 / 1.55]),
        "mode_index": np.arange(2),
    }
    shape = (2, 2, 1, 1, 2)
    values = {component: np.zeros(shape, dtype=np.complex128) for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    values["Ex"][..., 0] = 1.0
    values["Hy"][..., 0] = 1.0
    values["Ey"][..., 1] = 1.0
    values["Hx"][..., 1] = -1.0
    fields = {component: xr.DataArray(array, dims=dims, coords=coords) for component, array in values.items()}
    data = mm.Result(
        n_complex=xr.DataArray(
            [[2.0 + 0j, 1.8 + 0j]],
            dims=("f", "mode_index"),
            coords={"f": coords["f"], "mode_index": coords["mode_index"]},
        ),
        field_components=fields,
    )

    assert data.overlap(mode_index=0, kind="power") == pytest.approx(1.0 + 0.0j)
    assert data.overlap(mode_index=0, other_mode_index=1, kind="power") == pytest.approx(0.0 + 0.0j)
    assert data.overlap(mode_index=0, other_mode_index=1, kind="lorentz") == pytest.approx(0.0 + 0.0j)
    assert mm.overlap(data, mode_index=1, kind="electric") == pytest.approx(1.0 + 0.0j)
    np.testing.assert_allclose(data.overlap_matrix(kind="power").values, np.eye(2))
    np.testing.assert_allclose(data.overlap_matrix(kind="lorentz").values, np.eye(2))


def test_sweep_tracks_modes_by_overlap():
    dims = ("x", "y", "z", "f", "mode_index")
    coords = {
        "x": np.asarray([-0.5, 0.5]),
        "y": np.asarray([-0.25, 0.25]),
        "z": np.asarray([0.0]),
        "f": np.asarray([mm.C_0 / 1.55]),
        "mode_index": np.arange(2),
    }
    shape = (2, 2, 1, 1, 2)

    def result(n_values: list[float], order: tuple[str, str]) -> mm.Result:
        fields = {component: np.zeros(shape, dtype=np.complex128) for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        for mode_index, component in enumerate(order):
            fields[component][..., mode_index] = 1.0
        arrays = {component: xr.DataArray(values, dims=dims, coords=coords) for component, values in fields.items()}
        return mm.Result(
            n_complex=xr.DataArray(
                [np.asarray(n_values, dtype=np.complex128)],
                dims=("f", "mode_index"),
                coords={"f": coords["f"], "mode_index": coords["mode_index"]},
            ),
            field_components=arrays,
        )

    first = result([2.2, 1.9], ("Ex", "Ey"))
    second = result([1.91, 2.18], ("Ey", "Ex"))

    tracked = mm.track_modes_by_overlap([first, second], kind="electric")
    sweep = mm.Sweep(values=np.asarray([0.4, 0.5]), results=tracked, parameter_name="width")

    np.testing.assert_allclose(tracked[1].n_complex.values, [[2.18, 1.91]])
    np.testing.assert_allclose(sweep.n_eff[:, 0], [2.2, 2.18])
    assert {"width", "mode_index", "n_eff", "te_fraction"} <= set(sweep.to_dataframe().columns)


def test_spec_validation_for_core_options():
    spec = mm.Spec(
        num_modes=2,
        target_neff=2.0,
        angle_theta=0.1,
        angle_phi=0.2,
        bend_radius=5.0,
        bend_axis=1,
        pml=mm.PmlSpec(num_cells=(1, 2), sigma_max=1.5),
        boundary=mm.BoundarySpec(low=("pmc", "pec")),
    )

    assert spec.pml.num_cells == (1, 2)
    assert spec.pml.sigma_max == pytest.approx(1.5)
    assert spec.boundary.dmin_pmc == (True, False)
    assert spec.has_angle
    assert spec.has_bend
    assert spec.has_transform

    with pytest.raises(ValueError, match="num_modes"):
        mm.Spec(num_modes=0)
    with pytest.raises(ValueError, match="bend_axis"):
        mm.Spec(bend_radius=5.0)
