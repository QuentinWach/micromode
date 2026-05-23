from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks.compare_mode_solver_fixtures import _compare_local_case
from benchmarks.mode_solver.fixtures import (
    data_path,
    load_data_array,
    manifest_path,
    phase_aligned_relative_error,
    read_json,
    sha256_file,
    summary_path,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_FIXTURE_ROOT = ROOT / "fixtures" / "mode_solver" / "smoke"
EXTENDED_FIXTURE_ROOT = ROOT / "fixtures" / "mode_solver" / "extended"


def test_smoke_fixture_manifest_and_hashes_are_current():
    _assert_fixture_manifest(SMOKE_FIXTURE_ROOT)


def test_extended_fixture_manifest_and_hashes_are_current():
    _assert_fixture_manifest(EXTENDED_FIXTURE_ROOT)


def test_reference_fixture_files_do_not_embed_external_package_name():
    needle = bytes.fromhex("746964793364")
    for root in (SMOKE_FIXTURE_ROOT, EXTENDED_FIXTURE_ROOT):
        for path in root.rglob("*"):
            if path.is_file():
                assert needle not in path.read_bytes().lower(), path


def test_reference_hdf5_files_do_not_store_serialized_solver_metadata():
    import h5py

    for root in (SMOKE_FIXTURE_ROOT, EXTENDED_FIXTURE_ROOT):
        for entry in read_json(manifest_path(root))["cases"]:
            with h5py.File(data_path(root, entry["case_id"]), "r") as handle:
                assert "JSON_STRING" not in handle


def test_reference_n_complex_matches_summary_payload():
    for root in (SMOKE_FIXTURE_ROOT, EXTENDED_FIXTURE_ROOT):
        for entry in read_json(manifest_path(root))["cases"]:
            case_id = entry["case_id"]
            data = load_data_array(data_path(root, case_id), "n_complex")
            summary = read_json(summary_path(root, case_id))
            summary_payload = summary["scalars"]["n_complex"]
            expected = _array_from_summary_values(summary_payload["values"])

            assert list(data.dims) == summary_payload["dims"]
            assert list(data.shape) == summary_payload["values"]["shape"]
            np.testing.assert_allclose(data.values, expected)


def test_reference_field_signatures_match_summary_payload():
    for root in (SMOKE_FIXTURE_ROOT, EXTENDED_FIXTURE_ROOT):
        for entry in read_json(manifest_path(root))["cases"]:
            case_id = entry["case_id"]
            mode_data = data_path(root, case_id)
            summary = read_json(summary_path(root, case_id))
            for component, signature in summary["fields"].items():
                data = load_data_array(mode_data, component)
                assert list(data.dims) == signature["dims"]
                assert list(data.shape) == signature["shape"]
                assert str(data.dtype) == signature["dtype"]


def test_phase_aligned_relative_error_accepts_global_complex_phase():
    golden = np.array([1 + 2j, -3 + 1j, 0.5 - 0.25j])
    actual = golden * np.exp(1j * 0.73)

    rel, overlap_error = phase_aligned_relative_error(golden, actual)

    assert rel < 1e-14
    assert overlap_error < 1e-14


@pytest.mark.slow
def test_local_fixture_comparison_uses_staggered_rasterization_for_z_strips():
    manifest = read_json(manifest_path(EXTENDED_FIXTURE_ROOT))
    entries = {entry["case_id"]: entry for entry in manifest["cases"]}

    for case_id in ("strip_z_scalar_single", "group_index_silicon_strip"):
        status = _compare_local_case(EXTENDED_FIXTURE_ROOT, entries[case_id])
        assert status["status"] == "pass"
        assert status["n_complex_max_abs_error"] <= status["n_complex_atol"]


@pytest.mark.slow
def test_scipy_fixture_comparison_uses_staggered_rasterization_for_z_strips():
    pytest.importorskip("scipy")
    manifest = read_json(manifest_path(EXTENDED_FIXTURE_ROOT))
    entries = {entry["case_id"]: entry for entry in manifest["cases"]}

    for case_id in ("strip_z_scalar_single", "group_index_silicon_strip"):
        status = _compare_local_case(EXTENDED_FIXTURE_ROOT, entries[case_id], backend="scipy_reference")
        assert status["status"] == "pass"
        assert status["n_complex_max_abs_error"] <= status["n_complex_atol"]


@pytest.mark.slow
def test_local_production_fixture_matrix_passes():
    from benchmarks.compare_mode_solver_fixtures import _LOCAL_CASES

    manifest = read_json(manifest_path(EXTENDED_FIXTURE_ROOT))
    entries = {entry["case_id"]: entry for entry in manifest["cases"]}
    production_ids = [
        case_id
        for case_id, recipe in _LOCAL_CASES.items()
        if recipe.get("support", "production") == "production" and case_id in entries
    ]

    assert production_ids
    for case_id in production_ids:
        status = _compare_local_case(EXTENDED_FIXTURE_ROOT, entries[case_id])
        assert status["support"] == "production"
        assert status["status"] == "pass", f"{case_id}: {status['summary']}"
        assert status["n_complex_max_abs_error"] <= status["n_complex_atol"]


@pytest.mark.slow
def test_scipy_production_fixture_matrix_passes():
    pytest.importorskip("scipy")
    from benchmarks.compare_mode_solver_fixtures import _LOCAL_CASES

    manifest = read_json(manifest_path(EXTENDED_FIXTURE_ROOT))
    entries = {entry["case_id"]: entry for entry in manifest["cases"]}
    production_ids = [
        case_id
        for case_id, recipe in _LOCAL_CASES.items()
        if recipe.get("support", "production") == "production" and case_id in entries
    ]

    assert production_ids
    for case_id in production_ids:
        status = _compare_local_case(EXTENDED_FIXTURE_ROOT, entries[case_id], backend="scipy_reference")
        assert status["support"] == "production"
        assert status["status"] == "pass", f"{case_id}: {status['summary']}"
        assert status["n_complex_max_abs_error"] <= status["n_complex_atol"]


@pytest.mark.slow
def test_unsupported_fixture_matrix_is_explicit():
    from benchmarks.compare_mode_solver_fixtures import _LOCAL_CASES

    manifest = read_json(manifest_path(EXTENDED_FIXTURE_ROOT))
    entries = {entry["case_id"]: entry for entry in manifest["cases"]}
    allowed_support = {"outside_80_target", "future_feature", "future_fixture_harness", "metadata_missing"}

    unsupported_ids = [
        case_id
        for case_id, recipe in _LOCAL_CASES.items()
        if recipe.get("support", "production") != "production" and case_id in entries
    ]
    assert unsupported_ids
    for case_id in unsupported_ids:
        recipe = _LOCAL_CASES[case_id]
        assert recipe["support"] in allowed_support
        assert recipe.get("unsupported")
        status = _compare_local_case(EXTENDED_FIXTURE_ROOT, entries[case_id])
        assert status["status"] == "unsupported"
        assert status["support"] == recipe["support"]


def _assert_fixture_manifest(root: Path) -> None:
    manifest = read_json(manifest_path(root))
    expected_ids = [case["case_id"] for case in manifest["registered_cases"]]
    actual_ids = [case["case_id"] for case in manifest["cases"]]

    assert manifest["schema_version"] == 1
    assert actual_ids == expected_ids[: len(actual_ids)]
    assert manifest["case_count"] == len(actual_ids)
    assert "reference_solver_version" in manifest["environment"]

    for entry in manifest["cases"]:
        assert set(entry["files"]) == {"mode_data", "summary"}
        assert set(entry["sha256"]) == {"mode_data", "summary"}
        for key, rel_path in entry["files"].items():
            path = root / rel_path
            assert path.exists(), f"missing {key} file for {entry['case_id']}: {path}"
            assert sha256_file(path) == entry["sha256"][key]


def _array_from_summary_values(payload: dict) -> np.ndarray:
    if "real" in payload:
        return np.asarray(payload["real"]) + 1j * np.asarray(payload["imag"])
    return np.asarray(payload["values"])
