from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

SCHEMA_VERSION = 1
DEFAULT_FIXTURE_ROOT = Path("fixtures/mode_solver")
_XARRAY_VALUE_NAME = "__xarray_dataarray_variable__"
_PREFERRED_DIMS = ("x", "y", "z", "f", "mode_index")


def case_dir(root: Path, case_id: str) -> Path:
    return root / case_id


def data_path(root: Path, case_id: str) -> Path:
    return case_dir(root, case_id) / "mode_data.hdf5"


def summary_path(root: Path, case_id: str) -> Path:
    return case_dir(root, case_id) / "summary.json"


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def iter_manifest_entries(root: Path) -> tuple[dict[str, Any], ...]:
    return tuple(read_json(manifest_path(root))["cases"])


def load_data_array(path: Path, name: str) -> xr.DataArray:
    """Load one xarray-style data array from a committed reference HDF5 file."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency is part of the project.
        raise ImportError("h5py is required for reference fixture loading") from exc

    with h5py.File(path, "r") as handle:
        if name not in handle:
            raise KeyError(f"{path} does not contain data array {name!r}")
        return _read_xarray_group(handle[name])


def _read_xarray_group(group: Any) -> xr.DataArray:
    if _XARRAY_VALUE_NAME not in group:
        raise KeyError(f"HDF5 group {group.name!r} is missing {_XARRAY_VALUE_NAME!r}")
    values = group[_XARRAY_VALUE_NAME][()]
    coords = {name: group[name][()] for name in group if name != _XARRAY_VALUE_NAME and hasattr(group[name], "shape")}
    dims = _infer_dims(values.shape, coords)
    xarray_coords = {dim: coords[dim] for dim in dims if dim in coords}
    return xr.DataArray(values, dims=dims, coords=xarray_coords)


def _infer_dims(shape: tuple[int, ...], coords: dict[str, np.ndarray]) -> tuple[str, ...]:
    dims = tuple(dim for dim in _PREFERRED_DIMS if dim in coords)
    if len(dims) == len(shape) and all(len(coords[dim]) == size for dim, size in zip(dims, shape, strict=True)):
        return dims

    exact = [dim for dim in _PREFERRED_DIMS if dim in coords and len(coords[dim]) in set(shape)]
    if len(exact) == len(shape):
        return tuple(exact)
    raise ValueError(f"could not infer data-array dims for shape {shape} and coords {sorted(coords)}")


def phase_aligned_relative_error(golden: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    g = np.asarray(golden).reshape(-1)
    a = np.asarray(actual).reshape(-1)
    norm_g = float(np.linalg.norm(g))
    norm_a = float(np.linalg.norm(a))
    if max(norm_g, norm_a) < 1e-12:
        return 0.0, 0.0
    if norm_g == 0.0 and norm_a == 0.0:
        return 0.0, 0.0
    if norm_g == 0.0 or norm_a == 0.0:
        return float("inf"), 1.0

    overlap = np.vdot(a, g)
    if overlap == 0:
        aligned = a
        normalized_overlap = 0.0
    else:
        aligned = a * overlap / abs(overlap)
        normalized_overlap = abs(overlap) / (norm_a * norm_g)

    rel = float(np.linalg.norm(aligned - g) / max(norm_g, np.finfo(float).eps))
    overlap_error = float(abs(1.0 - normalized_overlap))
    return rel, overlap_error
