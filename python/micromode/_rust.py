from __future__ import annotations

import numpy as np

# Thin Python wrapper around the PyO3 extension. The public API works with
# NumPy arrays and dictionaries; the extension boundary only accepts plain
# Python lists of real/imag pairs, so the helpers below perform that conversion
# in one place.

C_0 = 2.997_924_58e14
EPSILON_0 = 8.854187812800384e-18


try:
    from ._core import (
        solve_diagonal_sparse_py as _solve_diagonal_sparse,
        solve_tensorial_sparse_py as _solve_tensorial_sparse,
    )
except Exception:  # pragma: no cover - exercised when extension is not built locally.
    _solve_diagonal_sparse = None
    _solve_tensorial_sparse = None


def solve_diagonal_sparse(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    num_modes: int,
    neff_guess: float,
    direction: str,
    derivative_scale: float | None = None,
    omega: float | None = None,
    num_pml: tuple[int, int] = (0, 0),
    pml_profile: dict[str, float | int] | None = None,
    dmin_pml: tuple[bool, bool] = (True, True),
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run the Rust sparse diagonal mode solver for a prepared 2D Yee grid."""
    # This is the normal path for diagonal material grids. The returned fields
    # are still flattened by mode/component; `raster.py` restores grid axes.
    if _solve_diagonal_sparse is None:
        raise RuntimeError("Rust extension is not built; sparse diagonal solver is unavailable")
    if eps_tensor.shape != mu_tensor.shape or eps_tensor.shape[:2] != (3, 3):
        raise ValueError("eps_tensor and mu_tensor must both have shape (3, 3, N)")
    nx = len(dlf[0])
    ny = len(dlf[1])
    expected_n = nx * ny
    if eps_tensor.shape[-1] != expected_n:
        raise ValueError("tensor length must match len(dlf[0]) * len(dlf[1])")

    pml_profile = _default_pml_profile(pml_profile)
    (
        n_pairs,
        field_pairs,
        residuals,
        power_norms,
        lorentz_norms,
        lorentz_orthogonality_error,
        backend,
        operator_size,
        operator_nnz,
    ) = _solve_diagonal_sparse(
        nx,
        ny,
        dlf[0].astype(float).tolist(),
        dlf[1].astype(float).tolist(),
        dlb[0].astype(float).tolist(),
        dlb[1].astype(float).tolist(),
        dmin_pmc[0],
        dmin_pmc[1],
        _tensor_payload(eps_tensor),
        _tensor_payload(mu_tensor),
        num_modes,
        neff_guess,
        direction,
        derivative_scale,
        int(num_pml[0]),
        int(num_pml[1]),
        float(pml_profile["sigma_max"]),
        float(pml_profile["kappa_min"]),
        float(pml_profile["kappa_max"]),
        int(pml_profile["order"]),
        bool(dmin_pml[0]),
        bool(dmin_pml[1]),
        omega,
        int(krylov_dim),
        None if initial_vector is None else _complex_vector_payload(initial_vector),
    )
    n_complex = _pairs_to_complex(n_pairs)
    fields = [_pairs_to_complex(component) for component in field_pairs]
    return n_complex, fields, _solver_info(
        residuals,
        power_norms,
        lorentz_norms,
        lorentz_orthogonality_error,
        backend,
        operator_size,
        operator_nnz,
    )


def solve_tensorial_sparse(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    num_modes: int,
    neff_guess: float,
    direction: str,
    derivative_scale: float | None = None,
    omega: float | None = None,
    num_pml: tuple[int, int] = (0, 0),
    pml_profile: dict[str, float | int] | None = None,
    dmin_pml: tuple[bool, bool] = (True, True),
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run the Rust sparse full-tensor mode solver for a prepared 2D Yee grid."""
    # Full tensor material grids, angled solves, and bent solves use this path
    # because coordinate transforms can introduce off-diagonal eps/mu terms.
    if _solve_tensorial_sparse is None:
        raise RuntimeError("Rust extension is not built; sparse tensorial solver is unavailable")
    if eps_tensor.shape != mu_tensor.shape or eps_tensor.shape[:2] != (3, 3):
        raise ValueError("eps_tensor and mu_tensor must both have shape (3, 3, N)")
    nx = len(dlf[0])
    ny = len(dlf[1])
    expected_n = nx * ny
    if eps_tensor.shape[-1] != expected_n:
        raise ValueError("tensor length must match len(dlf[0]) * len(dlf[1])")

    pml_profile = _default_pml_profile(pml_profile)
    (
        n_pairs,
        field_pairs,
        residuals,
        power_norms,
        lorentz_norms,
        lorentz_orthogonality_error,
        backend,
        operator_size,
        operator_nnz,
    ) = _solve_tensorial_sparse(
        nx,
        ny,
        dlf[0].astype(float).tolist(),
        dlf[1].astype(float).tolist(),
        dlb[0].astype(float).tolist(),
        dlb[1].astype(float).tolist(),
        dmin_pmc[0],
        dmin_pmc[1],
        _tensor_payload(eps_tensor),
        _tensor_payload(mu_tensor),
        num_modes,
        neff_guess,
        direction,
        derivative_scale,
        int(num_pml[0]),
        int(num_pml[1]),
        float(pml_profile["sigma_max"]),
        float(pml_profile["kappa_min"]),
        float(pml_profile["kappa_max"]),
        int(pml_profile["order"]),
        bool(dmin_pml[0]),
        bool(dmin_pml[1]),
        omega,
        int(krylov_dim),
        None if initial_vector is None else _complex_vector_payload(initial_vector),
    )
    n_complex = _pairs_to_complex(n_pairs)
    fields = [_pairs_to_complex(component) for component in field_pairs]
    return n_complex, fields, _solver_info(
        residuals,
        power_norms,
        lorentz_norms,
        lorentz_orthogonality_error,
        backend,
        operator_size,
        operator_nnz,
    )


def _default_pml_profile(profile: dict[str, float | int] | None) -> dict[str, float | int]:
    defaults: dict[str, float | int] = {
        "sigma_max": 2.0,
        "kappa_min": 1.0,
        "kappa_max": 3.0,
        "order": 3,
    }
    if profile is not None:
        defaults.update(profile)
    return defaults


def _solver_info(
    residuals,
    power_norms,
    lorentz_norms,
    lorentz_orthogonality_error,
    backend,
    operator_size,
    operator_nnz,
) -> dict[str, object]:
    # Keep raw backend diagnostics close to the extension boundary. Higher-level
    # context such as grid shape, PML, and normalization labels is added in
    # `raster.py`.
    return {
        "backend": str(backend),
        "operator_size": int(operator_size),
        "operator_nnz": int(operator_nnz),
        "residuals": np.asarray(residuals, dtype=float),
        "power_norms": np.asarray(power_norms, dtype=float),
        "lorentz_norms": _pairs_to_complex(lorentz_norms),
        "lorentz_orthogonality_error": float(lorentz_orthogonality_error),
    }


def _tensor_payload(tensor: np.ndarray) -> list[list[tuple[float, float]]]:
    # PyO3 can accept nested Python lists reliably across build targets. The
    # tensor is flattened as nine component vectors: xx, xy, xz, yx, ... zz.
    tensor = np.asarray(tensor)
    return [
        [(complex(value).real, complex(value).imag) for value in tensor[row, col, :]]
        for row in range(3)
        for col in range(3)
    ]


def _complex_vector_payload(vector: np.ndarray) -> list[tuple[float, float]]:
    # Initial vectors are complex NumPy arrays in Python but real/imag tuples at
    # the Rust boundary.
    return [(complex(value).real, complex(value).imag) for value in np.asarray(vector).reshape(-1)]


def _pairs_to_complex(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[..., 0] + 1j * array[..., 1]
