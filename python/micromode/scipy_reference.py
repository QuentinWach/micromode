"""Readable SciPy reference implementation for the diagonal mode solver.

This module intentionally mirrors the Rust diagonal sparse path in plain
Python/SciPy. It is slower and narrower than the production backend, but it
keeps the numerical contract inspectable by users who want to audit the
finite-difference operator against SciPy/ARPACK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ETA0 = 376.730_313_666_853_5


SparseSolveResult = tuple[np.ndarray, list[np.ndarray], dict[str, object]]


@dataclass
class _ModeFields:
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray

    def components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.ex, self.ey, self.ez, self.hx, self.hy, self.hz

    def add_scaled(self, other: _ModeFields, scale: complex) -> None:
        for left, right in zip(self.components(), other.components(), strict=True):
            left += scale * right


def solve_diagonal_scipy_reference(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    num_modes: int,
    neff_guess: float,
    direction: str,
    derivative_scale: float,
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> SparseSolveResult:
    """Solve the diagonal sparse eigenproblem with SciPy/ARPACK.

    Supported scope is deliberately small: diagonal material tensors, no PML,
    and the same reduced ``[Ex, Ey]`` transverse eigenproblem used by the Rust
    production backend.
    """

    sparse, spla, scipy_linalg = _import_scipy()
    nx = len(dlf[0])
    ny = len(dlf[1])
    n = nx * ny
    if eps_tensor.shape != mu_tensor.shape or eps_tensor.shape[:2] != (3, 3) or eps_tensor.shape[-1] != n:
        raise ValueError("eps_tensor and mu_tensor must both have shape (3, 3, len(dlf[0]) * len(dlf[1]))")
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    derivatives = _create_derivative_matrices(
        sparse,
        shape=(nx, ny),
        dlf=dlf,
        dlb=dlb,
        dmin_pmc=dmin_pmc,
        scale=float(derivative_scale),
    )
    operators = _assemble_diagonal_operators(sparse, eps_tensor, mu_tensor, derivatives)
    eig_guess = complex(-(neff_guess * neff_guess), 0.0)
    values, vectors = _selected_eigenpairs(
        operators["mat"],
        num_modes=num_modes,
        sigma=eig_guess,
        krylov_dim=krylov_dim,
        initial_vector=initial_vector,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )
    residuals = np.asarray(
        [
            np.linalg.norm(operators["mat"] @ vectors[:, index] - values[index] * vectors[:, index])
            for index in range(len(values))
        ],
        dtype=float,
    )
    vector_norms = np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(float).eps)
    residuals = residuals / vector_norms

    modes: list[tuple[complex, np.ndarray, float]] = []
    for value, vector, residual in zip(values, vectors.T, residuals, strict=True):
        modes.append((complex(np.sqrt(-value + 0j)), np.asarray(vector, dtype=np.complex128), float(residual)))
    modes.sort(key=lambda item: item[0].real, reverse=True)

    inv_eps_zz = sparse.diags(1.0 / eps_tensor[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu_tensor[2, 2, :], format="csc")
    dxf, dxb, dyf, dyb = derivatives
    cell_areas = np.repeat(np.asarray(dlf[0], dtype=float), ny) * np.tile(np.asarray(dlf[1], dtype=float), nx)

    n_complex: list[complex] = []
    mode_fields: list[_ModeFields] = []
    sorted_residuals: list[float] = []
    for mode_n, vector, residual in modes:
        ex = vector[:n].copy()
        ey = vector[n:].copy()
        denom = complex(-mode_n.imag, mode_n.real)

        h_field = operators["qmat"] @ vector
        hx = np.asarray(h_field[:n] / denom, dtype=np.complex128)
        hy = np.asarray(h_field[n:] / denom, dtype=np.complex128)

        hz_source = dxf @ ey - dyf @ ex
        hz = np.asarray(inv_mu_zz @ hz_source, dtype=np.complex128)

        h_partial = np.asarray((operators["q_ep"] @ vector) / denom, dtype=np.complex128)
        ez_source = dxb @ h_partial[n:] - dyb @ h_partial[:n]
        ez = np.asarray(inv_eps_zz @ ez_source, dtype=np.complex128)

        h_scale = -1j / ETA0
        hx *= h_scale
        hy *= h_scale
        hz *= h_scale
        if direction == "-":
            hx *= -1.0
            hy *= -1.0
            ez *= -1.0

        n_complex.append(mode_n)
        sorted_residuals.append(residual)
        mode_fields.append(_ModeFields(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz))

    orthogonalization = _lorentz_orthogonalize_and_normalize(mode_fields, cell_areas)
    fields = [
        np.asarray([getattr(mode, component) for mode in mode_fields], dtype=np.complex128)
        for component in ("ex", "ey", "ez", "hx", "hy", "hz")
    ]
    return (
        np.asarray(n_complex, dtype=np.complex128),
        fields,
        {
            "backend": "scipy_arpack_reference",
            "operator_size": int(operators["mat"].shape[0]),
            "operator_nnz": int(operators["mat"].nnz),
            "residuals": np.asarray(sorted_residuals, dtype=float),
            "power_norms": orthogonalization["power_norms"],
            "lorentz_norms": orthogonalization["lorentz_norms"],
            "lorentz_orthogonality_error": orthogonalization["lorentz_orthogonality_error"],
        },
    )


def _import_scipy():
    try:
        import scipy.linalg as scipy_linalg
        import scipy.sparse as sparse
        import scipy.sparse.linalg as spla
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError("the SciPy reference backend requires `pip install micromode[scipy]`") from exc
    return sparse, spla, scipy_linalg


def _create_derivative_matrices(
    sparse,
    *,
    shape: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    dmin_pmc: tuple[bool, bool],
    scale: float,
):
    matrices = (
        _make_dxf(sparse, np.asarray(dlf[0], dtype=float), shape, dmin_pmc[0]),
        _make_dxb(sparse, np.asarray(dlb[0], dtype=float), shape, dmin_pmc[0]),
        _make_dyf(sparse, np.asarray(dlf[1], dtype=float), shape, dmin_pmc[1]),
        _make_dyb(sparse, np.asarray(dlb[1], dtype=float), shape, dmin_pmc[1]),
    )
    return tuple(matrix * complex(scale, 0.0) for matrix in matrices)


def _make_dxf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    nx, ny = shape
    if nx == 1:
        return sparse.csc_matrix((ny, ny), dtype=np.complex128)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[ix]
            diagonal = 0.0 if ix == 0 and not pmc else -value
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if ix + 1 < nx:
                rows.append(row)
                cols.append((ix + 1) * ny + iy)
                data.append(value)
    return sparse.csc_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128)


def _make_dxb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    nx, ny = shape
    if nx == 1:
        return sparse.csc_matrix((ny, ny), dtype=np.complex128)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[ix]
            diagonal = 2.0 * value if ix == 0 and pmc else (0.0 if ix == 0 else value)
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if ix > 0:
                rows.append(row)
                cols.append((ix - 1) * ny + iy)
                data.append(-value)
    return sparse.csc_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128)


def _make_dyf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    nx, ny = shape
    if ny == 1:
        return sparse.csc_matrix((nx, nx), dtype=np.complex128)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[iy]
            diagonal = 0.0 if iy == 0 and not pmc else -value
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if iy + 1 < ny:
                rows.append(row)
                cols.append(ix * ny + iy + 1)
                data.append(value)
    return sparse.csc_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128)


def _make_dyb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    nx, ny = shape
    if ny == 1:
        return sparse.csc_matrix((nx, nx), dtype=np.complex128)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[iy]
            diagonal = 2.0 * value if iy == 0 and pmc else (0.0 if iy == 0 else value)
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if iy > 0:
                rows.append(row)
                cols.append(ix * ny + iy - 1)
                data.append(-value)
    return sparse.csc_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128)


def _assemble_diagonal_operators(sparse, eps: np.ndarray, mu: np.ndarray, der_mats) -> dict[str, object]:
    n = eps.shape[-1]
    zero = sparse.csc_matrix((n, n), dtype=np.complex128)
    dxf, dxb, dyf, dyb = der_mats
    inv_eps_zz = sparse.diags(1.0 / eps[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu[2, 2, :], format="csc")

    p_mu = sparse.bmat(
        [[zero, sparse.diags(mu[1, 1, :], format="csc")], [-sparse.diags(mu[0, 0, :], format="csc"), zero]],
        format="csc",
    )
    p_partial = sparse.bmat(
        [
            [-(dxf @ inv_eps_zz @ dyb), dxf @ inv_eps_zz @ dxb],
            [-(dyf @ inv_eps_zz @ dyb), dyf @ inv_eps_zz @ dxb],
        ],
        format="csc",
    )
    q_ep = sparse.bmat(
        [[zero, sparse.diags(eps[1, 1, :], format="csc")], [-sparse.diags(eps[0, 0, :], format="csc"), zero]],
        format="csc",
    )
    q_partial = sparse.bmat(
        [
            [-(dxb @ inv_mu_zz @ dyf), dxb @ inv_mu_zz @ dxf],
            [-(dyb @ inv_mu_zz @ dyf), dyb @ inv_mu_zz @ dxf],
        ],
        format="csc",
    )
    qmat = q_ep + q_partial
    mat = p_mu @ qmat + p_partial @ q_ep
    return {"q_ep": q_ep, "qmat": qmat, "mat": mat}


def _selected_eigenpairs(
    mat,
    *,
    num_modes: int,
    sigma: complex,
    krylov_dim: int,
    initial_vector: np.ndarray | None,
    spla,
    scipy_linalg,
) -> tuple[np.ndarray, np.ndarray]:
    size = mat.shape[0]
    if num_modes >= size - 1:
        values, vectors = scipy_linalg.eig(mat.toarray())
        order = np.argsort(np.abs(values - sigma))[:num_modes]
        return values[order], vectors[:, order]

    ncv = min(size, max(int(krylov_dim), num_modes + 2))
    if ncv <= num_modes + 1:
        ncv = min(size, num_modes + 2)
    values, vectors = spla.eigs(
        mat,
        k=num_modes,
        sigma=sigma,
        which="LM",
        v0=None if initial_vector is None else np.asarray(initial_vector, dtype=np.complex128),
        ncv=ncv,
        tol=1e-10,
    )
    return np.asarray(values, dtype=np.complex128), np.asarray(vectors, dtype=np.complex128)


def _lorentz_orthogonalize_and_normalize(modes: list[_ModeFields], cell_areas: np.ndarray) -> dict[str, object]:
    for mode in modes:
        _normalize_to_unit_power(mode, cell_areas)

    for mode_index, mode in enumerate(modes):
        for previous in modes[:mode_index]:
            denom = _lorentz_overlap(previous, previous, cell_areas)
            if abs(denom) <= np.finfo(float).eps:
                continue
            coeff = _lorentz_overlap(previous, mode, cell_areas) / denom
            mode.add_scaled(previous, -coeff)
        _normalize_to_unit_power(mode, cell_areas)
        _apply_dominant_e_phase_convention(mode)

    power_norms = np.asarray([abs(_transverse_power(mode, cell_areas)) for mode in modes], dtype=float)
    lorentz_norms = np.asarray([_lorentz_overlap(mode, mode, cell_areas) for mode in modes], dtype=np.complex128)
    error = 0.0
    for left_index, left in enumerate(modes):
        for right_index, right in enumerate(modes):
            if left_index == right_index:
                continue
            denom = float(np.sqrt(abs(lorentz_norms[left_index]) * abs(lorentz_norms[right_index])))
            if denom <= np.finfo(float).eps:
                continue
            error = max(error, abs(_lorentz_overlap(left, right, cell_areas)) / denom)
    return {
        "power_norms": power_norms,
        "lorentz_norms": lorentz_norms,
        "lorentz_orthogonality_error": float(error),
    }


def _normalize_to_unit_power(mode: _ModeFields, cell_areas: np.ndarray) -> float:
    norm = abs(_transverse_power(mode, cell_areas))
    if norm <= np.finfo(float).eps:
        return 0.0
    scale = 1.0 / np.sqrt(norm)
    for component in mode.components():
        component *= scale
    return abs(_transverse_power(mode, cell_areas))


def _transverse_power(mode: _ModeFields, cell_areas: np.ndarray) -> complex:
    return complex(np.sum((mode.ex * np.conj(mode.hy) - mode.ey * np.conj(mode.hx)) * cell_areas))


def _lorentz_overlap(left: _ModeFields, right: _ModeFields, cell_areas: np.ndarray) -> complex:
    left_cross_right = np.sum((left.ex * right.hy - left.ey * right.hx) * cell_areas)
    right_cross_left = np.sum((right.ex * left.hy - right.ey * left.hx) * cell_areas)
    return complex(0.5 * (left_cross_right + right_cross_left))


def _apply_dominant_e_phase_convention(mode: _ModeFields) -> None:
    electric = np.concatenate((mode.ex, mode.ey, mode.ez))
    if electric.size == 0:
        return
    anchor = electric[int(np.argmax(np.abs(electric) ** 2))]
    if abs(anchor) <= np.finfo(float).eps:
        return
    phase = np.conj(anchor) / abs(anchor)
    for component in mode.components():
        component *= phase
