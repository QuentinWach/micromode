"""Readable SciPy reference implementation for the diagonal mode solver.

This module intentionally mirrors the Rust sparse mode-solver paths in plain
Python/SciPy. It is slower and less portable than the production backend, but it
keeps the numerical contract inspectable by users who want to audit the
finite-difference operators against SciPy/ARPACK.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

ETA0 = 376.730_313_666_853_5
MU0 = 1.256_637_062_12e-12
C0 = 2.997_924_58e14
EPSILON0 = 1.0 / (MU0 * C0 * C0)


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
    omega: float | None = None,
    num_pml: tuple[int, int] = (0, 0),
    pml_profile: dict[str, float | int] | None = None,
    dmin_pml: tuple[bool, bool] = (True, True),
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> SparseSolveResult:
    """Solve the diagonal sparse eigenproblem with SciPy/ARPACK.

    This is the same reduced ``[Ex, Ey]`` transverse eigenproblem used by the
    Rust production backend for diagonal material tensors.
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
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        shape=(nx, ny),
        dlf=dlf,
        dlb=dlb,
        omega=omega,
        num_pml=num_pml,
        pml_profile=pml_profile,
        dmin_pml=dmin_pml,
        dmin_pmc=dmin_pmc,
        scale=float(derivative_scale),
    )
    operators = _assemble_diagonal_operators(sparse, eps_tensor, mu_tensor, derivatives)
    operator = operators["mat"]
    eig_guess = complex(-(neff_guess * neff_guess), 0.0)
    operator, arpack_initial_vector, arpack_guess = _real_arpack_problem_if_close(
        operator, initial_vector, eig_guess
    )
    values, vectors = _selected_eigenpairs(
        operator,
        num_modes=num_modes,
        sigma=arpack_guess,
        krylov_dim=krylov_dim,
        initial_vector=arpack_initial_vector,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )
    residuals = np.asarray(
        [
            np.linalg.norm(operator @ vectors[:, index] - values[index] * vectors[:, index])
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
            "operator_size": int(operator.shape[0]),
            "operator_nnz": int(operator.nnz),
            "residuals": np.asarray(sorted_residuals, dtype=float),
            "power_norms": orthogonalization["power_norms"],
            "lorentz_norms": orthogonalization["lorentz_norms"],
            "lorentz_orthogonality_error": orthogonalization["lorentz_orthogonality_error"],
        },
    )


def solve_tensorial_scipy_reference(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    num_modes: int,
    neff_guess: float,
    direction: str,
    derivative_scale: float,
    omega: float | None = None,
    num_pml: tuple[int, int] = (0, 0),
    pml_profile: dict[str, float | int] | None = None,
    dmin_pml: tuple[bool, bool] = (True, True),
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> SparseSolveResult:
    """Solve the first-order tensorial sparse eigenproblem with SciPy/ARPACK."""

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
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        shape=(nx, ny),
        dlf=dlf,
        dlb=dlb,
        omega=omega,
        num_pml=num_pml,
        pml_profile=pml_profile,
        dmin_pml=dmin_pml,
        dmin_pmc=dmin_pmc,
        scale=float(derivative_scale),
    )
    operator = _assemble_tensorial_operator(sparse, eps_tensor, mu_tensor, derivatives)
    values, vectors = _selected_eigenpairs(
        operator,
        num_modes=num_modes,
        sigma=complex(neff_guess, 0.0),
        krylov_dim=krylov_dim,
        initial_vector=initial_vector,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )
    residuals = np.asarray(
        [
            np.linalg.norm(operator @ vectors[:, index] - values[index] * vectors[:, index])
            for index in range(len(values))
        ],
        dtype=float,
    )
    residuals = residuals / np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(float).eps)

    modes = [
        (complex(value), np.asarray(vector, dtype=np.complex128), float(residual))
        for value, vector, residual in zip(values, vectors.T, residuals, strict=True)
    ]
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
        ey = vector[n : 2 * n].copy()
        hx = vector[2 * n : 3 * n].copy()
        hy = vector[3 * n : 4 * n].copy()

        hz_source = dxf @ ey - dyf @ ex - mu_tensor[2, 0, :] * hx - mu_tensor[2, 1, :] * hy
        hz = np.asarray(inv_mu_zz @ hz_source, dtype=np.complex128)

        ez_source = dxb @ hy - dyb @ hx - eps_tensor[2, 0, :] * ex - eps_tensor[2, 1, :] * ey
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
            "operator_size": int(operator.shape[0]),
            "operator_nnz": int(operator.nnz),
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
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    shape: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    omega: float | None,
    num_pml: tuple[int, int],
    pml_profile: dict[str, float | int] | None,
    dmin_pml: tuple[bool, bool],
    dmin_pmc: tuple[bool, bool],
    scale: float,
):
    matrices = (
        _make_dxf(sparse, np.asarray(dlf[0], dtype=float), shape, dmin_pmc[0]),
        _make_dxb(sparse, np.asarray(dlb[0], dtype=float), shape, dmin_pmc[0]),
        _make_dyf(sparse, np.asarray(dlf[1], dtype=float), shape, dmin_pmc[1]),
        _make_dyb(sparse, np.asarray(dlb[1], dtype=float), shape, dmin_pmc[1]),
    )
    if num_pml[0] > 0 or num_pml[1] > 0:
        if omega is None:
            raise ValueError("omega is required when num_pml is nonzero")
        pml_values = _create_s_diagonal_values(
            shape=shape,
            num_pml=num_pml,
            dlf=dlf,
            dlb=dlb,
            eps_tensor=eps_tensor,
            mu_tensor=mu_tensor,
            dmin_pml=dmin_pml,
            omega=float(omega),
            profile=pml_profile,
        )
        matrices = tuple(
            sparse.diags(values, format="csc") @ matrix for values, matrix in zip(pml_values, matrices, strict=True)
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
    return {"q_ep": _canonical_sparse(q_ep), "qmat": _canonical_sparse(qmat), "mat": _canonical_sparse(mat)}


def _assemble_tensorial_operator(sparse, eps: np.ndarray, mu: np.ndarray, der_mats):
    dxf, dxb, dyf, dyb = der_mats
    inv_eps_22 = sparse.diags(1.0 / eps[2, 2, :], format="csc")
    inv_mu_22 = sparse.diags(1.0 / mu[2, 2, :], format="csc")

    def diag(values):
        return sparse.diags(values, format="csc")

    eps_20_over_22 = eps[2, 0, :] / eps[2, 2, :]
    eps_21_over_22 = eps[2, 1, :] / eps[2, 2, :]
    eps_02_over_22 = eps[0, 2, :] / eps[2, 2, :]
    eps_12_over_22 = eps[1, 2, :] / eps[2, 2, :]
    mu_20_over_22 = mu[2, 0, :] / mu[2, 2, :]
    mu_21_over_22 = mu[2, 1, :] / mu[2, 2, :]
    mu_02_over_22 = mu[0, 2, :] / mu[2, 2, :]
    mu_12_over_22 = mu[1, 2, :] / mu[2, 2, :]

    mu_10_s = mu[1, 0, :] - mu[1, 2, :] * mu_20_over_22
    mu_11_s = mu[1, 1, :] - mu[1, 2, :] * mu_21_over_22
    mu_00_s = mu[0, 0, :] - mu[0, 2, :] * mu_20_over_22
    mu_01_s = mu[0, 1, :] - mu[0, 2, :] * mu_21_over_22
    eps_10_s = eps[1, 0, :] - eps[1, 2, :] * eps_20_over_22
    eps_11_s = eps[1, 1, :] - eps[1, 2, :] * eps_21_over_22
    eps_00_s = eps[0, 0, :] - eps[0, 2, :] * eps_20_over_22
    eps_01_s = eps[0, 1, :] - eps[0, 2, :] * eps_21_over_22

    axax = -(dxf @ diag(eps_20_over_22)) - diag(mu_12_over_22) @ dyf
    axay = -(dxf @ diag(eps_21_over_22)) + diag(mu_12_over_22) @ dxf
    axbx = -(dxf @ inv_eps_22 @ dyb) + diag(mu_10_s)
    axby = dxf @ inv_eps_22 @ dxb + diag(mu_11_s)

    ayax = -(dyf @ diag(eps_20_over_22)) + diag(mu_02_over_22) @ dyf
    ayay = -(dyf @ diag(eps_21_over_22)) - diag(mu_02_over_22) @ dxf
    aybx = -(dyf @ inv_eps_22 @ dyb) - diag(mu_00_s)
    ayby = dyf @ inv_eps_22 @ dxb - diag(mu_01_s)

    bxax = -(dxb @ inv_mu_22 @ dyf) + diag(eps_10_s)
    bxay = dxb @ inv_mu_22 @ dxf + diag(eps_11_s)
    bxbx = -(dxb @ diag(mu_20_over_22)) - diag(eps_12_over_22) @ dyb
    bxby = -(dxb @ diag(mu_21_over_22)) + diag(eps_12_over_22) @ dxb

    byax = -(dyb @ inv_mu_22 @ dyf) - diag(eps_00_s)
    byay = dyb @ inv_mu_22 @ dxf - diag(eps_01_s)
    bybx = -(dyb @ diag(mu_20_over_22)) + diag(eps_02_over_22) @ dyb
    byby = -(dyb @ diag(mu_21_over_22)) - diag(eps_02_over_22) @ dxb

    return _canonical_sparse(
        -1j
        * sparse.bmat(
            [
                [axax, axay, axbx, axby],
                [ayax, ayay, aybx, ayby],
                [bxax, bxay, bxbx, bxby],
                [byax, byay, bybx, byby],
            ],
            format="csc",
        )
    )


def _canonical_sparse(matrix):
    matrix = matrix.tocsc(copy=True)
    matrix.eliminate_zeros()
    return matrix


def _real_arpack_problem_if_close(matrix, initial_vector: np.ndarray | None, guess: complex):
    if matrix.nnz == 0:
        return matrix, initial_vector, guess
    matrix_imag = matrix.data.imag
    matrix_scale = max(float(np.max(np.abs(matrix.data))), 1.0)
    guess_is_real = abs(guess.imag) <= 1e-14 * max(abs(guess), 1.0)
    if np.max(np.abs(matrix_imag)) <= 1e-14 * matrix_scale and guess_is_real:
        real_vector = None if initial_vector is None else np.asarray(initial_vector.real, dtype=float)
        return matrix.real.astype(float), real_vector, float(guess.real)
    return matrix, initial_vector, guess


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning, module="scipy")
        values, vectors = spla.eigs(
            mat,
            k=num_modes,
            sigma=sigma,
            which="LM",
            v0=None if initial_vector is None else np.asarray(initial_vector),
            ncv=ncv,
            tol=1e-10,
        )
    return np.asarray(values, dtype=np.complex128), np.asarray(vectors, dtype=np.complex128)


def _create_s_diagonal_values(
    *,
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dmin_pml: tuple[bool, bool],
    omega: float,
    profile: dict[str, float | int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = shape
    avg_speed = _average_relative_speed(shape, num_pml, eps_tensor, mu_tensor)
    sx_f = _create_sfactor(
        "f", omega, np.asarray(dlf[0], dtype=float), nx, num_pml[0], dmin_pml[0], avg_speed[:2], profile
    )
    sx_b = _create_sfactor(
        "b", omega, np.asarray(dlb[0], dtype=float), nx, num_pml[0], dmin_pml[0], avg_speed[:2], profile
    )
    sy_f = _create_sfactor(
        "f", omega, np.asarray(dlf[1], dtype=float), ny, num_pml[1], dmin_pml[1], avg_speed[2:], profile
    )
    sy_b = _create_sfactor(
        "b", omega, np.asarray(dlb[1], dtype=float), ny, num_pml[1], dmin_pml[1], avg_speed[2:], profile
    )

    sx_f_vec = np.empty(nx * ny, dtype=np.complex128)
    sx_b_vec = np.empty(nx * ny, dtype=np.complex128)
    sy_f_vec = np.empty(nx * ny, dtype=np.complex128)
    sy_b_vec = np.empty(nx * ny, dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            index = ix * ny + iy
            sx_f_vec[index] = 1.0 / sx_f[ix]
            sx_b_vec[index] = 1.0 / sx_b[ix]
            sy_f_vec[index] = 1.0 / sy_f[iy]
            sy_b_vec[index] = 1.0 / sy_b[iy]
    return sx_f_vec, sx_b_vec, sy_f_vec, sy_b_vec


def _average_relative_speed(
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
) -> np.ndarray:
    eps_avg = _pml_average_all_sides(shape, num_pml, eps_tensor)
    mu_avg = _pml_average_all_sides(shape, num_pml, mu_tensor)
    return 1.0 / np.sqrt(eps_avg * mu_avg)


def _pml_average_all_sides(shape: tuple[int, int], num_pml: tuple[int, int], tensor: np.ndarray) -> np.ndarray:
    nx, ny = shape
    regions: list[list[complex]] = [[], [], [], []]
    for comp in range(3):
        for ix in range(nx):
            for iy in range(ny):
                value = complex(tensor[comp, comp, ix * ny + iy])
                if ix < num_pml[0]:
                    regions[0].append(value)
                if ix >= max(nx - num_pml[0], 0) + 1:
                    regions[1].append(value)
                if iy < num_pml[1]:
                    regions[2].append(value)
                if iy >= max(ny - num_pml[1], 0) + 1:
                    regions[3].append(value)
    out = np.ones(4, dtype=np.complex128)
    for index, values in enumerate(regions):
        if values:
            out[index] = sum(values) / len(values)
    return out


def _create_sfactor(
    direction: str,
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    if n_pml == 0:
        return np.ones(n, dtype=np.complex128)
    if direction == "f":
        return _create_sfactor_f(omega, dls, n, n_pml, dmin_pml, avg_speed, profile)
    if direction == "b":
        return _create_sfactor_b(omega, dls, n, n_pml, dmin_pml, avg_speed, profile)
    raise ValueError(f"direction value {direction} not recognized")


def _create_sfactor_f(
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    sfactor = np.ones(n, dtype=np.complex128)
    for i in range(n):
        if i < n_pml and dmin_pml:
            sfactor[i] = _s_value(dls[0], (n_pml - i - 0.5) / n_pml, omega, avg_speed[0], profile)
        elif i >= n - n_pml:
            sfactor[i] = _s_value(dls[-1], (i - (n - n_pml) + 0.5) / n_pml, omega, avg_speed[1], profile)
    return sfactor


def _create_sfactor_b(
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    sfactor = np.ones(n, dtype=np.complex128)
    for i in range(n):
        if i < n_pml and dmin_pml:
            sfactor[i] = _s_value(dls[0], (n_pml - i) / n_pml, omega, avg_speed[0], profile)
        elif i > n - n_pml:
            sfactor[i] = _s_value(dls[-1], (i - (n - n_pml)) / n_pml, omega, avg_speed[1], profile)
    return sfactor


def _s_value(
    dl: float,
    step: float,
    omega: float,
    avg_speed: complex,
    profile: dict[str, float | int] | None,
) -> complex:
    values = {
        "sigma_max": 2.0,
        "kappa_min": 1.0,
        "kappa_max": 3.0,
        "order": 3,
    }
    if profile is not None:
        values.update(profile)
    step_power = step ** int(values["order"])
    kappa = float(values["kappa_min"]) + (float(values["kappa_max"]) - float(values["kappa_min"])) * step_power
    sigma = avg_speed * (float(values["sigma_max"]) / (ETA0 * dl) * step_power)
    return complex(kappa, 0.0) + 1j * sigma / (omega * EPSILON0)


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
