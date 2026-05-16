//! Eigenvalue selection and shift-invert backends.
//!
//! The solver usually needs a handful of modes near a requested effective
//! index, not the whole spectrum. Shift-invert changes that local search into a
//! dominant-eigenvalue problem for `(A - sigma I)^-1`, which sparse Krylov
//! methods can solve efficiently on realistic grids.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;

use crate::sparse_matrix::SparseMatrix;

#[derive(Clone, Debug)]
pub struct Eigenpair {
    pub value: Complex64,
    pub vector: Vec<Complex64>,
    pub residual: f64,
    pub backend: &'static str,
}

#[derive(Clone, Debug)]
pub struct ShiftInvertOptions {
    pub krylov_dim: usize,
    pub tolerance: f64,
}

impl Default for ShiftInvertOptions {
    fn default() -> Self {
        Self {
            krylov_dim: 32,
            tolerance: 1e-10,
        }
    }
}

pub fn selected_sparse_shift_invert_eigenpairs(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
) -> Result<Vec<Eigenpair>, String> {
    // Prefer packaged sparse backends first. If an optional backend cannot
    // handle this matrix, fall through to the next available implementation.
    #[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
    {
        if let Ok(pairs) = selected_sparse_shift_invert_umfpack_eigenpairs(
            mat,
            num_modes,
            guess_value,
            initial_vector,
            options.clone(),
        ) {
            return Ok(pairs);
        }
    }

    #[cfg(feature = "arpack-backend")]
    {
        if let Ok(pairs) = selected_sparse_shift_invert_arpack_eigenpairs(
            mat,
            num_modes,
            guess_value,
            initial_vector,
            options.clone(),
        ) {
            return Ok(pairs);
        }
    }

    if mat.rows != mat.cols {
        return Err("eigenvalue matrix must be square".to_string());
    }
    if num_modes == 0 {
        return Err("num_modes must be positive".to_string());
    }
    let n = mat.rows;
    let krylov_dim = options.krylov_dim.min(n).max(num_modes + 2);
    if krylov_dim < num_modes {
        return Err("krylov_dim must be at least num_modes".to_string());
    }

    // Shift-invert solves `(A - sigma I) y = x` inside Arnoldi. Ritz values
    // theta of the inverse-shifted operator map back to lambda = sigma + 1/theta.
    let shifted = mat.shifted_diagonal(guess_value);
    let factorization = SparseLu::factor(&shifted)?;
    let start = initial_vector
        .map(|values| {
            if values.len() != n {
                return Err("initial vector length does not match matrix size".to_string());
            }
            Ok(values.to_vec())
        })
        .unwrap_or_else(|| Ok(default_initial_vector(n)))?;
    let mut q_vectors = vec![normalize_complex_vector(start)];
    let mut hessenberg = DMatrix::<Complex64>::zeros(krylov_dim + 1, krylov_dim);
    let mut actual_dim = 0usize;

    for col in 0..krylov_dim {
        // Native Arnoldi fallback with one reorthogonalization pass. The ARPACK
        // path is preferred for production, but this keeps the crate testable
        // when external sparse libraries are unavailable.
        let mut work = factorization.solve(&q_vectors[col])?;
        for row in 0..=col {
            let projection = complex_dot(&q_vectors[row], &work);
            hessenberg[(row, col)] = projection;
            axpy(&mut work, &q_vectors[row], -projection);
        }
        for row in 0..=col {
            let projection = complex_dot(&q_vectors[row], &work);
            hessenberg[(row, col)] += projection;
            axpy(&mut work, &q_vectors[row], -projection);
        }
        let norm = vector_norm(&work);
        actual_dim = col + 1;
        hessenberg[(col + 1, col)] = Complex64::new(norm, 0.0);
        if norm <= options.tolerance || col + 1 == krylov_dim {
            break;
        }
        scale_vector(&mut work, Complex64::new(1.0 / norm, 0.0));
        q_vectors.push(work);
    }

    // Solve the small projected problem, then lift each Ritz vector back into
    // the full grid basis using the stored Arnoldi vectors.
    let h_square = hessenberg
        .view((0, 0), (actual_dim, actual_dim))
        .into_owned();
    let theta_values = h_square
        .clone()
        .schur()
        .eigenvalues()
        .ok_or_else(|| "failed to compute Hessenberg eigenvalues".to_string())?;
    let mut candidates = Vec::new();
    for theta in theta_values.iter().copied() {
        if theta.norm() <= options.tolerance {
            continue;
        }
        let lambda = guess_value + Complex64::new(1.0, 0.0) / theta;
        let coeffs = null_vector_for_eigenvalue(&h_square, theta)?;
        let vector = combine_ritz_vector(&q_vectors, &coeffs);
        let residual = sparse_residual_norm(mat, &vector, lambda);
        candidates.push((lambda, vector, residual));
    }
    // Sort by closeness to the requested shift. Residual is the tie-breaker
    // because it is the direct measure of eigenpair quality.
    candidates.sort_by(|(left, _, left_res), (right, _, right_res)| {
        let left_distance = (*left - guess_value).norm();
        let right_distance = (*right - guess_value).norm();
        left_distance
            .partial_cmp(&right_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left_res
                    .partial_cmp(right_res)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    candidates.truncate(num_modes);
    Ok(candidates
        .into_iter()
        .map(|(value, vector, residual)| Eigenpair {
            value,
            vector,
            residual,
            backend: "native_shift_invert",
        })
        .collect())
}

#[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
pub fn selected_sparse_shift_invert_umfpack_eigenpairs(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
) -> Result<Vec<Eigenpair>, String> {
    // UMFPACK provides robust sparse LU factors, while ARPACK drives the Krylov
    // iteration. This is the current production sparse path when SuiteSparse is
    // available at build time.
    if mat.rows != mat.cols {
        return Err("eigenvalue matrix must be square".to_string());
    }
    if num_modes == 0 {
        return Err("num_modes must be positive".to_string());
    }
    let n = mat.rows;
    let ncv = options.krylov_dim.min(n).max(num_modes + 2);
    if ncv <= num_modes + 1 || ncv > n {
        return Err(
            "krylov_dim must satisfy num_modes + 2 <= krylov_dim <= matrix size".to_string(),
        );
    }

    let shifted = mat.shifted_diagonal(guess_value);
    let mut factorization = UmfpackLu::factor(&shifted)?;
    let maxiter = (10 * n).max(300).min(i32::MAX as usize);
    let (theta_values, vectors) = arpack_shift_invert_vectors(
        n,
        num_modes,
        ncv,
        maxiter,
        options.tolerance,
        initial_vector,
        |input| factorization.solve(input),
    )?;
    let mut candidates = Vec::with_capacity(theta_values.len());
    for mode_index in 0..theta_values.len() {
        let theta = theta_values[mode_index];
        if theta.norm() <= options.tolerance {
            continue;
        }
        let lambda = guess_value + Complex64::new(1.0, 0.0) / theta;
        let vector = normalize_complex_vector(vectors[mode_index].clone());
        let residual = sparse_residual_norm(mat, &vector, lambda);
        candidates.push((lambda, vector, residual));
    }
    candidates.sort_by(|(left, _, left_res), (right, _, right_res)| {
        let left_distance = (*left - guess_value).norm();
        let right_distance = (*right - guess_value).norm();
        left_distance
            .partial_cmp(&right_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left_res
                    .partial_cmp(right_res)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    candidates.truncate(num_modes);
    if candidates.len() != num_modes {
        return Err(format!(
            "UMFPACK/ARPACK returned {} usable eigenpairs, expected {}",
            candidates.len(),
            num_modes
        ));
    }
    Ok(candidates
        .into_iter()
        .map(|(value, vector, residual)| Eigenpair {
            value,
            vector,
            residual,
            backend: "umfpack_arpack",
        })
        .collect())
}

#[cfg(feature = "arpack-backend")]
pub fn selected_sparse_shift_invert_arpack_eigenpairs(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
) -> Result<Vec<Eigenpair>, String> {
    // ARPACK can also run with the small native LU factorization. It is less
    // robust than UMFPACK but avoids requiring SuiteSparse.
    if mat.rows != mat.cols {
        return Err("eigenvalue matrix must be square".to_string());
    }
    if num_modes == 0 {
        return Err("num_modes must be positive".to_string());
    }
    let n = mat.rows;
    let ncv = options.krylov_dim.min(n).max(num_modes + 2);
    if ncv <= num_modes + 1 || ncv > n {
        return Err(
            "krylov_dim must satisfy num_modes + 2 <= krylov_dim <= matrix size".to_string(),
        );
    }

    let shifted = mat.shifted_diagonal(guess_value);
    let factorization = SparseLu::factor(&shifted)?;
    let maxiter = (10 * n).max(300).min(i32::MAX as usize);
    let (theta_values, vectors) = arpack_shift_invert_vectors(
        n,
        num_modes,
        ncv,
        maxiter,
        options.tolerance,
        initial_vector,
        |input| factorization.solve(input),
    )?;
    let mut candidates = Vec::with_capacity(theta_values.len());
    for mode_index in 0..theta_values.len() {
        let theta = theta_values[mode_index];
        if theta.norm() <= options.tolerance {
            continue;
        }
        let lambda = guess_value + Complex64::new(1.0, 0.0) / theta;
        let vector = vectors[mode_index].clone();
        let vector = normalize_complex_vector(vector);
        let residual = sparse_residual_norm(mat, &vector, lambda);
        candidates.push((lambda, vector, residual));
    }
    candidates.sort_by(|(left, _, left_res), (right, _, right_res)| {
        let left_distance = (*left - guess_value).norm();
        let right_distance = (*right - guess_value).norm();
        left_distance
            .partial_cmp(&right_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left_res
                    .partial_cmp(right_res)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    candidates.truncate(num_modes);
    if candidates.len() != num_modes {
        return Err(format!(
            "ARPACK returned {} usable eigenpairs, expected {}",
            candidates.len(),
            num_modes
        ));
    }
    Ok(candidates
        .into_iter()
        .map(|(value, vector, residual)| Eigenpair {
            value,
            vector,
            residual,
            backend: "arpack",
        })
        .collect())
}

#[cfg(feature = "arpack-backend")]
static ARPACK_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "arpack-backend")]
fn arpack_shift_invert_vectors<F>(
    n: usize,
    num_modes: usize,
    ncv: usize,
    maxiter: usize,
    tolerance: f64,
    initial_vector: Option<&[Complex64]>,
    mut op_solve: F,
) -> Result<(Vec<Complex64>, Vec<Vec<Complex64>>), String>
where
    F: FnMut(&[Complex64]) -> Result<Vec<Complex64>, String>,
{
    // ARPACK uses reverse communication. `znaupd` repeatedly asks us to compute
    // y = OP*x by pointing into work arrays; `op_solve` supplies that product.
    // The mutex serializes access to ARPACK's Fortran state.
    use std::os::raw::c_int;

    use arpack_sys::{__BindgenComplex, znaupd_c, zneupd_c};

    let n_i32 = c_int::try_from(n).map_err(|_| "matrix size exceeds ARPACK c_int".to_string())?;
    let nev_i32 =
        c_int::try_from(num_modes).map_err(|_| "num_modes exceeds ARPACK c_int".to_string())?;
    let ncv_i32 =
        c_int::try_from(ncv).map_err(|_| "krylov_dim exceeds ARPACK c_int".to_string())?;
    let maxiter_i32 =
        c_int::try_from(maxiter).map_err(|_| "maxiter exceeds ARPACK c_int".to_string())?;
    let workd_len = n
        .checked_mul(3)
        .ok_or_else(|| "ARPACK workd size overflow".to_string())?;
    let v_len = n
        .checked_mul(ncv)
        .ok_or_else(|| "ARPACK basis size overflow".to_string())?;
    let lworkl = 3usize
        .checked_mul(ncv)
        .and_then(|value| value.checked_mul(ncv))
        .and_then(|value| value.checked_add(6 * ncv))
        .ok_or_else(|| "ARPACK workl size overflow".to_string())?;
    let lworkl_i32 =
        c_int::try_from(lworkl).map_err(|_| "ARPACK workl exceeds c_int".to_string())?;

    let zero = __BindgenComplex { re: 0.0, im: 0.0 };
    let mut ido: c_int = 0;
    let mut resid = vec![zero; n];
    let mut info: c_int = 0;
    if let Some(values) = initial_vector {
        if values.len() != n {
            return Err("initial vector length does not match matrix size".to_string());
        }
        write_raw_complex(&mut resid, values);
        info = 1;
    }
    let mut v = vec![zero; v_len];
    let mut iparam = [0 as c_int; 11];
    iparam[0] = 1;
    iparam[2] = maxiter_i32;
    iparam[3] = 1;
    iparam[6] = 1;
    let mut ipntr = [0 as c_int; 14];
    let mut workd = vec![zero; workd_len];
    let mut workl = vec![zero; lworkl];
    let mut rwork = vec![0.0; ncv];
    let bmat = b"I\0";
    let which = b"LM\0";

    let _guard = ARPACK_MUTEX
        .lock()
        .map_err(|_| "failed to lock ARPACK mutex".to_string())?;

    loop {
        unsafe {
            znaupd_c(
                &mut ido,
                bmat.as_ptr().cast(),
                n_i32,
                which.as_ptr().cast(),
                nev_i32,
                tolerance,
                resid.as_mut_ptr(),
                ncv_i32,
                v.as_mut_ptr(),
                n_i32,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl_i32,
                rwork.as_mut_ptr(),
                &mut info,
            );
        }

        match ido {
            -1 | 1 => {
                // `ipntr` contains one-based offsets into ARPACK's workspace.
                // Convert and bounds-check them before constructing Rust slices.
                let x_offset = usize::try_from(ipntr[0] - 1)
                    .map_err(|_| "ARPACK returned invalid input pointer".to_string())?;
                let y_offset = usize::try_from(ipntr[1] - 1)
                    .map_err(|_| "ARPACK returned invalid output pointer".to_string())?;
                if x_offset + n > workd.len() || y_offset + n > workd.len() {
                    return Err("ARPACK work pointer out of bounds".to_string());
                }
                let input = raw_complex_to_vec(&workd[x_offset..x_offset + n]);
                let output = op_solve(&input)?;
                if output.len() != n {
                    return Err("ARPACK operator returned vector with wrong length".to_string());
                }
                write_raw_complex(&mut workd[y_offset..y_offset + n], &output);
            }
            99 => break,
            other => return Err(format!("ARPACK returned unsupported ido={other}")),
        }
    }

    if !matches!(info, 0 | 1 | 2) {
        return Err(format!("ARPACK znaupd failed with info={info}"));
    }

    // Extract the converged Ritz values and vectors from ARPACK.
    let mut select = vec![0 as c_int; ncv];
    let mut theta = vec![zero; num_modes + 1];
    let mut z = vec![zero; n * num_modes];
    let mut workev = vec![zero; 2 * ncv];
    let mut info_eupd: c_int = 0;
    unsafe {
        zneupd_c(
            1,
            b"A\0".as_ptr().cast(),
            select.as_mut_ptr(),
            theta.as_mut_ptr(),
            z.as_mut_ptr(),
            n_i32,
            zero,
            workev.as_mut_ptr(),
            bmat.as_ptr().cast(),
            n_i32,
            which.as_ptr().cast(),
            nev_i32,
            tolerance,
            resid.as_mut_ptr(),
            ncv_i32,
            v.as_mut_ptr(),
            n_i32,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr(),
            workl.as_mut_ptr(),
            lworkl_i32,
            rwork.as_mut_ptr(),
            &mut info_eupd,
        );
    }
    if info_eupd != 0 {
        return Err(format!("ARPACK zneupd failed with info={info_eupd}"));
    }

    let values = raw_complex_to_vec(&theta[..num_modes]);
    let vectors = (0..num_modes)
        .map(|mode_index| {
            (0..n)
                .map(|row| raw_to_complex(z[mode_index * n + row]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok((values, vectors))
}

#[cfg(feature = "arpack-backend")]
fn raw_complex_to_vec(values: &[arpack_sys::__BindgenComplex<f64>]) -> Vec<Complex64> {
    values.iter().copied().map(raw_to_complex).collect()
}

#[cfg(feature = "arpack-backend")]
fn raw_to_complex(value: arpack_sys::__BindgenComplex<f64>) -> Complex64 {
    Complex64::new(value.re, value.im)
}

#[cfg(feature = "arpack-backend")]
fn write_raw_complex(target: &mut [arpack_sys::__BindgenComplex<f64>], source: &[Complex64]) {
    for (target, value) in target.iter_mut().zip(source) {
        target.re = value.re;
        target.im = value.im;
    }
}

#[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
struct UmfpackLu {
    n: usize,
    solver: russell_sparse::prelude::ComplexSolverUMFPACK,
}

#[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
impl UmfpackLu {
    fn factor(matrix: &SparseMatrix) -> Result<Self, String> {
        use russell_sparse::prelude::{ComplexCooMatrix, ComplexLinSolTrait, LinSolParams, Sym};

        if matrix.rows != matrix.cols {
            return Err("LU factorization requires a square matrix".to_string());
        }
        let mut coo = ComplexCooMatrix::new(matrix.rows, matrix.cols, matrix.nnz(), Sym::No)
            .map_err(|err| err.to_string())?;
        for col in 0..matrix.cols {
            for (row, value) in matrix.column_entries(col) {
                coo.put(row, col, russell_complex(value))
                    .map_err(|err| err.to_string())?;
            }
        }
        let mut solver =
            russell_sparse::prelude::ComplexSolverUMFPACK::new().map_err(|err| err.to_string())?;
        solver
            .factorize(&coo, Some(LinSolParams::new()))
            .map_err(|err| err.to_string())?;
        Ok(Self {
            n: matrix.rows,
            solver,
        })
    }

    fn solve(&mut self, rhs: &[Complex64]) -> Result<Vec<Complex64>, String> {
        use russell_sparse::prelude::ComplexLinSolTrait;

        if rhs.len() != self.n {
            return Err("right-hand side length does not match LU size".to_string());
        }
        let rhs_values = rhs.iter().copied().map(russell_complex).collect::<Vec<_>>();
        let rhs = russell_lab::ComplexVector::from(&rhs_values);
        let mut x = russell_lab::ComplexVector::new(self.n);
        self.solver
            .solve(&mut x, &rhs, false)
            .map_err(|err| err.to_string())?;
        Ok(x.as_data().iter().copied().map(num_complex).collect())
    }
}

#[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
fn russell_complex(value: Complex64) -> russell_lab::Complex64 {
    russell_lab::Complex64::new(value.re, value.im)
}

#[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
fn num_complex(value: russell_lab::Complex64) -> Complex64 {
    Complex64::new(value.re, value.im)
}

#[derive(Clone, Debug)]
struct SparseLu {
    n: usize,
    l: rlu::Matrix<usize, Complex64>,
    u: rlu::Matrix<usize, Complex64>,
    row_perm: Vec<Option<usize>>,
}

impl SparseLu {
    fn factor(matrix: &SparseMatrix) -> Result<Self, String> {
        // Native sparse LU is the fallback linear solve used by both the native
        // Arnoldi loop and the ARPACK-without-UMFPACK path. Validate pivots here
        // so `solve` can assume the factors are usable.
        if matrix.rows != matrix.cols {
            return Err("LU factorization requires a square matrix".to_string());
        }
        let (l, u, row_perm) = rlu::lu_decomposition(
            matrix.rows,
            matrix.row_indices(),
            matrix.col_ptrs(),
            matrix.values(),
            None::<&[usize]>,
            None,
            None,
            true,
        );
        if row_perm.iter().any(|value| value.is_none()) {
            return Err("sparse LU failed to find a complete pivot set".to_string());
        }
        Ok(Self {
            n: matrix.rows,
            l,
            u,
            row_perm,
        })
    }

    fn solve(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, String> {
        if rhs.len() != self.n {
            return Err("right-hand side length does not match LU size".to_string());
        }
        let mut out = vec![Complex64::new(0.0, 0.0); self.n];
        for (index, value) in rhs.iter().copied().enumerate() {
            out[self.row_perm[index].expect("validated pivot set")] = value;
        }
        rlu::lsolve(&self.l, &mut out);
        rlu::usolve(&self.u, &mut out);
        Ok(out)
    }
}

fn null_vector_for_eigenvalue(
    mat: &DMatrix<Complex64>,
    value: Complex64,
) -> Result<Vec<Complex64>, String> {
    // Given lambda, an eigenvector is in the null space of A - lambda I. The
    // smallest-singular right vector is a stable way to recover that direction
    // for the small projected Arnoldi matrix.
    let mut shifted = mat.clone();
    let dim = shifted.nrows();
    for index in 0..dim {
        shifted[(index, index)] -= value;
    }
    let svd = SVD::try_new(shifted, false, true, f64::EPSILON * 16.0, 0)
        .ok_or_else(|| "failed to compute null vector SVD".to_string())?;
    let v_t = svd
        .v_t
        .ok_or_else(|| "SVD did not return right singular vectors".to_string())?;
    let row = v_t.row(v_t.nrows() - 1);
    let vector = row.iter().map(|value| value.conj()).collect::<Vec<_>>();
    Ok(normalize_complex_vector(vector))
}

fn normalize_complex_vector(mut vector: Vec<Complex64>) -> Vec<Complex64> {
    let norm = vector_norm(&vector);
    if norm > 0.0 {
        scale_vector(&mut vector, Complex64::new(1.0 / norm, 0.0));
    }
    vector
}

pub(crate) fn vector_norm(vector: &[Complex64]) -> f64 {
    vector
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn scale_vector(vector: &mut [Complex64], scale: Complex64) {
    for value in vector {
        *value *= scale;
    }
}

pub(crate) fn complex_dot(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

fn axpy(target: &mut [Complex64], vector: &[Complex64], scale: Complex64) {
    assert_eq!(target.len(), vector.len());
    for (target, value) in target.iter_mut().zip(vector) {
        *target += scale * *value;
    }
}

fn default_initial_vector(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|index| {
            let x = (index + 1) as f64;
            Complex64::new((0.37 * x).sin(), (0.53 * x).cos())
        })
        .collect()
}

fn combine_ritz_vector(q_vectors: &[Vec<Complex64>], coeffs: &[Complex64]) -> Vec<Complex64> {
    let n = q_vectors[0].len();
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (basis, coeff) in q_vectors.iter().zip(coeffs) {
        axpy(&mut out, basis, *coeff);
    }
    normalize_complex_vector(out)
}

pub(crate) fn sparse_residual_norm(
    mat: &SparseMatrix,
    vector: &[Complex64],
    value: Complex64,
) -> f64 {
    // Relative residual ||A v - lambda v|| / ||v||. Python surfaces this in
    // solver diagnostics, and the tests use it to catch backend regressions.
    let mut residual = mat.matvec(vector);
    axpy(&mut residual, vector, -value);
    vector_norm(&residual) / vector_norm(vector).max(f64::EPSILON)
}

pub fn diagonal_eigs_to_effective_index(eigenvalues: &[Complex64]) -> Vec<Complex64> {
    eigenvalues
        .iter()
        .map(|value| (-*value + Complex64::new(0.0, 0.0)).sqrt())
        .collect()
}
