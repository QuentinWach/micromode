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
    selected_sparse_shift_invert_native_eigenpairs(
        mat,
        num_modes,
        guess_value,
        initial_vector,
        options,
    )
}

pub fn selected_sparse_shift_invert_native_eigenpairs(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
) -> Result<Vec<Eigenpair>, String> {
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
    selected_sparse_shift_invert_native_with_solver(
        mat,
        num_modes,
        guess_value,
        initial_vector,
        options,
        "native_shift_invert",
        |input| factorization.solve(input),
    )
}

fn selected_sparse_shift_invert_native_with_solver<F>(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
    backend: &'static str,
    mut solve: F,
) -> Result<Vec<Eigenpair>, String>
where
    F: FnMut(&[Complex64]) -> Result<Vec<Complex64>, String>,
{
    let n = mat.rows;
    let krylov_dim = options.krylov_dim.min(n).max(num_modes + 2);
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
        // Native Arnoldi fallback with one reorthogonalization pass.
        let mut work = solve(&q_vectors[col])?;
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
            backend,
        })
        .collect())
}

#[derive(Clone, Debug)]
struct SparseLu {
    n: usize,
    l: rlu::Matrix<usize, Complex64>,
    u: rlu::Matrix<usize, Complex64>,
    row_perm: Vec<Option<usize>>,
    col_perm: Option<Vec<usize>>,
}

impl SparseLu {
    fn factor(matrix: &SparseMatrix) -> Result<Self, String> {
        // Native sparse LU is the fallback linear solve. Validate pivots here
        // so `solve` can assume the factors are usable.
        if matrix.rows != matrix.cols {
            return Err("LU factorization requires a square matrix".to_string());
        }
        let col_perm = amd::order::<usize>(
            matrix.rows,
            matrix.col_ptrs(),
            matrix.row_indices(),
            &amd::Control::default(),
        )
        .map(|(perm, _, _)| perm)
        .ok();
        let (l, u, row_perm) = rlu::lu_decomposition(
            matrix.rows,
            matrix.row_indices(),
            matrix.col_ptrs(),
            matrix.values(),
            col_perm.as_deref(),
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
            col_perm,
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
        match &self.col_perm {
            Some(col_perm) => {
                let mut unpermuted = vec![Complex64::new(0.0, 0.0); self.n];
                for (index, value) in out.into_iter().enumerate() {
                    unpermuted[col_perm[index]] = value;
                }
                Ok(unpermuted)
            }
            None => Ok(out),
        }
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
