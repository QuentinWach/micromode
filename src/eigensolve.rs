//! Eigenvalue selection and shift-invert backends.
//!
//! The solver usually needs a handful of modes near a requested effective
//! index, not the whole spectrum. Shift-invert changes that local search into a
//! dominant-eigenvalue problem for `(A - sigma I)^-1`, which sparse Krylov
//! methods can solve efficiently on realistic grids.

use std::time::{Duration, Instant};

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

#[derive(Clone, Debug, Default)]
pub struct ShiftInvertProfile {
    pub pairs: Vec<Eigenpair>,
    pub total: Duration,
    pub shift_diagonal: Duration,
    pub amd_ordering: Duration,
    pub lu_factorization: Duration,
    pub lu_packing: Duration,
    pub linear_solves: Duration,
    pub arnoldi_orthogonalization: Duration,
    pub hessenberg_eigensolve: Duration,
    pub ritz_reconstruction: Duration,
    pub residuals: Duration,
    pub sorting: Duration,
    pub solve_calls: usize,
    pub arnoldi_steps: usize,
    pub candidate_count: usize,
    pub returned_pairs: usize,
    pub lu_l_nnz: usize,
    pub lu_u_nnz: usize,
    pub max_residual: f64,
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
    selected_sparse_shift_invert_native_eigenpairs_impl(
        mat,
        num_modes,
        guess_value,
        initial_vector,
        options,
        None,
    )
}

pub fn profile_sparse_shift_invert_native_eigenpairs(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
) -> Result<ShiftInvertProfile, String> {
    let total_start = Instant::now();
    let mut profile = ShiftInvertProfile::default();
    let pairs = selected_sparse_shift_invert_native_eigenpairs_impl(
        mat,
        num_modes,
        guess_value,
        initial_vector,
        options,
        Some(&mut profile),
    )?;
    profile.total = total_start.elapsed();
    profile.max_residual = pairs.iter().map(|pair| pair.residual).fold(0.0, f64::max);
    profile.returned_pairs = pairs.len();
    profile.pairs = pairs;
    Ok(profile)
}

fn selected_sparse_shift_invert_native_eigenpairs_impl(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
    mut profile: Option<&mut ShiftInvertProfile>,
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
    let shift_start = profile.as_ref().map(|_| Instant::now());
    let shifted = mat.shifted_diagonal(guess_value);
    if let (Some(profile), Some(start)) = (profile.as_mut(), shift_start) {
        profile.shift_diagonal += start.elapsed();
    }
    let factorization = SparseLu::factor_with_profile(&shifted, profile.as_deref_mut())?;
    let mut solve_workspace = vec![Complex64::new(0.0, 0.0); mat.rows];
    selected_sparse_shift_invert_native_with_solver(
        mat,
        num_modes,
        guess_value,
        initial_vector,
        options,
        "native_shift_invert",
        profile.as_deref_mut(),
        |input, output| factorization.solve_into(input, output, &mut solve_workspace),
    )
}

fn selected_sparse_shift_invert_native_with_solver<F>(
    mat: &SparseMatrix,
    num_modes: usize,
    guess_value: Complex64,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
    backend: &'static str,
    mut profile: Option<&mut ShiftInvertProfile>,
    mut solve: F,
) -> Result<Vec<Eigenpair>, String>
where
    F: FnMut(&[Complex64], &mut [Complex64]) -> Result<(), String>,
{
    let n = mat.rows;
    let krylov_dim = options.krylov_dim.min(n).max(num_modes + 2);
    let checkpoint_start = krylov_dim.min(((3 * krylov_dim + 3) / 4).max((num_modes + 8).max(16)));
    let checkpoint_interval = 4;
    let stability_tolerance = options.tolerance.sqrt();
    let mut previous_checkpoint_values: Option<Vec<Complex64>> = None;
    let start = initial_vector
        .map(|values| {
            if values.len() != n {
                return Err("initial vector length does not match matrix size".to_string());
            }
            Ok(values.to_vec())
        })
        .unwrap_or_else(|| Ok(default_initial_vector(n)))?;
    let mut q_basis = ArnoldiBasis::with_first(normalize_complex_vector(start), krylov_dim + 1);
    let mut hessenberg = DMatrix::<Complex64>::zeros(krylov_dim + 1, krylov_dim);
    let mut actual_dim = 0usize;

    for col in 0..krylov_dim {
        // Native Arnoldi fallback with one reorthogonalization pass.
        let solve_start = profile.as_ref().map(|_| Instant::now());
        let mut work = vec![Complex64::new(0.0, 0.0); n];
        solve(q_basis.vector(col), &mut work)?;
        if let (Some(profile), Some(start)) = (profile.as_mut(), solve_start) {
            profile.linear_solves += start.elapsed();
            profile.solve_calls += 1;
        }
        let orthogonalization_start = profile.as_ref().map(|_| Instant::now());
        for row in 0..=col {
            let basis_vector = q_basis.vector(row);
            let projection = complex_dot(basis_vector, &work);
            hessenberg[(row, col)] = projection;
            axpy(&mut work, basis_vector, -projection);
        }
        for row in 0..=col {
            let basis_vector = q_basis.vector(row);
            let projection = complex_dot(basis_vector, &work);
            hessenberg[(row, col)] += projection;
            axpy(&mut work, basis_vector, -projection);
        }
        let norm = vector_norm(&work);
        actual_dim = col + 1;
        hessenberg[(col + 1, col)] = Complex64::new(norm, 0.0);
        if let (Some(profile), Some(start)) = (profile.as_mut(), orthogonalization_start) {
            profile.arnoldi_orthogonalization += start.elapsed();
        }
        if let Some(profile) = profile.as_mut() {
            profile.arnoldi_steps = actual_dim;
        }

        let exhausted = norm <= options.tolerance || col + 1 == krylov_dim;
        let checkpoint_due = actual_dim >= checkpoint_start
            && ((actual_dim - checkpoint_start) % checkpoint_interval == 0 || exhausted);
        if checkpoint_due {
            let candidates = candidate_eigenpairs_from_hessenberg(
                mat,
                &q_basis,
                &hessenberg,
                actual_dim,
                num_modes,
                guess_value,
                backend,
                options.tolerance,
                profile.as_deref_mut(),
            )?;
            let converged = candidates_converged(&candidates, num_modes, options.tolerance);
            let stable = previous_checkpoint_values.as_ref().is_some_and(|previous| {
                candidates_stable(previous, &candidates, stability_tolerance)
            });
            previous_checkpoint_values =
                Some(candidates.iter().map(|candidate| candidate.value).collect());
            if exhausted || (converged && stable) {
                return Ok(candidates);
            }
        }
        if exhausted {
            break;
        }
        scale_vector(&mut work, Complex64::new(1.0 / norm, 0.0));
        q_basis.push(work);
    }
    if let Some(profile) = profile.as_mut() {
        profile.arnoldi_steps = actual_dim;
    }

    candidate_eigenpairs_from_hessenberg(
        mat,
        &q_basis,
        &hessenberg,
        actual_dim,
        num_modes,
        guess_value,
        backend,
        options.tolerance,
        profile.as_deref_mut(),
    )
}

fn candidate_eigenpairs_from_hessenberg(
    mat: &SparseMatrix,
    q_basis: &ArnoldiBasis,
    hessenberg: &DMatrix<Complex64>,
    actual_dim: usize,
    num_modes: usize,
    guess_value: Complex64,
    backend: &'static str,
    tolerance: f64,
    mut profile: Option<&mut ShiftInvertProfile>,
) -> Result<Vec<Eigenpair>, String> {
    let h_square = hessenberg
        .view((0, 0), (actual_dim, actual_dim))
        .into_owned();
    let hessenberg_start = profile.as_ref().map(|_| Instant::now());
    let theta_values = h_square
        .clone()
        .schur()
        .eigenvalues()
        .ok_or_else(|| "failed to compute Hessenberg eigenvalues".to_string())?;
    if let (Some(profile), Some(start)) = (profile.as_mut(), hessenberg_start) {
        profile.hessenberg_eigensolve += start.elapsed();
    }
    let mut theta_candidates = Vec::new();
    for theta in theta_values.iter().copied() {
        if theta.norm() <= tolerance {
            continue;
        }
        let lambda = guess_value + Complex64::new(1.0, 0.0) / theta;
        theta_candidates.push((lambda, theta));
    }
    theta_candidates.sort_by(|(left, _), (right, _)| {
        let left_distance = (*left - guess_value).norm();
        let right_distance = (*right - guess_value).norm();
        left_distance
            .partial_cmp(&right_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Residuals are only a tie-breaker after shift distance. Avoid the
    // SVD-based full-vector reconstruction for distant projected eigenvalues
    // that cannot survive the final truncation.
    let reconstruction_count = theta_candidates
        .len()
        .min((num_modes * 2).max(num_modes + 2));
    let mut candidates = Vec::with_capacity(reconstruction_count);
    for (lambda, theta) in theta_candidates.into_iter().take(reconstruction_count) {
        let ritz_start = profile.as_ref().map(|_| Instant::now());
        let coeffs = null_vector_for_eigenvalue(&h_square, theta)?;
        let vector = combine_ritz_vector(q_basis, &coeffs);
        if let (Some(profile), Some(start)) = (profile.as_mut(), ritz_start) {
            profile.ritz_reconstruction += start.elapsed();
        }
        let residual_start = profile.as_ref().map(|_| Instant::now());
        let residual = sparse_residual_norm(mat, &vector, lambda);
        if let (Some(profile), Some(start)) = (profile.as_mut(), residual_start) {
            profile.residuals += start.elapsed();
        }
        candidates.push((lambda, vector, residual));
    }
    if let Some(profile) = profile.as_mut() {
        profile.candidate_count = candidates.len();
    }
    // Sort by closeness to the requested shift. Residual is the tie-breaker
    // because it is the direct measure of eigenpair quality.
    let sorting_start = profile.as_ref().map(|_| Instant::now());
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
    if let (Some(profile), Some(start)) = (profile.as_mut(), sorting_start) {
        profile.sorting += start.elapsed();
    }
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

fn candidates_converged(candidates: &[Eigenpair], num_modes: usize, tolerance: f64) -> bool {
    candidates.len() == num_modes
        && candidates
            .iter()
            .all(|candidate| candidate.residual <= tolerance)
}

fn candidates_stable(
    previous_values: &[Complex64],
    candidates: &[Eigenpair],
    tolerance: f64,
) -> bool {
    previous_values.len() == candidates.len()
        && previous_values
            .iter()
            .zip(candidates)
            .all(|(previous, candidate)| (*previous - candidate.value).norm() <= tolerance)
}

#[derive(Clone, Debug)]
struct ArnoldiBasis {
    n: usize,
    values: Vec<Complex64>,
}

impl ArnoldiBasis {
    fn with_first(first: Vec<Complex64>, capacity: usize) -> Self {
        let n = first.len();
        let mut values = Vec::with_capacity(n * capacity);
        values.extend_from_slice(&first);
        Self { n, values }
    }

    fn vector(&self, index: usize) -> &[Complex64] {
        let start = index * self.n;
        &self.values[start..start + self.n]
    }

    fn push(&mut self, vector: Vec<Complex64>) {
        assert_eq!(vector.len(), self.n);
        self.values.extend_from_slice(&vector);
    }
}

#[derive(Clone, Debug)]
struct SparseLu {
    n: usize,
    l: PackedTriangularMatrix,
    u: PackedTriangularMatrix,
    row_perm: Vec<usize>,
    col_perm: Option<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct PackedTriangularMatrix {
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl SparseLu {
    fn factor_with_profile(
        matrix: &SparseMatrix,
        mut profile: Option<&mut ShiftInvertProfile>,
    ) -> Result<Self, String> {
        // Native sparse LU is the fallback linear solve. Validate pivots here
        // so `solve` can assume the factors are usable.
        if matrix.rows != matrix.cols {
            return Err("LU factorization requires a square matrix".to_string());
        }
        let ordering_start = profile.as_ref().map(|_| Instant::now());
        let col_perm = amd::order::<usize>(
            matrix.rows,
            matrix.col_ptrs(),
            matrix.row_indices(),
            &amd::Control::default(),
        )
        .map(|(perm, _, _)| perm)
        .ok();
        if let (Some(profile), Some(start)) = (profile.as_mut(), ordering_start) {
            profile.amd_ordering += start.elapsed();
        }
        let factor_start = profile.as_ref().map(|_| Instant::now());
        let (l_columns, u_columns, row_perm) = rlu::lu_decomposition(
            matrix.rows,
            matrix.row_indices(),
            matrix.col_ptrs(),
            matrix.values(),
            col_perm.as_deref(),
            None,
            None,
            true,
        );
        if let (Some(profile), Some(start)) = (profile.as_mut(), factor_start) {
            profile.lu_factorization += start.elapsed();
        }
        if row_perm.iter().any(|value| value.is_none()) {
            return Err("sparse LU failed to find a complete pivot set".to_string());
        }
        let row_perm = row_perm
            .into_iter()
            .map(|value| value.expect("validated pivot set"))
            .collect();
        let packing_start = profile.as_ref().map(|_| Instant::now());
        let l = PackedTriangularMatrix::from_columns(l_columns);
        let u = PackedTriangularMatrix::from_columns(u_columns);
        if let (Some(profile), Some(start)) = (profile.as_mut(), packing_start) {
            profile.lu_packing += start.elapsed();
            profile.lu_l_nnz = l.nnz();
            profile.lu_u_nnz = u.nnz();
        }
        Ok(Self {
            n: matrix.rows,
            l,
            u,
            row_perm,
            col_perm,
        })
    }

    fn solve_into(
        &self,
        rhs: &[Complex64],
        out: &mut [Complex64],
        work: &mut [Complex64],
    ) -> Result<(), String> {
        if rhs.len() != self.n {
            return Err("right-hand side length does not match LU size".to_string());
        }
        if out.len() != self.n || work.len() != self.n {
            return Err("solve workspace length does not match LU size".to_string());
        }

        if let Some(col_perm) = &self.col_perm {
            for (index, value) in rhs.iter().copied().enumerate() {
                work[self.row_perm[index]] = value;
            }
            self.l.lsolve(work);
            self.u.usolve(work);
            for (index, value) in work.iter().copied().enumerate() {
                out[col_perm[index]] = value;
            }
        } else {
            for (index, value) in rhs.iter().copied().enumerate() {
                out[self.row_perm[index]] = value;
            }
            self.l.lsolve(out);
            self.u.usolve(out);
        }
        Ok(())
    }
}

impl PackedTriangularMatrix {
    fn from_columns(columns: rlu::Matrix<usize, Complex64>) -> Self {
        let mut col_ptrs = Vec::with_capacity(columns.len() + 1);
        let nnz = columns.iter().map(Vec::len).sum();
        let mut row_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        col_ptrs.push(0);
        for column in columns {
            for (row, value) in column {
                row_indices.push(row);
                values.push(value);
            }
            col_ptrs.push(row_indices.len());
        }
        Self {
            col_ptrs,
            row_indices,
            values,
        }
    }

    fn nnz(&self) -> usize {
        self.values.len()
    }

    fn lsolve(&self, rhs: &mut [Complex64]) {
        for col in 0..rhs.len() {
            let value = rhs[col];
            for index in self.col_ptrs[col]..self.col_ptrs[col + 1] {
                rhs[self.row_indices[index]] -= self.values[index] * value;
            }
        }
    }

    fn usolve(&self, rhs: &mut [Complex64]) {
        for col in (0..rhs.len()).rev() {
            for index in (self.col_ptrs[col]..self.col_ptrs[col + 1]).rev() {
                let row = self.row_indices[index];
                if row == col {
                    rhs[col] /= self.values[index];
                } else {
                    rhs[row] -= self.values[index] * rhs[col];
                }
            }
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

fn combine_ritz_vector(q_basis: &ArnoldiBasis, coeffs: &[Complex64]) -> Vec<Complex64> {
    let n = q_basis.n;
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (basis_index, coeff) in coeffs.iter().copied().enumerate() {
        axpy(&mut out, q_basis.vector(basis_index), coeff);
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
