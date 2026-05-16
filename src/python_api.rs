//! PyO3 extension boundary.
//!
//! The Python package keeps user-facing objects in Python, while Rust owns the
//! numerical kernels. This module translates Python lists of real/imag pairs
//! into Rust tensors, runs the appropriate solver, and converts diagnostics and
//! fields back into simple Python values.

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{derivatives, diagonal_solver, sparse_matrix};

#[cfg(feature = "python")]
type SolvePayload = (
    // complex effective indices
    Vec<(f64, f64)>,
    // six field components, grouped as component -> mode -> flattened values
    Vec<Vec<Vec<(f64, f64)>>>,
    // eigenpair residuals
    Vec<f64>,
    // power-normalization checks
    Vec<f64>,
    // unconjugated Lorentz self-products and the largest normalized
    // off-diagonal product after orthogonalization
    Vec<(f64, f64)>,
    f64,
    // backend label and sparse operator diagnostics
    String,
    usize,
    usize,
);

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    nx,
    ny,
    dlf_x,
    dlf_y,
    dlb_x,
    dlb_y,
    pmc_x,
    pmc_y,
    eps_tensor,
    mu_tensor,
    num_modes,
    neff_guess,
    direction,
    derivative_scale = None,
    npml_x = 0,
    npml_y = 0,
    pml_sigma_max = 2.0,
    pml_kappa_min = 1.0,
    pml_kappa_max = 3.0,
    pml_order = 3,
    dmin_pml_x = true,
    dmin_pml_y = true,
    omega = None,
    krylov_dim = 32,
    initial_vector = None
))]
fn solve_diagonal_sparse_py(
    nx: usize,
    ny: usize,
    dlf_x: Vec<f64>,
    dlf_y: Vec<f64>,
    dlb_x: Vec<f64>,
    dlb_y: Vec<f64>,
    pmc_x: bool,
    pmc_y: bool,
    eps_tensor: Vec<Vec<(f64, f64)>>,
    mu_tensor: Vec<Vec<(f64, f64)>>,
    num_modes: usize,
    neff_guess: f64,
    direction: String,
    derivative_scale: Option<f64>,
    npml_x: usize,
    npml_y: usize,
    pml_sigma_max: f64,
    pml_kappa_min: f64,
    pml_kappa_max: f64,
    pml_order: i32,
    dmin_pml_x: bool,
    dmin_pml_y: bool,
    omega: Option<f64>,
    krylov_dim: usize,
    initial_vector: Option<Vec<(f64, f64)>>,
) -> PyResult<SolvePayload> {
    let n = nx * ny;
    let eps = derivatives::tensor_from_flat(&eps_tensor, n).map_err(PyValueError::new_err)?;
    let mu = derivatives::tensor_from_flat(&mu_tensor, n).map_err(PyValueError::new_err)?;
    let der_mats = sparse_derivative_matrices_for_solve(
        (nx, ny),
        (&dlf_x, &dlf_y),
        (&dlb_x, &dlb_y),
        (pmc_x, pmc_y),
        &eps,
        &mu,
        (npml_x, npml_y),
        derivatives::PmlProfile {
            sigma_max: pml_sigma_max,
            kappa_min: pml_kappa_min,
            kappa_max: pml_kappa_max,
            order: pml_order,
        },
        (dmin_pml_x, dmin_pml_y),
        omega,
        derivative_scale,
    )
    .map_err(PyValueError::new_err)?;
    let initial_vector = initial_vector.map(|values| {
        values
            .into_iter()
            .map(|(re, im)| num_complex::Complex64::new(re, im))
            .collect::<Vec<_>>()
    });
    let cell_areas = cell_areas_from_steps(&dlf_x, &dlf_y);
    let result = diagonal_solver::solve_diagonal_sparse(
        &eps,
        &mu,
        &der_mats,
        &cell_areas,
        num_modes,
        neff_guess,
        &direction,
        initial_vector.as_deref(),
        diagonal_solver::ShiftInvertOptions {
            krylov_dim,
            tolerance: 1e-10,
        },
    )
    .map_err(PyValueError::new_err)?;
    let n_complex = result
        .n_complex
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect();
    let fields = result
        .fields
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|mode| {
                    mode.into_iter()
                        .map(|value| (value.re, value.im))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    Ok((
        n_complex,
        fields,
        result.diagnostics.residuals,
        result.diagnostics.power_norms,
        result
            .diagnostics
            .lorentz_norms
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
        result.diagnostics.lorentz_orthogonality_error,
        result.diagnostics.backend,
        result.diagnostics.operator_size,
        result.diagnostics.operator_nnz,
    ))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    nx,
    ny,
    dlf_x,
    dlf_y,
    dlb_x,
    dlb_y,
    pmc_x,
    pmc_y,
    eps_tensor,
    mu_tensor,
    num_modes,
    neff_guess,
    direction,
    derivative_scale = None,
    npml_x = 0,
    npml_y = 0,
    pml_sigma_max = 2.0,
    pml_kappa_min = 1.0,
    pml_kappa_max = 3.0,
    pml_order = 3,
    dmin_pml_x = true,
    dmin_pml_y = true,
    omega = None,
    krylov_dim = 32,
    initial_vector = None
))]
fn solve_tensorial_sparse_py(
    nx: usize,
    ny: usize,
    dlf_x: Vec<f64>,
    dlf_y: Vec<f64>,
    dlb_x: Vec<f64>,
    dlb_y: Vec<f64>,
    pmc_x: bool,
    pmc_y: bool,
    eps_tensor: Vec<Vec<(f64, f64)>>,
    mu_tensor: Vec<Vec<(f64, f64)>>,
    num_modes: usize,
    neff_guess: f64,
    direction: String,
    derivative_scale: Option<f64>,
    npml_x: usize,
    npml_y: usize,
    pml_sigma_max: f64,
    pml_kappa_min: f64,
    pml_kappa_max: f64,
    pml_order: i32,
    dmin_pml_x: bool,
    dmin_pml_y: bool,
    omega: Option<f64>,
    krylov_dim: usize,
    initial_vector: Option<Vec<(f64, f64)>>,
) -> PyResult<SolvePayload> {
    let n = nx * ny;
    let eps = derivatives::tensor_from_flat(&eps_tensor, n).map_err(PyValueError::new_err)?;
    let mu = derivatives::tensor_from_flat(&mu_tensor, n).map_err(PyValueError::new_err)?;
    let der_mats = sparse_derivative_matrices_for_solve(
        (nx, ny),
        (&dlf_x, &dlf_y),
        (&dlb_x, &dlb_y),
        (pmc_x, pmc_y),
        &eps,
        &mu,
        (npml_x, npml_y),
        derivatives::PmlProfile {
            sigma_max: pml_sigma_max,
            kappa_min: pml_kappa_min,
            kappa_max: pml_kappa_max,
            order: pml_order,
        },
        (dmin_pml_x, dmin_pml_y),
        omega,
        derivative_scale,
    )
    .map_err(PyValueError::new_err)?;
    let initial_vector = initial_vector.map(|values| {
        values
            .into_iter()
            .map(|(re, im)| num_complex::Complex64::new(re, im))
            .collect::<Vec<_>>()
    });
    let cell_areas = cell_areas_from_steps(&dlf_x, &dlf_y);
    let result = diagonal_solver::solve_tensorial_sparse(
        &eps,
        &mu,
        &der_mats,
        &cell_areas,
        num_modes,
        neff_guess,
        &direction,
        initial_vector.as_deref(),
        diagonal_solver::ShiftInvertOptions {
            krylov_dim,
            tolerance: 1e-10,
        },
    )
    .map_err(PyValueError::new_err)?;
    let n_complex = result
        .n_complex
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect();
    let fields = result
        .fields
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|mode| {
                    mode.into_iter()
                        .map(|value| (value.re, value.im))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    Ok((
        n_complex,
        fields,
        result.diagnostics.residuals,
        result.diagnostics.power_norms,
        result
            .diagnostics
            .lorentz_norms
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
        result.diagnostics.lorentz_orthogonality_error,
        result.diagnostics.backend,
        result.diagnostics.operator_size,
        result.diagnostics.operator_nnz,
    ))
}

#[cfg(feature = "python")]
fn cell_areas_from_steps(dlf_x: &[f64], dlf_y: &[f64]) -> Vec<f64> {
    // Python supplies mode-plane edges; by this point they have become cell
    // widths. Flatten in the same x-major/y-minor order used by material tensors.
    let mut areas = Vec::with_capacity(dlf_x.len() * dlf_y.len());
    for dx in dlf_x {
        for dy in dlf_y {
            areas.push(dx.abs() * dy.abs());
        }
    }
    areas
}

#[cfg(feature = "python")]
fn sparse_derivative_matrices_for_solve(
    shape: (usize, usize),
    dlf: (&[f64], &[f64]),
    dlb: (&[f64], &[f64]),
    dmin_pmc: (bool, bool),
    eps: &derivatives::Tensor3,
    mu: &derivatives::Tensor3,
    npml: (usize, usize),
    pml_profile: derivatives::PmlProfile,
    dmin_pml: (bool, bool),
    omega: Option<f64>,
    derivative_scale: Option<f64>,
) -> Result<[sparse_matrix::SparseMatrix; 4], String> {
    // Build sparse derivative matrices, optionally preconditioned by PML
    // stretch matrices and scaled into the nondimensional coordinates used by
    // the operator assembly.
    let mut der_mats = derivatives::create_d_matrices_sparse(shape, dlf, dlb, dmin_pmc);
    if npml.0 > 0 || npml.1 > 0 {
        let omega = omega.ok_or_else(|| "omega is required when num_pml is nonzero".to_string())?;
        let pml_mats = derivatives::create_s_matrices_sparse_with_profile(
            omega,
            shape,
            npml,
            dlf,
            dlb,
            eps,
            mu,
            dmin_pml,
            &pml_profile,
        );
        for (der_mat, pml_mat) in der_mats.iter_mut().zip(pml_mats.iter()) {
            *der_mat = pml_mat.matmul(der_mat);
        }
    }
    if let Some(scale) = derivative_scale {
        for matrix in &mut der_mats {
            *matrix = matrix.scale(num_complex::Complex64::new(scale, 0.0));
        }
    }
    Ok(der_mats)
}

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_diagonal_sparse_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_tensorial_sparse_py, m)?)?;
    Ok(())
}
