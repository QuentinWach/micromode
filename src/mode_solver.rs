//! Mode-level solve orchestration.
//!
//! `operators` builds the matrices and `eigensolve` returns transverse
//! eigenvectors. This module turns those eigenvectors into user-facing mode
//! data: effective indices, all six field components, direction handling,
//! deterministic phase, and unit-power normalization.

use num_complex::Complex64;

use crate::derivatives::{Tensor3, ETA0};
use crate::eigensolve::{
    diagonal_eigs_to_effective_index, selected_sparse_shift_invert_eigenpairs, ShiftInvertOptions,
};
use crate::operators::{assemble_sparse_diagonal_operators, assemble_sparse_tensorial_operator};
use crate::sparse_matrix::SparseMatrix;

#[derive(Clone, Debug)]
pub struct SolveDiagnostics {
    pub backend: String,
    pub operator_size: usize,
    pub operator_nnz: usize,
    pub residuals: Vec<f64>,
    pub power_norms: Vec<f64>,
    pub lorentz_norms: Vec<Complex64>,
    pub lorentz_orthogonality_error: f64,
}

#[derive(Clone, Debug)]
pub struct DiagonalSolveResult {
    pub n_complex: Vec<Complex64>,
    pub fields: [Vec<Vec<Complex64>>; 6],
    pub diagnostics: SolveDiagnostics,
}

#[derive(Clone, Debug)]
struct ModeFields {
    ex: Vec<Complex64>,
    ey: Vec<Complex64>,
    ez: Vec<Complex64>,
    hx: Vec<Complex64>,
    hy: Vec<Complex64>,
    hz: Vec<Complex64>,
}

impl ModeFields {
    fn mutable_components(&mut self) -> [&mut Vec<Complex64>; 6] {
        [
            &mut self.ex,
            &mut self.ey,
            &mut self.ez,
            &mut self.hx,
            &mut self.hy,
            &mut self.hz,
        ]
    }

    fn add_scaled(&mut self, other: &ModeFields, scale: Complex64) {
        for (left, right) in self.ex.iter_mut().zip(&other.ex) {
            *left += scale * *right;
        }
        for (left, right) in self.ey.iter_mut().zip(&other.ey) {
            *left += scale * *right;
        }
        for (left, right) in self.ez.iter_mut().zip(&other.ez) {
            *left += scale * *right;
        }
        for (left, right) in self.hx.iter_mut().zip(&other.hx) {
            *left += scale * *right;
        }
        for (left, right) in self.hy.iter_mut().zip(&other.hy) {
            *left += scale * *right;
        }
        for (left, right) in self.hz.iter_mut().zip(&other.hz) {
            *left += scale * *right;
        }
    }
}

pub fn solve_diagonal_sparse(
    eps: &Tensor3,
    mu: &Tensor3,
    der_mats: &[SparseMatrix; 4],
    cell_areas: &[f64],
    num_modes: usize,
    neff_guess: f64,
    direction: &str,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
) -> Result<DiagonalSolveResult, String> {
    // Production diagonal-media path. The eigenvector contains only [Ex, Ey];
    // the remaining components are reconstructed from Maxwell curl equations.
    validate_cell_areas(cell_areas, eps[0][0].len())?;
    let operators = assemble_sparse_diagonal_operators(eps, mu, der_mats);
    let eig_guess = Complex64::new(-(neff_guess * neff_guess), 0.0);
    let pairs = selected_sparse_shift_invert_eigenpairs(
        &operators.mat,
        num_modes,
        eig_guess,
        initial_vector,
        options,
    )?;
    let operator_size = operators.mat.rows;
    let operator_nnz = operators.mat.nnz();
    let mut modes = pairs
        .into_iter()
        .map(|pair| {
            let n_complex = diagonal_eigs_to_effective_index(&[pair.value])[0];
            (n_complex, pair.vector, pair.residual, pair.backend)
        })
        .collect::<Vec<_>>();
    modes.sort_by(|(left, _, _, _), (right, _, _, _)| {
        left.re
            .partial_cmp(&right.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
    });

    let n = eps[0][0].len();
    let dxf = &der_mats[0];
    let dxb = &der_mats[1];
    let dyf = &der_mats[2];
    let dyb = &der_mats[3];
    let inv_eps_zz = SparseMatrix::diagonal(
        &eps[2][2]
            .iter()
            .map(|value| Complex64::new(1.0, 0.0) / *value)
            .collect::<Vec<_>>(),
    );
    let inv_mu_zz = SparseMatrix::diagonal(
        &mu[2][2]
            .iter()
            .map(|value| Complex64::new(1.0, 0.0) / *value)
            .collect::<Vec<_>>(),
    );

    let mut n_complex = Vec::with_capacity(modes.len());
    let mut mode_fields = Vec::with_capacity(modes.len());
    let mut residuals = Vec::with_capacity(modes.len());
    let backend = modes
        .first()
        .map(|(_, _, _, backend)| *backend)
        .unwrap_or("sparse_shift_invert");

    for (mode_n, vector, residual, _) in modes {
        // Eigenvector layout is [Ex, Ey] on the flattened local Yee grid.
        let ex = vector[..n].to_vec();
        let ey = vector[n..].to_vec();
        let denom = Complex64::new(-mode_n.im, mode_n.re);

        let h_field = operators.qmat.matvec(&vector);
        let mut hx = h_field[..n]
            .iter()
            .map(|value| *value / denom)
            .collect::<Vec<_>>();
        let mut hy = h_field[n..]
            .iter()
            .map(|value| *value / denom)
            .collect::<Vec<_>>();

        // Reconstruct longitudinal fields from the stored transverse fields.
        let dxf_ey = dxf.matvec(&ey);
        let dyf_ex = dyf.matvec(&ex);
        let hz_source = dxf_ey
            .iter()
            .zip(&dyf_ex)
            .map(|(left, right)| *left - *right)
            .collect::<Vec<_>>();
        let mut hz = inv_mu_zz.matvec(&hz_source);

        let h_partial_field = operators
            .q_ep
            .matvec(&vector)
            .into_iter()
            .map(|value| value / denom)
            .collect::<Vec<_>>();
        let dxb_hy = dxb.matvec(&h_partial_field[n..]);
        let dyb_hx = dyb.matvec(&h_partial_field[..n]);
        let ez_source = dxb_hy
            .iter()
            .zip(&dyb_hx)
            .map(|(left, right)| *left - *right)
            .collect::<Vec<_>>();
        let mut ez = inv_eps_zz.matvec(&ez_source);

        let h_scale = Complex64::new(0.0, -1.0) / ETA0;
        for component in [&mut hx, &mut hy, &mut hz] {
            for value in component {
                *value *= h_scale;
            }
        }

        if direction == "-" {
            for value in &mut hx {
                *value *= -1.0;
            }
            for value in &mut hy {
                *value *= -1.0;
            }
            for value in &mut ez {
                *value *= -1.0;
            }
        }
        n_complex.push(mode_n);
        residuals.push(residual);
        mode_fields.push(ModeFields {
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
        });
    }
    let orthogonalization = lorentz_orthogonalize_and_normalize(&mut mode_fields, cell_areas);
    let fields = collect_mode_fields(mode_fields);

    Ok(DiagonalSolveResult {
        n_complex,
        fields,
        diagnostics: SolveDiagnostics {
            backend: backend.to_string(),
            operator_size,
            operator_nnz,
            residuals,
            power_norms: orthogonalization.power_norms,
            lorentz_norms: orthogonalization.lorentz_norms,
            lorentz_orthogonality_error: orthogonalization.lorentz_orthogonality_error,
        },
    })
}

pub fn solve_tensorial_sparse(
    eps: &Tensor3,
    mu: &Tensor3,
    der_mats: &[SparseMatrix; 4],
    cell_areas: &[f64],
    num_modes: usize,
    neff_guess: f64,
    direction: &str,
    initial_vector: Option<&[Complex64]>,
    options: ShiftInvertOptions,
) -> Result<DiagonalSolveResult, String> {
    // Tensorial path for off-diagonal material tensors and angle/bend coordinate
    // transforms. The eigenvector keeps both transverse E and H components.
    validate_cell_areas(cell_areas, eps[0][0].len())?;
    let operator = assemble_sparse_tensorial_operator(eps, mu, der_mats);
    let eig_guess = Complex64::new(neff_guess, 0.0);
    let pairs = selected_sparse_shift_invert_eigenpairs(
        &operator,
        num_modes,
        eig_guess,
        initial_vector,
        options,
    )?;
    let operator_size = operator.rows;
    let operator_nnz = operator.nnz();
    let mut modes = pairs
        .into_iter()
        .map(|pair| (pair.value, pair.vector, pair.residual, pair.backend))
        .collect::<Vec<_>>();
    modes.sort_by(|(left, _, _, _), (right, _, _, _)| {
        left.re
            .partial_cmp(&right.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
    });

    let n = eps[0][0].len();
    let dxf = &der_mats[0];
    let dxb = &der_mats[1];
    let dyf = &der_mats[2];
    let dyb = &der_mats[3];
    let inv_eps_zz = SparseMatrix::diagonal(
        &eps[2][2]
            .iter()
            .map(|value| Complex64::new(1.0, 0.0) / *value)
            .collect::<Vec<_>>(),
    );
    let inv_mu_zz = SparseMatrix::diagonal(
        &mu[2][2]
            .iter()
            .map(|value| Complex64::new(1.0, 0.0) / *value)
            .collect::<Vec<_>>(),
    );

    let mut n_complex = Vec::with_capacity(modes.len());
    let mut mode_fields = Vec::with_capacity(modes.len());
    let mut residuals = Vec::with_capacity(modes.len());
    let backend = modes
        .first()
        .map(|(_, _, _, backend)| *backend)
        .unwrap_or("sparse_shift_invert");

    for (mode_n, vector, residual, _) in modes {
        if vector.len() != 4 * n {
            return Err("tensorial eigenvector has an unexpected length".to_string());
        }
        // Tensorial eigenvector layout is [Ex, Ey, Hx, Hy].
        let ex = vector[..n].to_vec();
        let ey = vector[n..2 * n].to_vec();
        let mut hx = vector[2 * n..3 * n].to_vec();
        let mut hy = vector[3 * n..4 * n].to_vec();

        // Reconstruct Hz/Ez while accounting for off-diagonal tensor coupling
        // into the eliminated z components.
        let dxf_ey = dxf.matvec(&ey);
        let dyf_ex = dyf.matvec(&ex);
        let hz_source = dxf_ey
            .iter()
            .zip(&dyf_ex)
            .zip(mu[2][0].iter())
            .zip(mu[2][1].iter())
            .zip(hx.iter())
            .zip(hy.iter())
            .map(|(((((dxf_ey, dyf_ex), mu_20), mu_21), hx), hy)| {
                *dxf_ey - *dyf_ex - *mu_20 * *hx - *mu_21 * *hy
            })
            .collect::<Vec<_>>();
        let mut hz = inv_mu_zz.matvec(&hz_source);

        let dxb_hy = dxb.matvec(&hy);
        let dyb_hx = dyb.matvec(&hx);
        let ez_source = dxb_hy
            .iter()
            .zip(&dyb_hx)
            .zip(eps[2][0].iter())
            .zip(eps[2][1].iter())
            .zip(ex.iter())
            .zip(ey.iter())
            .map(|(((((dxb_hy, dyb_hx), eps_20), eps_21), ex), ey)| {
                *dxb_hy - *dyb_hx - *eps_20 * *ex - *eps_21 * *ey
            })
            .collect::<Vec<_>>();
        let mut ez = inv_eps_zz.matvec(&ez_source);

        let h_scale = Complex64::new(0.0, -1.0) / ETA0;
        for component in [&mut hx, &mut hy, &mut hz] {
            for value in component {
                *value *= h_scale;
            }
        }

        if direction == "-" {
            for value in &mut hx {
                *value *= -1.0;
            }
            for value in &mut hy {
                *value *= -1.0;
            }
            for value in &mut ez {
                *value *= -1.0;
            }
        }
        n_complex.push(mode_n);
        residuals.push(residual);
        mode_fields.push(ModeFields {
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
        });
    }
    let orthogonalization = lorentz_orthogonalize_and_normalize(&mut mode_fields, cell_areas);
    let fields = collect_mode_fields(mode_fields);

    Ok(DiagonalSolveResult {
        n_complex,
        fields,
        diagnostics: SolveDiagnostics {
            backend: backend.to_string(),
            operator_size,
            operator_nnz,
            residuals,
            power_norms: orthogonalization.power_norms,
            lorentz_norms: orthogonalization.lorentz_norms,
            lorentz_orthogonality_error: orthogonalization.lorentz_orthogonality_error,
        },
    })
}

#[derive(Clone, Debug)]
struct OrthogonalizationDiagnostics {
    power_norms: Vec<f64>,
    lorentz_norms: Vec<Complex64>,
    lorentz_orthogonality_error: f64,
}

fn lorentz_orthogonalize_and_normalize(
    modes: &mut [ModeFields],
    cell_areas: &[f64],
) -> OrthogonalizationDiagnostics {
    // First normalize each reconstructed eigenmode to a sane amplitude. Then
    // apply modified Gram-Schmidt with the unconjugated Lorentz reciprocity
    // product:
    //
    //   L(a,b) = 1/2 integral[((Ea x Hb) + (Eb x Ha)) . z] dA
    //
    // The existing unit-power normalization uses H*, which fixes physical
    // amplitude. This Lorentz product intentionally does not conjugate either
    // mode; it removes residual mixing between modes in the reciprocal
    // eigenbasis.
    for mode in modes.iter_mut() {
        normalize_to_unit_power(mode.mutable_components(), cell_areas);
    }

    for mode_index in 0..modes.len() {
        for previous_index in 0..mode_index {
            let denom = lorentz_overlap(&modes[previous_index], &modes[previous_index], cell_areas);
            if denom.norm() <= f64::EPSILON {
                continue;
            }
            let numer = lorentz_overlap(&modes[previous_index], &modes[mode_index], cell_areas);
            let coeff = numer / denom;
            let previous = modes[previous_index].clone();
            modes[mode_index].add_scaled(&previous, -coeff);
        }
        normalize_to_unit_power(modes[mode_index].mutable_components(), cell_areas);
        apply_dominant_e_phase_convention(modes[mode_index].mutable_components());
    }

    let power_norms = modes
        .iter_mut()
        .map(|mode| transverse_power(&mode.mutable_components(), cell_areas).norm())
        .collect::<Vec<_>>();
    let lorentz_norms = modes
        .iter()
        .map(|mode| lorentz_overlap(mode, mode, cell_areas))
        .collect::<Vec<_>>();
    let mut lorentz_orthogonality_error: f64 = 0.0;
    for left in 0..modes.len() {
        for right in 0..modes.len() {
            if left == right {
                continue;
            }
            let denom = (lorentz_norms[left].norm() * lorentz_norms[right].norm()).sqrt();
            if denom <= f64::EPSILON {
                continue;
            }
            let normalized =
                lorentz_overlap(&modes[left], &modes[right], cell_areas).norm() / denom;
            lorentz_orthogonality_error = lorentz_orthogonality_error.max(normalized);
        }
    }

    OrthogonalizationDiagnostics {
        power_norms,
        lorentz_norms,
        lorentz_orthogonality_error,
    }
}

fn collect_mode_fields(modes: Vec<ModeFields>) -> [Vec<Vec<Complex64>>; 6] {
    let mut fields: [Vec<Vec<Complex64>>; 6] = std::array::from_fn(|_| Vec::new());
    for mode in modes {
        fields[0].push(mode.ex);
        fields[1].push(mode.ey);
        fields[2].push(mode.ez);
        fields[3].push(mode.hx);
        fields[4].push(mode.hy);
        fields[5].push(mode.hz);
    }
    fields
}

fn validate_cell_areas(cell_areas: &[f64], n: usize) -> Result<(), String> {
    // Integration weights come from the Python grid edges. Keep this validation
    // at the Rust boundary so normalization cannot silently use the wrong grid.
    if cell_areas.len() != n {
        return Err(format!(
            "cell area vector length {} does not match grid size {n}",
            cell_areas.len()
        ));
    }
    if cell_areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("cell areas must be finite and positive".to_string());
    }
    Ok(())
}

fn normalize_to_unit_power(mut components: [&mut Vec<Complex64>; 6], cell_areas: &[f64]) -> f64 {
    // Scale all six components together so the transverse Poynting product has
    // unit magnitude. This preserves E/H ratios while making returned modes
    // deterministic enough for injection and overlap calculations.
    let power = transverse_power(&components, cell_areas);
    let norm = power.norm();
    if norm <= f64::EPSILON {
        return 0.0;
    }
    let scale = 1.0 / norm.sqrt();
    for component in &mut components {
        for value in component.iter_mut() {
            *value *= scale;
        }
    }
    transverse_power(&components, cell_areas).norm()
}

fn transverse_power(components: &[&mut Vec<Complex64>; 6], cell_areas: &[f64]) -> Complex64 {
    // Local z-normal power flux: integral((E x H*) . z) dA.
    let ex = &components[0];
    let ey = &components[1];
    let hx = &components[3];
    let hy = &components[4];
    ex.iter()
        .zip(ey.iter())
        .zip(hx.iter())
        .zip(hy.iter())
        .zip(cell_areas.iter())
        .map(|((((ex, ey), hx), hy), area)| {
            (*ex * hy.conj() - *ey * hx.conj()) * Complex64::new(*area, 0.0)
        })
        .sum()
}

fn lorentz_overlap(left: &ModeFields, right: &ModeFields, cell_areas: &[f64]) -> Complex64 {
    // Symmetric unconjugated Lorentz product for z-normal mode planes. The
    // solver currently reconstructs local fields with propagation along local
    // z, and Python later permutes labels back to global axes if needed.
    let left_cross_right = left
        .ex
        .iter()
        .zip(&left.ey)
        .zip(&right.hx)
        .zip(&right.hy)
        .zip(cell_areas)
        .map(|((((ex, ey), hx), hy), area)| (*ex * *hy - *ey * *hx) * Complex64::new(*area, 0.0))
        .sum::<Complex64>();
    let right_cross_left = right
        .ex
        .iter()
        .zip(&right.ey)
        .zip(&left.hx)
        .zip(&left.hy)
        .zip(cell_areas)
        .map(|((((ex, ey), hx), hy), area)| (*ex * *hy - *ey * *hx) * Complex64::new(*area, 0.0))
        .sum::<Complex64>();
    (left_cross_right + right_cross_left) * Complex64::new(0.5, 0.0)
}

fn apply_dominant_e_phase_convention(mut components: [&mut Vec<Complex64>; 6]) {
    // Eigenvectors have arbitrary complex phase. Anchor the dominant electric
    // sample to be real-positive so plots and saved fixtures are stable between
    // runs/backends.
    let mut anchor = Complex64::new(0.0, 0.0);
    let mut anchor_norm = 0.0;
    for component in components.iter().take(3) {
        for value in component.iter() {
            let norm = value.norm_sqr();
            if norm > anchor_norm {
                anchor = *value;
                anchor_norm = norm;
            }
        }
    }
    if anchor_norm <= f64::EPSILON {
        return;
    }
    let phase = anchor.conj() / Complex64::new(anchor_norm.sqrt(), 0.0);
    for component in &mut components {
        for value in component.iter_mut() {
            *value *= phase;
        }
    }
}
