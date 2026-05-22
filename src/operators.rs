//! Maxwell operator assembly.
//!
//! This module contains only matrix construction. It does not choose modes,
//! reconstruct fields, normalize amplitudes, or know about Python. Keeping the
//! algebra here isolated makes it easier to replace individual pieces without
//! touching the eigensolver.

use num_complex::Complex64;

use crate::derivatives::Tensor3;
use crate::sparse_matrix::SparseMatrix;

#[derive(Clone, Debug)]
pub struct SparseDiagonalOperators {
    pub p_mu: SparseMatrix,
    pub p_partial: SparseMatrix,
    pub q_ep: SparseMatrix,
    pub q_partial: SparseMatrix,
    pub qmat: SparseMatrix,
    pub mat: SparseMatrix,
}

pub fn assemble_sparse_diagonal_operators(
    eps: &Tensor3,
    mu: &Tensor3,
    der_mats: &[SparseMatrix; 4],
) -> SparseDiagonalOperators {
    // Diagonal media reduce to a 2N x 2N transverse-electric eigenproblem for
    // [Ex, Ey]. The P/Q block notation follows the standard vectorial FDFD
    // mode formulation: Q maps transverse E to transverse H up to the
    // propagation constant, while P maps transverse H back to transverse E.
    // Ez, Hz, and the physical H scale are reconstructed after the eigen solve.
    let n = eps[0][0].len();
    let eps_xx = &eps[0][0];
    let eps_yy = &eps[1][1];
    let eps_zz = &eps[2][2];
    let mu_xx = &mu[0][0];
    let mu_yy = &mu[1][1];
    let mu_zz = &mu[2][2];
    let dxf = &der_mats[0];
    let dxb = &der_mats[1];
    let dyf = &der_mats[2];
    let dyb = &der_mats[3];

    let zero = SparseMatrix::zeros(n, n);
    let inv_eps_zz = SparseMatrix::diagonal(
        &eps_zz
            .iter()
            .map(|value| Complex64::new(1.0, 0.0) / *value)
            .collect::<Vec<_>>(),
    );
    let inv_mu_zz = SparseMatrix::diagonal(
        &mu_zz
            .iter()
            .map(|value| Complex64::new(1.0, 0.0) / *value)
            .collect::<Vec<_>>(),
    );

    // Material-only part of P. The signs encode z-normal cross products:
    // transverse H couples to [Ey, -Ex] through the diagonal mu block.
    let p_mu = SparseMatrix::block_2x2(
        &zero,
        &SparseMatrix::diagonal(mu_yy),
        &SparseMatrix::diagonal(mu_xx).scale(Complex64::new(-1.0, 0.0)),
        &zero,
    );

    // Longitudinal electric elimination. The derivative sandwich
    // Df * inv(eps_zz) * Db is the Schur-complement contribution from Ez.
    let p00 = dxf
        .matmul(&inv_eps_zz)
        .matmul(dyb)
        .scale(Complex64::new(-1.0, 0.0));
    let p01 = dxf.matmul(&inv_eps_zz).matmul(dxb);
    let p10 = dyf
        .matmul(&inv_eps_zz)
        .matmul(dyb)
        .scale(Complex64::new(-1.0, 0.0));
    let p11 = dyf.matmul(&inv_eps_zz).matmul(dxb);
    let p_partial = SparseMatrix::block_2x2(&p00, &p01, &p10, &p11);

    // Material-only part of Q. It is the epsilon-side analogue of p_mu.
    let q_ep = SparseMatrix::block_2x2(
        &zero,
        &SparseMatrix::diagonal(eps_yy),
        &SparseMatrix::diagonal(eps_xx).scale(Complex64::new(-1.0, 0.0)),
        &zero,
    );

    // Longitudinal magnetic elimination. This mirrors p_partial with mu_zz and
    // backward/forward derivatives swapped for Yee staggering.
    let q00 = dxb
        .matmul(&inv_mu_zz)
        .matmul(dyf)
        .scale(Complex64::new(-1.0, 0.0));
    let q01 = dxb.matmul(&inv_mu_zz).matmul(dxf);
    let q10 = dyb
        .matmul(&inv_mu_zz)
        .matmul(dyf)
        .scale(Complex64::new(-1.0, 0.0));
    let q11 = dyb.matmul(&inv_mu_zz).matmul(dxf);
    let q_partial = SparseMatrix::block_2x2(&q00, &q01, &q10, &q11);

    let qmat = q_ep.add(&q_partial);
    let mat = p_mu.matmul(&qmat).add(&p_partial.matmul(&q_ep));

    SparseDiagonalOperators {
        p_mu,
        p_partial,
        q_ep,
        q_partial,
        qmat,
        mat,
    }
}

pub fn assemble_sparse_tensorial_operator(
    eps: &Tensor3,
    mu: &Tensor3,
    der_mats: &[SparseMatrix; 4],
) -> SparseMatrix {
    // Full tensor media and coordinate transforms cannot use the simpler
    // diagonal reduction. Here the eigenvector is [Ex, Ey, Hx, Hy], and Ez/Hz
    // are still reconstructed after the solve. Off-diagonal material terms are
    // Schur-complemented through eps_zz and mu_zz.
    let one = Complex64::new(1.0, 0.0);
    let i_scale = Complex64::new(0.0, -1.0);
    let dxf = &der_mats[0];
    let dxb = &der_mats[1];
    let dyf = &der_mats[2];
    let dyb = &der_mats[3];

    let inv_eps_22 = SparseMatrix::diagonal(
        &eps[2][2]
            .iter()
            .map(|value| one / *value)
            .collect::<Vec<_>>(),
    );
    let inv_mu_22 = SparseMatrix::diagonal(
        &mu[2][2]
            .iter()
            .map(|value| one / *value)
            .collect::<Vec<_>>(),
    );

    // Precompute tensor ratios such as eps_zx / eps_zz. These appear repeatedly
    // when eliminating the longitudinal field components.
    let eps_20_over_22 = component_div(&eps[2][0], &eps[2][2]);
    let eps_21_over_22 = component_div(&eps[2][1], &eps[2][2]);
    let eps_02_over_22 = component_div(&eps[0][2], &eps[2][2]);
    let eps_12_over_22 = component_div(&eps[1][2], &eps[2][2]);
    let mu_20_over_22 = component_div(&mu[2][0], &mu[2][2]);
    let mu_21_over_22 = component_div(&mu[2][1], &mu[2][2]);
    let mu_02_over_22 = component_div(&mu[0][2], &mu[2][2]);
    let mu_12_over_22 = component_div(&mu[1][2], &mu[2][2]);

    // Schur-complemented transverse material blocks. The suffix `_s` means the
    // effect of the eliminated z component has already been included.
    let mu_10_s = component_sub(&mu[1][0], &component_mul(&mu[1][2], &mu_20_over_22));
    let mu_11_s = component_sub(&mu[1][1], &component_mul(&mu[1][2], &mu_21_over_22));
    let mu_00_s = component_sub(&mu[0][0], &component_mul(&mu[0][2], &mu_20_over_22));
    let mu_01_s = component_sub(&mu[0][1], &component_mul(&mu[0][2], &mu_21_over_22));
    let eps_10_s = component_sub(&eps[1][0], &component_mul(&eps[1][2], &eps_20_over_22));
    let eps_11_s = component_sub(&eps[1][1], &component_mul(&eps[1][2], &eps_21_over_22));
    let eps_00_s = component_sub(&eps[0][0], &component_mul(&eps[0][2], &eps_20_over_22));
    let eps_01_s = component_sub(&eps[0][1], &component_mul(&eps[0][2], &eps_21_over_22));

    let diag = |values: &[Complex64]| SparseMatrix::diagonal(values);

    // The 4x4 block grid below is the first-order tensorial Maxwell operator.
    // The names encode destination/source blocks: a = electric transverse
    // components, b = magnetic transverse components, x/y = local grid axes.
    let axax = dxf
        .matmul(&diag(&eps_20_over_22))
        .scale(-one)
        .sub(&diag(&mu_12_over_22).matmul(dyf));
    let axay = dxf
        .matmul(&diag(&eps_21_over_22))
        .scale(-one)
        .add(&diag(&mu_12_over_22).matmul(dxf));
    let axbx = dxf
        .matmul(&inv_eps_22)
        .matmul(dyb)
        .scale(-one)
        .add(&diag(&mu_10_s));
    let axby = dxf.matmul(&inv_eps_22).matmul(dxb).add(&diag(&mu_11_s));

    let ayax = dyf
        .matmul(&diag(&eps_20_over_22))
        .scale(-one)
        .add(&diag(&mu_02_over_22).matmul(dyf));
    let ayay = dyf
        .matmul(&diag(&eps_21_over_22))
        .scale(-one)
        .sub(&diag(&mu_02_over_22).matmul(dxf));
    let aybx = dyf
        .matmul(&inv_eps_22)
        .matmul(dyb)
        .scale(-one)
        .sub(&diag(&mu_00_s));
    let ayby = dyf.matmul(&inv_eps_22).matmul(dxb).sub(&diag(&mu_01_s));

    let bxax = dxb
        .matmul(&inv_mu_22)
        .matmul(dyf)
        .scale(-one)
        .add(&diag(&eps_10_s));
    let bxay = dxb.matmul(&inv_mu_22).matmul(dxf).add(&diag(&eps_11_s));
    let bxbx = dxb
        .matmul(&diag(&mu_20_over_22))
        .scale(-one)
        .sub(&diag(&eps_12_over_22).matmul(dyb));
    let bxby = dxb
        .matmul(&diag(&mu_21_over_22))
        .scale(-one)
        .add(&diag(&eps_12_over_22).matmul(dxb));

    let byax = dyb
        .matmul(&inv_mu_22)
        .matmul(dyf)
        .scale(-one)
        .sub(&diag(&eps_00_s));
    let byay = dyb.matmul(&inv_mu_22).matmul(dxf).sub(&diag(&eps_01_s));
    let bybx = dyb
        .matmul(&diag(&mu_20_over_22))
        .scale(-one)
        .add(&diag(&eps_02_over_22).matmul(dyb));
    let byby = dyb
        .matmul(&diag(&mu_21_over_22))
        .scale(-one)
        .sub(&diag(&eps_02_over_22).matmul(dxb));

    SparseMatrix::block_grid(&[
        vec![&axax, &axay, &axbx, &axby],
        vec![&ayax, &ayay, &aybx, &ayby],
        vec![&bxax, &bxay, &bxbx, &bxby],
        vec![&byax, &byay, &bybx, &byby],
    ])
    .scale(i_scale)
}

fn component_div(left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| *left / *right)
        .collect()
}

fn component_mul(left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| *left * *right)
        .collect()
}

fn component_sub(left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| *left - *right)
        .collect()
}
