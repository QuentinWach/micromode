use num_complex::Complex64;

use crate::sparse_matrix::SparseMatrix;

pub const C0: f64 = 2.997_924_58e14;
pub const MU0: f64 = 1.256_637_062_12e-12;
pub const EPSILON0: f64 = 1.0 / (MU0 * C0 * C0);
pub const ETA0: f64 = 376.730_313_666_853_5;

pub type Tensor3 = [[Vec<Complex64>; 3]; 3];

#[derive(Clone, Debug)]
pub struct PmlProfile {
    pub sigma_max: f64,
    pub kappa_min: f64,
    pub kappa_max: f64,
    pub order: i32,
}

impl Default for PmlProfile {
    fn default() -> Self {
        Self {
            sigma_max: 2.0,
            kappa_min: 1.0,
            kappa_max: 3.0,
            order: 3,
        }
    }
}

pub fn make_dxf_sparse(dls: &[f64], shape: (usize, usize), pmc: bool) -> SparseMatrix {
    let (nx, ny) = shape;
    if nx == 1 {
        return SparseMatrix::zeros(ny, ny);
    }
    let mut triplets = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            let row = ix * ny + iy;
            let scale = 1.0 / dls[ix];
            let diagonal = if ix == 0 && !pmc { 0.0 } else { -scale };
            if diagonal != 0.0 {
                triplets.push((row, row, Complex64::new(diagonal, 0.0)));
            }
            if ix + 1 < nx {
                let col = (ix + 1) * ny + iy;
                triplets.push((row, col, Complex64::new(scale, 0.0)));
            }
        }
    }
    SparseMatrix::from_triplets(nx * ny, nx * ny, triplets)
}

pub fn make_dxb_sparse(dls: &[f64], shape: (usize, usize), pmc: bool) -> SparseMatrix {
    let (nx, ny) = shape;
    if nx == 1 {
        return SparseMatrix::zeros(ny, ny);
    }
    let mut triplets = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            let row = ix * ny + iy;
            let scale = 1.0 / dls[ix];
            let diagonal = if ix == 0 {
                if pmc {
                    2.0 * scale
                } else {
                    0.0
                }
            } else {
                scale
            };
            if diagonal != 0.0 {
                triplets.push((row, row, Complex64::new(diagonal, 0.0)));
            }
            if ix > 0 {
                let col = (ix - 1) * ny + iy;
                triplets.push((row, col, Complex64::new(-scale, 0.0)));
            }
        }
    }
    SparseMatrix::from_triplets(nx * ny, nx * ny, triplets)
}

pub fn make_dyf_sparse(dls: &[f64], shape: (usize, usize), pmc: bool) -> SparseMatrix {
    let (nx, ny) = shape;
    if ny == 1 {
        return SparseMatrix::zeros(nx, nx);
    }
    let mut triplets = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            let row = ix * ny + iy;
            let scale = 1.0 / dls[iy];
            let diagonal = if iy == 0 && !pmc { 0.0 } else { -scale };
            if diagonal != 0.0 {
                triplets.push((row, row, Complex64::new(diagonal, 0.0)));
            }
            if iy + 1 < ny {
                let col = ix * ny + iy + 1;
                triplets.push((row, col, Complex64::new(scale, 0.0)));
            }
        }
    }
    SparseMatrix::from_triplets(nx * ny, nx * ny, triplets)
}

pub fn make_dyb_sparse(dls: &[f64], shape: (usize, usize), pmc: bool) -> SparseMatrix {
    let (nx, ny) = shape;
    if ny == 1 {
        return SparseMatrix::zeros(nx, nx);
    }
    let mut triplets = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            let row = ix * ny + iy;
            let scale = 1.0 / dls[iy];
            let diagonal = if iy == 0 {
                if pmc {
                    2.0 * scale
                } else {
                    0.0
                }
            } else {
                scale
            };
            if diagonal != 0.0 {
                triplets.push((row, row, Complex64::new(diagonal, 0.0)));
            }
            if iy > 0 {
                let col = ix * ny + iy - 1;
                triplets.push((row, col, Complex64::new(-scale, 0.0)));
            }
        }
    }
    SparseMatrix::from_triplets(nx * ny, nx * ny, triplets)
}

pub fn create_d_matrices_sparse(
    shape: (usize, usize),
    dlf: (&[f64], &[f64]),
    dlb: (&[f64], &[f64]),
    dmin_pmc: (bool, bool),
) -> [SparseMatrix; 4] {
    [
        make_dxf_sparse(dlf.0, shape, dmin_pmc.0),
        make_dxb_sparse(dlb.0, shape, dmin_pmc.0),
        make_dyf_sparse(dlf.1, shape, dmin_pmc.1),
        make_dyb_sparse(dlb.1, shape, dmin_pmc.1),
    ]
}

pub fn create_s_matrices_sparse(
    omega: f64,
    shape: (usize, usize),
    npml: (usize, usize),
    dlf: (&[f64], &[f64]),
    dlb: (&[f64], &[f64]),
    eps_tensor: &Tensor3,
    mu_tensor: &Tensor3,
    dmin_pml: (bool, bool),
) -> [SparseMatrix; 4] {
    create_s_matrices_sparse_with_profile(
        omega,
        shape,
        npml,
        dlf,
        dlb,
        eps_tensor,
        mu_tensor,
        dmin_pml,
        &PmlProfile::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn create_s_matrices_sparse_with_profile(
    omega: f64,
    shape: (usize, usize),
    npml: (usize, usize),
    dlf: (&[f64], &[f64]),
    dlb: (&[f64], &[f64]),
    eps_tensor: &Tensor3,
    mu_tensor: &Tensor3,
    dmin_pml: (bool, bool),
    profile: &PmlProfile,
) -> [SparseMatrix; 4] {
    create_s_diagonal_values(
        omega, shape, npml, dlf, dlb, eps_tensor, mu_tensor, dmin_pml, profile,
    )
    .map(|values| SparseMatrix::diagonal(&values))
}

#[allow(clippy::too_many_arguments)]
pub fn create_s_diagonal_values(
    omega: f64,
    shape: (usize, usize),
    npml: (usize, usize),
    dlf: (&[f64], &[f64]),
    dlb: (&[f64], &[f64]),
    eps_tensor: &Tensor3,
    mu_tensor: &Tensor3,
    dmin_pml: (bool, bool),
    profile: &PmlProfile,
) -> [Vec<Complex64>; 4] {
    let (nx, ny) = shape;
    let n = nx * ny;
    let avg_speed = average_relative_speed(shape, npml, eps_tensor, mu_tensor);

    let sx_f = create_sfactor(
        "f",
        omega,
        dlf.0,
        nx,
        npml.0,
        dmin_pml.0,
        (avg_speed[0], avg_speed[1]),
        profile,
    );
    let sx_b = create_sfactor(
        "b",
        omega,
        dlb.0,
        nx,
        npml.0,
        dmin_pml.0,
        (avg_speed[0], avg_speed[1]),
        profile,
    );
    let sy_f = create_sfactor(
        "f",
        omega,
        dlf.1,
        ny,
        npml.1,
        dmin_pml.1,
        (avg_speed[2], avg_speed[3]),
        profile,
    );
    let sy_b = create_sfactor(
        "b",
        omega,
        dlb.1,
        ny,
        npml.1,
        dmin_pml.1,
        (avg_speed[2], avg_speed[3]),
        profile,
    );

    let mut sx_f_vec = vec![Complex64::new(0.0, 0.0); n];
    let mut sx_b_vec = vec![Complex64::new(0.0, 0.0); n];
    let mut sy_f_vec = vec![Complex64::new(0.0, 0.0); n];
    let mut sy_b_vec = vec![Complex64::new(0.0, 0.0); n];

    for ix in 0..nx {
        for iy in 0..ny {
            let index = ix * ny + iy;
            sx_f_vec[index] = Complex64::new(1.0, 0.0) / sx_f[ix];
            sx_b_vec[index] = Complex64::new(1.0, 0.0) / sx_b[ix];
            sy_f_vec[index] = Complex64::new(1.0, 0.0) / sy_f[iy];
            sy_b_vec[index] = Complex64::new(1.0, 0.0) / sy_b[iy];
        }
    }

    [sx_f_vec, sx_b_vec, sy_f_vec, sy_b_vec]
}

pub fn average_relative_speed(
    shape: (usize, usize),
    npml: (usize, usize),
    eps_tensor: &Tensor3,
    mu_tensor: &Tensor3,
) -> [Complex64; 4] {
    let eps_avg = pml_average_all_sides(shape, npml, eps_tensor);
    let mu_avg = pml_average_all_sides(shape, npml, mu_tensor);
    let mut out = [Complex64::new(1.0, 0.0); 4];
    for i in 0..4 {
        out[i] = Complex64::new(1.0, 0.0) / (eps_avg[i] * mu_avg[i]).sqrt();
    }
    out
}

fn pml_average_all_sides(
    shape: (usize, usize),
    npml: (usize, usize),
    tensor: &Tensor3,
) -> [Complex64; 4] {
    let (nx, ny) = shape;
    let mut regions = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for comp in 0..3 {
        for ix in 0..nx {
            for iy in 0..ny {
                let value = tensor[comp][comp][ix * ny + iy];
                if ix < npml.0 {
                    regions[0].push(value);
                }
                if ix >= nx.saturating_sub(npml.0).saturating_add(1) {
                    regions[1].push(value);
                }
                if iy < npml.1 {
                    regions[2].push(value);
                }
                if iy >= ny.saturating_sub(npml.1).saturating_add(1) {
                    regions[3].push(value);
                }
            }
        }
    }

    let mut out = [Complex64::new(1.0, 0.0); 4];
    for (index, values) in regions.iter().enumerate() {
        if !values.is_empty() {
            out[index] = values.iter().copied().sum::<Complex64>() / values.len() as f64;
        }
    }
    out
}

pub fn create_sfactor(
    direction: &str,
    omega: f64,
    dls: &[f64],
    n: usize,
    n_pml: usize,
    dmin_pml: bool,
    avg_speed: (Complex64, Complex64),
    profile: &PmlProfile,
) -> Vec<Complex64> {
    if n_pml == 0 {
        return vec![Complex64::new(1.0, 0.0); n];
    }
    match direction {
        "f" => create_sfactor_f(omega, dls, n, n_pml, dmin_pml, avg_speed, profile),
        "b" => create_sfactor_b(omega, dls, n, n_pml, dmin_pml, avg_speed, profile),
        _ => panic!("direction value {direction} not recognized"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn create_sfactor_f(
    omega: f64,
    dls: &[f64],
    n: usize,
    n_pml: usize,
    dmin_pml: bool,
    avg_speed: (Complex64, Complex64),
    profile: &PmlProfile,
) -> Vec<Complex64> {
    let mut sfactor = vec![Complex64::new(1.0, 0.0); n];
    for (i, value) in sfactor.iter_mut().enumerate() {
        if i <= n_pml - 1 && dmin_pml {
            *value = s_value(
                dls[0],
                (n_pml as f64 - i as f64 - 0.5) / n_pml as f64,
                omega,
                avg_speed.0,
                profile,
            );
        } else if i >= n - n_pml {
            *value = s_value(
                dls[dls.len() - 1],
                (i as f64 - (n - n_pml) as f64 + 0.5) / n_pml as f64,
                omega,
                avg_speed.1,
                profile,
            );
        }
    }
    sfactor
}

#[allow(clippy::too_many_arguments)]
pub fn create_sfactor_b(
    omega: f64,
    dls: &[f64],
    n: usize,
    n_pml: usize,
    dmin_pml: bool,
    avg_speed: (Complex64, Complex64),
    profile: &PmlProfile,
) -> Vec<Complex64> {
    let mut sfactor = vec![Complex64::new(1.0, 0.0); n];
    for (i, value) in sfactor.iter_mut().enumerate() {
        if i < n_pml && dmin_pml {
            *value = s_value(
                dls[0],
                (n_pml as f64 - i as f64) / n_pml as f64,
                omega,
                avg_speed.0,
                profile,
            );
        } else if i > n - n_pml {
            *value = s_value(
                dls[dls.len() - 1],
                (i as f64 - (n - n_pml) as f64) / n_pml as f64,
                omega,
                avg_speed.1,
                profile,
            );
        }
    }
    sfactor
}

pub fn s_value(
    dl: f64,
    step: f64,
    omega: f64,
    avg_speed: Complex64,
    profile: &PmlProfile,
) -> Complex64 {
    let step_power = step.powi(profile.order);
    let kappa = profile.kappa_min + (profile.kappa_max - profile.kappa_min) * step_power;
    let sigma = avg_speed * (profile.sigma_max / (ETA0 * dl) * step_power);
    Complex64::new(kappa, 0.0) + Complex64::new(0.0, 1.0) * sigma / (omega * EPSILON0)
}

pub fn tensor_from_flat(flat: &[Vec<(f64, f64)>], n: usize) -> Result<Tensor3, String> {
    if flat.len() != 9 {
        return Err("tensor must contain 9 flattened components".to_string());
    }
    let mut tensor: Tensor3 = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for row in 0..3 {
        for col in 0..3 {
            let values = &flat[row * 3 + col];
            if values.len() != n {
                return Err("tensor component length does not match grid shape".to_string());
            }
            tensor[row][col] = values
                .iter()
                .map(|(real, imag)| Complex64::new(*real, *imag))
                .collect();
        }
    }
    Ok(tensor)
}

#[cfg(test)]
mod sparse_tests {
    use super::*;

    fn sample_tensor(shape: (usize, usize), base: f64) -> Tensor3 {
        let n = shape.0 * shape.1;
        let mut tensor: Tensor3 = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for row in 0..3 {
            for col in 0..3 {
                tensor[row][col] = vec![Complex64::new(0.0, 0.0); n];
            }
        }
        tensor[0][0] = (0..n)
            .map(|index| Complex64::new(base + 0.03 * index as f64, 0.0))
            .collect();
        tensor[1][1] = (0..n)
            .map(|index| Complex64::new(base + 0.2 + 0.02 * index as f64, 0.0))
            .collect();
        tensor[2][2] = (0..n)
            .map(|index| Complex64::new(base + 0.6 + 0.01 * index as f64, 0.0))
            .collect();
        tensor
    }

    #[test]
    fn sparse_derivative_matrices_have_expected_shape() {
        let shape = (4, 5);
        let dlf = (
            vec![0.09, 0.10, 0.13, 0.18],
            vec![0.08, 0.11, 0.12, 0.14, 0.19],
        );
        let dlb = (
            vec![0.09, 0.095, 0.115, 0.15],
            vec![0.08, 0.10, 0.115, 0.13, 0.17],
        );
        let dmin_pmc = (true, false);
        let sparse = create_d_matrices_sparse(shape, (&dlf.0, &dlf.1), (&dlb.0, &dlb.1), dmin_pmc);

        for matrix in &sparse {
            assert_eq!(
                (matrix.rows, matrix.cols),
                (shape.0 * shape.1, shape.0 * shape.1)
            );
            assert!(matrix.nnz() > 0);
        }
    }

    #[test]
    fn sparse_pml_matrices_have_expected_shape() {
        let shape = (4, 5);
        let dlf = (
            vec![0.09, 0.10, 0.13, 0.18],
            vec![0.08, 0.11, 0.12, 0.14, 0.19],
        );
        let dlb = (
            vec![0.09, 0.095, 0.115, 0.15],
            vec![0.08, 0.10, 0.115, 0.13, 0.17],
        );
        let eps = sample_tensor(shape, 2.0);
        let mu = sample_tensor(shape, 1.0);
        let omega = 2.0 * std::f64::consts::PI * 193.414_489e12;
        let npml = (2, 1);
        let dmin_pml = (true, false);

        let sparse = create_s_matrices_sparse(
            omega,
            shape,
            npml,
            (&dlf.0, &dlf.1),
            (&dlb.0, &dlb.1),
            &eps,
            &mu,
            dmin_pml,
        );

        for matrix in &sparse {
            assert_eq!(
                (matrix.rows, matrix.cols),
                (shape.0 * shape.1, shape.0 * shape.1)
            );
            assert!(matrix.nnz() > 0);
        }
    }
}
