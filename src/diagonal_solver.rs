#[cfg(feature = "arpack-backend")]
pub use crate::eigensolve::selected_sparse_shift_invert_arpack_eigenpairs;
#[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
pub use crate::eigensolve::selected_sparse_shift_invert_umfpack_eigenpairs;
pub use crate::eigensolve::{
    diagonal_eigs_to_effective_index, selected_sparse_shift_invert_eigenpairs, Eigenpair,
    ShiftInvertOptions,
};
pub use crate::mode_solver::{
    solve_diagonal_sparse, solve_tensorial_sparse, DiagonalSolveResult, SolveDiagnostics,
};
pub use crate::operators::{
    assemble_sparse_diagonal_operators, assemble_sparse_tensorial_operator, SparseDiagonalOperators,
};

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use crate::derivatives;
    use crate::derivatives::Tensor3;
    use crate::eigensolve::sparse_residual_norm;
    use num_complex::Complex64;

    fn sample_tensor(shape: (usize, usize), base: f64) -> Tensor3 {
        let n = shape.0 * shape.1;
        let grid = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let mut tensor: Tensor3 = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for row in 0..3 {
            for col in 0..3 {
                tensor[row][col] = vec![Complex64::new(0.0, 0.0); n];
            }
        }
        tensor[0][0] = grid
            .iter()
            .map(|value| Complex64::new(base + 0.03 * value, 0.0))
            .collect();
        tensor[1][1] = grid
            .iter()
            .map(|value| Complex64::new(base + 0.2 + 0.02 * value, 0.0))
            .collect();
        tensor[2][2] = grid
            .iter()
            .map(|value| Complex64::new(base + 0.6 + 0.01 * value, 0.0))
            .collect();
        tensor
    }

    #[test]
    fn sparse_diagonal_operator_assembly_has_expected_shape() {
        let shape = (3, 4);
        let dlf = (vec![0.17, 0.19, 0.23], vec![0.11, 0.13, 0.17, 0.21]);
        let dlb = (vec![0.16, 0.18, 0.21], vec![0.10, 0.12, 0.15, 0.19]);
        let dmin_pmc = (false, true);
        let sparse_derivatives = derivatives::create_d_matrices_sparse(
            shape,
            (&dlf.0, &dlf.1),
            (&dlb.0, &dlb.1),
            dmin_pmc,
        );
        let eps = sample_tensor(shape, 2.0);
        let mu = sample_tensor(shape, 1.0);

        let sparse = assemble_sparse_diagonal_operators(&eps, &mu, &sparse_derivatives);

        assert_eq!((sparse.mat.rows, sparse.mat.cols), (24, 24));
        assert_eq!((sparse.qmat.rows, sparse.qmat.cols), (24, 24));
        assert_eq!((sparse.q_ep.rows, sparse.q_ep.cols), (24, 24));
        assert!(sparse.mat.nnz() > 0);
        assert!(sparse.qmat.nnz() > 0);
    }

    #[test]
    fn sparse_shift_invert_eigenpairs_have_small_residuals() {
        let shape = (3, 4);
        let dlf = (vec![0.17, 0.19, 0.23], vec![0.11, 0.13, 0.17, 0.21]);
        let dlb = (vec![0.16, 0.18, 0.21], vec![0.10, 0.12, 0.15, 0.19]);
        let sparse_derivatives = derivatives::create_d_matrices_sparse(
            shape,
            (&dlf.0, &dlf.1),
            (&dlb.0, &dlb.1),
            (false, false),
        );
        let eps = sample_tensor(shape, 2.0);
        let mu = sample_tensor(shape, 1.0);
        let sparse = assemble_sparse_diagonal_operators(&eps, &mu, &sparse_derivatives);
        let guess = Complex64::new(-(2.2 * 2.2), 0.0);

        let actual = selected_sparse_shift_invert_eigenpairs(
            &sparse.mat,
            2,
            guess,
            None,
            ShiftInvertOptions {
                krylov_dim: 20,
                tolerance: 1e-11,
            },
        )
        .unwrap();

        assert_eq!(actual.len(), 2);
        for actual in &actual {
            let residual = sparse_residual_norm(&sparse.mat, &actual.vector, actual.value);
            assert!(residual < 1e-6, "residual was {residual}");
        }
    }

    #[cfg(feature = "arpack-backend")]
    #[test]
    fn arpack_shift_invert_eigenpairs_have_small_residuals() {
        let shape = (3, 4);
        let dlf = (vec![0.17, 0.19, 0.23], vec![0.11, 0.13, 0.17, 0.21]);
        let dlb = (vec![0.16, 0.18, 0.21], vec![0.10, 0.12, 0.15, 0.19]);
        let sparse_derivatives = derivatives::create_d_matrices_sparse(
            shape,
            (&dlf.0, &dlf.1),
            (&dlb.0, &dlb.1),
            (false, false),
        );
        let eps = sample_tensor(shape, 2.0);
        let mu = sample_tensor(shape, 1.0);
        let sparse = assemble_sparse_diagonal_operators(&eps, &mu, &sparse_derivatives);
        let guess = Complex64::new(-(2.2 * 2.2), 0.0);

        let actual = selected_sparse_shift_invert_arpack_eigenpairs(
            &sparse.mat,
            2,
            guess,
            None,
            ShiftInvertOptions {
                krylov_dim: 20,
                tolerance: 1e-11,
            },
        )
        .unwrap();

        assert_eq!(actual.len(), 2);
        for actual in &actual {
            let residual = sparse_residual_norm(&sparse.mat, &actual.vector, actual.value);
            assert!(residual < 1e-8, "residual was {residual}");
        }
    }

    #[cfg(all(feature = "arpack-backend", feature = "umfpack-backend"))]
    #[test]
    fn umfpack_shift_invert_eigenpairs_have_small_residuals() {
        let shape = (3, 4);
        let dlf = (vec![0.17, 0.19, 0.23], vec![0.11, 0.13, 0.17, 0.21]);
        let dlb = (vec![0.16, 0.18, 0.21], vec![0.10, 0.12, 0.15, 0.19]);
        let sparse_derivatives = derivatives::create_d_matrices_sparse(
            shape,
            (&dlf.0, &dlf.1),
            (&dlb.0, &dlb.1),
            (false, false),
        );
        let eps = sample_tensor(shape, 2.0);
        let mu = sample_tensor(shape, 1.0);
        let sparse = assemble_sparse_diagonal_operators(&eps, &mu, &sparse_derivatives);
        let guess = Complex64::new(-(2.2 * 2.2), 0.0);

        let actual = selected_sparse_shift_invert_umfpack_eigenpairs(
            &sparse.mat,
            2,
            guess,
            None,
            ShiftInvertOptions {
                krylov_dim: 20,
                tolerance: 1e-11,
            },
        )
        .unwrap();

        assert_eq!(actual.len(), 2);
        for actual in &actual {
            let residual = sparse_residual_norm(&sparse.mat, &actual.vector, actual.value);
            assert!(residual < 1e-8, "residual was {residual}");
        }
    }

    #[test]
    fn sparse_diagonal_solve_recovers_normalized_fields() {
        let shape = (3, 4);
        let dlf = (vec![0.17, 0.19, 0.23], vec![0.11, 0.13, 0.17, 0.21]);
        let dlb = (vec![0.16, 0.18, 0.21], vec![0.10, 0.12, 0.15, 0.19]);
        let sparse_derivatives = derivatives::create_d_matrices_sparse(
            shape,
            (&dlf.0, &dlf.1),
            (&dlb.0, &dlb.1),
            (false, false),
        );
        let eps = sample_tensor(shape, 2.0);
        let mu = sample_tensor(shape, 1.0);
        let cell_areas = test_cell_areas(&dlf.0, &dlf.1);

        let sparse = solve_diagonal_sparse(
            &eps,
            &mu,
            &sparse_derivatives,
            &cell_areas,
            2,
            2.2,
            "+",
            None,
            ShiftInvertOptions {
                krylov_dim: 20,
                tolerance: 1e-11,
            },
        )
        .unwrap();

        assert_eq!(sparse.n_complex.len(), 2);
        assert!(sparse.n_complex[0].re >= sparse.n_complex[1].re);
        for value in &sparse.n_complex {
            assert!(value.re.is_finite());
            assert!(value.im.is_finite());
        }
        for power_norm in &sparse.diagnostics.power_norms {
            assert!((*power_norm - 1.0).abs() < 1e-10);
        }

        for mode_index in 0..2 {
            for component in &sparse.fields {
                assert_eq!(component[mode_index].len(), shape.0 * shape.1);
            }
        }
    }

    fn test_cell_areas(dlf_x: &[f64], dlf_y: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(dlf_x.len() * dlf_y.len());
        for dx in dlf_x {
            for dy in dlf_y {
                out.push(dx * dy);
            }
        }
        out
    }
}
