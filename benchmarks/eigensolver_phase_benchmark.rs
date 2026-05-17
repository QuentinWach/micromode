use std::hint::black_box;
use std::time::{Duration, Instant};

use micromode_core::eigensolve::{
    profile_sparse_shift_invert_native_eigenpairs, ShiftInvertOptions, ShiftInvertProfile,
};

mod eigensolver_benchmark_problem;
use eigensolver_benchmark_problem::{parse_f64_arg, parse_grids, parse_usize_arg, strip_operator};

fn main() -> Result<(), String> {
    let grids = parse_grids();
    let repeats = parse_usize_arg("--repeats").unwrap_or(3);
    let krylov_dim = parse_usize_arg("--krylov-dim").unwrap_or(56);
    let num_modes = parse_usize_arg("--num-modes").unwrap_or(2);
    let target_neff = parse_f64_arg("--target-neff").unwrap_or(2.5);

    println!(
        "grid,operator_size,operator_nnz,repeat,assembly_ms,total_ms,shift_diagonal_ms,amd_ordering_ms,lu_factorization_ms,linear_solves_ms,arnoldi_orthogonalization_ms,hessenberg_eigensolve_ms,ritz_reconstruction_ms,residuals_ms,sorting_ms,solve_calls,arnoldi_steps,candidate_count,returned_pairs,lu_l_nnz,lu_u_nnz,lu_fill_ratio,max_residual"
    );
    for (nx, ny) in grids {
        let (warm_mat, warm_guess) = strip_operator(nx, ny, target_neff);
        let warm_profile = profile_sparse_shift_invert_native_eigenpairs(
            &warm_mat,
            num_modes,
            warm_guess,
            None,
            ShiftInvertOptions {
                krylov_dim,
                tolerance: 1e-10,
            },
        )?;
        black_box(&warm_profile);

        for repeat in 0..repeats {
            let assembly_start = Instant::now();
            let (mat, guess) = strip_operator(nx, ny, target_neff);
            let assembly = assembly_start.elapsed();
            let profile = profile_sparse_shift_invert_native_eigenpairs(
                &mat,
                num_modes,
                guess,
                None,
                ShiftInvertOptions {
                    krylov_dim,
                    tolerance: 1e-10,
                },
            )?;
            black_box(&profile);
            print_row(nx, ny, repeat, mat.rows, mat.nnz(), assembly, &profile);
        }
    }
    Ok(())
}

fn print_row(
    nx: usize,
    ny: usize,
    repeat: usize,
    operator_size: usize,
    operator_nnz: usize,
    assembly: Duration,
    profile: &ShiftInvertProfile,
) {
    let lu_nnz = profile.lu_l_nnz + profile.lu_u_nnz;
    let fill_ratio = lu_nnz as f64 / operator_nnz.max(1) as f64;
    println!(
        "{}x{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{:.3},{:.3e}",
        nx,
        ny,
        operator_size,
        operator_nnz,
        repeat,
        ms(assembly),
        ms(profile.total),
        ms(profile.shift_diagonal),
        ms(profile.amd_ordering),
        ms(profile.lu_factorization),
        ms(profile.linear_solves),
        ms(profile.arnoldi_orthogonalization),
        ms(profile.hessenberg_eigensolve),
        ms(profile.ritz_reconstruction),
        ms(profile.residuals),
        ms(profile.sorting),
        profile.solve_calls,
        profile.arnoldi_steps,
        profile.candidate_count,
        profile.returned_pairs,
        profile.lu_l_nnz,
        profile.lu_u_nnz,
        fill_ratio,
        profile.max_residual,
    );
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
