use std::hint::black_box;
use std::time::{Duration, Instant};

use micromode_core::eigensolve::{
    selected_sparse_shift_invert_native_eigenpairs, Eigenpair, ShiftInvertOptions,
};

mod eigensolver_benchmark_problem;
use eigensolver_benchmark_problem::{parse_f64_arg, parse_grids, parse_usize_arg, strip_operator};

fn main() -> Result<(), String> {
    let grids = parse_grids();
    let repeats = parse_usize_arg("--repeats").unwrap_or(5);
    let krylov_dim = parse_usize_arg("--krylov-dim").unwrap_or(40);
    let num_modes = parse_usize_arg("--num-modes").unwrap_or(2);
    let target_neff = parse_f64_arg("--target-neff").unwrap_or(2.5);

    println!("backend,grid,operator_size,operator_nnz,repeats,best_ms,mean_ms,max_residual");
    for (nx, ny) in grids {
        let (mat, guess) = strip_operator(nx, ny, target_neff);
        let options = ShiftInvertOptions {
            krylov_dim,
            tolerance: 1e-10,
        };

        let native = benchmark_backend(repeats, || {
            selected_sparse_shift_invert_native_eigenpairs(
                &mat,
                num_modes,
                guess,
                None,
                options.clone(),
            )
        })?;

        print_row("native", nx, ny, mat.rows, mat.nnz(), repeats, &native);
    }
    Ok(())
}

struct BenchResult {
    durations: Vec<Duration>,
    pairs: Vec<Eigenpair>,
}

fn benchmark_backend<F>(repeats: usize, mut solve: F) -> Result<BenchResult, String>
where
    F: FnMut() -> Result<Vec<Eigenpair>, String>,
{
    let warmup = solve()?;
    black_box(&warmup);

    let mut durations = Vec::with_capacity(repeats);
    let mut pairs = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        pairs = solve()?;
        durations.push(start.elapsed());
        black_box(&pairs);
    }
    Ok(BenchResult { durations, pairs })
}

fn print_row(
    backend: &str,
    nx: usize,
    ny: usize,
    operator_size: usize,
    operator_nnz: usize,
    repeats: usize,
    result: &BenchResult,
) {
    let best = result.durations.iter().min().copied().unwrap_or_default();
    let mean = result
        .durations
        .iter()
        .map(Duration::as_secs_f64)
        .sum::<f64>()
        / repeats as f64;
    let max_residual = result
        .pairs
        .iter()
        .map(|pair| pair.residual)
        .fold(0.0, f64::max);
    println!(
        "{backend},{nx}x{ny},{operator_size},{operator_nnz},{repeats},{:.3},{:.3},{:.3e}",
        best.as_secs_f64() * 1000.0,
        mean * 1000.0,
        max_residual,
    );
}
