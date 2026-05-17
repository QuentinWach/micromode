use micromode_core::derivatives::{self, Tensor3};
use micromode_core::operators::assemble_sparse_diagonal_operators;
use micromode_core::sparse_matrix::SparseMatrix;
use num_complex::Complex64;

pub fn strip_operator(nx: usize, ny: usize, target_neff: f64) -> (SparseMatrix, Complex64) {
    let x_edges = linspace(-1.2, 1.2, nx + 1);
    let y_edges = linspace(-0.8, 0.8, ny + 1);
    let dlf_x = steps(&x_edges);
    let dlf_y = steps(&y_edges);
    let dlb_x = dlf_x.clone();
    let dlb_y = dlf_y.clone();
    let derivatives = derivatives::create_d_matrices_sparse(
        (nx, ny),
        (&dlf_x, &dlf_y),
        (&dlb_x, &dlb_y),
        (false, false),
    );
    let eps = strip_tensor(nx, ny, &x_edges, &y_edges);
    let mu = uniform_tensor(nx * ny, 1.0);
    let operators = assemble_sparse_diagonal_operators(&eps, &mu, &derivatives);
    let guess = Complex64::new(-(target_neff * target_neff), 0.0);
    (operators.mat, guess)
}

pub fn parse_grids() -> Vec<(usize, usize)> {
    let args = std::env::args().collect::<Vec<_>>();
    let mut grids = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--grid" {
            if let Some(value) = args.get(index + 1) {
                if let Some(grid) = parse_grid(value) {
                    grids.push(grid);
                }
            }
            index += 1;
        }
        index += 1;
    }
    if grids.is_empty() {
        vec![(10, 8), (16, 12), (24, 18)]
    } else {
        grids
    }
}

pub fn parse_usize_arg(name: &str) -> Option<usize> {
    parse_arg(name).and_then(|value| value.parse().ok())
}

pub fn parse_f64_arg(name: &str) -> Option<f64> {
    parse_arg(name).and_then(|value| value.parse().ok())
}

fn strip_tensor(nx: usize, ny: usize, x_edges: &[f64], y_edges: &[f64]) -> Tensor3 {
    let mut tensor = uniform_tensor(nx * ny, 1.44 * 1.44);
    for ix in 0..nx {
        let x = 0.5 * (x_edges[ix] + x_edges[ix + 1]);
        for iy in 0..ny {
            let y = 0.5 * (y_edges[iy] + y_edges[iy + 1]);
            let eps = if x.abs() <= 0.25 && y.abs() <= 0.11 {
                3.48 * 3.48
            } else {
                1.44 * 1.44
            };
            let index = ix * ny + iy;
            for component in 0..3 {
                tensor[component][component][index] = Complex64::new(eps, 0.0);
            }
        }
    }
    tensor
}

fn uniform_tensor(n: usize, diagonal: f64) -> Tensor3 {
    let mut tensor: Tensor3 = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for row in 0..3 {
        for col in 0..3 {
            tensor[row][col] = vec![Complex64::new(0.0, 0.0); n];
        }
    }
    for component in 0..3 {
        tensor[component][component] = vec![Complex64::new(diagonal, 0.0); n];
    }
    tensor
}

fn linspace(start: f64, stop: f64, len: usize) -> Vec<f64> {
    let step = (stop - start) / (len - 1) as f64;
    (0..len).map(|index| start + step * index as f64).collect()
}

fn steps(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect()
}

fn parse_grid(value: &str) -> Option<(usize, usize)> {
    let (left, right) = value.split_once('x')?;
    Some((left.parse().ok()?, right.parse().ok()?))
}

fn parse_arg(name: &str) -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}
