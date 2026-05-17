# Mode Solver Methods

`solve_modes(...)` is the main entry point. It validates the `Materials` grid,
resolves frequencies or wavelengths, builds the Yee derivative matrices, chooses
the diagonal or tensorial Rust backend, solves one frequency at a time, and
returns a coordinate-aware `Result`.

## Material Grids

- `Materials.from_diagonal(...)` builds scalar or diagonal-anisotropic grids.
  These usually use the faster diagonal sparse formulation.
- `Materials.from_components(...)` accepts full \(3\times3\) material tensors.
  Any off-diagonal component routes the solve through the tensorial sparse
  formulation.
- `Materials.from_slice(...)` creates a 1D mode-plane slice with an invariant
  width so integrations and overlaps still have physical weights.
- `Materials.from_subpixel_diagonal(...)` downsamples a high-resolution
  diagonal raster with arithmetic, harmonic, geometric, min, or max averaging.

## Solver Controls

- `num_modes`: number of modes returned near the requested target.
- `target_neff`: center of the shift-invert search. If omitted, MicroMode uses
  the largest local material index as a practical guided-mode default.
- `pml`: absorbing boundary thickness and stretch profile via `PmlSpec`.
- `boundary`: low-edge PEC/PMC symmetry settings via `BoundarySpec`.
- `direction`: `"+"` or `"-"` propagation; the backward solve flips the
  appropriate magnetic and longitudinal electric signs.
- `components`: optional subset of returned field components.
- `krylov_dim`: dimension of the Arnoldi search space.
- `angle_theta`, `angle_phi`, `bend_radius`, `bend_axis`: transformation-optics
  controls that update \(\epsilon\) and \(\mu\) before the sparse solve.

## Eigenpair Selection

Internally, eigenpairs are selected with sparse shift-invert Arnoldi. For a
matrix \(A\) and shift \(\sigma\), Arnoldi is applied to

$$
(A-\sigma I)^{-1},
\qquad
\lambda = \sigma + 1/\theta,
$$

where \(\theta\) is a Ritz value of the inverse-shifted operator. The diagonal
backend uses \(\sigma=-\texttt{target_neff}^2\); the tensorial backend uses
\(\sigma=\texttt{target_neff}\).

Returned modes are sorted by decreasing real effective index, normalized to
unit transverse power,

$$
\int (\mathbf{E}\times\mathbf{H}^*)\cdot\hat{\mathbf{n}}\,dA,
$$

and orthogonalized with the unconjugated Lorentz product

$$
L(a,b)=\frac{1}{2}\int
\left[(\mathbf{E}_a\times\mathbf{H}_b)
+(\mathbf{E}_b\times\mathbf{H}_a)\right]\cdot\hat{\mathbf{n}}\,dA.
$$

## Result Helpers

`Result` exposes the post-processing methods users normally need: `n_eff`,
`k_eff`, `mode_area`, `pol_fraction`, `pol_fraction_waveguide`, `modes_info`,
`to_dataframe()`, `overlap()`, `overlap_matrix()`, `plot_field()`,
`plot_field_components()`, `to_hdf5()`, and `Result.from_hdf5()`.
