# Changelog

## 0.1.0a4 - 2026-05-20

- Fixed y-normal field mapping so returned global fields use a right-handed
  basis and physical +y power overlap is positive.
- Added regression coverage for the y-normal power-overlap sign and tightened
  API type-checking coverage.
- Added a Tidy3D modal sources and monitors example with README documentation
  and refreshed generated documentation assets.
- Updated the ridge-waveguide README example to show an angled slab on a
  substrate with complex fundamental mode fields.
- Added stricter lint, type-check, and release metadata checks to CI.
- Hardened example input validation, HDF5 solver metadata loading, and sparse
  solver edge-case handling.

## 0.1.0a3

Initial alpha release candidate.

- Added grid-first Python API for rasterized mode solving.
- Added Rust sparse shift-invert solver backend.
- Added a portable native Rust sparse eigensolver path with AMD ordering, packed
  LU solves, adaptive Arnoldi stopping, and no external solver-stack dependency.
- Added 2D cross-section solves and 1D slice solves.
- Added scalar, diagonal anisotropic, and tensor material grids.
- Added six-component field reconstruction.
- Added deterministic phase convention, unit-power normalization, and Lorentz orthogonalization.
- Added mode metrics, overlaps, plotting helpers, HDF5 save/load, and dataframe export.
- Added ridge-waveguide README example and generated plot asset.
- Added Apache-2.0 license, CI, coverage, and release packaging setup.
