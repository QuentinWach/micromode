# Changelog

## 0.1.0a4 - Unreleased

- Fixed y-normal field mapping so returned global fields use a right-handed
  basis and physical +y power overlap is positive.
- Added a regression test for y-normal power-overlap sign.

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
