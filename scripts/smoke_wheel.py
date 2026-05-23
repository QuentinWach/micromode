"""Smoke-test an installed MicroMode wheel in a clean environment."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import micromode as mm


def main() -> None:
    """Solve a tiny problem and round-trip HDF5 in an installed wheel."""
    x_edges = np.linspace(-0.8, 0.8, 7)
    y_edges = np.linspace(-0.6, 0.6, 6)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    eps = np.where((np.abs(xx) < 0.28) & (np.abs(yy) < 0.22), 3.4**2, 1.44**2)

    result = mm.solve_grid(
        eps_xx=eps,
        x_edges=tuple(float(value) for value in x_edges),
        y_edges=tuple(float(value) for value in y_edges),
        wavelength=1.55,
        num_modes=1,
        target_neff=2.4,
        krylov_dim=16,
    )
    if result.n_eff.shape != (1, 1):
        raise RuntimeError(f"unexpected n_eff shape: {result.n_eff.shape}")
    if not np.isfinite(result.n_eff.values).all():
        raise RuntimeError("n_eff contains non-finite values")
    if abs(result.overlap(mode_index=0, kind="power") - 1.0) > 1e-8:
        raise RuntimeError("power-normalized self overlap is not one")

    with tempfile.TemporaryDirectory() as tmp:
        path = result.to_hdf5(Path(tmp) / "smoke.h5")
        loaded = mm.Result.from_hdf5(path)
        np.testing.assert_allclose(loaded.n_eff.values, result.n_eff.values)

    print(f"micromode smoke test passed; n_eff={result.n_eff.values[0, 0]:.6f}")


if __name__ == "__main__":
    main()
