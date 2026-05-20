# Physics Model

MicroMode solves source-free, frequency-domain Maxwell's equations on a
rasterized mode plane, following the same FDFD starting point used by
MaxwellFDFD [1]:

```math
\nabla \times \mathbf{E}(\mathbf{r})
=
-i\omega\mu(\mathbf{r},\omega)\mathbf{H}(\mathbf{r}),
\qquad
\nabla \times \mathbf{H}(\mathbf{r})
=
i\omega\epsilon(\mathbf{r},\omega)\mathbf{E}(\mathbf{r}).
```

Here:

- $\mathbf{r}$ is position in the local mode-coordinate system;
- $\mathbf{E}$ and $\mathbf{H}$ are the electric and magnetic mode fields;
- $\omega$ is the angular frequency;
- $\epsilon$ and $\mu$ are the supplied material tensors.

Unlike a driven FDFD field solve, MicroMode is a mode solver: there are no
electric or magnetic current sources. It assumes fields vary along the local
propagation axis as

```math
\mathbf{E}(x, y, z) = \mathbf{e}(x, y) e^{i k_0 n_\mathrm{eff} z},
\qquad
\mathbf{H}(x, y, z) = \mathbf{h}(x, y) e^{i k_0 n_\mathrm{eff} z},
```

where $k_0 = 2\pi / \lambda_0$ and $n_\mathrm{eff}$ is the unknown complex
effective index. The transverse fields are discretized by the
finite-difference frequency-domain method on a regular Yee grid [2].

## Discretization

The Rust kernels use relative material tensors $\epsilon_r(x,y)$,
$\mu_r(x,y)$ and scale transverse derivatives by $1/k_0$, so the sparse
operators are dimensionless. On the local Yee grid, the four derivative
matrices are

```math
D_{xf}, D_{xb}, D_{yf}, D_{yb}
\approx
\frac{1}{k_0}\partial_x^\mathrm{forward/backward},
\frac{1}{k_0}\partial_y^\mathrm{forward/backward}.
```

Low-edge PEC/PMC boundary settings modify the derivative stencils, and PMLs
premultiply derivatives by complex stretch matrices:

```math
D \leftarrow S^{-1}D,\qquad
s(u) = \kappa(u) + i\frac{\sigma(u)}{\omega\epsilon_0}.
```

The stretch profiles are polynomial functions controlled by `PmlSpec`. The
stretched-coordinate PML form follows the frequency-domain Maxwell literature
summarized by Shin and Fan [3].

## Diagonal Materials

For diagonal material tensors, MicroMode reduces Maxwell's equations to a
transverse electric eigenproblem. With

```math
\mathbf{e}_t =
\begin{bmatrix} E_x \\ E_y \end{bmatrix},
\qquad
A_\mathrm{diag} =
P_\mu Q + P_\partial Q_\epsilon,
```

the solved eigenproblem is

```math
A_\mathrm{diag}\mathbf{e}_t = -n_\mathrm{eff}^2 \mathbf{e}_t.
```

The block operators are assembled from the Yee derivatives and diagonal tensor
components:

```math
P_\mu =
\begin{bmatrix}
0 & \mu_{yy} \\
-\mu_{xx} & 0
\end{bmatrix},
\qquad
Q_\epsilon =
\begin{bmatrix}
0 & \epsilon_{yy} \\
-\epsilon_{xx} & 0
\end{bmatrix},
```

```math
P_\partial =
\begin{bmatrix}
-D_{xf}\epsilon_{zz}^{-1}D_{yb} & D_{xf}\epsilon_{zz}^{-1}D_{xb} \\
-D_{yf}\epsilon_{zz}^{-1}D_{yb} & D_{yf}\epsilon_{zz}^{-1}D_{xb}
\end{bmatrix},
```

```math
Q_\partial =
\begin{bmatrix}
-D_{xb}\mu_{zz}^{-1}D_{yf} & D_{xb}\mu_{zz}^{-1}D_{xf} \\
-D_{yb}\mu_{zz}^{-1}D_{yf} & D_{yb}\mu_{zz}^{-1}D_{xf}
\end{bmatrix},
\qquad
Q = Q_\epsilon + Q_\partial.
```

After the transverse solve, the remaining field components are reconstructed
from the curl equations:

```math
\begin{bmatrix} H_x \\ H_y \end{bmatrix}
\propto
\frac{1}{i n_\mathrm{eff}}Q\mathbf{e}_t,
\qquad
H_z \propto \mu_{zz}^{-1}(D_{xf}E_y - D_{yf}E_x),
```

```math
E_z \propto \epsilon_{zz}^{-1}(D_{xb}H_y - D_{yb}H_x).
```

## Tensorial Materials

For full tensor media, including off-diagonal $\epsilon$/$\mu$ terms and
angle or bend coordinate transforms, MicroMode switches to a first-order
tensorial eigenproblem:

```math
A_\mathrm{tensor}
\begin{bmatrix} E_x \\ E_y \\ H_x \\ H_y \end{bmatrix}
=
n_\mathrm{eff}
\begin{bmatrix} E_x \\ E_y \\ H_x \\ H_y \end{bmatrix}.
```

The longitudinal tensor couplings are eliminated through local Schur
complements such as

```math
\epsilon^{(s)}_{\alpha\beta}
=
\epsilon_{\alpha\beta}
-
\epsilon_{\alpha z}\epsilon_{z\beta}/\epsilon_{zz},
\qquad
\mu^{(s)}_{\alpha\beta}
=
\mu_{\alpha\beta}
-
\mu_{\alpha z}\mu_{z\beta}/\mu_{zz},
```

then $E_z$ and $H_z$ are reconstructed with the off-diagonal coupling terms
included. This is the path used automatically for `Materials.from_components`,
angled solves, and bend solves whenever the transformed tensors are no longer
diagonal.

## References

[1] W. Shin, [MaxwellFDFD webpage](https://www.mit.edu/~wsshin/maxwellfdfd.html), 2015.

[2] K. S. Yee, "Numerical solution of initial boundary value problems involving
Maxwell's equations in isotropic media," *IEEE Transactions on Antennas and
Propagation*, vol. 14, no. 3, pp. 302-307, 1966.
doi:[10.1109/TAP.1966.1138693](https://doi.org/10.1109/TAP.1966.1138693).

[3] W. Shin and S. Fan, "Choice of the perfectly matched layer boundary
condition for frequency-domain Maxwell's equations solvers," *Journal of
Computational Physics*, vol. 231, no. 8, pp. 3406-3431, 2012.
doi:[10.1016/j.jcp.2012.01.013](https://doi.org/10.1016/j.jcp.2012.01.013).
