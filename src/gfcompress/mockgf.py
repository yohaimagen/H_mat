"""Analytic-kernel mock Green's-function operator (Task 2.2).

`MockGF` is a `MatVecOperator` (Task 2.1) backed by a smooth, tensor-valued
kernel `K(x_i, x_j) -> R^{dof_row x dof_col}` evaluated between patch
centroids of a `FaultMesh`. It plays the role of the elastostatic Green's
function in tests: it has genuine, smoothly-decaying singular values on
well-separated (admissible) blocks, and is not low rank on near-diagonal
blocks -- exactly the structure a real elastostatic kernel has, and exactly
what is needed to validate a compressor (per CLAUDE.md, a random dense matrix
cannot do this).

Kernel
------
For source point `y` and target point `x` with separation vector `r_vec = x -
y` and distance `r = ||r_vec||`, a simplified Kelvin-type tensor is used:

    K(x, y) = 1 / (r + eps)^d * (I_d + r_vec r_vec^T / (r + eps)^2)[:, :dof_col]

i.e. the usual "isotropic + dipole" Kelvin combination, scaled to decay like
`1/r^d` (matching the `1/r^d` far-field decay assumed by `suggest_eta` in
`gfcompress.interactions`), with a small regularization `eps` (proportional to
the patches' characteristic lengths) avoiding the `r -> 0` singularity for
coincident centroids. Only the first `dof_col = d - 1` columns of the `d x d`
tensor are kept, giving the `dof_row x dof_col` block required by the
patch-major flattening convention (`dof_row = d`, `dof_col = d - 1`).

Assembly
--------
`MockGF` assembles the full dense, patch-major-flattened operator
`A`, shape `(dof_row * N, dof_col * N)` (`2N x N` in 2D, `3N x 2N` in 3D), once
at construction time, and implements `matvec`/`rmatvec`/`shape` via this dense
array. `block(row_patches, col_patches)` extracts the dense sub-block
`A[I_rows, I_cols]` (`I_rows`/`I_cols` the flattened row/column indices of the
given patch subsets) for ground-truth tests -- e.g. checking that admissible
(well-separated) blocks are numerically low rank while near-diagonal blocks
are not.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.operators import MatVecOperator


def kernel_block(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dof_row: int,
    dof_col: int,
    eps: float = 1e-3,
) -> NDArray[np.float64]:
    """Evaluate the smooth tensor kernel `K(x, y) -> R^{dof_row x dof_col}`.

    A simplified Kelvin-type elastostatic kernel: with `r_vec = x - y`,
    `r = ||r_vec||`,

        K(x, y) = (I_d + r_vec r_vec^T / (r + eps)^2) / (r + eps)^d

    truncated to its first `dof_col` columns. Decays like `1/r^d` for `r >>
    eps`; `eps` regularizes the `r -> 0` (coincident-point) singularity.

    Args:
        x: Target point, shape `(d,)`, `d in {2, 3}`.
        y: Source point, shape `(d,)`, same `d` as `x`.
        dof_row: Number of output rows, must equal `d`.
        dof_col: Number of output columns, must equal `d - 1`.
        eps: Regularization length scale, `> 0`.

    Returns:
        The `(dof_row, dof_col)` kernel block `K(x, y)`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x.shape[0]
    if y.shape[0] != d:
        raise ValueError(f"x and y must have the same dimension, got {x.shape} and {y.shape}")
    if dof_row != d:
        raise ValueError(f"dof_row must equal d={d}, got {dof_row}")
    if dof_col != d - 1:
        raise ValueError(f"dof_col must equal d-1={d - 1}, got {dof_col}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    r_vec = x - y
    r = float(np.linalg.norm(r_vec))
    denom = r + eps

    tensor = np.eye(d) + np.outer(r_vec, r_vec) / (denom * denom)
    tensor = tensor / (denom**d)
    block: NDArray[np.float64] = tensor[:, :dof_col]
    return block


class MockGF(MatVecOperator):
    """Dense-assembled mock Green's-function operator over a `FaultMesh`.

    Implements `MatVecOperator` (Task 2.1) by assembling, once at
    construction time, the dense patch-major-flattened operator `A` of shape
    `(dof_row * N, dof_col * N)` whose `(dof_row, dof_col)` patch blocks are
    `kernel_block(centroids[i], centroids[j], dof_row, dof_col, eps)`.

    Attributes:
        mesh: The underlying `FaultMesh`.
        eps: Regularization length scale used by `kernel_block`.
        A: The dense assembled operator, shape `(mesh.n_rows, mesh.n_cols)`.
    """

    def __init__(self, mesh: FaultMesh, eps: float = 1e-3) -> None:
        """Assemble the dense mock operator over `mesh`.

        Args:
            mesh: `FaultMesh` providing the `N` patch centroids, `dof_row`,
                and `dof_col`.
            eps: Regularization length scale passed to `kernel_block`.
                Defaults to `1e-3`.
        """
        self.mesh = mesh
        self.eps = eps
        self.A: NDArray[np.float64] = _assemble(mesh, eps)

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `A @ omega`. See `MatVecOperator.matvec`."""
        return self.A @ omega

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `A.conj().T @ psi`. See `MatVecOperator.rmatvec`."""
        a_adj: NDArray[np.floating] = self.A.conj().T
        return a_adj @ psi

    @property
    def shape(self) -> tuple[int, int]:
        """`(dof_row * N, dof_col * N)`: `2N x N` in 2D, `3N x 2N` in 3D."""
        return (self.A.shape[0], self.A.shape[1])

    def block(
        self,
        row_patches: NDArray[np.integer],
        col_patches: NDArray[np.integer],
    ) -> NDArray[np.float64]:
        """Direct ground-truth accessor: the dense sub-block `A[I_rows,
        I_cols]`.

        Args:
            row_patches: Integer array of patch indices selecting rows.
            col_patches: Integer array of patch indices selecting columns.

        Returns:
            Dense array of shape `(dof_row * len(row_patches), dof_col *
            len(col_patches))`, the sub-block of `A` corresponding to the
            flattened row/column indices of `row_patches`/`col_patches`
            (patch-major, block-interleaved, via
            `FaultMesh.patch_to_rows`/`patch_to_cols`).
        """
        rows = self.mesh.patch_to_rows(np.asarray(row_patches, dtype=np.intp))
        cols = self.mesh.patch_to_cols(np.asarray(col_patches, dtype=np.intp))
        return self.A[np.ix_(rows, cols)]


def _assemble(mesh: FaultMesh, eps: float) -> NDArray[np.float64]:
    """Assemble the dense patch-major-flattened operator `A` for `mesh`.

    Args:
        mesh: The `FaultMesh` providing centroids, `dof_row`, `dof_col`.
        eps: Regularization length scale passed to `kernel_block`.

    Returns:
        Dense array of shape `(mesh.n_rows, mesh.n_cols)`.
    """
    n = mesh.n_patches
    dof_row = mesh.dof_row
    dof_col = mesh.dof_col
    a = np.empty((mesh.n_rows, mesh.n_cols), dtype=np.float64)
    for i in range(n):
        x = mesh.centroids[i]
        row_slice = slice(dof_row * i, dof_row * (i + 1))
        for j in range(n):
            y = mesh.centroids[j]
            col_slice = slice(dof_col * j, dof_col * (j + 1))
            a[row_slice, col_slice] = kernel_block(x, y, dof_row, dof_col, eps)
    return a
