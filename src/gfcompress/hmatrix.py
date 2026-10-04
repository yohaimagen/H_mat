"""`HMatrix` container: `dot`/`rdot` and `block_partition` (Task 5.5).

An `HMatrix` is the final output of Algorithm 4.1's level loop plus Task
5.4's leaf extraction: a flat `gfcompress.peeling.Factors` list (one
`BlockFactor` per admissible pair across every level `2, ..., L`) and a flat
list of `gfcompress.leaf.DenseLeaf`s (one per inadmissible neighbor pair at
the leaf level `L`). Together these two lists cover every `(alpha, beta)`
patch-pair exactly once (`block_partition` below is the test hook for that
invariant).

`dot(x)`/`rdot(y)` apply the compressed operator through those factors only
-- never assembling a dense matrix. Peeling constructs each admissible block
as `U (B (V* x))`, then finalization stores the identical `U_tilde (V* x)`
form after absorbing `B` into `U`. The finished H-matrix therefore retains no
square core and never materializes the dense `(dof_row*|alpha|) x
(dof_col*|beta|)` far-field block (`gfcompress.peeling.apply_truncated`/
`apply_truncated_T` implement this; `HMatrix` adds dense leaves only). Per
CLAUDE.md, `dot: R^{dof_col*N} -> R^{dof_row*N}` and `rdot` goes the other
way (`2N x N` in 2D, `3N x 2N` in 3D) -- the two are never interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.leaf import DenseLeaf
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import Factors, apply_truncated, apply_truncated_T
from gfcompress.tree import TreeNode


@dataclass
class HMatrix(MatVecOperator):
    """Compressed operator: admissible-block factors + dense leaf blocks.

    Attributes:
        root: Root of the geometric cluster tree shared by rows and columns.
        mesh: The `FaultMesh` underlying `root` (supplies `n_rows`/`n_cols`).
        factors: Flat list of `BlockFactor`s, one per admissible pair across
            every level `2, ..., L`.
        leaves: Flat list of `DenseLeaf`s, one per inadmissible neighbor pair
            at the leaf level `L`.
    """

    root: TreeNode
    mesh: FaultMesh
    factors: Factors = field(default_factory=list)
    leaves: list[DenseLeaf] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        """`(mesh.n_rows, mesh.n_cols)`: `2N x N` in 2D, `3N x 2N` in 3D."""
        return (self.mesh.n_rows, self.mesh.n_cols)

    def dot(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the compressed operator: `A_H @ x`.

        Args:
            x: Array of shape `(n_cols,)` or `(n_cols, w)`, in `A`'s domain
                (indexed by `beta.col_indices` for every stored block).

        Returns:
            Array of shape `(n_rows,)` or `(n_rows, w)` matching `x`'s
            trailing shape: the sum of every admissible block's finalized
            `U_tilde(V* x)` contribution (via `apply_truncated`) plus every dense leaf's
            `block @ x` contribution, scattered into their respective
            `alpha.row_indices`.
        """
        x = np.asarray(x, dtype=np.float64)
        result = apply_truncated(self.factors, x, self.mesh.n_rows)
        for leaf in self.leaves:
            x_beta = x[leaf.beta.col_indices, ...]
            result[leaf.alpha.row_indices, ...] += leaf.block @ x_beta
        return result

    def rdot(self, y: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the compressed operator's conjugate transpose: `A_H* @ y`.

        Args:
            y: Array of shape `(n_rows,)` or `(n_rows, w)`, in `A`'s range
                (indexed by `alpha.row_indices` for every stored block).

        Returns:
            Array of shape `(n_cols,)` or `(n_cols, w)` matching `y`'s
            trailing shape: the sum of every admissible block's finalized
            `V(U_tilde* y)` contribution (via `apply_truncated_T`) plus every
            dense leaf's `block* @ y` contribution, scattered into their
            respective `beta.col_indices`.
        """
        y = np.asarray(y, dtype=np.float64)
        result = apply_truncated_T(self.factors, y, self.mesh.n_cols)
        for leaf in self.leaves:
            y_alpha = y[leaf.alpha.row_indices, ...]
            result[leaf.beta.col_indices, ...] += leaf.block.conj().T @ y_alpha
        return result

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """`MatVecOperator` interface: same as `dot`."""
        return self.dot(omega)

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """`MatVecOperator` interface: same as `rdot`."""
        return self.rdot(psi)

    def block_partition(self) -> list[frozenset[tuple[int, int]]]:
        """The patch-pair set covered by every stored block.

        Returns:
            One `frozenset` of `(row_patch, col_patch)` pairs per stored
            block (factors first, then leaves), each pair set being the
            Cartesian product `alpha.patch_indices x beta.patch_indices` for
            that block's `(alpha, beta)`. A correctly built `HMatrix` covers
            every pair in `{0, ..., N-1} x {0, ..., N-1}` exactly once across
            the whole list (complete, disjoint cover, per CLAUDE.md).
        """
        return [_pair_set(f.alpha, f.beta) for f in self.factors] + [
            _pair_set(leaf.alpha, leaf.beta) for leaf in self.leaves
        ]


def _pair_set(alpha: TreeNode, beta: TreeNode) -> frozenset[tuple[int, int]]:
    """The Cartesian product of two boxes' patch indices, as `(row, col)`
    integer pairs."""
    return frozenset((int(i), int(j)) for i in alpha.patch_indices for j in beta.patch_indices)


__all__ = ["HMatrix"]
