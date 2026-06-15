"""Per-level column bases `U_{alpha,beta}` for admissible blocks (Task 5.2).

This is the first of the three per-level compression passes of Stage 5
(`column_bases` here, `row_bases` in Task 5.3, `core_matrices` in Task 5.4).
For tree level `l`, with the low-rank factors of levels `2, ..., l-1` already
stored as a `gfcompress.peeling.Factors` list:

1. Build the level's fixed `6x...x6` periodic admissible test matrices
   `Omega` (Task 4.2, `gfcompress.fixed_pattern.build_admissible_test_matrices`)
   -- per CLAUDE.md, these satisfy the Eq. 4.4 sampling constraints for every
   admissible pair `(alpha, beta)` at `l` and are reused verbatim, not
   reinvented.
2. For each such `Omega`, compute the level-`l` column sample
   `Y = (A - A^{(l-1)}) @ Omega` via `gfcompress.peeling.peeled_matvec`
   (Task 5.1).
3. For every admissible pair `(alpha, beta)` at `l` (from `L^int`, Task 1.5),
   look up the unique `Omega`/`Y` whose active boxes include `beta` (Eq. 4.4
   guarantees `beta`'s columns of that `Omega` are an independent Gaussian
   block and every other box interacting with `alpha` is zeroed there),
   restrict to `alpha`'s row-index set `I_alpha = alpha.row_indices`
   (`Y(I_alpha, :)`), and set `U_{alpha,beta} = qr(Y(I_alpha, :), k)`
   (`gfcompress.randomized.orth`, Task 3.1) -- an orthonormal basis for the
   block's (approximate rank-`k`) column space.

Per CLAUDE.md's shape conventions, `Y` lives in `A`'s range
(`R^{dof_row * N}`), so `I_alpha` must be `alpha`'s `dof_row`-expanded
`row_indices`; `U_{alpha,beta}` has shape `(len(alpha.row_indices), k)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gfcompress.fixed_pattern import build_admissible_test_matrices
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import interaction_lists
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import Factors, peeled_matvec
from gfcompress.randomized import orth
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class ColumnBasis:
    """Column-space basis `U_{alpha,beta}` for one admissible block `(alpha,
    beta)` at a single level.

    Attributes:
        alpha: The row box. `alpha.row_indices` (length `dof_row * |alpha|`)
            indexes the global row space `{0, ..., n_rows - 1}`.
        beta: The col box. `beta.col_indices` (length `dof_col * |beta|`)
            indexes the global col space `{0, ..., n_cols - 1}`.
        u: Orthonormal column-space basis `U_{alpha,beta}`, shape
            `(len(alpha.row_indices), k)`, satisfying `u.conj().T @ u ~= I_k`.
    """

    alpha: TreeNode
    beta: TreeNode
    u: NDArray[np.float64]


def column_bases(
    operator: MatVecOperator,
    root: TreeNode,
    mesh: FaultMesh,
    level: int,
    factors: Factors,
    k: int,
    p: int = 0,
    seed: int | None = None,
) -> list[ColumnBasis]:
    """Compute the level-`level` column bases `U_{alpha,beta}` for every
    admissible pair `(alpha, beta)` at `level`.

    Args:
        operator: The black-box operator `A` (accessed only via
            `operator.matvec`, through `peeled_matvec`).
        root: Root of the geometric cluster tree.
        mesh: The `FaultMesh` underlying `root` (provides `n_cols` for sizing
            the test matrices).
        level: The tree level whose admissible pairs (`L^int`, Task 1.5) are
            processed.
        factors: Flat list of `BlockFactor`s for levels `2, ..., level - 1`
            (Task 5.1's `peeled_matvec` subtracts their contribution before
            sampling). Empty for the coarsest level with admissible pairs.
        k: Target rank for each block's column basis.
        p: Oversampling parameter for the test matrices. Defaults to `0`.
        seed: Optional base seed forwarded to
            `gfcompress.fixed_pattern.build_admissible_test_matrices` for
            reproducibility.

    Returns:
        A list of `ColumnBasis`, one per admissible pair `(alpha, beta)` at
        `level` (in the order `L^int` yields them: outer loop over
        `root.nodes_at_level(level)`, inner loop over each box's interaction
        list). Each `u` has orthonormal columns and shape
        `(len(alpha.row_indices), k)`.
    """
    level_nodes = root.nodes_at_level(level)
    il = interaction_lists(root)[level]

    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=seed)

    # Map each box (by identity) to the Y sample from the Omega whose
    # active_boxes include it -- Eq. 4.4 guarantees this Omega is unique per
    # box.
    y_for_box: dict[int, NDArray[np.floating]] = {}
    for tm in test_matrices:
        y = peeled_matvec(operator, tm.omega, factors)
        for box in tm.active_boxes:
            y_for_box[id(box)] = y

    result: list[ColumnBasis] = []
    for i, alpha in enumerate(level_nodes):
        for beta in il[i]:
            y = y_for_box[id(beta)]
            y_alpha = y[alpha.row_indices, :]
            u = orth(y_alpha, k)
            result.append(ColumnBasis(alpha=alpha, beta=beta, u=u))

    return result


__all__ = ["ColumnBasis", "column_bases"]
