"""Per-level row bases `V_{alpha,beta}` and core matrices `B_{alpha,beta}`
(Task 5.3, revised by Task F.5/F.4/FIXPLAN #4/#6).

This is the mirror of Task 5.2/F.5's `gfcompress.column_basis.column_bases`
through `peeled_rmatvec` (Task 5.1) with the `side="row"` fixed periodic test
matrices `Psi` (`gfcompress.fixed_pattern.build_admissible_test_matrices`,
Task F.4) -- reused verbatim, not reinvented (FIXPLAN defect #6: an earlier
attempt duplicated `fixed_pattern` with rows swapped in; that is forbidden).

For tree level `l`, with the low-rank factors of levels `2, ..., l-1` already
stored as a `gfcompress.peeling.Factors` list:

1. Build the level's fixed `Psi` (`side="row"`).
2. For each such `Psi`, compute the level-`l` row sample
   `Z = (A - A^{(l-1)})* @ Psi` via `gfcompress.peeling.peeled_rmatvec` --
   exactly one peeled rmatvec of width `k + p` per emitted `Psi`.
3. For every admissible pair `(alpha, beta)` at `l`, look up the unique
   `Psi`/`Z` whose active boxes include `alpha` (Eq. 4.4's transposed
   constraint, per the `fixed_pattern` module docstring), restrict to
   `beta`'s col-index set `I_beta = beta.col_indices` (`Z(I_beta, :)`), and
   set `V_{alpha,beta} = qr(Z(I_beta, :), k)` (pivoted `orth`).

Per Eq. 4.3, the core-matrix solve also needs the Gaussian sketch block
`G_alpha` used to generate `Z(I_beta, :)` -- `RowBasis` retains it
(`tm.blocks[alpha]`) rather than re-deriving it from seeds.

`core_matrices` combines a level's `ColumnBasis` list (`u`, `y_alpha`,
`g_beta`) with its `RowBasis` list (`v`, `g_alpha`) via
`gfcompress.randomized.core_matrix_solve` (Eq. 4.3) into one `BlockFactor`
per admissible pair.

`compress_level` (Algorithm 4.1's level loop: `column_bases` + `row_bases` +
`core_matrices`) has moved to `gfcompress.compress` (Task 5.6): a module named
after this one pass should not also export the level driver that combines it
with `column_basis`'s pass and the outer loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gfcompress.column_basis import ColumnBasis
from gfcompress.fixed_pattern import build_admissible_test_matrices
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import BlockFactor, Factors, peeled_rmatvec
from gfcompress.randomized import core_matrix_solve, orth
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class RowBasis:
    """Row-space basis `V_{alpha,beta}` for one admissible block `(alpha,
    beta)` at a single level, plus the Gaussian sketch quantity Eq. 4.3's
    core-matrix solve needs.

    Attributes:
        alpha: The row box. `alpha.row_indices` (length `dof_row * |alpha|`)
            indexes the global row space `{0, ..., n_rows - 1}`.
        beta: The col box. `beta.col_indices` (length `dof_col * |beta|`)
            indexes the global col space `{0, ..., n_cols - 1}`.
        v: Orthonormal row-space basis `V_{alpha,beta}`, shape
            `(len(beta.col_indices), k)`, satisfying `v.conj().T @ v ~= I_k`.
        g_alpha: The Gaussian sketch block used to generate the row sample
            `Z(I_beta, :)`, `tm.blocks[alpha]`, shape
            `(len(alpha.row_indices), k + p)` -- the `G_alpha` of Eq. 4.3.
    """

    alpha: TreeNode
    beta: TreeNode
    v: NDArray[np.float64]
    g_alpha: NDArray[np.float64]


def row_bases(
    operator: MatVecOperator,
    root: TreeNode,
    lists: TreeLists,
    mesh: FaultMesh,
    level: int,
    factors: Factors,
    k: int,
    p: int = 0,
    seed: int | None = None,
) -> list[RowBasis]:
    """Compute the level-`level` row bases `V_{alpha,beta}` for every
    admissible pair `(alpha, beta)` at `level`.

    Args:
        operator: The black-box operator `A` (accessed only via
            `operator.rmatvec`, through `peeled_rmatvec`).
        root: Root of the geometric cluster tree.
        lists: Precomputed `TreeLists` (`gfcompress.interactions.build_lists`)
            for the tree rooted at `root`; supplies `L^int` without
            recomputing it.
        mesh: The `FaultMesh` underlying `root` (provides `n_rows` for sizing
            the test matrices).
        level: The tree level whose admissible pairs (`L^int`) are processed.
        factors: Flat list of `BlockFactor`s for levels `2, ..., level - 1`
            (`peeled_rmatvec` subtracts their adjoint contribution before
            sampling). Empty for the coarsest level with admissible pairs.
        k: Target rank for each block's row basis.
        p: Oversampling parameter for the test matrices. Defaults to `0`.
        seed: Optional base seed forwarded to
            `gfcompress.fixed_pattern.build_admissible_test_matrices` for
            reproducibility. Passing the same `seed` as `column_bases` is
            intentional -- `Omega` (`side="col"`) and `Psi` (`side="row"`)
            are drawn from independent streams regardless.

    Returns:
        A list of `RowBasis`, one per admissible pair `(alpha, beta)` at
        `level`, in the same order as `column_bases` (outer loop over
        `root.nodes_at_level(level)`, inner loop over each box's interaction
        list). Each `v` has orthonormal columns and shape
        `(len(beta.col_indices), k)`.
    """
    level_nodes = root.nodes_at_level(level)

    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=seed, side="row")

    # Map each box (by identity) to the Z sample and G_alpha block from the
    # Psi whose active_boxes include it -- Eq. 4.4's transposed constraint
    # guarantees this Psi is unique per box.
    z_for_box: dict[TreeNode, NDArray[np.float64]] = {}
    g_for_box: dict[TreeNode, NDArray[np.float64]] = {}
    for tm in test_matrices:
        z = np.asarray(peeled_rmatvec(operator, tm.omega, factors), dtype=np.float64)
        for box in tm.active_boxes:
            z_for_box[box] = z
            g_for_box[box] = tm.blocks[box]

    result: list[RowBasis] = []
    for alpha in level_nodes:
        for beta in lists.interaction[alpha]:
            z = z_for_box[alpha]
            z_beta = z[beta.col_indices, :]
            v = orth(z_beta, k)
            result.append(RowBasis(alpha=alpha, beta=beta, v=v, g_alpha=g_for_box[alpha]))

    return result


def core_matrices(col_bases: list[ColumnBasis], row_bases_: list[RowBasis]) -> Factors:
    """Combine a level's column and row bases into `BlockFactor`s (Eq. 4.3).

    For each admissible pair `(alpha, beta)`, forms the core matrix
    `B_{alpha,beta}` via `gfcompress.randomized.core_matrix_solve` from
    `col_bases`'s `u`/`y_alpha`/`g_beta` and `row_bases_`'s `v`/`g_alpha`.

    Args:
        col_bases: Column bases for the level, from `column_bases`.
        row_bases_: Row bases for the level, from `row_bases`, covering the
            same set of admissible pairs as `col_bases` (order-independent:
            matched by `(alpha, beta)` identity).

    Returns:
        A `Factors` list (one `BlockFactor` per admissible pair), in
        `col_bases`'s order.

    Raises:
        KeyError: If some pair in `col_bases` has no matching entry in
            `row_bases_`.
    """
    row_by_pair = {(rb.alpha, rb.beta): rb for rb in row_bases_}

    result: Factors = []
    for cb in col_bases:
        rb = row_by_pair[(cb.alpha, cb.beta)]
        b = core_matrix_solve(cb.u, rb.v, cb.y_alpha, rb.g_alpha, cb.g_beta)
        result.append(BlockFactor(alpha=cb.alpha, beta=cb.beta, u=cb.u, b=b, v=rb.v))

    return result


__all__ = ["RowBasis", "core_matrices", "row_bases"]
