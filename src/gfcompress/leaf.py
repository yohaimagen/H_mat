"""Leaf/inadmissible dense-block extraction through the residual operator
`A - A^{(L)}` (Task 5.4, Sec. 4.1.3, Algorithm 4.1).

At the leaf level `L` of the geometric cluster tree, every same-level pair
`(alpha, beta)` with `beta in L^nei(alpha)` (Task F.2's
`gfcompress.interactions.TreeLists.nei`, which -- per
`gfcompress.neighbors.neighbor_lists` -- includes `alpha` itself) is
*inadmissible*: it is never compressed into a low-rank factor, but extracted
directly as a dense block `A(I_alpha, I_beta)`.

Task 4.3's `gfcompress.fixed_pattern.build_leaf_test_matrices` builds the
fixed, period-`3` periodic test matrices `Omega` needed to read these blocks
off a sample of the *residual* operator `A - A^{(L)}` (all admissible blocks
of levels `2, ..., L` already peeled off and stored as `Factors`, Task 5.1):
for the unique `Omega` in which `beta` is active,

    ((A - A^{(L)}) @ Omega)[alpha.row_indices, :w_beta] == A(I_alpha, I_beta)

because (i) peeling `A^{(L)}` makes every far-field (admissible) contribution
to that sample vanish, leaving only `alpha`'s own neighbor blocks, and (ii)
the period-3 pattern guarantees no other member of `L^nei(alpha)` (`alpha`
included) is active in that same `Omega` -- so those neighbor blocks other
than `beta` also contribute nothing. Per CLAUDE.md and the `fixed_pattern`
module docstring, isolation comes from these two facts, *not* from giving
each box a private column slot: every active box in one `Omega` shares the
same `w_beta`-wide column range, keeping the total probe width over all
`Omega`'s at `<= 3^d * w_max` (`w_max = max_beta len(beta.col_indices)`),
independent of `N`.

`extract_leaves` is the thin driver that applies those probes through
`gfcompress.peeling.peeled_matvec` (Task 5.1) with the full set of stored
`Factors` (levels `2, ..., L`) and reads off one `DenseLeaf` per neighbor
pair. It never assembles a dense `A`; the dense block it stores is exactly
the sample it read, not a call to the underlying kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gfcompress.fixed_pattern import build_leaf_test_matrices, leaf_probe_owners
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import Factors, peeled_matvec
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class DenseLeaf:
    """Dense (uncompressed) block for one inadmissible neighbor pair
    `(alpha, beta)` at the leaf level.

    Attributes:
        alpha: The row box. `alpha.row_indices` (length `dof_row * |alpha|`)
            indexes the global row space `{0, ..., n_rows - 1}`.
        beta: The col box, `beta in L^nei(alpha)` (may equal `alpha`).
            `beta.col_indices` (length `dof_col * |beta|`) indexes the global
            col space `{0, ..., n_cols - 1}`.
        block: The dense sub-block `A(I_alpha, I_beta)`, shape
            `(len(alpha.row_indices), len(beta.col_indices))`.
    """

    alpha: TreeNode
    beta: TreeNode
    block: NDArray[np.float64]


def extract_leaves(
    operator: MatVecOperator,
    root: TreeNode,
    lists: TreeLists,
    mesh: FaultMesh,
    level: int,
    factors: Factors,
) -> list[DenseLeaf]:
    """Extract every inadmissible neighbor block `(alpha, beta)` at the leaf
    `level`, from the residual operator `A - A^{(L)}`.

    Args:
        operator: The black-box operator `A` (accessed only via
            `operator.matvec`, through `peeled_matvec`).
        root: Root of the geometric cluster tree.
        lists: Precomputed `TreeLists` (`gfcompress.interactions.build_lists`)
            for the tree rooted at `root`; supplies `L^nei` without
            recomputing it.
        mesh: The `FaultMesh` underlying `root` (provides `n_cols` for sizing
            the leaf test matrices).
        level: The leaf tree level `L` whose neighbor pairs are extracted.
        factors: Flat list of `BlockFactor`s for every admissible pair of
            levels `2, ..., L` (i.e. including `level` itself), so that
            `peeled_matvec` samples the full residual `A - A^{(L)}` in which
            only neighbor blocks survive.

    Returns:
        A list of `DenseLeaf`, one per pair `(alpha, beta)` with `alpha`
        ranging over `root.nodes_at_level(level)` and `beta` over
        `lists.nei[alpha]` (in that nested order). Issues exactly one peeled
        matvec per emitted leaf test matrix (`<= 3 ** mesh.tree_dim`), each of width
        `w_max = max_beta len(beta.col_indices)`, for a total probe width
        `<= 3 ** mesh.tree_dim * w_max`.
    """
    level_nodes = root.nodes_at_level(level)

    test_matrices = build_leaf_test_matrices(root, level, mesh)

    owners = leaf_probe_owners(root, lists, level, test_matrices)
    result: dict[tuple[TreeNode, TreeNode], DenseLeaf] = {}
    for probe in test_matrices:
        y = np.asarray(peeled_matvec(operator, probe.realize(), factors), dtype=np.float64)
        for (alpha, beta), owner in owners.items():
            if owner is probe:
                width = len(beta.col_indices)
                block = np.array(y[np.ix_(alpha.row_indices, np.arange(width))], copy=True)
                result[(alpha, beta)] = DenseLeaf(alpha=alpha, beta=beta, block=block)
        del y
    return [result[(alpha, beta)] for alpha in level_nodes for beta in lists.nei[alpha]]


__all__ = ["DenseLeaf", "extract_leaves"]
