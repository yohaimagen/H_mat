"""Fixed periodic admissible test matrices (`<= 6^d`, paper Sec. 4.1.4, Task 4.2)
and fixed periodic leaf/inadmissible test matrices (`<= 3^d`, Sec. 4.1.3,
Task 4.3).

Per Levitt & Martinsson (2024), Sec. 4.1.4, the boxes at a level of the
geometric cluster tree live on a common dyadic grid of `2^level` cells per
axis (see `gfcompress.build_tree.build_tree`). Tiling that grid periodically
with period `6` along every axis assigns each box a **pattern cell**
`(i_0 mod 6, ..., i_{d-1} mod 6) in {0, ..., 5}^d`, where `(i_0, ..., i_{d-1})`
are the box's dyadic grid coordinates.

The period `6` is large enough that, for any box `alpha` at the level, no two
boxes in `{beta} | L^nei(alpha) | L^int(alpha)` share a pattern cell: `L^nei`
reaches at most one cell away from `alpha` along each axis, and `L^int`
reaches at most two cells away (children of the parent's neighbors), so the
full neighborhood `L^nei(alpha) | L^int(alpha)` spans grid coordinates within
`+-2` of `alpha`'s along each axis -- a `5x...x5` window, strictly smaller than
the `6x...x6` period. Hence activating *every* box that shares a single
pattern cell simultaneously (filling its columns with an independent Gaussian
block, zeroing everything else) cannot put two boxes from the same
neighborhood `{beta} | L^nei(alpha) | L^int(alpha)` into the same "active"
group: whichever pattern cell `beta` belongs to, `alpha`'s other neighbors and
interaction-list partners are guaranteed to fall into *different* pattern
cells (or be inactive, i.e. zero) for that same test matrix. Consequently, for
every admissible pair `(alpha, beta)`, the single test matrix `Omega` whose
active set contains `beta` automatically satisfies the Eq. 4.4 sampling
constraint for `(alpha, beta)`: `beta`'s columns are Gaussian and every column
in `L^nei(alpha) | L^int(alpha) \\ {beta}` is zero.

`build_admissible_test_matrices` emits one such `Omega` per non-empty pattern
cell (at most `6**d`), each of shape `(mesh.n_cols, k + p)`.

Leaf/inadmissible test matrices (Task 4.3, Sec. 4.1.3)
--------------------------------------------------------
For a same-level pair `(alpha, beta)` with `beta in L^nei(alpha)` (an
inadmissible "neighbor" pair), the dense block `A_{alpha,beta}` is extracted
directly rather than compressed. Reading off `A_{alpha,beta}` from a matvec
`A @ Omega` requires `Omega` to contain a column block `C` of width
`w = len(beta.col_indices)` such that `Omega[beta.col_indices, C] = I_w` and
`Omega[gamma.col_indices, C] = 0` for every other box `gamma != beta` active
in `Omega` -- then `(A @ Omega)[alpha.row_indices, C] = A_{alpha,beta} @ I_w +
sum_{gamma != beta} A_{alpha,gamma} @ 0 = A_{alpha,beta}` exactly, regardless
of whether those other active boxes interact with `alpha`.

`L^nei(alpha)` reaches at most one dyadic cell away from `alpha` along each
axis -- a `3x...x3` window. Tiling the dyadic grid periodically with period
`3` along every axis assigns each box a **leaf pattern cell**
`(i_0 mod 3, ..., i_{d-1} mod 3) in {0, ..., 2}^d`. Two distinct boxes whose
grid coordinates differ by at most `2` along every axis (as any two boxes
within a `3x...x3` window of `alpha` -- including `alpha` itself -- do)
cannot share a leaf pattern cell: a collision would require their coordinate
difference along some axis to be a nonzero multiple of `3`, but the
difference is bounded in `{-2, ..., 2}`, and `0` only occurs for identical
coordinates (the same box). Hence, for every inadmissible pair `(alpha,
beta)`, the single test matrix `Omega` whose active set contains `beta` is
the *unique* one in which `beta` is active among `L^nei(alpha)` (including
`alpha` itself).

Concretely, every box sharing a leaf pattern cell is activated
*simultaneously* in the same `Omega`, each given its own dedicated column
slot: `Omega` has shape `(mesh.n_cols, sum(w_beta for beta in active_boxes))`,
and the `j`-th active box `beta` gets `I_{w_beta}` placed in
`Omega[beta.col_indices, col_slices[j]]`, with every other entry of that
column block -- including the rows of every other active box -- zero. This is
"identity-like" rather than a full `n_cols x n_cols` identity (each `Omega` is
narrow, width `O(N / 3**d)`), while the matrix count stays at `<= 3**d`.

`build_leaf_test_matrices` emits one such `Omega` per non-empty leaf pattern
cell (at most `3**d`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.randomized import gaussian
from gfcompress.tree import TreeNode

#: Periodic pattern period for admissible test matrices (Sec. 4.1.4).
PERIOD = 6

#: Periodic pattern period for leaf/inadmissible test matrices (Sec. 4.1.3).
LEAF_PERIOD = 3


@dataclass(frozen=True)
class PeriodicTestMatrix:
    """One emitted test matrix for the fixed `6x...x6` periodic pattern.

    Attributes:
        omega: The test matrix `Omega`, shape `(mesh.n_cols, k + p)`.
        pattern: The pattern-cell offset `(i_0 mod 6, ..., i_{d-1} mod 6)`
            shared by every box in `active_boxes`, length `d`.
        active_boxes: The level-`level` nodes whose pattern cell equals
            `pattern`; their `col_indices` rows of `omega` hold independent
            Gaussian blocks, every other row is zero.
    """

    omega: NDArray[np.float64]
    pattern: tuple[int, ...]
    active_boxes: list[TreeNode]


def grid_coordinates(node: TreeNode, root: TreeNode) -> tuple[int, ...]:
    """Compute `node`'s dyadic grid coordinates `(i_0, ..., i_{d-1})`.

    `node.bounding_box` is a cell of the dyadic grid obtained by subdividing
    `root.bounding_box` into `2**node.level` equal cells per axis (see
    `gfcompress.build_tree.build_tree`). The grid coordinate along axis `a` is

        i_a = round((node.bounding_box[a, 0] - root.bounding_box[a, 0]) / cell_width_a)

    where `cell_width_a = (root.bounding_box[a, 1] - root.bounding_box[a, 0])
    / 2**node.level`. Rounding guards against floating-point error in the
    bisection arithmetic.

    Args:
        node: The node whose grid coordinates are computed.
        root: Root of the geometric cluster tree (provides the level-0 domain
            box).

    Returns:
        Tuple of `d` integers, each in `{0, ..., 2**node.level - 1}`.
    """
    level = node.level
    n_cells = 2**level
    d = root.bounding_box.shape[0]

    coords = []
    for axis in range(d):
        root_lo = root.bounding_box[axis, 0]
        root_hi = root.bounding_box[axis, 1]
        cell_width = (root_hi - root_lo) / n_cells
        raw = (node.bounding_box[axis, 0] - root_lo) / cell_width
        i_a = int(round(raw))
        coords.append(i_a)

    return tuple(coords)


def pattern_cell(node: TreeNode, root: TreeNode) -> tuple[int, ...]:
    """Compute `node`'s periodic pattern cell `(i_0 mod 6, ..., i_{d-1} mod 6)`.

    Args:
        node: The node whose pattern cell is computed.
        root: Root of the geometric cluster tree.

    Returns:
        Tuple of `d` integers, each in `{0, ..., PERIOD - 1}`.
    """
    coords = grid_coordinates(node, root)
    return tuple(i % PERIOD for i in coords)


def build_admissible_test_matrices(
    root: TreeNode,
    level: int,
    mesh: FaultMesh,
    k: int,
    p: int = 0,
    seed: int | None = None,
) -> list[PeriodicTestMatrix]:
    """Build the fixed `6x...x6` periodic admissible test matrices for `level`.

    Groups the level-`level` nodes by their periodic pattern cell
    (`pattern_cell`), and emits one `Omega` of shape `(mesh.n_cols, k + p)` per
    non-empty pattern cell: for each "active" box `beta` in that cell,
    `Omega[beta.col_indices, :]` is filled with an independent `(|beta.
    col_indices|, k + p)` Gaussian block (`gfcompress.randomized.gaussian`);
    all other rows of `Omega` are zero.

    Per the module docstring, this guarantees that for every admissible pair
    `(alpha, beta)` at `level`, the unique emitted `Omega` whose
    `active_boxes` contains `beta` satisfies the Eq. 4.4 sampling constraint
    for `(alpha, beta)`.

    Args:
        root: Root of the geometric cluster tree.
        level: The tree level to build test matrices for.
        mesh: The `FaultMesh` (provides `n_cols`).
        k: Target rank (number of "signal" columns of each `Omega`).
        p: Oversampling parameter. Defaults to `0`.
        seed: Optional base seed for `gfcompress.randomized.gaussian`. Each
            active box's Gaussian block is drawn with a seed derived
            deterministically from `seed` and a running counter, so the whole
            generator is reproducible given `seed`.

    Returns:
        A list of `PeriodicTestMatrix`, one per non-empty pattern cell, in a
        deterministic order (pattern cells sorted lexicographically). At most
        `6 ** mesh.d` entries.
    """
    level_nodes = root.nodes_at_level(level)

    groups: dict[tuple[int, ...], list[TreeNode]] = {}
    for node in level_nodes:
        cell = pattern_cell(node, root)
        groups.setdefault(cell, []).append(node)

    n_cols = mesh.n_cols
    k_p = k + p

    result: list[PeriodicTestMatrix] = []
    box_counter = 0
    for cell in sorted(groups.keys()):
        active_boxes = groups[cell]
        omega = np.zeros((n_cols, k_p), dtype=np.float64)
        for beta in active_boxes:
            block_seed = None if seed is None else seed + box_counter
            box_counter += 1
            block = gaussian(len(beta.col_indices), k, p, seed=block_seed)
            omega[beta.col_indices, :] = block
        result.append(PeriodicTestMatrix(omega=omega, pattern=cell, active_boxes=active_boxes))

    return result


@dataclass(frozen=True)
class PeriodicLeafTestMatrix:
    """One emitted test matrix for the fixed `3x...x3` leaf periodic pattern.

    Attributes:
        omega: The test matrix `Omega`, shape `(mesh.n_cols, w)` with
            `w = sum(len(beta.col_indices) for beta in active_boxes)`.
        pattern: The leaf pattern-cell offset `(i_0 mod 3, ..., i_{d-1} mod
            3)` shared by every box in `active_boxes`, length `d`.
        active_boxes: The level-`level` nodes whose leaf pattern cell equals
            `pattern`, in the order their column slots appear in `omega`.
        col_slices: For each `beta` in `active_boxes` (same order, same
            length), the slice `omega[:, col_slices[j]]` is the
            `(mesh.n_cols, len(beta.col_indices))` block where
            `omega[beta.col_indices, col_slices[j]] = I_{len(beta.
            col_indices)}`; every other entry of `omega`, including
            `omega[gamma.col_indices, col_slices[j]]` for `gamma != beta`, is
            zero.
    """

    omega: NDArray[np.float64]
    pattern: tuple[int, ...]
    active_boxes: list[TreeNode]
    col_slices: list[slice]


def leaf_pattern_cell(node: TreeNode, root: TreeNode) -> tuple[int, ...]:
    """Compute `node`'s periodic leaf pattern cell `(i_0 mod 3, ..., i_{d-1}
    mod 3)`.

    Args:
        node: The node whose leaf pattern cell is computed.
        root: Root of the geometric cluster tree.

    Returns:
        Tuple of `d` integers, each in `{0, ..., LEAF_PERIOD - 1}`.
    """
    coords = grid_coordinates(node, root)
    return tuple(i % LEAF_PERIOD for i in coords)


def build_leaf_test_matrices(
    root: TreeNode,
    level: int,
    mesh: FaultMesh,
) -> list[PeriodicLeafTestMatrix]:
    """Build the fixed `3x...x3` periodic leaf/inadmissible test matrices for
    `level`.

    Groups the level-`level` nodes by their leaf pattern cell
    (`leaf_pattern_cell`), and emits one `Omega` per non-empty leaf pattern
    cell. Each active box `beta` in the cell gets its own dedicated column
    slot of width `w_beta = len(beta.col_indices)`: `Omega` has shape
    `(mesh.n_cols, sum(w_beta for beta in active_boxes))`, and for the `j`-th
    active box, `Omega[beta.col_indices, col_slices[j]] = I_{w_beta}` while
    every other entry of column block `col_slices[j]` -- including the rows
    of every other active box `gamma != beta` -- is zero.

    Because each active box owns a disjoint column slot, `(A @ Omega)
    [alpha.row_indices, col_slices[j]] = A_{alpha,beta} @ I_{w_beta} +
    sum_{gamma != beta} A_{alpha,gamma} @ 0 = A_{alpha,beta}` exactly, with no
    contamination from any other active box (whether or not it interacts
    with `alpha`).

    Per the module docstring, the period-3 collision-avoidance property
    further guarantees that for every inadmissible pair `(alpha, beta)` at
    `level` with `beta in L^nei(alpha)`, the unique emitted `Omega` whose
    `active_boxes` contains `beta` is the only one isolating
    `A_{alpha,beta}`: no other box in `L^nei(alpha)` (including `alpha`
    itself) can be active in that same `Omega`.

    Args:
        root: Root of the geometric cluster tree.
        level: The tree level to build leaf test matrices for (typically the
            deepest/leaf level).
        mesh: The `FaultMesh` (provides `n_cols`).

    Returns:
        A list of `PeriodicLeafTestMatrix`, one per non-empty leaf pattern
        cell, in a deterministic order (pattern cells sorted
        lexicographically). At most `3 ** mesh.d` entries.
    """
    level_nodes = root.nodes_at_level(level)

    groups: dict[tuple[int, ...], list[TreeNode]] = {}
    for node in level_nodes:
        cell = leaf_pattern_cell(node, root)
        groups.setdefault(cell, []).append(node)

    n_cols = mesh.n_cols

    result: list[PeriodicLeafTestMatrix] = []
    for cell in sorted(groups.keys()):
        active_boxes = groups[cell]
        width = sum(len(beta.col_indices) for beta in active_boxes)
        omega = np.zeros((n_cols, width), dtype=np.float64)

        col_slices: list[slice] = []
        offset = 0
        for beta in active_boxes:
            w_beta = len(beta.col_indices)
            sl = slice(offset, offset + w_beta)
            omega[np.ix_(beta.col_indices, np.arange(offset, offset + w_beta))] = np.eye(w_beta)
            col_slices.append(sl)
            offset += w_beta

        result.append(
            PeriodicLeafTestMatrix(
                omega=omega, pattern=cell, active_boxes=active_boxes, col_slices=col_slices
            )
        )

    return result


__all__ = [
    "LEAF_PERIOD",
    "PERIOD",
    "PeriodicLeafTestMatrix",
    "PeriodicTestMatrix",
    "build_admissible_test_matrices",
    "build_leaf_test_matrices",
    "grid_coordinates",
    "leaf_pattern_cell",
    "pattern_cell",
]
