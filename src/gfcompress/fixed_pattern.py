"""Fixed periodic admissible test matrices (`<= 6^d`, paper Sec. 4.1.4, Task 4.2).

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


__all__ = [
    "PERIOD",
    "PeriodicTestMatrix",
    "grid_coordinates",
    "pattern_cell",
    "build_admissible_test_matrices",
]
