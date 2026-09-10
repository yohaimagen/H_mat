"""Fixed periodic admissible test matrices (`<= 6^d`, paper Sec. 4.1.4)
and fixed periodic leaf/inadmissible test matrices (`<= 3^d`, Sec. 4.1.3).

Per Levitt & Martinsson (2024), Sec. 4.1.4, the boxes at a level of the
geometric cluster tree live on a common dyadic grid of `2^level` cells per
axis (see `gfcompress.build_tree.build_tree`). Tiling that grid periodically
with period `6` along every axis assigns each box a **pattern cell**
`(i_0 mod 6, ..., i_{d-1} mod 6) in {0, ..., 5}^d`, where `(i_0, ..., i_{d-1})`
are the box's dyadic grid coordinates.

The period `6` is the *minimal* one for which, for any box `alpha` at the
level, no two boxes in `{beta} | L^nei(alpha) | L^int(alpha)` share a pattern
cell. `L^int(alpha)` consists of the children of `alpha`'s parent's neighbors
(minus `alpha`'s own neighbors). With `i` the grid coordinate of `alpha` along
an axis, the parent sits at `i // 2` and the parent's neighbors at
`i // 2 + {-1, 0, 1}`; their children therefore span grid coordinates
`2 * (i // 2) - 2 ... 2 * (i // 2) + 3`, i.e. offsets `-2 ... +3` relative to
`alpha` when `i` is even and `-3 ... +2` when `i` is odd. Either way `L^nei |
L^int` spans a **6-wide** window along each axis (`L^nei` alone spans only
`-1 ... +1`, a 3-wide window -- hence period `3` for the leaf matrices below).
Two distinct boxes inside a 6-wide window differ by at most `5` along each
axis, so they can only collide mod `6` if their difference is `0` along every
axis -- i.e. if they are the same box. Hence activating *every* box that shares a single
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

`build_admissible_test_matrices` emits one lightweight description per
non-empty pattern cell (at most `6**d`); each is realized only for its one
matvec/rmatvec, at shape `(mesh.n_cols, k + p)`.

The very same combinatorics applies to *row* sampling (`Psi`, used to build
the row bases): the admissibility relation is symmetric, so grouping the boxes
by the same period-6 pattern cell and filling `Psi[alpha.row_indices, :]` with
a Gaussian block yields, for every admissible pair `(alpha, beta)`, a unique
`Psi` in which `alpha` is Gaussian and every `gamma in L^nei(beta) |
L^int(beta) \\ {alpha}` is zero. `side="row"` selects this variant; the only
differences are the index set used (`row_indices`, so `Psi` has `mesh.n_rows`
rows -- `A` is not square) and an independent seed stream, so `Omega` and
`Psi` never share random entries.

Leaf/inadmissible test matrices (Task 4.3, Sec. 4.1.3)
--------------------------------------------------------
For a same-level pair `(alpha, beta)` with `beta in L^nei(alpha)` (an
inadmissible "neighbor" pair), the dense block `A_{alpha,beta}` is extracted
directly rather than compressed, from a matvec of the *residual* operator
`A - A^{(L)}` (all admissible blocks of all levels already peeled off). In
that residual only the neighbor blocks of `alpha` survive, so a sample
`Y = (A - A^{(L)}) @ Omega` satisfies

    Y[alpha.row_indices, :] = sum_{gamma in L^nei(alpha)}
        A_{alpha,gamma} @ Omega[gamma.col_indices, :].

Hence `A_{alpha,beta}` is read off directly as long as `Omega` restricted to
`beta`'s columns is an identity block and *no other member of `L^nei(alpha)`
is active in that same `Omega`*.

`L^nei(alpha)` spans grid offsets `-1 ... +1` along each axis -- a `3x...x3`
window. Tiling the dyadic grid periodically with period `3` along every axis
assigns each box a **leaf pattern cell**
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

Because the active boxes of one `Omega` never collide inside any `L^nei`,
they can all **share the same column slot** instead of each getting a
dedicated one: with `w_max = max(len(beta.col_indices) for beta in the
level)`, `Omega` has shape `(mesh.n_cols, w_max)` and every active `beta`
carries `Omega[beta.col_indices, :w_beta] = I_{w_beta}` (`w_beta =
len(beta.col_indices)`). The consumer reads `Y[alpha.row_indices, :w_beta]`.
This keeps each `Omega` as narrow as a single box (`w_max` columns, not
`O(N / 3**d)`) while the matrix count stays at `<= 3**d`, so the whole leaf
extraction costs at most `3**d * w_max` matvecs.

`build_leaf_test_matrices` likewise emits descriptions, realizing one `Omega`
at a time for each non-empty leaf pattern cell (at most `3**d`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists
from gfcompress.randomized import gaussian
from gfcompress.tree import TreeNode

#: Periodic pattern period for admissible test matrices (Sec. 4.1.4).
PERIOD = 6

#: Periodic pattern period for leaf/inadmissible test matrices (Sec. 4.1.3).
LEAF_PERIOD = 3

#: Which of `A`'s two (different-sized) spaces a test matrix lives in:
#: `"col"` -> `Omega` in `R^{n_cols x (k+p)}` (domain of `A`, sampled with
#: `matvec`), `"row"` -> `Psi` in `R^{n_rows x (k+p)}` (range of `A`, sampled
#: with `rmatvec`).
Side = Literal["col", "row"]


@dataclass(frozen=True)
class PeriodicTestMatrix:
    """A lightweight description of one fixed `6x...x6` probe.

    Attributes:
        pattern: The pattern-cell offset `(i_0 mod 6, ..., i_{d-1} mod 6)`
            shared by every box in `active_boxes`, length `d`.
        active_boxes: The level-`level` nodes whose pattern cell equals
            `pattern`; their `col_indices` (resp. `row_indices`) rows of
            `omega` hold independent Gaussian blocks, every other row is zero.
        The dense probe is deliberately not retained. Call :meth:`realize`
        immediately before its one matvec/rmatvec. ``omega`` and ``blocks``
        remain compatibility properties for diagnostics; production consumers
        must use ``realize`` and release its result before the next probe.
    """

    pattern: tuple[int, ...]
    active_boxes: list[TreeNode]
    n_dofs: int
    k: int
    p: int
    seed: int | None
    level: int
    side: Side

    def realize(self) -> RealizedPeriodicTestMatrix:
        """Materialize this probe and its independent active-box sketches."""
        omega = np.zeros((self.n_dofs, self.k + self.p), dtype=np.float64)
        blocks: dict[TreeNode, NDArray[np.float64]] = {}
        for box in self.active_boxes:
            indices = box.col_indices if self.side == "col" else box.row_indices
            block = gaussian(
                len(indices),
                self.k,
                self.p,
                seed=_block_seed(self.seed, self.level, self.side, box.cell_coords),
            )
            omega[indices, :] = block
            blocks[box] = block
        return RealizedPeriodicTestMatrix(omega=omega, blocks=blocks)

    @property
    def omega(self) -> NDArray[np.float64]:
        """Materialize a diagnostic copy of the dense test matrix."""
        return self.realize().omega

    @property
    def blocks(self) -> dict[TreeNode, NDArray[np.float64]]:
        """Materialize diagnostic sketch blocks without retaining a probe."""
        return self.realize().blocks


@dataclass(frozen=True)
class RealizedPeriodicTestMatrix:
    """Ephemeral dense realization of :class:`PeriodicTestMatrix`."""

    omega: NDArray[np.float64]
    blocks: dict[TreeNode, NDArray[np.float64]]


def grid_coordinates(node: TreeNode, root: TreeNode) -> tuple[int, ...]:
    """Return `node`'s stored dyadic grid coordinates `(i_0, ..., i_{d-1})`.

    `build_tree` assigns exact integer cell coordinates while subdividing the
    padded root hypercube. Reusing those integers avoids reconstructing a
    coordinate from floating-point cell geometry.

    Args:
        node: The node whose grid coordinates are computed.
        root: Root of the geometric cluster tree, used to validate the
            coordinate dimensionality.

    Returns:
        Tuple of `d` integers, each in `{0, ..., 2**node.level - 1}`.
    """
    if len(node.cell_coords) != len(root.cell_coords):
        raise ValueError("node does not carry a dyadic cell coordinate")
    return node.cell_coords


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
    p: int = 10,
    seed: int | None = None,
    side: Side = "col",
) -> list[PeriodicTestMatrix]:
    """Build the fixed `6x...x6` periodic admissible test matrices for `level`.

    Groups the level-`level` nodes by their periodic pattern cell
    (`pattern_cell`), and emits one test matrix per non-empty pattern cell:
    for each "active" box `beta` in that cell, the rows of `beta`'s index set
    are filled with an independent Gaussian block
    (`gfcompress.randomized.gaussian`); all other rows are zero.

    With `side="col"` the index set is `beta.col_indices` and the result has
    shape `(mesh.n_cols, k + p)` -- the `Omega` used with `matvec`. With
    `side="row"` it is `beta.row_indices` and the shape is `(mesh.n_rows,
    k + p)` -- the `Psi` used with `rmatvec`. `A` is not square, so the two
    are genuinely different objects; their random streams are independent, so
    `Omega` and `Psi` built from the same `seed` never share entries.

    Per the module docstring, this guarantees that for every admissible pair
    `(alpha, beta)` at `level`, the unique emitted `Omega` whose
    `active_boxes` contains `beta` satisfies the Eq. 4.4 sampling constraint
    for `(alpha, beta)` (and, for `side="row"`, the unique `Psi` whose
    `active_boxes` contains `alpha` satisfies the transposed constraint).

    Args:
        root: Root of the geometric cluster tree.
        level: The tree level to build test matrices for.
        mesh: The `FaultMesh` (provides `n_cols` / `n_rows`).
        k: Target rank (number of "signal" columns of each test matrix).
        p: Oversampling parameter. Defaults to `10`; pass `p=0` to disable
            oversampling explicitly.
        seed: Optional base seed for `gfcompress.randomized.gaussian`. Each
            active box's Gaussian block is drawn with a seed derived
            deterministically from `seed`, `level`, `side`, and the box's
            integer dyadic cell coordinates. This is independent of pattern
            group or traversal order, while the two sides stay independent.
        side: `"col"` (default) for `Omega`, `"row"` for `Psi`.

    Returns:
        A list of `PeriodicTestMatrix`, one per non-empty pattern cell, in a
        deterministic order (pattern cells sorted lexicographically). At most
        `6 ** mesh.tree_dim` entries.

    Raises:
        ValueError: If `side` is not `"col"` or `"row"`.
    """
    if side not in ("col", "row"):
        raise ValueError(f"side must be 'col' or 'row', got {side!r}")

    level_nodes = root.nodes_at_level(level)

    groups: dict[tuple[int, ...], list[TreeNode]] = {}
    for node in level_nodes:
        cell = pattern_cell(node, root)
        groups.setdefault(cell, []).append(node)

    n_dofs = mesh.n_cols if side == "col" else mesh.n_rows
    result: list[PeriodicTestMatrix] = []
    for cell in sorted(groups.keys()):
        active_boxes = sorted(groups[cell], key=lambda node: grid_coordinates(node, root))
        result.append(
            PeriodicTestMatrix(
                pattern=cell,
                active_boxes=active_boxes,
                n_dofs=n_dofs,
                k=k,
                p=p,
                seed=seed,
                level=level,
                side=side,
            )
        )

    return result


def admissible_probe_owners(
    root: TreeNode,
    lists: TreeLists,
    level: int,
    probes: list[PeriodicTestMatrix],
    side: Side,
) -> dict[tuple[TreeNode, TreeNode], PeriodicTestMatrix]:
    """Assign every admissible pair to its explicitly selected fixed probe.

    The ownership map is the consumption contract used by both basis passes:
    callers look up a *pair*, never merely the currently active source box.
    """
    by_box = {box: probe for probe in probes for box in probe.active_boxes}
    owners: dict[tuple[TreeNode, TreeNode], PeriodicTestMatrix] = {}
    for alpha in root.nodes_at_level(level):
        for beta in lists.interaction[alpha]:
            owner = by_box[beta if side == "col" else alpha]
            owners[(alpha, beta)] = owner
    return owners


def validate_admissible_probe_owners(
    owners: dict[tuple[TreeNode, TreeNode], PeriodicTestMatrix], side: Side
) -> None:
    """Reject an explicit pair assignment whose required sketch is absent."""
    for (alpha, beta), probe in owners.items():
        required = beta if side == "col" else alpha
        if required not in probe.active_boxes:
            raise ValueError("probe does not activate the pair's required sketch box")


def _block_seed(seed: int | None, level: int, side: Side, cell: tuple[int, ...]) -> int | None:
    """Derive an independent per-box seed from stable sketch coordinates.

    Returns `None` (i.e. OS entropy) when `seed` is `None`. Otherwise
    `numpy.random.SeedSequence` derives the stream from `(seed, level, side,
    cell)`, rather than the order in which groups or boxes happen to be
    visited. The signed Python integer is losslessly encoded as a sign and
    little-endian 32-bit words; this does not depend on Python's randomized
    hash implementation or alias integers modulo a machine word.
    """
    if seed is None:
        return None
    base = int(seed)
    magnitude = abs(base)
    words: list[int] = []
    while magnitude:
        words.append(magnitude & 0xFFFFFFFF)
        magnitude >>= 32
    entropy = [int(base < 0), len(words), *words, int(level), 0 if side == "col" else 1, *cell]
    state = np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


@dataclass(frozen=True)
class PeriodicLeafTestMatrix:
    """One emitted test matrix for the fixed `3x...x3` leaf periodic pattern.

    Attributes:
        pattern: The leaf pattern-cell offset `(i_0 mod 3, ..., i_{d-1} mod
            3)` shared by every box in `active_boxes`, length `d`.
        active_boxes: The level-`level` nodes whose leaf pattern cell equals
            `pattern`.
    """

    pattern: tuple[int, ...]
    active_boxes: list[TreeNode]
    n_cols: int
    w_max: int

    def realize(self) -> NDArray[np.float64]:
        """Materialize this probe immediately before its residual matvec."""
        omega = np.zeros((self.n_cols, self.w_max), dtype=np.float64)
        for beta in self.active_boxes:
            width = len(beta.col_indices)
            omega[np.ix_(beta.col_indices, np.arange(width))] = np.eye(width)
        return omega

    @property
    def omega(self) -> NDArray[np.float64]:
        """Materialize a diagnostic copy of the dense test matrix."""
        return self.realize()


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
    cell. All active boxes of a cell **share one column slot** of width
    `w_max = max(len(beta.col_indices))` over the level: `Omega` has shape
    `(mesh.n_cols, w_max)` and `Omega[beta.col_indices, :w_beta] =
    I_{w_beta}` for every active `beta` (`w_beta = len(beta.col_indices)`).

    Per the module docstring, the period-3 collision-avoidance property
    guarantees that for every inadmissible pair `(alpha, beta)` at `level`
    with `beta in L^nei(alpha)`, `beta` is the *only* member of
    `L^nei(alpha)` (including `alpha` itself) active in the emitted `Omega`
    that contains it. Sampling the residual operator, in which only
    `alpha`'s neighbor blocks survive, therefore gives
    `((A - A^{(L)}) @ Omega)[alpha.row_indices, :w_beta] = A_{alpha,beta}`:
    the other active boxes of that `Omega` contribute zero because they are
    not neighbors of `alpha`.

    Args:
        root: Root of the geometric cluster tree.
        level: The tree level to build leaf test matrices for (typically the
            deepest/leaf level).
        mesh: The `FaultMesh` (provides `n_cols`).

    Returns:
        A list of `PeriodicLeafTestMatrix`, one per non-empty leaf pattern
        cell, in a deterministic order (pattern cells sorted
        lexicographically). At most `3 ** mesh.tree_dim` entries.
    """
    level_nodes = root.nodes_at_level(level)

    groups: dict[tuple[int, ...], list[TreeNode]] = {}
    for node in level_nodes:
        cell = leaf_pattern_cell(node, root)
        groups.setdefault(cell, []).append(node)

    n_cols = mesh.n_cols
    w_max = max(len(node.col_indices) for node in level_nodes)

    result: list[PeriodicLeafTestMatrix] = []
    for cell in sorted(groups.keys()):
        active_boxes = groups[cell]
        result.append(
            PeriodicLeafTestMatrix(
                pattern=cell, active_boxes=active_boxes, n_cols=n_cols, w_max=w_max
            )
        )

    return result


def leaf_probe_owners(
    root: TreeNode,
    lists: TreeLists,
    level: int,
    probes: list[PeriodicLeafTestMatrix],
) -> dict[tuple[TreeNode, TreeNode], PeriodicLeafTestMatrix]:
    """Assign every leaf neighbor pair to the probe that isolates it."""
    by_box = {box: probe for probe in probes for box in probe.active_boxes}
    return {
        (alpha, beta): by_box[beta]
        for alpha in root.nodes_at_level(level)
        for beta in lists.nei[alpha]
    }


__all__ = [
    "LEAF_PERIOD",
    "PERIOD",
    "PeriodicLeafTestMatrix",
    "PeriodicTestMatrix",
    "RealizedPeriodicTestMatrix",
    "Side",
    "admissible_probe_owners",
    "build_admissible_test_matrices",
    "build_leaf_test_matrices",
    "grid_coordinates",
    "leaf_pattern_cell",
    "leaf_probe_owners",
    "pattern_cell",
    "validate_admissible_probe_owners",
]
