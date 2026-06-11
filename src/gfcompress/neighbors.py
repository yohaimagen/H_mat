"""Neighbor lists `L^nei` over the geometric cluster tree (Task 1.4).

Per CLAUDE.md, all combinatorics (neighbors, interaction lists, admissibility,
coloring, peeling) run on the `N` boxes of the single geometric cluster tree
and are unchanged by dof multiplicity.

Because `build_tree` (Task 1.3) constructs a fixed uniform **dyadic grid** --
at every level `l`, every surviving node's `bounding_box` is a cell of one
common `2^l`-per-axis subdivision of the root domain box, so same-level cells
are grid-aligned and congruent -- same-level adjacency reduces to a simple
touch-or-overlap test on axis-aligned boxes (`boxes_adjacent`). Two boxes are
neighbors iff their dyadic cells touch (share a face, edge, or corner) or
overlap; a box is always its own neighbor. On a full grid an interior box has
exactly `3^d` neighbors (itself plus the `3^d - 1` cells sharing a face,
edge, or corner), and a boundary box has fewer.

`neighbor_lists(root)` builds the per-level neighbor map by brute-force
all-pairs comparison among same-level nodes using `boxes_adjacent`. This is
`O(n_l^2)` per level `l`, which is fine for the tree sizes used in this
project's tests; if it ever becomes a bottleneck, it can be replaced by a
spatial hash keyed on dyadic-grid cell indices without changing the API.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.tree import TreeNode

#: Default absolute tolerance for the touch-or-overlap test, used to treat
#: dyadic cells that abut along a shared boundary (which differ only by
#: floating-point round-off) as touching rather than disjoint.
_DEFAULT_TOL = 1e-9


def boxes_adjacent(
    box_a: NDArray[np.float64], box_b: NDArray[np.float64], tol: float = _DEFAULT_TOL
) -> bool:
    """Return whether two axis-aligned boxes touch or overlap.

    Two boxes are adjacent iff, along *every* axis, their `[lo, hi]` extents
    overlap or share an endpoint (within `tol`). This is the standard
    separating-axis test: the boxes are adjacent unless some axis separates
    them by a positive gap.

    Args:
        box_a: Axis-aligned box, shape `(d, 2)`, `box_a[i] = (lo_i, hi_i)`.
        box_b: Axis-aligned box, shape `(d, 2)`, `box_b[i] = (lo_i, hi_i)`.
        tol: Absolute tolerance on each axis: a gap of size `<= tol` between
            the boxes on that axis still counts as touching. This absorbs
            floating-point error at the shared boundary of abutting dyadic
            cells produced by `build_tree`.

    Returns:
        `True` if the boxes touch or overlap on every axis (so they are
        adjacent / neighbors), `False` if some axis separates them by a gap
        `> tol`.
    """
    if box_a.shape != box_b.shape:
        raise ValueError(f"box shapes must match, got {box_a.shape} and {box_b.shape}")

    lo_a, hi_a = box_a[:, 0], box_a[:, 1]
    lo_b, hi_b = box_b[:, 0], box_b[:, 1]

    # On each axis, the boxes are separated by a gap iff one box's hi is
    # strictly less than the other box's lo (beyond tol).
    gap = np.maximum(lo_a - hi_b, lo_b - hi_a)
    return bool(np.all(gap <= tol))


def neighbor_lists(
    root: TreeNode, tol: float = _DEFAULT_TOL
) -> dict[int, dict[int, list[TreeNode]]]:
    """Build the per-level neighbor-list map `L^nei` for the tree rooted at
    `root`.

    For each level `l` present in the tree, and for each node `alpha` at
    level `l`, `L^nei(alpha)` is the list of all same-level nodes `beta`
    (including `alpha` itself) whose `bounding_box` touches or overlaps
    `alpha`'s `bounding_box`, per `boxes_adjacent`.

    `TreeNode` is a mutable dataclass and therefore unhashable, so nodes
    cannot be used directly as dict keys. Instead, each per-level map is
    keyed by the node's position in `nodes_at_level(level)`'s output order
    (i.e. `level_nodes[i]` corresponds to key `i`).

    Args:
        root: Root of the geometric cluster tree (e.g. from `build_tree`).
        tol: Adjacency tolerance forwarded to `boxes_adjacent`.

    Returns:
        Mapping `level -> {index_in_level: [neighbor nodes, including the
        node itself]}`. Each per-level dict has one entry per node at that
        level, keyed by that node's index in `root.nodes_at_level(level)`.
    """
    result: dict[int, dict[int, list[TreeNode]]] = {}
    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        level_map: dict[int, list[TreeNode]] = {}
        for i, alpha in enumerate(level_nodes):
            neighbors = [
                beta
                for beta in level_nodes
                if boxes_adjacent(alpha.bounding_box, beta.bounding_box, tol=tol)
            ]
            level_map[i] = neighbors
        result[level] = level_map
    return result
