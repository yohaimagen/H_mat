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

`neighbor_lists(root)` builds the neighbor map by brute-force all-pairs
comparison among same-level nodes, applying `boxes_adjacent`'s touch-or-
overlap test to every pair at once via vectorized numpy broadcasting (rather
than one Python-level `boxes_adjacent` call per pair). This keeps the
`O(n_l^2)` per-level complexity but removes the Python-call overhead that
made it a bottleneck at the tree sizes used for the Stage 7 (coloring)
sampling constraint (Task F.2); if it ever becomes a bottleneck again, it can
be replaced by a spatial hash keyed on dyadic-grid cell indices without
changing the API.

`TreeNode` is hashable by identity (Task F.1, `@dataclass(eq=False)`), so the
result is a single flat, node-keyed dict rather than the earlier
`level -> index_in_level -> list` nesting; per-level iteration is available
via `root.nodes_at_level(level)` and indexing the dict directly.
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


def neighbor_lists(root: TreeNode, tol: float = _DEFAULT_TOL) -> dict[TreeNode, list[TreeNode]]:
    """Build the neighbor-list map `L^nei` for the tree rooted at `root`.

    For each node `alpha` at every level of the tree, `L^nei(alpha)` is the
    list of all same-level nodes `beta` (including `alpha` itself) whose
    `bounding_box` touches or overlaps `alpha`'s `bounding_box`, per
    `boxes_adjacent`.

    Args:
        root: Root of the geometric cluster tree (e.g. from `build_tree`).
        tol: Adjacency tolerance forwarded to `boxes_adjacent`.

    Returns:
        A flat mapping `node -> [neighbor nodes, including the node itself]`,
        with one entry per node in the tree.
    """
    result: dict[TreeNode, list[TreeNode]] = {}
    for level_nodes in root.iter_levels():
        boxes = np.stack([node.bounding_box for node in level_nodes])  # (n, d, 2)
        lo, hi = boxes[:, :, 0], boxes[:, :, 1]  # each (n, d)

        # Same separating-axis test as `boxes_adjacent`, applied to every
        # pair (i, j) at once: gap[i, j, axis] = max(lo_i - hi_j, lo_j - hi_i).
        gap = np.maximum(
            lo[:, None, :] - hi[None, :, :], lo[None, :, :] - hi[:, None, :]
        )  # (n, n, d)
        adjacent = np.all(gap <= tol, axis=-1)  # (n, n)

        for i, alpha in enumerate(level_nodes):
            result[alpha] = [level_nodes[j] for j in np.flatnonzero(adjacent[i])]
    return result
