"""Neighbor lists `L^nei` over the geometric cluster tree (Task 1.4).

Per CLAUDE.md, all combinatorics (neighbors, interaction lists, admissibility,
coloring, peeling) run on the `N` boxes of the single geometric cluster tree
and are unchanged by dof multiplicity.

Because `build_tree` constructs a fixed uniform **dyadic grid** --
at every level `l`, every surviving node's `bounding_box` is a cell of one
common `2^l`-per-axis subdivision of the root domain box, so same-level cells
are grid-aligned and congruent -- same-level adjacency reduces to a simple
touch-or-overlap test on axis-aligned boxes (`boxes_adjacent`). Two boxes are
neighbors iff their dyadic cells touch (share a face, edge, or corner) or
overlap; a box is always its own neighbor. On a full grid an interior box has
exactly `3^d` neighbors (itself plus the `3^d - 1` cells sharing a face,
edge, or corner), and a boundary box has fewer.

`neighbor_lists(root)` uses the integer dyadic cell coordinate carried by
each node.  It probes only the `{-1, 0, 1}^d` stencil in an occupied-cell map;
there is no floating-point adjacency tolerance or same-level all-pairs array
in the production path.

`TreeNode` is hashable by identity (Task F.1, `@dataclass(eq=False)`), so the
result is a single flat, node-keyed dict rather than the earlier
`level -> index_in_level -> list` nesting; per-level iteration is available
via `root.nodes_at_level(level)` and indexing the dict directly.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from gfcompress.tree import TreeNode


def boxes_adjacent(
    box_a: NDArray[np.float64], box_b: NDArray[np.float64], tol: float = 0.0
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


def neighbor_lists(root: TreeNode) -> dict[TreeNode, list[TreeNode]]:
    """Build the neighbor-list map `L^nei` for the tree rooted at `root`.

    For each node `alpha` at every level of the tree, `L^nei(alpha)` is the
    list of occupied cells whose integer coordinates differ from
    `alpha.cell_coords` by at most one along every axis.

    Args:
        root: Root of the geometric cluster tree (e.g. from `build_tree`).
    Returns:
        A flat mapping `node -> [neighbor nodes, including the node itself]`,
        with one entry per node in the tree.
    """
    result: dict[TreeNode, list[TreeNode]] = {}
    for level_nodes in root.iter_levels():
        occupied = {node.cell_coords: node for node in level_nodes}
        offsets = tuple(itertools.product((-1, 0, 1), repeat=len(root.cell_coords)))
        for alpha in level_nodes:
            result[alpha] = [
                beta
                for offset in offsets
                if (
                    beta := occupied.get(
                        tuple(a + b for a, b in zip(alpha.cell_coords, offset, strict=True))
                    )
                )
                is not None
            ]
    return result
