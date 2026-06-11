"""Recursive geometric bisection builder for the dual-index cluster tree
(Task 1.3).

Splitting strategy
-------------------
This follows the construction in Levitt & Martinsson (2024), §3 (p.5):
the domain is refined as a **fixed uniform dyadic grid**. Level 0 consists of
a single box -- the root's bounding box, computed once from all of `mesh`'s
centroids. The boxes belonging to level `l + 1` are obtained by bisecting
each box of level `l` along *every* spatial axis at that box's **geometric
midpoint** (not the median of its points), producing up to `2^d` smaller
boxes. Boxes that contain no points are omitted. The splitting procedure is
applied recursively to boxes that contain `>= m` points; a box with `< m`
points becomes a leaf.

Because every split is at the geometric midpoint of the *current cell* (and
the current cell is itself a dyadic sub-box of the root domain), all boxes
surviving at level `l` are cells of one common dyadic grid: the root domain
subdivided into `2^l` equal parts along each axis. This is the regular-grid
structure that the neighbor list (`<= 3^d` entries, Task 1.4) and interaction
list (`<= 6^d - 3^d` entries, Task 1.5) machinery -- and the fixed periodic
test patterns of Stage 4 -- rely on.

Concretely, each node carries its **dyadic cell** `[lo, hi]^d`, derived from
the root domain box and the node's path in the tree (root cell -> bisect at
its midpoint -> child cell -> ...). Patch membership in a child cell is
decided by comparing each patch's centroid coordinate against the cell's
midpoint along each axis (`< mid` -> lower half, `>= mid` -> upper half).
The geometry used for splitting is therefore always the dyadic cell, never a
node's shrink-wrapped centroid bounding box.

We store this dyadic cell directly as `TreeNode.bounding_box` (overwriting
the shrink-wrapped centroid bounds that `make_node` initially computes), with
`center`/`diam` recomputed from the cell. This is the choice called for by
the design note in Task 1.3: downstream neighbor/interaction-list machinery
(Tasks 1.4/1.5) needs *congruent, grid-aligned* boxes whose split planes
coincide across siblings -- a property the dyadic cell guarantees and a
shrink-wrapped centroid bounding box does not.

Recursion stops -- the node becomes a leaf -- once its patch count is `< m`,
or if a split would make no progress (all patches fall into a single child
cell, e.g. coincident centroids), to guarantee termination.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.tree import TreeNode, make_node


def build_tree(mesh: FaultMesh, m: int) -> TreeNode:
    """Build the geometric bisection cluster tree over `mesh`'s patches.

    Implements the fixed uniform dyadic-grid refinement of Levitt &
    Martinsson (2024), §3: level 0 is a single box (the bounding box of all
    centroids); level `l + 1` boxes are obtained by bisecting each level-`l`
    box along every spatial axis at its geometric midpoint, forming up to
    `2^d` children. Boxes containing no patches are omitted. Recursion stops
    once a box holds fewer than `m` patches, which becomes a leaf.

    Args:
        mesh: The `FaultMesh` providing centroids, `d`, and the
            `patch_to_rows`/`patch_to_cols` index-expansion helpers.
        m: Leaf stop threshold: a node with `< m` patches is not split
            further. Must be `>= 1`.

    Returns:
        The root `TreeNode` of the cluster tree, with `row_indices` /
        `col_indices` populated on every node (root, internal, and leaf).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    all_patches = np.arange(mesh.n_patches, dtype=np.intp)
    root_cell = _root_domain_box(mesh.centroids)
    root = make_node(mesh, all_patches, level=0, parent=None)
    _set_cell_geometry(root, root_cell)
    _split(mesh, root, m, root_cell)
    return root


def _set_cell_geometry(node: TreeNode, cell: NDArray[np.float64]) -> None:
    """Overwrite `node`'s `bounding_box`/`center`/`diam` with those of its
    dyadic `cell`, replacing the shrink-wrapped centroid bounds that
    `make_node` initially computes.

    Args:
        node: The node to update in place.
        cell: The node's dyadic cell, shape `(d, 2)`, `cell[i] = (lo_i,
            hi_i)`.
    """
    node.bounding_box = cell
    node.center = 0.5 * (cell[:, 0] + cell[:, 1])
    node.diam = float(np.linalg.norm(cell[:, 1] - cell[:, 0]))


def _root_domain_box(centroids: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the root domain box `[lo, hi]^d` enclosing all `centroids`.

    The upper edge of each axis is nudged up by a tiny relative amount so
    that points lying exactly on the global maximum are consistently
    assigned to the *lower* half-cell at every split (the dyadic split rule
    used by `_bisect_cell` is `coord < mid` -> lower, `coord >= mid` ->
    upper), matching the half-open `[lo, hi)` convention for all but the
    final cell.

    Args:
        centroids: Centroids of all patches, shape `(N, d)`.

    Returns:
        Array of shape `(d, 2)`, `box[i] = (lo_i, hi_i)`.
    """
    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    span = maxs - mins
    # Guard against a zero-width axis (all centroids share that coordinate)
    # by giving it a tiny nonzero width so midpoint splits are well defined.
    eps = np.where(span > 0, span * 1e-9, 1e-12)
    return np.stack([mins, maxs + eps], axis=1)


def _split(mesh: FaultMesh, node: TreeNode, m: int, cell: NDArray[np.float64]) -> None:
    """Recursively split `node` in place, attaching children until every
    leaf has `< m` patches.

    Args:
        mesh: The `FaultMesh` providing centroids and index helpers.
        node: The node to (possibly) split.
        m: Leaf stop threshold.
        cell: This node's dyadic cell, shape `(d, 2)`, `cell[i] = (lo_i,
            hi_i)`.
    """
    patch_indices = node.patch_indices
    if patch_indices.shape[0] < m:
        return

    # A single non-empty sub-cell covering everyone means the dyadic split
    # at this resolution did not separate the patches. If the cell still has
    # nonzero width, keep bisecting *in place* (without creating a sibling)
    # -- this may eventually separate distinct-but-close centroids while
    # keeping every split plane on the dyadic grid. If a bisection is a
    # geometric no-op (`child_cell == cell`, i.e. the cell has underflowed
    # to zero width, e.g. coincident centroids), stop to guarantee
    # termination.
    while True:
        child_partitions = _bisect_cell(mesh.centroids, patch_indices, cell)
        if len(child_partitions) != 1:
            break
        _, only_cell = child_partitions[0]
        if np.array_equal(only_cell, cell):
            return
        cell = only_cell

    for child_patches, child_cell in child_partitions:
        child = make_node(mesh, child_patches, level=node.level + 1, parent=node)
        _set_cell_geometry(child, child_cell)
        node.children.append(child)
        _split(mesh, child, m, child_cell)


def _bisect_cell(
    centroids: NDArray[np.float64],
    patch_indices: NDArray[np.intp],
    cell: NDArray[np.float64],
) -> list[tuple[NDArray[np.intp], NDArray[np.float64]]]:
    """Bisect `cell` along every axis at its geometric midpoint, partitioning
    `patch_indices` into the resulting `2^d` sub-cells by centroid.

    Args:
        centroids: Centroids of all patches, shape `(N, d)`.
        patch_indices: Global patch indices covered by `cell`, shape `(n,)`.
        cell: The cell to bisect, shape `(d, 2)`, `cell[i] = (lo_i, hi_i)`.

    Returns:
        A list of `(child_patch_indices, child_cell)` pairs, one per
        non-empty sub-cell of the `2 x ... x 2` (`d` axes) dyadic refinement
        of `cell`. Sub-cells with no centroids are omitted, so the result has
        between `1` and `2^d` entries.
    """
    d = cell.shape[0]
    lo = cell[:, 0]
    hi = cell[:, 1]
    mid = 0.5 * (lo + hi)

    pts = centroids[patch_indices]
    # side[:, axis] = 0 if centroid is on the lower (< mid) side of that
    # axis's midpoint, 1 if on the upper (>= mid) side.
    side = (pts >= mid[None, :]).astype(np.intp)

    partitions: list[tuple[NDArray[np.intp], NDArray[np.float64]]] = []
    for half in itertools.product((0, 1), repeat=d):
        half_arr = np.array(half, dtype=np.intp)
        mask = np.all(side == half_arr[None, :], axis=1)
        if not np.any(mask):
            continue
        child_cell = np.empty_like(cell)
        for axis in range(d):
            if half_arr[axis] == 0:
                child_cell[axis] = (lo[axis], mid[axis])
            else:
                child_cell[axis] = (mid[axis], hi[axis])
        partitions.append((patch_indices[mask], child_cell))
    return partitions
