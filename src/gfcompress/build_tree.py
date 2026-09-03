"""Level-synchronous geometric bisection builder for the dual-index cluster
tree (Task F.1, revising Task 1.3).

Splitting strategy
-------------------
This follows the construction in Levitt & Martinsson (2024), §3 (p.5):
the domain is refined as a **fixed uniform dyadic grid**. Level 0 consists of
a single box -- the root's bounding box, computed once from all of `mesh`'s
centroids. The boxes belonging to level `l + 1` are obtained by bisecting
*every* box of level `l` (not just the ones that still need splitting) along
every spatial axis at that box's **geometric midpoint** (not the median of
its points), producing up to `2^d` smaller boxes. Boxes that contain no
points are omitted. Because every node at a level is bisected in lock-step,
**all leaves end up at the same uniform depth `L`** -- a property the
neighbor/interaction-list machinery (Tasks 1.4/1.5) and Alg. 4.1's "neighbor
pairs in level `L`" rely on.

The loop keeps bisecting the whole frontier as long as *any* node in it still
holds `> m` patches (the paper's threshold; `= m` is already a leaf). A node
whose own bisection yields exactly one non-empty sub-cell (no separation at
this resolution, e.g. two clusters far from the current cell's midpoint)
still advances by one level with that single child -- there is no recursive
"keep bisecting the same node in place" shortcut, since that would let
different branches of the tree reach different depths.

Because every split is at the geometric midpoint of the *current cell* (and
the current cell is itself a dyadic sub-box of the root domain), all boxes
surviving at level `l` are cells of one common dyadic grid: the root domain
subdivided into `2^l` equal parts along each axis. Concretely, each node
carries its **dyadic cell** `[lo, hi]^d`, derived from the root domain box and
the node's path in the tree. Patch membership in a child cell is decided by
comparing each patch's centroid coordinate against the cell's midpoint along
each axis (`< mid` -> lower half, `>= mid` -> upper half). We store this
dyadic cell directly as `TreeNode.bounding_box` (overwriting the
shrink-wrapped centroid bounds that `make_node` initially computes), with
`center`/`diam` recomputed from the cell.

Recursion stops -- the current frontier becomes the leaf level `L` -- once
every node holds `<= m` patches, or `max_depth` levels have been built
(guards against pathological inputs), or refining would be a floating-point
no-op: if a cell's geometric midpoint equals one of its edges on some axis
(the cell has underflowed to numerically zero width, e.g. exactly coincident
centroids), further bisection cannot make progress and could loop without
bound, so building stops and the current frontier is final.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.tree import TreeNode, make_node


def build_tree(mesh: FaultMesh, m: int, max_depth: int = 64) -> TreeNode:
    """Build the geometric bisection cluster tree over `mesh`'s patches.

    Implements the fixed uniform dyadic-grid refinement of Levitt &
    Martinsson (2024), §3, as a level-synchronous loop: level `l + 1` is
    obtained by bisecting *every* box of level `l` along every spatial axis
    at its geometric midpoint, forming up to `2^d` children each. Boxes
    containing no patches are omitted. The loop continues while any node
    still holds `> m` patches, so all leaves land at the same uniform depth.

    Args:
        mesh: The `FaultMesh` providing centroids, `d`, and the
            `patch_to_rows`/`patch_to_cols` index-expansion helpers.
        m: Leaf stop threshold: a node with `<= m` patches does not need
            further splitting (paper's `> m` continuation rule). Must be
            `>= 1`.
        max_depth: Hard cap on the number of levels built, to guarantee
            termination on pathological inputs (e.g. many coincident
            centroids) where patch counts never drop to `<= m`.

    Returns:
        The root `TreeNode` of the cluster tree, with `row_indices` /
        `col_indices` and `index_in_level` populated on every node (root,
        internal, and leaf).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    all_patches = np.arange(mesh.n_patches, dtype=np.intp)
    centroids = mesh.centroids
    # Axes along which every centroid coincides (e.g. a planar fault's
    # constant depth) never separate points and get a tiny cosmetic fallback
    # width in `_root_domain_box`; that width is unrelated to the actual
    # point spread and must never be allowed to trip the underflow guard
    # below, or a large-magnitude constant coordinate (e.g. z=5000) rounds
    # `mid` to `lo` on the very first split and halts the whole build.
    degenerate_axis = (centroids.max(axis=0) - centroids.min(axis=0)) <= 0
    root_cell = _root_domain_box(centroids)
    root = make_node(mesh, all_patches, level=0, parent=None)
    _set_cell_geometry(root, root_cell)
    root.index_in_level = 0

    level_nodes = [root]
    depth = 0
    while (
        any(node.patch_indices.shape[0] > m for node in level_nodes)
        and depth < max_depth
        and not _cell_underflowed(level_nodes, degenerate_axis)
    ):
        next_level: list[TreeNode] = []
        for node in level_nodes:
            partitions = _bisect_cell(centroids, node.patch_indices, node.bounding_box)
            children = []
            for child_patches, child_cell in partitions:
                child = make_node(mesh, child_patches, level=node.level + 1, parent=node)
                _set_cell_geometry(child, child_cell)
                children.append(child)
            node.children = children
            next_level.extend(children)
        for index, node in enumerate(next_level):
            node.index_in_level = index
        level_nodes = next_level
        depth += 1

    if any(node.patch_indices.shape[0] > m for node in level_nodes):
        warnings.warn(
            f"build_tree: stopped with leaves holding > m={m} patches "
            f"(max_depth={max_depth} reached or a cell underflowed); "
            "the tree is not fully refined.",
            stacklevel=2,
        )

    return root


def _cell_underflowed(level_nodes: list[TreeNode], degenerate_axis: NDArray[np.bool_]) -> bool:
    """Whether bisecting any node's cell at its geometric midpoint would be a
    floating-point no-op along some non-degenerate axis (`mid == lo` or
    `mid == hi`), i.e. the cell has underflowed to (numerically) zero width.

    Axes in `degenerate_axis` (constant across every centroid, e.g. a planar
    fault's depth) are excluded: their cell width is a cosmetic fallback
    (see `_root_domain_box`), never shrinks the point spread, and must not
    veto refinement along the other, genuinely discriminating axes.

    Args:
        level_nodes: Nodes of the current frontier, about to be bisected.
        degenerate_axis: Boolean mask, shape `(d,)`, `True` for axes along
            which every centroid coincides.

    Returns:
        `True` if any node's cell cannot be meaningfully bisected further
        along some non-degenerate axis.
    """
    for node in level_nodes:
        lo = node.bounding_box[:, 0]
        hi = node.bounding_box[:, 1]
        mid = 0.5 * (lo + hi)
        stuck = (mid == lo) | (mid == hi)
        if np.any(stuck & ~degenerate_axis):
            return True
    return False


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
    # Scaled to the coordinate's own magnitude (not a bare 1e-12), since a
    # fixed absolute epsilon is meaningless once coordinates are far from
    # the origin (e.g. z=5000 m): this width is purely cosmetic geometry for
    # a degenerate axis and is excluded from the underflow guard regardless.
    fallback = np.maximum(np.abs(mins), 1.0) * 1e-9
    eps = np.where(span > 0, span * 1e-9, fallback)
    return np.stack([mins, maxs + eps], axis=1)


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
