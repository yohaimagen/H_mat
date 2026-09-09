"""Level-synchronous geometric bisection builder for the dual-index cluster
tree (Task C.1).

Splitting strategy
-------------------
This implementation uses the paper's dyadic construction in Levitt &
Martinsson (2024), §3 (p.5), with one simplifying adaptation:
**level-synchronous subdivision to a common leaf depth**. This is the
implementation's choice for its neighbor/interaction and leaf-sampling
machinery, not a universal requirement of the paper.

The domain is refined as a **fixed uniform dyadic grid**. Level 0 is a padded
physical hypercube enclosing the centroids. The boxes belonging to level
`l + 1` are obtained by bisecting
*every* box of level `l` (not just the ones that still need splitting) along
every spatial axis at that box's **geometric midpoint** (not the median of
its points), producing up to `2^d` smaller boxes. Boxes that contain no
points are omitted. Because every node at a level is bisected in lock-step,
**this implementation's leaves end up at the same uniform depth `L`** -- a property the
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

    Implements fixed uniform dyadic-grid refinement inspired by Levitt &
    Martinsson (2024), §3, as this implementation's level-synchronous loop:
    level `l + 1` is
    obtained by bisecting *every* box of level `l` along every spatial axis
    at its geometric midpoint, forming up to `2^d` children each. Boxes
    containing no patches are omitted. The loop continues while any node
    still holds `> m` patches, so this implementation's leaves land at a
    common depth. This common-depth policy is a simplifying implementation
    adaptation, not a universal paper requirement.

    Args:
        mesh: The `FaultMesh` providing centroids, `tree_dim`, and the
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
    degenerate_axis = centroids.max(axis=0) == centroids.min(axis=0)
    root_cell = _root_domain_box(centroids)
    root = make_node(mesh, all_patches, level=0, parent=None)
    _set_cell_geometry(root, root_cell)
    root.index_in_level = 0
    root.cell_coords = (0,) * mesh.tree_dim

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
            for child_patches, child_cell, half in partitions:
                child = make_node(mesh, child_patches, level=node.level + 1, parent=node)
                _set_cell_geometry(child, child_cell)
                child.cell_coords = tuple(
                    2 * coord + direction
                    for coord, direction in zip(node.cell_coords, half, strict=True)
                )
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
        mid = _midpoint(lo, hi)
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
    node.center = _midpoint(cell[:, 0], cell[:, 1])
    with np.errstate(over="ignore"):
        diam = float(np.hypot.reduce(cell[:, 1] - cell[:, 0]))
    if not np.isfinite(diam):
        raise ValueError("dyadic cell diagonal cannot be represented as a finite float")
    node.diam = diam


def _root_domain_box(centroids: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the root domain box `[lo, hi]^d` enclosing all `centroids`.

    The cube is centered on the centroid bounding box and uses the largest
    physical span on every axis. Its side is moved to the next representable
    float so points on an extreme remain inside the padded root.

    Args:
        centroids: Centroids of all patches, shape `(N, d)`.

    Returns:
        Array of shape `(d, 2)`, enclosing the centroids with one common
        physical side length in all retained coordinates.
    """
    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    with np.errstate(over="ignore"):
        spans = maxs - mins
    if not np.all(np.isfinite(spans)):
        raise ValueError("centroid extent cannot be represented by a finite root hypercube")
    center = _midpoint(mins, maxs)
    span = float(np.max(spans))
    if span == 0.0:
        # A point cloud still needs a representable cell for the depth guard.
        span = float(np.maximum(np.max(np.abs(center)), 1.0) * 1e-9)
    # Increase the common side by the few ULPs needed to make the rounded
    # endpoints enclose every extreme coordinate. Computing the midpoint as
    # `lo + (hi - lo) / 2` avoids overflow for large positive coordinates.
    side = np.nextafter(span, np.inf)
    while np.isfinite(side):
        half_side = 0.5 * side
        lo = center - half_side
        hi = center + half_side
        if (
            np.all(np.isfinite(lo))
            and np.all(np.isfinite(hi))
            and np.all(lo <= mins)
            and np.all(hi >= maxs)
        ):
            return np.stack([lo, hi], axis=1)
        next_side = np.nextafter(side, np.inf)
        if next_side == side:
            break
        side = next_side
    raise ValueError("centroids cannot be enclosed by a finite padded root hypercube")


def _midpoint(lo: NDArray[np.float64], hi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a finite midpoint when the interval extent is representable."""
    return lo + 0.5 * (hi - lo)


def _bisect_cell(
    centroids: NDArray[np.float64],
    patch_indices: NDArray[np.intp],
    cell: NDArray[np.float64],
) -> list[tuple[NDArray[np.intp], NDArray[np.float64], tuple[int, ...]]]:
    """Bisect `cell` along every axis at its geometric midpoint, partitioning
    `patch_indices` into the resulting `2^d` sub-cells by centroid.

    Args:
        centroids: Centroids of all patches, shape `(N, d)`.
        patch_indices: Global patch indices covered by `cell`, shape `(n,)`.
        cell: The cell to bisect, shape `(d, 2)`, `cell[i] = (lo_i, hi_i)`.

    Returns:
        A list of `(child_patch_indices, child_cell, half)` triples, one per
        non-empty sub-cell of the `2 x ... x 2` (`d` axes) dyadic refinement
        of `cell`. Sub-cells with no centroids are omitted, so the result has
        between `1` and `2^d` entries.
    """
    d = cell.shape[0]
    lo = cell[:, 0]
    hi = cell[:, 1]
    mid = _midpoint(lo, hi)

    pts = centroids[patch_indices]
    # side[:, axis] = 0 if centroid is on the lower (< mid) side of that
    # axis's midpoint, 1 if on the upper (>= mid) side.
    side = (pts >= mid[None, :]).astype(np.intp)

    partitions: list[tuple[NDArray[np.intp], NDArray[np.float64], tuple[int, ...]]] = []
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
        partitions.append((patch_indices[mask], child_cell, tuple(int(x) for x in half_arr)))
    return partitions
