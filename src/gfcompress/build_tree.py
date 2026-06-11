"""Recursive geometric bisection builder for the dual-index cluster tree
(Task 1.3).

Splitting strategy
-------------------
Per CLAUDE.md / PLAN.md, downstream tasks (1.4/1.5) need a *regular-grid*
structure: same-level boxes have ``<= 3^d`` neighbors and ``<= 6^d``
interaction-list entries. A binary kd-tree (one axis split per level, two
children) does not give that directly -- two levels of a kd-tree are needed
to make a "grid refinement" in 2D, three in 3D, which shifts the regular-grid
counts to alternating levels.

Instead we build a **quadtree (2D) / octree (3D)**: at each internal node we
split *every* spatial axis simultaneously at its median, producing up to
``2^d`` child cells arranged on a regular ``2 x ... x 2`` sub-grid of the
parent's bounding box. This is the "uniform-style partition into ``2^d``
children" called for in PLAN.md, and it gives every level its own regular
grid of boxes, which is what the neighbor/interaction-list machinery
(Tasks 1.4/1.5) and the fixed periodic test patterns (Stage 4) assume.

Splitting along all ``d`` axes at once requires choosing a median per axis.
We use the coordinate-wise median of the node's centroids (so each axis is
split as evenly as possible), assign each centroid to a cell based on which
side of each per-axis median it falls on, and drop empty cells (so a node
may have fewer than ``2^d`` children if centroids are degenerate along some
axis, e.g. all share the same x-coordinate). "Bisecting along the longest
axis" is realized degenerately when only one axis actually separates the
points; in the common case all ``d`` axes are split, matching the
``2^d``-children description.

Recursion stops -- the node becomes a leaf -- once its patch count is
``< m``.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.tree import TreeNode, make_node


def build_tree(mesh: FaultMesh, m: int) -> TreeNode:
    """Build the geometric bisection cluster tree over `mesh`'s patches.

    Recursively splits the patch set by bisecting every spatial axis at its
    median, producing up to `2^d` children arranged on a regular sub-grid of
    the parent's bounding box (a quadtree in 2D, an octree in 3D). Recursion
    stops once a node holds fewer than `m` patches, which becomes a leaf.

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
    root = make_node(mesh, all_patches, level=0, parent=None)
    _split(mesh, root, m)
    return root


def _split(mesh: FaultMesh, node: TreeNode, m: int) -> None:
    """Recursively split `node` in place, attaching children until every
    leaf has `< m` patches."""
    patch_indices = node.patch_indices
    if patch_indices.shape[0] < m:
        return

    child_partitions = _bisect_patches(mesh.centroids[patch_indices], patch_indices)

    # If the split failed to make progress (all centroids identical, so a
    # single "child" contains every patch), stop: further recursion would
    # not terminate.
    if len(child_partitions) <= 1:
        return

    for child_patches in child_partitions:
        child = make_node(mesh, child_patches, level=node.level + 1, parent=node)
        node.children.append(child)
        _split(mesh, child, m)


def _bisect_patches(
    centroids: NDArray[np.float64], patch_indices: NDArray[np.intp]
) -> list[NDArray[np.intp]]:
    """Partition `patch_indices` into up to `2^d` cells by bisecting every
    axis of `centroids` at its median.

    Args:
        centroids: Centroids of the patches in `node`, shape `(n, d)`.
        patch_indices: The corresponding global patch indices, shape `(n,)`.

    Returns:
        A list of integer arrays (subsets of `patch_indices`), one per
        non-empty cell of the `2 x ... x 2` (`d` axes) sub-grid. Cells with
        no centroids are omitted, so the result has between `1` and `2^d`
        entries.
    """
    d = centroids.shape[1]
    medians = np.median(centroids, axis=0)

    # side[:, axis] = 0 if centroid is on the lower (<= median) side of that
    # axis's median, 1 if on the upper (> median) side.
    side = (centroids > medians[None, :]).astype(np.intp)

    partitions: list[NDArray[np.intp]] = []
    for cell in itertools.product((0, 1), repeat=d):
        mask = np.all(side == np.array(cell, dtype=np.intp)[None, :], axis=1)
        if np.any(mask):
            partitions.append(patch_indices[mask])
    return partitions
