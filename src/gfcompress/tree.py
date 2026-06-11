"""`TreeNode`: container for the dual-index geometric cluster tree.

A single geometric cluster tree over the `N` patch centroids serves as both
the row tree and the column tree. Each node holds the set of
patch indices it covers (`patch_indices`) plus their patch-major flattened
row/column index sets (`row_indices`/`col_indices`, via
`FaultMesh.patch_to_rows`/`patch_to_cols`), axis-aligned bounding-box
geometry (`bounding_box`, `center`, `diam`), and parent/child links.

This module provides only the node container, geometry helpers, and
traversal utilities. The recursive bisection builder is Task 1.3.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh


@dataclass
class TreeNode:
    """A node of the geometric cluster tree.

    Attributes:
        patch_indices: Integer array of patch indices covered by this node,
            shape `(n_patches_in_node,)`.
        row_indices: Flattened row indices (`dof_row` per patch, patch-major
            block-interleaved), shape `(dof_row * n_patches_in_node,)`.
        col_indices: Flattened column indices (`dof_col` per patch,
            patch-major block-interleaved), shape
            `(dof_col * n_patches_in_node,)`.
        level: Depth of this node in the tree (root is level 0).
        bounding_box: Axis-aligned bounding box of the node's patch
            centroids, shape `(d, 2)` with `bounding_box[i] = (min_i, max_i)`.
        center: Center of the bounding box, shape `(d,)`.
        diam: Diameter (Euclidean length of the bounding-box diagonal).
        parent: Parent node, or `None` for the root.
        children: Child nodes (empty for a leaf).
    """

    patch_indices: NDArray[np.intp]
    row_indices: NDArray[np.intp]
    col_indices: NDArray[np.intp]
    level: int
    bounding_box: NDArray[np.float64]
    center: NDArray[np.float64]
    diam: float
    parent: TreeNode | None = None
    children: list[TreeNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""
        return len(self.children) == 0

    def leaves(self) -> list[TreeNode]:
        """Return all leaf nodes in the subtree rooted at `self`, in
        depth-first, left-to-right order."""
        if self.is_leaf:
            return [self]
        result: list[TreeNode] = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def nodes_at_level(self, level: int) -> list[TreeNode]:
        """Return all nodes in the subtree rooted at `self` at depth `level`
        (relative to the tree's root, i.e. matching `node.level == level`).

        If `level < self.level`, returns an empty list. If `level ==
        self.level`, returns `[self]`.
        """
        if level < self.level:
            return []
        if level == self.level:
            return [self]
        result: list[TreeNode] = []
        for child in self.children:
            result.extend(child.nodes_at_level(level))
        return result

    def iter_levels(self) -> Iterator[list[TreeNode]]:
        """Iterate over the subtree level by level (breadth-first), starting
        at `self.level`.

        Yields:
            Lists of nodes, one list per level, ordered from `self.level`
            down to the deepest level present in the subtree.
        """
        current = [self]
        while current:
            yield current
            nxt: list[TreeNode] = []
            for node in current:
                nxt.extend(node.children)
            current = nxt


def make_node(
    mesh: FaultMesh,
    patch_indices: NDArray[np.integer],
    level: int,
    parent: TreeNode | None = None,
) -> TreeNode:
    """Construct a `TreeNode` for the given patch subset, computing its
    row/column index sets and bounding-box geometry from `mesh`.

    Args:
        mesh: The `FaultMesh` providing centroids and index-expansion
            helpers.
        patch_indices: Integer array of patch indices covered by the node.
        level: Depth of the node in the tree.
        parent: Parent node, or `None` for the root.

    Returns:
        A `TreeNode` with `children=[]`; the caller (recursive builder) is
        responsible for attaching children.
    """
    patch_indices = np.asarray(patch_indices, dtype=np.intp).reshape(-1)
    row_indices = mesh.patch_to_rows(patch_indices)
    col_indices = mesh.patch_to_cols(patch_indices)
    bounding_box, center, diam = _compute_geometry(mesh.centroids[patch_indices])
    return TreeNode(
        patch_indices=patch_indices,
        row_indices=row_indices,
        col_indices=col_indices,
        level=level,
        bounding_box=bounding_box,
        center=center,
        diam=diam,
        parent=parent,
        children=[],
    )


def _compute_geometry(
    centroids: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute the axis-aligned bounding box, center, and diameter of a set
    of centroids.

    Args:
        centroids: Array of shape `(n, d)`, `n >= 1`.

    Returns:
        Tuple `(bounding_box, center, diam)`:
            - `bounding_box`: shape `(d, 2)`, `bounding_box[i] = (min_i, max_i)`.
            - `center`: shape `(d,)`, midpoint of the bounding box.
            - `diam`: Euclidean length of the bounding-box diagonal.
    """
    if centroids.ndim != 2 or centroids.shape[0] == 0:
        raise ValueError(f"centroids must have shape (n, d) with n >= 1, got {centroids.shape}")
    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    bounding_box = np.stack([mins, maxs], axis=1)
    center = 0.5 * (mins + maxs)
    diam = float(np.linalg.norm(maxs - mins))
    return bounding_box, center, diam
