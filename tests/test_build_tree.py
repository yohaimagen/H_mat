"""Tests for the geometric bisection tree builder (Task 1.3)."""

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.geometry import FaultMesh
from gfcompress.tree import TreeNode


def _random_mesh(d: int, n: int, seed: int = 0) -> FaultMesh:
    rng = np.random.default_rng(seed)
    centroids = rng.uniform(-1.0, 1.0, size=(n, d))
    L = rng.uniform(0.1, 0.5, size=n)
    return FaultMesh(centroids=centroids, L=L)


def _grid_mesh_2d(nx: int = 8, ny: int = 8) -> FaultMesh:
    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    centroids = np.stack([xs.ravel(), ys.ravel()], axis=1)
    L = np.full(centroids.shape[0], 0.1)
    return FaultMesh(centroids=centroids, L=L)


def _grid_mesh_3d(nx: int = 4, ny: int = 4, nz: int = 4) -> FaultMesh:
    xs, ys, zs = np.meshgrid(
        np.arange(nx, dtype=float), np.arange(ny, dtype=float), np.arange(nz, dtype=float)
    )
    centroids = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    L = np.full(centroids.shape[0], 0.05)
    return FaultMesh(centroids=centroids, L=L)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_root_covers_all_patches_2d() -> None:
    mesh = _random_mesh(d=2, n=50, seed=1)
    root = build_tree(mesh, m=5)

    assert isinstance(root, TreeNode)
    assert root.level == 0
    assert root.parent is None
    np.testing.assert_array_equal(np.sort(root.patch_indices), np.arange(mesh.n_patches))


def test_root_covers_all_patches_3d() -> None:
    mesh = _random_mesh(d=3, n=60, seed=2)
    root = build_tree(mesh, m=5)

    np.testing.assert_array_equal(np.sort(root.patch_indices), np.arange(mesh.n_patches))


def test_leaves_respect_min_size_2d() -> None:
    mesh = _random_mesh(d=2, n=64, seed=3)
    m = 4
    root = build_tree(mesh, m=m)

    for leaf in root.leaves():
        assert leaf.is_leaf
        # Either small enough to stop, or splitting failed to make progress
        # (which cannot happen for distinct random points).
        assert leaf.patch_indices.shape[0] < m


def test_leaves_respect_min_size_3d() -> None:
    mesh = _random_mesh(d=3, n=80, seed=4)
    m = 6
    root = build_tree(mesh, m=m)

    for leaf in root.leaves():
        assert leaf.patch_indices.shape[0] < m


def test_internal_nodes_have_up_to_2_pow_d_children() -> None:
    for d, n, seed in [(2, 64, 5), (3, 80, 6)]:
        mesh = _random_mesh(d=d, n=n, seed=seed)
        root = build_tree(mesh, m=4)
        max_children = 2**d
        for level_nodes in root.iter_levels():
            for node in level_nodes:
                if not node.is_leaf:
                    assert 1 < len(node.children) <= max_children


# ---------------------------------------------------------------------------
# Grid alignment: split planes lie on the dyadic refinement of the root
# domain box (paper §3, p.5: bisect each box at its geometric midpoint along
# every axis -- NOT the median of its points). This is the regression that
# distinguishes geometric-midpoint bisection from coordinate-wise-median
# bisection: on a non-uniform mesh, median splits would NOT be dyadic
# subdivisions of the root box, and sibling cells from different parents
# would not share grid lines.
# ---------------------------------------------------------------------------


def _assert_dyadic_subdivision(
    child_lo: float, child_hi: float, root_lo: float, root_hi: float, tol: float = 1e-9
) -> None:
    """Assert that `[child_lo, child_hi]` is a dyadic sub-interval of
    `[root_lo, root_hi]`, i.e. both endpoints are `root_lo + k * width /
    2**p` for some common non-negative integer `p` and integers `k`."""
    width = root_hi - root_lo
    assert width > 0
    for edge in (child_lo, child_hi):
        frac = (edge - root_lo) / width
        # frac must be k / 2**p for some integers k, p >= 0; search small p.
        for p in range(0, 60):
            scaled = frac * (2**p)
            if abs(scaled - round(scaled)) < tol * (2**p):
                break
        else:
            raise AssertionError(
                f"edge {edge} (frac={frac}) is not a dyadic subdivision of "
                f"[{root_lo}, {root_hi}]"
            )


def _assert_children_tile_parent_at_midpoint(node: TreeNode) -> None:
    """Assert that every child's `bounding_box` is obtained by bisecting
    `node`'s `bounding_box` at its geometric midpoint along every axis (with
    possibly several such bisections collapsed if intermediate cells were
    empty along the way -- so each child cell must itself be a dyadic
    sub-cell of `node`'s cell, on the side of `node`'s midpoint consistent
    with where its centroids lie)."""
    parent_box = node.bounding_box
    d = parent_box.shape[0]
    parent_mid = 0.5 * (parent_box[:, 0] + parent_box[:, 1])

    for child in node.children:
        child_box = child.bounding_box
        for axis in range(d):
            p_lo, p_hi = parent_box[axis]
            c_lo, c_hi = child_box[axis]
            mid = parent_mid[axis]
            # The child cell must be entirely within the parent's lower half
            # [p_lo, mid] or entirely within the upper half [mid, p_hi]
            # (up to floating-point tolerance), i.e. the parent's midpoint is
            # itself one of the child's bounding edges or outside the child
            # cell on the appropriate side.
            tol = 1e-9 * max(p_hi - p_lo, 1.0)
            in_lower = c_hi <= mid + tol
            in_upper = c_lo >= mid - tol
            assert in_lower or in_upper, (
                f"child cell axis {axis} [{c_lo}, {c_hi}] straddles parent "
                f"midpoint {mid} of [{p_lo}, {p_hi}]"
            )
            # And the child cell must lie within the parent cell.
            assert c_lo >= p_lo - tol and c_hi <= p_hi + tol


def test_split_planes_are_dyadic_subdivisions_of_root_2d() -> None:
    mesh = _random_mesh(d=2, n=60, seed=21)
    root = build_tree(mesh, m=4)
    root_box = root.bounding_box

    def check(node: TreeNode) -> None:
        for axis in range(root_box.shape[0]):
            lo, hi = node.bounding_box[axis]
            _assert_dyadic_subdivision(lo, hi, root_box[axis, 0], root_box[axis, 1])
        _assert_children_tile_parent_at_midpoint(node)
        for child in node.children:
            check(child)

    check(root)


def test_split_planes_are_dyadic_subdivisions_of_root_3d() -> None:
    mesh = _random_mesh(d=3, n=70, seed=22)
    root = build_tree(mesh, m=5)
    root_box = root.bounding_box

    def check(node: TreeNode) -> None:
        for axis in range(root_box.shape[0]):
            lo, hi = node.bounding_box[axis]
            _assert_dyadic_subdivision(lo, hi, root_box[axis, 0], root_box[axis, 1])
        _assert_children_tile_parent_at_midpoint(node)
        for child in node.children:
            check(child)

    check(root)


def test_sibling_subtrees_share_grid_lines() -> None:
    """At a given level, the distinct split-plane coordinates per axis must
    come from one common dyadic grid: cells from different parent subtrees
    must share grid lines, not just be internally consistent with their own
    parent."""
    mesh = _random_mesh(d=2, n=80, seed=23)
    root = build_tree(mesh, m=3)
    root_box = root.bounding_box
    d = root_box.shape[0]

    for level_nodes in root.iter_levels():
        if len(level_nodes) <= 1:
            continue
        for axis in range(d):
            # Collect the dyadic "k / 2**p" coordinate of each node's lower
            # edge on this axis, all at a common resolution p_max sufficient
            # for every node at this level.
            p_max = 0
            for node in level_nodes:
                lo, hi = node.bounding_box[axis]
                width = hi - lo
                root_width = root_box[axis, 1] - root_box[axis, 0]
                ratio = root_width / width
                p = round(np.log2(ratio))
                p_max = max(p_max, p)

            scale = 2**p_max
            root_lo = root_box[axis, 0]
            root_width = root_box[axis, 1] - root_box[axis, 0]
            for node in level_nodes:
                lo, hi = node.bounding_box[axis]
                for edge in (lo, hi):
                    k = (edge - root_lo) / root_width * scale
                    assert abs(k - round(k)) < 1e-6, (
                        f"edge {edge} on axis {axis} at level {node.level} is "
                        f"not aligned to the common 1/2**{p_max} grid"
                    )


def test_uniform_grid_produces_perfect_dyadic_grid() -> None:
    mesh = _grid_mesh_2d(8, 8)
    root = build_tree(mesh, m=4)
    root_box = root.bounding_box

    def check(node: TreeNode) -> None:
        for axis in range(root_box.shape[0]):
            lo, hi = node.bounding_box[axis]
            _assert_dyadic_subdivision(lo, hi, root_box[axis, 0], root_box[axis, 1])
        _assert_children_tile_parent_at_midpoint(node)
        for child in node.children:
            check(child)

    check(root)


# ---------------------------------------------------------------------------
# Critical invariant: leaf row/col index sets exactly partition {0..dof*N-1}.
# ---------------------------------------------------------------------------


def _check_leaf_index_partition(root: TreeNode, mesh: FaultMesh) -> None:
    leaves = root.leaves()

    # Patch indices: disjoint and complete over {0..N-1}.
    all_patches = np.concatenate([leaf.patch_indices for leaf in leaves])
    assert np.unique(all_patches).shape[0] == all_patches.shape[0]
    np.testing.assert_array_equal(np.sort(all_patches), np.arange(mesh.n_patches))

    # Row indices: disjoint and complete over {0..dof_row*N-1}.
    all_rows = np.concatenate([leaf.row_indices for leaf in leaves])
    assert np.unique(all_rows).shape[0] == all_rows.shape[0]
    np.testing.assert_array_equal(np.sort(all_rows), np.arange(mesh.n_rows))

    # Col indices: disjoint and complete over {0..dof_col*N-1}.
    all_cols = np.concatenate([leaf.col_indices for leaf in leaves])
    assert np.unique(all_cols).shape[0] == all_cols.shape[0]
    np.testing.assert_array_equal(np.sort(all_cols), np.arange(mesh.n_cols))


def test_leaf_index_partition_2d_random() -> None:
    mesh = _random_mesh(d=2, n=53, seed=7)
    root = build_tree(mesh, m=5)
    assert mesh.n_rows == 2 * mesh.n_patches
    assert mesh.n_cols == mesh.n_patches
    _check_leaf_index_partition(root, mesh)


def test_leaf_index_partition_3d_random() -> None:
    mesh = _random_mesh(d=3, n=71, seed=8)
    root = build_tree(mesh, m=6)
    assert mesh.n_rows == 3 * mesh.n_patches
    assert mesh.n_cols == 2 * mesh.n_patches
    _check_leaf_index_partition(root, mesh)


def test_leaf_index_partition_2d_grid() -> None:
    mesh = _grid_mesh_2d(8, 8)
    root = build_tree(mesh, m=4)
    _check_leaf_index_partition(root, mesh)


def test_leaf_index_partition_3d_grid() -> None:
    mesh = _grid_mesh_3d(4, 4, 4)
    root = build_tree(mesh, m=4)
    _check_leaf_index_partition(root, mesh)


def test_leaf_index_partition_various_m() -> None:
    mesh = _random_mesh(d=2, n=37, seed=9)
    for m in (1, 2, 3, 10, 50, 100):
        root = build_tree(mesh, m=m)
        _check_leaf_index_partition(root, mesh)


# ---------------------------------------------------------------------------
# Internal-node index sets are unions of their children's index sets.
# ---------------------------------------------------------------------------


def test_internal_node_indices_are_union_of_children() -> None:
    mesh = _random_mesh(d=2, n=40, seed=10)
    root = build_tree(mesh, m=3)

    def check(node: TreeNode) -> None:
        if node.is_leaf:
            return
        child_rows = np.concatenate([c.row_indices for c in node.children])
        child_cols = np.concatenate([c.col_indices for c in node.children])
        np.testing.assert_array_equal(np.sort(child_rows), np.sort(node.row_indices))
        np.testing.assert_array_equal(np.sort(child_cols), np.sort(node.col_indices))
        for c in node.children:
            check(c)

    check(root)


# ---------------------------------------------------------------------------
# Single-patch / tiny meshes (degenerate cases).
# ---------------------------------------------------------------------------


def test_single_patch_mesh_is_a_leaf() -> None:
    mesh = _random_mesh(d=2, n=1, seed=11)
    root = build_tree(mesh, m=1)
    assert root.is_leaf
    np.testing.assert_array_equal(root.patch_indices, np.array([0]))
    _check_leaf_index_partition(root, mesh)


def test_small_mesh_smaller_than_m_is_a_leaf() -> None:
    mesh = _random_mesh(d=2, n=3, seed=12)
    root = build_tree(mesh, m=10)
    assert root.is_leaf
    _check_leaf_index_partition(root, mesh)


def test_degenerate_collinear_centroids_terminates() -> None:
    # All centroids share the same x-coordinate: bisection along x makes no
    # progress, but bisection along y still splits the points.
    centroids = np.array([[0.0, float(i)] for i in range(10)])
    L = np.full(10, 0.1)
    mesh = FaultMesh(centroids=centroids, L=L)
    root = build_tree(mesh, m=2)
    _check_leaf_index_partition(root, mesh)


def test_all_identical_centroids_terminates_as_single_leaf() -> None:
    centroids = np.zeros((5, 2))
    L = np.full(5, 0.1)
    mesh = FaultMesh(centroids=centroids, L=L)
    root = build_tree(mesh, m=2)
    # No axis can separate identical points: split makes no progress, so the
    # root remains a single leaf despite having >= m patches.
    assert root.is_leaf
    _check_leaf_index_partition(root, mesh)
