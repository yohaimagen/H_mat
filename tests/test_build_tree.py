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
