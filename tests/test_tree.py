"""Tests for `TreeNode` container, geometry, and traversal helpers (Task 1.2)."""

import numpy as np

from gfcompress.geometry import FaultMesh
from gfcompress.tree import TreeNode, make_node


def _grid_mesh_2d() -> FaultMesh:
    # 4 patches at the corners of a unit square.
    centroids = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
        ]
    )
    L = np.full(4, 0.1)
    return FaultMesh(centroids=centroids, L=L)


def _grid_mesh_3d() -> FaultMesh:
    centroids = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    L = np.full(5, 0.2)
    return FaultMesh(centroids=centroids, L=L)


# ---------------------------------------------------------------------------
# make_node: index sets
# ---------------------------------------------------------------------------


def test_make_node_index_sets_2d() -> None:
    mesh = _grid_mesh_2d()
    patch_ids = np.array([0, 2, 3])
    node = make_node(mesh, patch_ids, level=0)

    np.testing.assert_array_equal(node.patch_indices, patch_ids)
    np.testing.assert_array_equal(node.row_indices, mesh.patch_to_rows(patch_ids))
    np.testing.assert_array_equal(node.col_indices, mesh.patch_to_cols(patch_ids))

    assert node.row_indices.shape == (mesh.dof_row * len(patch_ids),)
    assert node.col_indices.shape == (mesh.dof_col * len(patch_ids),)


def test_make_node_index_sets_3d() -> None:
    mesh = _grid_mesh_3d()
    patch_ids = np.array([1, 3, 4])
    node = make_node(mesh, patch_ids, level=1)

    np.testing.assert_array_equal(node.row_indices, mesh.patch_to_rows(patch_ids))
    np.testing.assert_array_equal(node.col_indices, mesh.patch_to_cols(patch_ids))
    assert node.row_indices.shape == (3 * len(patch_ids),)
    assert node.col_indices.shape == (2 * len(patch_ids),)
    assert node.level == 1


# ---------------------------------------------------------------------------
# Geometry: bounding_box, center, diam
# ---------------------------------------------------------------------------


def test_geometry_full_set_2d() -> None:
    mesh = _grid_mesh_2d()
    node = make_node(mesh, np.arange(4), level=0)

    np.testing.assert_allclose(node.bounding_box, np.array([[0.0, 2.0], [0.0, 1.0]]))
    np.testing.assert_allclose(node.center, np.array([1.0, 0.5]))
    # Diagonal of a 2x1 box: sqrt(4 + 1) = sqrt(5).
    np.testing.assert_allclose(node.diam, np.sqrt(5.0))


def test_geometry_single_patch_zero_diam() -> None:
    mesh = _grid_mesh_2d()
    node = make_node(mesh, np.array([1]), level=2)

    np.testing.assert_allclose(node.bounding_box, np.array([[2.0, 2.0], [0.0, 0.0]]))
    np.testing.assert_allclose(node.center, np.array([2.0, 0.0]))
    np.testing.assert_allclose(node.diam, 0.0)


def test_geometry_3d() -> None:
    mesh = _grid_mesh_3d()
    node = make_node(mesh, np.arange(5), level=0)

    expected_bbox = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    np.testing.assert_allclose(node.bounding_box, expected_bbox)
    np.testing.assert_allclose(node.center, np.array([0.5, 1.0, 1.5]))
    np.testing.assert_allclose(node.diam, np.sqrt(1.0 + 4.0 + 9.0))


# ---------------------------------------------------------------------------
# parent/children links
# ---------------------------------------------------------------------------


def test_parent_child_links() -> None:
    mesh = _grid_mesh_2d()
    root = make_node(mesh, np.arange(4), level=0)
    left = make_node(mesh, np.array([0, 2]), level=1, parent=root)
    right = make_node(mesh, np.array([1, 3]), level=1, parent=root)
    root.children = [left, right]

    assert root.parent is None
    assert root.children == [left, right]
    assert left.parent is root
    assert right.parent is root
    assert left.is_leaf
    assert right.is_leaf
    assert not root.is_leaf


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------


def _build_small_tree(mesh: FaultMesh) -> TreeNode:
    """Build a depth-2 binary tree over 4 patches by hand for traversal tests.

    root (level 0): patches [0,1,2,3]
      left  (level 1): patches [0,1]
        ll (level 2): patches [0]
        lr (level 2): patches [1]
      right (level 1): patches [2,3]
        rl (level 2): patches [2]
        rr (level 2): patches [3]
    """
    root = make_node(mesh, np.arange(4), level=0)
    left = make_node(mesh, np.array([0, 1]), level=1, parent=root)
    right = make_node(mesh, np.array([2, 3]), level=1, parent=root)
    root.children = [left, right]

    ll = make_node(mesh, np.array([0]), level=2, parent=left)
    lr = make_node(mesh, np.array([1]), level=2, parent=left)
    left.children = [ll, lr]

    rl = make_node(mesh, np.array([2]), level=2, parent=right)
    rr = make_node(mesh, np.array([3]), level=2, parent=right)
    right.children = [rl, rr]

    return root


def test_leaves_depth_first_order() -> None:
    mesh = _grid_mesh_2d()
    root = _build_small_tree(mesh)

    leaves = root.leaves()
    assert len(leaves) == 4
    assert all(leaf.is_leaf for leaf in leaves)

    leaf_patches = [leaf.patch_indices.tolist() for leaf in leaves]
    assert leaf_patches == [[0], [1], [2], [3]]


def test_leaves_of_single_node_is_itself() -> None:
    mesh = _grid_mesh_2d()
    node = make_node(mesh, np.array([1]), level=2)
    assert node.leaves() == [node]


def test_nodes_at_level() -> None:
    mesh = _grid_mesh_2d()
    root = _build_small_tree(mesh)

    level0 = root.nodes_at_level(0)
    assert len(level0) == 1
    assert level0[0] is root

    level1 = root.nodes_at_level(1)
    assert len(level1) == 2
    assert all(n.level == 1 for n in level1)

    level2 = root.nodes_at_level(2)
    assert len(level2) == 4
    assert all(n.level == 2 for n in level2)
    patch_sets = sorted(n.patch_indices.tolist()[0] for n in level2)
    assert patch_sets == [0, 1, 2, 3]


def test_nodes_at_level_below_root_is_empty() -> None:
    mesh = _grid_mesh_2d()
    root = _build_small_tree(mesh)
    left = root.children[0]

    # A level above the subtree's own level returns nothing.
    assert left.nodes_at_level(0) == []


def test_nodes_at_level_out_of_range() -> None:
    mesh = _grid_mesh_2d()
    root = _build_small_tree(mesh)
    assert root.nodes_at_level(3) == []


def test_iter_levels() -> None:
    mesh = _grid_mesh_2d()
    root = _build_small_tree(mesh)

    levels = list(root.iter_levels())
    assert len(levels) == 3
    assert [n.level for n in levels[0]] == [0]
    assert [n.level for n in levels[1]] == [1, 1]
    assert [n.level for n in levels[2]] == [2, 2, 2, 2]

    # Each level's union of patch_indices reproduces the full patch set.
    for level_nodes in levels:
        all_patches = sorted(int(p) for n in level_nodes for p in n.patch_indices.tolist())
        assert all_patches == [0, 1, 2, 3]


def test_iter_levels_from_subtree() -> None:
    mesh = _grid_mesh_2d()
    root = _build_small_tree(mesh)
    left = root.children[0]

    levels = list(left.iter_levels())
    assert len(levels) == 2
    assert levels[0] == [left]
    assert [n.level for n in levels[1]] == [2, 2]
