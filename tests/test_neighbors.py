"""Tests for box-adjacency and neighbor lists `L^nei` (Task 1.4).

`FaultMesh` requires `d in (2, 3)` (see `geometry.py`), so a 1D mesh cannot be
constructed and a true 1D grid test is not representable here. The "3 in 1D"
expectation from the task description is instead exercised indirectly: a 2D
grid that is uniform along x but has a single row along y reduces, along the
x-axis, to the 1D adjacency pattern (each interior box has neighbors at
`{-1, 0, +1}` along x, all in the single row along y), giving the expected
`3` neighbors for interior boxes. Full 2D and 3D grids cover the `9` and `27`
cases directly.
"""

from __future__ import annotations

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.geometry import FaultMesh
from gfcompress.neighbors import boxes_adjacent, neighbor_lists
from gfcompress.tree import TreeNode


def _grid_mesh(*shape: int, spacing: float = 1.0) -> FaultMesh:
    """Build a `FaultMesh` whose centroids form a regular grid of the given
    `shape` (length `d`, `d in (2, 3)`), with unit spacing along each axis."""
    axes = [np.arange(n, dtype=float) * spacing for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    L = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=L)


def _deepest_full_level(root: TreeNode, expected_shape: tuple[int, ...]) -> int:
    """Return the deepest level whose node count equals
    `prod(expected_shape)`, i.e. the finest level at which the dyadic grid
    has exactly one box per grid cell of `expected_shape`."""
    target = 1
    for n in expected_shape:
        target *= n
    best = 0
    for level_nodes in root.iter_levels():
        if len(level_nodes) == target:
            best = level_nodes[0].level
    return best


# ---------------------------------------------------------------------------
# boxes_adjacent
# ---------------------------------------------------------------------------


def test_boxes_adjacent_overlapping() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.5, 1.5], [0.5, 1.5]])
    assert boxes_adjacent(a, b)
    assert boxes_adjacent(b, a)


def test_boxes_adjacent_touching_face() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[1.0, 2.0], [0.0, 1.0]])
    assert boxes_adjacent(a, b)


def test_boxes_adjacent_touching_corner() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert boxes_adjacent(a, b)


def test_boxes_adjacent_self() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert boxes_adjacent(a, a)


def test_boxes_not_adjacent_disjoint() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[1.5, 2.5], [0.0, 1.0]])
    assert not boxes_adjacent(a, b)


def test_boxes_adjacent_floating_point_tolerance() -> None:
    a = np.array([[0.0, 1.0]])
    # b's lower edge is a hair below a's upper edge (round-off from a dyadic
    # split): without a tolerance this would register a tiny gap of 0.
    b = np.array([[1.0 - 1e-13, 2.0]])
    assert boxes_adjacent(a, b)


# ---------------------------------------------------------------------------
# neighbor_lists: 2D grid -> 9 for interior boxes
# ---------------------------------------------------------------------------


def test_neighbor_lists_2d_interior_has_9() -> None:
    nx, ny = 8, 8
    mesh = _grid_mesh(nx, ny)
    root = build_tree(mesh, m=2)

    level = _deepest_full_level(root, (nx, ny))
    level_nodes = root.nodes_at_level(level)
    assert len(level_nodes) == nx * ny

    nei = neighbor_lists(root)[level]

    # Identify an interior box: its dyadic cell does not touch the root
    # domain boundary.
    root_box = root.bounding_box
    found_interior = False
    for i, node in enumerate(level_nodes):
        box = node.bounding_box
        is_boundary = np.any(
            np.isclose(box[:, 0], root_box[:, 0]) | np.isclose(box[:, 1], root_box[:, 1])
        )
        if not is_boundary:
            found_interior = True
            assert len(nei[i]) == 9, f"interior box {i} has {len(nei[i])} neighbors"
    assert found_interior, "expected at least one interior box on an 8x8 grid"


def test_neighbor_lists_2d_corner_has_4() -> None:
    nx, ny = 8, 8
    mesh = _grid_mesh(nx, ny)
    root = build_tree(mesh, m=2)
    level = _deepest_full_level(root, (nx, ny))
    level_nodes = root.nodes_at_level(level)
    nei = neighbor_lists(root)[level]

    root_box = root.bounding_box
    for i, node in enumerate(level_nodes):
        box = node.bounding_box
        touches_lo = np.isclose(box[:, 0], root_box[:, 0])
        touches_hi = np.isclose(box[:, 1], root_box[:, 1])
        n_boundary_axes = int(np.sum(touches_lo | touches_hi))
        if n_boundary_axes == 2:
            # A corner box: only 4 neighbors (2x2 patch of the 3x3 stencil).
            assert len(nei[i]) == 4, f"corner box {i} has {len(nei[i])} neighbors"


# ---------------------------------------------------------------------------
# neighbor_lists: 3D grid -> 27 for interior boxes
# ---------------------------------------------------------------------------


def test_neighbor_lists_3d_interior_has_27() -> None:
    nx, ny, nz = 4, 4, 4
    mesh = _grid_mesh(nx, ny, nz)
    root = build_tree(mesh, m=2)

    level = _deepest_full_level(root, (nx, ny, nz))
    level_nodes = root.nodes_at_level(level)
    assert len(level_nodes) == nx * ny * nz

    nei = neighbor_lists(root)[level]

    root_box = root.bounding_box
    found_interior = False
    for i, node in enumerate(level_nodes):
        box = node.bounding_box
        is_boundary = np.any(
            np.isclose(box[:, 0], root_box[:, 0]) | np.isclose(box[:, 1], root_box[:, 1])
        )
        if not is_boundary:
            found_interior = True
            assert len(nei[i]) == 27, f"interior box {i} has {len(nei[i])} neighbors"
    assert found_interior, "expected at least one interior box on a 4x4x4 grid"


# ---------------------------------------------------------------------------
# "1D-equivalent": 2D grid that is a single row in y -> 3 for interior boxes
# along x, since neighbors along the trivial y-axis are always the (single)
# row itself.
# ---------------------------------------------------------------------------


def test_neighbor_lists_1d_equivalent_interior_has_3() -> None:
    nx = 8
    mesh = _grid_mesh(nx, 1)
    root = build_tree(mesh, m=2)

    level = _deepest_full_level(root, (nx, 1))
    level_nodes = root.nodes_at_level(level)
    assert len(level_nodes) == nx

    nei = neighbor_lists(root)[level]

    root_box = root.bounding_box
    found_interior = False
    for i, node in enumerate(level_nodes):
        box = node.bounding_box
        # Only the x-axis (axis 0) has nontrivial extent / boundary to check.
        is_boundary = np.isclose(box[0, 0], root_box[0, 0]) or np.isclose(box[0, 1], root_box[0, 1])
        if not is_boundary:
            found_interior = True
            assert len(nei[i]) == 3, f"interior box {i} has {len(nei[i])} neighbors"
    assert found_interior, "expected at least one interior box on an 8x1 grid"


# ---------------------------------------------------------------------------
# General properties: self-inclusion, symmetry, complete cover by levels.
# ---------------------------------------------------------------------------


def test_box_includes_itself() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    nei = neighbor_lists(root)
    for level, level_map in nei.items():
        level_nodes = root.nodes_at_level(level)
        for i, node in enumerate(level_nodes):
            assert any(
                node is b for b in level_map[i]
            ), f"node {i} at level {level} does not include itself"


def test_neighbor_relation_is_symmetric_2d() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    nei = neighbor_lists(root)
    for level, level_map in nei.items():
        level_nodes = root.nodes_at_level(level)
        for i, alpha in enumerate(level_nodes):
            for beta in level_map[i]:
                j = next(k for k, n in enumerate(level_nodes) if n is beta)
                assert any(n is alpha for n in level_map[j]), (
                    f"neighbor relation not symmetric at level {level} "
                    f"between boxes {i} and {j}"
                )


def test_neighbor_relation_is_symmetric_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    nei = neighbor_lists(root)
    for level, level_map in nei.items():
        level_nodes = root.nodes_at_level(level)
        for i, alpha in enumerate(level_nodes):
            for beta in level_map[i]:
                j = next(k for k, n in enumerate(level_nodes) if n is beta)
                assert any(n is alpha for n in level_map[j]), (
                    f"neighbor relation not symmetric at level {level} "
                    f"between boxes {i} and {j}"
                )


def test_neighbor_count_at_most_3_pow_d() -> None:
    for shape, m in [((8, 8), 2), ((4, 4, 4), 2)]:
        mesh = _grid_mesh(*shape)
        root = build_tree(mesh, m=m)
        d = len(shape)
        nei = neighbor_lists(root)
        for level_map in nei.values():
            for neighbors in level_map.values():
                assert len(neighbors) <= 3**d
                assert len(neighbors) >= 1


def test_boundary_box_has_fewer_neighbors_than_interior_3d() -> None:
    nx, ny, nz = 4, 4, 4
    mesh = _grid_mesh(nx, ny, nz)
    root = build_tree(mesh, m=2)
    level = _deepest_full_level(root, (nx, ny, nz))
    level_nodes = root.nodes_at_level(level)
    nei = neighbor_lists(root)[level]

    root_box = root.bounding_box
    counts_interior = []
    counts_boundary = []
    for i, node in enumerate(level_nodes):
        box = node.bounding_box
        is_boundary = np.any(
            np.isclose(box[:, 0], root_box[:, 0]) | np.isclose(box[:, 1], root_box[:, 1])
        )
        if is_boundary:
            counts_boundary.append(len(nei[i]))
        else:
            counts_interior.append(len(nei[i]))

    assert counts_interior, "no interior boxes found"
    assert counts_boundary, "no boundary boxes found"
    assert max(counts_boundary) < max(counts_interior)
