"""Tests for the Eq. 4.4 sampling-constraint descriptor (Task 4.1).

For an admissible pair `(alpha, beta)`, `build_sampling_constraint` marks
`beta` (the col-box) as *random* and every col-box `gamma in L^nei(alpha) |
L^int(alpha) \\ {beta}` as *zero*, with column-index sets expressed via
`TreeNode.col_indices` (`dof_col`-based patch-major flattening, from
`gfcompress.geometry.FaultMesh.patch_to_cols`).

Two small grids (2D and 3D) are used. For sample admissible pairs we check:

- `random_box is beta` and `random_cols == beta.col_indices`.
- `zero_boxes` is exactly `(L^nei(alpha) | L^int(alpha)) \\ {beta}` (by
  identity), with no duplicates and `beta` excluded even if it would
  otherwise appear.
- `zero_cols` is the concatenation of the zero boxes' `col_indices`, disjoint
  from `random_cols`.
- the underlying `L^nei`/`L^int` lists used satisfy the disjoint, complete
  cover and admissibility-consistency invariants from CLAUDE.md (a
  sanity-check, not a re-test of Stage 1's full coverage).
"""

from __future__ import annotations

import numpy as np
import pytest

from gfcompress.build_tree import build_tree
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import DEFAULT_ETA, interaction_lists, is_admissible
from gfcompress.neighbors import neighbor_lists
from gfcompress.sampling import SamplingConstraint, build_sampling_constraint
from gfcompress.tree import TreeNode


def _grid_mesh(*shape: int, spacing: float = 1.0) -> FaultMesh:
    """Build a `FaultMesh` whose centroids form a regular grid of the given
    `shape` (length `d`, `d in (2, 3)`), with unit spacing along each axis."""
    axes = [np.arange(n, dtype=float) * spacing for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    L = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=L)


def _deepest_level(root: TreeNode) -> int:
    deepest = 0
    for level_nodes in root.iter_levels():
        deepest = level_nodes[0].level
    return deepest


def _find_admissible_pair(root: TreeNode, level: int) -> tuple[TreeNode, TreeNode]:
    """Return the first admissible same-level pair `(alpha, beta)` at
    `level`, via `L^int`."""
    level_nodes = root.nodes_at_level(level)
    ints = interaction_lists(root)[level]
    for i, alpha in enumerate(level_nodes):
        if ints[i]:
            return alpha, ints[i][0]
    raise AssertionError(f"no admissible pair found at level {level}")


# ---------------------------------------------------------------------------
# build_sampling_constraint: random box is beta, zero set is L^nei | L^int \ {beta}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_constraint_random_box_is_beta(shape: tuple[int, ...], m: int) -> None:
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    level = _deepest_level(root)
    alpha, beta = _find_admissible_pair(root, level)

    constraint = build_sampling_constraint(alpha, beta, root)

    assert constraint.alpha is alpha
    assert constraint.beta is beta
    assert constraint.random_box is beta
    np.testing.assert_array_equal(constraint.random_cols, beta.col_indices)


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_constraint_zero_set_is_nei_union_int_minus_beta(shape: tuple[int, ...], m: int) -> None:
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    level = _deepest_level(root)
    alpha, beta = _find_admissible_pair(root, level)

    constraint = build_sampling_constraint(alpha, beta, root)

    level_nodes = root.nodes_at_level(level)
    alpha_idx = next(idx for idx, node in enumerate(level_nodes) if node is alpha)
    nei = neighbor_lists(root)[level][alpha_idx]
    ints = interaction_lists(root)[level][alpha_idx]

    expected_ids = {id(g) for g in (*nei, *ints)} - {id(beta)}
    actual_ids = {id(g) for g in constraint.zero_boxes}

    assert actual_ids == expected_ids
    # beta itself is never in the zero set.
    assert id(beta) not in actual_ids
    # No duplicates.
    assert len(constraint.zero_boxes) == len(actual_ids)


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_constraint_zero_cols_match_col_indices_concatenation(
    shape: tuple[int, ...], m: int
) -> None:
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    level = _deepest_level(root)
    alpha, beta = _find_admissible_pair(root, level)

    constraint = build_sampling_constraint(alpha, beta, root)

    expected = (
        np.concatenate([gamma.col_indices for gamma in constraint.zero_boxes])
        if constraint.zero_boxes
        else np.array([], dtype=np.intp)
    )
    np.testing.assert_array_equal(constraint.zero_cols, expected)

    # dof_col-based patch-major flattening: zero_cols are a subset of
    # {0, ..., n_cols - 1}.
    assert np.all(constraint.zero_cols >= 0)
    assert np.all(constraint.zero_cols < mesh.n_cols)


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_constraint_random_and_zero_cols_disjoint(shape: tuple[int, ...], m: int) -> None:
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    level = _deepest_level(root)
    alpha, beta = _find_admissible_pair(root, level)

    constraint = build_sampling_constraint(alpha, beta, root)

    random_set = set(constraint.random_cols.tolist())
    zero_set = set(constraint.zero_cols.tolist())
    assert random_set.isdisjoint(zero_set)


# ---------------------------------------------------------------------------
# beta also present in L^nei|L^int (e.g. beta == alpha): excluded from zeros
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_constraint_beta_excluded_even_if_neighbor(shape: tuple[int, ...], m: int) -> None:
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    alpha = level_nodes[0]

    # beta = alpha is always in L^nei(alpha) (a node is its own neighbor).
    constraint = build_sampling_constraint(alpha, alpha, root)

    assert constraint.random_box is alpha
    assert all(gamma is not alpha for gamma in constraint.zero_boxes)


# ---------------------------------------------------------------------------
# Underlying L^nei/L^int invariants (sanity check, per CLAUDE.md) for the
# pairs used above.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_underlying_lists_disjoint_complete_cover(shape: tuple[int, ...], m: int) -> None:
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    nei = neighbor_lists(root)[level]
    ints = interaction_lists(root)[level]

    for i, alpha in enumerate(level_nodes):
        nei_ids = {id(b) for b in nei[i]}
        int_ids = {id(b) for b in ints[i]}

        # Disjoint.
        assert nei_ids.isdisjoint(int_ids)

        for beta in nei[i]:
            assert not is_admissible(alpha, beta, DEFAULT_ETA)
        for beta in ints[i]:
            assert is_admissible(alpha, beta, DEFAULT_ETA)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_constraint_different_levels_raises() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    alpha = root.nodes_at_level(_deepest_level(root))[0]
    beta = root.nodes_at_level(_deepest_level(root) - 1)[0]

    with pytest.raises(ValueError):
        build_sampling_constraint(alpha, beta, root)


# ---------------------------------------------------------------------------
# SamplingConstraint defaults
# ---------------------------------------------------------------------------


def test_sampling_constraint_default_zero_boxes_empty() -> None:
    mesh = _grid_mesh(4, 4)
    root = build_tree(mesh, m=1)
    alpha = root.nodes_at_level(_deepest_level(root))[0]
    constraint = SamplingConstraint(alpha=alpha, beta=alpha)
    assert constraint.zero_boxes == []
    np.testing.assert_array_equal(constraint.zero_cols, np.array([], dtype=np.intp))
