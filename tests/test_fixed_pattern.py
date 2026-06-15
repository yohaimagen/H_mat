"""Tests for fixed periodic admissible test matrices (`<= 6^d`, Task 4.2) and
fixed periodic leaf/inadmissible test matrices (`<= 3^d`, Task 4.3).

`build_admissible_test_matrices` tiles the level's boxes into a `6x...x6`
periodic pattern (`pattern_cell`, derived from each box's dyadic grid
coordinates `grid_coordinates`) and emits one `Omega` per non-empty pattern
cell, filling each active box's column rows with an independent Gaussian
block and zeroing everything else.

The coverage test is the PLAN's required output: build a 2D grid deep enough
that some level has `>= 7` boxes along an axis, so pattern cells wrap around
and some cells contain more than one active box -- exercising the period-6
collision-avoidance property (a smaller grid where every box gets a unique
pattern cell would pass trivially even with a wrong period). For every
admissible pair `(alpha, beta)` at that level, the unique `Omega` whose
`active_boxes` contains `beta` must satisfy `(alpha, beta)`'s Eq. 4.4 sampling
constraint: `beta`'s columns are the Gaussian block, and the
`L^nei(alpha) | L^int(alpha) \\ {beta}` columns are all zero.

`build_leaf_test_matrices` (Task 4.3) tiles the level's boxes into a `3x...x3`
periodic pattern (`leaf_pattern_cell`) and emits one `Omega` per non-empty
leaf pattern cell. Each active box `beta` gets its own dedicated column slot
(`col_slices`) holding `I_{len(beta.col_indices)}` on `beta`'s rows and zeros
on every other active box's rows. For every inadmissible pair `(alpha, beta)`
with `beta in L^nei(alpha)`, the unique `Omega` whose `active_boxes` contains
`beta` isolates the dense block `A_{alpha,beta}`:
`(A @ Omega)[alpha.row_indices, col_slice] == A_{alpha,beta}`, because no other
box active in that `Omega` contributes to `beta`'s column slot.
"""

from __future__ import annotations

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.fixed_pattern import (
    LEAF_PERIOD,
    PERIOD,
    build_admissible_test_matrices,
    build_leaf_test_matrices,
    grid_coordinates,
    leaf_pattern_cell,
    pattern_cell,
)
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import interaction_lists
from gfcompress.mockgf import MockGF
from gfcompress.neighbors import neighbor_lists
from gfcompress.sampling import build_sampling_constraint
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


# ---------------------------------------------------------------------------
# grid_coordinates / pattern_cell
# ---------------------------------------------------------------------------


def test_grid_coordinates_distinct_and_in_range() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    n_cells = 2**level
    seen = set()
    for node in level_nodes:
        coords = grid_coordinates(node, root)
        assert len(coords) == 2
        for c in coords:
            assert 0 <= c < n_cells
        assert coords not in seen, f"duplicate grid coordinates {coords}"
        seen.add(coords)


def test_pattern_cell_is_elementwise_mod_period() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    for node in level_nodes:
        coords = grid_coordinates(node, root)
        cell = pattern_cell(node, root)
        assert cell == tuple(c % PERIOD for c in coords)
        assert len(cell) == 2
        for c in cell:
            assert 0 <= c < PERIOD


def test_pattern_cell_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    n_cells = 2**level
    seen = set()
    for node in level_nodes:
        coords = grid_coordinates(node, root)
        assert len(coords) == 3
        for c in coords:
            assert 0 <= c < n_cells
        assert coords not in seen
        seen.add(coords)

        cell = pattern_cell(node, root)
        assert cell == tuple(c % PERIOD for c in coords)


# ---------------------------------------------------------------------------
# Coverage test (2D, period wrap-around)
# ---------------------------------------------------------------------------


def test_admissible_test_matrices_coverage_2d_wraparound() -> None:
    # A grid deep enough that the deepest level has >= 7 boxes along an axis,
    # so pattern cells wrap around (2**level >= 7 -> level >= 3, i.e. >= 8
    # cells per axis at the deepest level).
    nx, ny = 8, 8
    mesh = _grid_mesh(nx, ny, spacing=1.0)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    n_cells = 2**level
    assert n_cells >= 7, f"expected >= 7 cells per axis, got {n_cells}"

    k, p = 3, 2
    seed = 12345
    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=seed)

    d = mesh.d
    assert len(test_matrices) <= PERIOD**d

    # Every emitted Omega has the right shape.
    for tm in test_matrices:
        assert tm.omega.shape == (mesh.n_cols, k + p)

    # Some pattern cells must contain more than one active box (the
    # period-wraparound collision-avoidance property is only exercised if
    # this holds).
    assert any(len(tm.active_boxes) > 1 for tm in test_matrices)

    # Map each box (by id) to the Omega whose active_boxes contains it.
    box_to_tm = {}
    for tm in test_matrices:
        for box in tm.active_boxes:
            assert id(box) not in box_to_tm, "box appears in multiple Omegas"
            box_to_tm[id(box)] = tm

    # Every level node appears in exactly one Omega.
    assert set(box_to_tm.keys()) == {id(node) for node in level_nodes}

    ints = interaction_lists(root)[level]

    n_checked = 0
    for i, alpha in enumerate(level_nodes):
        for beta in ints[i]:
            tm = box_to_tm[id(beta)]
            constraint = build_sampling_constraint(alpha, beta, root)

            # All required-zero columns are zero in this Omega.
            if constraint.zero_cols.size:
                zero_block = tm.omega[constraint.zero_cols, :]
                np.testing.assert_array_equal(zero_block, np.zeros_like(zero_block))

            # The random columns hold beta's Gaussian block (nonzero with
            # overwhelming probability).
            random_block = tm.omega[constraint.random_cols, :]
            assert random_block.shape == (len(beta.col_indices), k + p)
            assert not np.allclose(random_block, 0.0)

            n_checked += 1

    assert n_checked > 0


# ---------------------------------------------------------------------------
# 3D smoke test
# ---------------------------------------------------------------------------


def test_admissible_test_matrices_3d_smoke() -> None:
    nx, ny, nz = 4, 4, 4
    mesh = _grid_mesh(nx, ny, nz, spacing=1.0)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    k, p = 2, 1
    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=7)

    d = mesh.d
    assert len(test_matrices) <= PERIOD**d

    for tm in test_matrices:
        assert tm.omega.shape == (mesh.n_cols, k + p)

    box_to_tm = {}
    for tm in test_matrices:
        for box in tm.active_boxes:
            box_to_tm[id(box)] = tm

    assert set(box_to_tm.keys()) == {id(node) for node in level_nodes}

    ints = interaction_lists(root)[level]

    found_admissible = False
    for i, alpha in enumerate(level_nodes):
        for beta in ints[i]:
            tm = box_to_tm[id(beta)]
            constraint = build_sampling_constraint(alpha, beta, root)

            if constraint.zero_cols.size:
                zero_block = tm.omega[constraint.zero_cols, :]
                np.testing.assert_array_equal(zero_block, np.zeros_like(zero_block))

            random_block = tm.omega[constraint.random_cols, :]
            assert random_block.shape == (len(beta.col_indices), k + p)
            assert not np.allclose(random_block, 0.0)
            found_admissible = True

    assert found_admissible


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_build_admissible_test_matrices_reproducible_with_seed() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)

    tm1 = build_admissible_test_matrices(root, level, mesh, k=2, p=1, seed=42)
    tm2 = build_admissible_test_matrices(root, level, mesh, k=2, p=1, seed=42)

    assert len(tm1) == len(tm2)
    for a, b in zip(tm1, tm2, strict=True):
        np.testing.assert_array_equal(a.omega, b.omega)
        assert a.pattern == b.pattern


def test_build_admissible_test_matrices_no_seed_runs() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)

    test_matrices = build_admissible_test_matrices(root, level, mesh, k=2, p=1)
    assert len(test_matrices) > 0
    for tm in test_matrices:
        assert tm.omega.shape == (mesh.n_cols, 3)


# ---------------------------------------------------------------------------
# leaf_pattern_cell (Task 4.3)
# ---------------------------------------------------------------------------


def test_leaf_pattern_cell_is_elementwise_mod_3() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    for node in level_nodes:
        coords = grid_coordinates(node, root)
        cell = leaf_pattern_cell(node, root)
        assert cell == tuple(c % LEAF_PERIOD for c in coords)
        assert len(cell) == 2
        for c in cell:
            assert 0 <= c < LEAF_PERIOD


def test_leaf_pattern_cell_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    for node in level_nodes:
        coords = grid_coordinates(node, root)
        cell = leaf_pattern_cell(node, root)
        assert cell == tuple(c % LEAF_PERIOD for c in coords)
        assert len(cell) == 3
        for c in cell:
            assert 0 <= c < LEAF_PERIOD


# ---------------------------------------------------------------------------
# build_leaf_test_matrices coverage (2D, period wrap-around)
# ---------------------------------------------------------------------------


def test_leaf_test_matrices_coverage_2d_wraparound() -> None:
    # A grid deep enough that the deepest level has >= 4 boxes along an axis,
    # so leaf pattern cells (period 3) wrap around and some cells contain
    # more than one active box (a smaller grid where every box gets a unique
    # leaf pattern cell would pass trivially even with a wrong period).
    nx, ny = 8, 8
    mesh = _grid_mesh(nx, ny, spacing=1.0)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    n_cells = 2**level
    assert n_cells >= 4, f"expected >= 4 cells per axis, got {n_cells}"

    test_matrices = build_leaf_test_matrices(root, level, mesh)

    d = mesh.d
    assert len(test_matrices) <= LEAF_PERIOD**d

    # Every emitted Omega has n_cols rows.
    for tm in test_matrices:
        assert tm.omega.shape[0] == mesh.n_cols

    # Some leaf pattern cells must contain more than one active box (the
    # period-wraparound collision-avoidance property is only exercised if
    # this holds).
    assert any(len(tm.active_boxes) > 1 for tm in test_matrices)

    # Map each box (by id) to the (Omega, col_slice) whose active_boxes
    # contains it.
    box_to_tm = {}
    for tm in test_matrices:
        for box, sl in zip(tm.active_boxes, tm.col_slices, strict=True):
            assert id(box) not in box_to_tm, "box appears in multiple Omegas"
            box_to_tm[id(box)] = (tm, sl)

    # Every level node appears in exactly one Omega.
    assert set(box_to_tm.keys()) == {id(node) for node in level_nodes}

    nei = neighbor_lists(root)[level]

    n_checked = 0
    for i, _alpha in enumerate(level_nodes):
        for beta in nei[i]:
            tm, sl = box_to_tm[id(beta)]
            w_beta = len(beta.col_indices)
            assert sl.stop - sl.start == w_beta

            # beta's columns hold I_{w_beta} on its own column slot.
            identity_block = tm.omega[np.ix_(beta.col_indices, np.arange(sl.start, sl.stop))]
            np.testing.assert_array_equal(identity_block, np.eye(w_beta))

            # Every other active box (whether or not it neighbors alpha) has
            # zeroed entries in this column slot.
            for gamma in tm.active_boxes:
                if gamma is beta:
                    continue
                gamma_block = tm.omega[gamma.col_indices, sl]
                np.testing.assert_array_equal(gamma_block, np.zeros_like(gamma_block))

            n_checked += 1

    assert n_checked > 0


def test_leaf_test_matrices_isolate_each_inadmissible_block_exactly_once() -> None:
    """Each inadmissible leaf block `(alpha, beta)` (`beta in L^nei(alpha)`)
    is isolated by exactly one `Omega`: the unique one whose `active_boxes`
    contains `beta`."""
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    test_matrices = build_leaf_test_matrices(root, level, mesh)

    box_to_tm = {}
    for tm in test_matrices:
        for box, sl in zip(tm.active_boxes, tm.col_slices, strict=True):
            box_to_tm[id(box)] = (tm, sl)

    nei = neighbor_lists(root)[level]

    for i, _alpha in enumerate(level_nodes):
        for beta in nei[i]:
            isolating = [tm for tm in test_matrices if any(box is beta for box in tm.active_boxes)]
            assert len(isolating) == 1, f"beta isolated by {len(isolating)} matrices, want 1"
            assert box_to_tm[id(beta)][0] is isolating[0]


# ---------------------------------------------------------------------------
# build_leaf_test_matrices exactness against MockGF
# ---------------------------------------------------------------------------


def test_leaf_test_matrices_recover_dense_neighbor_blocks_exactly() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    gf = MockGF(mesh)
    test_matrices = build_leaf_test_matrices(root, level, mesh)

    box_to_tm = {}
    for tm in test_matrices:
        for box, sl in zip(tm.active_boxes, tm.col_slices, strict=True):
            box_to_tm[id(box)] = (tm, sl)

    nei = neighbor_lists(root)[level]

    # Precompute A @ Omega for every emitted test matrix.
    samples = {id(tm): gf.matvec(tm.omega) for tm in test_matrices}

    n_checked = 0
    for i, alpha in enumerate(level_nodes):
        for beta in nei[i]:
            tm, sl = box_to_tm[id(beta)]
            y = samples[id(tm)]

            recovered = y[np.ix_(alpha.row_indices, np.arange(sl.start, sl.stop))]
            expected = gf.block(alpha.patch_indices, beta.patch_indices)

            np.testing.assert_allclose(recovered, expected, atol=1e-10)
            n_checked += 1

    assert n_checked > 0


# ---------------------------------------------------------------------------
# 3D smoke test
# ---------------------------------------------------------------------------


def test_leaf_test_matrices_3d_smoke() -> None:
    nx, ny, nz = 4, 4, 4
    mesh = _grid_mesh(nx, ny, nz, spacing=1.0)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    test_matrices = build_leaf_test_matrices(root, level, mesh)

    d = mesh.d
    assert len(test_matrices) <= LEAF_PERIOD**d

    for tm in test_matrices:
        assert tm.omega.shape[0] == mesh.n_cols

    box_to_tm = {}
    for tm in test_matrices:
        for box, sl in zip(tm.active_boxes, tm.col_slices, strict=True):
            box_to_tm[id(box)] = (tm, sl)

    assert set(box_to_tm.keys()) == {id(node) for node in level_nodes}

    nei = neighbor_lists(root)[level]

    found_neighbor = False
    for i, _alpha in enumerate(level_nodes):
        for beta in nei[i]:
            tm, sl = box_to_tm[id(beta)]
            w_beta = len(beta.col_indices)

            identity_block = tm.omega[np.ix_(beta.col_indices, np.arange(sl.start, sl.stop))]
            np.testing.assert_array_equal(identity_block, np.eye(w_beta))

            for gamma in tm.active_boxes:
                if gamma is beta:
                    continue
                gamma_block = tm.omega[gamma.col_indices, sl]
                np.testing.assert_array_equal(gamma_block, np.zeros_like(gamma_block))

            found_neighbor = True

    assert found_neighbor
