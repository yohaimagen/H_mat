"""Tests for fixed periodic admissible test matrices (`<= 6^d`, Task 4.2) and
fixed periodic leaf/inadmissible test matrices (`<= 3^d`, Task 4.3).

`build_admissible_test_matrices` tiles the level's boxes into a `6x...x6`
periodic pattern (`pattern_cell`, derived from each box's dyadic grid
coordinates `grid_coordinates`) and emits one `Omega` per non-empty pattern
cell, filling each active box's column rows with an independent Gaussian
block and zeroing everything else.

The coverage test is the PLAN's required output: build a 2D grid deep enough
that some level has `>= 12` boxes along an axis, so pattern cells wrap around
and some cells contain more than one active box -- exercising the period-6
collision-avoidance property (a smaller grid where every box gets a unique
pattern cell would pass trivially even with a wrong period). For every
admissible pair `(alpha, beta)` at that level, the unique `Omega` whose
`active_boxes` contains `beta` must satisfy `(alpha, beta)`'s Eq. 4.4 sampling
constraint: `beta`'s columns are the Gaussian block, and the
`L^nei(alpha) | L^int(alpha) \\ {beta}` columns are all zero.

`side="row"` builds the transposed variant `Psi` (shape `(n_rows, k + p)`,
filled on `row_indices`) from the same combinatorics and an independent random
stream.

`build_leaf_test_matrices` (Task 4.3) tiles the level's boxes into a `3x...x3`
periodic pattern (`leaf_pattern_cell`) and emits one `Omega` per non-empty
leaf pattern cell. All active boxes **share one column slot** of width
`w_max`: `Omega[beta.col_indices, :w_beta] = I_{w_beta}`. Recovery of the
dense neighbor block is therefore exact only through the *residual* operator
`A - A^{(L)}`, in which the non-neighbor active boxes of the same `Omega`
contribute nothing: `((A - A^{(L)}) @ Omega)[alpha.row_indices, :w_beta] ==
A_{alpha,beta}`.
"""

from __future__ import annotations

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.fixed_pattern import (
    LEAF_PERIOD,
    PERIOD,
    PeriodicLeafTestMatrix,
    build_admissible_test_matrices,
    build_leaf_test_matrices,
    grid_coordinates,
    leaf_pattern_cell,
    pattern_cell,
)
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.mockgf import MockGF
from gfcompress.peeling import BlockFactor, Factors, peeled_matvec
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
    # A grid deep enough that the tested level has >= 12 boxes along an axis:
    # the L^nei | L^int window is 6 wide, so >= 12 cells per axis means every
    # pattern cell wraps at least twice and the 6-wide window is genuinely
    # exercised (a smaller grid would pass even with a wrong period).
    nx, ny = 16, 16
    mesh = _grid_mesh(nx, ny, spacing=1.0)
    root = build_tree(mesh, m=1)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    n_cells = 2**level
    assert n_cells >= 12, f"expected >= 12 cells per axis, got {n_cells}"

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

    lists = build_lists(root)
    ints = lists.interaction

    n_checked = 0
    for alpha in level_nodes:
        for beta in ints[alpha]:
            tm = box_to_tm[id(beta)]
            constraint = build_sampling_constraint(alpha, beta, lists)

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
# Row side (Psi) and the retained Gaussian blocks
# ---------------------------------------------------------------------------


def test_admissible_test_matrices_row_side_coverage_2d() -> None:
    """For every admissible `(alpha, beta)`, the unique `Psi` whose
    `active_boxes` contains `alpha` is Gaussian on `alpha.row_indices` and
    zero on `gamma.row_indices` for every `gamma in L^nei(beta) | L^int(beta)
    \\ {alpha}`."""
    mesh = _grid_mesh(16, 16, spacing=1.0)
    root = build_tree(mesh, m=1)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    assert 2**level >= 12

    k, p = 3, 2
    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=5, side="row")

    assert len(test_matrices) <= PERIOD**mesh.d
    for tm in test_matrices:
        assert tm.omega.shape == (mesh.n_rows, k + p)

    box_to_tm = {}
    for tm in test_matrices:
        for box in tm.active_boxes:
            assert id(box) not in box_to_tm
            box_to_tm[id(box)] = tm
    assert set(box_to_tm.keys()) == {id(node) for node in level_nodes}

    lists = build_lists(root)

    n_checked = 0
    for alpha in level_nodes:
        for beta in lists.interaction[alpha]:
            tm = box_to_tm[id(alpha)]

            random_block = tm.omega[alpha.row_indices, :]
            assert random_block.shape == (len(alpha.row_indices), k + p)
            assert not np.allclose(random_block, 0.0)

            for gamma in lists.nei[beta] + lists.interaction[beta]:
                if gamma is alpha:
                    continue
                zero_block = tm.omega[gamma.row_indices, :]
                np.testing.assert_array_equal(zero_block, np.zeros_like(zero_block))

            n_checked += 1

    assert n_checked > 0


def test_blocks_match_written_rows_both_sides() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)

    col_matrices = build_admissible_test_matrices(root, level, mesh, k=3, p=2, seed=11)
    for tm in col_matrices:
        assert set(map(id, tm.blocks)) == {id(b) for b in tm.active_boxes}
        for box in tm.active_boxes:
            np.testing.assert_array_equal(tm.blocks[box], tm.omega[box.col_indices, :])

    row_matrices = build_admissible_test_matrices(root, level, mesh, k=3, p=2, seed=11, side="row")
    for tm in row_matrices:
        assert set(map(id, tm.blocks)) == {id(b) for b in tm.active_boxes}
        for box in tm.active_boxes:
            np.testing.assert_array_equal(tm.blocks[box], tm.omega[box.row_indices, :])


def test_row_and_col_sides_use_independent_random_streams() -> None:
    """Same base seed, different sides: no Gaussian block may be shared."""
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)

    cols = build_admissible_test_matrices(root, level, mesh, k=3, p=2, seed=11)
    rows = build_admissible_test_matrices(root, level, mesh, k=3, p=2, seed=11, side="row")

    for tm_c, tm_r in zip(cols, rows, strict=True):
        assert tm_c.pattern == tm_r.pattern
        for box in tm_c.active_boxes:
            block_c = tm_c.blocks[box]
            # Row blocks are dof_row-tall vs dof_col-tall; compare the
            # overlapping leading rows.
            block_r = tm_r.blocks[box][: block_c.shape[0], :]
            assert not np.allclose(block_c, block_r)

    # side="row" is still reproducible on its own.
    rows_again = build_admissible_test_matrices(root, level, mesh, k=3, p=2, seed=11, side="row")
    for a, b in zip(rows, rows_again, strict=True):
        np.testing.assert_array_equal(a.omega, b.omega)


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

    lists = build_lists(root)
    ints = lists.interaction

    found_admissible = False
    for alpha in level_nodes:
        for beta in ints[alpha]:
            tm = box_to_tm[id(beta)]
            constraint = build_sampling_constraint(alpha, beta, lists)

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


def _leaf_box_to_tm(
    test_matrices: list[PeriodicLeafTestMatrix],
) -> dict[int, PeriodicLeafTestMatrix]:
    """Map `id(box) -> the unique test matrix in which `box` is active."""
    box_to_tm: dict[int, PeriodicLeafTestMatrix] = {}
    for tm in test_matrices:
        for box in tm.active_boxes:
            assert id(box) not in box_to_tm, "box appears in multiple Omegas"
            box_to_tm[id(box)] = tm
    return box_to_tm


def test_leaf_test_matrices_shared_slots_and_isolation_2d() -> None:
    # 16x16 patches, m=4 -> leaf level has 8x8 boxes of 4 patches, so leaf
    # pattern cells (period 3) wrap around and some cells hold several boxes.
    mesh = _grid_mesh(16, 16, spacing=1.0)
    root = build_tree(mesh, m=4)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    assert 2**level >= 4

    test_matrices = build_leaf_test_matrices(root, level, mesh)
    d = mesh.d
    w_max = max(len(box.col_indices) for box in level_nodes)

    assert len(test_matrices) <= LEAF_PERIOD**d

    # Shared slots: every Omega is exactly w_max wide, so the total number of
    # matvec columns is at most 3^d * w_max -- far below a full identity.
    total_width = 0
    for tm in test_matrices:
        assert tm.omega.shape == (mesh.n_cols, w_max)
        total_width += tm.omega.shape[1]
    assert total_width <= LEAF_PERIOD**d * w_max
    assert total_width < mesh.n_cols // 4, f"{total_width} not << {mesh.n_cols}"

    # The period-wraparound property is only exercised if some cell holds
    # more than one active box.
    assert any(len(tm.active_boxes) > 1 for tm in test_matrices)

    box_to_tm = _leaf_box_to_tm(test_matrices)
    assert set(box_to_tm.keys()) == {id(node) for node in level_nodes}

    nei = build_lists(root).nei

    n_checked = 0
    for alpha in level_nodes:
        for beta in nei[alpha]:
            isolating = [tm for tm in test_matrices if any(b is beta for b in tm.active_boxes)]
            assert len(isolating) == 1, f"beta active in {len(isolating)} matrices, want 1"
            tm = isolating[0]
            assert tm is box_to_tm[id(beta)]

            w_beta = len(beta.col_indices)
            identity_block = tm.omega[np.ix_(beta.col_indices, np.arange(w_beta))]
            np.testing.assert_array_equal(identity_block, np.eye(w_beta))

            # No other member of L^nei(alpha) (alpha included) is active here.
            collisions = [
                gamma
                for gamma in [alpha, *nei[alpha]]
                if gamma is not beta and any(b is gamma for b in tm.active_boxes)
            ]
            assert not collisions, f"{len(collisions)} L^nei(alpha) collisions in one Omega"

            n_checked += 1

    assert n_checked > 0


# ---------------------------------------------------------------------------
# Leaf recovery through the residual operator A - A^(L)
# ---------------------------------------------------------------------------


def _exact_factors(gf: MockGF, root: TreeNode, lists: TreeLists) -> Factors:
    """Ground-truth (test-only) factors: an exact full-rank SVD of every
    admissible block at every level, so `A - A^(L)` peels the whole far field
    to round-off and only the leaf-level neighbor blocks survive."""
    factors: Factors = []
    for level_nodes in root.iter_levels():
        for alpha in level_nodes:
            for beta in lists.interaction[alpha]:
                block = gf.block(alpha.patch_indices, beta.patch_indices)
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                factors.append(
                    BlockFactor(alpha=alpha, beta=beta, u=u, b=np.diag(s), v=vt.conj().T)
                )
    return factors


def test_leaf_test_matrices_recover_neighbor_blocks_through_peeling() -> None:
    mesh = _grid_mesh(16, 16, spacing=1.0)
    root = build_tree(mesh, m=4)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)
    lists = build_lists(root)

    gf = MockGF(mesh)
    factors = _exact_factors(gf, root, lists)
    assert factors, "expected admissible blocks to peel"

    test_matrices = build_leaf_test_matrices(root, level, mesh)
    box_to_tm = _leaf_box_to_tm(test_matrices)
    samples = {id(tm): peeled_matvec(gf, tm.omega, factors) for tm in test_matrices}

    n_checked = 0
    for alpha in level_nodes:
        for beta in lists.nei[alpha]:
            tm = box_to_tm[id(beta)]
            w_beta = len(beta.col_indices)

            recovered = samples[id(tm)][np.ix_(alpha.row_indices, np.arange(w_beta))]
            expected = gf.block(alpha.patch_indices, beta.patch_indices)

            np.testing.assert_allclose(recovered, expected, rtol=1e-8, atol=1e-8)
            n_checked += 1

    assert n_checked > 0


# ---------------------------------------------------------------------------
# 3D smoke test
# ---------------------------------------------------------------------------


def test_leaf_test_matrices_3d_smoke() -> None:
    mesh = _grid_mesh(4, 4, 4, spacing=1.0)
    root = build_tree(mesh, m=2)
    level = _deepest_level(root)
    level_nodes = root.nodes_at_level(level)

    test_matrices = build_leaf_test_matrices(root, level, mesh)
    w_max = max(len(box.col_indices) for box in level_nodes)

    assert len(test_matrices) <= LEAF_PERIOD**mesh.d
    for tm in test_matrices:
        assert tm.omega.shape == (mesh.n_cols, w_max)

    box_to_tm = _leaf_box_to_tm(test_matrices)
    assert set(box_to_tm.keys()) == {id(node) for node in level_nodes}

    nei = build_lists(root).nei

    found_neighbor = False
    for alpha in level_nodes:
        for beta in nei[alpha]:
            tm = box_to_tm[id(beta)]
            w_beta = len(beta.col_indices)

            identity_block = tm.omega[np.ix_(beta.col_indices, np.arange(w_beta))]
            np.testing.assert_array_equal(identity_block, np.eye(w_beta))

            for gamma in [alpha, *nei[alpha]]:
                if gamma is beta:
                    continue
                assert not any(b is gamma for b in tm.active_boxes)

            found_neighbor = True

    assert found_neighbor
