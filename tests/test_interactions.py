"""Tests for interaction lists `L^int` and the admissibility predicate
(Task 1.5).

Two structural invariants are checked at every level of the tree:

1. **Disjointness**: for each box `alpha`, `L^nei(alpha)` and `L^int(alpha)`
   share no boxes.
2. **Completeness**: `L^nei(alpha) | L^int(alpha)` equals exactly the set of
   children of `alpha.parent`'s neighbors (the only candidates `L^int` is
   drawn from), union `alpha`'s own neighbors (which may include boxes that
   are not children of a parent-neighbor, e.g. across a leaf/internal-level
   mismatch -- on the regular grids built by `build_tree` this does not
   happen, and the two sets coincide).

On top of the combinatorial cover, `is_admissible` is checked for *geometric*
consistency with the combinatorial split: every box in `L^int(alpha)` must
test admissible against `alpha`, and every box in `L^nei(alpha)` (including
`alpha` itself) must test inadmissible.

The "Fig. 3 tessellation" (paper Fig. 3, p.6: an H^1 matrix of depth 3 over a
grid on `[0,1]`) is reproduced structurally: a small uniform grid is built to
depth 3, and at the deepest level every same-level pair `(alpha, beta)` is
classified as exactly one of {inadmissible neighbor, admissible interaction,
or "not present" (neither a neighbor nor a child of a parent-neighbor)} -- the
disjoint, complete tessellation that Fig. 3 depicts in red/blue.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from gfcompress.build_tree import build_tree
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import (
    DEFAULT_ETA,
    box_dist,
    interaction_list,
    interaction_lists,
    is_admissible,
    suggest_eta,
)
from gfcompress.neighbors import neighbor_lists
from gfcompress.tree import TreeNode


def _grid_mesh(*shape: int, spacing: float = 1.0) -> FaultMesh:
    """Build a `FaultMesh` whose centroids form a regular grid of the given
    `shape` (length `d`, `d in (2, 3)`), with unit spacing along each axis."""
    axes = [np.arange(n, dtype=float) * spacing for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    L = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=L)


# ---------------------------------------------------------------------------
# box_dist
# ---------------------------------------------------------------------------


def test_box_dist_overlapping_is_zero() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.5, 1.5], [0.5, 1.5]])
    assert box_dist(a, b) == 0.0


def test_box_dist_touching_face_is_zero() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[1.0, 2.0], [0.0, 1.0]])
    assert box_dist(a, b) == 0.0


def test_box_dist_touching_corner_is_zero() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert box_dist(a, b) == 0.0


def test_box_dist_separated_along_one_axis() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[2.0, 3.0], [0.0, 1.0]])
    assert box_dist(a, b) == pytest.approx(1.0)


def test_box_dist_separated_diagonally() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[2.0, 3.0], [2.0, 3.0]])
    assert box_dist(a, b) == pytest.approx(np.sqrt(2.0))


def test_box_dist_symmetric() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[2.0, 3.0], [3.0, 5.0]])
    assert box_dist(a, b) == pytest.approx(box_dist(b, a))


# ---------------------------------------------------------------------------
# interaction_lists / interaction_list: complete, disjoint cover with L^nei
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_interaction_and_neighbor_lists_disjoint(shape: tuple[int, ...], m: int) -> None:
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    nei = neighbor_lists(root)
    ints = interaction_lists(root)

    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        for i in range(len(level_nodes)):
            nei_ids = {id(b) for b in nei[level][i]}
            int_ids = {id(b) for b in ints[level][i]}
            assert nei_ids.isdisjoint(int_ids), f"level {level} box {i}: L^nei and L^int overlap"


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_interaction_list_is_complement_of_neighbors_within_candidates(
    shape: tuple[int, ...], m: int
) -> None:
    """For every non-root box `alpha`, `L^int(alpha)` together with
    `L^nei(alpha)` exactly covers the set of children of `alpha.parent`'s
    neighbors -- the candidate set `L^int` is drawn from. I.e. `L^int(alpha)
    = candidates \\ L^nei(alpha)` and every neighbor of `alpha` that is itself
    a child of a parent-neighbor is excluded from `L^int`.
    """
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    nei = neighbor_lists(root)
    ints = interaction_lists(root)

    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        if level == 0:
            assert ints[level][0] == []
            continue

        parent_level_nodes = root.nodes_at_level(level - 1)
        parent_nei = nei[level - 1]

        for i, alpha in enumerate(level_nodes):
            assert alpha.parent is not None
            parent_idx = next(
                idx for idx, node in enumerate(parent_level_nodes) if node is alpha.parent
            )
            candidates: set[int] = set()
            for parent_neighbor in parent_nei[parent_idx]:
                for child in parent_neighbor.children:
                    candidates.add(id(child))

            nei_ids = {id(b) for b in nei[level][i]}
            int_ids = {id(b) for b in ints[level][i]}

            # Every interaction-list box is a candidate not in L^nei(alpha).
            assert int_ids <= candidates
            assert int_ids == candidates - nei_ids

            # alpha itself is always a candidate (it's a child of its own
            # parent, which is its own neighbor) and always in L^nei.
            assert id(alpha) in candidates
            assert id(alpha) in nei_ids
            assert id(alpha) not in int_ids


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_interaction_list_size_bound(shape: tuple[int, ...], m: int) -> None:
    """On a regular grid, `|L^int(alpha)| <= 6^d - 3^d`."""
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    ints = interaction_lists(root)
    d = len(shape)
    bound = 6**d - 3**d

    for level_map in ints.values():
        for entries in level_map.values():
            assert len(entries) <= bound


def test_interaction_list_single_node_matches_full_map() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    ints = interaction_lists(root)

    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        for i, alpha in enumerate(level_nodes):
            single = interaction_list(alpha, root)
            full = ints[level][i]
            assert [id(b) for b in single] == [id(b) for b in full]


def test_root_has_empty_interaction_list() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    assert interaction_list(root, root) == []
    assert interaction_lists(root)[0] == {0: []}


# ---------------------------------------------------------------------------
# is_admissible: geometric predicate consistent with the combinatorial split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,m", [((8, 8), 2), ((4, 4, 4), 2)])
def test_admissibility_consistent_with_interaction_and_neighbor_lists(
    shape: tuple[int, ...], m: int
) -> None:
    """At `DEFAULT_ETA`, every box in `L^int(alpha)` is admissible against
    `alpha`, and every box in `L^nei(alpha)` (including `alpha` itself) is
    inadmissible. This is the geometric predicate agreeing with the
    combinatorial interaction-list/neighbor-list split."""
    mesh = _grid_mesh(*shape)
    root = build_tree(mesh, m=m)
    nei = neighbor_lists(root)
    ints = interaction_lists(root)

    checked_int = 0
    checked_nei = 0
    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        for i, alpha in enumerate(level_nodes):
            for beta in nei[level][i]:
                assert not is_admissible(
                    alpha, beta, DEFAULT_ETA
                ), f"level {level} box {i}: neighbor tested admissible"
                checked_nei += 1
            for beta in ints[level][i]:
                assert is_admissible(
                    alpha, beta, DEFAULT_ETA
                ), f"level {level} box {i}: interaction-list box tested inadmissible"
                checked_int += 1

    assert checked_nei > 0
    assert checked_int > 0


def test_admissibility_self_is_inadmissible() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    for level_nodes in root.iter_levels():
        for alpha in level_nodes:
            assert not is_admissible(alpha, alpha, DEFAULT_ETA)


def test_admissibility_symmetric() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    level_nodes = root.nodes_at_level(_deepest_level(root))
    for alpha, beta in itertools.product(level_nodes, repeat=2):
        assert is_admissible(alpha, beta, DEFAULT_ETA) == is_admissible(beta, alpha, DEFAULT_ETA)


def _deepest_level(root: TreeNode) -> int:
    deepest = 0
    for level_nodes in root.iter_levels():
        deepest = level_nodes[0].level
    return deepest


# ---------------------------------------------------------------------------
# "Fig. 3" tessellation: small uniform grid, complete + disjoint admissibility
# cover at the deepest level.
#
# Per the task description, this validates the *structural* cover property
# rather than pixel-matching Fig. 3: at the deepest level of a small uniform
# grid (analogous to the depth-3, 16-leaf tree of Fig. 3 over [0, 1]), every
# ordered pair of same-level boxes (alpha, beta) is classified as exactly one
# of:
#   - an inadmissible neighbor pair (red blocks in Fig. 3),
#   - an admissible interaction-list pair (blue blocks in Fig. 3), or
#   - "not present": neither alpha's neighbor nor in alpha's interaction list
#     (beta's parent is not a neighbor of alpha's parent -- too far apart for
#     either category at this level; in Fig. 3 these correspond to blocks
#     handled at a coarser level, e.g. A_{4,6}/A_{4,7} etc., not drawn within
#     the depth-3 leaf grid itself).
# These three cases are disjoint and exhaustive by construction.
# ---------------------------------------------------------------------------


def test_fig3_tessellation_2d_depth3() -> None:
    # 4x4 grid of patches over [0, 1]^2, dyadic-split to depth 2 (4 leaves
    # along each axis -> 16 leaves total), mirroring the 16-leaf, depth-3
    # tree of Fig. 3 (the figure's "depth 3" counts the root as level 0).
    nx, ny = 4, 4
    mesh = _grid_mesh(nx, ny, spacing=1.0 / 4)
    root = build_tree(mesh, m=1)

    deepest = _deepest_level(root)
    assert deepest >= 2
    level_nodes = root.nodes_at_level(deepest)
    assert len(level_nodes) == nx * ny

    nei = neighbor_lists(root)[deepest]
    ints = interaction_lists(root)[deepest]

    n_admissible = 0
    n_inadmissible = 0
    n_absent = 0

    for i, alpha in enumerate(level_nodes):
        nei_ids = {id(b) for b in nei[i]}
        int_ids = {id(b) for b in ints[i]}

        # Disjoint.
        assert nei_ids.isdisjoint(int_ids)

        for beta in level_nodes:
            is_nei = id(beta) in nei_ids
            is_int = id(beta) in int_ids

            # Exactly one of the three categories.
            assert is_nei + is_int <= 1
            admissible = is_admissible(alpha, beta, DEFAULT_ETA)

            if is_nei:
                n_inadmissible += 1
                assert not admissible
            elif is_int:
                n_admissible += 1
                assert admissible
            else:
                n_absent += 1
                # "Not present" pairs are neither neighbors nor in the
                # interaction list; on this small grid they may still test
                # geometrically admissible (they are even farther apart than
                # the interaction list), but they are not part of either
                # tessellation category at this level.

    # The diagonal (alpha, alpha) is always an inadmissible neighbor pair.
    assert n_inadmissible >= len(level_nodes)
    # At least some admissible interaction-list pairs exist on a 4x4 grid.
    assert n_admissible > 0
    assert n_inadmissible + n_admissible + n_absent == len(level_nodes) ** 2


def test_fig3_tessellation_3d() -> None:
    nx, ny, nz = 4, 4, 4
    mesh = _grid_mesh(nx, ny, nz, spacing=1.0 / 4)
    root = build_tree(mesh, m=1)

    deepest = _deepest_level(root)
    level_nodes = root.nodes_at_level(deepest)
    assert len(level_nodes) == nx * ny * nz

    nei = neighbor_lists(root)[deepest]
    ints = interaction_lists(root)[deepest]

    for i, alpha in enumerate(level_nodes):
        nei_ids = {id(b) for b in nei[i]}
        int_ids = {id(b) for b in ints[i]}
        assert nei_ids.isdisjoint(int_ids)

        for beta_id, beta in [(id(b), b) for b in level_nodes]:
            is_nei = beta_id in nei_ids
            is_int = beta_id in int_ids
            assert is_nei + is_int <= 1
            if is_nei:
                assert not is_admissible(alpha, beta, DEFAULT_ETA)
            elif is_int:
                assert is_admissible(alpha, beta, DEFAULT_ETA)


# ---------------------------------------------------------------------------
# suggest_eta
# ---------------------------------------------------------------------------


def test_suggest_eta_default_is_positive_and_o1() -> None:
    eta = suggest_eta()
    assert eta > 0.0
    assert eta < 10.0


def test_suggest_eta_smaller_target_error_increases_eta() -> None:
    eta_loose = suggest_eta(gamma=0.5, target_rel_error=1e-1)
    eta_tight = suggest_eta(gamma=0.5, target_rel_error=1e-3)
    assert eta_tight > eta_loose


def test_suggest_eta_larger_gamma_increases_eta() -> None:
    eta_small_gamma = suggest_eta(gamma=0.1, target_rel_error=1e-2)
    eta_large_gamma = suggest_eta(gamma=1.0, target_rel_error=1e-2)
    assert eta_large_gamma > eta_small_gamma


def test_suggest_eta_zero_gamma_is_zero() -> None:
    assert suggest_eta(gamma=0.0) == 0.0


def test_suggest_eta_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        suggest_eta(gamma=-1.0)
    with pytest.raises(ValueError):
        suggest_eta(target_rel_error=0.0)
    with pytest.raises(ValueError):
        suggest_eta(target_rel_error=1.0)
    with pytest.raises(ValueError):
        suggest_eta(d=4)


# ---------------------------------------------------------------------------
# box_dist input validation
# ---------------------------------------------------------------------------


def test_box_dist_shape_mismatch_raises() -> None:
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        box_dist(a, b)
