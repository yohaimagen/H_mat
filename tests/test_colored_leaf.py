"""C.15 neighbor-only graphs, identity isolation, and residual leaf accuracy."""

import weakref
from itertools import combinations

import numpy as np
import pytest

from gfcompress.build_tree import build_tree
from gfcompress.colored_leaf import build_colored_leaf_schedule
from gfcompress.colored_probes import build_colored_admissible_schedule
from gfcompress.coloring import fixed_pattern_coloring
from gfcompress.column_basis import column_bases
from gfcompress.compress import CountingOperator
from gfcompress.constraint_graph import build_constraint_graph
from gfcompress.error import relative_error
from gfcompress.fixed_pattern import PeriodicLeafTestMatrix, build_leaf_schedule
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import build_lists
from gfcompress.leaf import extract_leaves
from gfcompress.mockgf import MockGF
from gfcompress.peeling import BlockFactor, finalize_factors, peeled_matvec
from gfcompress.row_basis import core_matrices, row_bases


def _fixture(dim=2, n=60, seed=20):
    mesh = FaultMesh(np.random.default_rng(seed).random((n, dim)), np.full(n, 0.1))
    root = build_tree(mesh, 4)
    return mesh, root, build_lists(root), root.leaves()[0].level


def _approximate_factors(op, mesh, root, lists, level, k, seed):
    factors = []
    for lev in range(2, level + 1):
        col, _ = build_colored_admissible_schedule(root, lists, lev, mesh, k, 10, seed, "col")
        row, _ = build_colored_admissible_schedule(root, lists, lev, mesh, k, 10, seed, "row")
        cb = column_bases(op, root, lists, mesh, lev, factors, k, 10, seed, col)
        rb = row_bases(op, root, lists, mesh, lev, factors, k, 10, seed, row)
        factors += core_matrices(cb, rb)
    return factors


@pytest.mark.parametrize("dim", [2, 3])
def test_neighbor_graph_oracle_and_transpose(dim: int) -> None:
    _, root, lists, level = _fixture(dim)
    col = build_constraint_graph(root, lists, level, neighbors_only=True)
    row = build_constraint_graph(root, lists, level, "row", neighbors_only=True)
    expected = {(a, b) for a in root.leaves() for b in lists.nei[a]}
    for side, graph in (("col", col), ("row", row)):
        assert set(graph.owners) == expected
        for (a, b), vertex in graph.owners.items():
            target, source = (a, b) if side == "col" else (b, a)
            assert graph.constraints[vertex].random_box is source
            assert graph.constraints[vertex].zero_boxes == set(lists.nei[target]) - {source}
        edges = {
            (i, j)
            for i, j in combinations(range(len(graph.constraints)), 2)
            if graph.constraints[i].random_box in graph.constraints[j].zero_boxes
            or graph.constraints[j].random_box in graph.constraints[i].zero_boxes
        }
        assert edges == {
            (i, j) for i, neighbors in enumerate(graph.adjacency) for j in neighbors if i < j
        }
        assert all(i not in neighbors for i, neighbors in enumerate(graph.adjacency))
        assert graph.edge_count == len(edges)
    for a, b in expected:
        assert col.constraints[col.owners[a, b]] == row.constraints[row.owners[b, a]]


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_factor_isolation_uneven_occupancy_and_counts(
    dim: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh, root, lists, level = _fixture(dim)
    op = MockGF(mesh, eps=0.1)
    factors = [
        BlockFactor(
            a, b, op.block(a.patch_indices, b.patch_indices), None, np.eye(len(b.col_indices))
        )
        for nodes in root.iter_levels()
        for a in nodes
        for b in lists.interaction[a]
    ]
    schedule, metrics = build_colored_leaf_schedule(root, lists, level, mesh)
    fixed = build_leaf_schedule(root, lists, level, mesh)
    width = max(len(b.col_indices) for b in root.leaves())
    assert len({len(b.col_indices) for b in root.leaves()}) > 1
    assert metrics.width == width
    assert metrics.fixed_columns == sum(p.w_max for p in fixed.probes)
    if dim == 2:
        boxes = [b for p in schedule.probes for b in p.active_boxes]
        assert len(boxes) > len(set(boxes))
        assert metrics.strategy == "dsatur"
    for probe in schedule.probes:
        omega = probe.realize()
        assert probe.w_max == width and omega.shape == (mesh.n_cols, width)
        for b in probe.active_boxes:
            expected = np.zeros((len(b.col_indices), width))
            expected[:, : len(b.col_indices)] = np.eye(len(b.col_indices))
            np.testing.assert_array_equal(omega[b.col_indices], expected)
        for (a, b), owner in schedule.owners.items():
            if owner is probe:
                assert set(probe.active_boxes) & set(lists.nei[a]) == {b}
    del omega
    probe_refs, sample_refs = [], []
    realize = PeriodicLeafTestMatrix.realize

    def tracked_realize(probe):
        assert all(ref() is None for ref in probe_refs + sample_refs)
        value = realize(probe)
        probe_refs.append(weakref.ref(value))
        return value

    def tracked_sample(*args):
        value = peeled_matvec(*args)
        sample_refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(PeriodicLeafTestMatrix, "realize", tracked_realize)
    monkeypatch.setattr("gfcompress.leaf.peeled_matvec", tracked_sample)
    counted = CountingOperator(op)
    leaves = extract_leaves(counted, root, lists, mesh, level, factors, schedule)
    assert all(ref() is None for ref in probe_refs + sample_refs)
    pairs = [(leaf.alpha, leaf.beta) for leaf in leaves]
    assert len(pairs) == len(set(pairs)) == len(schedule.owners)
    assert set(pairs) == {(a, b) for a in root.leaves() for b in lists.nei[a]}
    assert counted.matvec_calls == len(schedule.probes)
    assert counted.matvec_columns == len(schedule.probes) * width <= metrics.fixed_columns
    assert counted.rmatvec_calls == counted.rmatvec_columns == 0
    predicted = metrics.dsatur_columns if metrics.strategy == "dsatur" else metrics.fixed_columns
    assert counted.matvec_columns == predicted
    for leaf in leaves:
        np.testing.assert_allclose(
            leaf.block,
            op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices),
            rtol=1e-8,
            atol=1e-8,
        )
        assert leaf.block.flags.owndata


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("seed", [11, 23, 37])
def test_approximate_peeling_preserves_near_field_accuracy(dim: int, seed: int) -> None:
    mesh, root, lists, level = _fixture(dim, n=80, seed=0)
    op = MockGF(mesh, eps=0.02)
    # The 3D rank-4/seed-23 residual exceeds the 5% leaf criterion (see below).
    # Keep that criterion and use rank 6, still below some block dimensions.
    k = 4 if dim == 2 else 6
    factors = _approximate_factors(op, mesh, root, lists, level, k, seed)
    schedule, metrics = build_colored_leaf_schedule(root, lists, level, mesh)
    counted = CountingOperator(op)
    colored = extract_leaves(counted, root, lists, mesh, level, factors, schedule)
    assert counted.matvec_columns == len(schedule.probes) * metrics.width <= metrics.fixed_columns
    fixed = extract_leaves(op, root, lists, mesh, level, factors)
    max_errors = []
    for leaves in (colored, fixed):
        errors = []
        for leaf in leaves:
            exact = op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices)
            errors.append(np.linalg.norm(leaf.block - exact) / np.linalg.norm(exact))
        max_errors.append(max(errors))
        assert max(errors) < 5e-2  # Existing near-field criterion, unchanged.
    assert max_errors[0] > 1e-10  # Approximate peeling is not exact extraction.
    finalized = finalize_factors(factors)
    colored_h = HMatrix(root, mesh, finalized, colored)
    fixed_h = HMatrix(root, mesh, finalized, fixed)
    errors = [relative_error(h, op, seed=101) for h in (colored_h, fixed_h)]
    assert max(errors) < 5e-4
    assert errors[0] < max(3 * errors[1], 1e-12)
    print(dim, seed, counted.matvec_columns, metrics.fixed_columns, max_errors, errors)


def test_approximate_peeling_can_cross_quality_threshold() -> None:
    """Cost selection cannot guarantee the accuracy of an approximate residual."""
    mesh, root, lists, level = _fixture(3, n=80, seed=0)
    op = MockGF(mesh, eps=0.02)
    factors = _approximate_factors(op, mesh, root, lists, level, 4, 23)
    schedule, _ = build_colored_leaf_schedule(root, lists, level, mesh)
    worst = []
    for selected in (schedule, None):
        leaves = extract_leaves(op, root, lists, mesh, level, factors, selected)
        errors = []
        for leaf in leaves:
            exact = op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices)
            errors.append(np.linalg.norm(leaf.block - exact) / np.linalg.norm(exact))
        worst.append(max(errors))
    print("3D rank-4/seed-23 negative case, colored/fixed leaf errors:", worst)
    assert worst[0] > 5e-2
    assert worst[1] < 5e-2


def test_fixed_fallback_tie_and_lazy_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh, root, lists, level = _fixture()
    graph = build_constraint_graph(root, lists, level, neighbors_only=True)
    fixed_colors, _ = fixed_pattern_coloring(graph, root, level, leaf=True)

    def forbidden(*args):
        raise AssertionError("schedule building must not realize a probe")

    monkeypatch.setattr(PeriodicLeafTestMatrix, "realize", forbidden)
    _, natural = build_colored_leaf_schedule(root, lists, level, mesh)
    assert natural.strategy == "dsatur"
    # Valid, deliberately nonoptimal coloring exercises the cheaper fallback.
    monkeypatch.setattr(
        "gfcompress.coloring.dsatur", lambda adjacency: tuple(range(len(adjacency)))
    )
    schedule, metrics = build_colored_leaf_schedule(root, lists, level, mesh)
    assert metrics.reason == "fixed_cheaper"
    assert schedule == build_leaf_schedule(root, lists, level, mesh)
    monkeypatch.setattr("gfcompress.coloring.dsatur", lambda adjacency: fixed_colors)
    schedule, metrics = build_colored_leaf_schedule(root, lists, level, mesh)
    assert metrics.reason == "fixed_tie"
    assert schedule == build_leaf_schedule(root, lists, level, mesh)


def test_invalid_leaf_level() -> None:
    mesh, root, lists, _ = _fixture()
    for level in (0, 100):
        with pytest.raises(ValueError, match="leaf level"):
            build_colored_leaf_schedule(root, lists, level, mesh)


@pytest.mark.parametrize("dim", [2, 3])
def test_single_leaf_fixed_tie_has_exact_observed_count(dim: int) -> None:
    mesh = FaultMesh(np.random.default_rng(0).random((3, dim)), np.full(3, 0.1))
    root = build_tree(mesh, 4)
    lists = build_lists(root)
    schedule, metrics = build_colored_leaf_schedule(root, lists, 0, mesh)
    assert metrics.strategy == "fixed" and metrics.reason == "fixed_tie"
    op = MockGF(mesh, eps=0.1)
    counted = CountingOperator(op)
    leaves = extract_leaves(counted, root, lists, mesh, 0, [], schedule)
    assert len(leaves) == 1
    assert counted.matvec_columns == metrics.fixed_columns == mesh.n_cols
    assert counted.matvec_calls == metrics.fixed_probes == 1
    np.testing.assert_allclose(leaves[0].block, op.block(root.patch_indices, root.patch_indices))
