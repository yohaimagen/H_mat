"""C.14: pair-owned streaming samples and multilevel numerical checks."""

import weakref

import numpy as np
import pytest

from gfcompress.build_tree import build_tree
from gfcompress.colored_probes import build_colored_admissible_schedule
from gfcompress.column_basis import column_bases
from gfcompress.compress import CountingOperator, compress
from gfcompress.error import relative_error
from gfcompress.fixed_pattern import PeriodicTestMatrix, Side, build_admissible_schedule
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import build_lists
from gfcompress.leaf import extract_leaves
from gfcompress.mockgf import MockGF
from gfcompress.peeling import BlockFactor, finalize_factors, peeled_matvec, peeled_rmatvec
from gfcompress.row_basis import core_matrices, row_bases
from gfcompress.sampling import build_sampling_constraint


def _fixture(dim=2):
    points = np.random.default_rng(0).random((80, dim))
    mesh = FaultMesh(points, np.full(80, 0.1))
    root = build_tree(mesh, 4)
    return mesh, root, build_lists(root)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("side", ["col", "row"])
def test_isolation_streams_shapes_and_direct_samples(dim: int, side: Side) -> None:
    mesh, root, lists = _fixture(dim)
    op = MockGF(mesh, eps=0.1)
    prior = []
    selected_dsatur = False
    for nodes in root.iter_levels():
        level = nodes[0].level
        if level < 2:
            continue
        schedule, selection = build_colored_admissible_schedule(
            root, lists, level, mesh, 3, 5, 11, side
        )
        selected_dsatur |= selection.strategy == "dsatur"
        fixed = build_admissible_schedule(root, lists, level, mesh, 3, 5, 11, side)
        expected_pairs = {(a, b) for a in nodes for b in lists.interaction[a]}
        assert set(schedule.owners) == expected_pairs
        assert len(schedule.probes) * 8 <= selection.fixed_columns
        for probe in schedule.probes:
            realization = probe.realize()
            assert realization.omega.shape == (mesh.n_cols if side == "col" else mesh.n_rows, 8)
            sample = (peeled_matvec if side == "col" else peeled_rmatvec)(
                op, realization.omega, prior
            )
            for (a, b), owner in schedule.owners.items():
                if owner is not probe:
                    continue
                target, source = (a, b) if side == "col" else (b, a)
                constraint = build_sampling_constraint(target, source, lists)
                assert source in probe.active_boxes
                assert not set(probe.active_boxes) & set(constraint.zero_boxes)
                for zero in constraint.zero_boxes:
                    indices = zero.col_indices if side == "col" else zero.row_indices
                    assert not np.any(realization.omega[indices])
                block = op.block(a.patch_indices, b.patch_indices)
                direct = (block if side == "col" else block.T) @ realization.blocks[source]
                indices = a.row_indices if side == "col" else b.col_indices
                np.testing.assert_allclose(sample[indices], direct, rtol=1e-10, atol=1e-9)
                np.testing.assert_array_equal(
                    realization.blocks[source], fixed.owners[a, b].realize().blocks[source]
                )
        # Exact smooth-kernel coarse blocks, solely as a peeling test oracle.
        for a, b in expected_pairs:
            block = op.block(a.patch_indices, b.patch_indices)
            prior.append(BlockFactor(a, b, block, None, np.eye(block.shape[1])))
    if dim == 2:
        assert selected_dsatur


@pytest.mark.parametrize("seed", [11, None])
@pytest.mark.parametrize("side", ["col", "row"])
def test_repeated_sources_keep_actual_pair_sketches_and_stream_lifetimes(
    seed: int | None, side: Side, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh, root, lists = _fixture()
    level = 4
    schedule, selection = build_colored_admissible_schedule(
        root, lists, level, mesh, 3, 5, seed, side
    )
    assert selection.strategy == "dsatur"
    boxes = [box for probe in schedule.probes for box in probe.active_boxes]
    assert len(boxes) > len(set(boxes))
    expected = {
        pair: probe.realize().blocks[pair[1] if side == "col" else pair[0]]
        for pair, probe in schedule.owners.items()
    }
    refs = []
    sample_refs = []
    original = PeriodicTestMatrix.realize

    def realize(probe):
        assert all(ref() is None for ref in refs)
        assert all(ref() is None for ref in sample_refs)
        value = original(probe)
        refs.append(weakref.ref(value.omega))
        return value

    monkeypatch.setattr(PeriodicTestMatrix, "realize", realize)
    original_sample = peeled_matvec if side == "col" else peeled_rmatvec

    def sample(*args):
        value = original_sample(*args)
        sample_refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(
        (
            "gfcompress.column_basis.peeled_matvec"
            if side == "col"
            else "gfcompress.row_basis.peeled_rmatvec"
        ),
        sample,
    )
    op = CountingOperator(MockGF(mesh, eps=0.1))
    bases = (column_bases if side == "col" else row_bases)(
        op, root, lists, mesh, level, [], 3, 5, seed, schedule
    )
    assert all(ref() is None for ref in refs)
    assert all(ref() is None for ref in sample_refs)
    assert op.matvec_columns + op.rmatvec_columns == len(schedule.probes) * 8
    assert op.matvec_calls + op.rmatvec_calls == len(schedule.probes)
    for basis in bases:
        actual = basis.g_beta if side == "col" else basis.g_alpha
        np.testing.assert_array_equal(actual, expected[basis.alpha, basis.beta])
        assert actual.flags.owndata
    if seed is None:
        repeated = next(box for box in boxes if boxes.count(box) > 1)
        sketches = [
            expected[pair] for pair in expected if pair[1 if side == "col" else 0] is repeated
        ]
        assert any(not np.array_equal(sketches[0], other) for other in sketches)


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_multilevel_quality_and_counts(seed: int) -> None:
    mesh, root, lists = _fixture()
    op = MockGF(mesh, eps=0.02)
    counted = CountingOperator(op)
    factors = []
    expected_col = expected_row = fixed_columns = 0
    selected = []
    for level in range(2, root.leaves()[0].level + 1):
        col, cm = build_colored_admissible_schedule(root, lists, level, mesh, 4, 10, seed, "col")
        row, rm = build_colored_admissible_schedule(root, lists, level, mesh, 4, 10, seed, "row")
        cb = column_bases(counted, root, lists, mesh, level, factors, 4, 10, seed, col)
        rb = row_bases(counted, root, lists, mesh, level, factors, 4, 10, seed, row)
        factors += core_matrices(cb, rb)
        expected_col += len(col.probes) * 14
        expected_row += len(row.probes) * 14
        fixed_columns += cm.fixed_columns + rm.fixed_columns
        selected += [cm.strategy, rm.strategy]
    assert "dsatur" in selected
    assert counted.matvec_columns == expected_col and counted.rmatvec_columns == expected_row
    assert expected_col + expected_row <= fixed_columns
    leaves = extract_leaves(counted, root, lists, mesh, root.leaves()[0].level, factors)
    colored = HMatrix(root, mesh, finalize_factors(factors), leaves)
    fixed = compress(op, mesh, m=4, k=4, p=10, seed=seed)
    errors = [relative_error(h, op, seed=101) for h in (colored, fixed)]
    leaves_error = relative_error(HMatrix(root, mesh, [], leaves), op, seed=101)
    print(seed, expected_col + expected_row, fixed_columns, errors, leaves_error)
    assert max(errors) < 5e-4
    assert max(errors) * 100 < leaves_error
    assert errors[0] < max(3 * errors[1], 1e-12)
    # Confirm genuine rank truncation occurs, not only exact small blocks.
    singular = [
        np.linalg.svd(op.block(a.patch_indices, b.patch_indices), compute_uv=False)
        for a in root.nodes_at_level(2)
        for b in lists.interaction[a]
    ]
    assert any(len(s) > 4 and s[4] > 1e-8 * s[0] for s in singular)


def test_invalid_ranks() -> None:
    mesh, root, lists = _fixture()
    for k, p in [(0, 2), (2, -1), (True, 2), (2, 0.5)]:
        with pytest.raises(ValueError, match="integers"):
            build_colored_admissible_schedule(root, lists, 2, mesh, k, p)


def test_schedule_construction_is_lazy_and_fixed_fallback_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, root, lists = _fixture()

    def forbidden(*args):
        raise AssertionError("schedule construction must not realize probes")

    monkeypatch.setattr(PeriodicTestMatrix, "realize", forbidden)
    for level in (2, 4, 100):
        schedule, selection = build_colored_admissible_schedule(root, lists, level, mesh, 3, 5, 11)
        assert len(schedule.probes) * 8 <= selection.fixed_columns
        if level == 2:
            assert selection.strategy == "fixed"
            fixed = build_admissible_schedule(root, lists, level, mesh, 3, 5, 11)
            assert schedule == fixed
        if level == 100:
            assert schedule.probes == [] and schedule.owners == {}
