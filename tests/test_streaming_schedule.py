"""C.5 streaming schedules: pair ownership and bounded dense lifetimes."""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import numpy as np

import gfcompress.column_basis as column_module
import gfcompress.leaf as leaf_module
import gfcompress.row_basis as row_module
from gfcompress.build_tree import build_tree
from gfcompress.column_basis import column_bases
from gfcompress.fixed_pattern import (
    AdmissibleProbeSchedule,
    LeafProbeSchedule,
    PeriodicLeafTestMatrix,
    PeriodicTestMatrix,
    build_admissible_schedule,
    build_leaf_schedule,
)
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.leaf import extract_leaves
from gfcompress.mockgf import MockGF
from gfcompress.row_basis import row_bases
from gfcompress.tree import TreeNode


def _problem() -> tuple[FaultMesh, TreeNode, TreeLists, MockGF, int]:
    axis = np.arange(8, dtype=float)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    mesh = FaultMesh(centroids=np.stack((x.ravel(), y.ravel()), axis=1), L=np.full(64, 0.1))
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    level = next(nodes[0].level for nodes in root.iter_levels() if lists.interaction[nodes[0]])
    return mesh, root, lists, MockGF(mesh), level


def test_pair_owners_drive_distinct_column_row_and_leaf_probes() -> None:
    mesh, root, lists, op, level = _problem()
    beta = next(node for node in root.nodes_at_level(level) if len(lists.interaction[node]) >= 2)
    alpha0, alpha1 = lists.interaction[beta][:2]

    col = build_admissible_schedule(root, lists, level, mesh, 2, 1, seed=1, side="col")
    col_alt = replace(col.owners[(alpha0, beta)], active_boxes=[beta], seed=91)
    col_schedule = AdmissibleProbeSchedule(
        probes=[*col.probes, col_alt], owners={**col.owners, (alpha1, beta): col_alt}
    )
    cb = next(
        item
        for item in column_bases(op, root, lists, mesh, level, [], 2, 1, 1, schedule=col_schedule)
        if (item.alpha, item.beta) == (alpha1, beta)
    )
    np.testing.assert_allclose(
        cb.y_alpha, op.block(alpha1.patch_indices, beta.patch_indices) @ cb.g_beta
    )
    assert not np.array_equal(cb.g_beta, col.owners[(alpha0, beta)].realize().blocks[beta])

    row = build_admissible_schedule(root, lists, level, mesh, 2, 1, seed=1, side="row")
    alpha = next(node for node in root.nodes_at_level(level) if len(lists.interaction[node]) >= 2)
    beta0, beta1 = lists.interaction[alpha][:2]
    row_alt = replace(row.owners[(alpha, beta0)], active_boxes=[alpha], seed=92)
    row_schedule = AdmissibleProbeSchedule(
        probes=[*row.probes, row_alt], owners={**row.owners, (alpha, beta1): row_alt}
    )
    rb = next(
        item
        for item in row_bases(op, root, lists, mesh, level, [], 2, 1, 1, schedule=row_schedule)
        if (item.alpha, item.beta) == (alpha, beta1)
    )
    expected_z = op.rmatvec(row_alt.realize().omega)[beta1.col_indices, :]
    np.testing.assert_allclose(
        expected_z, op.block(alpha.patch_indices, beta1.patch_indices).T @ rb.g_alpha
    )
    assert not np.array_equal(rb.g_alpha, row.owners[(alpha, beta0)].realize().blocks[alpha])

    leaf_level = max(nodes[0].level for nodes in root.iter_levels())
    leaf_beta = root.nodes_at_level(leaf_level)[0]
    leaf_alpha0, leaf_alpha1 = [
        node for node in root.nodes_at_level(leaf_level) if leaf_beta in lists.nei[node]
    ][:2]
    leaf = build_leaf_schedule(root, lists, leaf_level, mesh)
    leaf_alt = replace(leaf.owners[(leaf_alpha0, leaf_beta)], active_boxes=[leaf_beta])
    leaf_schedule = LeafProbeSchedule(
        probes=[*leaf.probes, leaf_alt], owners={**leaf.owners, (leaf_alpha1, leaf_beta): leaf_alt}
    )
    found = next(
        item
        for item in extract_leaves(op, root, lists, mesh, leaf_level, [], schedule=leaf_schedule)
        if (item.alpha, item.beta) == (leaf_alpha1, leaf_beta)
    )
    width = len(leaf_beta.col_indices)
    expected_y = op.matvec(leaf_alt.realize())
    np.testing.assert_allclose(
        found.block, expected_y[np.ix_(leaf_alpha1.row_indices, np.arange(width))]
    )


def test_dense_probe_and_sample_objects_are_released_as_schedule_grows(monkeypatch) -> None:
    mesh, root, lists, op, level = _problem()
    col = build_admissible_schedule(root, lists, level, mesh, 2, 1, seed=3, side="col")
    row = build_admissible_schedule(root, lists, level, mesh, 2, 1, seed=3, side="row")
    leaf_level = max(nodes[0].level for nodes in root.iter_levels())
    leaf = build_leaf_schedule(root, lists, leaf_level, mesh)

    def observe(refs, value) -> None:
        gc.collect()
        assert all(ref() is None for ref in refs)
        refs.append(weakref.ref(value))

    col_probes: list[weakref.ReferenceType[np.ndarray]] = []
    row_probes: list[weakref.ReferenceType[np.ndarray]] = []
    leaf_probes: list[weakref.ReferenceType[np.ndarray]] = []
    col_samples: list[weakref.ReferenceType[np.ndarray]] = []
    row_samples: list[weakref.ReferenceType[np.ndarray]] = []
    leaf_samples: list[weakref.ReferenceType[np.ndarray]] = []
    periodic_realize = PeriodicTestMatrix.realize
    leaf_realize = PeriodicLeafTestMatrix.realize
    col_matvec = column_module.peeled_matvec
    row_rmatvec = row_module.peeled_rmatvec
    leaf_matvec = leaf_module.peeled_matvec

    def realize_periodic(probe):
        realized = periodic_realize(probe)
        observe(col_probes if probe.side == "col" else row_probes, realized.omega)
        return realized

    def realize_leaf(probe):
        omega = leaf_realize(probe)
        observe(leaf_probes, omega)
        return omega

    def sampled_col(*args, **kwargs):
        sample = col_matvec(*args, **kwargs)
        observe(col_samples, sample)
        return sample

    def sampled_row(*args, **kwargs):
        sample = row_rmatvec(*args, **kwargs)
        observe(row_samples, sample)
        return sample

    def sampled_leaf(*args, **kwargs):
        sample = leaf_matvec(*args, **kwargs)
        observe(leaf_samples, sample)
        return sample

    monkeypatch.setattr(PeriodicTestMatrix, "realize", realize_periodic)
    monkeypatch.setattr(PeriodicLeafTestMatrix, "realize", realize_leaf)
    monkeypatch.setattr(column_module, "peeled_matvec", sampled_col)
    monkeypatch.setattr(row_module, "peeled_rmatvec", sampled_row)
    monkeypatch.setattr(leaf_module, "peeled_matvec", sampled_leaf)

    for multiplier in (1, 2, 3):
        col_result = column_bases(
            op,
            root,
            lists,
            mesh,
            level,
            [],
            2,
            1,
            3,
            schedule=replace(col, probes=col.probes * multiplier),
        )
        row_result = row_bases(
            op,
            root,
            lists,
            mesh,
            level,
            [],
            2,
            1,
            3,
            schedule=replace(row, probes=row.probes * multiplier),
        )
        leaf_result = extract_leaves(
            op,
            root,
            lists,
            mesh,
            leaf_level,
            [],
            schedule=replace(leaf, probes=leaf.probes * multiplier),
        )
        assert col_result and row_result and leaf_result
        assert all(item.y_alpha.base is None and item.u.base is None for item in col_result)
        gc.collect()
        for refs, count in (
            (col_probes, len(col.probes)),
            (row_probes, len(row.probes)),
            (leaf_probes, len(leaf.probes)),
            (col_samples, len(col.probes)),
            (row_samples, len(row.probes)),
            (leaf_samples, len(leaf.probes)),
        ):
            assert len(refs) == sum(range(1, multiplier + 1)) * count
            assert all(ref() is None for ref in refs)
