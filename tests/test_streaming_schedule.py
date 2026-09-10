"""C.5 streaming schedules: pair ownership and bounded dense lifetimes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.column_basis import column_bases
from gfcompress.fixed_pattern import (
    AdmissibleProbeSchedule,
    LeafProbeSchedule,
    ProbeLifetime,
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


def test_dense_probe_and_sample_peaks_stay_constant_as_schedule_grows() -> None:
    mesh, root, lists, op, level = _problem()
    col = build_admissible_schedule(root, lists, level, mesh, 2, 1, seed=3, side="col")
    row = build_admissible_schedule(root, lists, level, mesh, 2, 1, seed=3, side="row")
    leaf_level = max(nodes[0].level for nodes in root.iter_levels())
    leaf = build_leaf_schedule(root, lists, leaf_level, mesh)

    for multiplier in (1, 2, 3):
        col_life, row_life, leaf_life = ProbeLifetime(), ProbeLifetime(), ProbeLifetime()
        column_bases(
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
            lifetime=col_life,
        )
        row_bases(
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
            lifetime=row_life,
        )
        extract_leaves(
            op,
            root,
            lists,
            mesh,
            leaf_level,
            [],
            schedule=replace(leaf, probes=leaf.probes * multiplier),
            lifetime=leaf_life,
        )
        for lifetime, count in (
            (col_life, len(col.probes)),
            (row_life, len(row.probes)),
            (leaf_life, len(leaf.probes)),
        ):
            assert lifetime.realized == multiplier * count
            assert lifetime.live_probes == lifetime.live_samples == 0
            assert lifetime.peak_live_probes == lifetime.peak_live_samples == 1
