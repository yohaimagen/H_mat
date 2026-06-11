"""Tests for the analytic-kernel mock GF operator `MockGF` (Task 2.2).

`MockGF` wraps a smooth, `1/r^d`-decaying tensor kernel `K(x_i, x_j) ->
R^{dof_row x dof_col}` and assembles the dense patch-major-flattened operator
`A`, shape `(dof_row * N, dof_col * N)` (`2N x N` in 2D, `3N x 2N` in 3D). The
key structural property tested here is the one CLAUDE.md calls out: blocks
between well-separated (admissible) patch clusters must be numerically low
rank (decaying singular values), while near-diagonal blocks must not be.
"""

from __future__ import annotations

import numpy as np

from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF, kernel_block
from gfcompress.operators import MatVecOperator


def _grid_mesh(*shape: int, spacing: float = 1.0, origin: float = 0.0) -> FaultMesh:
    """A `FaultMesh` whose centroids form a regular grid of `shape`,
    translated by `origin` along every axis."""
    axes = [np.arange(n, dtype=float) * spacing + origin for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    L = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=L)


def _two_cluster_mesh(d: int, n_side: int = 4, gap: float = 100.0) -> FaultMesh:
    """A mesh with two well-separated clusters of `n_side^d` points each: a
    unit-spacing grid near the origin, and a congruent grid shifted by `gap`
    along the first axis. The clusters' separation (`~gap`) is huge relative
    to each cluster's diameter (`O(n_side)`), so the cross block between them
    is a strongly-admissible far-field interaction."""
    if d == 2:
        cluster_a = _grid_mesh(n_side, n_side)
        cluster_b = _grid_mesh(n_side, n_side, origin=gap)
    else:
        cluster_a = _grid_mesh(n_side, n_side, n_side)
        cluster_b = _grid_mesh(n_side, n_side, n_side, origin=gap)

    centroids = np.concatenate([cluster_a.centroids, cluster_b.centroids], axis=0)
    L = np.concatenate([cluster_a.L, cluster_b.L], axis=0)
    return FaultMesh(centroids=centroids, L=L)


# ---------------------------------------------------------------------------
# kernel_block
# ---------------------------------------------------------------------------


def test_kernel_block_shape_2d() -> None:
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    block = kernel_block(x, y, dof_row=2, dof_col=1)
    assert block.shape == (2, 1)


def test_kernel_block_shape_3d() -> None:
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([1.0, 2.0, 3.0])
    block = kernel_block(x, y, dof_row=3, dof_col=2)
    assert block.shape == (3, 2)


def test_kernel_block_decays_like_one_over_r_to_the_d() -> None:
    """Far field: ||K(x,y)|| ~ 1/r^d for r >> eps."""
    eps = 1e-6
    origin = np.array([0.0, 0.0])

    near = kernel_block(origin, np.array([1.0, 0.0]), dof_row=2, dof_col=1, eps=eps)
    far = kernel_block(origin, np.array([10.0, 0.0]), dof_row=2, dof_col=1, eps=eps)

    ratio = np.linalg.norm(far) / np.linalg.norm(near)
    # r increases by 10x, d=2 => norm should drop by ~100x.
    np.testing.assert_allclose(ratio, 1e-2, rtol=0.05)


def test_kernel_block_finite_for_coincident_points() -> None:
    x = np.array([1.0, 2.0])
    block = kernel_block(x, x, dof_row=2, dof_col=1, eps=1e-3)
    assert np.all(np.isfinite(block))


# ---------------------------------------------------------------------------
# MockGF: construction, shape, matvec/rmatvec contract
# ---------------------------------------------------------------------------


def test_mockgf_is_matvec_operator() -> None:
    mesh = _grid_mesh(3, 3)
    op = MockGF(mesh)
    assert isinstance(op, MatVecOperator)


def test_mockgf_shape_2d_is_2n_by_n() -> None:
    mesh = _grid_mesh(4, 4)
    n = mesh.n_patches
    op = MockGF(mesh)
    assert op.shape == (2 * n, n)


def test_mockgf_shape_3d_is_3n_by_2n() -> None:
    mesh = _grid_mesh(3, 3, 3)
    n = mesh.n_patches
    op = MockGF(mesh)
    assert op.shape == (3 * n, 2 * n)


def test_matvec_matches_dense_assembly_2d() -> None:
    mesh = _grid_mesh(4, 4)
    op = MockGF(mesh)
    rng = np.random.default_rng(0)

    omega = rng.standard_normal(op.shape[1])
    np.testing.assert_allclose(op.matvec(omega), op.A @ omega)

    omega_thick = rng.standard_normal((op.shape[1], 5))
    np.testing.assert_allclose(op.matvec(omega_thick), op.A @ omega_thick)


def test_rmatvec_matches_dense_assembly_3d() -> None:
    mesh = _grid_mesh(3, 3, 3)
    op = MockGF(mesh)
    rng = np.random.default_rng(1)

    psi = rng.standard_normal(op.shape[0])
    np.testing.assert_allclose(op.rmatvec(psi), op.A.T @ psi)

    psi_thick = rng.standard_normal((op.shape[0], 5))
    np.testing.assert_allclose(op.rmatvec(psi_thick), op.A.T @ psi_thick)


def test_matvec_and_rmatvec_act_on_different_spaces() -> None:
    mesh = _grid_mesh(4, 4)
    op = MockGF(mesh)
    n_rows, n_cols = op.shape
    assert n_rows != n_cols

    omega = np.zeros(n_cols)
    assert op.matvec(omega).shape == (n_rows,)

    psi = np.zeros(n_rows)
    assert op.rmatvec(psi).shape == (n_cols,)


# ---------------------------------------------------------------------------
# block() ground-truth accessor
# ---------------------------------------------------------------------------


def test_block_matches_dense_assembly() -> None:
    mesh = _grid_mesh(4, 4)
    op = MockGF(mesh)

    row_patches = np.array([0, 1, 5, 7])
    col_patches = np.array([2, 3, 9])
    block = op.block(row_patches, col_patches)

    expected_rows = mesh.patch_to_rows(row_patches)
    expected_cols = mesh.patch_to_cols(col_patches)
    np.testing.assert_allclose(block, op.A[np.ix_(expected_rows, expected_cols)])

    dof_row, dof_col = mesh.dof_row, mesh.dof_col
    assert block.shape == (dof_row * len(row_patches), dof_col * len(col_patches))


def test_block_full_mesh_recovers_a() -> None:
    mesh = _grid_mesh(3, 3)
    op = MockGF(mesh)
    all_patches = np.arange(mesh.n_patches)
    np.testing.assert_allclose(op.block(all_patches, all_patches), op.A)


# ---------------------------------------------------------------------------
# Rank structure: the central test of Task 2.2
# ---------------------------------------------------------------------------


def test_admissible_block_singular_values_decay_2d() -> None:
    """Cross block between two well-separated clusters must be numerically
    low rank: singular values decay rapidly."""
    mesh = _two_cluster_mesh(d=2, n_side=4, gap=100.0)
    op = MockGF(mesh)

    n_cluster = 4 * 4
    cluster_a = np.arange(n_cluster)
    cluster_b = np.arange(n_cluster, 2 * n_cluster)

    block = op.block(cluster_a, cluster_b)
    sv = np.linalg.svd(block, compute_uv=False)

    assert sv[0] > 0.0
    # Numerically low rank: trailing singular values are tiny relative to
    # the largest, i.e. the effective rank is well below full rank.
    rel = sv / sv[0]
    assert rel[-1] < 1e-6


def test_admissible_block_singular_values_decay_3d() -> None:
    mesh = _two_cluster_mesh(d=3, n_side=3, gap=100.0)
    op = MockGF(mesh)

    n_cluster = 3 * 3 * 3
    cluster_a = np.arange(n_cluster)
    cluster_b = np.arange(n_cluster, 2 * n_cluster)

    block = op.block(cluster_a, cluster_b)
    sv = np.linalg.svd(block, compute_uv=False)

    assert sv[0] > 0.0
    rel = sv / sv[0]
    assert rel[-1] < 1e-6


def test_near_diagonal_block_is_not_low_rank() -> None:
    """A block of nearby (non-separated) patches must NOT be numerically low
    rank: its singular values stay close to the largest, i.e. it is close to
    full rank."""
    mesh = _grid_mesh(6, 6)
    op = MockGF(mesh)

    # A contiguous patch of nearby patches (small, dense local grid).
    patches = np.arange(16)  # a 4x4 sub-block of the 6x6 grid, patch-major order
    block = op.block(patches, patches)
    sv = np.linalg.svd(block, compute_uv=False)

    rel = sv / sv[0]
    # Full-rank-ish: the smallest singular value is not vanishingly small
    # relative to the largest, unlike the admissible far-field block.
    assert rel[-1] > 1e-3


def test_near_diagonal_vs_admissible_decay_contrast() -> None:
    """Directly contrast decay rates: the admissible block's singular-value
    spectrum decays far faster than the near-diagonal block's."""
    mesh = _two_cluster_mesh(d=2, n_side=4, gap=100.0)
    op = MockGF(mesh)
    n_cluster = 16

    cluster_a = np.arange(n_cluster)
    cluster_b = np.arange(n_cluster, 2 * n_cluster)

    far_block = op.block(cluster_a, cluster_b)
    near_block = op.block(cluster_a, cluster_a)

    far_sv = np.linalg.svd(far_block, compute_uv=False)
    near_sv = np.linalg.svd(near_block, compute_uv=False)

    far_decay = far_sv[-1] / far_sv[0]
    near_decay = near_sv[-1] / near_sv[0]

    assert far_decay < near_decay
    assert far_decay < 1e-6
    assert near_decay > 1e-3
