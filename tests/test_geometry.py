"""Tests for FaultMesh / Patch geometry helpers (Task 1.1)."""

import numpy as np
import pytest

from gfcompress.geometry import FaultMesh, Patch, pairwise_distances


def _random_mesh(d: int, n: int, seed: int = 0) -> FaultMesh:
    rng = np.random.default_rng(seed)
    centroids = rng.uniform(-1.0, 1.0, size=(n, d))
    L = rng.uniform(0.1, 0.5, size=n)
    return FaultMesh(centroids=centroids, L=L)


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------


def test_patch_2d() -> None:
    p = Patch(centroid=np.array([1.0, 2.0]), L=0.5)
    assert p.centroid.shape == (2,)
    assert p.L == 0.5


def test_patch_3d() -> None:
    p = Patch(centroid=np.array([1.0, 2.0, 3.0]), L=0.25)
    assert p.centroid.shape == (3,)


def test_patch_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        Patch(centroid=np.array([1.0]), L=0.5)
    with pytest.raises(ValueError):
        Patch(centroid=np.array([1.0, 2.0, 3.0, 4.0]), L=0.5)


# ---------------------------------------------------------------------------
# FaultMesh construction & derived dof
# ---------------------------------------------------------------------------


def test_faultmesh_2d_dof() -> None:
    mesh = _random_mesh(d=2, n=5)
    assert mesh.d == 2
    assert mesh.dof_row == 2
    assert mesh.dof_col == 1
    assert mesh.n_patches == 5
    assert mesh.n_rows == 2 * 5
    assert mesh.n_cols == 1 * 5


def test_faultmesh_3d_dof() -> None:
    mesh = _random_mesh(d=3, n=7)
    assert mesh.d == 3
    assert mesh.dof_row == 3
    assert mesh.dof_col == 2
    assert mesh.n_patches == 7
    assert mesh.n_rows == 3 * 7
    assert mesh.n_cols == 2 * 7


def test_faultmesh_bad_centroid_shape_raises() -> None:
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros((5, 4)), L=np.ones(5))
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros(5), L=np.ones(5))


def test_faultmesh_mismatched_L_raises() -> None:
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros((5, 2)), L=np.ones(4))


# ---------------------------------------------------------------------------
# patch_to_rows / patch_to_cols
# ---------------------------------------------------------------------------


def test_patch_to_rows_2d_full_set_size_and_partition() -> None:
    n = 6
    mesh = _random_mesh(d=2, n=n)
    all_patches = np.arange(n)
    rows = mesh.patch_to_rows(all_patches)
    assert rows.shape == (mesh.dof_row * n,)
    assert rows.shape[0] == 2 * n
    # Full set must be a permutation of 0..2N-1 (complete, disjoint cover).
    assert sorted(rows.tolist()) == list(range(2 * n))


def test_patch_to_cols_2d_full_set_size_and_partition() -> None:
    n = 6
    mesh = _random_mesh(d=2, n=n)
    all_patches = np.arange(n)
    cols = mesh.patch_to_cols(all_patches)
    assert cols.shape == (mesh.dof_col * n,)
    assert cols.shape[0] == 1 * n
    assert sorted(cols.tolist()) == list(range(n))


def test_patch_to_rows_3d_full_set_size_and_partition() -> None:
    n = 5
    mesh = _random_mesh(d=3, n=n)
    all_patches = np.arange(n)
    rows = mesh.patch_to_rows(all_patches)
    assert rows.shape == (3 * n,)
    assert sorted(rows.tolist()) == list(range(3 * n))


def test_patch_to_cols_3d_full_set_size_and_partition() -> None:
    n = 5
    mesh = _random_mesh(d=3, n=n)
    all_patches = np.arange(n)
    cols = mesh.patch_to_cols(all_patches)
    assert cols.shape == (2 * n,)
    assert sorted(cols.tolist()) == list(range(2 * n))


def test_patch_to_rows_block_interleaved_2d() -> None:
    mesh = _random_mesh(d=2, n=4)
    # Patch-major, block-interleaved: patch i -> rows [2i, 2i+1].
    rows = mesh.patch_to_rows(np.array([0, 2, 3]))
    np.testing.assert_array_equal(rows, np.array([0, 1, 4, 5, 6, 7]))


def test_patch_to_cols_block_interleaved_3d() -> None:
    mesh = _random_mesh(d=3, n=4)
    # Patch-major, block-interleaved: patch i -> cols [2i, 2i+1] (dof_col=2).
    cols = mesh.patch_to_cols(np.array([1, 3]))
    np.testing.assert_array_equal(cols, np.array([2, 3, 6, 7]))


def test_patch_to_rows_3d_block_interleaved() -> None:
    mesh = _random_mesh(d=3, n=4)
    # patch i -> rows [3i, 3i+1, 3i+2] (dof_row=3).
    rows = mesh.patch_to_rows(np.array([0, 2]))
    np.testing.assert_array_equal(rows, np.array([0, 1, 2, 6, 7, 8]))


def test_disjoint_subsets_partition_full_index_set_2d() -> None:
    n = 8
    mesh = _random_mesh(d=2, n=n)
    left = np.arange(0, n // 2)
    right = np.arange(n // 2, n)

    rows_left = mesh.patch_to_rows(left)
    rows_right = mesh.patch_to_rows(right)

    assert set(rows_left.tolist()).isdisjoint(set(rows_right.tolist()))
    union = sorted(rows_left.tolist() + rows_right.tolist())
    assert union == list(range(mesh.n_rows))

    cols_left = mesh.patch_to_cols(left)
    cols_right = mesh.patch_to_cols(right)
    assert set(cols_left.tolist()).isdisjoint(set(cols_right.tolist()))
    union_cols = sorted(cols_left.tolist() + cols_right.tolist())
    assert union_cols == list(range(mesh.n_cols))


# ---------------------------------------------------------------------------
# pairwise_distances
# ---------------------------------------------------------------------------


def test_pairwise_distances_shape_and_zero_diagonal() -> None:
    mesh = _random_mesh(d=2, n=10)
    dist = pairwise_distances(mesh.centroids)
    assert dist.shape == (10, 10)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-12)
    # Symmetric.
    np.testing.assert_allclose(dist, dist.T, atol=1e-12)


def test_pairwise_distances_known_values_2d() -> None:
    centroids = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    dist = pairwise_distances(centroids)
    expected = np.array(
        [
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]
    )
    np.testing.assert_allclose(dist, expected, atol=1e-12)


def test_pairwise_distances_known_values_3d() -> None:
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    dist = pairwise_distances(centroids)
    np.testing.assert_allclose(dist[0, 1], 3.0, atol=1e-12)
    np.testing.assert_allclose(dist[1, 0], 3.0, atol=1e-12)
