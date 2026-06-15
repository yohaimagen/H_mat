"""Tests for Gaussian sampling, orthonormalization, two-sample compression,
and core-matrix-solve helpers (Tasks 3.1-3.3).

`gaussian(n, k, p, seed)` draws an `n x (k+p)` standard-normal matrix
(seedable via `numpy.random.default_rng`); `orth(Y)` and `orth(Y, k)` wrap
`scipy.linalg.qr` to return an orthonormal basis `Q` for `range(Y)` (thin /
rank-`k`-truncated). `two_sample_compress(Y, Z, k)` (Algorithm 2.1) returns
`(U, V) = (qr(Y, k), qr(Z, k))`, orthonormal bases for a single block's
column and row spaces. `core_matrix_solve(U, V, Y_alpha, G_alpha, G_beta)`
(Eq. 4.3) returns the `k x k` core matrix `B` such that `U @ B @ V*`
approximates the block.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF
from gfcompress.randomized import core_matrix_solve, gaussian, orth, two_sample_compress

# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------


def test_gaussian_shape() -> None:
    omega = gaussian(10, 3, 2, seed=0)
    assert omega.shape == (10, 5)


def test_gaussian_default_oversampling_is_zero() -> None:
    omega = gaussian(8, 4, seed=0)
    assert omega.shape == (8, 4)


def test_gaussian_is_seedable_and_reproducible() -> None:
    a = gaussian(6, 3, 2, seed=42)
    b = gaussian(6, 3, 2, seed=42)
    np.testing.assert_array_equal(a, b)


def test_gaussian_different_seeds_differ() -> None:
    a = gaussian(6, 3, 2, seed=1)
    b = gaussian(6, 3, 2, seed=2)
    assert not np.allclose(a, b)


def test_gaussian_entries_are_standard_normal() -> None:
    omega = gaussian(2000, 1, 0, seed=7)
    assert omega.mean() == pytest.approx(0.0, abs=0.1)
    assert omega.std() == pytest.approx(1.0, rel=0.1)


def test_gaussian_rejects_negative_dimensions() -> None:
    with pytest.raises(ValueError):
        gaussian(-1, 3, 2)
    with pytest.raises(ValueError):
        gaussian(10, -1, 2)
    with pytest.raises(ValueError):
        gaussian(10, 3, -1)


# ---------------------------------------------------------------------------
# orth
# ---------------------------------------------------------------------------


def test_orth_returns_orthonormal_columns_for_tall_matrix() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal((20, 5))

    q = orth(y)

    assert q.shape == (20, 5)
    np.testing.assert_allclose(q.T @ q, np.eye(5), atol=1e-10)


def test_orth_returns_orthonormal_columns_for_wide_matrix() -> None:
    rng = np.random.default_rng(1)
    y = rng.standard_normal((4, 9))

    q = orth(y)

    assert q.shape == (4, 4)
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-10)


def test_orth_spans_same_column_space() -> None:
    rng = np.random.default_rng(2)
    y = rng.standard_normal((15, 4))

    q = orth(y)

    # Q Q^T Y == Y up to numerical error (Y already lies in range(Q)).
    np.testing.assert_allclose(q @ (q.T @ y), y, atol=1e-10)


def test_orth_rank_k_truncation_shape_and_orthonormality() -> None:
    rng = np.random.default_rng(3)
    y = rng.standard_normal((30, 8))

    q_full = orth(y)
    q_k = orth(y, k=3)

    assert q_k.shape == (30, 3)
    np.testing.assert_allclose(q_k.T @ q_k, np.eye(3), atol=1e-10)
    # Truncation keeps the leading columns of the full Q.
    np.testing.assert_allclose(q_k, q_full[:, :3])


def test_orth_rank_k_zero_returns_empty_basis() -> None:
    rng = np.random.default_rng(4)
    y = rng.standard_normal((10, 5))

    q_0 = orth(y, k=0)

    assert q_0.shape == (10, 0)


def test_orth_rank_k_full_rank_matches_untruncated() -> None:
    rng = np.random.default_rng(5)
    y = rng.standard_normal((12, 4))

    q_full = orth(y)
    q_k = orth(y, k=4)

    np.testing.assert_allclose(q_k, q_full)


def test_orth_rejects_k_out_of_range() -> None:
    rng = np.random.default_rng(6)
    y = rng.standard_normal((10, 4))

    with pytest.raises(ValueError):
        orth(y, k=-1)
    with pytest.raises(ValueError):
        orth(y, k=5)


# ---------------------------------------------------------------------------
# Integration: orth on a sketch Y = A @ Omega built from gaussian()
# ---------------------------------------------------------------------------


def test_orth_of_gaussian_sketch_of_low_rank_matrix_recovers_range() -> None:
    """For a rank-k matrix A and a sketch Y = A @ Omega with Omega drawn from
    gaussian(), orth(Y, k) should recover an orthonormal basis for range(A)
    (up to numerical error), since k+p >= k columns suffice."""
    rng = np.random.default_rng(10)
    n, r = 25, 3
    u = orth(rng.standard_normal((n, r)))
    a = u @ rng.standard_normal((r, n))  # rank-r matrix, shape (n, n)

    omega = gaussian(n, r, p=5, seed=11)
    y = a @ omega

    q = orth(y, k=r)

    assert q.shape == (n, r)
    np.testing.assert_allclose(q.T @ q, np.eye(r), atol=1e-10)
    # range(Q) should equal range(A) == range(U).
    np.testing.assert_allclose(q @ (q.T @ u), u, atol=1e-8)


# ---------------------------------------------------------------------------
# two_sample_compress (Algorithm 2.1, Task 3.2)
# ---------------------------------------------------------------------------


def _synthetic_rank_k_block(
    m: int, n: int, r: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """A genuinely rank-`r` block `block = U_true @ S @ V_true.T` of shape
    `(m, n)`, built from thin orthonormal factors (per CLAUDE.md, not a
    full-rank random dense matrix).

    Returns `(block, U_true, V_true)`: `U_true` (shape `(m, r)`) spans
    `range(block)`, `V_true` (shape `(n, r)`) spans `range(block.T)`, both
    orthonormal.
    """
    rng = np.random.default_rng(seed)
    u_true = orth(rng.standard_normal((m, r)))
    v_true = orth(rng.standard_normal((n, r)))
    # Distinct positive singular values -> rank exactly r, no degeneracy.
    s = np.diag(np.linspace(1.0, 1.0 / r, r))
    block = u_true @ s @ v_true.T
    return block, u_true, v_true


def test_two_sample_compress_recovers_column_and_row_spaces_of_synthetic_block() -> None:
    """Block has shape (dof_row * |alpha|, dof_col * |beta|) = (8, 5), i.e.
    non-square per CLAUDE.md's row/col conventions; Y and Z accordingly have
    different numbers of rows."""
    m, n, r = 8, 5, 3
    block, u_true, v_true = _synthetic_rank_k_block(m, n, r, seed=0)

    p = 4  # oversampling
    omega = gaussian(n, r, p, seed=1)
    psi = gaussian(m, r, p, seed=2)

    y = block @ omega  # column sample, shape (m, r+p)
    z = block.T @ psi  # row sample, shape (n, r+p)

    u, v = two_sample_compress(y, z, k=r)

    assert u.shape == (m, r)
    assert v.shape == (n, r)

    # Orthonormal bases.
    np.testing.assert_allclose(u.T @ u, np.eye(r), atol=1e-10)
    np.testing.assert_allclose(v.T @ v, np.eye(r), atol=1e-10)

    # U spans the true column space: U U^T projects U_true onto itself.
    np.testing.assert_allclose(u @ (u.T @ u_true), u_true, atol=1e-8)
    # V spans the true row space: V V^T projects V_true onto itself.
    np.testing.assert_allclose(v @ (v.T @ v_true), v_true, atol=1e-8)

    # The block is exactly reconstructed from the recovered subspaces.
    reconstructed = (u @ u.T) @ block @ (v @ v.T)
    np.testing.assert_allclose(reconstructed, block, atol=1e-8)


def test_two_sample_compress_returns_orth_of_inputs() -> None:
    """two_sample_compress reuses orth(., k) verbatim -- no QR
    reimplementation."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal((10, 6))
    z = rng.standard_normal((7, 6))

    u, v = two_sample_compress(y, z, k=4)

    np.testing.assert_allclose(u, orth(y, k=4))
    np.testing.assert_allclose(v, orth(z, k=4))


def test_two_sample_compress_on_mockgf_admissible_block() -> None:
    """On a real admissible (low-rank) MockGF block, the recovered U/V
    subspaces capture the dominant column/row space: A_block - (UU^T) A_block
    (VV^T) is small relative to A_block."""
    n_side = 4
    gap = 100.0

    axes = [np.arange(n_side, dtype=float), np.arange(n_side, dtype=float)]
    grids = np.meshgrid(*axes, indexing="ij")
    cluster_a_centroids = np.stack([g.ravel() for g in grids], axis=1)
    cluster_b_centroids = cluster_a_centroids + np.array([gap, 0.0])

    centroids = np.concatenate([cluster_a_centroids, cluster_b_centroids], axis=0)
    lengths = np.full(centroids.shape[0], 0.1)
    mesh = FaultMesh(centroids=centroids, L=lengths)
    op = MockGF(mesh)

    n_cluster = n_side * n_side
    alpha = np.arange(n_cluster)
    beta = np.arange(n_cluster, 2 * n_cluster)

    block = op.block(alpha, beta)  # shape (dof_row * |alpha|, dof_col * |beta|)
    m, n = block.shape

    k, p = 3, 4
    omega = gaussian(n, k, p, seed=4)
    psi = gaussian(m, k, p, seed=5)

    y = block @ omega
    z = block.T @ psi

    u, v = two_sample_compress(y, z, k=k)

    assert u.shape == (m, k)
    assert v.shape == (n, k)
    np.testing.assert_allclose(u.T @ u, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-10)

    reconstructed = (u @ u.T) @ block @ (v @ v.T)
    rel_err = np.linalg.norm(reconstructed - block) / np.linalg.norm(block)
    assert rel_err < 1e-3


def test_two_sample_compress_rejects_k_out_of_range() -> None:
    rng = np.random.default_rng(6)
    y = rng.standard_normal((10, 4))
    z = rng.standard_normal((8, 4))

    with pytest.raises(ValueError):
        two_sample_compress(y, z, k=-1)
    with pytest.raises(ValueError):
        two_sample_compress(y, z, k=5)


# ---------------------------------------------------------------------------
# core_matrix_solve (Eq. 4.3, Task 3.3)
# ---------------------------------------------------------------------------


def test_core_matrix_solve_reconstructs_synthetic_rank_k_block() -> None:
    """Block has shape (dof_row * |alpha|, dof_col * |beta|) = (8, 5), a
    genuinely rank-r factorization (per CLAUDE.md, not a full-rank random
    dense matrix). U @ B @ V* should reconstruct it to tolerance."""
    m, n, r = 8, 5, 3
    block, _, _ = _synthetic_rank_k_block(m, n, r, seed=20)

    p = 4  # oversampling
    k = r
    g_beta = gaussian(n, k, p, seed=21)  # column-sample sketch, (n, k+p)
    g_alpha = gaussian(m, k, p, seed=22)  # row-sample sketch, (m, k+p)

    y = block @ g_beta  # column sample, shape (m, k+p)
    z = block.T @ g_alpha  # row sample, shape (n, k+p)

    u, v = two_sample_compress(y, z, k=k)

    b = core_matrix_solve(u, v, y_alpha=y, g_alpha=g_alpha, g_beta=g_beta)

    assert b.shape == (k, k)

    reconstructed = u @ b @ v.conj().T
    np.testing.assert_allclose(reconstructed, block, atol=1e-8)


def test_core_matrix_solve_on_mockgf_admissible_block() -> None:
    """On a real admissible (low-rank) MockGF block, U @ B @ V* reconstructs
    the block to a small relative error in Frobenius norm."""
    n_side = 4
    gap = 100.0

    axes = [np.arange(n_side, dtype=float), np.arange(n_side, dtype=float)]
    grids = np.meshgrid(*axes, indexing="ij")
    cluster_a_centroids = np.stack([g.ravel() for g in grids], axis=1)
    cluster_b_centroids = cluster_a_centroids + np.array([gap, 0.0])

    centroids = np.concatenate([cluster_a_centroids, cluster_b_centroids], axis=0)
    lengths = np.full(centroids.shape[0], 0.1)
    mesh = FaultMesh(centroids=centroids, L=lengths)
    op = MockGF(mesh)

    n_cluster = n_side * n_side
    alpha = np.arange(n_cluster)
    beta = np.arange(n_cluster, 2 * n_cluster)

    block = op.block(alpha, beta)  # shape (dof_row * |alpha|, dof_col * |beta|)
    m, n = block.shape

    k, p = 3, 6
    g_beta = gaussian(n, k, p, seed=23)
    g_alpha = gaussian(m, k, p, seed=24)

    y = block @ g_beta
    z = block.T @ g_alpha

    u, v = two_sample_compress(y, z, k=k)

    b = core_matrix_solve(u, v, y_alpha=y, g_alpha=g_alpha, g_beta=g_beta)

    assert b.shape == (k, k)

    reconstructed = u @ b @ v.conj().T
    rel_err = np.linalg.norm(reconstructed - block) / np.linalg.norm(block)
    assert rel_err < 1e-3


def test_core_matrix_solve_rejects_inconsistent_shapes() -> None:
    rng = np.random.default_rng(25)
    k, p = 3, 2
    m, n = 8, 5

    u = orth(rng.standard_normal((m, k + 2)))[:, :k]
    v = orth(rng.standard_normal((n, k + 2)))[:, :k]
    y_alpha = rng.standard_normal((m, k + p))
    g_alpha = rng.standard_normal((m, k + p))
    g_beta = rng.standard_normal((n, k + p))

    # u and v have mismatched number of columns.
    with pytest.raises(ValueError):
        core_matrix_solve(u, v[:, :-1], y_alpha, g_alpha, g_beta)

    # g_alpha has the wrong number of rows.
    with pytest.raises(ValueError):
        core_matrix_solve(u, v, y_alpha, g_alpha[:-1, :], g_beta)

    # g_beta has the wrong number of rows.
    with pytest.raises(ValueError):
        core_matrix_solve(u, v, y_alpha, g_alpha, g_beta[:-1, :])
