"""Tests for Gaussian sampling and orthonormalization helpers (Task 3.1).

`gaussian(n, k, p, seed)` draws an `n x (k+p)` standard-normal matrix
(seedable via `numpy.random.default_rng`); `orth(Y)` and `orth(Y, k)` wrap
`scipy.linalg.qr` to return an orthonormal basis `Q` for `range(Y)` (thin /
rank-`k`-truncated).
"""

from __future__ import annotations

import numpy as np
import pytest

from gfcompress.randomized import gaussian, orth

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
