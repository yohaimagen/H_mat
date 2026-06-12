"""Tests for the relative-error utility via the power method (Task 2.3).

`DifferenceOperator` exposes `A_approx - A` via `matvec`/`rmatvec` only;
`spectral_norm_power_method` estimates `||op||_2` via ~20 power-method
iterations on `op* op`; `relative_error` is the ratio `||A_approx -
A||_2 / ||A||_2`. All three are checked against known small dense cases
(`numpy.linalg.svd`/`numpy.linalg.norm`), including rectangular shapes
(`2N x N`, `3N x 2N`) per CLAUDE.md's shape conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from gfcompress.error import (
    DifferenceOperator,
    relative_error,
    spectral_norm_power_method,
)
from gfcompress.operators import DenseOperator, MatVecOperator


def _spectral_norm_dense(a: np.ndarray) -> float:
    return float(np.linalg.svd(a, compute_uv=False)[0])


# ---------------------------------------------------------------------------
# DifferenceOperator
# ---------------------------------------------------------------------------


def test_difference_operator_matvec_matches_dense_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 3))
    b = rng.standard_normal((6, 3))
    diff = DifferenceOperator(DenseOperator(b), DenseOperator(a))

    omega = rng.standard_normal(3)
    np.testing.assert_allclose(diff.matvec(omega), (b - a) @ omega)


def test_difference_operator_rmatvec_matches_dense_difference() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 3))
    b = rng.standard_normal((6, 3))
    diff = DifferenceOperator(DenseOperator(b), DenseOperator(a))

    psi = rng.standard_normal(6)
    np.testing.assert_allclose(diff.rmatvec(psi), (b - a).T @ psi)


def test_difference_operator_matvec_on_thick_matrix() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((6, 3))
    b = rng.standard_normal((6, 3))
    diff = DifferenceOperator(DenseOperator(b), DenseOperator(a))

    omega = rng.standard_normal((3, 4))
    result = diff.matvec(omega)
    assert result.shape == (6, 4)
    np.testing.assert_allclose(result, (b - a) @ omega)


def test_difference_operator_is_matvec_operator() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 2))
    diff = DifferenceOperator(DenseOperator(a), DenseOperator(a))
    assert isinstance(diff, MatVecOperator)
    assert diff.shape == (4, 2)


def test_difference_operator_rejects_mismatched_shapes() -> None:
    rng = np.random.default_rng(4)
    a = rng.standard_normal((6, 3))
    b = rng.standard_normal((6, 4))
    with pytest.raises(ValueError):
        DifferenceOperator(DenseOperator(b), DenseOperator(a))


def test_difference_operator_rectangular_2n_by_n() -> None:
    """2D convention: A maps R^N -> R^{2N}."""
    n = 5
    rng = np.random.default_rng(5)
    a = rng.standard_normal((2 * n, n))
    b = rng.standard_normal((2 * n, n))
    diff = DifferenceOperator(DenseOperator(b), DenseOperator(a))
    assert diff.shape == (2 * n, n)

    omega = rng.standard_normal(n)
    np.testing.assert_allclose(diff.matvec(omega), (b - a) @ omega)

    psi = rng.standard_normal(2 * n)
    np.testing.assert_allclose(diff.rmatvec(psi), (b - a).T @ psi)


# ---------------------------------------------------------------------------
# spectral_norm_power_method
# ---------------------------------------------------------------------------


def test_spectral_norm_matches_svd_for_square_matrix() -> None:
    rng = np.random.default_rng(10)
    a = rng.standard_normal((20, 20))
    op = DenseOperator(a)

    estimate = spectral_norm_power_method(op, n_iters=50, seed=0)
    expected = _spectral_norm_dense(a)

    assert estimate == pytest.approx(expected, rel=1e-3)


def test_spectral_norm_matches_svd_for_rectangular_2n_by_n() -> None:
    """2D convention: A maps R^N -> R^{2N}."""
    n = 10
    rng = np.random.default_rng(11)
    a = rng.standard_normal((2 * n, n))
    op = DenseOperator(a)

    estimate = spectral_norm_power_method(op, n_iters=50, seed=1)
    expected = _spectral_norm_dense(a)

    assert estimate == pytest.approx(expected, rel=1e-3)


def test_spectral_norm_matches_svd_for_rectangular_3n_by_2n() -> None:
    """3D convention: A maps R^{2N} -> R^{3N}."""
    n = 8
    rng = np.random.default_rng(12)
    a = rng.standard_normal((3 * n, 2 * n))
    op = DenseOperator(a)

    estimate = spectral_norm_power_method(op, n_iters=50, seed=2)
    expected = _spectral_norm_dense(a)

    assert estimate == pytest.approx(expected, rel=1e-3)


def test_spectral_norm_of_diagonal_matrix_is_largest_singular_value() -> None:
    diag = np.array([1.0, 2.0, 3.0, 10.0])
    a = np.diag(diag)
    op = DenseOperator(a)

    estimate = spectral_norm_power_method(op, n_iters=20, seed=3)

    assert estimate == pytest.approx(10.0, rel=1e-6)


def test_spectral_norm_of_zero_operator_is_zero() -> None:
    a = np.zeros((4, 3))
    op = DenseOperator(a)

    estimate = spectral_norm_power_method(op, n_iters=20, seed=4)

    assert estimate == 0.0


def test_spectral_norm_rejects_non_positive_iters() -> None:
    a = np.eye(3)
    op = DenseOperator(a)
    with pytest.raises(ValueError):
        spectral_norm_power_method(op, n_iters=0)


# ---------------------------------------------------------------------------
# relative_error
# ---------------------------------------------------------------------------


def test_relative_error_is_zero_for_identical_operators() -> None:
    rng = np.random.default_rng(20)
    a = rng.standard_normal((10, 5))
    op_a = DenseOperator(a)
    op_b = DenseOperator(a.copy())

    err = relative_error(op_b, op_a, n_iters=20, seed=0)

    assert err == pytest.approx(0.0, abs=1e-10)


def test_relative_error_matches_dense_ratio_for_small_perturbation() -> None:
    rng = np.random.default_rng(21)
    n = 12
    a = rng.standard_normal((2 * n, n))
    perturbation = 1e-3 * rng.standard_normal((2 * n, n))
    b = a + perturbation

    op_a = DenseOperator(a)
    op_b = DenseOperator(b)

    err = relative_error(op_b, op_a, n_iters=50, seed=1)
    expected = _spectral_norm_dense(perturbation) / _spectral_norm_dense(a)

    assert err == pytest.approx(expected, rel=1e-2)


def test_relative_error_for_known_rank_one_difference() -> None:
    """A_approx = A + sigma * u v^T (rank-1 update): error = sigma /
    ||A||_2 when u, v are the leading singular vectors of A (worst case)."""
    a = np.diag(np.array([10.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    # Perturb the smallest singular direction so ||A|| is unaffected.
    delta = 0.5
    perturbation = np.zeros_like(a)
    perturbation[-1, -1] = delta
    b = a + perturbation

    op_a = DenseOperator(a)
    op_b = DenseOperator(b)

    err = relative_error(op_b, op_a, n_iters=50, seed=2)
    expected = delta / 10.0

    assert err == pytest.approx(expected, rel=1e-2)


def test_relative_error_is_large_for_unrelated_operators() -> None:
    rng = np.random.default_rng(22)
    n = 10
    a = rng.standard_normal((2 * n, n))
    b = 100.0 * rng.standard_normal((2 * n, n))

    op_a = DenseOperator(a)
    op_b = DenseOperator(b)

    err = relative_error(op_b, op_a, n_iters=50, seed=3)

    assert err > 10.0


def test_relative_error_rejects_mismatched_shapes() -> None:
    rng = np.random.default_rng(23)
    a = rng.standard_normal((6, 3))
    b = rng.standard_normal((6, 4))
    with pytest.raises(ValueError):
        relative_error(DenseOperator(b), DenseOperator(a))


def test_relative_error_with_zero_reference_and_nonzero_approx_is_inf() -> None:
    a = np.zeros((4, 3))
    b = np.eye(4, 3)
    err = relative_error(DenseOperator(b), DenseOperator(a), n_iters=10, seed=0)
    assert err == float("inf")


def test_relative_error_with_both_zero_is_zero() -> None:
    a = np.zeros((4, 3))
    b = np.zeros((4, 3))
    err = relative_error(DenseOperator(b), DenseOperator(a), n_iters=10, seed=0)
    assert err == 0.0


# ---------------------------------------------------------------------------
# Integration with MockGF (rectangular, genuine rank structure)
# ---------------------------------------------------------------------------


def test_relative_error_against_low_rank_approximation_of_mockgf() -> None:
    """A low-rank truncation of a MockGF block should have small but
    nonzero relative error against the full block, decreasing with rank."""
    from gfcompress.geometry import FaultMesh
    from gfcompress.mockgf import MockGF

    rng = np.random.default_rng(30)
    n = 16
    centroids = np.stack(
        [
            np.repeat(np.arange(4, dtype=float), 4),
            np.tile(np.arange(4, dtype=float), 4),
        ],
        axis=1,
    )
    centroids += 0.01 * rng.standard_normal(centroids.shape)
    lengths = np.full(n, 0.1)
    mesh = FaultMesh(centroids=centroids, L=lengths)
    gf = MockGF(mesh)

    a = gf.A
    u, s, vt = np.linalg.svd(a, full_matrices=False)

    k_small = 2
    k_large = min(8, s.shape[0])
    approx_small = (u[:, :k_small] * s[:k_small]) @ vt[:k_small, :]
    approx_large = (u[:, :k_large] * s[:k_large]) @ vt[:k_large, :]

    err_small = relative_error(DenseOperator(approx_small), gf, n_iters=30, seed=0)
    err_large = relative_error(DenseOperator(approx_large), gf, n_iters=30, seed=0)

    assert err_small > 0.0
    assert err_large < err_small
