"""Tests for `MatVecOperator` and `DenseOperator` (Task 2.1).

`DenseOperator` is a trivial dense-backed implementation used to check the
`MatVecOperator` contract: `matvec`/`rmatvec` agree with direct dense
matrix-vector and matrix-matrix products, on both vectors and "thick"
(stacked-column) inputs, and `shape` reports `(n_rows, n_cols)` -- which is
rectangular (`n_rows != n_cols`) for the `dof_row * N x dof_col * N` shapes
used elsewhere in this package (`2N x N` in 2D, `3N x 2N` in 3D).
"""

from __future__ import annotations

import numpy as np
import pytest

from gfcompress.operators import DenseOperator, MatVecOperator


def test_dense_operator_is_matvec_operator() -> None:
    a = np.arange(6.0).reshape(2, 3)
    op = DenseOperator(a)
    assert isinstance(op, MatVecOperator)


def test_dense_operator_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        DenseOperator(np.zeros(5))
    with pytest.raises(ValueError):
        DenseOperator(np.zeros((2, 3, 4)))


def test_dense_operator_shape_is_rectangular_2n_by_n() -> None:
    """2D convention: A maps R^N -> R^{2N} (dof_row=2, dof_col=1)."""
    n = 5
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2 * n, n))
    op = DenseOperator(a)
    assert op.shape == (2 * n, n)
    assert op.shape[0] != op.shape[1]


def test_dense_operator_shape_is_rectangular_3n_by_2n() -> None:
    """3D convention: A maps R^{2N} -> R^{3N} (dof_row=3, dof_col=2)."""
    n = 4
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3 * n, 2 * n))
    op = DenseOperator(a)
    assert op.shape == (3 * n, 2 * n)


def test_matvec_matches_dense_product_on_vector() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((6, 3))
    op = DenseOperator(a)

    omega = rng.standard_normal(3)
    result = op.matvec(omega)

    assert result.shape == (6,)
    np.testing.assert_allclose(result, a @ omega)


def test_matvec_matches_dense_product_on_thick_matrix() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 3))
    op = DenseOperator(a)

    omega = rng.standard_normal((3, 4))
    result = op.matvec(omega)

    assert result.shape == (6, 4)
    np.testing.assert_allclose(result, a @ omega)


def test_rmatvec_matches_dense_product_on_vector() -> None:
    rng = np.random.default_rng(4)
    a = rng.standard_normal((6, 3))
    op = DenseOperator(a)

    psi = rng.standard_normal(6)
    result = op.rmatvec(psi)

    assert result.shape == (3,)
    np.testing.assert_allclose(result, a.T @ psi)


def test_rmatvec_matches_dense_product_on_thick_matrix() -> None:
    rng = np.random.default_rng(5)
    a = rng.standard_normal((6, 3))
    op = DenseOperator(a)

    psi = rng.standard_normal((6, 4))
    result = op.rmatvec(psi)

    assert result.shape == (3, 4)
    np.testing.assert_allclose(result, a.T @ psi)


def test_matvec_and_rmatvec_act_on_different_spaces_when_rectangular() -> None:
    """A and A* must not be assumed interchangeable: domain/range differ."""
    n_rows, n_cols = 6, 3
    rng = np.random.default_rng(6)
    a = rng.standard_normal((n_rows, n_cols))
    op = DenseOperator(a)

    omega = rng.standard_normal(n_cols)
    assert op.matvec(omega).shape == (n_rows,)

    psi = rng.standard_normal(n_rows)
    assert op.rmatvec(psi).shape == (n_cols,)

    # matvec input shape must not match rmatvec output shape trivially
    # unless n_rows == n_cols -- here they differ.
    assert n_rows != n_cols


def test_rmatvec_is_adjoint_for_complex_dense_operator() -> None:
    """rmatvec applies the conjugate transpose A*."""
    rng = np.random.default_rng(7)
    real = rng.standard_normal((4, 3))
    imag = rng.standard_normal((4, 3))
    a = real + 1j * imag
    op = DenseOperator(a)

    psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    result = op.rmatvec(psi)

    np.testing.assert_allclose(result, a.conj().T @ psi)


def test_matvec_operator_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        MatVecOperator()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated() -> None:
    class Incomplete(MatVecOperator):
        def matvec(self, omega: np.ndarray) -> np.ndarray:
            return omega

        # rmatvec and shape intentionally not implemented

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
