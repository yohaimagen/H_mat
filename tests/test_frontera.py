"""Tiny fixtures for the native-dense Frontera reader."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from benchmarks.frontera import estimate_storage, stream_check

from gfcompress.build_tree import build_tree
from gfcompress.frontera import (
    PETSC_MAT_CLASSID,
    NativeDenseGF,
    read_native_coordinates,
    read_native_dense_header,
)
from gfcompress.interactions import build_lists


def _fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((60, 40))
    coordinates = rng.standard_normal((20, 3))
    matrix_path = tmp_path / "gf_mat.bin"
    with matrix_path.open("wb") as stream:
        stream.write(struct.pack(">6i", 8, 40, PETSC_MAT_CLASSID, 60, 40, -1))
        matrix.astype(">f8").tofile(stream)
    coordinate_path = tmp_path / "gf_coords.bin"
    coordinates.astype(">f8").tofile(coordinate_path)
    return matrix_path, coordinate_path, matrix, coordinates


def test_native_dense_header_and_coordinates(tmp_path: Path) -> None:
    matrix_path, coordinate_path, matrix, coordinates = _fixture(tmp_path)
    header = read_native_dense_header(matrix_path)
    assert (header.rows, header.cols) == matrix.shape
    assert header.predicted_size == matrix_path.stat().st_size
    loaded = read_native_coordinates(coordinate_path, 20)
    assert loaded.dtype.isnative
    np.testing.assert_array_equal(loaded, coordinates)


def test_native_dense_products_are_linear_and_adjoint(tmp_path: Path) -> None:
    matrix_path, coordinate_path, matrix, _ = _fixture(tmp_path)
    operator = NativeDenseGF(matrix_path, coordinate_path, chunk_rows=7)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((40, 2))
    y = rng.standard_normal((40, 2))
    u = rng.standard_normal((60, 2))
    v = rng.standard_normal((60, 2))

    np.testing.assert_allclose(operator.matvec(x), matrix @ x)
    np.testing.assert_allclose(operator.rmatvec(u), matrix.T @ u)
    np.testing.assert_allclose(operator.matvec(x + y), operator.matvec(x) + operator.matvec(y))
    np.testing.assert_allclose(operator.rmatvec(u + v), operator.rmatvec(u) + operator.rmatvec(v))
    assert np.vdot(operator.matvec(x[:, 0]), u[:, 0]) == pytest.approx(
        np.vdot(x[:, 0], operator.rmatvec(u[:, 0]))
    )
    assert operator.maximum_chunk_bytes == 2 * 7 * 40 * 8


def test_two_pass_stream_check_validates_both_linearities(tmp_path: Path) -> None:
    matrix_path, coordinate_path, _, _ = _fixture(tmp_path)
    result = stream_check(NativeDenseGF(matrix_path, coordinate_path, chunk_rows=7))
    assert result["full_passes"] == 2
    assert result["bytes_read"] == 2 * 60 * 40 * 8
    assert result["forward_linearity_relative"] < 1e-14
    assert result["adjoint_linearity_relative"] < 1e-14
    assert result["adjoint_identity_relative"] < 1e-14


def test_storage_prediction_matches_tree_pair_enumeration(tmp_path: Path) -> None:
    matrix_path, coordinate_path, _, _ = _fixture(tmp_path)
    operator = NativeDenseGF(matrix_path, coordinate_path)
    root = build_tree(operator.mesh, m=4)
    lists = build_lists(root)
    levels = list(root.iter_levels())
    expected_factors = sum(
        min(3, alpha.row_indices.size, beta.col_indices.size)
        * (alpha.row_indices.size + beta.col_indices.size)
        for nodes in levels[2:]
        for alpha in nodes
        for beta in lists.interaction[alpha]
    )
    expected_leaves = sum(
        alpha.row_indices.size * beta.col_indices.size
        for alpha in levels[-1]
        for beta in lists.nei[alpha]
    )
    result = estimate_storage(root, k=3)
    assert result["factor_entries_upper_bound"] == expected_factors
    assert result["leaf_entries"] == expected_leaves
    assert result["total_bytes_upper_bound"] == 8 * (expected_factors + expected_leaves)


def test_native_dense_vectors_and_diagnostic_block(tmp_path: Path) -> None:
    matrix_path, coordinate_path, matrix, _ = _fixture(tmp_path)
    operator = NativeDenseGF(matrix_path, coordinate_path, chunk_rows=3)
    rng = np.random.default_rng(10)
    x = rng.standard_normal(40)
    u = rng.standard_normal(60)
    np.testing.assert_allclose(operator.matvec(x), matrix @ x)
    np.testing.assert_allclose(operator.rmatvec(u), matrix.T @ u)
    rows = np.array([19, 2, 33, 2])
    cols = np.array([7, 1, 31])
    np.testing.assert_array_equal(operator.read_block(rows, cols), matrix[np.ix_(rows, cols)])


def test_native_dense_rejects_incomplete_or_mismatched_files(tmp_path: Path) -> None:
    matrix_path, coordinate_path, _, _ = _fixture(tmp_path)
    with matrix_path.open("r+b") as stream:
        stream.write(struct.pack(">2i", 8, 39))
    with pytest.raises(ValueError, match="incomplete"):
        read_native_dense_header(matrix_path)

    matrix_path, coordinate_path, _, _ = _fixture(tmp_path)
    coordinate_path.write_bytes(coordinate_path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="file size"):
        NativeDenseGF(matrix_path, coordinate_path)


def test_native_dense_rhs_shapes_are_checked(tmp_path: Path) -> None:
    matrix_path, coordinate_path, _, _ = _fixture(tmp_path)
    operator = NativeDenseGF(matrix_path, coordinate_path)
    with pytest.raises(ValueError, match="omega"):
        operator.matvec(np.ones(39))
    with pytest.raises(ValueError, match="psi"):
        operator.rmatvec(np.ones((60, 1, 1)))
