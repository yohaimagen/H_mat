"""Tiny fixtures for the native-dense Frontera reader."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.frontera import (
    _environment,
    _known_live_product_workspace_bytes,
    _provenance,
    _rank_for_relative_tail,
    _scalar_indices,
    construction_prediction,
    estimate_storage,
    full_compression_gate,
    ordering_evidence,
    stream_check,
)

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


def _element_blocked_fixture(tmp_path: Path) -> NativeDenseGF:
    n, nbf = 20, 10
    coordinates = np.column_stack((np.r_[np.arange(10), np.arange(20, 30)], np.zeros((20, 2))))
    distance = np.abs(coordinates[:, 0, None] - coordinates[:, 0])
    strength = 1.0 / (1.0 + distance) ** 2
    component = np.arange(1, 7, dtype=float).reshape(3, 2)
    physical = np.einsum("ij,ab->iajb", strength, component).reshape(3 * n, 2 * n)
    row_map = _scalar_indices(np.arange(n), 3, n, nbf, "element_blocked")
    col_map = _scalar_indices(np.arange(n), 2, n, nbf, "element_blocked")
    raw = np.empty_like(physical)
    raw[np.ix_(row_map, col_map)] = physical
    matrix_path = tmp_path / "blocked_mat.bin"
    with matrix_path.open("wb") as stream:
        stream.write(struct.pack(">6i", 2, 40, PETSC_MAT_CLASSID, 60, 40, -1))
        raw.astype(">f8").tofile(stream)
    coordinate_path = tmp_path / "blocked_coords.bin"
    coordinates.astype(">f8").tofile(coordinate_path)
    return NativeDenseGF(matrix_path, coordinate_path, chunk_rows=7)


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
    assert operator.ordering_status == "unresolved"
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
    assert operator.maximum_chunk_bytes == 3 * 7 * 40 * 8


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


def test_ordering_compares_element_blocked_but_remains_unresolved(tmp_path: Path) -> None:
    evidence = ordering_evidence(_element_blocked_fixture(tmp_path))
    blocked = "element_blocked_rows__element_blocked_cols"
    patch = "patch_major_rows__patch_major_cols"
    assert (
        evidence["hypotheses"][blocked]["median_spearman"]
        < evidence["hypotheses"][patch]["median_spearman"]
    )
    assert evidence["strongest_decay_hypothesis"].endswith("__element_blocked_cols")
    assert evidence["status"] == "unresolved"
    assert evidence["passes_gate"] is False


def test_rank_boundary_is_inclusive_and_uses_next_singular_value() -> None:
    relative = np.array([1.0, 0.1, 0.01])
    assert _rank_for_relative_tail(relative, 0.1) == 1
    assert _rank_for_relative_tail(relative, 0.099) == 2
    assert _rank_for_relative_tail(relative, 0.001) is None


def test_prediction_names_runtime_upper_and_models_three_outputs(tmp_path: Path) -> None:
    matrix_path, coordinate_path, _, _ = _fixture(tmp_path)
    operator = NativeDenseGF(matrix_path, coordinate_path, chunk_rows=7)
    stream = {"forward_seconds": 2.0, "adjoint_seconds": 3.0, "rhs_columns_per_pass": 3}
    prediction = construction_prediction(operator, build_tree(operator.mesh, m=4), stream)
    assert "deliberately_pessimistic_linear_rhs_upper_seconds" in prediction
    assert "linear_rhs_seconds_from_width3_products" not in prediction
    peak = prediction["peak_memory_model"]
    assert (
        peak["modeled_known_live_array_bytes"] >= prediction["storage"]["total_bytes_upper_bound"]
    )
    expected = 8 * 5 * (40 + 3 * 60) + operator.maximum_chunk_bytes
    assert (
        _known_live_product_workspace_bytes(
            operator.shape, 5, operator.maximum_chunk_bytes, transpose=False
        )
        == expected
    )


def test_gate_and_provenance_make_unresolved_state_explicit(monkeypatch) -> None:
    gate = full_compression_gate({"status": "unresolved"}, {"bytes_read": 2_000_000_000_000})
    assert gate["status"] == "gated_not_run"
    assert "physical row/column ordering is unresolved" in gate["reasons"]
    assert any("1000000000000" in reason for reason in gate["reasons"])

    args = SimpleNamespace(
        revision="review-correction",
        dataset=Path("dataset"),
        output=Path("output.json"),
        chunk_rows=64,
        run=True,
    )
    monkeypatch.setattr("sys.argv", ["runner", "--run"])
    provenance = _provenance(args)
    assert provenance["revision_label"] == "review-correction"
    assert provenance["command"][-1] == "--run"
    assert "src/gfcompress/frontera.py" in provenance["source_sha256"]
    assert "Git commands" in provenance["dirty_state"]
    assert _environment()["scipy"]


def test_tracked_correction_record_preserves_gate_and_timing_provenance() -> None:
    report = json.loads(Path("benchmarks/results/frontera_np20k_preflight.json").read_text())
    assert report["ordering"]["status"] == "unresolved"
    assert report["conditional_block_preflight"]["used_for_gate"] is False
    assert report["full_compression"]["status"] == "gated_not_run"
    assert report["derived_correction"]["streaming_timings_reused_without_full_matrix_rescan"]
    assert (
        report["derived_correction"]["original_measured_record_sha256"]
        == "89696e68ed460d776fd302ad766d1bf0ff879aee0ac8674a3dcc866d1a5f32d8"
    )
    assert "deliberately_pessimistic_linear_rhs_upper_seconds" in report["prediction"]
    peak = report["prediction"]["peak_memory_model"]
    assert report["streaming"]["maximum_converted_chunk_bytes"] == 61_870_080
    assert peak["maximum_forward_workspace_bytes"] == 1_323_761_920
    assert peak["maximum_adjoint_workspace_bytes"] == 144_524_640
    assert peak["modeled_known_live_array_bytes"] == 4_509_852_944
    assert "full, truncated, and subtraction-result" in peak["included_lifetimes"]
    assert report["derived_correction"]["final_chunk_lifetime_adjustment"] == (
        "deterministic three-chunk lifetime correction; no dataset scan or new " "measurement"
    )
    assert report["provenance"]["command"][-1] == "--run"
    source = Path("src/gfcompress/frontera.py")
    assert (
        report["provenance"]["source_sha256"][str(source)]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


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
