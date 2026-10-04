"""Focused C.8 accounting checks; deliberately small and serial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gfcompress.benchmark import (
    benchmark_operator,
    predict_fixed_products,
    representation_storage,
    synthetic_report,
)
from gfcompress.build_tree import build_tree
from gfcompress.compress import CountingOperator, ProductCounts, compress
from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF
from gfcompress.operators import MatVecOperator


def _mesh() -> FaultMesh:
    axis = np.arange(8, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return FaultMesh(np.column_stack((xx.ravel(), yy.ravel())), np.ones(64))


def test_count_snapshot_and_reset_are_phase_boundaries() -> None:
    mesh = _mesh()
    counted = CountingOperator(MockGF(mesh))
    counted.matvec(np.ones((mesh.n_cols, 2)))
    before = counted.reset()
    assert before == ProductCounts(1, 2, 0, 0)
    assert counted.snapshot() == ProductCounts(0, 0, 0, 0)
    counted.rmatvec(np.ones(mesh.n_rows))
    assert counted.snapshot() == ProductCounts(0, 0, 1, 1)


def test_fixed_prediction_matches_observed_columns_and_retained_storage(tmp_path: Path) -> None:
    mesh = _mesh()
    root = build_tree(mesh, m=2)
    predicted = predict_fixed_products(root, mesh, k=2, p=2, validation=ProductCounts(0, 0, 0, 0))
    counted = CountingOperator(MockGF(mesh))
    hmat = compress(counted, mesh, m=2, k=2, p=2, seed=0)
    observed = counted.snapshot()
    assert observed == ProductCounts(
        predicted.construction.matvec_calls + predicted.leaves.matvec_calls,
        predicted.construction.matvec_columns + predicted.leaves.matvec_columns,
        predicted.construction.rmatvec_calls + predicted.leaves.rmatvec_calls,
        predicted.construction.rmatvec_columns + predicted.leaves.rmatvec_columns,
    )
    cache = tmp_path / "cache.bin"
    cache.write_bytes(b"cache")
    storage = representation_storage(
        hmat, cache_path=cache, dense_reference_bytes=counted.inner.A.nbytes
    )
    assert storage.factor_bytes + storage.leaf_bytes == storage.numerical_bytes
    assert storage.retained_allocation_bytes >= storage.numerical_bytes + storage.tree_index_bytes
    assert storage.tree_index_bytes > 0
    assert storage.disk_cache_bytes == 5
    assert storage.dense_reference_bytes == counted.inner.A.nbytes
    cache_dir = tmp_path / "cache-dir"
    cache_dir.mkdir()
    (cache_dir / "nested").mkdir()
    (cache_dir / "nested" / "part.bin").write_bytes(b"nested")
    assert representation_storage(hmat, cache_path=cache_dir).disk_cache_bytes == 6


def test_synthetic_schema_separates_nonzero_validation_and_timing_phases() -> None:
    report = synthetic_report(n_side=4, m=1, k=1, p=1, repeats=1)
    products = report["products"]
    assert products["budget"]["construction"] == products["construction_observed"]
    assert products["budget"]["leaves"] == products["leaves_observed"]
    for direction in ("matvec_calls", "matvec_columns", "rmatvec_calls", "rmatvec_columns"):
        assert (
            products["validation_observed"][direction]
            <= products["budget"]["validation_budget"][direction]
        )
    assert products["total_observed"] == {
        key: products["construction_total_observed"][key] + products["validation_observed"][key]
        for key in products["total_observed"]
    }
    timing = report["timing_seconds"]
    assert all(
        timing[name] >= 0
        for name in (
            "loading_conversion",
            "geometry",
            "sampling_peeling_exclusive",
            "local_factorizations",
            "total_setup",
        )
    )
    assert timing["total_setup"] >= timing["loading_conversion"] + timing["geometry"]


def test_shared_benchmark_measures_finalized_production_factors() -> None:
    mesh = _mesh()
    report = benchmark_operator(
        MockGF(mesh), mesh, dataset={"identity": "finalized"}, m=2, k=2, p=2, repeats=1
    )
    hmat = compress(MockGF(mesh), mesh, m=2, k=2, p=2, seed=0)
    assert all(factor.b is None for factor in hmat.factors)
    assert report["storage"]["factor_bytes"] == representation_storage(hmat).factor_bytes


def test_exact_recovery_uses_less_than_validation_budget() -> None:
    mesh = _mesh()
    report = benchmark_operator(
        MockGF(mesh), mesh, dataset={"identity": "exact leaf"}, m=128, k=1, p=1, repeats=1
    )
    products = report["products"]
    assert report["products"]["relative_error"] == pytest.approx(0.0)
    assert (
        products["validation_observed"]["matvec_calls"]
        < products["budget"]["validation_budget"]["matvec_calls"]
    )
    assert products["total_observed"] == {
        key: products["construction_total_observed"][key] + products["validation_observed"][key]
        for key in products["total_observed"]
    }


def test_subset_ready_report_runs_shared_measurement_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import subset as subset_cli

    mesh = _mesh()
    source = MockGF(mesh)

    class TinySubset(MatVecOperator):
        def __init__(self) -> None:
            self.mesh = mesh
            self.patch_ids = np.arange(mesh.n_patches, dtype=np.intp)
            self.row_map = np.arange(mesh.n_rows, dtype=np.intp)
            self.col_map = np.arange(mesh.n_cols, dtype=np.intp)

        @property
        def shape(self) -> tuple[int, int]:
            return source.shape

        def matvec(self, omega: np.ndarray) -> np.ndarray:
            return source.matvec(omega)

        def rmatvec(self, psi: np.ndarray) -> np.ndarray:
            return source.rmatvec(psi)

    data_dir = tmp_path / "gf_bp3"
    data_dir.mkdir()
    (data_dir / "gf_mat.bin").write_bytes(b"present")
    (data_dir / "bp3_fbf_coords.csv").write_text("present", encoding="utf-8")
    monkeypatch.setattr(subset_cli, "RealGF", lambda *_: object())
    monkeypatch.setattr(subset_cli, "representative_subset", lambda *_args, **_kwargs: TinySubset())
    report = subset_cli.report(tmp_path, "bp3", 4, 64)
    assert report["status"] == "ready"
    assert report["parameters"] == {
        "m": 12,
        "k": 4,
        "p": 2,
        "sampling": "fixed",
        "regions": 4,
        "max_patches": 64,
    }
    assert report["products"]["validation_observed"]["matvec_calls"] > 0
    assert report["storage"]["dense_reference_bytes"] == mesh.n_rows * mesh.n_cols * 8


def test_subset_rss_finalization_uses_comparable_peaks() -> None:
    from benchmarks.subset import with_rss

    report = with_rss({"storage": {}}, 300, 200)
    assert report["storage"] == {
        "compressed_peak_rss_bytes": 300,
        "dense_peak_rss_bytes": 200,
        "rss_ratio": 1.5,
    }
