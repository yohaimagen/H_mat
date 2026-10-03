"""Focused C.8 accounting checks; deliberately small and serial."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gfcompress.benchmark import predict_fixed_products, representation_storage, synthetic_report
from gfcompress.build_tree import build_tree
from gfcompress.compress import CountingOperator, ProductCounts, compress
from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF


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
    predicted = predict_fixed_products(
        root, mesh, k=2, p=2, validation_matvec_columns=0, validation_rmatvec_columns=0
    )
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
    assert products["predicted"]["construction"] == products["construction_observed"]
    assert products["predicted"]["leaves"] == products["leaves_observed"]
    assert products["predicted"]["validation"] == products["validation_observed"]
    assert products["predicted"]["total"] == products["total_observed"]
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
