"""Focused C.9 checks for finalized far-field factors."""

from __future__ import annotations

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.compress import compress, compress_level
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import build_lists
from gfcompress.leaf import extract_leaves
from gfcompress.mockgf import MockGF
from gfcompress.peeling import factor_economy, finalize_factors


def _mesh() -> FaultMesh:
    axis = np.arange(8, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return FaultMesh(np.column_stack((xx.ravel(), yy.ravel())), np.ones(64))


def _raw_hmatrix(mesh: FaultMesh) -> HMatrix:
    operator = MockGF(mesh)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    factors = []
    for level in range(2, 4):
        factors += compress_level(operator, root, lists, mesh, level, factors, k=2, p=2, seed=0)
    leaves = extract_leaves(operator, root, lists, mesh, 3, factors)
    return HMatrix(root=root, mesh=mesh, factors=factors, leaves=leaves)


def test_finalization_preserves_products_and_adjoint_and_removes_cores() -> None:
    mesh = _mesh()
    raw = _raw_hmatrix(mesh)
    finalized = HMatrix(
        root=raw.root, mesh=mesh, factors=finalize_factors(raw.factors), leaves=raw.leaves
    )
    rng = np.random.default_rng(5)
    x = rng.standard_normal((mesh.n_cols, 3))
    y = rng.standard_normal((mesh.n_rows, 3))
    np.testing.assert_allclose(finalized.dot(x), raw.dot(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(finalized.rdot(y), raw.rdot(y), rtol=1e-12, atol=1e-12)
    assert all(factor.b is None for factor in finalized.factors)
    economy = factor_economy(raw.factors)
    assert economy.before_entries - economy.finalized_entries == sum(
        factor.u.shape[1] ** 2 for factor in raw.factors
    )
    assert economy.uneconomical_blocks > 0


def test_public_compress_finalizes_only_after_leaf_extraction() -> None:
    mesh = _mesh()
    hmat = compress(MockGF(mesh), mesh, m=2, k=2, p=2, seed=0)
    assert hmat.factors
    assert all(factor.b is None for factor in hmat.factors)


def test_c9_report_records_measurement_and_all_deferrals() -> None:
    from gfcompress.benchmark import c9_synthetic_report

    report = c9_synthetic_report(n_side=4, m=1, k=1, p=1, repeats=1)
    far_field = report["measurements"]["far_field"]
    assert far_field["blocks"] > 0
    assert 0 <= far_field["uneconomical_fraction"] <= 1
    assert far_field["before_entries"] > far_field["after_core_absorption_entries"]
    assert report["measurements"]["equivalence"]["forward_absolute_difference"] < 1e-12
    assert report["measurements"]["equivalence"]["adjoint_absolute_difference"] < 1e-12
    assert "deferred" in report["decisions"]["dense_far_field_fallback"]
    assert "not adopted" in report["decisions"]["tree_order_permutation"]
    assert "not adopted" in report["decisions"]["native_endian_cache"]
