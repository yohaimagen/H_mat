"""C.16 public selection, accounting, benchmark, and full-path regressions."""

import json

import numpy as np
import pytest
from benchmarks.coloring import paired_comparison
from benchmarks.coloring import report as subset_report

from gfcompress.benchmark import benchmark_operator
from gfcompress.compress import CountingOperator, compress
from gfcompress.error import relative_error
from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF
from gfcompress.operators import DenseOperator


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("seed", [11, 23, 37])
def test_public_full_path_quality_counts_permutation_and_adjoint(dim: int, seed: int) -> None:
    points = np.random.default_rng(0).random((80, dim))
    mesh = FaultMesh(points, np.full(80, 0.1))
    op = MockGF(mesh, eps=0.02)
    k = 4 if dim == 2 else 6  # Retains C.15's visible rank-4 negative case.
    errors = []
    for strategy in ("fixed", "coloring"):
        counted = CountingOperator(op)
        hmat = compress(counted, mesh, m=4, k=k, p=10, seed=seed, sampling=strategy)
        report = hmat.sampling_report
        assert report["requested"] == strategy
        actual = counted.snapshot()
        for name in ("matvec_calls", "matvec_columns", "rmatvec_calls", "rmatvec_columns"):
            assert getattr(actual, name) == sum(p["predicted"][name] for p in report["passes"])
        for record in report["passes"]:
            assert record["predicted"] == record["observed"]
            assert record["selected_columns"] <= record["fixed_columns"]
            assert record["graph_build_seconds"] >= 0 and record["dsatur_seconds"] >= 0
            if strategy == "fixed":
                assert record["vertices"] is None
        if strategy == "coloring":
            assert any(p["selected_columns"] < p["fixed_columns"] for p in report["passes"])
            assert all(p["vertices"] is not None for p in report["passes"])
        partition = hmat.block_partition()
        assert len(set().union(*partition)) == sum(map(len, partition)) == 80**2
        rng = np.random.default_rng(101)
        x, y = rng.normal(size=mesh.n_cols), rng.normal(size=mesh.n_rows)
        np.testing.assert_allclose(y @ hmat.matvec(x), x @ hmat.rmatvec(y), rtol=1e-12)
        errors.append(relative_error(hmat, op, seed=101))
        for leaf in hmat.leaves:
            expected = op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices)
            assert np.linalg.norm(leaf.block - expected) / np.linalg.norm(expected) < 0.05
        permutation = rng.permutation(80)
        rows, cols = mesh.patch_to_rows(permutation), mesh.patch_to_cols(permutation)
        permuted_mesh = FaultMesh(points[permutation], mesh.L[permutation])
        permuted_op = DenseOperator(op.A[np.ix_(rows, cols)])
        permuted = compress(
            permuted_op, permuted_mesh, m=4, k=k, p=10, seed=seed, sampling=strategy
        )
        assert relative_error(permuted, permuted_op, seed=103) < 5e-4
    assert max(errors) < 5e-4
    assert errors[1] < 3 * errors[0]


def test_default_fixed_and_invalid_before_work() -> None:
    mesh = FaultMesh(np.random.default_rng(0).random((20, 2)), np.full(20, 0.1))
    op = MockGF(mesh)
    explicit = compress(op, mesh, 4, 3, seed=11, sampling="fixed")
    default = compress(op, mesh, 4, 3, seed=11)
    x = np.random.default_rng(4).normal(size=mesh.n_cols)
    np.testing.assert_array_equal(explicit.matvec(x), default.matvec(x))
    counted = CountingOperator(op)
    with pytest.raises(ValueError, match="sampling"):
        compress(counted, mesh, 4, 3, sampling="invalid")
    assert counted.matvec_calls == counted.rmatvec_calls == 0


def test_benchmark_uses_public_pipeline() -> None:
    mesh = FaultMesh(np.random.default_rng(0).random((30, 2)), np.full(30, 0.1))
    op = MockGF(mesh)
    report = benchmark_operator(
        op, mesh, dataset={"identity": "test"}, m=4, k=3, seed=11, sampling="coloring"
    )
    hmat = compress(op, mesh, m=4, k=3, p=2, seed=11, sampling="coloring")
    assert report["parameters"]["sampling"] == "coloring"
    assert [p["predicted"] for p in report["sampling_report"]["passes"]] == [
        p["predicted"] for p in hmat.sampling_report["passes"]
    ]
    for name in ("matvec_calls", "matvec_columns", "rmatvec_calls", "rmatvec_columns"):
        observed = report["products"]["construction_total_observed"][name]
        predicted = (
            report["products"]["budget"]["construction"][name]
            + report["products"]["budget"]["leaves"][name]
        )
        assert observed == predicted


def test_predeclared_paired_comparison_is_falsifiable() -> None:
    def diagnostic(error, global_error):
        return {
            "global_relative_error": global_error,
            "selected_block_output_errors": {},
            **{
                side: {
                    "component_index_group_errors": {"0": {"absolute": error}},
                    "input_errors": {},
                }
                for side in ("forward", "adjoint")
            },
        }

    fixed = diagnostic(0.1, 1e-4)
    json.dumps(paired_comparison(fixed, diagnostic(1.09, 1.09e-4), 1.0, 0.1))
    assert paired_comparison(fixed, diagnostic(1.09, 1.09e-4), 1.0, 0.1)["all_pass"]
    assert not paired_comparison(fixed, diagnostic(1.11, 1.09e-4), 1.0, 0.1)["all_pass"]
    assert not paired_comparison(fixed, diagnostic(0.1, 1.11e-4), 1.0, 0.1)["all_pass"]


def test_subset_runner_is_opt_in_and_records_sources(tmp_path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("must not load a real matrix without --run")

    monkeypatch.setattr("benchmarks.coloring.RealGF", forbidden)
    blocked = subset_report(tmp_path, "bp3", "test-base", False)
    assert blocked["status"] == "blocked"
    directory = tmp_path / "gf_bp3"
    directory.mkdir()
    (directory / "gf_mat.bin").touch()
    (directory / "bp3_fbf_coords.csv").touch()
    skipped = subset_report(tmp_path, "bp3", "test-base", False)
    assert skipped["status"] == "not_run"
    assert skipped["revision"] == "test-base"
    assert "src/gfcompress/compress.py" in skipped["source_sha256"]
    assert skipped["config"]["construction_seeds"] == [11, 23, 37]
    assert skipped["config"]["validation_seeds"] == [101, 103]
    json.dumps(skipped)
