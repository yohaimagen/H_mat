"""Full-run accounting contracts on a small smooth operator."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.diagnose import block_diagnostics, diagnostic_provenance
from benchmarks.full import main, measure_seed, prepare
from benchmarks.screen import select_promotions

from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF
from gfcompress.randomized import orth


def test_full_accounting_keeps_validation_separate_and_failed_gates_visible() -> None:
    config = json.loads(Path("benchmarks/configs/bp3_fixed.json").read_text())
    config["selection"]["primary"] = {"m": 2, "k": 1, "p": 1}
    config["validation"]["power_iterations"] = 1
    mesh = FaultMesh(np.column_stack((np.arange(32), np.zeros(32))), np.ones(32))
    result = measure_seed(MockGF(mesh), mesh, config, 11, 2.0)
    products = result["products"]
    assert products["predicted_total"] == products["observed_total"]
    assert products["validation_excluded"]["matvec_columns"] > 0
    assert [r["validation_seed"] for r in result["records"]] == [101, 103]
    assert result["all_pass"] == all(result["gates"].values())
    assert result["gates"]["frozen_accuracy"] is False
    timing = result["timing_seconds"]
    assert timing["total_setup"] == timing["construction"] + 2.0
    for side in ("forward", "adjoint"):
        values = timing[side]
        if values["dense"] <= values["compressed"]:
            assert values["setup_break_even_applies"] is None
    assert result["storage"]["numerical_entries"] * 8 == result["storage"]["numerical_bytes"]


def test_native_preparation_preserves_file_order() -> None:
    original = np.arange(780).reshape(260, 3).astype(">f8")
    source = SimpleNamespace(mat=original, shape=original.shape)
    prepare(source)
    assert source.mat.dtype.isnative
    np.testing.assert_array_equal(source.mat, original)


def test_cli_is_opt_in_and_requires_single_thread_caps(tmp_path, monkeypatch) -> None:
    output = tmp_path / "report.json"
    argv = ["full", "--dataset", "bp3", "--revision", "test", "--output", str(output)]
    monkeypatch.setattr("sys.argv", argv)
    main()
    report = json.loads(output.read_text())
    assert report["status"] == "not_run"
    assert report["runs"] == []
    monkeypatch.setattr("sys.argv", [*argv, "--run"])
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with pytest.raises(ValueError, match="OPENBLAS_NUM_THREADS=1"):
        main()


def test_block_diagnostics_distinguishes_sample_contamination() -> None:
    mesh = FaultMesh(np.column_stack((np.arange(64), np.zeros(64))), np.ones(64))
    a = MockGF(mesh).matvec(np.eye(mesh.n_cols))[:32, 48:64]
    rng = np.random.default_rng(11)
    ga, gb = rng.normal(size=(32, 7)), rng.normal(size=(16, 7))
    y = a @ gb
    u, v = orth(y, 3), orth(a.T @ ga, 3)
    clean = block_diagnostics(a, u, v, y, ga, gb, 2)
    assert clean["column_sample_contamination"] == 0
    assert clean["errors"]["optimal_rank_k"]["all"] <= clean["errors"]["reconstructed"]["all"]
    dirty = block_diagnostics(
        a, u, v, y + 0.1 * np.linalg.norm(y) * rng.normal(size=y.shape), ga, gb, 2
    )
    assert dirty["column_sample_contamination"] > 0
    assert dirty["errors"]["reconstructed"]["all"] > clean["errors"]["reconstructed"]["all"]
    assert dirty["errors"]["exact_sample_same_bases"] == clean["errors"]["exact_sample_same_bases"]


def test_cli_rejects_over_budget_before_preparing_operator(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "gf_bp3"
    directory.mkdir()
    (directory / "gf_mat.bin").touch()
    (directory / "test_fbf_coords.csv").touch()
    mesh = FaultMesh(np.array([[0.0, 0.0], [1.0, 0.0]]), np.ones(2))
    monkeypatch.setattr(
        "benchmarks.full.RealGF", lambda *args: SimpleNamespace(mesh=mesh, shape=(4, 2))
    )
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        "sys.argv",
        ["full", "--dataset", "bp3", "--revision", "test", "--data-root", str(tmp_path), "--run"],
    )
    with pytest.raises(ValueError, match="strictly below n_cols"):
        main()


def test_diagnostic_provenance_tracks_revision_coordinates_and_helpers(tmp_path, monkeypatch):
    import hashlib

    from benchmarks.diagnose import main as diagnose_main

    coordinate = tmp_path / "coords.csv"
    coordinate.write_text("coordinate fixture")
    config = Path("benchmarks/configs/bp7_full_fixed.json")
    result = diagnostic_provenance("explicit-revision", config, coordinate, coordinate)
    assert result["provenance"]["revision_supplied_by_orchestrator"] == "explicit-revision"
    assert result["source"]["coordinates"]["size"] == coordinate.stat().st_size
    for helper in ["benchmarks/c10.py", "benchmarks/full.py"]:
        assert (
            result["provenance"]["source_sha256"][helper]
            == hashlib.sha256(Path(helper).read_bytes()).hexdigest()
        )
    monkeypatch.setattr("sys.argv", ["diagnose", "--run"])
    with pytest.raises(SystemExit):
        diagnose_main()


def test_screen_promotion_rule_is_bounded_and_requires_both_metrics():
    policy = {"component_worst_ratio_max": 0.5, "all_worst_ratio_max": 1.0, "maximum_promotions": 2}
    results = [
        {"parameters": {"k": k}, "component_worst": comp, "all_worst": whole}
        for k, comp, whole in [
            (47, 2.0, 1.0),
            (41, 0.9, 0.9),
            (44, 0.8, 1.1),
            (49, 1.1, 0.8),
            (53, 0.7, 1.0),
            (55, 0.95, 0.8),
        ]
    ]
    assert select_promotions(results, policy) == [{"k": 53}, {"k": 41}]
