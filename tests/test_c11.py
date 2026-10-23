"""Full-run accounting contracts on a small smooth operator."""

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import benchmarks.screen as screen
import numpy as np
import pytest
from benchmarks.diagnose import block_diagnostics, diagnostic_provenance
from benchmarks.full import main, measure_seed, prepare
from benchmarks.remediate import (
    ComponentScaledOperator,
    component_scales_from_blocks,
    compose_raw_maps,
    inverse_scaled,
    screen_block,
)
from benchmarks.screen import select_promotions

from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF
from gfcompress.operators import DenseOperator
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
    policy = {"component_worst_ratio_max": 0.5, "all_worst_ratio_max": 1.0, "maximum_promotions": 1}
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
    assert select_promotions(results, policy) == [{"k": 53}]


def test_screen_plan_declares_three_maximum_width_alternative_m_values():
    plan = json.loads(Path("benchmarks/configs/bp7_screen.json").read_text())
    assert len(plan["candidates"]) == 3
    assert all(candidate["m"] != plan["incumbent"]["m"] for candidate in plan["candidates"])
    assert all(
        candidate["k"] + candidate["p"] == candidate["width"] for candidate in plan["candidates"]
    )
    assert plan["promotion"]["maximum_promotions"] == 1


def test_screen_rejects_a_nonmaximum_declared_width(monkeypatch):
    monkeypatch.setattr(
        screen,
        "construction_budget",
        lambda _root, _mesh, width: (width, {6: 9, 7: 10, 8: 11}[width]),
    )
    assert screen.assert_maximum_feasible_width(None, None, 7, 11) == 7
    with pytest.raises(ValueError, match="maximum strictly feasible"):
        screen.assert_maximum_feasible_width(None, None, 6, 11)


def test_component_scaled_operator_preserves_rectangular_black_box_products():
    matrix = np.arange(24, dtype=float).reshape(6, 4) + 1
    operator = DenseOperator(matrix)
    dr = np.array([2.0, 3.0, 2.0, 3.0, 2.0, 3.0])
    dc = np.array([5.0, 7.0, 5.0, 7.0])
    scaled = ComponentScaledOperator(operator, dr, dc)
    omega = np.arange(8, dtype=float).reshape(4, 2) + 1
    psi = np.arange(12, dtype=float).reshape(6, 2) + 1
    np.testing.assert_allclose(scaled.matvec(omega), (dr[:, None] * matrix / dc) @ omega)
    np.testing.assert_allclose(scaled.rmatvec(psi), (dr[:, None] * matrix / dc).T @ psi)
    np.testing.assert_allclose(inverse_scaled(dr[:, None] * matrix / dc, dr, dc), matrix)


def test_remediation_screen_reports_all_rectangular_component_errors():
    rng = np.random.default_rng(4)
    block = rng.normal(size=(12, 8))
    result = screen_block(
        block,
        rank=3,
        width=9,
        seed=11,
        dr=np.ones(12),
        dc=np.ones(8),
        dof_row=3,
        dof_col=2,
    )
    assert result["rank"] == 3
    assert result["width"] == 9
    for error in (result["projection_errors"], result["reconstruction_errors"]):
        assert set(error["output"]) == {"0", "1", "2"}
        assert set(error["input"]) == {"0", "1"}
        assert error["worst_component"] >= error["all"]


def test_subset_maps_compose_patch_major_indices_with_source_raw_permutations():
    source = SimpleNamespace(
        row_pm_to_raw=np.array([4, 0, 5, 1, 6, 2]),
        col_pm_to_raw=np.array([3, 0, 4, 1]),
    )
    rows, cols = compose_raw_maps(source, np.array([5, 0, 2]), np.array([3, 0]))
    np.testing.assert_array_equal(rows, [2, 4, 5])
    np.testing.assert_array_equal(cols, [1, 3])


def test_oracle_input_scale_uses_output_scaled_blocks():
    block = np.array([[1.0, 8.0], [2.0, 4.0], [3.0, 2.0], [4.0, 1.0]])
    dr, dc = component_scales_from_blocks([block], dof_row=2, dof_col=2)
    row_energy = np.array([np.linalg.norm(block[0::2]) ** 2, np.linalg.norm(block[1::2]) ** 2])
    expected_dr = np.exp(np.mean(np.log(np.sqrt(row_energy)))) / np.sqrt(row_energy)
    scaled = np.tile(expected_dr, 2)[:, None] * block
    col_energy = np.array([np.linalg.norm(scaled[:, 0]) ** 2, np.linalg.norm(scaled[:, 1]) ** 2])
    expected_dc = np.sqrt(col_energy) / np.exp(np.mean(np.log(np.sqrt(col_energy))))
    np.testing.assert_allclose(dr, expected_dr)
    np.testing.assert_allclose(dc, expected_dc)


def test_remediation_plan_predeclares_bounded_strata_calibration_and_gate():
    plan = json.loads(Path("benchmarks/configs/bp7_remediation.json").read_text())
    assert plan["strata"]["quantiles"] == [0.05, 0.35, 0.65, 0.95]
    assert plan["strata"]["maximum_entries"] == 2_000_000
    assert plan["black_box_scaling"]["total_columns"] == 4
    assert plan["promotion"]["component_error_max"] == 0.08
    assert {policy["name"] for policy in plan["adaptive_policies"]} == {
        "incumbent",
        "coarse_priority",
        "fine_priority",
    }


def _frontera_config():
    return json.loads(Path("benchmarks/configs/frontera_full_fixed.json").read_text())


def _frontera_fixture(tmp_path):
    """Independent physical MockGF encoded to Tandem component blocks."""
    axis = np.arange(8, dtype=float)
    x, y = np.meshgrid(axis, axis)
    coordinates = np.column_stack((x.ravel(), y.ravel(), np.zeros(64)))
    mesh = FaultMesh(coordinates, np.ones(64), dof_row=3)
    physical = MockGF(mesh).matvec(np.eye(128))
    raw = physical.reshape(16, 4, 3, 16, 4, 2).transpose(0, 2, 1, 3, 5, 4)
    (tmp_path / "gf").mkdir()
    with (tmp_path / "gf/gf_mat.bin").open("wb") as stream:
        stream.write(struct.pack(">6i", 1, 128, 1211216, 192, 128, -1))
        raw.astype(">f8").tofile(stream)
    coordinates.astype(">f8").tofile(tmp_path / "gf_coords.bin")
    config = _frontera_config()
    config["reader"] = {"chunk_rows": 16, "basis_functions_per_element": 4}
    config["guards"]["minimum_external_free_bytes"] = 1
    config["expected_data"] = {
        "shape": [192, 128],
        "matrix_file_bytes": 24 + physical.nbytes,
        "coordinate_sha256": hashlib.sha256((tmp_path / "gf_coords.bin").read_bytes()).hexdigest(),
    }
    return config, physical


def test_frontera_contract_freezes_independent_seeds_thresholds_and_new_success_pair():
    from benchmarks.frontera_full import acceptance_contract, validate_config

    config = _frontera_config()
    validate_config(config)
    assert config["construction_seeds"] == [11, 23, 37]
    assert config["validation_seeds"] == [101, 103]
    assert config["acceptance"]["global_relative_error_max"] == 1e-5
    assert config["acceptance"]["leaves_only_improvement_min"] == 100
    bp3 = json.loads(Path("benchmarks/results/bp3_fixed.json").read_text())
    bp7 = json.loads(Path("benchmarks/results/bp7_fixed.json").read_text())
    pending = {"status": "not_run", "dataset": "frontera", "config": config}
    gate = acceptance_contract(bp3, pending, bp7)
    assert gate["required_success"] == {"bp3": True, "frontera": False}
    assert not gate["coloring_unlocked"]
    assert gate["bp7"]["all_pass"] is False
    # Contract plumbing only: use a complete passing fixture, not claimed Frontera evidence.
    passing_fixture = {
        **bp3,
        "dataset": "frontera",
        "config": {**bp3["config"], "dataset": "frontera"},
    }
    assert acceptance_contract(bp3, passing_fixture, bp7)["c11_accepted"]
    assert not acceptance_contract(bp3, bp3, bp7)["c11_accepted"]
    incomplete = {**passing_fixture, "runs": bp3["runs"][:2]}
    assert not acceptance_contract(bp3, incomplete, bp7)["c11_accepted"]
    config["construction_seeds"] = [11, 23, 20261012]
    with pytest.raises(ValueError, match="pilot"):
        validate_config(config)


def test_frontera_v2_candidate_is_screen_gated_with_fresh_confirmation_seeds():
    from benchmarks.frontera_full import validate_config
    from benchmarks.frontera_screen import promote

    candidate = json.loads(Path("benchmarks/configs/frontera_full_fixed_v2.json").read_text())
    validate_config(candidate)
    assert candidate["construction_seeds"] == [41, 43, 47]
    assert candidate["validation_seeds"] == [107, 109]
    assert candidate["expected_prediction"]["construction_columns"] == 22_792
    assert candidate["confirmation"]["requires_screen_promotion"]
    results = [
        {"parameters": {"k": 64}, "l4_all": 1.0, "representative_all": 1.0},
        {"parameters": {"k": 96}, "l4_all": 0.7, "representative_all": 1.0},
        {"parameters": {"k": 128}, "l4_all": 0.6, "representative_all": 1.1},
    ]
    assert promote(results, {"l4_all_ratio_max": 0.8, "representative_all": 1.0}) == {"k": 96}


def test_frontera_stream_guards_prevent_excess_width_reads_rss_and_low_disk(tmp_path, monkeypatch):
    import benchmarks.frontera_full as formal

    config, physical = _frontera_fixture(tmp_path)
    source = formal.GuardedNativeDenseGF(tmp_path, config)
    np.testing.assert_allclose(source.matvec(np.ones(128)), physical @ np.ones(128))
    np.testing.assert_allclose(source.rmatvec(np.ones(192)), physical.T @ np.ones(192))
    assert source.completed_passes == 2
    config["guards"]["maximum_rhs_columns"] = 1
    with pytest.raises(RuntimeError, match="width"):
        source.matvec(np.ones((128, 2)))
    config["guards"]["maximum_total_passes_per_seed"] = 2
    with pytest.raises(RuntimeError, match="budget"):
        source.matvec(np.ones(128))
    monkeypatch.setattr(formal, "_peak_rss_bytes", lambda: config["guards"]["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="RSS"):
        source.check_resources()
    monkeypatch.setattr(formal, "_peak_rss_bytes", lambda: 0)
    monkeypatch.setattr(formal.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="free-space"):
        source.check_resources()


def test_frontera_preflight_rejects_memory_and_all_phase_traffic(tmp_path, monkeypatch):
    import benchmarks.frontera_full as formal

    config, _ = _frontera_fixture(tmp_path)
    source = formal.GuardedNativeDenseGF(tmp_path, config)
    prediction = {
        "full_matrix_passes": 137,
        "combined_columns_below_n_cols": True,
        "storage": {"total_bytes_upper_bound": 1},
        "peak_memory_model": {
            "modeled_known_live_array_bytes": 100,
            "maximum_leaf_probe_columns": 712,
            "maximum_admissible_probe_columns": 57,
        },
        "io_floor_seconds_from_width3_products": 0,
        "deliberately_pessimistic_linear_rhs_upper_seconds": 0,
        "linear_rhs_upper_method": "unused",
    }
    monkeypatch.setattr(
        formal, "construction_prediction", lambda *args: json.loads(json.dumps(prediction))
    )
    result = formal.preflight(source, config)
    assert result["all_phase_passes_upper_bound"] > 137 + 12
    config["guards"]["modeled_array_limit_bytes"] = 99
    with pytest.raises(RuntimeError, match="array guard"):
        formal.preflight(source, config)
    config["guards"]["modeled_array_limit_bytes"] = 100
    config["guards"]["maximum_total_passes_per_seed"] = 137
    with pytest.raises(RuntimeError, match="traffic"):
        formal.preflight(source, config)


def test_frontera_seed_checkpoint_keeps_all_diagnostics_and_interruption(tmp_path, monkeypatch):
    import benchmarks.frontera_full as formal

    config, _ = _frontera_fixture(tmp_path)
    config["selection"]["primary"] = {"m": 4, "k": 2, "p": 1}
    config["validation"]["power_iterations"] = 1
    monkeypatch.setattr(formal, "preflight", lambda *args: {"fixture_only": True})
    identity = {
        "data": {
            name: {"stat": formal._file_identity(path), "sha256": formal.sha256(path)}
            for name, path in (
                ("matrix", tmp_path / "gf/gf_mat.bin"),
                ("coordinates", tmp_path / "gf_coords.bin"),
            )
        },
        "source_sha256": {},
    }
    path = tmp_path / "seed_11.json"
    run = formal.execute_seed(tmp_path, config, 11, path, identity, 2.0)
    assert json.loads(path.read_text())["status"] == "completed"
    assert run["products"]["predicted_total"] == run["products"]["observed_total"]
    counts = run["products"]
    assert (
        run["streamed_passes_all_phases"]
        == sum(
            counts[group][key]
            for group in ("observed_total", "validation_excluded")
            for key in ("matvec_calls", "rmatvec_calls")
        )
        + 12
    )
    for record in run["records"]:
        diagnostics = record["diagnostics"]
        assert diagnostics["selected_block_output_errors"]
        assert diagnostics["global_relative_error"] >= 0
        for side in ("forward", "adjoint"):
            assert set(diagnostics[side]["input_errors"]) == {"random", "localized", "smooth"}
            assert diagnostics[side]["component_index_group_errors"]
            assert diagnostics[side]["leaves_only_improvement_factor"]
    config["guards"]["maximum_total_passes_per_seed"] = 1
    with pytest.raises(RuntimeError, match="budget"):
        formal.execute_seed(tmp_path, config, 11, path, identity, 2.0)
    interrupted = json.loads(path.read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["previous_attempts"][0]["status"] == "completed"


def test_frontera_cli_is_read_only_without_run_and_requires_thread_caps(
    tmp_path, monkeypatch, capsys
):
    import benchmarks.frontera_full as formal

    argv = ["frontera_full", "--dataset", str(tmp_path), "--revision", "fixture"]
    monkeypatch.setattr("sys.argv", argv)
    formal.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "not_run"
    assert not (tmp_path / "benchmark_results").exists()
    monkeypatch.setattr("sys.argv", [*argv, "--run"])
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with pytest.raises(ValueError, match="OPENBLAS_NUM_THREADS"):
        formal.main()


def test_frontera_resume_preserves_completed_seeds_and_rejects_changed_config(
    tmp_path, monkeypatch
):
    import benchmarks.frontera_full as formal

    config, _ = _frontera_fixture(tmp_path)
    config["selection"]["primary"] = {"m": 4, "k": 2, "p": 1}
    config["validation"]["power_iterations"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    pilot = tmp_path / config["calibration"]["path"]
    pilot.parent.mkdir()
    pilot.write_text('{"status": "fixture-only"}')
    monkeypatch.setattr(formal, "preflight", lambda *args: {"fixture_only": True})
    calls = []

    def isolated(function, *args):
        calls.append(args[2])
        result = function(*args)
        return result, formal._peak_rss_bytes()

    monkeypatch.setattr(formal, "isolated_peak_rss", isolated)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(name, "1")
    argv = [
        "frontera_full",
        "--dataset",
        str(tmp_path),
        "--config",
        str(config_path),
        "--revision",
        "fixture",
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)
    formal.main()
    aggregate = tmp_path / "benchmark_results/formal_fixed/frontera_fixed.json"
    original = json.loads(aggregate.read_text())
    assert calls == [11, 23, 37]
    assert original["identity"]["data"]["matrix"]["sha256"] == formal.sha256(
        tmp_path / "gf/gf_mat.bin"
    )
    assert original["acceptance_contract"]["bp7"]["all_pass"] is False
    monkeypatch.setattr("sys.argv", [*argv, "--resume"])
    formal.main()
    resumed = json.loads(aggregate.read_text())
    assert calls == [11, 23, 37]
    assert resumed["runs"] == original["runs"]
    config["acceptance"]["global_relative_error_max"] /= 2
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identity mismatch"):
        formal.main()
    assert json.loads(aggregate.read_text()) == resumed
