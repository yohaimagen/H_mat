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
    assert candidate["confirmation"]["requires_exploratory_pass"]
    original = _frontera_config()["acceptance"]
    assert {k: v for k, v in candidate["acceptance"].items() if k != "reason"} == {
        k: v for k, v in original.items() if k != "reason"
    }
    exploration = json.loads(Path("benchmarks/configs/frontera_exploratory_v2.json").read_text())
    assert exploration["candidate"] == candidate["selection"]["primary"]
    assert exploration["early_gates"] == {
        "random_leaves_only_improvement_min": 150.0,
        "selected_block_error_max": 5e-4,
    }
    seeds = candidate["construction_seeds"] + candidate["validation_seeds"] + [11, 23, 37, 101, 103]
    assert exploration["construction_seed"] not in seeds
    assert exploration["validation_seed"] not in seeds
    results = [
        {"parameters": {"k": 64}, "l4_all": 1.0, "representative_all": 1.0},
        {"parameters": {"k": 96}, "l4_all": 0.7, "representative_all": 1.0},
        {"parameters": {"k": 128}, "l4_all": 0.6, "representative_all": 1.1},
    ]
    assert promote(results, {"l4_all_ratio_max": 0.8, "representative_all": 1.0}) == {"k": 96}
    results[2]["representative_all"] = 0.5
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


def _screen_evidence():
    from benchmarks.frontera_screen import PROBE_POLICY, array_hash, probes, seal

    plan = json.loads(Path("benchmarks/configs/frontera_screen_v1.json").read_text())
    manifest = [{**b, "shape": [132, 130]} for b in plan["blocks"]]
    results = []
    for candidate, error in zip(
        [plan["comparator"], *plan["candidates"]], [1.0, 0.7, 0.6], strict=True
    ):
        records = []
        for spec in manifest:
            for seed in plan["screen_seeds"]:
                ga, gb = probes(spec, seed, 160)
                width = candidate["k"] + candidate["p"]
                records.append(
                    {
                        "block": spec["name"],
                        "seed": seed,
                        "probe_sha256": [array_hash(ga[:, :width]), array_hash(gb[:, :width])],
                        "errors": {"clean_two_sample": {"all": error}},
                    }
                )
        results.append(
            {
                "parameters": candidate,
                "records": records,
                "l4_all": error,
                "representative_all": error,
            }
        )
    binding = {"config": plan, "fixture": True}
    return (
        seal(
            {
                "schema_version": 2,
                "kind": "screen",
                "binding": binding,
                "manifest": manifest,
                "probe_policy": PROBE_POLICY,
                "actual_read_bytes": sum(b["expected_read_bytes"] for b in manifest),
                "results": results,
                "promotion": plan["candidates"][0],
            }
        ),
        binding,
        manifest,
    )


def test_screen_nested_paired_probes_and_separate_streams():
    from benchmarks.frontera_screen import _record, probes

    spec = {"shape": [32, 16], "level": 4, "alpha_index": 16, "beta_index": 2}
    ga, gb = probes(spec, 401, 12)
    again = probes(spec, 401, 12)
    np.testing.assert_array_equal(ga, again[0])
    np.testing.assert_array_equal(gb, again[1])
    assert not np.array_equal(ga[:16], gb)
    mesh = FaultMesh(np.column_stack((np.arange(64), np.zeros(64))), np.ones(64))
    block = MockGF(mesh).matvec(np.eye(64))[:32, 48:64]
    from benchmarks.frontera_screen import array_hash

    for k, p in [(3, 2), (6, 3)]:
        record = _record(block, {"k": k, "p": p}, ga, gb, 2)
        assert record["probe_sha256"] == [array_hash(ga[:, : k + p]), array_hash(gb[:, : k + p])]


@pytest.mark.parametrize(
    "damage",
    [
        "bare",
        "tamper",
        "revision",
        "seed",
        "candidate",
        "block",
        "probe",
        "aggregate",
        "promotion",
        "bytes",
        "nan",
    ],
)
def test_screen_rejects_incomplete_or_tampered_evidence(damage):
    from benchmarks.frontera_screen import seal, validate_screen

    report, binding, manifest = _screen_evidence()
    assert validate_screen(report, binding, manifest) == {"m": 512, "k": 96, "p": 32}
    if damage == "bare":
        report = {"promotion": report["promotion"]}
    elif damage == "tamper":
        report["promotion"] = None
    elif damage == "revision":
        binding = {**binding, "revision": "different"}
    else:
        if damage == "seed":
            report["results"][0]["records"].pop()
        elif damage == "candidate":
            report["results"].reverse()
        elif damage == "block":
            report["manifest"][0]["alpha_index"] = 7
            manifest = [{**b} for b in _screen_evidence()[2]]
        elif damage == "probe":
            report["results"][0]["records"][0]["probe_sha256"][0] = "wrong"
        elif damage == "aggregate":
            report["results"][0]["l4_all"] = 0.2
        elif damage == "promotion":
            report["promotion"] = report["results"][2]["parameters"]
        elif damage == "bytes":
            report["actual_read_bytes"] += 1
        elif damage == "nan":
            report["results"][0]["records"][0]["errors"]["clean_two_sample"]["all"] = float("nan")
            with pytest.raises(ValueError):
                seal(report)
            return
        report = seal(report)
    with pytest.raises(ValueError):
        validate_screen(report, binding, manifest)


def test_evidence_atomic_no_overwrite_and_dataset_source_binding(tmp_path):
    from benchmarks.frontera_screen import evidence_binding, seal, verify, write_once

    config, _ = _frontera_fixture(tmp_path)
    binding = evidence_binding(tmp_path, config, "reviewed-fixture")
    record = {"schema_version": 2, "kind": "fixture", "binding": binding}
    path = tmp_path / "report.json"
    write_once(path, record)
    original = path.read_bytes()
    verify(json.loads(original), "fixture", binding)
    with pytest.raises(FileExistsError):
        write_once(path, {"replacement": True})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".evidence-*"))
    changed = {**binding, "sources": {}}
    with pytest.raises(ValueError):
        verify(seal(record), "fixture", changed)
    with (tmp_path / "gf_coords.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="coordinate hash"):
        evidence_binding(tmp_path, config, "reviewed-fixture")


def test_screen_read_accounting_and_resource_guards(tmp_path, monkeypatch):
    import benchmarks.frontera_screen as screen

    from gfcompress.frontera import NativeDenseGF

    config, matrix = _frontera_fixture(tmp_path)
    source = NativeDenseGF(
        tmp_path / "gf/gf_mat.bin", tmp_path / "gf_coords.bin", **config["reader"]
    )
    spec = {"shape": [3, 2], "rows": [0, 4, 9], "cols": [1, 8], "expected_read_bytes": 3 * 128 * 8}
    plan = {"guards": {"rss_limit_bytes": 10**12, "minimum_external_free_bytes": 1}}
    block, actual = screen.read_block_once(source, spec, plan)
    np.testing.assert_array_equal(block, matrix[np.ix_(spec["rows"], spec["cols"])])
    assert actual == spec["expected_read_bytes"]
    with pytest.raises(RuntimeError, match="byte mismatch"):
        screen.read_block_once(source, {**spec, "expected_read_bytes": 1}, plan)
    monkeypatch.setattr(screen, "_peak_rss_bytes", lambda: 10**13)
    with pytest.raises(RuntimeError, match="RSS"):
        screen.check_resources(tmp_path, plan)
    monkeypatch.setattr(screen, "_peak_rss_bytes", lambda: 0)
    monkeypatch.setattr(screen.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="free-disk"):
        screen.check_resources(tmp_path, plan)


def _exploratory_evidence(tmp_path):
    import benchmarks.frontera_full as formal
    from benchmarks.frontera_screen import array_hash, seal

    config, _ = _frontera_fixture(tmp_path)
    source = formal.GuardedNativeDenseGF(tmp_path, config)
    exploration = json.loads(formal.EXPLORATORY.read_text())
    exploration["candidate"] = {"m": 4, "k": 2, "p": 1}
    manifest = formal.exploratory_manifest(source, exploration)
    records = {
        "random": {
            side: {
                "error": 1.0,
                "leaves_error": 150.0,
                "reference_norm": 2.0,
                "probe_sha256": array_hash(x),
            }
            for side, x in formal.exploratory_probes(source, exploration).items()
        },
        "selected_blocks": [
            {"identity": b, "error": 0.0005, "reference_norm": 1.0} for b in manifest
        ],
    }
    gates = formal.early_gates(records, exploration)
    binding = {"config": exploration}
    screen = {"content_sha256": "screen-fixture"}
    report = seal(
        {
            "schema_version": 2,
            "kind": "exploratory",
            "binding": binding,
            "screen_content_sha256": screen["content_sha256"],
            "matrix_sha256": "a" * 64,
            "probe_policy": formal.EARLY_PROBE_POLICY,
            "manifest": manifest,
            "run": {
                "construction_seed": 431,
                "validation_seed": 433,
                "records": records,
                "gates": gates,
                "all_pass": True,
                "later_diagnostics_run": False,
            },
        }
    )
    return report, binding, source, screen


@pytest.mark.parametrize(
    "damage", ["random", "selected", "seed", "missing", "probe", "matrix", "screen", "later"]
)
def test_exploratory_evidence_requires_every_early_gate_and_identity(tmp_path, damage):
    import benchmarks.frontera_full as formal
    from benchmarks.frontera_screen import seal

    report, binding, source, screen = _exploratory_evidence(tmp_path)
    formal.validate_exploratory(report, binding, source, screen, "a" * 64)
    run = report["run"]
    if damage == "random":
        run["records"]["random"]["forward"]["leaves_error"] = 149.9
    elif damage == "selected":
        run["records"]["selected_blocks"][0]["error"] = 0.000501
    elif damage == "seed":
        run["construction_seed"] = 41
    elif damage == "missing":
        run["records"]["selected_blocks"].pop()
    elif damage == "probe":
        run["records"]["random"]["adjoint"]["probe_sha256"] = "wrong"
    elif damage == "matrix":
        report["matrix_sha256"] = "wrong"
    elif damage == "screen":
        report["screen_content_sha256"] = "wrong"
    elif damage == "later":
        run["later_diagnostics_run"] = True
    with pytest.raises(ValueError):
        formal.validate_exploratory(seal(report), binding, source, screen, "a" * 64)


def test_exploratory_constructs_once_and_stops_before_later_diagnostics(tmp_path, monkeypatch):
    import benchmarks.frontera_full as formal

    _, binding, source, _ = _exploratory_evidence(tmp_path)
    calls = []
    original = formal.compress

    def construct(*args, **kwargs):
        calls.append(kwargs["seed"])
        return original(*args, **kwargs)

    monkeypatch.setattr(formal, "compress", construct)
    monkeypatch.setattr(formal, "measure_seed", lambda *args: pytest.fail("late diagnostics"))
    run = formal.measure_exploratory(
        source, binding["config"], formal.exploratory_manifest(source, binding["config"])
    )
    assert calls == [431]
    assert not run["later_diagnostics_run"]
    assert not run["all_pass"]


def test_prediction_explicit_candidates_do_not_mutate_global_state(tmp_path):
    from benchmarks.frontera import CANDIDATE, construction_prediction
    from benchmarks.frontera_full import GuardedNativeDenseGF

    from gfcompress.build_tree import build_tree

    config, _ = _frontera_fixture(tmp_path)
    source = GuardedNativeDenseGF(tmp_path, config)
    root = build_tree(source.mesh, 4)
    stream = {"forward_seconds": 1, "adjoint_seconds": 1, "rhs_columns_per_pass": 3}
    original = CANDIDATE.copy()
    first = construction_prediction(source, root, stream, {"m": 4, "k": 2, "p": 1})
    second = construction_prediction(source, root, stream, {"m": 4, "k": 3, "p": 2})
    assert construction_prediction(source, root, stream, {"m": 4, "k": 2, "p": 1}) == first
    assert first["combined_construction_columns"] != second["combined_construction_columns"]
    assert CANDIDATE == original


@pytest.mark.parametrize(
    "damage", [None, "cells", "shape", "hash", "bytes", "entries", "cap", "rank"]
)
def test_screen_manifest_guards_whole_block_geometry_and_cost(damage):
    from benchmarks.frontera_screen import array_hash, block_manifest

    from gfcompress.build_tree import build_tree
    from gfcompress.interactions import build_lists

    x, y = np.meshgrid(np.arange(48), np.arange(48))
    mesh = FaultMesh(
        np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))), np.ones(x.size), dof_row=3
    )
    root = build_tree(mesh, 200)
    levels = list(root.iter_levels())
    lists = build_lists(root)
    a = next(a for a in levels[2] if lists.interaction[a])
    b = lists.interaction[a][0]
    spec = {
        "name": "fixture",
        "level": 2,
        "alpha_index": a.index_in_level,
        "beta_index": b.index_in_level,
        "shape": [a.row_indices.size, b.col_indices.size],
        "index_sha256": [array_hash(i.astype("<i8")) for i in (a.row_indices, b.col_indices)],
        "expected_read_bytes": a.row_indices.size * mesh.n_cols * 8,
    }
    plan = {
        "comparator": {"m": 200},
        "expected_data": {"shape": [mesh.n_rows, mesh.n_cols]},
        "occupied_cells_by_level": [len(n) for n in levels],
        "blocks": [spec],
        "maximum_block_entries": 8_000_000,
        "guards": {"maximum_read_bytes": 20_000_000_000},
    }
    source = SimpleNamespace(shape=(mesh.n_rows, mesh.n_cols), mesh=mesh)
    if damage is None:
        result = block_manifest(source, plan)
        assert result[0]["rows"] == a.row_indices.tolist()
        return
    if damage == "cells":
        plan["occupied_cells_by_level"][0] = 2
    elif damage == "shape":
        spec["shape"][0] += 1
    elif damage == "hash":
        spec["index_sha256"][0] = "wrong"
    elif damage == "bytes":
        spec["expected_read_bytes"] += 1
    elif damage == "entries":
        plan["maximum_block_entries"] = 1
    elif damage == "cap":
        plan["guards"]["maximum_read_bytes"] = 1
    elif damage == "rank":
        plan["comparator"]["m"] = 1
        plan["occupied_cells_by_level"] = [len(n) for n in build_tree(mesh, 1).iter_levels()]
        spec["level"] = len(plan["occupied_cells_by_level"]) - 1
    with pytest.raises(ValueError):
        block_manifest(source, plan)


def test_screen_reads_each_block_once_and_reuses_maximum_probes(tmp_path, monkeypatch):
    import benchmarks.frontera_screen as screen

    _, binding, manifest = _screen_evidence()
    reads, seen = [], []
    monkeypatch.setattr(screen, "check_resources", lambda *args: None)
    monkeypatch.setattr(screen, "evidence_binding", lambda *args: binding)
    monkeypatch.setattr(screen, "NativeDenseGF", lambda *args, **kwargs: SimpleNamespace(dof_row=3))
    monkeypatch.setattr(screen, "block_manifest", lambda *args: manifest)

    def read(_source, spec, _plan):
        reads.append(spec["name"])
        return np.zeros(spec["shape"]), spec["expected_read_bytes"]

    def record(_block, candidate, ga, gb, _dof):
        seen.append((ga, gb))
        width = candidate["k"] + candidate["p"]
        return {
            "probe_sha256": [screen.array_hash(ga[:, :width]), screen.array_hash(gb[:, :width])],
            "errors": {"clean_two_sample": {"all": {64: 1.0, 96: 0.7, 128: 0.6}[candidate["k"]]}},
        }

    monkeypatch.setattr(screen, "read_block_once", read)
    monkeypatch.setattr(screen, "_record", record)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        "sys.argv", ["screen", "--dataset", str(tmp_path), "--revision", "fixture", "--run"]
    )
    screen.main()
    assert reads == ["failed_l4", "representative_l4"]
    for i in range(0, len(seen), 3):
        assert all(seen[j][0] is seen[i][0] and seen[j][1] is seen[i][1] for j in range(i, i + 3))
        assert seen[i][0].shape[1] == 160
    with pytest.raises(FileExistsError):
        screen.main()


@pytest.mark.parametrize("evidence", ["none", "bare", "screen_only", "stale_exploration"])
def test_formal_cli_rejects_bad_evidence_before_full_matrix_work(tmp_path, monkeypatch, evidence):
    import benchmarks.frontera_full as formal

    report, binding, manifest = _screen_evidence()
    argv = [
        "formal",
        "--config",
        str(formal.V2_CONFIG),
        "--dataset",
        str(tmp_path),
        "--revision",
        "fixture",
        "--run",
    ]
    monkeypatch.setattr(formal, "NativeDenseGF", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(formal, "evidence_binding", lambda *args: binding)
    monkeypatch.setattr(formal, "block_manifest", lambda *args: manifest)
    monkeypatch.setattr(
        formal, "GuardedNativeDenseGF", lambda *args: pytest.fail("full matrix work")
    )
    if evidence != "none":
        path = tmp_path / "screen.json"
        path.write_text(
            json.dumps({"promotion": report["promotion"]} if evidence == "bare" else report)
        )
        argv += ["--screen-report", str(path)]
    if evidence == "stale_exploration":
        path = tmp_path / "exploratory.json"
        path.write_text('{"schema_version": 1, "all_pass": true}')
        argv += ["--exploratory-report", str(path)]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError):
        formal.main()
