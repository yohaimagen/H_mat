"""C.10 study contracts, kept compact for serial resource-safe testing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF
from gfcompress.operators import MatVecOperator
from gfcompress.study import configuration_schema, fixed_sweep, validate_seed_plan


def _mesh() -> FaultMesh:
    axis = np.arange(4, dtype=float)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return FaultMesh(np.column_stack((x.ravel(), y.ravel())), np.ones(16))


def test_seed_plan_rejects_favorable_selection_and_sweep_records_all_pairs() -> None:
    with pytest.raises(ValueError, match="three"):
        validate_seed_plan([1, 2], [3])
    with pytest.raises(ValueError, match="independent"):
        validate_seed_plan([1, 2, 3], [3])
    mesh = _mesh()
    records = fixed_sweep(
        MockGF(mesh),
        mesh,
        m_values=[1],
        k_values=[1],
        p_values=[1],
        construction_seeds=[1, 2, 3],
        validation_seeds=[11],
        absolute_floor=1e-10,
    )
    assert {(r["construction_seed"], r["validation_seed"]) for r in records} == {
        (1, 11),
        (2, 11),
        (3, 11),
    }
    diagnostics = records[0]["diagnostics"]
    for direction, dof in (("forward", mesh.dof_col), ("adjoint", mesh.dof_row)):
        assert {"random", "localized", "smooth"} == set(diagnostics[direction]["input_errors"])
        assert set(diagnostics[direction]["component_index_group_errors"]) == {
            str(index) for index in range(dof)
        }
        assert all(
            "leaves_only" in value for value in diagnostics[direction]["input_errors"].values()
        )
        assert min(diagnostics[direction]["leaves_only_improvement_factor"].values()) > 1
    assert all(
        value["retained_rank"] < min(value["row_dimension"], value["col_dimension"])
        for value in diagnostics["selected_block_output_errors"].values()
    )


def test_unlocked_configuration_and_missing_data_record_are_honest(tmp_path: Path) -> None:
    config = configuration_schema("bp3")
    assert config["selection"]["primary"] is None
    assert config["acceptance"]["frozen"] is False
    for dataset in ("bp3", "bp7"):
        path = Path(f"benchmarks/configs/{dataset}_fixed.json")
        tracked = json.loads(path.read_text(encoding="utf-8"))
        assert tracked["dataset"] == dataset
        assert {
            "sweep",
            "construction_seeds",
            "validation_seeds",
            "validation",
            "selection",
            "acceptance",
        } <= set(tracked)
        assert {"m", "k", "p"} <= set(tracked["sweep"])
        assert {"power_iterations", "absolute_response_floor", "status"} <= set(
            tracked["validation"]
        )
    spec = importlib.util.spec_from_file_location("c10", Path("benchmarks/c10.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.report(tmp_path, "bp3", run=False)
    assert result["status"] == "blocked"
    assert "not evaluated" in result["acceptance"]


def test_bp3_selected_block_criterion_rejects_zero_output() -> None:
    report = json.loads(Path("benchmarks/results/bp3_c10_sweep.json").read_text(encoding="utf-8"))
    config = json.loads(Path("benchmarks/configs/bp3_fixed.json").read_text(encoding="utf-8"))
    primary = config["selection"]["primary"]
    records = [
        record
        for record in report["records"]
        if all(record["parameters"][name] == primary[name] for name in ("m", "k", "p"))
    ]
    threshold = config["acceptance"]["selected_block_output_error_max"]
    selected = [
        value
        for record in records
        for value in record["diagnostics"]["selected_block_output_errors"].values()
    ]
    assert max(value["relative_to_response"] for value in selected) <= threshold
    # Relative to its own nonzero unit-drive response, zero output is exactly
    # one; every selected genuinely truncated block must fail the criterion.
    assert threshold < 1.0


def test_bp7_primary_selected_blocks_are_genuinely_truncated_at_every_level() -> None:
    report = json.loads(Path("benchmarks/results/bp7_c10_sweep.json").read_text(encoding="utf-8"))
    config = json.loads(Path("benchmarks/configs/bp7_fixed.json").read_text(encoding="utf-8"))
    primary = config["selection"]["primary"]
    records = [
        record
        for record in report["records"]
        if all(record["parameters"][name] == primary[name] for name in ("m", "k", "p"))
    ]
    selected = [
        (name, value)
        for record in records
        for name, value in record["diagnostics"]["selected_block_output_errors"].items()
    ]
    assert {name.split(":", 1)[0] for name, _ in selected} == {"L2", "L3", "L4"}
    assert all(
        value["retained_rank"] < min(value["row_dimension"], value["col_dimension"])
        for _, value in selected
    )
    assert (
        max(value["relative_to_response"] for _, value in selected)
        <= config["acceptance"]["selected_block_output_error_max"]
        < 1.0
    )


def test_runner_consumes_injected_tracked_config_and_records_source_subset_identity(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("c10", Path("benchmarks/c10.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = configuration_schema("bp3")
    config["sweep"] = {"m": [1], "k": [1], "p": [1]}
    config["validation"] = {
        "power_iterations": 3,
        "absolute_response_floor": 1e-9,
        "status": "provisional",
    }
    config_path = tmp_path / "locked-later.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    directory = tmp_path / "gf_bp3"
    directory.mkdir()
    (directory / "gf_mat.bin").write_bytes(b"matrix")
    (directory / "bp3_fbf_coords.csv").write_text("coordinates", encoding="utf-8")
    mesh = _mesh()

    class Subset(MatVecOperator):
        def __init__(self) -> None:
            self._inner = MockGF(mesh)
            self.mesh = mesh
            self.patch_ids = np.arange(mesh.n_patches, dtype=np.intp)
            self.row_map = np.arange(mesh.n_rows, dtype=np.intp)
            self.col_map = np.arange(mesh.n_cols, dtype=np.intp)

        @property
        def shape(self) -> tuple[int, int]:
            return self._inner.shape

        def matvec(self, omega: np.ndarray) -> np.ndarray:
            return self._inner.matvec(omega)

        def rmatvec(self, psi: np.ndarray) -> np.ndarray:
            return self._inner.rmatvec(psi)

    result = module.report(
        tmp_path,
        "bp3",
        run=True,
        config_path=config_path,
        source_type=lambda *_: object(),
        subset_builder=lambda *_args, **_kwargs: Subset(),
    )
    assert result["config"] == config
    assert result["source"]["matrix"]["size"] == len(b"matrix")
    assert result["subset"]["patch_ids_hash"]
    record = result["records"][0]
    assert record["absolute_floor"] == 1e-9
    assert record["selection_measurements"]["storage"]["dense_reference_bytes"] > 0
    assert record["selection_measurements"]["storage"]["dense_storage_ratio"] is not None
    assert record["diagnostics"]["global_relative_error"] >= 0
    assert result["environment"]["python"]
    assert result["timing_protocol"]["rhs_columns"] == 1
    assert "warm" in result["timing_protocol"]["cache"]

    config["selection"]["primary"] = {"m": 1, "k": 1, "p": 1}
    config["selection"]["alternatives"] = [{"m": 2, "k": 1, "p": 1}]
    config["acceptance"].update(
        {
            "frozen": True,
            "global_relative_error_max": 0.1,
            "selected_block_output_error_max": 1.0,
            "component_response_error_max": 1.0,
            "input_response_error_max": 1.0,
            "leaves_only_improvement_min": 1.0,
            "comparison_relative_slack": 0.01,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    locked = module.report(
        tmp_path,
        "bp3",
        run=True,
        config_path=config_path,
        source_type=lambda *_: object(),
        subset_builder=lambda *_args, **_kwargs: Subset(),
    )
    assert locked["status"] == "measured_locked"
    assert "frozen configuration was consumed" in locked["acceptance"]
    assert locked["config"]["selection"]["primary"] == {"m": 1, "k": 1, "p": 1}
    assert not locked["acceptance_evaluation"]["evaluated"]
    assert "genuinely truncated" in locked["acceptance_evaluation"]["reason"]
