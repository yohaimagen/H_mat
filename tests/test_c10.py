"""C.10 study contracts, kept compact for serial resource-safe testing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF
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
    assert {"random", "localized", "smooth"} == set(diagnostics["input_errors"])
    assert diagnostics["selected_block_output_errors"]
    assert all("leaves_only" in value for value in diagnostics["input_errors"].values())


def test_unlocked_configuration_and_missing_data_record_are_honest(tmp_path: Path) -> None:
    config = configuration_schema("bp3")
    assert config["selection"]["primary"] is None
    assert config["acceptance"]["frozen"] is False
    for dataset in ("bp3", "bp7"):
        path = Path(f"benchmarks/configs/{dataset}_fixed.json")
        assert json.loads(path.read_text(encoding="utf-8")) == configuration_schema(dataset)
    spec = importlib.util.spec_from_file_location("c10", Path("benchmarks/c10.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.report(tmp_path, "bp3", run=False)
    assert result["status"] == "blocked"
    assert "not evaluated" in result["acceptance"]
