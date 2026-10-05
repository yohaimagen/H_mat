"""Full-run accounting contracts on a small smooth operator."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.full import main, measure_seed, prepare

from gfcompress.geometry import FaultMesh
from gfcompress.mockgf import MockGF


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
