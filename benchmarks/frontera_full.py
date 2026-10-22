#!/usr/bin/env python3
"""Reviewed, opt-in three-seed Frontera confirmation; no dense copy or saved factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from benchmarks.c10 import _file_identity, _load_config
from benchmarks.frontera import (
    DEFAULT_DATASET,
    _environment,
    _peak_rss_bytes,
    construction_prediction,
)
from benchmarks.full import measure_seed
from numpy.typing import NDArray

from gfcompress.benchmark import isolated_peak_rss
from gfcompress.build_tree import build_tree
from gfcompress.frontera import NativeDenseGF
from gfcompress.study import validate_seed_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "benchmarks/configs/frontera_full_fixed.json"


def sha256(path: Path) -> str:
    """Hash even the matrix in bounded memory; this is a separately charged scan."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(path: Path, record: dict[str, Any]) -> None:
    """Atomically replace a small JSON record; never serialize numerical factors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    validate_seed_plan(config["construction_seeds"], config["validation_seeds"])
    if len(config["construction_seeds"]) != 3 or len(set(config["validation_seeds"])) != 2:
        raise ValueError(
            "formal confirmation requires exactly three seeds and two validation starts"
        )
    if config["calibration"]["construction_seed"] in config["construction_seeds"]:
        raise ValueError("pilot cannot be a formal construction seed")
    if not config["acceptance"]["frozen"] or config["sampling"] != "fixed":
        raise ValueError("formal confirmation requires frozen fixed-path acceptance")
    for name in ("m", "k", "p"):
        value = config["selection"]["primary"][name]
        if not isinstance(value, int) or value < (0 if name == "p" else 1):
            raise ValueError(f"invalid primary {name}")
    for name in (
        "global_relative_error_max",
        "component_response_error_max",
        "input_response_error_max",
        "selected_block_output_error_max",
    ):
        if not 0 < config["acceptance"][name] < 1:
            raise ValueError(f"invalid discriminating threshold {name}")
    if config["acceptance"]["leaves_only_improvement_min"] <= 1:
        raise ValueError("leaves-only improvement must exceed one")
    if config["validation"]["power_iterations"] < 1:
        raise ValueError("power_iterations must be positive")
    if (
        config["validation"]["absolute_response_floor"]
        != config["acceptance"]["absolute_response_floor"]
    ):
        raise ValueError("validation and acceptance floors must agree")
    if config["timing"] != {"repeats": 5, "warmups": 1, "rhs_columns": 1, "seed": 101}:
        raise ValueError("timing must match the shared full-baseline protocol")
    if any(value <= 0 for value in config["guards"].values()):
        raise ValueError("resource guards must be positive")


class GuardedNativeDenseGF(NativeDenseGF):
    """Stream unchanged file data with RSS, width, traffic, and free-space guards."""

    def __init__(self, dataset: Path, config: dict[str, Any]) -> None:
        self.dataset = dataset
        self.limits = config["guards"]
        self.completed_passes = 0
        self.check_resources()
        super().__init__(dataset / "gf/gf_mat.bin", dataset / "gf_coords.bin", **config["reader"])

    def check_resources(self) -> None:
        if _peak_rss_bytes() > self.limits["rss_limit_bytes"]:
            raise RuntimeError("RSS guard exceeded")
        if shutil.disk_usage(self.dataset).free < self.limits["minimum_external_free_bytes"]:
            raise RuntimeError("external free-space guard exceeded")

    def _chunks(self, stream: BinaryIO) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
        # Chunk-boundary checks also cover memory growth within a streamed pass.
        for item in super()._chunks(stream):
            self.check_resources()
            yield item

    def _product(self, value: NDArray[np.floating], *, transpose: bool) -> NDArray[np.floating]:
        self.check_resources()
        width = 1 if value.ndim == 1 else value.shape[1]
        if width > self.limits["maximum_rhs_columns"]:
            raise RuntimeError("RHS width guard exceeded")
        passes = self.completed_passes + 1
        if (
            passes > self.limits["maximum_total_passes_per_seed"]
            or passes * self.header.value_bytes > self.limits["maximum_total_read_bytes_per_seed"]
        ):
            raise RuntimeError("streamed-read budget exceeded")
        result = super().rmatvec(value) if transpose else super().matvec(value)
        self.completed_passes = passes
        self.check_resources()
        return result

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._product(omega, transpose=False)

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._product(psi, transpose=True)


def preflight(source: GuardedNativeDenseGF, config: dict[str, Any]) -> dict[str, Any]:
    """Predict before construction; no operator product or matrix materialization."""
    expected = config["expected_data"]
    if (
        list(source.shape) != expected["shape"]
        or source.mat_path.stat().st_size != expected["matrix_file_bytes"]
    ):
        raise ValueError("dataset shape/size differs from frozen configuration")
    primary = config["selection"]["primary"]
    # This helper's fixed candidate is deliberately checked before reuse.
    if primary != {"m": 512, "k": 47, "p": 10}:
        raise ValueError("Frontera resource prediction requires the frozen pilot parameters")
    root = build_tree(source.mesh, primary["m"])
    prediction = construction_prediction(
        source,
        root,
        {"forward_seconds": 0.0, "adjoint_seconds": 0.0, "rhs_columns_per_pass": 3},
    )
    for name in (
        "io_floor_seconds_from_width3_products",
        "deliberately_pessimistic_linear_rhs_upper_seconds",
        "linear_rhs_upper_method",
    ):
        prediction.pop(name)
    memory = prediction["peak_memory_model"]
    limits = config["guards"]
    if memory["modeled_known_live_array_bytes"] > limits["modeled_array_limit_bytes"]:
        raise RuntimeError("modeled array guard exceeded")
    if (
        max(memory["maximum_leaf_probe_columns"], memory["maximum_admissible_probe_columns"])
        > limits["maximum_rhs_columns"]
    ):
        raise RuntimeError("predicted RHS width exceeds guard")
    if not prediction["combined_columns_below_n_cols"]:
        raise RuntimeError("construction columns must be below n_cols")
    if prediction["storage"]["total_bytes_upper_bound"] >= source.header.value_bytes:
        raise RuntimeError("predicted numerical storage must be below dense")
    # C.10: 17 direction/component calls, <= one selected block per active level,
    # and four calls per power iteration. Timing adds 2*(1+5) dense products.
    levels = len(list(root.iter_levels())) - 2
    validation_bound = len(config["validation_seeds"]) * (
        17 + max(levels, 0) + 4 * config["validation"]["power_iterations"]
    )
    total_passes = prediction["full_matrix_passes"] + validation_bound + 12
    if (
        total_passes > limits["maximum_total_passes_per_seed"]
        or total_passes * source.header.value_bytes > limits["maximum_total_read_bytes_per_seed"]
    ):
        raise RuntimeError("predicted validation/construction/timing traffic exceeds guard")
    prediction["all_phase_passes_upper_bound"] = total_passes
    prediction["all_phase_read_bytes_upper_bound"] = total_passes * source.header.value_bytes
    return prediction


def formal_success(report: dict[str, Any], dataset: str) -> bool:
    """Require every declared construction/start record, not just an aggregate flag."""
    config = report["config"]
    runs = report.get("runs", [])
    if report.get("dataset") != dataset or config.get("dataset") != dataset:
        return False
    if len(set(config["construction_seeds"])) < 3 or not config["acceptance"]["frozen"]:
        return False
    if report.get("status") != "passed" or report.get("all_pass") is not True:
        return False
    if sorted(run["construction_seed"] for run in runs) != sorted(config["construction_seeds"]):
        return False
    return all(
        run["all_pass"]
        and all(run["gates"].values())
        and sorted(record["validation_seed"] for record in run["records"])
        == sorted(config["validation_seeds"])
        for run in runs
    )


def acceptance_contract(
    bp3: dict[str, Any], frontera: dict[str, Any], bp7: dict[str, Any]
) -> dict[str, Any]:
    passing = {"bp3": formal_success(bp3, "bp3"), "frontera": formal_success(frontera, "frontera")}
    return {
        "required_success": passing,
        "c11_accepted": all(passing.values()),
        "coloring_unlocked": all(passing.values()),
        "bp7": {
            "role": "negative evidence, not a success gate",
            "status": bp7["status"],
            "all_pass": bp7["all_pass"],
        },
    }


def check_identity(dataset: Path, identity: dict[str, Any]) -> None:
    for name, relative in (("matrix", "gf/gf_mat.bin"), ("coordinates", "gf_coords.bin")):
        if _file_identity(dataset / relative) != identity["data"][name]["stat"]:
            raise RuntimeError("data identity changed since hashing")
    for name, digest in identity["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError("source identity changed during confirmation")


def execute_seed(
    dataset: Path,
    config: dict[str, Any],
    seed: int,
    path: Path,
    identity: dict[str, Any],
    hash_seconds: float,
) -> dict[str, Any]:
    """Fresh process per seed; interrupted seeds restart, completed records persist."""
    record: dict[str, Any] = {"status": "running", "construction_seed": seed, "identity": identity}
    if path.exists():
        previous = json.loads(path.read_text())
        record["previous_attempts"] = [
            *previous.get("previous_attempts", []),
            {key: value for key, value in previous.items() if key != "previous_attempts"},
        ]
    checkpoint(path, record)
    try:
        check_identity(dataset, identity)
        start = time.perf_counter()
        source = GuardedNativeDenseGF(dataset, config)
        record["preflight"] = preflight(source, config)
        preparation = time.perf_counter() - start + hash_seconds
        checkpoint(path, record)
        run = measure_seed(source, source.mesh, config, seed, preparation)
        source.check_resources()
        check_identity(dataset, identity)
        run["storage"]["peak_rss_bytes"] = _peak_rss_bytes()
        run["streamed_passes_all_phases"] = source.completed_passes
        run["streamed_bytes_all_phases"] = source.completed_passes * source.header.value_bytes
        json.dumps(run, allow_nan=False)  # nonfinite diagnostics cannot become a passing checkpoint
        record.update(status="completed", run=run)
        checkpoint(path, record)
        return run
    except Exception as exc:
        record.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
        checkpoint(path, record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = _load_config("frontera", args.config)
    validate_config(config)
    output = (args.output_dir or args.dataset / "benchmark_results/formal_fixed").resolve()
    if not output.is_relative_to(args.dataset.resolve()) or output.is_relative_to(ROOT):
        raise ValueError("output directory must be inside the external dataset, outside repository")
    sources = [
        Path(__file__),
        ROOT / "benchmarks/full.py",
        ROOT / "benchmarks/c10.py",
        ROOT / "benchmarks/frontera.py",
        ROOT / "BP_COLORING_PLAN.md",
        *sorted((ROOT / "src/gfcompress").glob("*.py")),
    ]
    identity: dict[str, Any] = {
        "revision_supplied_by_orchestrator": args.revision,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "config_sha256": sha256(args.config),
        "environment": _environment(),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "frontera",
        "config": config,
        "identity": identity,
        "environment": _environment(),
        "command": [sys.executable, *sys.argv],
        "status": "not_run",
        "runs": [],
        "all_pass": False,
        "timing_protocol": {
            **config["timing"],
            "dense": "streamed endian-converted NativeDenseGF; no native dense copy",
            "setup": (
                "reader/preflight plus bounded full-data hashing charged "
                "to each independently usable seed"
            ),
        },
    }
    if not args.run:
        print(json.dumps(report, indent=2))
        return
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("output exists; use --resume with identical source/config/data identity")
    source = GuardedNativeDenseGF(args.dataset, config)
    report["preflight"] = preflight(source, config)
    start = time.perf_counter()
    identity["data"] = {
        name: {"stat": _file_identity(path), "sha256": sha256(path)}
        for name, path in (("matrix", source.mat_path), ("coordinates", source.coordinate_path))
    }
    hash_seconds = time.perf_counter() - start
    if identity["data"]["coordinates"]["sha256"] != config["expected_data"]["coordinate_sha256"]:
        raise ValueError("coordinate hash differs from frozen dataset")
    pilot = args.dataset / config["calibration"]["path"]
    report["pilot_identity"] = {"stat": _file_identity(pilot), "sha256": sha256(pilot)}
    report["hashing"] = {
        "seconds": hash_seconds,
        "matrix_scans": 1,
        "bytes": source.mat_path.stat().st_size + source.coordinate_path.stat().st_size,
    }
    references = {}
    for dataset in ("bp3", "bp7"):
        path = ROOT / f"benchmarks/results/{dataset}_fixed.json"
        references[dataset] = json.loads(path.read_text())
        identity[dataset + "_record_sha256"] = sha256(path)
    aggregate = output / "frontera_fixed.json"
    if aggregate.exists() and json.loads(aggregate.read_text())["identity"] != identity:
        raise ValueError("resume identity mismatch; use a new output directory")
    report["status"] = "running"
    checkpoint(aggregate, report)
    for seed in config["construction_seeds"]:
        path = output / f"seed_{seed}.json"
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing and existing["identity"] != identity:
            raise ValueError("seed checkpoint identity mismatch")
        if existing and existing["status"] == "completed":
            run = existing["run"]
        else:
            print(f"Frontera seed {seed}: starting", flush=True)
            try:
                run, peak = isolated_peak_rss(
                    execute_seed, args.dataset, config, seed, path, identity, hash_seconds
                )
                run["storage"]["peak_rss_bytes"] = peak
                seed_record = json.loads(path.read_text())
                seed_record["run"] = run
                checkpoint(path, seed_record)
            except Exception as exc:
                report.update(status="interrupted", error=str(exc))
                checkpoint(aggregate, report)
                raise
        report["runs"].append(run)
        checkpoint(aggregate, report)
    report["all_pass"] = all(run["all_pass"] for run in report["runs"])
    report["status"] = "passed" if report["all_pass"] else "failed_gates"
    report["acceptance_contract"] = acceptance_contract(
        references["bp3"], report, references["bp7"]
    )
    checkpoint(aggregate, report)
    print(aggregate, flush=True)


if __name__ == "__main__":
    main()
