#!/usr/bin/env python3
"""Opt-in C.11 full fixed baselines; invoke one dataset per process."""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.c10 import _acceptance_evaluation, _file_identity, _load_config

from gfcompress.benchmark import (
    _add_counts,
    environment,
    predict_fixed_products,
    representation_storage,
    time_products,
    write_report,
)
from gfcompress.build_tree import build_tree
from gfcompress.compress import CountingOperator, ProductCounts, compress
from gfcompress.geometry import FaultMesh
from gfcompress.operators import MatVecOperator
from gfcompress.realgf import RealGF
from gfcompress.study import _diagnostics, validate_seed_plan


def measure_seed(
    reference: MatVecOperator,
    mesh: FaultMesh,
    config: dict[str, Any],
    seed: int,
    preparation_seconds: float,
) -> dict[str, Any]:
    """Construct once, snapshot costs, then run every frozen validation start."""
    parameters = {name: config["selection"]["primary"][name] for name in ("m", "k", "p")}
    operator = CountingOperator(reference)
    start = time.perf_counter()
    hmat = compress(operator, mesh, **parameters, seed=seed, sampling="fixed")
    construction_seconds = time.perf_counter() - start
    observed = operator.reset()
    predicted = predict_fixed_products(
        hmat.root, mesh, parameters["k"], parameters["p"], ProductCounts(0, 0, 0, 0)
    )
    expected = _add_counts(predicted.construction, predicted.leaves)
    records = [
        {
            "parameters": parameters,
            "construction_seed": seed,
            "validation_seed": validation_seed,
            "diagnostics": _diagnostics(
                hmat,
                operator,
                mesh,
                validation_seed,
                config["validation"]["absolute_response_floor"],
                config["validation"]["power_iterations"],
            ),
        }
        for validation_seed in config["validation_seeds"]
    ]
    validation = operator.reset()
    timing: dict[str, Any] = {
        "operator_preparation": preparation_seconds,
        "construction": construction_seconds,
        "total_setup": preparation_seconds + construction_seconds,
    }
    for side, transpose in (("forward", False), ("adjoint", True)):
        compressed = time_products(hmat, transpose=transpose, rhs_columns=1, repeats=5, seed=101)
        dense = time_products(reference, transpose=transpose, rhs_columns=1, repeats=5, seed=101)
        timing[side] = {
            "compressed": compressed,
            "dense": dense,
            "setup_break_even_applies": (
                timing["total_setup"] / (dense - compressed) if dense > compressed else None
            ),
        }
    dense_entries = mesh.n_rows * mesh.n_cols
    storage = asdict(representation_storage(hmat, dense_reference_bytes=dense_entries * 8))
    storage["numerical_entries"] = storage["numerical_bytes"] // 8
    storage["dense_entries"] = dense_entries
    acceptance = _acceptance_evaluation(config, records)
    gates = {
        "frozen_accuracy": acceptance.get("all_pass", False),
        "numerical_entries_below_dense": storage["numerical_entries"] < dense_entries,
        "construction_columns_below_n_cols": observed.matvec_columns + observed.rmatvec_columns
        < mesh.n_cols,
        "predicted_equals_observed": expected == observed,
    }
    levels = list(hmat.root.iter_levels())
    return {
        "construction_seed": seed,
        "records": records,
        "acceptance": acceptance,
        "gates": gates,
        "all_pass": all(gates.values()),
        "storage": storage,
        "timing_seconds": timing,
        "products": {
            "predicted_admissible": asdict(predicted.construction),
            "predicted_leaves": asdict(predicted.leaves),
            "predicted_total": asdict(expected),
            "observed_total": asdict(observed),
            "validation_excluded": asdict(validation),
        },
        "tree": {
            "dimension": mesh.tree_dim,
            "depth": len(levels) - 1,
            "leaf_count": len(levels[-1]),
            "oversized_leaves": sum(len(n.patch_indices) > parameters["m"] for n in levels[-1]),
        },
    }


def prepare(source: RealGF) -> None:
    """Benchmark-only native endian preparation, retaining file-order mappings."""
    native = np.empty(source.shape, dtype=np.float64)
    for start in range(0, source.shape[0], 128):
        native[start : start + 128] = source.mat[start : start + 128]
    source.mat = native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bp3", "bp7"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("real_gfs"))
    parser.add_argument("--revision", required=True, help="orchestrator-supplied source revision")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, help="explicit full-run configuration revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config or Path(f"benchmarks/configs/{args.dataset}_fixed.json")
    config = _load_config(args.dataset, config_path)
    validate_seed_plan(config["construction_seeds"], config["validation_seeds"])
    if not config["acceptance"]["frozen"] or config["selection"]["primary"] is None:
        raise ValueError("full baselines require a frozen primary configuration")
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": args.dataset,
        "config": config,
        "provenance": {
            "revision_supplied_by_orchestrator": args.revision,
            "command": [sys.executable, *sys.argv],
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path("benchmarks/c10.py"),
                    *sorted(Path("src/gfcompress").glob("*.py")),
                ]
            },
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "config_path": str(config_path),
        },
        "environment": {
            **environment(),
            "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS", "unset"),
        },
        "status": "not_run",
        "runs": [],
        "timing_protocol": {
            "rhs_columns": 1,
            "repeats": 5,
            "warmups": 1,
            "dense": "same native-endian file-order operator and permutations",
            "setup": "preparation charged to each independently usable seed",
        },
        "limitations": [
            "RSS is process lifetime high-water including reference and preparation",
            "retained allocation bytes exclude Python object overhead",
            "row component layout remains the R.1 inference",
        ],
    }
    output = args.output or Path(f"benchmarks/results/{args.dataset}_fixed.json")
    if args.run:
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            if os.environ.get(name) != "1":
                raise ValueError(f"{name}=1 is required for resource-safe full runs")
        directory = args.data_root / f"gf_{args.dataset}"
        matrix = directory / "gf_mat.bin"
        (coordinates,) = directory.glob("*_fbf_coords.csv")
        report["source"] = {
            "matrix": _file_identity(matrix),
            "coordinates": _file_identity(coordinates),
        }
        start = time.perf_counter()
        source = RealGF(matrix, coordinates)
        primary = config["selection"]["primary"]
        budget = predict_fixed_products(
            build_tree(source.mesh, primary["m"]),
            source.mesh,
            primary["k"],
            primary["p"],
            ProductCounts(0, 0, 0, 0),
        ).total_budget
        report["preflight_construction"] = asdict(budget)
        if budget.matvec_columns + budget.rmatvec_columns >= source.shape[1]:
            raise ValueError("predicted construction columns must be strictly below n_cols")
        prepare(source)
        preparation = time.perf_counter() - start
        report["shape"] = list(source.shape)
        report["status"] = "running"
        write_report(output, report)
        for seed in config["construction_seeds"]:
            print(f"{args.dataset} seed {seed}: starting", flush=True)
            run = measure_seed(source, source.mesh, config, seed, preparation)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            run["storage"]["peak_rss_bytes"] = int(
                peak if sys.platform == "darwin" else peak * 1024
            )
            report["runs"].append(run)
            write_report(output, report)
            print(f"{args.dataset} seed {seed}: {run['gates']}", flush=True)
        report["all_pass"] = all(run["all_pass"] for run in report["runs"])
        report["status"] = "passed" if report["all_pass"] else "failed_gates"
    write_report(output, report)
    print(output, flush=True)


if __name__ == "__main__":
    main()
