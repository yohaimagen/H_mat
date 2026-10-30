"""C.16 opt-in paired 192-patch regression; never a full-matrix experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.c10 import _acceptance_evaluation, _file_identity, _load_config

from gfcompress.benchmark import environment, write_report
from gfcompress.compress import CountingOperator, compress
from gfcompress.error import relative_error
from gfcompress.geometry import FaultMesh
from gfcompress.operators import DenseOperator
from gfcompress.realgf import RealGF, representative_subset
from gfcompress.study import _diagnostics, validate_seed_plan


def _responses(diagnostic: dict[str, Any]) -> dict[str, Any]:
    values = dict(diagnostic["selected_block_output_errors"])
    for side in ("forward", "adjoint"):
        values.update(
            {
                f"{side}/component/{key}": value
                for key, value in diagnostic[side]["component_index_group_errors"].items()
            }
        )
        values.update(
            {
                f"{side}/input/{key}": value["compressed"]
                for key, value in diagnostic[side]["input_errors"].items()
            }
        )
    return values


def paired_comparison(
    fixed: dict[str, Any], colored: dict[str, Any], floor: float, slack: float
) -> dict[str, Any]:
    """Predeclared C.16 comparison; absolute acceptance is checked separately."""
    f, c = _responses(fixed), _responses(colored)
    if f.keys() != c.keys():
        raise ValueError("paired diagnostics must use identical response probes")
    responses = {
        key: {
            "fixed_absolute": f[key]["absolute"],
            "colored_absolute": c[key]["absolute"],
            "limit": (1 + slack) * max(f[key]["absolute"], floor),
        }
        for key in f
    }
    for value in responses.values():
        value["pass"] = value["colored_absolute"] <= value["limit"]
    limit = float((1 + slack) * fixed["global_relative_error"] + 64 * np.finfo(float).eps)
    return {
        "responses": responses,
        "global": {
            "fixed": fixed["global_relative_error"],
            "colored": colored["global_relative_error"],
            "limit": limit,
            "pass": colored["global_relative_error"] <= limit,
        },
        "all_pass": all(value["pass"] for value in responses.values())
        and colored["global_relative_error"] <= limit,
    }


def paired_subset(reference, mesh, dense, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every configured seed/start at one locked operating point."""
    validate_seed_plan(config["construction_seeds"], config["validation_seeds"])
    parameters = {name: config["selection"]["primary"][name] for name in ("m", "k", "p")}
    floor = config["acceptance"]["absolute_response_floor"]
    iterations = config["validation"]["power_iterations"]
    raw: dict[str, list[dict[str, Any]]] = {"fixed": [], "coloring": []}
    runs, comparisons = [], []
    permutation = np.random.default_rng(701).permutation(mesh.n_patches)
    rows, cols = mesh.patch_to_rows(permutation), mesh.patch_to_cols(permutation)
    perm_mesh = FaultMesh(mesh.centroids[permutation], mesh.L[permutation], dof_row=mesh.dof_row)
    perm_reference = DenseOperator(dense[np.ix_(rows, cols)])
    for seed in config["construction_seeds"]:
        seed_diagnostics = {}
        for strategy in ("fixed", "coloring"):
            counted = CountingOperator(reference)
            hmat = compress(counted, mesh, **parameters, seed=seed, sampling=strategy)
            construction = asdict(counted.snapshot())
            coverage = hmat.block_partition()
            pairs = set().union(*coverage)
            complete = len(pairs) == mesh.n_patches**2 == sum(map(len, coverage))
            rng = np.random.default_rng(709)
            x, y = rng.standard_normal(mesh.n_cols), rng.standard_normal(mesh.n_rows)
            lhs, rhs = float(y @ hmat.matvec(x)), float(x @ hmat.rmatvec(y))
            adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
            leaf_errors = []
            leaf_details = []
            for leaf in hmat.leaves:
                exact = dense[np.ix_(leaf.alpha.row_indices, leaf.beta.col_indices)]
                absolute = float(np.linalg.norm(leaf.block - exact))
                norm = float(np.linalg.norm(exact))
                leaf_errors.append(absolute / max(norm, np.finfo(float).tiny))
                leaf_details.append(
                    {
                        "alpha_cell": leaf.alpha.cell_coords,
                        "beta_cell": leaf.beta.cell_coords,
                        "row_patches": leaf.alpha.patch_indices.tolist(),
                        "col_patches": leaf.beta.patch_indices.tolist(),
                        "absolute_error": absolute,
                        "reference_norm": norm,
                        "relative_error": leaf_errors[-1],
                    }
                )
            permuted = compress(
                perm_reference, perm_mesh, **parameters, seed=seed, sampling=strategy
            )
            permutation_error = relative_error(
                permuted, perm_reference, seed=101, n_iters=iterations
            )
            diagnostics = []
            for start in config["validation_seeds"]:
                diag = _diagnostics(hmat, reference, mesh, start, floor, iterations)
                raw[strategy].append({"parameters": parameters, "diagnostics": diag})
                diagnostics.append(diag)
            seed_diagnostics[strategy] = diagnostics
            runs.append(
                {
                    "seed": seed,
                    "sampling": strategy,
                    "construction": construction,
                    "sampling_report": hmat.sampling_report,
                    "coverage_complete_disjoint": complete,
                    "adjoint_relative_discrepancy": adjoint_error,
                    "permuted_global_error": permutation_error,
                    "near_field_relative_error_max": max(leaf_errors),
                    "worst_near_field_block": leaf_details[int(np.argmax(leaf_errors))],
                    "global_errors": [d["global_relative_error"] for d in diagnostics],
                    "structural_and_near_field_pass": complete
                    and adjoint_error < 1e-12
                    and max(leaf_errors) < 5e-2
                    and permutation_error <= config["acceptance"]["global_relative_error_max"],
                }
            )
        for start, f, c in zip(
            config["validation_seeds"],
            seed_diagnostics["fixed"],
            seed_diagnostics["coloring"],
            strict=True,
        ):
            comparisons.append(
                {
                    "seed": seed,
                    "validation_seed": start,
                    **paired_comparison(
                        f, c, floor, config["acceptance"]["comparison_relative_slack"]
                    ),
                }
            )
    acceptance = {
        strategy: _acceptance_evaluation(config, records) for strategy, records in raw.items()
    }
    within_budget = all(
        p["selected_columns"] <= p["fixed_columns"]
        for run in runs
        for p in run["sampling_report"]["passes"]
    )
    strict_reduction = any(
        p["selected_columns"] < p["fixed_columns"]
        for run in runs
        for p in run["sampling_report"]["passes"]
    )
    passed = (
        all(value["all_pass"] for value in acceptance.values())
        and all(c["all_pass"] for c in comparisons)
        and all(r["structural_and_near_field_pass"] for r in runs)
        and within_budget
    )
    return {
        "status": "passed" if passed else "failed",
        "all_pass": passed,
        "parameters": parameters,
        "acceptance": acceptance,
        "runs": runs,
        "paired_comparisons": comparisons,
        "selected_columns_within_fixed": within_budget,
        "strict_reduction": strict_reduction,
        "comparison_contract": {
            "response_errors": (
                "colored_absolute <= (1+slack)*max(fixed_absolute, absolute_response_floor)"
            ),
            "global": "colored_global <= (1+slack)*fixed_global + 64*float64_eps",
            "near_field_relative_error_max": 0.05,
            "absolute_response_floor": floor,
            "slack": config["acceptance"]["comparison_relative_slack"],
        },
    }


def report(data_root: Path, dataset: str, revision: str, run: bool) -> dict[str, Any]:
    config_path = Path(f"benchmarks/configs/{dataset}_fixed.json")
    config = _load_config(dataset, config_path)
    directory = data_root / f"gf_{dataset}"
    matrix, coords = directory / "gf_mat.bin", sorted(directory.glob("*_fbf_coords.csv"))
    paths = [
        config_path,
        Path(__file__),
        Path("benchmarks/c10.py"),
        *sorted(Path("src/gfcompress").glob("*.py")),
    ]
    common = {
        "schema_version": 1,
        "dataset": dataset,
        "revision": revision,
        "revision_meaning": (
            "supplied committed base; source hashes bind the working implementation "
            "used for this run"
        ),
        "environment": environment(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "config": config,
        "subset_limit": 192,
        "regions": 4,
    }
    if not matrix.is_file() or len(coords) != 1:
        return {
            **common,
            "status": "blocked",
            "reason": "missing/ambiguous BP matrix or coordinates",
        }
    if not run:
        return {**common, "status": "not_run", "reason": "bounded subset runs require --run"}
    source = RealGF(matrix, coords[0])
    subset = representative_subset(source, max_patches=192, regions=4)
    dense = np.ascontiguousarray(subset.matvec(np.eye(subset.shape[1])))
    result = paired_subset(DenseOperator(dense), subset.mesh, dense, config)
    return {
        **common,
        **result,
        "matrix": _file_identity(matrix),
        "coordinates": _file_identity(coords[0]),
        "coordinates_sha256": hashlib.sha256(coords[0].read_bytes()).hexdigest(),
        "subset": {
            "patch_ids": subset.patch_ids.tolist(),
            "shape": list(subset.shape),
            "row_map_sha256": hashlib.sha256(subset.row_map.tobytes()).hexdigest(),
            "col_map_sha256": hashlib.sha256(subset.col_map.tobytes()).hexdigest(),
            "native_dense_sha256": hashlib.sha256(dense.tobytes()).hexdigest(),
        },
        "limitations": (
            "192-patch native-endian reference only; no full-matrix, RSS, or speedup claim. "
            "C.15 rank-4 negative case remains valid."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bp3", "bp7"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("real_gfs"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = report(args.data_root, args.dataset, args.revision, args.run)
    value["command"] = [sys.executable, "-m", "benchmarks.coloring", *sys.argv[1:]]
    write_report(args.output, value)
    print(json.dumps({"status": value["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
