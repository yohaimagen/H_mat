#!/usr/bin/env python3
"""Opt-in bounded Frontera block screen; never launches formal confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.diagnose import block_diagnostics

from gfcompress.build_tree import build_tree
from gfcompress.frontera import NativeDenseGF
from gfcompress.interactions import build_lists
from gfcompress.randomized import orth

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/Volumes/Extreme SSD/H_mat/frontera_np20k_3d")
DEFAULT_PLAN = ROOT / "benchmarks/configs/frontera_screen_v1.json"


def _pairs(root: Any, lists: Any, plan: dict[str, Any]) -> list[tuple[int, Any, Any]]:
    chosen: list[tuple[int, Any, Any]] = []
    target = plan["target"]
    nodes = root.nodes_at_level(target["level"])
    alpha, beta = nodes[target["alpha_index"]], nodes[target["beta_index"]]
    if beta not in lists.interaction[alpha]:
        raise ValueError("declared failed L4 pair is not an interaction")
    chosen.append((target["level"], alpha, beta))
    bound = plan["maximum_block_entries"]
    for level in plan["representative_levels"]:
        candidates = [
            (a, b)
            for a in root.nodes_at_level(level)
            for b in lists.interaction[a]
            if a.row_indices.size * b.col_indices.size <= bound
        ]
        if not candidates:
            raise ValueError(f"no bounded representative at level {level}")
        alpha, beta = max(
            candidates, key=lambda pair: pair[0].row_indices.size * pair[1].col_indices.size
        )
        chosen.append((level, alpha, beta))
    return chosen


def _record(block: np.ndarray, candidate: dict[str, int], seed: int, dof: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    width = candidate["k"] + candidate["p"]
    gb = rng.standard_normal((block.shape[1], width))
    ga = rng.standard_normal((block.shape[0], width))
    rank = min(candidate["k"], *block.shape)
    y = block @ gb
    u, v = orth(y, rank), orth(block.T @ ga, rank)
    return block_diagnostics(block, u, v, y, ga, gb, dof)


def promote(results: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, int] | None:
    comparator = results[0]
    eligible = [
        result
        for result in results[1:]
        if result["l4_all"] <= comparator["l4_all"] * policy["l4_all_ratio_max"]
        and result["representative_all"] <= comparator["representative_all"]
    ]
    return min(eligible, key=lambda result: result["l4_all"])["parameters"] if eligible else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.run:
        print(json.dumps({"status": "not_run", "plan": plan}, indent=2))
        return
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    output = (args.output_dir or args.dataset / "benchmark_results/frontera_screen_v1").resolve()
    if not output.is_relative_to(args.dataset.resolve()) or output.is_relative_to(ROOT):
        raise ValueError("output directory must be external to the repository")
    output.mkdir(parents=True, exist_ok=True)
    source = NativeDenseGF(
        args.dataset / "gf/gf_mat.bin", args.dataset / "gf_coords.bin", **plan["reader"]
    )
    root, lists = build_tree(source.mesh, plan["comparator"]["m"]), None
    lists = build_lists(root)
    pairs = _pairs(root, lists, plan)
    results = []
    for candidate in [plan["comparator"], *plan["candidates"]]:
        records = []
        for level, alpha, beta in pairs:
            block = source.read_block(alpha.row_indices, beta.col_indices)
            for seed in plan["screen_seeds"]:
                records.append(
                    {
                        "level": level,
                        "alpha_index": alpha.index_in_level,
                        "beta_index": beta.index_in_level,
                        "seed": seed,
                        **_record(block, candidate, seed, source.dof_row),
                    }
                )
        l4 = [r["errors"]["clean_two_sample"]["all"] for r in records if r["level"] == 4]
        representative = [
            r["errors"]["clean_two_sample"]["all"] for r in records if r["level"] != 4
        ]
        results.append(
            {
                "parameters": candidate,
                "records": records,
                "l4_all": max(l4),
                "representative_all": max(representative),
            }
        )
    report = {
        "schema_version": 1,
        "status": "screened",
        "revision_supplied_by_orchestrator": args.revision,
        "source_sha256": {
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "plan": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        },
        "plan": plan,
        "results": results,
        "promotion": promote(results, plan["promotion"]),
        "limitations": (
            "Exact bounded physical blocks with clean randomized reconstruction; no peeling, "
            "factors, dense matrix copy, or formal confirmation."
        ),
    }
    (output / "frontera_screen.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(output / "frontera_screen.json")


if __name__ == "__main__":
    main()
