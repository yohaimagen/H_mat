"""Predeclared C.11 clean-block screening; never substitutes for full validation."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from benchmarks.diagnose import block_diagnostics, diagnostic_provenance

from gfcompress.benchmark import environment, predict_fixed_products, write_report
from gfcompress.build_tree import build_tree
from gfcompress.compress import ProductCounts
from gfcompress.fixed_pattern import build_admissible_schedule
from gfcompress.interactions import build_lists
from gfcompress.randomized import orth
from gfcompress.realgf import RealGF


def select_promotions(results, policy):
    baseline = results[0]
    eligible = [
        r
        for r in results[1:]
        if r["component_worst"] <= baseline["component_worst"] * policy["component_worst_ratio_max"]
        and r["all_worst"] <= baseline["all_worst"] * policy["all_worst_ratio_max"]
    ]
    return [
        r["parameters"]
        for r in sorted(eligible, key=lambda r: r["component_worst"])[
            : policy["maximum_promotions"]
        ]
    ]


def construction_budget(root, mesh, width: int):
    """Return the combined fixed construction count at one sketch width."""
    budget = predict_fixed_products(root, mesh, width, 0, ProductCounts(0, 0, 0, 0))
    return budget, budget.total_budget.matvec_columns + budget.total_budget.rmatvec_columns


def assert_maximum_feasible_width(root, mesh, width: int, n_cols: int):
    """Require the declared width to be the last one under the strict budget."""
    budget, columns = construction_budget(root, mesh, width)
    _, next_columns = construction_budget(root, mesh, width + 1)
    if columns >= n_cols or next_columns < n_cols:
        raise ValueError("screen width must be the maximum strictly feasible construction width")
    return budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--plan", type=Path, default=Path("benchmarks/configs/bp7_screen.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/bp7_screen.json"))
    args = parser.parse_args()
    if not args.run:
        parser.error("real-data screening requires --run")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    plan = json.loads(args.plan.read_text())
    matrix = Path("real_gfs/gf_bp7/gf_mat.bin")
    coordinates = next(Path("real_gfs/gf_bp7").glob("*_fbf_coords.csv"))
    source = RealGF(matrix, coordinates)

    def selected_blocks(root, lists):
        blocks = []
        for level in range(2, len(list(root.iter_levels()))):
            pairs = [(a, b) for a in root.nodes_at_level(level) for b in lists.interaction[a]]
            selected = sorted(
                pairs, key=lambda pair: -len(pair[0].row_indices) * len(pair[1].col_indices)
            )[:1]
            if pairs and pairs[0] != selected[0]:
                selected.append(pairs[0])
            for a, b in selected:
                if len(a.row_indices) * len(b.col_indices) > 8_000_000:
                    raise ValueError("diagnostic block exceeds eight-million-entry bound")
                blocks.append(
                    (
                        level,
                        a,
                        b,
                        np.array(
                            source.mat[
                                np.ix_(
                                    source.row_pm_to_raw[a.row_indices],
                                    source.col_pm_to_raw[b.col_indices],
                                )
                            ],
                            dtype=float,
                        ),
                    )
                )
        return blocks

    report = {
        **diagnostic_provenance(args.revision, args.plan, matrix, coordinates),
        "environment": environment(),
        "plan": plan,
        "results": [],
        "limitations": (
            "Clean block-only sketches exclude peeling; screening is not full validation."
        ),
    }
    report["provenance"]["source_sha256"][str(Path(__file__))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    for parameters in [plan["incumbent"], *plan["candidates"]]:
        k, p = parameters["k"], parameters["p"]
        width = parameters["width"]
        if k + p != width:
            raise ValueError("candidate rank and oversampling must equal its declared width")
        root = build_tree(source.mesh, parameters["m"])
        lists = build_lists(root)
        budget = assert_maximum_feasible_width(root, source.mesh, width, source.shape[1])
        blocks = selected_blocks(root, lists)
        records = []
        for seed in plan["construction_seeds"]:
            for level, a, b, block in blocks:
                col = build_admissible_schedule(root, lists, level, source.mesh, k, p, seed, "col")
                row = build_admissible_schedule(root, lists, level, source.mesh, k, p, seed, "row")
                gb = col.owners[a, b].realize().blocks[b]
                ga = row.owners[a, b].realize().blocks[a]
                rank = min(k, *block.shape)
                y = block @ gb
                value = block_diagnostics(
                    block, orth(y, rank), orth(block.T @ ga, rank), y, ga, gb, source.dof_row
                )
                records.append(
                    {
                        "seed": seed,
                        "level": level,
                        "alpha": list(a.cell_coords),
                        "beta": list(b.cell_coords),
                        **value,
                    }
                )
        result = {
            "parameters": parameters,
            "predicted_full_construction": asdict(budget.total_budget),
            "records": records,
            "component_worst": max(r["errors"]["reconstructed"]["component0"] for r in records),
            "all_worst": max(r["errors"]["reconstructed"]["all"] for r in records),
        }
        report["results"].append(result)
        write_report(args.output, report)
        print(parameters, result["component_worst"], result["all_worst"], flush=True)
    report["promotions"] = select_promotions(report["results"], plan["promotion"])
    write_report(args.output, report)


if __name__ == "__main__":
    main()
