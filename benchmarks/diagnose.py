"""Bounded C.11 block diagnostics; no changes to compressor mathematics."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.c10 import _file_identity, _load_config
from benchmarks.full import prepare

from gfcompress.benchmark import environment, predict_fixed_products, write_report
from gfcompress.build_tree import build_tree
from gfcompress.column_basis import column_bases
from gfcompress.compress import ProductCounts
from gfcompress.interactions import build_lists
from gfcompress.randomized import core_matrix_solve, orth
from gfcompress.realgf import RealGF
from gfcompress.row_basis import core_matrices, row_bases


def block_diagnostics(a, u, v, y, ga, gb, dof: int) -> dict[str, Any]:
    """Compare actual samples against clean block-only samples and optimal SVD.

    All errors here are Frobenius-relative block diagnostics, not substitutes
    for the frozen operator-wide acceptance checks.
    """
    k = u.shape[1]
    left, s, right = np.linalg.svd(a, full_matrices=False)
    exact_y, exact_z = a @ gb, a.T @ ga
    clean_u, clean_v = orth(exact_y, k), orth(exact_z, k)
    projected = u @ (u.T @ a @ v) @ v.T
    actual = u @ core_matrix_solve(u, v, y, ga, gb) @ v.T
    exact_sample = u @ core_matrix_solve(u, v, exact_y, ga, gb) @ v.T
    clean = clean_u @ core_matrix_solve(clean_u, clean_v, exact_y, ga, gb) @ clean_v.T
    best = (left[:, :k] * s[:k]) @ right[:k]
    approximations = {
        "column_projection": u @ (u.T @ a),
        "row_projection": (a @ v) @ v.T,
        "two_sided_projection": projected,
        "reconstructed": actual,
        "exact_sample_same_bases": exact_sample,
        "clean_two_sample": clean,
        "optimal_rank_k": best,
    }
    norms = {"all": np.linalg.norm(a), "component0": np.linalg.norm(a[::dof])}
    errors = {
        name: {
            "all": float(np.linalg.norm(a - value) / norms["all"]),
            "component0": float(
                np.linalg.norm((a - value)[::dof]) / max(norms["component0"], np.finfo(float).tiny)
            ),
        }
        for name, value in approximations.items()
    }
    return {
        "shape": list(a.shape),
        "rank": k,
        "reference_frobenius": {name: float(value) for name, value in norms.items()},
        "singular_value_ratios": {
            str(i + 1): float(s[i] / s[0])
            for i in [0, k - 1, k, min(2 * k, len(s) - 1)]
            if i < len(s)
        },
        "component0_singular_tail": float(
            np.linalg.norm(np.linalg.svd(a[::dof], compute_uv=False)[k:])
            / max(norms["component0"], np.finfo(float).tiny)
        ),
        "errors": errors,
        "column_sample_contamination": float(np.linalg.norm(y - exact_y) / np.linalg.norm(exact_y)),
        "left_sketch_condition": float(np.linalg.cond(ga.T @ u)),
        "right_sketch_condition": float(np.linalg.cond(gb.T @ v)),
        "core_vs_projection_error_ratio": errors["reconstructed"]["all"]
        / max(errors["two_sided_projection"]["all"], np.finfo(float).tiny),
    }


def diagnostic_provenance(
    revision: str, config: Path, matrix: Path, coordinates: Path
) -> dict[str, Any]:
    """Identify the supplied revision, measured inputs, and executed helpers."""
    return {
        "provenance": {
            "revision_supplied_by_orchestrator": revision,
            "command": [sys.executable, *sys.argv],
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(__file__),
                    Path("benchmarks/c10.py"),
                    Path("benchmarks/full.py"),
                    config,
                    *sorted(Path("src/gfcompress").glob("*.py")),
                ]
            },
        },
        "source": {"matrix": _file_identity(matrix), "coordinates": _file_identity(coordinates)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--revision", required=True, help="orchestrator-supplied source revision")
    parser.add_argument(
        "--config", type=Path, default=Path("benchmarks/configs/bp7_full_fixed.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/bp7_block_diagnostics_v2.json")
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("bounded real diagnostics require --run")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    config = _load_config("bp7", args.config)
    parameters = config["selection"]["primary"]
    matrix = Path("real_gfs/gf_bp7/gf_mat.bin")
    coordinates = next(Path("real_gfs/gf_bp7").glob("*_fbf_coords.csv"))
    source = RealGF(matrix, coordinates)
    root = build_tree(source.mesh, parameters["m"])
    lists = build_lists(root)
    k, p = parameters["k"], parameters["p"]
    predicted = predict_fixed_products(root, source.mesh, k, p, ProductCounts(0, 0, 0, 0))
    if (
        predicted.total_budget.matvec_columns + predicted.total_budget.rmatvec_columns
        >= source.shape[1]
    ):
        raise ValueError("predicted construction columns must be strictly below n_cols")
    prepare(source)
    records, factors = [], []
    for level in range(2, len(list(root.iter_levels()))):
        cb = column_bases(source, root, lists, source.mesh, level, factors, k, p, seed=11)
        rb = row_bases(source, root, lists, source.mesh, level, factors, k, p, seed=11)
        rows = {(x.alpha, x.beta): x for x in rb}
        # Two deterministic representatives per level: largest and first pair.
        selected = sorted(cb, key=lambda x: -len(x.alpha.row_indices) * len(x.beta.col_indices))[:1]
        if cb and cb[0] is not selected[0]:
            selected.append(cb[0])
        for c in selected:
            r = rows[c.alpha, c.beta]
            # Bounded reference blocks are allowed for diagnostics, never in the compressor.
            if len(c.alpha.row_indices) * len(c.beta.col_indices) > 8_000_000:
                raise ValueError("diagnostic block exceeds the eight-million-entry bound")
            a = source.mat[
                np.ix_(
                    source.row_pm_to_raw[c.alpha.row_indices],
                    source.col_pm_to_raw[c.beta.col_indices],
                )
            ]
            record = block_diagnostics(a, c.u, r.v, c.y_alpha, r.g_alpha, c.g_beta, source.dof_row)
            record.update(
                {
                    "level": level,
                    "alpha": list(c.alpha.cell_coords),
                    "beta": list(c.beta.cell_coords),
                }
            )
            records.append(record)
            print(level, record["shape"], record["errors"], flush=True)
        factors += core_matrices(cb, rb)
    write_report(
        args.output,
        {
            **diagnostic_provenance(args.revision, args.config, matrix, coordinates),
            "environment": environment(),
            "config": config,
            "construction_seed": 11,
            "selection": "largest scalar-area and first interaction pair per level",
            "predicted_full_construction": asdict(predicted.total_budget),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
