#!/usr/bin/env python3
"""Bounded Frontera native-dense preflight; never starts full compression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import spearmanr

from gfcompress.benchmark import predict_fixed_products
from gfcompress.build_tree import build_tree
from gfcompress.compress import ProductCounts
from gfcompress.frontera import NativeDenseGF
from gfcompress.interactions import box_dist, build_lists
from gfcompress.tree import TreeNode

DEFAULT_DATASET = Path("/Volumes/Extreme SSD/H_mat/frontera_np20k_3d")
CANDIDATE = {"m": 512, "k": 47, "p": 10}
SAMPLE_PATCHES = 64
MAX_FULL_READ_BYTES = 1_000_000_000_000


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _identity(path: Path, *, digest: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if digest:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _relative(residual: np.ndarray, scale: np.ndarray) -> float:
    return float(np.linalg.norm(residual) / max(np.linalg.norm(scale), np.finfo(float).tiny))


def stream_check(operator: NativeDenseGF, seed: int = 20261012) -> dict[str, Any]:
    """Check both linearities and the adjoint in exactly two full passes."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(operator.shape[1])
    y = rng.standard_normal(operator.shape[1])
    u = rng.standard_normal(operator.shape[0])
    v = rng.standard_normal(operator.shape[0])
    omega = np.column_stack((x, y, x + y))
    psi = np.column_stack((u, v, u + v))

    start = time.perf_counter()
    forward = operator.matvec(omega)
    forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    adjoint = operator.rmatvec(psi)
    adjoint_seconds = time.perf_counter() - start
    matrix_bytes = operator.header.value_bytes
    left = float(np.vdot(forward[:, 0], u))
    right = float(np.vdot(x, adjoint[:, 0]))
    return {
        "full_passes": 2,
        "rhs_columns_per_pass": 3,
        "bytes_read": 2 * matrix_bytes,
        "forward_seconds": forward_seconds,
        "adjoint_seconds": adjoint_seconds,
        "forward_gb_per_second": matrix_bytes / forward_seconds / 1e9,
        "adjoint_gb_per_second": matrix_bytes / adjoint_seconds / 1e9,
        "forward_linearity_relative": _relative(
            forward[:, 2] - forward[:, 0] - forward[:, 1], forward[:, 2]
        ),
        "adjoint_linearity_relative": _relative(
            adjoint[:, 2] - adjoint[:, 0] - adjoint[:, 1], adjoint[:, 2]
        ),
        "adjoint_identity_relative": abs(left - right) / max(abs(left), abs(right), 1.0),
        "peak_rss_bytes": _peak_rss_bytes(),
        "maximum_converted_chunk_bytes": operator.maximum_chunk_bytes,
        "persistent_permutation_bytes": operator.permutation_bytes,
    }


def ordering_evidence(operator: NativeDenseGF) -> dict[str, Any]:
    """Record authoritative executable mapping and a spatial sanity check."""
    n = operator.mesh.n_patches
    probes = np.unique(np.linspace(0, n - 1, 5, dtype=np.intp))
    all_cols = np.arange(operator.shape[1], dtype=np.intp)
    correlations = []
    location_ranks = []
    for patch in probes:
        distance = np.linalg.norm(operator.coordinates - operator.coordinates[patch], axis=1)
        keep = distance > 0
        rows = operator.mesh.patch_to_rows(np.array([patch]))
        grouped = operator.read_block(rows, all_cols).reshape(3, n, 2)
        strength = np.linalg.norm(grouped, axis=(0, 2))
        correlations.append(float(spearmanr(distance[keep], strength[keep]).statistic))
        location_ranks.append(int(np.count_nonzero(strength > strength[patch])))
    return {
        "status": operator.ordering_status,
        "passes_gate": True,
        "authoritative_writer_mapping_available": True,
        "raw_order": "component-major within each nbf-element",
        "public_order": "patch-major",
        "basis_functions_per_element": operator.basis_functions_per_element,
        "mapping": "i=e*nbf+n; raw row=e*(3*nbf)+alpha*nbf+n; raw col=e*(2*nbf)+beta*nbf+n",
        "source_evidence": {
            "host": "hise_jump",
            "checkout": "/home/ymagen/tandem_h_matrix/tandem",
            "branch": "ymagen/strumpack-production",
            "commit": "c4eb1b4615b0",
            "function": "app/form/StrumpackGFFullOperator.cpp::build_index_maps",
            "basis": (
                "executable index-map and blocked-assembly code; "
                "native-dense and gf_coords features match dataset"
            ),
            "contradicted_comments": (
                "later dump_fault_node_coords comment and downloaded "
                "analyze_gf.py patch-major helpers"
            ),
        },
        "probe_patches": probes.tolist(),
        "median_spearman": float(np.median(correlations)),
        "same_location_rank_median_zero_based": float(np.median(location_ranks)),
        "reason": (
            "ordering resolved from executable Tandem source; "
            "spatial decay is only a sanity check"
        ),
    }


def _sample(values: np.ndarray, count: int) -> np.ndarray:
    if values.size <= count:
        return np.sort(values)
    positions = np.linspace(0, values.size - 1, count, dtype=np.intp)
    return np.sort(values)[positions]


def _rank_for_relative_tail(relative: np.ndarray, tolerance: float) -> int | None:
    """Smallest retained rank whose next singular ratio is at most tolerance."""
    below = np.flatnonzero(relative <= tolerance)
    return int(below[0]) if below.size else None


def _rank_summary(block: np.ndarray, candidate_rank: int) -> dict[str, Any]:
    singular = np.linalg.svd(block, compute_uv=False)
    relative = singular / singular[0]
    result: dict[str, Any] = {
        "shape": list(block.shape),
        "relative_sigma_at_candidate_rank": float(relative[min(candidate_rank, relative.size) - 1]),
        "rank_semantics": (
            "rank r retains sigma_1..sigma_r; the reported tail criterion uses "
            "sigma_(r+1)/sigma_1 <= tolerance, inclusive"
        ),
    }
    for tolerance in (1e-2, 1e-4, 1e-6):
        result[f"rank_for_relative_tail_at_most_{tolerance:g}"] = _rank_for_relative_tail(
            relative, tolerance
        )
    return result


def _pair_score(pair: tuple[TreeNode, TreeNode]) -> float:
    alpha, beta = pair
    return box_dist(alpha.bounding_box, beta.bounding_box) / max(alpha.diam, beta.diam)


def block_preflight(
    operator: NativeDenseGF, root: TreeNode, candidate_rank: int
) -> list[dict[str, Any]]:
    """Measure close/far interaction-list subsamples at every active level."""
    lists = build_lists(root)
    records = []
    for nodes in list(root.iter_levels())[2:]:
        pairs = [(alpha, beta) for alpha in nodes for beta in lists.interaction[alpha]]
        if not pairs:
            continue
        ranked = sorted(pairs, key=_pair_score)
        selected = [("closest", ranked[0]), ("farthest", ranked[-1])]
        for stratum, (alpha, beta) in selected:
            row_patches = _sample(alpha.patch_indices, SAMPLE_PATCHES)
            col_patches = _sample(beta.patch_indices, SAMPLE_PATCHES)
            block = operator.read_block(
                operator.mesh.patch_to_rows(row_patches),
                operator.mesh.patch_to_cols(col_patches),
            )
            records.append(
                {
                    "level": alpha.level,
                    "stratum": stratum,
                    "separation_over_diameter": _pair_score((alpha, beta)),
                    "source_block_patch_shape": [
                        int(alpha.patch_indices.size),
                        int(beta.patch_indices.size),
                    ],
                    "sample_patch_shape": [int(row_patches.size), int(col_patches.size)],
                    **_rank_summary(block, candidate_rank),
                }
            )
    return records


def _known_live_product_workspace_bytes(
    shape: tuple[int, int], width: int, chunk_bytes: int, *, transpose: bool
) -> int:
    """Known simultaneous probe, full, truncated, and residual arrays."""
    rows, cols = shape
    input_rows, output_rows = (rows, cols) if transpose else (cols, rows)
    # Conservatively include raw input and raw output during output gathering.
    return 8 * width * (2 * input_rows + 4 * output_rows) + chunk_bytes


def estimate_storage(root: TreeNode, k: int) -> dict[str, int]:
    """Upper-bound finalized factor and exact scheduled dense-leaf entries."""
    lists = build_lists(root)
    levels = list(root.iter_levels())
    factor_entries = 0
    factor_left_entries = 0
    factor_core_entries = 0
    factor_blocks = 0
    for nodes in levels[2:]:
        for alpha in nodes:
            for beta in lists.interaction[alpha]:
                rank = min(k, alpha.row_indices.size, beta.col_indices.size)
                factor_entries += rank * (alpha.row_indices.size + beta.col_indices.size)
                factor_left_entries += rank * alpha.row_indices.size
                factor_core_entries += rank * rank
                factor_blocks += 1
    leaf_entries = 0
    leaf_blocks = 0
    for alpha in levels[-1]:
        for beta in lists.nei[alpha]:
            leaf_entries += alpha.row_indices.size * beta.col_indices.size
            leaf_blocks += 1
    return {
        "factor_blocks": factor_blocks,
        "factor_entries_upper_bound": int(factor_entries),
        "factor_left_entries_upper_bound": int(factor_left_entries),
        "factor_core_entries_upper_bound": int(factor_core_entries),
        "leaf_blocks": leaf_blocks,
        "leaf_entries": int(leaf_entries),
        "total_entries_upper_bound": int(factor_entries + leaf_entries),
        "total_bytes_upper_bound": int(8 * (factor_entries + leaf_entries)),
    }


def construction_prediction(
    operator: NativeDenseGF,
    root: TreeNode,
    stream: dict[str, Any],
    candidate: dict[str, int] | None = None,
) -> dict[str, Any]:
    candidate = CANDIDATE if candidate is None else candidate
    products = predict_fixed_products(
        root,
        operator.mesh,
        candidate["k"],
        candidate["p"],
        ProductCounts(0, 0, 0, 0),
    )
    budget = products.total_budget
    calls = budget.matvec_calls + budget.rmatvec_calls
    columns = budget.matvec_columns + budget.rmatvec_columns
    matrix_bytes = operator.header.value_bytes
    storage = estimate_storage(root, candidate["k"])
    maximum_leaf_width = max(node.col_indices.size for node in list(root.iter_levels())[-1])
    admissible_width = candidate["k"] + candidate["p"]
    forward_workspace = _known_live_product_workspace_bytes(
        operator.shape,
        int(maximum_leaf_width),
        operator.maximum_chunk_bytes,
        transpose=False,
    )
    adjoint_workspace = _known_live_product_workspace_bytes(
        operator.shape,
        admissible_width,
        operator.maximum_chunk_bytes,
        transpose=True,
    )
    peeling_factor_bytes = 8 * (
        storage["factor_entries_upper_bound"] + storage["factor_core_entries_upper_bound"]
    )
    leaf_bytes = 8 * storage["leaf_entries"]
    leaf_extraction_peak = (
        peeling_factor_bytes + leaf_bytes + forward_workspace + operator.permutation_bytes
    )
    finalization_peak = (
        peeling_factor_bytes
        + leaf_bytes
        + 8 * storage["factor_left_entries_upper_bound"]
        + operator.permutation_bytes
    )
    io_floor_seconds = (
        budget.matvec_calls * stream["forward_seconds"]
        + budget.rmatvec_calls * stream["adjoint_seconds"]
    )
    linear_rhs_seconds = (
        budget.matvec_columns * stream["forward_seconds"]
        + budget.rmatvec_columns * stream["adjoint_seconds"]
    ) / stream["rhs_columns_per_pass"]
    return {
        "candidate": candidate,
        "products": asdict(budget),
        "combined_construction_columns": columns,
        "combined_columns_below_n_cols": columns < operator.shape[1],
        "full_matrix_passes": calls,
        "bytes_read": calls * matrix_bytes,
        "io_floor_seconds_from_width3_products": io_floor_seconds,
        "deliberately_pessimistic_linear_rhs_upper_seconds": linear_rhs_seconds,
        "linear_rhs_upper_method": (
            "charges one third of the measured width-3 pass time to every scheduled column; "
            "this ignores wider-BLAS efficiency and is not a compute-time estimate"
        ),
        "flop_estimate": 2 * operator.header.rows * operator.header.cols * columns,
        "storage": storage,
        "peak_memory_model": {
            "maximum_leaf_probe_columns": int(maximum_leaf_width),
            "maximum_admissible_probe_columns": admissible_width,
            "maximum_forward_workspace_bytes": int(forward_workspace),
            "maximum_adjoint_workspace_bytes": int(adjoint_workspace),
            "persistent_permutation_bytes": operator.permutation_bytes,
            "modeled_known_live_array_bytes": int(max(leaf_extraction_peak, finalization_peak)),
            "included_lifetimes": (
                "probe plus full, truncated, and subtraction-result arrays during peeling; "
                "raw permuted input/output buffers and persistent index maps; "
                "retained three-factor blocks and leaves; finalization overlap"
            ),
            "excludes": (
                "Python/BLAS overhead, allocator fragmentation, and unmodeled LAPACK "
                "workspace; not a measured or strict upper-bound RSS"
            ),
        },
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS", "unset"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(args: argparse.Namespace) -> dict[str, Any]:
    sources = [
        Path(__file__),
        Path("src/gfcompress/frontera.py"),
        Path("src/gfcompress/benchmark.py"),
        Path("src/gfcompress/build_tree.py"),
        Path("src/gfcompress/interactions.py"),
    ]
    return {
        "revision_label": args.revision,
        "command": [sys.executable, *sys.argv],
        "arguments": {
            "dataset": str(args.dataset),
            "output": str(args.output),
            "chunk_rows": args.chunk_rows,
            "run": args.run,
        },
        "source_sha256": {str(path): _sha256(path) for path in sources},
        "dirty_state": "not collected: this task prohibited Git commands",
    }


def full_compression_gate(ordering: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    reasons = [
        "no frozen Frontera operator-wide accuracy tolerance or independent validation plan",
        "physical block subsamples cannot certify the candidate rank for every full block",
    ]
    if ordering["status"] != "resolved_authoritative":
        reasons.append("physical row/column ordering is unresolved")
    if prediction["bytes_read"] > MAX_FULL_READ_BYTES:
        reasons.append(f"predicted construction reads exceed {MAX_FULL_READ_BYTES} bytes")
    return {
        "status": "gated_not_run",
        "reasons": reasons,
        "does_not_fix_bp7": True,
        "does_not_complete_c11": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/frontera_np20k_preflight.json")
    )
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--revision", default="not-supplied")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    matrix = args.dataset / "gf" / "gf_mat.bin"
    coordinates = args.dataset / "gf_coords.bin"
    report: dict[str, Any] = {
        "schema_version": 3,
        "status": "not_run",
        "purpose": "bounded Frontera feasibility evidence; not a C.11 baseline",
        "environment": _environment(),
        "provenance": _provenance(args),
        "source": {
            "matrix": _identity(matrix),
            "coordinates": _identity(coordinates, digest=True),
            "metadata": {
                name: _identity(args.dataset / name, digest=True)
                for name in ("np20k_summary.json", "f400x100_gen.toml", "f400x100_bench.toml")
            },
        },
        "limitations": [
            "physical block subsamples do not certify all full-block ranks",
            "no frozen Frontera operator-wide tolerance or validation plan exists",
            "the linear-RHS wall time is a deliberately pessimistic upper extrapolation",
        ],
    }
    if not args.run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(args.output)
        return

    operator = NativeDenseGF(matrix, coordinates, chunk_rows=args.chunk_rows)
    report["status"] = "running"
    report["header"] = asdict(operator.header)
    report["coordinates"] = {
        "shape": list(operator.coordinates.shape),
        "minimum": operator.coordinates.min(axis=0).tolist(),
        "maximum": operator.coordinates.max(axis=0).tolist(),
        "unique": int(np.unique(operator.coordinates, axis=0).shape[0]),
        "pca_variance_ratio": operator.alignment.variance_ratio.tolist(),
        "tree_dimension": operator.mesh.tree_dim,
        "patch_length_quantiles": np.quantile(operator.mesh.L, [0, 0.5, 1]).tolist(),
    }
    report["ordering"] = ordering_evidence(operator)
    report["streaming"] = stream_check(operator)
    root = build_tree(operator.mesh, CANDIDATE["m"])
    levels = list(root.iter_levels())
    report["tree"] = {
        "depth": len(levels) - 1,
        "occupied_boxes_by_level": [len(nodes) for nodes in levels],
        "leaf_patch_count_max": max(node.patch_indices.size for node in levels[-1]),
    }
    report["physical_block_preflight"] = {
        "status": "authoritative_ordering_bounded_subsamples",
        "logical_ordering": "patch_major",
        "used_for_gate": False,
        "records": block_preflight(operator, root, CANDIDATE["k"]),
    }
    report["prediction"] = construction_prediction(operator, root, report["streaming"])
    report["disk"] = {
        "free_bytes_before_exit": shutil.disk_usage(args.dataset).free,
        "large_outputs_written": False,
    }
    report["full_compression"] = full_compression_gate(report["ordering"], report["prediction"])
    report["status"] = "completed_preflight"
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        report["timing_history"] = [
            *previous.get("timing_history", []),
            {
                "record_sha256": _sha256(args.output),
                "ordering_status": previous.get("ordering", {}).get("status"),
                "streaming": previous.get("streaming"),
                "provenance": previous.get("provenance"),
                "derived_correction": previous.get("derived_correction"),
                "note": (
                    "historical measurement retained; current streaming is a "
                    "fresh corrected-order measurement"
                ),
            },
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
