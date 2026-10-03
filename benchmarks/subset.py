#!/usr/bin/env python3
"""C.7 representative-subset benchmark, explicitly skipped without its data."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from gfcompress.benchmark import benchmark_operator, isolated_peak_rss, provenance, write_report
from gfcompress.operators import DenseOperator
from gfcompress.realgf import RealGF, representative_subset


def _identity(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def report(data_root: Path, dataset: str, regions: int, max_patches: int) -> dict[str, Any]:
    directory = data_root / f"gf_{dataset}"
    matrix = directory / "gf_mat.bin"
    csvs = sorted(directory.glob("*_fbf_coords.csv"))
    common: dict[str, Any] = {
        "schema_version": 1,
        "provenance": provenance(),
        "dataset": {"identity": dataset, "data_root": str(data_root)},
        "parameters": {"regions": regions, "max_patches": max_patches, "sampling": "fixed"},
    }
    if not matrix.is_file() or len(csvs) != 1:
        return {
            **common,
            "status": "skipped",
            "reason": "C.7 real matrix/coordinate files are absent or ambiguous",
            "required": {
                "matrix": str(matrix),
                "coordinates_glob": str(directory / "*_fbf_coords.csv"),
            },
        }
    load_start = time.perf_counter()
    source = RealGF(matrix, csvs[0])
    subset = representative_subset(source, max_patches=max_patches, regions=regions)
    # This is intentionally bounded to the selected subset, then made native endian.
    identity = np.eye(subset.shape[1])
    dense = np.ascontiguousarray(subset.matvec(identity), dtype=np.float64)
    measured = benchmark_operator(
        subset,
        subset.mesh,
        dataset=common["dataset"],
        m=12,
        k=4,
        p=2,
        loading_seconds=time.perf_counter() - load_start,
        dense_reference=DenseOperator(dense),
    )
    return {
        **measured,
        "status": "ready",
        "parameters": {
            **measured["parameters"],
            "regions": regions,
            "max_patches": max_patches,
        },
        "dataset": {
            **common["dataset"],
            "source": {
                "matrix": str(matrix),
                "matrix_size": matrix.stat().st_size,
                "matrix_mtime_ns": matrix.stat().st_mtime_ns,
                "coordinates": str(csvs[0]),
                "coordinates_size": csvs[0].stat().st_size,
                "coordinates_mtime_ns": csvs[0].stat().st_mtime_ns,
            },
        },
        "subset": {
            "patch_ids": subset.patch_ids.tolist(),
            "patch_ids_hash": _identity(subset.patch_ids),
            "row_map_hash": _identity(subset.row_map),
            "col_map_hash": _identity(subset.col_map),
            "shape": list(subset.shape),
            "dense_reference_bytes": int(dense.nbytes),
            "dense_reference": "native-endian, subset-only; same RHS/timing protocol required",
        },
        "limitations": (
            "Direct report() omits RSS; the CLI attaches comparable fresh-process dense and "
            "compressed peaks."
        ),
    }


def dense_only(data_root: Path, dataset: str, regions: int, max_patches: int) -> None:
    """Load and touch exactly the bounded native dense reference, without an H-matrix."""
    directory = data_root / f"gf_{dataset}"
    matrix = directory / "gf_mat.bin"
    csvs = sorted(directory.glob("*_fbf_coords.csv"))
    if not matrix.is_file() or len(csvs) != 1:
        return
    subset = representative_subset(
        RealGF(matrix, csvs[0]), max_patches=max_patches, regions=regions
    )
    dense = np.ascontiguousarray(subset.matvec(np.eye(subset.shape[1])), dtype=np.float64)
    DenseOperator(dense).matvec(np.ones(subset.shape[1]))


def with_rss(report_value: dict[str, Any], compressed_peak: int, dense_peak: int) -> dict[str, Any]:
    """Attach comparable fresh-process peaks to a completed ready report."""
    storage = report_value["storage"]
    storage["compressed_peak_rss_bytes"] = compressed_peak
    storage["dense_peak_rss_bytes"] = dense_peak
    storage["rss_ratio"] = compressed_peak / dense_peak if dense_peak else None
    return report_value


def _available(data_root: Path, dataset: str) -> bool:
    directory = data_root / f"gf_{dataset}"
    return (directory / "gf_mat.bin").is_file() and len(
        list(directory.glob("*_fbf_coords.csv"))
    ) == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("real_gfs"))
    parser.add_argument("--dataset", choices=("bp3", "bp7"), required=True)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--max-patches", type=int, default=192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"benchmarks/results/{args.dataset}_subset.json")
    if _available(args.data_root, args.dataset):
        measured, compressed_peak = isolated_peak_rss(
            report, args.data_root, args.dataset, args.regions, args.max_patches
        )
        _, dense_peak = isolated_peak_rss(
            dense_only, args.data_root, args.dataset, args.regions, args.max_patches
        )
        value = with_rss(measured, compressed_peak, dense_peak)
    else:
        value = report(args.data_root, args.dataset, args.regions, args.max_patches)
    write_report(output, value)
    print(output)


if __name__ == "__main__":
    main()
