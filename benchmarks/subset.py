#!/usr/bin/env python3
"""C.7 representative-subset benchmark, explicitly skipped without its data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from gfcompress.benchmark import provenance, write_report
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
    source = RealGF(matrix, csvs[0])
    subset = representative_subset(source, max_patches=max_patches, regions=regions)
    # This is intentionally bounded to the selected subset, then made native endian.
    identity = np.eye(subset.shape[1])
    dense = np.ascontiguousarray(subset.matvec(identity), dtype=np.float64)
    return {
        **common,
        "status": "ready",
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
            "This data-aware record establishes the bounded reference contract; run compression "
            "accounting with the shared helpers after selecting C.10 operating parameters."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("real_gfs"))
    parser.add_argument("--dataset", choices=("bp3", "bp7"), required=True)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--max-patches", type=int, default=192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"benchmarks/results/{args.dataset}_subset.json")
    write_report(output, report(args.data_root, args.dataset, args.regions, args.max_patches))
    print(output)


if __name__ == "__main__":
    main()
