#!/usr/bin/env python3
"""C.10 bounded fixed-sampling study; real runs are explicit opt-in."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gfcompress.benchmark import provenance, write_report
from gfcompress.realgf import RealGF, representative_subset
from gfcompress.study import configuration_schema, fixed_sweep


def report(data_root: Path, dataset: str, run: bool) -> dict[str, Any]:
    directory = data_root / f"gf_{dataset}"
    matrix, coordinates = directory / "gf_mat.bin", sorted(directory.glob("*_fbf_coords.csv"))
    config = configuration_schema(dataset)
    common: dict[str, Any] = {"schema_version": 1, "provenance": provenance(), "config": config}
    if not matrix.is_file() or len(coordinates) != 1:
        return {
            **common,
            "status": "blocked",
            "reason": "C.10 requires the missing real gf_mat.bin and one coordinate file",
            "required": {
                "matrix": str(matrix),
                "coordinates_glob": str(directory / "*_fbf_coords.csv"),
            },
            "acceptance": "not evaluated; no metrics, operating point, or tolerance is fabricated",
        }
    if not run:
        return {
            **common,
            "status": "not_run",
            "reason": "bounded real sweep is opt-in; rerun with --run after checking resources",
            "acceptance": "not evaluated",
        }
    source = RealGF(matrix, coordinates[0])
    subset = representative_subset(source, max_patches=192, regions=4)
    sweep = config["sweep"]
    records = fixed_sweep(
        subset,
        subset.mesh,
        m_values=sweep["m"],
        k_values=sweep["k"],
        p_values=sweep["p"],
        construction_seeds=config["construction_seeds"],
        validation_seeds=config["validation_seeds"],
    )
    return {
        **common,
        "status": "measured_unlocked",
        "subset": {"patch_count": subset.mesh.n_patches, "regions": 4, "max_patches": 192},
        "records": records,
        "acceptance": "selection/tolerance locking remains a human-reviewed measurement step",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("real_gfs"))
    parser.add_argument("--dataset", choices=("bp3", "bp7"), required=True)
    parser.add_argument("--run", action="store_true", help="run the bounded real-data sweep")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"benchmarks/results/{args.dataset}_c10_sweep.json")
    write_report(output, report(args.data_root, args.dataset, args.run))
    print(output)


if __name__ == "__main__":
    main()
