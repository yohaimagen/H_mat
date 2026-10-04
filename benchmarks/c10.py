#!/usr/bin/env python3
"""C.10 bounded fixed-sampling study; real runs are explicit opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gfcompress.benchmark import provenance, write_report
from gfcompress.operators import DenseOperator
from gfcompress.realgf import RealGF, representative_subset
from gfcompress.study import fixed_sweep


def _load_config(dataset: str, config_path: Path | None) -> dict[str, Any]:
    path = config_path or Path(f"benchmarks/configs/{dataset}_fixed.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping) or not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"{path}: configuration must be a JSON object with string keys")
    config: dict[str, Any] = dict(loaded)
    if config.get("dataset") != dataset or config.get("sampling") != "fixed":
        raise ValueError(f"{path}: expected fixed configuration for {dataset}")
    required = (
        "sweep",
        "construction_seeds",
        "validation_seeds",
        "validation",
        "selection",
        "acceptance",
    )
    if any(
        not isinstance(config.get(name), Mapping)
        for name in ("sweep", "validation", "selection", "acceptance")
    ) or any(name not in config for name in required):
        raise ValueError(f"{path}: missing required C.10 configuration fields")
    for name in ("m", "k", "p"):
        if name not in config["sweep"]:
            raise ValueError(f"{path}: sweep.{name} is required")
    for name in ("power_iterations", "absolute_response_floor", "status"):
        if name not in config["validation"]:
            raise ValueError(f"{path}: validation.{name} is required")
    if not isinstance(config["acceptance"].get("frozen"), bool):
        raise ValueError(f"{path}: acceptance.frozen must be boolean")
    return config


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _array_hash(array: Any) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def report(
    data_root: Path,
    dataset: str,
    run: bool,
    config_path: Path | None = None,
    source_type: Any = RealGF,
    subset_builder: Any = representative_subset,
) -> dict[str, Any]:
    directory = data_root / f"gf_{dataset}"
    matrix, coordinates = directory / "gf_mat.bin", sorted(directory.glob("*_fbf_coords.csv"))
    config = _load_config(dataset, config_path)
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
    source = source_type(matrix, coordinates[0])
    subset = subset_builder(source, max_patches=192, regions=4)
    # C.7/C.8 permit this bounded native-endian subset reference.  It is
    # built once per report and supplied only for dense timing/storage baselines.
    dense = np.ascontiguousarray(subset.matvec(np.eye(subset.shape[1])), dtype=np.float64)
    dense_reference = DenseOperator(dense)
    sweep = config["sweep"]
    metadata: dict[str, Any] = {}
    records = fixed_sweep(
        subset,
        subset.mesh,
        m_values=sweep["m"],
        k_values=sweep["k"],
        p_values=sweep["p"],
        construction_seeds=config["construction_seeds"],
        validation_seeds=config["validation_seeds"],
        absolute_floor=config["validation"]["absolute_response_floor"],
        validation_iterations=config["validation"]["power_iterations"],
        dense_reference=dense_reference,
        metadata=metadata,
    )
    frozen = config["acceptance"]["frozen"]
    return {
        **common,
        "status": "measured_locked" if frozen else "measured_unlocked",
        **metadata,
        "source": {"matrix": _file_identity(matrix), "coordinates": _file_identity(coordinates[0])},
        "subset": {
            "patch_count": subset.mesh.n_patches,
            "regions": 4,
            "max_patches": 192,
            "patch_ids_hash": _array_hash(subset.patch_ids),
            "row_map_hash": _array_hash(subset.row_map),
            "col_map_hash": _array_hash(subset.col_map),
        },
        "records": records,
        "acceptance": (
            "frozen configuration was consumed; recorded thresholds still require review against "
            "the measured records"
            if frozen
            else "selection/tolerance locking remains a human-reviewed measurement step"
        ),
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
