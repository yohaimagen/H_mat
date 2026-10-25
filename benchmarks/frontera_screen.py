#!/usr/bin/env python3
"""Opt-in bounded physical-block screen and content-bound evidence validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.c10 import _file_identity
from benchmarks.diagnose import block_diagnostics
from benchmarks.frontera import _peak_rss_bytes

from gfcompress.build_tree import build_tree
from gfcompress.frontera import NativeDenseGF
from gfcompress.interactions import build_lists
from gfcompress.randomized import orth

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/Volumes/Extreme SSD/H_mat/frontera_np20k_3d")
DEFAULT_PLAN = ROOT / "benchmarks/configs/frontera_screen_v1.json"
PROBE_POLICY = "PCG64 SeedSequence([seed,level,alpha,beta,side]); maximum width then prefixes"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def evidence_binding(dataset: Path, config: dict[str, Any], revision: str) -> dict[str, Any]:
    """Bind to reviewed code/config and dataset identity without scanning the matrix."""
    if not revision.strip():
        raise ValueError("reviewed revision is required")
    expected = config["expected_data"]
    matrix, coords = dataset / "gf/gf_mat.bin", dataset / "gf_coords.bin"
    if matrix.stat().st_size != expected["matrix_file_bytes"]:
        raise ValueError("matrix size differs from frozen dataset")
    if file_hash(coords) != expected["coordinate_sha256"]:
        raise ValueError("coordinate hash differs from frozen dataset")
    paths = [
        ROOT / "BP_COLORING_PLAN.md",
        *sorted((ROOT / "benchmarks").glob("*.py")),
        *sorted((ROOT / "src/gfcompress").glob("*.py")),
        DEFAULT_PLAN,
        ROOT / "benchmarks/configs/frontera_exploratory_v2.json",
        ROOT / "benchmarks/configs/frontera_full_fixed_v2.json",
    ]
    return {
        "revision": revision,
        "config": config,
        "config_sha256": digest(config),
        "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
        "matrix": _file_identity(matrix),
        "coordinates": {**_file_identity(coords), "sha256": file_hash(coords)},
        "expected_data": expected,
    }


def seal(report: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in report.items() if k != "content_sha256"}
    return {**payload, "content_sha256": digest(payload)}


def verify(report: dict[str, Any], kind: str, binding: dict[str, Any]) -> None:
    if (
        report.get("schema_version") != 2
        or report.get("kind") != kind
        or report.get("binding") != binding
        or report.get("content_sha256") != seal(report)["content_sha256"]
    ):
        raise ValueError("stale, tampered, or incompatible evidence")


def write_once(path: Path, report: dict[str, Any]) -> None:
    """Publish complete evidence atomically; even concurrent writers cannot overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".evidence-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(seal(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def check_resources(dataset: Path, plan: dict[str, Any]) -> None:
    if _peak_rss_bytes() > plan["guards"]["rss_limit_bytes"]:
        raise RuntimeError("screen RSS guard exceeded")
    if shutil.disk_usage(dataset).free < plan["guards"]["minimum_external_free_bytes"]:
        raise RuntimeError("screen free-disk guard exceeded")


def block_manifest(source: NativeDenseGF, plan: dict[str, Any]) -> list[dict[str, Any]]:
    if list(source.shape) != plan["expected_data"]["shape"]:
        raise ValueError("screen shape mismatch")
    root = build_tree(source.mesh, plan["comparator"]["m"])
    levels = list(root.iter_levels())
    if [len(nodes) for nodes in levels] != plan["occupied_cells_by_level"]:
        raise ValueError("screen cell-count mismatch")
    lists = build_lists(root)
    manifest = []
    for spec in plan["blocks"]:
        alpha, beta = (
            levels[spec["level"]][spec["alpha_index"]],
            levels[spec["level"]][spec["beta_index"]],
        )
        rows, cols = alpha.row_indices, beta.col_indices
        shape = [int(rows.size), int(cols.size)]
        if beta not in lists.interaction[alpha] or min(shape) <= 128:
            raise ValueError("screen must use whole genuinely truncated interaction blocks")
        if shape != spec["shape"] or rows.size * cols.size > plan["maximum_block_entries"]:
            raise ValueError("screen block shape/entry guard")
        hashes = [array_hash(x.astype("<i8")) for x in (rows, cols)]
        if hashes != spec["index_sha256"]:
            raise ValueError("screen index hash mismatch")
        expected_bytes = int(rows.size * source.shape[1] * 8)
        if expected_bytes != spec["expected_read_bytes"]:
            raise ValueError("screen orientation/read-cost mismatch")
        manifest.append({**spec, "rows": rows.tolist(), "cols": cols.tolist()})
    total = sum(b["expected_read_bytes"] for b in manifest)
    if total > min(plan["guards"]["maximum_read_bytes"], 20_000_000_000):
        raise ValueError("screen diagnostic read cap exceeded")
    return manifest


def read_block_once(
    source: NativeDenseGF, spec: dict[str, Any], plan: dict[str, Any]
) -> tuple[np.ndarray, int]:
    """Read whole raw rows in declared orientation, counting actual returned bytes."""
    result = np.empty(spec["shape"], dtype=np.float64)
    cols = source.col_logical_to_raw[spec["cols"]]
    width = source.shape[1] * 8
    actual = 0
    with source.mat_path.open("rb") as stream:
        for i, row in enumerate(source.row_logical_to_raw[spec["rows"]]):
            check_resources(source.mat_path.parent.parent, plan)
            stream.seek(source.header.values_offset + int(row) * width)
            raw = stream.read(width)
            actual += len(raw)
            if len(raw) != width:
                raise OSError("short screen block read")
            result[i] = np.frombuffer(raw, dtype=">f8")[cols]
    if actual != spec["expected_read_bytes"]:
        raise RuntimeError("screen expected-vs-actual byte mismatch")
    return result, actual


def probes(spec: dict[str, Any], seed: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    identity = [seed, spec["level"], spec["alpha_index"], spec["beta_index"]]
    return tuple(  # type: ignore[return-value]
        np.random.default_rng(np.random.SeedSequence([*identity, side])).standard_normal((n, width))
        for side, n in enumerate(spec["shape"])
    )


def _record(
    block: np.ndarray, candidate: dict[str, int], ga: np.ndarray, gb: np.ndarray, dof: int
) -> dict[str, Any]:
    width = candidate["k"] + candidate["p"]
    ga, gb = ga[:, :width], gb[:, :width]
    rank = min(candidate["k"], *block.shape)
    y = block @ gb
    u, v = orth(y, rank), orth(block.T @ ga, rank)
    return {
        "probe_sha256": [array_hash(ga), array_hash(gb)],
        **block_diagnostics(block, u, v, y, ga, gb, dof),
    }


def promote(results: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, int] | None:
    comparator = results[0]
    for result in results[1:]:
        if (
            result["l4_all"] <= comparator["l4_all"] * policy["l4_all_ratio_max"]
            and result["representative_all"] <= comparator["representative_all"]
        ):
            return result["parameters"]
    return None


def validate_screen(
    report: dict[str, Any], binding: dict[str, Any], manifest: list[dict[str, Any]]
) -> dict[str, int] | None:
    verify(report, "screen", binding)
    plan = binding["config"]
    candidates = [plan["comparator"], *plan["candidates"]]
    width = max(c["k"] + c["p"] for c in candidates)
    if (
        report.get("manifest") != manifest
        or report.get("probe_policy") != PROBE_POLICY
        or report.get("actual_read_bytes") != sum(b["expected_read_bytes"] for b in manifest)
        or [r.get("parameters") for r in report.get("results", [])] != candidates
    ):
        raise ValueError("incomplete screen identities/results/read accounting")
    for result in report["results"]:
        expected = [(b, s) for b in manifest for s in plan["screen_seeds"]]
        if len(result["records"]) != len(expected):
            raise ValueError("incomplete screen seeds/blocks")
        for record, (spec, seed) in zip(result["records"], expected, strict=True):
            ga, gb = probes(spec, seed, width)
            w = result["parameters"]["k"] + result["parameters"]["p"]
            if (
                record["block"] != spec["name"]
                or record["seed"] != seed
                or record["probe_sha256"] != [array_hash(ga[:, :w]), array_hash(gb[:, :w])]
            ):
                raise ValueError("screen seed/block/probe mismatch")
            value = record["errors"]["clean_two_sample"]["all"]
            if not np.isfinite(value) or value < 0:
                raise ValueError("invalid screen error")
        for metric, name in (("l4_all", "failed_l4"), ("representative_all", "representative_l4")):
            worst = max(
                r["errors"]["clean_two_sample"]["all"]
                for r in result["records"]
                if r["block"] == name
            )
            if result[metric] != worst:
                raise ValueError("screen aggregate mismatch")
    promotion = promote(report["results"], plan["promotion"])
    if report.get("promotion") != promotion:
        raise ValueError("screen promotion mismatch")
    return promotion


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
    if plan != json.loads(DEFAULT_PLAN.read_text()):
        raise ValueError("screen requires the reviewed frozen plan")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    output = (args.output_dir or args.dataset / "benchmark_results/frontera_screen_v1").resolve()
    if not output.is_relative_to(args.dataset.resolve()) or output.is_relative_to(ROOT):
        raise ValueError("output directory must be external to the repository")
    target = output / "frontera_screen.json"
    if target.exists():
        raise FileExistsError(target)
    check_resources(args.dataset, plan)
    binding = evidence_binding(args.dataset, plan, args.revision)
    source = NativeDenseGF(
        args.dataset / "gf/gf_mat.bin", args.dataset / "gf_coords.bin", **plan["reader"]
    )
    manifest = block_manifest(source, plan)
    candidates = [plan["comparator"], *plan["candidates"]]
    width = max(c["k"] + c["p"] for c in candidates)
    results: list[dict[str, Any]] = [{"parameters": c, "records": []} for c in candidates]
    actual_bytes = 0
    for spec in manifest:
        block, actual = read_block_once(source, spec, plan)
        actual_bytes += actual
        for seed in plan["screen_seeds"]:
            ga, gb = probes(spec, seed, width)
            for result in results:
                check_resources(args.dataset, plan)
                result["records"].append(
                    {
                        "block": spec["name"],
                        "seed": seed,
                        **_record(block, result["parameters"], ga, gb, source.dof_row),
                    }
                )
                check_resources(args.dataset, plan)
    for result in results:
        for metric, name in (("l4_all", "failed_l4"), ("representative_all", "representative_l4")):
            result[metric] = max(
                r["errors"]["clean_two_sample"]["all"]
                for r in result["records"]
                if r["block"] == name
            )
    if evidence_binding(args.dataset, plan, args.revision) != binding:
        raise ValueError("dataset/source changed during screen")
    report = seal(
        {
            "schema_version": 2,
            "kind": "screen",
            "binding": binding,
            "manifest": manifest,
            "probe_policy": PROBE_POLICY,
            "actual_read_bytes": actual_bytes,
            "results": results,
            "promotion": promote(results, plan["promotion"]),
        }
    )
    validate_screen(report, binding, manifest)
    write_once(target, report)
    print(target)


if __name__ == "__main__":
    main()
