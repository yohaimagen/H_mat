"""Bounded C.11 BP7 component/remediation screen; never runs a full matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.c10 import _file_identity
from benchmarks.diagnose import diagnostic_provenance

from gfcompress.benchmark import environment, write_report
from gfcompress.build_tree import build_tree
from gfcompress.fixed_pattern import build_admissible_schedule, build_leaf_schedule
from gfcompress.interactions import build_lists
from gfcompress.operators import MatVecOperator
from gfcompress.randomized import core_matrix_solve, orth
from gfcompress.realgf import RealGF, representative_subset


def _component_indices(size: int, dof: int, component: int) -> np.ndarray:
    return np.arange(component, size, dof, dtype=np.intp)


class ComponentScaledOperator(MatVecOperator):
    """Benchmark-only ``B = Dr A Dc^-1`` wrapper using only black-box products."""

    def __init__(self, operator: MatVecOperator, dr: np.ndarray, dc: np.ndarray) -> None:
        self.operator = operator
        self.dr = np.asarray(dr, dtype=float)
        self.dc = np.asarray(dc, dtype=float)
        if self.dr.shape != (operator.shape[0],) or self.dc.shape != (operator.shape[1],):
            raise ValueError("component scales must match the rectangular operator")
        if np.any(self.dr <= 0) or np.any(self.dc <= 0):
            raise ValueError("component scales must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.operator.shape

    def matvec(self, omega: np.ndarray) -> np.ndarray:
        if omega.ndim == 2:
            return self.dr[:, None] * self.operator.matvec(omega / self.dc[:, None])
        return self.dr * self.operator.matvec(omega / self.dc)

    def rmatvec(self, psi: np.ndarray) -> np.ndarray:
        if psi.ndim == 2:
            return self.operator.rmatvec(psi * self.dr[:, None]) / self.dc[:, None]
        return self.operator.rmatvec(psi * self.dr) / self.dc


def inverse_scaled(approximation: np.ndarray, dr: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Map an approximation of ``Dr A Dc^-1`` back to A's coordinates."""
    return approximation * dc[None, :] / dr[:, None]


def component_scales_from_probes(
    operator: MatVecOperator, dof_row: int, dof_col: int, probes: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate global component RMS scales with independent black-box probes."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((operator.shape[1], probes))
    y = operator.matvec(omega)
    row_rms = np.array([np.linalg.norm(y[component::dof_row]) for component in range(dof_row)])
    dr_component = np.exp(np.mean(np.log(row_rms))) / row_rms
    dr = np.tile(dr_component, operator.shape[0] // dof_row)
    psi = rng.standard_normal((operator.shape[0], probes)) * dr[:, None]
    z = operator.rmatvec(psi)
    col_rms = np.array([np.linalg.norm(z[component::dof_col]) for component in range(dof_col)])
    dc_component = col_rms / np.exp(np.mean(np.log(col_rms)))
    return np.tile(dr_component, operator.shape[0] // dof_row), np.tile(
        dc_component, operator.shape[1] // dof_col
    )


def component_scales_from_blocks(
    blocks: list[np.ndarray], dof_row: int, dof_col: int
) -> tuple[np.ndarray, np.ndarray]:
    """Entry-based calibration used only to quantify an oracle scaling ceiling."""
    row_energy = np.zeros(dof_row)
    col_energy = np.zeros(dof_col)
    for block in blocks:
        for component in range(dof_row):
            row_energy[component] += np.linalg.norm(block[component::dof_row]) ** 2
        for component in range(dof_col):
            col_energy[component] += np.linalg.norm(block[:, component::dof_col]) ** 2
    dr_component = np.exp(np.mean(np.log(np.sqrt(row_energy)))) / np.sqrt(row_energy)
    dc_component = np.sqrt(col_energy) / np.exp(np.mean(np.log(np.sqrt(col_energy))))
    return dr_component, dc_component


def _expanded(component_scale: np.ndarray, size: int) -> np.ndarray:
    return np.tile(component_scale, size // len(component_scale))


def _relative(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(
        np.linalg.norm(reference - approximation)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _component_errors(
    a: np.ndarray, approximation: np.ndarray, dof_row: int, dof_col: int
) -> dict[str, Any]:
    output = {
        str(component): _relative(a[component::dof_row], approximation[component::dof_row])
        for component in range(dof_row)
    }
    input_ = {
        str(component): _relative(a[:, component::dof_col], approximation[:, component::dof_col])
        for component in range(dof_col)
    }
    return {
        "all": _relative(a, approximation),
        "output": output,
        "input": input_,
        "worst_component": max(*output.values(), *input_.values()),
    }


def _spectrum(a: np.ndarray, rank_grid: list[int]) -> dict[str, Any]:
    values = np.linalg.svd(a, compute_uv=False)
    norm = max(np.linalg.norm(values), np.finfo(float).tiny)
    return {
        "rank": int(len(values)),
        "sigma_over_sigma1": {
            str(rank): float(values[rank - 1] / values[0])
            for rank in rank_grid
            if rank <= len(values)
        },
        "optimal_tail": {
            str(rank): float(np.linalg.norm(values[rank:]) / norm)
            for rank in rank_grid
            if rank < len(values)
        },
    }


def component_spectra(
    a: np.ndarray, dof_row: int, dof_col: int, rank_grid: list[int]
) -> dict[str, Any]:
    """Compact singular spectra for whole, output-component, and input-component blocks."""
    return {
        "all": _spectrum(a, rank_grid),
        "output": {
            str(component): _spectrum(a[component::dof_row], rank_grid)
            for component in range(dof_row)
        },
        "input": {
            str(component): _spectrum(a[:, component::dof_col], rank_grid)
            for component in range(dof_col)
        },
    }


def screen_block(
    a: np.ndarray,
    rank: int,
    width: int,
    seed: int,
    dr: np.ndarray,
    dc: np.ndarray,
    dof_row: int,
    dof_col: int,
) -> dict[str, Any]:
    """Use clean two-sided sketches, reporting errors after inverse scaling."""
    rank = min(rank, *a.shape)
    width = min(width, *a.shape)
    scaled = dr[:, None] * a / dc[None, :]
    rng = np.random.default_rng(seed)
    gb = rng.standard_normal((a.shape[1], width))
    ga = rng.standard_normal((a.shape[0], width))
    u = orth(scaled @ gb, rank)
    v = orth(scaled.T @ ga, rank)
    projection = inverse_scaled(u @ (u.T @ scaled @ v) @ v.T, dr, dc)
    reconstruction = inverse_scaled(u @ core_matrix_solve(u, v, scaled @ gb, ga, gb) @ v.T, dr, dc)
    return {
        "rank": rank,
        "width": width,
        "projection_errors": _component_errors(a, projection, dof_row, dof_col),
        "reconstruction_errors": _component_errors(a, reconstruction, dof_row, dof_col),
    }


def select_strata(
    root: Any, lists: Any, quantiles: list[float], maximum_entries: int
) -> list[dict[str, Any]]:
    """Select deterministic area quantiles, rather than a largest/first convenience pair."""
    selected: list[dict[str, Any]] = []
    for level in range(2, len(list(root.iter_levels()))):
        pairs = sorted(
            (
                (alpha, beta)
                for alpha in root.nodes_at_level(level)
                for beta in lists.interaction[alpha]
            ),
            key=lambda pair: (
                len(pair[0].row_indices) * len(pair[1].col_indices),
                pair[0].cell_coords,
                pair[1].cell_coords,
            ),
        )
        seen: set[tuple[Any, Any]] = set()
        for quantile in quantiles:
            alpha, beta = pairs[round((len(pairs) - 1) * quantile)]
            if (alpha, beta) in seen:
                continue
            seen.add((alpha, beta))
            entries = len(alpha.row_indices) * len(beta.col_indices)
            if entries > maximum_entries:
                raise ValueError("selected diagnostic block exceeds its declared entry limit")
            selected.append(
                {
                    "level": level,
                    "quantile": quantile,
                    "alpha": alpha,
                    "beta": beta,
                    "entries": entries,
                }
            )
    return selected


def adaptive_budget(
    root: Any, mesh: Any, policy: dict[str, Any], calibration_columns: int
) -> dict[str, int]:
    """Exact fixed-pattern columns for a per-level policy, including leaves and calibration."""
    lists = build_lists(root)
    leaf_level = len(list(root.iter_levels())) - 1
    forward_calls = forward_columns = transpose_calls = transpose_columns = 0
    for level in range(2, leaf_level + 1):
        parameters = policy["levels"].get(str(level), policy["levels"]["3"])
        width = parameters["k"] + parameters["p"]
        col = build_admissible_schedule(
            root, lists, level, mesh, parameters["k"], parameters["p"], side="col"
        )
        row = build_admissible_schedule(
            root, lists, level, mesh, parameters["k"], parameters["p"], side="row"
        )
        forward_calls += len(col.probes)
        forward_columns += len(col.probes) * width
        transpose_calls += len(row.probes)
        transpose_columns += len(row.probes) * width
    leaves = build_leaf_schedule(root, lists, leaf_level, mesh)
    leaf_columns = sum(probe.w_max for probe in leaves.probes)
    return {
        "forward_admissible_calls": forward_calls,
        "forward_admissible_columns": forward_columns,
        "transpose_admissible_calls": transpose_calls,
        "transpose_admissible_columns": transpose_columns,
        "leaf_forward_calls": len(leaves.probes),
        "leaf_forward_columns": leaf_columns,
        "calibration_forward_columns": calibration_columns // 2,
        "calibration_transpose_columns": calibration_columns // 2,
        "total_forward_columns": forward_columns + leaf_columns + calibration_columns // 2,
        "total_transpose_columns": transpose_columns + calibration_columns // 2,
        "total_columns": forward_columns + transpose_columns + leaf_columns + calibration_columns,
    }


def _block_array(
    source: RealGF, row_map: np.ndarray, col_map: np.ndarray, record: dict[str, Any]
) -> np.ndarray:
    alpha, beta = record["alpha"], record["beta"]
    return np.asarray(
        source.mat[np.ix_(row_map[alpha.row_indices], col_map[beta.col_indices])], dtype=float
    )


def _screen_records(
    blocks: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    scales: dict[str, tuple[np.ndarray, np.ndarray]],
    dof_row: int,
    dof_col: int,
    seed: int,
    rank_grid: list[int],
    combinations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    policy_by_name = {policy["name"]: policy for policy in policies}
    for block in blocks:
        a = block["array"]
        base = {key: value for key, value in block.items() if key not in {"alpha", "beta", "array"}}
        base["alpha"] = list(block["alpha"].cell_coords)
        base["beta"] = list(block["beta"].cell_coords)
        base["shape"] = list(a.shape)
        base["spectra"] = component_spectra(a, dof_row, dof_col, rank_grid)
        variants = []
        for combination in combinations:
            policy = policy_by_name[combination["policy"]]
            parameters = policy["levels"].get(str(block["level"]), policy["levels"]["3"])
            dr, dc = scales[combination["scaling"]]
            variants.append(
                {
                    **combination,
                    **screen_block(
                        a,
                        parameters["k"],
                        parameters["k"] + parameters["p"],
                        seed,
                        _expanded(dr, a.shape[0]),
                        _expanded(dc, a.shape[1]),
                        dof_row,
                        dof_col,
                    ),
                }
            )
        records.append({**base, "variants": variants})
    return records


def _worst(records: list[dict[str, Any]], policy: str, scaling: str) -> dict[str, float]:
    selected = [
        variant
        for record in records
        for variant in record["variants"]
        if variant["policy"] == policy and variant["scaling"] == scaling
    ]
    return {
        "component": max(item["reconstruction_errors"]["worst_component"] for item in selected),
        "whole_block": max(item["reconstruction_errors"]["all"] for item in selected),
    }


def _plot_aggregates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Small per-level maxima are ready for a plot without parsing dense diagnostics."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        for variant in record["variants"]:
            key = (variant["policy"], variant["scaling"], record["level"])
            groups.setdefault(key, []).append(variant)
    return [
        {
            "policy": policy,
            "scaling": scaling,
            "level": level,
            "block_count": len(values),
            "worst_component_reconstruction": max(
                value["reconstruction_errors"]["worst_component"] for value in values
            ),
            "worst_whole_block_reconstruction": max(
                value["reconstruction_errors"]["all"] for value in values
            ),
        }
        for (policy, scaling, level), values in sorted(groups.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--plan", type=Path, default=Path("benchmarks/configs/bp7_remediation.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/bp7_remediation.json")
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("bounded real remediation requires --run")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    plan = json.loads(args.plan.read_text())
    matrix = Path("real_gfs/gf_bp7/gf_mat.bin")
    coordinates = next(Path("real_gfs/gf_bp7").glob("*_fbf_coords.csv"))
    source = RealGF(matrix, coordinates)
    primary = plan["full_tree"]
    root = build_tree(source.mesh, primary["m"])
    lists = build_lists(root)
    strata = select_strata(
        root, lists, plan["strata"]["quantiles"], plan["strata"]["maximum_entries"]
    )
    for record in strata:
        record["array"] = _block_array(source, source.row_pm_to_raw, source.col_pm_to_raw, record)
    calibration = plan["black_box_scaling"]
    black_box = component_scales_from_probes(
        source, source.dof_row, source.dof_col, calibration["probes_per_side"], calibration["seed"]
    )
    oracle_components = component_scales_from_blocks(
        [record["array"] for record in strata], source.dof_row, source.dof_col
    )
    scales = {
        "identity": (np.ones(source.dof_row), np.ones(source.dof_col)),
        "black_box_rms": (black_box[0][: source.dof_row], black_box[1][: source.dof_col]),
        "oracle_stratified_rms": oracle_components,
    }
    policies = plan["adaptive_policies"]
    budgets = {
        policy["name"]: adaptive_budget(root, source.mesh, policy, calibration["total_columns"])
        for policy in policies
    }
    records = _screen_records(
        strata,
        policies,
        scales,
        source.dof_row,
        source.dof_col,
        plan["screen_seed"],
        plan["rank_grid"],
        plan["combinations"],
    )
    subset = representative_subset(
        source, max_patches=plan["subset"]["max_patches"], regions=plan["subset"]["regions"]
    )
    subset_root = build_tree(subset.mesh, plan["subset"]["m"])
    subset_strata = select_strata(
        subset_root,
        build_lists(subset_root),
        plan["strata"]["quantiles"],
        plan["strata"]["maximum_entries"],
    )
    for record in subset_strata:
        record["array"] = _block_array(source, subset.row_map, subset.col_map, record)
    subset_scales = {"black_box_rms": scales["black_box_rms"]}
    subset_records = _screen_records(
        subset_strata,
        policies,
        subset_scales,
        source.dof_row,
        source.dof_col,
        plan["screen_seed"],
        plan["rank_grid"],
        [item for item in plan["combinations"] if item["scaling"] == "black_box_rms"],
    )
    summary = {
        f"{item['policy']}+{item['scaling']}": _worst(records, item["policy"], item["scaling"])
        for item in plan["combinations"]
    }
    gate_policy = plan["promotion"]["policy"]
    gate_scaling = plan["promotion"]["scaling"]
    gate = summary[f"{gate_policy}+{gate_scaling}"]
    authorized = (
        gate["component"] <= plan["promotion"]["component_error_max"]
        and gate["whole_block"] <= plan["promotion"]["whole_block_max"]
        and budgets[gate_policy]["total_columns"] < source.shape[1]
    )
    report = {
        **diagnostic_provenance(args.revision, args.plan, matrix, coordinates),
        "environment": environment(),
        "plan": plan,
        "source": {"matrix": _file_identity(matrix), "coordinates": _file_identity(coordinates)},
        "scales": {
            name: {"dr_component": dr.tolist(), "dc_component": dc.tolist()}
            for name, (dr, dc) in scales.items()
        },
        "full_budgets": budgets,
        "records": records,
        "plot_aggregates": _plot_aggregates(records),
        "subset": {"patch_count": subset.mesh.n_patches, "records": subset_records},
        "summary": summary,
        "promotion": {
            "predeclared_gate": plan["promotion"],
            "observed": gate,
            "full_run_authorized": authorized,
            "decision": "no full run was launched by this bounded screen",
        },
        "limitations": [
            (
                "Block arrays, SVD spectra, block-fitted scales, and block-specific ranks are "
                "oracle diagnostics; no compressor entry access changed."
            ),
            (
                "The black-box scale wrapper and its independent calibration probes are "
                "implementable, but this codebase exposes only one global k,p; per-level "
                "policies need a small scheduling API extension and block-specific ranks "
                "require a redesign."
            ),
            (
                "Whole-block reconstruction is only a screen proxy for the frozen global "
                "error; it cannot establish the full operator gate."
            ),
        ],
    }
    report["provenance"]["source_sha256"][str(Path(__file__))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    write_report(args.output, report)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
