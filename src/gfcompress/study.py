"""Small, data-ready fixed-path study helpers used by the C.10 runner.

The routines deliberately use only the operator interface.  They do not
select a best seed: every supplied construction/validation pair is retained.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from gfcompress.benchmark import benchmark_operator
from gfcompress.compress import compress
from gfcompress.error import relative_error
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.operators import MatVecOperator


def validate_seed_plan(construction: Sequence[int], validation: Sequence[int]) -> None:
    """Require three construction seeds and disjoint validation starts."""
    if len(set(construction)) < 3:
        raise ValueError("C.10 requires at least three distinct construction seeds")
    if not validation or set(construction) & set(validation):
        raise ValueError("validation seeds must be nonempty and independent of construction seeds")


def response_error(
    approximate: MatVecOperator, reference: MatVecOperator, x: np.ndarray, floor: float
) -> dict[str, float]:
    """Return stable response error, using ``floor`` for near-zero responses."""
    reference_y = reference.matvec(x)
    absolute = float(np.linalg.norm(approximate.matvec(x) - reference_y))
    reference_norm = float(np.linalg.norm(reference_y))
    return {
        "absolute": absolute,
        "reference_norm": reference_norm,
        "relative_with_floor": absolute / max(reference_norm, floor),
    }


def _inputs(mesh: FaultMesh, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    patch = int(np.argmin(np.sum((mesh.centroids - mesh.centroids.mean(0)) ** 2, axis=1)))
    distance = np.linalg.norm(mesh.centroids - mesh.centroids[patch], axis=1)
    width = max(float(np.max(distance)) / 4.0, 1.0)
    localized = np.repeat(np.exp(-((distance / width) ** 2)), mesh.dof_col)
    span = np.ptp(mesh.centroids, axis=0)
    scaled = (mesh.centroids - mesh.centroids.min(0)) / np.where(span == 0, 1.0, span)
    return {
        "random": rng.standard_normal(mesh.n_cols),
        "localized": localized,
        "smooth": np.repeat(np.cos(np.pi * scaled.sum(axis=1)), mesh.dof_col),
    }


def _response(
    approximate: MatVecOperator,
    reference: MatVecOperator,
    x: np.ndarray,
    floor: float,
    rows: np.ndarray | None = None,
) -> dict[str, float]:
    actual = reference.matvec(x)
    estimate = approximate.matvec(x)
    if rows is not None:
        actual, estimate = actual[rows], estimate[rows]
    absolute = float(np.linalg.norm(estimate - actual))
    reference_norm = float(np.linalg.norm(actual))
    return {
        "absolute": absolute,
        "reference_norm": reference_norm,
        "relative_with_floor": absolute / max(reference_norm, floor),
    }


def _diagnostics(
    hmat: HMatrix, reference: MatVecOperator, mesh: FaultMesh, validation_seed: int, floor: float
) -> dict[str, Any]:
    leaves_only = HMatrix(root=hmat.root, mesh=mesh, leaves=hmat.leaves)
    inputs = _inputs(mesh, validation_seed)
    input_errors = {
        name: {
            "compressed": _response(hmat, reference, x, floor),
            "leaves_only": _response(leaves_only, reference, x, floor),
        }
        for name, x in inputs.items()
    }
    groups: dict[str, dict[str, float]] = {}
    for component in range(mesh.dof_col):
        x = np.zeros(mesh.n_cols)
        x[component :: mesh.dof_col] = inputs["random"][component :: mesh.dof_col]
        groups[str(component)] = _response(hmat, reference, x, floor)
    selected: dict[str, dict[str, float]] = {}
    for factor in hmat.factors[:3]:
        x = np.zeros(mesh.n_cols)
        x[factor.beta.col_indices] = inputs["random"][factor.beta.col_indices]
        selected[
            f"L{factor.alpha.level}:{factor.alpha.index_in_level},{factor.beta.index_in_level}"
        ] = _response(hmat, reference, x, floor, factor.alpha.row_indices)
    improvement = {
        name: value["leaves_only"]["relative_with_floor"]
        / max(value["compressed"]["relative_with_floor"], floor)
        for name, value in input_errors.items()
    }
    return {
        "global_relative_error": relative_error(hmat, reference, n_iters=4, seed=validation_seed),
        "selected_block_output_errors": selected,
        "component_index_group_errors": groups,
        "input_errors": input_errors,
        "leaves_only_improvement_factor": improvement,
    }


def fixed_sweep(
    reference: MatVecOperator,
    mesh: FaultMesh,
    *,
    m_values: Sequence[int],
    k_values: Sequence[int],
    p_values: Sequence[int],
    construction_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    absolute_floor: float = 1e-12,
) -> list[dict[str, Any]]:
    """Run a bounded all-seed fixed sweep without outcome-dependent filtering."""
    validate_seed_plan(construction_seeds, validation_seeds)
    records: list[dict[str, Any]] = []
    for m, k, p, construction_seed in product(m_values, k_values, p_values, construction_seeds):
        # Reuse C.8's accounting core so configuration selection receives
        # storage, sampled-column, and warm-apply measurements as well as error.
        measurement = benchmark_operator(
            reference,
            mesh,
            dataset={"identity": "C.10 supplied subset"},
            m=m,
            k=k,
            p=p,
            seed=construction_seed,
            validation_seed=validation_seeds[0],
            validation_iters=4,
        )
        hmat = compress(reference, mesh, m=m, k=k, p=p, seed=construction_seed, sampling="fixed")
        for validation_seed in validation_seeds:
            records.append(
                {
                    "parameters": {"m": m, "k": k, "p": p, "sampling": "fixed"},
                    "construction_seed": construction_seed,
                    "validation_seed": validation_seed,
                    "absolute_floor": absolute_floor,
                    "selection_measurements": {
                        "storage": measurement["storage"],
                        "sampled_columns": measurement["products"],
                        "apply_cost_seconds": {
                            key: value
                            for key, value in measurement["timing_seconds"].items()
                            if key.startswith(("hmat_", "dense_"))
                        },
                    },
                    "diagnostics": _diagnostics(
                        hmat, reference, mesh, validation_seed, absolute_floor
                    ),
                }
            )
    return records


def configuration_schema(dataset: str) -> dict[str, Any]:
    """Return an intentionally unlocked C.10 configuration record."""
    return {
        "schema_version": 1,
        "dataset": dataset,
        "sampling": "fixed",
        "full_runs": "opt-in",
        "construction_seeds": [11, 23, 37],
        "validation_seeds": [101, 103],
        "sweep": {"m": [8, 12], "k": [3, 4], "p": [1, 2]},
        "selection": {
            "primary": None,
            "alternatives": [],
            "required_measures": [
                "global_error",
                "selected_block_errors",
                "component_index_group_errors",
                "localized_smooth_errors",
                "leaves_only_improvement",
                "storage",
                "sampled_columns",
                "apply_cost",
            ],
            "rule": "retain every seed; never choose a favorable seed or subset",
            "locking_rule": (
                "a primary requires a measured leaves-only improvement factor and all "
                "additional block/input checks"
            ),
        },
        "acceptance": {
            "frozen": False,
            "absolute_response_floor": None,
            "global_relative_error_max": None,
            "leaves_only_improvement_min": None,
            "comparison_relative_slack": None,
            "reason": "requires measured BP3/BP7 sweep records before thresholds can be derived",
        },
    }
