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


def _inputs(mesh: FaultMesh, seed: int, dof: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    patch = int(np.argmin(np.sum((mesh.centroids - mesh.centroids.mean(0)) ** 2, axis=1)))
    distance = np.linalg.norm(mesh.centroids - mesh.centroids[patch], axis=1)
    width = max(float(np.max(distance)) / 4.0, 1.0)
    localized = np.repeat(np.exp(-((distance / width) ** 2)), dof)
    span = np.ptp(mesh.centroids, axis=0)
    scaled = (mesh.centroids - mesh.centroids.min(0)) / np.where(span == 0, 1.0, span)
    return {
        "random": rng.standard_normal(mesh.n_patches * dof),
        "localized": localized,
        "smooth": np.repeat(np.cos(np.pi * scaled.sum(axis=1)), dof),
    }


def _response(
    approximate: MatVecOperator,
    reference: MatVecOperator,
    x: np.ndarray,
    floor: float,
    rows: np.ndarray | None = None,
    transpose: bool = False,
) -> dict[str, float]:
    product = "rmatvec" if transpose else "matvec"
    actual = getattr(reference, product)(x)
    estimate = getattr(approximate, product)(x)
    if rows is not None:
        actual, estimate = actual[rows], estimate[rows]
    absolute = float(np.linalg.norm(estimate - actual))
    reference_norm = float(np.linalg.norm(actual))
    return {
        "absolute": absolute,
        "reference_norm": reference_norm,
        "relative_with_floor": absolute / max(reference_norm, floor),
    }


def _direction_diagnostics(
    hmat: HMatrix,
    reference: MatVecOperator,
    mesh: FaultMesh,
    validation_seed: int,
    floor: float,
    *,
    transpose: bool,
) -> dict[str, Any]:
    leaves_only = HMatrix(root=hmat.root, mesh=mesh, leaves=hmat.leaves)
    dof = mesh.dof_row if transpose else mesh.dof_col
    inputs = _inputs(mesh, validation_seed, dof)
    input_errors = {
        name: {
            "compressed": _response(hmat, reference, x, floor, transpose=transpose),
            "leaves_only": _response(leaves_only, reference, x, floor, transpose=transpose),
        }
        for name, x in inputs.items()
    }
    groups: dict[str, dict[str, float]] = {}
    for component in range(dof):
        x = np.zeros(mesh.n_patches * dof)
        x[component::dof] = inputs["random"][component::dof]
        groups[str(component)] = _response(hmat, reference, x, floor, transpose=transpose)
    improvement = {
        name: value["leaves_only"]["relative_with_floor"]
        / max(value["compressed"]["relative_with_floor"], np.finfo(float).tiny)
        for name, value in input_errors.items()
    }
    return {
        "component_index_group_errors": groups,
        "input_errors": input_errors,
        "leaves_only_improvement_factor": improvement,
    }


def _diagnostics(
    hmat: HMatrix,
    reference: MatVecOperator,
    mesh: FaultMesh,
    validation_seed: int,
    floor: float,
    validation_iterations: int,
) -> dict[str, Any]:
    forward = _direction_diagnostics(hmat, reference, mesh, validation_seed, floor, transpose=False)
    adjoint = _direction_diagnostics(hmat, reference, mesh, validation_seed, floor, transpose=True)
    selected: dict[str, dict[str, Any]] = {}
    selected_factors = _truncated_factors_by_level(hmat)
    for factor in selected_factors:
        x = np.zeros(mesh.n_cols)
        # A unit source-box drive is deterministic and avoids accidentally
        # judging a block on a cancelling random response.  It is independent
        # of compression/validation seeds and makes a zero block falsifiable.
        x[factor.beta.col_indices] = 1.0
        row_dimension, col_dimension = len(factor.alpha.row_indices), len(factor.beta.col_indices)
        error = _response(hmat, reference, x, floor, factor.alpha.row_indices)
        selected[
            f"L{factor.alpha.level}:{factor.alpha.index_in_level},{factor.beta.index_in_level}"
        ] = {
            **error,
            "relative_to_response": error["absolute"]
            / max(error["reference_norm"], np.finfo(float).tiny),
            "row_dimension": row_dimension,
            "col_dimension": col_dimension,
            "retained_rank": factor.u.shape[1],
            "probe": "unit source-box drive",
        }
    return {
        "global_relative_error": relative_error(
            hmat, reference, n_iters=validation_iterations, seed=validation_seed
        ),
        "selected_block_output_errors": selected,
        "forward": forward,
        "adjoint": adjoint,
    }


def _truncated_factors_by_level(hmat: HMatrix) -> list[Any]:
    """Pick one largest genuinely truncated admissible block per level.

    C.10 must test the approximation rather than blocks whose retained rank is
    already the full local dimension.  Geometry breaks ties deterministically.
    """
    selected: list[Any] = []
    for level in sorted({factor.alpha.level for factor in hmat.factors}):
        eligible = [
            factor
            for factor in hmat.factors
            if factor.alpha.level == level
            and factor.u.shape[1] < min(len(factor.alpha.row_indices), len(factor.beta.col_indices))
        ]
        if eligible:
            selected.append(
                sorted(
                    eligible,
                    key=lambda factor: (
                        -min(len(factor.alpha.row_indices), len(factor.beta.col_indices)),
                        factor.alpha.cell_coords,
                        factor.beta.cell_coords,
                    ),
                )[0]
            )
    return selected


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
    validation_iterations: int = 4,
    dense_reference: MatVecOperator | None = None,
    metadata: dict[str, Any] | None = None,
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
            validation_iters=validation_iterations,
            dense_reference=dense_reference,
        )
        if metadata is not None and not metadata:
            metadata.update(
                {
                    "environment": measurement["environment"],
                    "timing_protocol": measurement["timing_protocol"],
                }
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
                        hmat,
                        reference,
                        mesh,
                        validation_seed,
                        absolute_floor,
                        validation_iterations,
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
        "validation": {
            "power_iterations": 4,
            "absolute_response_floor": 1e-12,
            "status": "provisional: freeze only from measured BP3/BP7 behavior",
        },
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
