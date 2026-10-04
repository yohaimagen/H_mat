"""Small, JSON-ready accounting helpers for reproducible compression runs.

This module deliberately observes the compressor from outside: it does not
change its sampling or factorization mathematics.  A construction snapshot is
taken before any optional error estimator, so validation products are never
silently charged to construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from gfcompress.build_tree import build_tree
from gfcompress.compress import CountingOperator, ProductCounts, compress_level
from gfcompress.error import relative_error
from gfcompress.fixed_pattern import build_admissible_schedule, build_leaf_schedule
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import build_lists
from gfcompress.leaf import extract_leaves
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import Factors, factor_economy, finalize_factors
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class PredictedProducts:
    """Exact scheduled construction work plus maximum validation work."""

    construction: ProductCounts
    leaves: ProductCounts
    validation_budget: ProductCounts
    total_budget: ProductCounts


@dataclass(frozen=True)
class Storage:
    """Byte accounting that keeps represented storage separate from process RSS."""

    factor_bytes: int
    leaf_bytes: int
    numerical_bytes: int
    retained_allocation_bytes: int
    tree_index_bytes: int
    mesh_bytes: int
    peak_rss_bytes: int | None
    disk_cache_bytes: int
    dense_reference_bytes: int


def _deepest_level(root: TreeNode) -> int:
    return max(nodes[0].level for nodes in root.iter_levels())


def predict_fixed_products(
    root: TreeNode,
    mesh: FaultMesh,
    k: int,
    p: int,
    validation: ProductCounts,
    lists: Any | None = None,
) -> PredictedProducts:
    """Predict fixed-path work from occupied groups, not period upper bounds."""
    lists = build_lists(root) if lists is None else lists
    leaf_level = _deepest_level(root)
    forward_calls = forward_columns = transpose_calls = transpose_columns = 0
    for level in range(2, leaf_level + 1):
        width = k + p
        forward = build_admissible_schedule(root, lists, level, mesh, k, p, side="col")
        transpose = build_admissible_schedule(root, lists, level, mesh, k, p, side="row")
        forward_calls += len(forward.probes)
        forward_columns += len(forward.probes) * width
        transpose_calls += len(transpose.probes)
        transpose_columns += len(transpose.probes) * width
    leaf = build_leaf_schedule(root, lists, leaf_level, mesh)
    leaf_columns = sum(probe.w_max for probe in leaf.probes)
    construction = ProductCounts(forward_calls, forward_columns, transpose_calls, transpose_columns)
    leaves = ProductCounts(len(leaf.probes), leaf_columns, 0, 0)
    return PredictedProducts(
        construction, leaves, validation, _add_counts(_add_counts(construction, leaves), validation)
    )


def _add_counts(left: ProductCounts, right: ProductCounts) -> ProductCounts:
    return ProductCounts(
        left.matvec_calls + right.matvec_calls,
        left.matvec_columns + right.matvec_columns,
        left.rmatvec_calls + right.rmatvec_calls,
        left.rmatvec_columns + right.rmatvec_columns,
    )


def _subtract_counts(total: ProductCounts, part: ProductCounts) -> ProductCounts:
    return ProductCounts(
        total.matvec_calls - part.matvec_calls,
        total.matvec_columns - part.matvec_columns,
        total.rmatvec_calls - part.rmatvec_calls,
        total.rmatvec_columns - part.rmatvec_columns,
    )


def _allocation_bytes(arrays: Sequence[NDArray[Any]]) -> int:
    """Count each NumPy backing allocation once, even when passed views."""
    seen: set[int] = set()
    total = 0
    for array in arrays:
        owner: Any = array
        while isinstance(getattr(owner, "base", None), np.ndarray):
            owner = owner.base
        key = id(owner)
        if key not in seen:
            seen.add(key)
            total += int(owner.nbytes)
    return total


def _tree_arrays(root: TreeNode) -> list[NDArray[Any]]:
    """Bytes in tree geometry and index arrays, counted by backing allocation."""
    arrays: list[NDArray[Any]] = []
    for nodes in root.iter_levels():
        for node in nodes:
            arrays.extend(
                (
                    node.patch_indices,
                    node.row_indices,
                    node.col_indices,
                    node.bounding_box,
                    node.center,
                )
            )
    return arrays


def tree_index_bytes(root: TreeNode) -> int:
    """Bytes in tree geometry and index arrays, counted by backing allocation."""
    return _allocation_bytes(_tree_arrays(root))


def _cache_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def representation_storage(
    hmat: HMatrix,
    *,
    peak_rss_bytes: int | None = None,
    cache_path: Path | None = None,
    dense_reference_bytes: int = 0,
) -> Storage:
    """Measure the retained H-matrix, explicitly excluding reference storage."""
    factor_arrays = [
        array
        for factor in hmat.factors
        for array in (factor.u, factor.b, factor.v)
        if array is not None
    ]
    leaf_arrays = [leaf.block for leaf in hmat.leaves]
    factor_bytes = _allocation_bytes(factor_arrays)
    leaf_bytes = _allocation_bytes(leaf_arrays)
    numerical = _allocation_bytes([*factor_arrays, *leaf_arrays])
    tree_bytes = tree_index_bytes(hmat.root)
    mesh_arrays = [hmat.mesh.centroids, hmat.mesh.L]
    mesh_bytes = _allocation_bytes(mesh_arrays)
    retained = _allocation_bytes(
        [*factor_arrays, *leaf_arrays, *_tree_arrays(hmat.root), *mesh_arrays]
    )
    cache_bytes = _cache_bytes(cache_path)
    return Storage(
        factor_bytes=factor_bytes,
        leaf_bytes=leaf_bytes,
        numerical_bytes=numerical,
        retained_allocation_bytes=retained,
        tree_index_bytes=tree_bytes,
        mesh_bytes=mesh_bytes,
        peak_rss_bytes=peak_rss_bytes,
        disk_cache_bytes=cache_bytes,
        dense_reference_bytes=dense_reference_bytes,
    )


def native_dense_reference_bytes(operator: MatVecOperator) -> int:
    """Return dense-reference bytes only when an exposed array is native-endian."""
    array = getattr(operator, "A", None)
    if not isinstance(array, np.ndarray):
        return 0
    if not array.dtype.isnative:
        raise ValueError("dense reference must be native-endian; byte swapping is not a benchmark")
    return int(array.nbytes)


def time_products(
    operator: MatVecOperator,
    *,
    transpose: bool,
    rhs_columns: int,
    repeats: int,
    seed: int,
) -> float:
    """Mean seconds per same-width product, with inputs outside the timer."""
    n = operator.shape[0] if transpose else operator.shape[1]
    rhs = np.random.default_rng(seed).standard_normal((n, rhs_columns))
    product = operator.rmatvec if transpose else operator.matvec
    product(rhs)  # one warm-cache product; callers record this convention.
    start = time.perf_counter()
    for _ in range(repeats):
        product(rhs)
    return (time.perf_counter() - start) / repeats


T = TypeVar("T")


def _isolated_target(conn: Any, function: Callable[..., T], args: tuple[Any, ...]) -> None:
    try:
        value = function(*args)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        peak = int(rss if sys.platform == "darwin" else rss * 1024)
        conn.send((value, peak, None))
    except BaseException as exc:  # pragma: no cover - exercised by caller propagation
        conn.send((None, None, repr(exc)))
    finally:
        conn.close()


def isolated_peak_rss(function: Callable[..., T], *args: Any) -> tuple[T, int]:
    """Run a picklable top-level function in a fresh process and return its RSS.

    ``spawn`` avoids inheriting the caller's high-water mark.  The benchmark
    entry point uses this for complete runs, never a forked child.
    """
    import multiprocessing as mp

    parent, child = mp.get_context("spawn").Pipe(duplex=False)
    process = mp.get_context("spawn").Process(target=_isolated_target, args=(child, function, args))
    process.start()
    child.close()
    value, peak, error = parent.recv()
    process.join()
    if error is not None or process.exitcode != 0:
        raise RuntimeError(f"isolated benchmark failed: {error or process.exitcode}")
    return value, int(peak)


def environment() -> dict[str, str]:
    """Stable, small environment record; thread caps are recorded, not imposed."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }


def provenance() -> dict[str, Any]:
    """Read revision and dirty state from Git; never trust an environment label."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    return {
        "commit": commit,
        "dirty": bool(diff),
        "diff_hash": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a deterministic, human-diffable JSON benchmark result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def benchmark_operator(
    reference: MatVecOperator,
    mesh: FaultMesh,
    *,
    dataset: dict[str, Any],
    m: int = 2,
    k: int = 2,
    p: int = 2,
    seed: int = 0,
    repeats: int = 3,
    validation_seed: int = 1,
    validation_iters: int = 2,
    loading_seconds: float = 0.0,
    lists: Any | None = None,
    dense_reference: MatVecOperator | None = None,
) -> dict[str, Any]:
    """Run the shared C.8 measurement core for any black-box operator."""
    operator = CountingOperator(reference)
    geometry_start = time.perf_counter()
    root = build_tree(mesh, m)
    lists = build_lists(root) if lists is None else lists
    geometry_seconds = time.perf_counter() - geometry_start
    validation_counts = ProductCounts(
        2 * validation_iters, 2 * validation_iters, 2 * validation_iters, 2 * validation_iters
    )
    predicted = predict_fixed_products(root, mesh, k, p, validation_counts, lists)
    setup_start = time.perf_counter()
    # Reuse the measured tree/list pair: this is the compressor's public loop,
    # split here only to expose geometry versus sampling/peeling time.
    import gfcompress.column_basis as column_basis_module
    import gfcompress.row_basis as row_basis_module

    local_seconds = 0.0
    original_column_orth = getattr(column_basis_module, "orth")  # noqa: B009
    original_row_orth = getattr(row_basis_module, "orth")  # noqa: B009
    original_core = getattr(row_basis_module, "core_matrix_solve")  # noqa: B009

    def timed(function: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal local_seconds
            start = time.perf_counter()
            result = function(*args, **kwargs)
            local_seconds += time.perf_counter() - start
            return result

        return wrapper

    setattr(column_basis_module, "orth", timed(original_column_orth))  # noqa: B010
    setattr(row_basis_module, "orth", timed(original_row_orth))  # noqa: B010
    setattr(row_basis_module, "core_matrix_solve", timed(original_core))  # noqa: B010
    try:
        factors: Factors = []
        for level in range(2, _deepest_level(root) + 1):
            factors += compress_level(operator, root, lists, mesh, level, factors, k, p, seed)
        admissible_observed = operator.snapshot()
        leaves = extract_leaves(operator, root, lists, mesh, _deepest_level(root), factors)
    finally:
        setattr(column_basis_module, "orth", original_column_orth)  # noqa: B010
        setattr(row_basis_module, "orth", original_row_orth)  # noqa: B010
        setattr(row_basis_module, "core_matrix_solve", original_core)  # noqa: B010
    hmat = HMatrix(root=root, mesh=mesh, factors=factors, leaves=leaves)
    sampling_seconds = time.perf_counter() - setup_start
    construction_total_observed = operator.reset()
    leaves_observed = _subtract_counts(construction_total_observed, admissible_observed)
    error = relative_error(hmat, operator, n_iters=validation_iters, seed=validation_seed)
    validation_observed = operator.snapshot()
    total_observed = _add_counts(construction_total_observed, validation_observed)
    forward = time_products(hmat, transpose=False, rhs_columns=1, repeats=repeats, seed=seed)
    adjoint = time_products(hmat, transpose=True, rhs_columns=1, repeats=repeats, seed=seed + 1)
    dense_operator = dense_reference or operator.inner
    dense_forward = time_products(
        dense_operator, transpose=False, rhs_columns=1, repeats=repeats, seed=seed
    )
    dense_adjoint = time_products(
        dense_operator, transpose=True, rhs_columns=1, repeats=repeats, seed=seed + 1
    )
    dense_bytes = native_dense_reference_bytes(dense_operator)
    storage = representation_storage(hmat, dense_reference_bytes=dense_bytes)
    return {
        "schema_version": 1,
        "provenance": provenance(),
        "dataset": dataset,
        "environment": environment(),
        "seeds": {"compression": seed, "apply": seed, "validation": validation_seed},
        "parameters": {"m": m, "k": k, "p": p, "sampling": "fixed"},
        "timing_protocol": {
            "rhs_columns": 1,
            "repeats": repeats,
            "cache": "one warm product before repeated warm timings",
            "dense_reference": (
                "native-endian on the same process, thread settings, RHS width, and convention"
            ),
            "warning": (
                "sampled columns and wall time are reported separately; no byte-swap speedup claim"
            ),
        },
        "products": {
            "budget": asdict(predicted),
            "construction_observed": asdict(admissible_observed),
            "leaves_observed": asdict(leaves_observed),
            "construction_total_observed": asdict(construction_total_observed),
            "validation_observed": asdict(validation_observed),
            "total_observed": asdict(total_observed),
            "relative_error": error,
            "validation_iterations": validation_iters,
        },
        "timing_seconds": {
            "loading_conversion": loading_seconds,
            "geometry": geometry_seconds,
            "sampling_peeling_exclusive": max(0.0, sampling_seconds - local_seconds),
            "local_factorizations": local_seconds,
            "total_setup": loading_seconds + geometry_seconds + sampling_seconds,
            "hmat_forward_warm": forward,
            "hmat_adjoint_warm": adjoint,
            "dense_forward_warm": dense_forward,
            "dense_adjoint_warm": dense_adjoint,
        },
        "storage": {
            **asdict(storage),
            "dense_storage_ratio": (
                storage.retained_allocation_bytes / dense_bytes if dense_bytes else None
            ),
            "rss_ratio": None,
            "reference_excluded_from_representation": True,
            "reference_included_in_process_rss": True,
        },
    }


def synthetic_report(
    *, n_side: int = 8, m: int = 2, k: int = 2, p: int = 2, seed: int = 0, repeats: int = 3
) -> dict[str, Any]:
    """Run the compact MockGF report used by ``benchmarks/synthetic.py``."""
    from gfcompress.mockgf import MockGF

    axis = np.arange(n_side, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    mesh = FaultMesh(np.column_stack((xx.ravel(), yy.ravel())), np.ones(n_side * n_side))
    start = time.perf_counter()
    reference = MockGF(mesh)
    return benchmark_operator(
        reference,
        mesh,
        dataset={"identity": "MockGF regular-grid", "n_side": n_side, "patches": mesh.n_patches},
        m=m,
        k=k,
        p=p,
        seed=seed,
        repeats=repeats,
        loading_seconds=time.perf_counter() - start,
    )


def c9_synthetic_report(
    *, n_side: int = 8, m: int = 2, k: int = 2, p: int = 2, seed: int = 0, repeats: int = 20
) -> dict[str, Any]:
    """Measure C.9's final-factor representation on the compact MockGF case.

    This deliberately compares identical factors before and after core
    absorption.  It is not a compression-accuracy experiment and does not
    claim that this tiny synthetic case predicts BP3/BP7 timing.
    """
    from gfcompress.mockgf import MockGF

    axis = np.arange(n_side, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    mesh = FaultMesh(np.column_stack((xx.ravel(), yy.ravel())), np.ones(n_side * n_side))
    reference = MockGF(mesh)
    root = build_tree(mesh, m)
    lists = build_lists(root)
    factors: Factors = []
    for level in range(2, _deepest_level(root) + 1):
        factors += compress_level(reference, root, lists, mesh, level, factors, k, p, seed)
    leaves = extract_leaves(reference, root, lists, mesh, _deepest_level(root), factors)
    before = HMatrix(root=root, mesh=mesh, factors=factors, leaves=leaves)
    after = HMatrix(root=root, mesh=mesh, factors=finalize_factors(factors), leaves=leaves)
    rng = np.random.default_rng(seed + 91)
    x = rng.standard_normal(mesh.n_cols)
    y = rng.standard_normal(mesh.n_rows)
    forward_difference = float(np.linalg.norm(before.dot(x) - after.dot(x)))
    adjoint_difference = float(np.linalg.norm(before.rdot(y) - after.rdot(y)))
    economy = factor_economy(factors)
    before_storage = representation_storage(before, dense_reference_bytes=reference.A.nbytes)
    after_storage = representation_storage(after, dense_reference_bytes=reference.A.nbytes)
    return {
        "schema_version": 1,
        "provenance": provenance(),
        "dataset": {"identity": "MockGF regular-grid", "n_side": n_side, "patches": mesh.n_patches},
        "parameters": {"m": m, "k": k, "p": p, "sampling": "fixed"},
        "environment": environment(),
        "measurements": {
            "far_field": {
                "blocks": economy.blocks,
                "uneconomical_blocks": economy.uneconomical_blocks,
                "uneconomical_fraction": (
                    economy.uneconomical_blocks / economy.blocks if economy.blocks else 0.0
                ),
                "before_entries": economy.before_entries,
                "after_core_absorption_entries": economy.finalized_entries,
            },
            "equivalence": {
                "forward_absolute_difference": forward_difference,
                "adjoint_absolute_difference": adjoint_difference,
            },
            "storage_bytes": {
                "before": before_storage.factor_bytes,
                "after": after_storage.factor_bytes,
                "saved": before_storage.factor_bytes - after_storage.factor_bytes,
            },
            "warm_seconds": {
                "repeats": repeats,
                "before_forward": time_products(
                    before, transpose=False, rhs_columns=1, repeats=repeats, seed=seed
                ),
                "after_forward": time_products(
                    after, transpose=False, rhs_columns=1, repeats=repeats, seed=seed
                ),
                "before_adjoint": time_products(
                    before, transpose=True, rhs_columns=1, repeats=repeats, seed=seed + 1
                ),
                "after_adjoint": time_products(
                    after, transpose=True, rhs_columns=1, repeats=repeats, seed=seed + 1
                ),
            },
        },
        "decisions": {
            "core_absorption": "adopted: exact algebraic rewrite removes each retained k-by-k core",
            "dense_far_field_fallback": (
                "deferred: uneconomical count is measured only; production far-field blocks remain "
                "low-rank"
            ),
            "tree_order_permutation": (
                "not adopted: C.8's compact case provides no demonstrated material gather/scatter "
                "bottleneck; preserve external order and maps"
            ),
            "native_endian_cache": (
                "not adopted: MockGF is in-memory and BP3/BP7 matrices are unavailable; no cache "
                "is justified without real-data evidence"
            ),
        },
        "limitations": [
            "Timing is a compact synthetic measurement, not a BP3/BP7 performance claim.",
            "No dense far-field block or permuted full operator is retained.",
        ],
    }


def synthetic_dense_only(n_side: int = 8) -> None:
    """Allocate only the native dense reference for its fresh-process RSS."""
    from gfcompress.mockgf import MockGF

    axis = np.arange(n_side, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    mesh = FaultMesh(np.column_stack((xx.ravel(), yy.ravel())), np.ones(n_side * n_side))
    MockGF(mesh)


__all__ = [
    "PredictedProducts",
    "Storage",
    "environment",
    "provenance",
    "isolated_peak_rss",
    "native_dense_reference_bytes",
    "predict_fixed_products",
    "representation_storage",
    "synthetic_report",
    "synthetic_dense_only",
    "time_products",
    "tree_index_bytes",
    "write_report",
]
