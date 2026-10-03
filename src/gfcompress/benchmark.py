"""Small, JSON-ready accounting helpers for reproducible compression runs.

This module deliberately observes the compressor from outside: it does not
change its sampling or factorization mathematics.  A construction snapshot is
taken before any optional error estimator, so validation products are never
silently charged to construction.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from gfcompress.build_tree import build_tree
from gfcompress.compress import CountingOperator, compress_level
from gfcompress.fixed_pattern import build_admissible_schedule, build_leaf_schedule
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import build_lists
from gfcompress.leaf import extract_leaves
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import Factors
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class PredictedProducts:
    """Actual scheduled product widths, including optional validation work."""

    matvec_calls: int
    matvec_columns: int
    rmatvec_calls: int
    rmatvec_columns: int
    leaf_matvec_calls: int
    leaf_matvec_columns: int
    validation_matvec_columns: int = 0
    validation_rmatvec_columns: int = 0


@dataclass(frozen=True)
class Storage:
    """Byte accounting that keeps represented storage separate from process RSS."""

    factor_bytes: int
    leaf_bytes: int
    numerical_bytes: int
    retained_allocation_bytes: int
    tree_index_bytes: int
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
    validation_matvec_columns: int = 0,
    validation_rmatvec_columns: int = 0,
) -> PredictedProducts:
    """Predict fixed-path work from occupied groups, not period upper bounds."""
    lists = build_lists(root)
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
    return PredictedProducts(
        matvec_calls=forward_calls + len(leaf.probes),
        matvec_columns=forward_columns + leaf_columns + validation_matvec_columns,
        rmatvec_calls=transpose_calls,
        rmatvec_columns=transpose_columns + validation_rmatvec_columns,
        leaf_matvec_calls=len(leaf.probes),
        leaf_matvec_columns=leaf_columns,
        validation_matvec_columns=validation_matvec_columns,
        validation_rmatvec_columns=validation_rmatvec_columns,
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


def tree_index_bytes(root: TreeNode) -> int:
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
    return _allocation_bytes(arrays)


def representation_storage(
    hmat: HMatrix,
    *,
    peak_rss_bytes: int | None = None,
    cache_path: Path | None = None,
    dense_reference_bytes: int = 0,
) -> Storage:
    """Measure the retained H-matrix, explicitly excluding reference storage."""
    factor_arrays = [array for factor in hmat.factors for array in (factor.u, factor.b, factor.v)]
    leaf_arrays = [leaf.block for leaf in hmat.leaves]
    factor_bytes = _allocation_bytes(factor_arrays)
    leaf_bytes = _allocation_bytes(leaf_arrays)
    numerical = _allocation_bytes([*factor_arrays, *leaf_arrays])
    tree_bytes = tree_index_bytes(hmat.root)
    cache_bytes = 0 if cache_path is None or not cache_path.exists() else cache_path.stat().st_size
    return Storage(
        factor_bytes=factor_bytes,
        leaf_bytes=leaf_bytes,
        numerical_bytes=numerical,
        retained_allocation_bytes=numerical + tree_bytes,
        tree_index_bytes=tree_bytes,
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


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a deterministic, human-diffable JSON benchmark result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def synthetic_report(
    *,
    n_side: int = 8,
    m: int = 2,
    k: int = 2,
    p: int = 2,
    seed: int = 0,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run the compact MockGF report used by ``benchmarks/synthetic.py``."""
    from gfcompress.mockgf import MockGF

    axis = np.arange(n_side, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    mesh = FaultMesh(np.column_stack((xx.ravel(), yy.ravel())), np.ones(n_side * n_side))
    load_start = time.perf_counter()
    operator = CountingOperator(MockGF(mesh))
    load_seconds = time.perf_counter() - load_start
    geometry_start = time.perf_counter()
    root = build_tree(mesh, m)
    build_lists(root)
    geometry_seconds = time.perf_counter() - geometry_start
    predicted = predict_fixed_products(root, mesh, k, p)
    setup_start = time.perf_counter()
    # Reuse the measured tree/list pair: this is the compressor's public loop,
    # split here only to expose geometry versus sampling/peeling time.
    lists = build_lists(root)
    factors: Factors = []
    for level in range(2, _deepest_level(root) + 1):
        factors += compress_level(operator, root, lists, mesh, level, factors, k, p, seed)
    leaves = extract_leaves(operator, root, lists, mesh, _deepest_level(root), factors)
    hmat = HMatrix(root=root, mesh=mesh, factors=factors, leaves=leaves)
    sampling_seconds = time.perf_counter() - setup_start
    construction = operator.snapshot()
    forward = time_products(hmat, transpose=False, rhs_columns=1, repeats=repeats, seed=seed)
    adjoint = time_products(hmat, transpose=True, rhs_columns=1, repeats=repeats, seed=seed + 1)
    dense_forward = time_products(
        operator.inner, transpose=False, rhs_columns=1, repeats=repeats, seed=seed
    )
    dense_adjoint = time_products(
        operator.inner, transpose=True, rhs_columns=1, repeats=repeats, seed=seed + 1
    )
    dense_bytes = native_dense_reference_bytes(operator.inner)
    storage = representation_storage(hmat, dense_reference_bytes=dense_bytes)
    return {
        "schema_version": 1,
        "revision": os.environ.get("GFCOMPRESS_REVISION", "unknown"),
        "dataset": {"identity": "MockGF regular-grid", "n_side": n_side, "patches": mesh.n_patches},
        "environment": environment(),
        "seeds": {"compression": seed, "apply": seed},
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
        "products": {"predicted": asdict(predicted), "construction_observed": asdict(construction)},
        "timing_seconds": {
            "loading_conversion": load_seconds,
            "geometry": geometry_seconds,
            "sampling_peeling": sampling_seconds,
            "local_factorizations": None,
            "total_setup": geometry_seconds + sampling_seconds,
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


__all__ = [
    "PredictedProducts",
    "Storage",
    "environment",
    "isolated_peak_rss",
    "native_dense_reference_bytes",
    "predict_fixed_products",
    "representation_storage",
    "synthetic_report",
    "time_products",
    "tree_index_bytes",
    "write_report",
]
