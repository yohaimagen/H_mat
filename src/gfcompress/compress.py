"""Top-level `compress()` driver (Algorithm 4.1) and `CountingOperator`
(Task 5.6).

`compress(operator, mesh, m, k, p, seed, sampling="fixed") -> HMatrix` is the
whole non-uniform H1 pipeline: build the geometric cluster tree and its
`TreeLists` (Tasks 1.3/1.5), run Algorithm 4.1's level loop -- Task 5.3's
`compress_level` for every level `2, ..., L` with admissible pairs,
accumulating `BlockFactor`s -- then Task 5.4's leaf extraction at the deepest
level `L`, and package the result as a Task 5.5 `HMatrix`.

`compress_level` itself is *relocated* here from `gfcompress.row_basis`
(mechanical move only, no behaviour change): a module named after one of
Algorithm 4.1's three per-level passes (row bases) should not also export the
level driver that combines all three passes plus the outer loop.
`gfcompress.row_basis` keeps `RowBasis`/`row_bases`/`core_matrices`.

`sampling` accepts only `"fixed"` (the periodic test-matrix path of Tasks
4.2-4.3); `"coloring"` (Stage 7's graph-coloring optimization) is out of scope
here and rejected.

`CountingOperator` wraps any `MatVecOperator` and counts the **columns**
(not just calls) issued through `matvec`/`rmatvec` -- Sec. 4.1's matvec-count
cost claim is about column width, since each call may carry a `k+p`-wide (or
leaf-probe-wide) block of columns, not a single vector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.build_tree import build_tree
from gfcompress.column_basis import column_bases
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.leaf import extract_leaves
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import BlockFactor, Factors
from gfcompress.row_basis import core_matrices, row_bases
from gfcompress.tree import TreeNode

#: Sampling strategies accepted by `compress`. Only `"fixed"` (Tasks 4.2-4.3's
#: periodic test matrices) is implemented; `"coloring"` is Stage 7.
SUPPORTED_SAMPLING = ("fixed",)


def compress_level(
    operator: MatVecOperator,
    root: TreeNode,
    lists: TreeLists,
    mesh: FaultMesh,
    level: int,
    factors: Factors,
    k: int,
    p: int = 10,
    seed: int | None = None,
) -> list[BlockFactor]:
    """Compress every admissible pair `(alpha, beta)` at `level` (Algorithm
    4.1's level loop): `column_bases` + `row_bases` + `core_matrices`.

    Relocated verbatim from `gfcompress.row_basis` (Task 5.6); see that
    module for `column_bases`/`row_bases`/`core_matrices`.

    Args:
        operator: The black-box operator `A` (accessed only via `matvec`/
            `rmatvec`).
        root: Root of the geometric cluster tree.
        lists: Precomputed `TreeLists`.
        mesh: The `FaultMesh` underlying `root`.
        level: The tree level whose admissible pairs are compressed.
        factors: Flat list of `BlockFactor`s for levels `2, ..., level - 1`.
        k: Target rank for each block's factorization.
        p: Oversampling parameter. Defaults to `10`; pass `p=0` to disable
            oversampling explicitly.
        seed: Optional base seed forwarded to both `column_bases` and
            `row_bases`.

    Returns:
        A `Factors` list (one `BlockFactor` per admissible pair at `level`).
        Issues exactly `n_Omega + n_Psi <= 2 * 6 ** mesh.tree_dim` products of width
        `k + p` with `A`/`A*` (one peeled matvec per `Omega`, one peeled
        rmatvec per `Psi`).
    """
    cb = column_bases(operator, root, lists, mesh, level, factors, k, p, seed=seed)
    rb = row_bases(operator, root, lists, mesh, level, factors, k, p, seed=seed)
    return core_matrices(cb, rb)


def _deepest_level(root: TreeNode) -> int:
    """The level of the tree's leaves (the deepest level present)."""
    deepest = root.level
    for level_nodes in root.iter_levels():
        deepest = level_nodes[0].level
    return deepest


def compress(
    operator: MatVecOperator,
    mesh: FaultMesh,
    m: int,
    k: int,
    p: int = 10,
    seed: int | None = None,
    sampling: str = "fixed",
) -> HMatrix:
    """Compress `operator` into an `HMatrix` (Algorithm 4.1, end to end).

    Builds the geometric cluster tree (`gfcompress.build_tree.build_tree`)
    and its `TreeLists` (`gfcompress.interactions.build_lists`), runs
    `compress_level` for every level `l = 2, ..., L` (`L` the leaf level),
    accumulating `BlockFactor`s, then extracts the leaf-level dense
    inadmissible blocks (`gfcompress.leaf.extract_leaves`) from the fully
    peeled residual. Never assembles a dense `A` (per CLAUDE.md); `operator`
    is touched only through `matvec`/`rmatvec`.

    Args:
        operator: The black-box operator `A` to compress.
        mesh: The `FaultMesh` underlying `operator` (patch centroids/lengths;
            supplies `n_rows`/`n_cols` and the tree's index expansion).
        m: Leaf stop threshold for `build_tree` (a node with `<= m` patches
            does not need further splitting).
        k: Positive target rank. Each block uses the common effective rank
            `min(k, len(alpha.row_indices), len(beta.col_indices))`; this
            permits small rectangular blocks without narrowing the `k+p`
            probes. Rank zero is not supported because the fixed-pattern and
            leaf paths do not provide a meaningful zero-width compression.
        p: Oversampling parameter. Defaults to `10`; pass `p=0` to disable
            oversampling explicitly.
        seed: Optional base seed forwarded to every level's test matrices.
        sampling: Test-matrix strategy. Only `"fixed"` (the default) is
            implemented; `"coloring"` (Stage 7) is not yet supported.

    Returns:
        The compressed `HMatrix`.

    Raises:
        ValueError: If an option, mesh, or operator shape is invalid.
    """
    _validate_inputs(operator, mesh, m, k, p, seed, sampling)

    if sampling not in SUPPORTED_SAMPLING:
        raise ValueError(
            f"sampling={sampling!r} not supported; only {SUPPORTED_SAMPLING!r} "
            "is implemented ('coloring' is Stage 7, out of scope here)"
        )

    root = build_tree(mesh, m)
    lists = build_lists(root)
    leaf_level = _deepest_level(root)

    factors: Factors = []
    for level in range(2, leaf_level + 1):
        level_factors = compress_level(operator, root, lists, mesh, level, factors, k, p, seed=seed)
        factors = factors + level_factors

    leaves = extract_leaves(operator, root, lists, mesh, leaf_level, factors)
    return HMatrix(root=root, mesh=mesh, factors=factors, leaves=leaves)


def _validate_inputs(
    operator: MatVecOperator,
    mesh: FaultMesh,
    m: int,
    k: int,
    p: int,
    seed: int | None,
    sampling: str,
) -> None:
    """Validate public compression inputs before constructing any probes."""
    if not isinstance(mesh, FaultMesh):
        raise ValueError("mesh must be a FaultMesh")
    if (
        mesh.centroids.ndim != 2
        or mesh.centroids.shape[0] == 0
        or mesh.L.shape != (mesh.centroids.shape[0],)
        or mesh.tree_dim != mesh.centroids.shape[1]
        or mesh.dof_col != mesh.dof_row - 1
    ):
        raise ValueError("mesh has inconsistent centroid, length, or degree-of-freedom shapes")
    for name, value, minimum in (("m", m, 1), ("k", k, 1), ("p", p, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")
    if not isinstance(sampling, str):
        raise ValueError(f"sampling must be a string, got {sampling!r}")
    for name in ("matvec", "rmatvec"):
        if not callable(getattr(operator, name, None)):
            raise ValueError(f"operator must provide callable {name}")
    try:
        shape = operator.shape
    except AttributeError as exc:
        raise ValueError("operator must provide shape=(n_rows, n_cols)") from exc
    expected = (mesh.n_rows, mesh.n_cols)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(size, bool) or not isinstance(size, (int, np.integer)) for size in shape)
        or shape != expected
    ):
        raise ValueError(f"operator shape must be {expected}, got {shape!r}")


class CountingOperator(MatVecOperator):
    """`MatVecOperator` wrapper counting `matvec`/`rmatvec` columns issued.

    Sec. 4.1's matvec-count cost claim (`Sigma_l (n_Omega(l) + n_Psi(l)) *
    (k+p) + Sigma leaf-probe width`) is about the number of *columns* pushed
    through `A`/`A*`, not the number of calls: every fixed-pattern test
    matrix is `k+p` (or `w_max`) columns wide in one call. `matvec_calls`/
    `rmatvec_calls` are exposed too, since they fall out for free.

    Attributes:
        inner: The wrapped operator.
        matvec_calls: Number of `matvec` calls issued.
        matvec_columns: Total number of columns issued across all `matvec`
            calls (a 1-D `omega` counts as 1 column).
        rmatvec_calls: Number of `rmatvec` calls issued.
        rmatvec_columns: Total number of columns issued across all `rmatvec`
            calls.
    """

    def __init__(self, inner: MatVecOperator) -> None:
        """Wrap `inner`, resetting all counters to zero."""
        self.inner = inner
        self.matvec_calls = 0
        self.matvec_columns = 0
        self.rmatvec_calls = 0
        self.rmatvec_columns = 0

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward to `inner.matvec`, counting the call and its columns."""
        self.matvec_calls += 1
        self.matvec_columns += omega.shape[1] if omega.ndim == 2 else 1
        return self.inner.matvec(omega)

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward to `inner.rmatvec`, counting the call and its columns."""
        self.rmatvec_calls += 1
        self.rmatvec_columns += psi.shape[1] if psi.ndim == 2 else 1
        return self.inner.rmatvec(psi)

    @property
    def shape(self) -> tuple[int, int]:
        """`inner.shape`."""
        return self.inner.shape


__all__ = ["SUPPORTED_SAMPLING", "CountingOperator", "compress", "compress_level"]
