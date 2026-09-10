"""Tests for per-level row bases `V_{alpha,beta}`, core matrices, and
`compress_level` (Task 5.3, revised by Task F.5).

`row_bases(operator, root, lists, mesh, level, factors, k, p, seed)` mirrors
`gfcompress.column_basis.column_bases` through `peeled_rmatvec` with the
`side="row"` fixed periodic test matrices `Psi` (Task F.4): for every
admissible pair `(alpha, beta)` at `level` it extracts `Z(I_beta, :)` and
`G_alpha`, sets `V_{alpha,beta} = qr(Z(I_beta, :), k)`, and retains `G_alpha`
for the Eq. 4.3 core-matrix solve. `core_matrices` combines a level's
`ColumnBasis`/`RowBasis` lists via `core_matrix_solve` into `BlockFactor`s;
`compress_level` is the body of Algorithm 4.1's level loop.

Per CLAUDE.md's testing rule, the low-rank/subspace checks below use
admissible blocks of the smooth `MockGF` kernel (whose singular values
genuinely decay), never a random dense matrix, with `k` strictly below the
block's numerical rank (FIXPLAN defect #5).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.build_tree import build_tree
from gfcompress.column_basis import column_bases
from gfcompress.compress import compress_level
from gfcompress.fixed_pattern import build_admissible_test_matrices
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.mockgf import MockGF
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import BlockFactor, peeled_rmatvec
from gfcompress.row_basis import RowBasis, core_matrices, row_bases
from gfcompress.tree import TreeNode, make_node


def _grid_mesh(*shape: int, spacing: float = 1.0) -> FaultMesh:
    """Build a `FaultMesh` whose centroids form a regular grid of the given
    `shape` (length `d`, `d in (2, 3)`), with unit spacing along each axis."""
    axes = [np.arange(n, dtype=float) * spacing for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    L = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=L)


def _admissible_pairs(
    root: TreeNode, level: int, lists: TreeLists
) -> list[tuple[TreeNode, TreeNode]]:
    """All admissible `(alpha, beta)` pairs at `level`, from `L^int`."""
    level_nodes = root.nodes_at_level(level)
    pairs: list[tuple[TreeNode, TreeNode]] = []
    for alpha in level_nodes:
        for beta in lists.interaction[alpha]:
            pairs.append((alpha, beta))
    return pairs


def _first_admissible_level(root: TreeNode, lists: TreeLists) -> int:
    """The shallowest level at which some box has a non-empty interaction
    list."""
    for level_nodes in root.iter_levels():
        level = level_nodes[0].level
        if _admissible_pairs(root, level, lists):
            return level
    raise AssertionError("no level has admissible pairs")


class _CountingOperator(MatVecOperator):
    """Local shim counting matvec/rmatvec calls and total probe width."""

    def __init__(self, inner: MatVecOperator) -> None:
        self.inner = inner
        self.matvec_calls = 0
        self.matvec_columns = 0
        self.rmatvec_calls = 0
        self.rmatvec_columns = 0

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        self.matvec_calls += 1
        self.matvec_columns += omega.shape[1] if omega.ndim == 2 else 1
        return self.inner.matvec(omega)

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        self.rmatvec_calls += 1
        self.rmatvec_columns += psi.shape[1] if psi.ndim == 2 else 1
        return self.inner.rmatvec(psi)

    @property
    def shape(self) -> tuple[int, int]:
        return self.inner.shape


def test_one_patch_rectangular_blocks_share_effective_rank_and_recover_exactly() -> None:
    """A smooth MockGF 2-by-1 patch block remains valid when k is larger."""
    mesh = FaultMesh(
        centroids=np.array([[0.0, 0.0], [0.02, 0.01]]),
        L=np.array([0.01, 0.01]),
    )
    root = make_node(mesh, np.array([0, 1]), level=0)
    root.cell_coords = (0, 0)
    alpha = make_node(mesh, np.array([0]), level=1, parent=root)
    beta = make_node(mesh, np.array([1]), level=1, parent=root)
    alpha.cell_coords = (0, 0)
    beta.cell_coords = (3, 0)
    root.children = [alpha, beta]
    lists = TreeLists(
        nei={root: [root], alpha: [alpha], beta: [beta]},
        interaction={root: [], alpha: [beta], beta: [alpha]},
    )

    factors = compress_level(MockGF(mesh), root, lists, mesh, 1, factors=[], k=5, p=2, seed=4)
    assert len(factors) == 2
    for factor in factors:
        assert factor.u.shape == (2, 1)
        assert factor.v.shape == factor.b.shape == (1, 1)
        exact = MockGF(mesh).block(factor.alpha.patch_indices, factor.beta.patch_indices)
        np.testing.assert_allclose(factor.u @ factor.b @ factor.v.T, exact, atol=1e-11)


# ---------------------------------------------------------------------------
# 1. 2D, real low-rank check with a genuine singular-value gap: V captures
#    the dominant row space with k < rank.
# ---------------------------------------------------------------------------


def test_row_bases_capture_dominant_row_space_2d() -> None:
    mesh = _grid_mesh(32, 32)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = 2  # 4x4 boxes of 64 patches each; blocks are 128x64.
    k, p = 10, 10

    bases = row_bases(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=0)
    assert len(bases) == len(_admissible_pairs(root, level, lists))
    assert len(bases) > 0

    for rb in bases:
        assert isinstance(rb, RowBasis)
        block = op.block(rb.alpha.patch_indices, rb.beta.patch_indices)

        singular_values = np.linalg.svd(block, compute_uv=False)
        gap = singular_values[k] / singular_values[0]
        assert gap > 1e-8, f"block is not genuinely low rank, sigma_{{k+1}}/sigma_1={gap}"

        v = rb.v
        assert v.shape == (len(rb.beta.col_indices), k)
        np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-8)

        projected = (block @ v) @ v.T
        rel_err = np.linalg.norm(block - projected) / np.linalg.norm(block)
        assert rel_err < 1e-3, f"rel_err={rel_err} for block shape {block.shape}"


# ---------------------------------------------------------------------------
# 2. g_alpha shapes and Eq. 4.3's `Z(I_beta,:) = A_{alpha,beta}* @ G_alpha`
#    on the coarsest admissible level (no peeling).
# ---------------------------------------------------------------------------


def test_g_alpha_shapes_and_identity() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = _first_admissible_level(root, lists)
    k, p = 2, 2

    # Independently recompute Z(I_beta, :) via the same Psi/peeled_rmatvec
    # row_bases uses internally, to check the raw Eq. 4.3 sample identity
    # `Z(I_beta, :) = A_{alpha,beta}* @ G_alpha` directly (rather than
    # through the rank-k-truncated v, which need not reproduce the full
    # k+p-wide sample).
    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=2, side="row")
    z_for_box: dict[TreeNode, NDArray[np.float64]] = {}
    g_for_box: dict[TreeNode, NDArray[np.float64]] = {}
    for tm in test_matrices:
        z = np.asarray(peeled_rmatvec(op, tm.omega, []), dtype=np.float64)
        for box in tm.active_boxes:
            z_for_box[box] = z
            g_for_box[box] = tm.blocks[box]

    bases = row_bases(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=2)
    assert len(bases) == len(_admissible_pairs(root, level, lists))
    assert len(bases) > 0

    for rb in bases:
        assert rb.g_alpha.shape == (len(rb.alpha.row_indices), k + p)
        np.testing.assert_allclose(rb.g_alpha, g_for_box[rb.alpha])

        block = op.block(rb.alpha.patch_indices, rb.beta.patch_indices)
        z_beta = z_for_box[rb.alpha][rb.beta.col_indices, :]
        expected = block.T @ rb.g_alpha
        np.testing.assert_allclose(z_beta, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Matvec count: one peeled rmatvec of width k+p per emitted Psi.
# ---------------------------------------------------------------------------


def test_rmatvec_count_matches_test_matrix_count() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    inner = MockGF(mesh)
    counting_op = _CountingOperator(inner)

    level = _first_admissible_level(root, lists)
    k, p = 2, 2

    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=3, side="row")
    assert len(test_matrices) <= 6**mesh.tree_dim

    row_bases(counting_op, root, lists, mesh, level, factors=[], k=k, p=p, seed=3)

    assert counting_op.matvec_calls == 0
    assert counting_op.rmatvec_calls == len(test_matrices)
    assert counting_op.rmatvec_columns == len(test_matrices) * (k + p)


# ---------------------------------------------------------------------------
# 4. Empty admissible level.
# ---------------------------------------------------------------------------


def test_row_bases_empty_when_level_has_no_admissible_pairs() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = 1  # no admissible pairs at level 1 for this tree
    assert _admissible_pairs(root, level, lists) == []

    bases = row_bases(op, root, lists, mesh, level, factors=[], k=4, p=6, seed=4)
    assert bases == []


# ---------------------------------------------------------------------------
# 5. core_matrices / compress_level: per-block ||A - U B V*|| / ||A|| small.
# ---------------------------------------------------------------------------


def test_compress_level_per_block_accuracy_2d() -> None:
    mesh = _grid_mesh(32, 32)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = 2
    k, p = 10, 20

    factors = compress_level(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=5)
    assert len(factors) == len(_admissible_pairs(root, level, lists))
    assert len(factors) > 0

    for factor in factors:
        assert isinstance(factor, BlockFactor)
        block = op.block(factor.alpha.patch_indices, factor.beta.patch_indices)
        approx = factor.u @ (factor.b @ factor.v.T)
        rel_err = np.linalg.norm(block - approx) / np.linalg.norm(block)
        assert rel_err < 1e-3, f"rel_err={rel_err} for block shape {block.shape}"


def test_core_matrices_matches_compress_level() -> None:
    """`compress_level` == `column_bases` + `row_bases` + `core_matrices`."""
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = _first_admissible_level(root, lists)
    k, p = 4, 4

    cb = column_bases(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=6)
    rb = row_bases(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=6)
    factors_direct = core_matrices(cb, rb)

    factors_level = compress_level(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=6)

    assert len(factors_direct) == len(factors_level)
    by_pair = {(f.alpha, f.beta): f for f in factors_level}
    for f in factors_direct:
        g = by_pair[(f.alpha, f.beta)]
        np.testing.assert_allclose(f.u, g.u)
        np.testing.assert_allclose(f.b, g.b)
        np.testing.assert_allclose(f.v, g.v)


# ---------------------------------------------------------------------------
# 6. Peeled second level: factors from level 2 fed into level 3 reach the
#    same per-block accuracy against the raw MockGF ground truth (the
#    nested H-matrix structure guarantees level-2 factors never overlap a
#    level-3 admissible pair's index set, so the residual equals the raw
#    block there).
# ---------------------------------------------------------------------------


def test_peeled_level_3_reaches_same_accuracy() -> None:
    mesh = _grid_mesh(32, 32)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    op = MockGF(mesh)

    # NOTE: p=10 (matching the other tests in this file) leaves the worst of
    # the 1116 level-3 blocks at rel_err ~1.06e-3 for this seed pair -- just
    # over the 1e-3 threshold. The excess does not come from level-3 sampling
    # variance: swapping in *exact* SVD factors for level 2 (same seeds, same
    # p) drops that block's error to ~3.2e-4. The real cause is the core
    # solve at level 3 consuming samples contaminated by the *approximate*
    # level-2 factors leaking through the peeled residual -- on far active
    # boxes gamma where A - A^(2) is only approximately (not exactly) zero.
    # Raising oversampling to p=20 (k unchanged) helps chiefly because it
    # improves the level-2 factors themselves (their own worst-block error
    # roughly halves), which reduces that leakage and brings the level-3
    # worst block to <1e-3 robustly across several seed pairs, so we use
    # p=20 here rather than weakening the threshold.
    k, p = 10, 20

    factors_2 = compress_level(op, root, lists, mesh, 2, factors=[], k=k, p=p, seed=7)

    level_3_pairs = _admissible_pairs(root, 3, lists)
    assert len(level_3_pairs) > 0

    factors_3 = compress_level(op, root, lists, mesh, 3, factors=factors_2, k=k, p=p, seed=8)
    assert len(factors_3) == len(level_3_pairs)

    for factor in factors_3:
        block = op.block(factor.alpha.patch_indices, factor.beta.patch_indices)
        approx = factor.u @ (factor.b @ factor.v.T)
        rel_err = np.linalg.norm(block - approx) / np.linalg.norm(block)
        assert rel_err < 1e-3, f"rel_err={rel_err} for block shape {block.shape}"


# ---------------------------------------------------------------------------
# 7. compress_level matvec/rmatvec budget: <= 2 * 6^d products of width k+p.
# ---------------------------------------------------------------------------


def test_compress_level_matvec_budget() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    inner = MockGF(mesh)
    counting_op = _CountingOperator(inner)

    level = _first_admissible_level(root, lists)
    k, p = 2, 2

    compress_level(counting_op, root, lists, mesh, level, factors=[], k=k, p=p, seed=9)

    assert counting_op.matvec_calls <= 6**mesh.tree_dim
    assert counting_op.rmatvec_calls <= 6**mesh.tree_dim
    assert counting_op.matvec_calls + counting_op.rmatvec_calls <= 2 * 6**mesh.tree_dim
    assert counting_op.matvec_columns == counting_op.matvec_calls * (k + p)
    assert counting_op.rmatvec_columns == counting_op.rmatvec_calls * (k + p)
