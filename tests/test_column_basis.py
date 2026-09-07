"""Tests for per-level column bases `U_{alpha,beta}` (Task 5.2, revised by
Task F.5).

`column_bases(operator, root, lists, mesh, level, factors, k, p, seed)` builds
the level's fixed `6x...x6` periodic admissible test matrices (Task 4.4),
applies them through `peeled_matvec` (Task 5.1), and for every admissible
pair `(alpha, beta)` at `level` extracts `Y(I_alpha, :)` and `G_beta`, sets
`U_{alpha,beta} = qr(Y(I_alpha, :), k)`, and retains `Y(I_alpha, :)`/`G_beta`
for the Eq. 4.3 core-matrix solve.

Per CLAUDE.md's testing rule, the low-rank/subspace checks below use
admissible blocks of the smooth `MockGF` kernel (whose singular values
genuinely decay), never a random dense matrix, and assert both the
projection error *and* a real singular-value gap (`sigma_{k+1}/sigma_1`)
so a bug that discards oversampling (e.g. an unpivoted-QR truncation) would
be caught.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.build_tree import build_tree
from gfcompress.column_basis import ColumnBasis, column_bases
from gfcompress.fixed_pattern import build_admissible_test_matrices
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.mockgf import MockGF
from gfcompress.operators import MatVecOperator
from gfcompress.tree import TreeNode


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
    """Local shim counting the total number of columns passed to `matvec`."""

    def __init__(self, inner: MatVecOperator) -> None:
        self.inner = inner
        self.matvec_calls = 0
        self.column_count = 0

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        self.matvec_calls += 1
        self.column_count += omega.shape[1] if omega.ndim == 2 else 1
        return self.inner.matvec(omega)

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.inner.rmatvec(psi)

    @property
    def shape(self) -> tuple[int, int]:
        return self.inner.shape


# ---------------------------------------------------------------------------
# 1. 2D, real low-rank check with a genuine singular-value gap
# ---------------------------------------------------------------------------


def test_column_bases_capture_dominant_column_space_2d() -> None:
    mesh = _grid_mesh(32, 32)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = 2  # 4x4 boxes of 64 patches each; blocks are 128x64.
    k, p = 10, 10

    bases = column_bases(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=0)
    assert len(bases) == len(_admissible_pairs(root, level, lists))
    assert len(bases) > 0

    for cb in bases:
        assert isinstance(cb, ColumnBasis)
        block = op.block(cb.alpha.patch_indices, cb.beta.patch_indices)

        singular_values = np.linalg.svd(block, compute_uv=False)
        gap = singular_values[k] / singular_values[0]
        assert gap > 1e-8, f"block is not genuinely low rank, sigma_{{k+1}}/sigma_1={gap}"

        u = cb.u
        projected = u @ (u.T @ block)
        rel_err = np.linalg.norm(block - projected) / np.linalg.norm(block)
        assert rel_err < 1e-3, f"rel_err={rel_err} for block shape {block.shape}"


# ---------------------------------------------------------------------------
# 2. 3D, same real low-rank check
# ---------------------------------------------------------------------------


def test_column_bases_capture_dominant_column_space_3d() -> None:
    # NOTE: deviates from FIXPLAN F.5's literal "8x8x8 grid, k=6, p=6": with
    # that mesh the first admissible level is also the leaf level (blocks
    # only 16 columns wide) and, more fundamentally, MockGF's 3D kernel
    # decays much slower per rank than the 2D one -- even the *exact*
    # best rank-6 truncation of the closest admissible block leaves several
    # percent relative error there, so no U (however constructed) can reach
    # 1e-3. A 12x12x12 grid keeps the first admissible level (2) internal
    # (not a leaf) with 27-patch boxes, and k=30 is the rank actually needed
    # for this kernel to cross 1e-3 -- still comfortably < the block's own
    # column count, and the sigma_{k+1}/sigma_1 gap is still measured to
    # confirm the truncation is real (not just "k >= full rank").
    mesh = _grid_mesh(12, 12, 12)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = _first_admissible_level(root, lists)
    k, p = 30, 30

    bases = column_bases(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=1)
    assert len(bases) == len(_admissible_pairs(root, level, lists))
    assert len(bases) > 0

    for cb in bases:
        block = op.block(cb.alpha.patch_indices, cb.beta.patch_indices)

        singular_values = np.linalg.svd(block, compute_uv=False)
        gap = singular_values[k] / singular_values[0]
        assert gap > 1e-8, f"block is not genuinely low rank, sigma_{{k+1}}/sigma_1={gap}"

        u = cb.u
        projected = u @ (u.T @ block)
        rel_err = np.linalg.norm(block - projected) / np.linalg.norm(block)
        assert rel_err < 1e-3, f"rel_err={rel_err} for block shape {block.shape}"


# ---------------------------------------------------------------------------
# 3. y_alpha / g_beta: shapes and Eq. 4.3's `Y(I_alpha,:) = A_{alpha,beta} @
#    G_beta` on the coarsest admissible level (no peeling)
# ---------------------------------------------------------------------------


def test_y_alpha_and_g_beta_shapes_and_identity() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = _first_admissible_level(root, lists)
    k, p = 2, 2

    bases = column_bases(op, root, lists, mesh, level, factors=[], k=k, p=p, seed=2)
    assert len(bases) == len(_admissible_pairs(root, level, lists))
    assert len(bases) > 0

    for cb in bases:
        assert cb.y_alpha.shape == (len(cb.alpha.row_indices), k + p)
        assert cb.g_beta.shape == (len(cb.beta.col_indices), k + p)

        block = op.block(cb.alpha.patch_indices, cb.beta.patch_indices)
        expected = block @ cb.g_beta
        np.testing.assert_allclose(cb.y_alpha, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Matvec count: one peeled matvec of width k+p per emitted Omega
# ---------------------------------------------------------------------------


def test_matvec_count_matches_test_matrix_count() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    inner = MockGF(mesh)
    counting_op = _CountingOperator(inner)

    level = _first_admissible_level(root, lists)
    k, p = 2, 2

    test_matrices = build_admissible_test_matrices(root, level, mesh, k, p, seed=3, side="col")
    assert len(test_matrices) <= 6**mesh.tree_dim

    column_bases(counting_op, root, lists, mesh, level, factors=[], k=k, p=p, seed=3)

    assert counting_op.matvec_calls == len(test_matrices)
    assert counting_op.column_count == len(test_matrices) * (k + p)


# ---------------------------------------------------------------------------
# Empty admissible level
# ---------------------------------------------------------------------------


def test_column_bases_empty_when_level_has_no_admissible_pairs() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    op = MockGF(mesh)

    level = 1  # no admissible pairs at level 1 for this tree
    assert _admissible_pairs(root, level, lists) == []

    bases = column_bases(op, root, lists, mesh, level, factors=[], k=4, p=6, seed=4)
    assert bases == []
