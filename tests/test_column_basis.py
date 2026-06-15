"""Tests for per-level column bases `U_{alpha,beta}` (Task 5.2).

`column_bases(operator, root, mesh, level, factors, k, p, seed)` builds the
level's fixed `6x...x6` periodic admissible test matrices (Task 4.2), applies
them through `peeled_matvec` (Task 5.1), and for every admissible pair
`(alpha, beta)` at `level` extracts `Y(I_alpha, :)` and sets
`U_{alpha,beta} = qr(Y(I_alpha, :), k)`.

Two properties are checked, per the task's "Output":

1. **Orthogonality**: every `U_{alpha,beta}` has orthonormal columns
   (`U.T @ U ~= I_k`).
2. **Subspace accuracy against `MockGF`**: for the genuinely low-rank
   admissible blocks `A(I_alpha, I_beta)` of the smooth `MockGF` kernel (per
   CLAUDE.md, never a random dense matrix), the column space of
   `U_{alpha,beta}` captures the block's dominant column space -- the
   projection residual `||A(I_alpha,I_beta) - U U^T A(I_alpha,I_beta)||` is
   small relative to `||A(I_alpha,I_beta)||`.
"""

from __future__ import annotations

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.column_basis import ColumnBasis, column_bases
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import interaction_lists
from gfcompress.mockgf import MockGF
from gfcompress.tree import TreeNode


def _grid_mesh(*shape: int, spacing: float = 1.0) -> FaultMesh:
    """Build a `FaultMesh` whose centroids form a regular grid of the given
    `shape` (length `d`, `d in (2, 3)`), with unit spacing along each axis."""
    axes = [np.arange(n, dtype=float) * spacing for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    L = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=L)


def _admissible_pairs(root: TreeNode, level: int) -> list[tuple[TreeNode, TreeNode]]:
    """All admissible `(alpha, beta)` pairs at `level`, from `L^int`."""
    level_nodes = root.nodes_at_level(level)
    il = interaction_lists(root)[level]
    pairs: list[tuple[TreeNode, TreeNode]] = []
    for i, alpha in enumerate(level_nodes):
        for beta in il[i]:
            pairs.append((alpha, beta))
    return pairs


# ---------------------------------------------------------------------------
# Orthogonality
# ---------------------------------------------------------------------------


def test_column_bases_are_orthonormal_2d() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 2  # coarsest level with admissible pairs
    k, p = 4, 6

    bases = column_bases(op, root, mesh, level, factors=[], k=k, p=p, seed=0)
    assert len(bases) == len(_admissible_pairs(root, level))

    for cb in bases:
        assert isinstance(cb, ColumnBasis)
        assert cb.u.shape == (len(cb.alpha.row_indices), k)
        gram = cb.u.T @ cb.u
        np.testing.assert_allclose(gram, np.eye(k), atol=1e-10)


def test_column_bases_are_orthonormal_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    deepest = 0
    for level_nodes in root.iter_levels():
        deepest = level_nodes[0].level

    level = None
    for candidate in range(1, deepest + 1):
        if _admissible_pairs(root, candidate):
            level = candidate
            break
    assert level is not None

    # At this level each box covers a single patch (dof_row * |alpha| = 3
    # rows), so k can be at most 3.
    k, p = 3, 4
    bases = column_bases(op, root, mesh, level, factors=[], k=k, p=p, seed=1)
    assert len(bases) == len(_admissible_pairs(root, level))

    for cb in bases:
        assert cb.u.shape == (len(cb.alpha.row_indices), k)
        gram = cb.u.T @ cb.u
        np.testing.assert_allclose(gram, np.eye(k), atol=1e-10)


# ---------------------------------------------------------------------------
# Subspace accuracy against MockGF
# ---------------------------------------------------------------------------


def test_column_bases_capture_dominant_column_space_2d() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 2
    k, p = 6, 8

    bases = column_bases(op, root, mesh, level, factors=[], k=k, p=p, seed=2)

    for cb in bases:
        block = op.block(cb.alpha.patch_indices, cb.beta.patch_indices)
        u = cb.u

        projected = u @ (u.T @ block)
        residual_norm = np.linalg.norm(block - projected)
        block_norm = np.linalg.norm(block)

        rel_err = residual_norm / block_norm
        assert rel_err < 1e-2, f"rel_err={rel_err} for block shape {block.shape}"


def test_column_bases_capture_dominant_column_space_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    deepest = 0
    for level_nodes in root.iter_levels():
        deepest = level_nodes[0].level

    level = None
    for candidate in range(1, deepest + 1):
        if _admissible_pairs(root, candidate):
            level = candidate
            break
    assert level is not None

    # At this level each box covers a single patch, so the block is
    # (dof_row, dof_col) = (3, 2): rank <= 2. Use k = 2 < 3 = dof_row * |alpha|
    # so the projection is a genuine (non-trivial) subspace test.
    k, p = 2, 6
    bases = column_bases(op, root, mesh, level, factors=[], k=k, p=p, seed=3)

    for cb in bases:
        block = op.block(cb.alpha.patch_indices, cb.beta.patch_indices)
        u = cb.u

        projected = u @ (u.T @ block)
        residual_norm = np.linalg.norm(block - projected)
        block_norm = np.linalg.norm(block)

        rel_err = residual_norm / block_norm
        assert rel_err < 1e-2, f"rel_err={rel_err} for block shape {block.shape}"


# ---------------------------------------------------------------------------
# Empty admissible level
# ---------------------------------------------------------------------------


def test_column_bases_empty_when_level_has_no_admissible_pairs() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 1  # no admissible pairs at level 1 for this tree
    assert _admissible_pairs(root, level) == []

    bases = column_bases(op, root, mesh, level, factors=[], k=4, p=6, seed=4)
    assert bases == []
