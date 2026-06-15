"""Tests for the level-truncation operator `(A - A^{(l-1)})` (Task 5.1).

`apply_truncated(factors, omega, n_rows)` reconstructs `A^{(l-1)} @ omega`
from a flat list of `BlockFactor`s (one per admissible block at levels
`2, ..., l-1`), summing `P_alpha @ (U_{alpha,beta} @ (B_{alpha,beta} @
(V_{alpha,beta}.T @ omega[I_beta, :])))` over the stored blocks.
`apply_truncated_T` is the conjugate-transpose counterpart.
`peeled_matvec`/`peeled_rmatvec` combine these with `operator.matvec`/
`operator.rmatvec` to give `(A - A^{(l-1)}) @ omega` / `(A - A^{(l-1)})* @
psi`.

Two properties are checked:

1. At the coarsest level with admissible pairs (`factors == []`, since levels
   `2, ..., l-1` is empty), `peeled_matvec`/`peeled_rmatvec` reduce exactly to
   `operator.matvec`/`operator.rmatvec`.
2. Building genuine low-rank `BlockFactor`s for the admissible pairs of one
   level from `MockGF` (via the Stage 3 `two_sample_compress` /
   `core_matrix_solve` primitives -- never a random dense matrix, per
   CLAUDE.md), `apply_truncated`/`apply_truncated_T` faithfully reproduce
   `sum_{(alpha,beta)} P_alpha U B V* P_beta* @ omega` directly from the
   stored factors, and `peeled_matvec`/`peeled_rmatvec` subtract exactly that
   sum from the raw matvec/rmatvec.
"""

from __future__ import annotations

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import interaction_lists
from gfcompress.mockgf import MockGF
from gfcompress.peeling import (
    BlockFactor,
    apply_truncated,
    apply_truncated_T,
    peeled_matvec,
    peeled_rmatvec,
)
from gfcompress.randomized import core_matrix_solve, gaussian, two_sample_compress
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


def _block_factor(
    op: MockGF, alpha: TreeNode, beta: TreeNode, k: int, p: int, seed: int
) -> BlockFactor:
    """Build a genuine low-rank `BlockFactor` for `(alpha, beta)` from a
    `MockGF` block via the Stage 3 two-sample-compression / core-matrix-solve
    primitives (Tasks 3.1-3.3) -- never a random dense matrix."""
    block = op.block(alpha.patch_indices, beta.patch_indices)
    m, n = block.shape

    g_beta = gaussian(n, k, p, seed=seed)
    g_alpha = gaussian(m, k, p, seed=seed + 1)

    y = block @ g_beta
    z = block.T @ g_alpha

    u, v = two_sample_compress(y, z, k=k)
    b = core_matrix_solve(u, v, y_alpha=y, g_alpha=g_alpha, g_beta=g_beta)

    return BlockFactor(alpha=alpha, beta=beta, u=u, b=b, v=v)


# ---------------------------------------------------------------------------
# Coarsest level: empty factors -> peeled matvec/rmatvec == raw matvec/rmatvec
# ---------------------------------------------------------------------------


def test_peeled_matvec_with_empty_factors_equals_raw_matvec_2d() -> None:
    mesh = _grid_mesh(8, 8)
    op = MockGF(mesh)

    rng = np.random.default_rng(0)
    omega = rng.standard_normal((op.shape[1], 4))

    y_peeled = peeled_matvec(op, omega, factors=[])
    y_raw = op.matvec(omega)

    np.testing.assert_array_equal(y_peeled, y_raw)


def test_peeled_rmatvec_with_empty_factors_equals_raw_rmatvec_2d() -> None:
    mesh = _grid_mesh(8, 8)
    op = MockGF(mesh)

    rng = np.random.default_rng(1)
    psi = rng.standard_normal((op.shape[0], 4))

    z_peeled = peeled_rmatvec(op, psi, factors=[])
    z_raw = op.rmatvec(psi)

    np.testing.assert_array_equal(z_peeled, z_raw)


def test_peeled_matvec_with_empty_factors_equals_raw_matvec_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    op = MockGF(mesh)

    rng = np.random.default_rng(2)
    omega = rng.standard_normal(op.shape[1])  # single vector, shape (n_cols,)

    y_peeled = peeled_matvec(op, omega, factors=[])
    y_raw = op.matvec(omega)

    np.testing.assert_array_equal(y_peeled, y_raw)
    assert y_peeled.shape == (op.shape[0],)


def test_apply_truncated_empty_factors_is_zero() -> None:
    mesh = _grid_mesh(8, 8)
    op = MockGF(mesh)

    omega = np.ones((op.shape[1], 3))
    result = apply_truncated([], omega, n_rows=op.shape[0])

    assert result.shape == (op.shape[0], 3)
    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_apply_truncated_t_empty_factors_is_zero() -> None:
    mesh = _grid_mesh(8, 8)
    op = MockGF(mesh)

    psi = np.ones((op.shape[0], 3))
    result = apply_truncated_T([], psi, n_cols=op.shape[1])

    assert result.shape == (op.shape[1], 3)
    np.testing.assert_array_equal(result, np.zeros_like(result))


# ---------------------------------------------------------------------------
# Reconstruction: apply_truncated reproduces sum_{(alpha,beta)} U B V* omega
# ---------------------------------------------------------------------------


def test_apply_truncated_reproduces_stored_factors_sum() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 2  # coarsest level with admissible pairs (see module docstring)
    pairs = _admissible_pairs(root, level)
    assert len(pairs) > 0

    k, p = 3, 4
    factors = [
        _block_factor(op, alpha, beta, k=k, p=p, seed=100 + 7 * i)
        for i, (alpha, beta) in enumerate(pairs)
    ]

    rng = np.random.default_rng(42)
    omega = rng.standard_normal((op.shape[1], 5))

    result = apply_truncated(factors, omega, n_rows=op.shape[0])

    # Directly sum the per-block contributions and compare.
    expected = np.zeros((op.shape[0], 5))
    for factor in factors:
        omega_beta = omega[factor.beta.col_indices, :]
        contribution = factor.u @ (factor.b @ (factor.v.T @ omega_beta))
        expected[factor.alpha.row_indices, :] += contribution

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_apply_truncated_t_reproduces_stored_factors_sum() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 2
    pairs = _admissible_pairs(root, level)
    assert len(pairs) > 0

    k, p = 3, 4
    factors = [
        _block_factor(op, alpha, beta, k=k, p=p, seed=200 + 7 * i)
        for i, (alpha, beta) in enumerate(pairs)
    ]

    rng = np.random.default_rng(43)
    psi = rng.standard_normal((op.shape[0], 5))

    result = apply_truncated_T(factors, psi, n_cols=op.shape[1])

    expected = np.zeros((op.shape[1], 5))
    for factor in factors:
        psi_alpha = psi[factor.alpha.row_indices, :]
        contribution = factor.v @ (factor.b.T @ (factor.u.T @ psi_alpha))
        expected[factor.beta.col_indices, :] += contribution

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_peeled_matvec_subtracts_stored_factors_from_raw_matvec() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 2
    pairs = _admissible_pairs(root, level)
    assert len(pairs) > 0

    k, p = 3, 4
    factors = [
        _block_factor(op, alpha, beta, k=k, p=p, seed=300 + 7 * i)
        for i, (alpha, beta) in enumerate(pairs)
    ]

    rng = np.random.default_rng(44)
    omega = rng.standard_normal((op.shape[1], 5))

    peeled = peeled_matvec(op, omega, factors=factors)
    expected = op.matvec(omega) - apply_truncated(factors, omega, n_rows=op.shape[0])

    np.testing.assert_allclose(peeled, expected, atol=1e-12)


def test_peeled_rmatvec_subtracts_stored_factors_from_raw_rmatvec() -> None:
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 2
    pairs = _admissible_pairs(root, level)
    assert len(pairs) > 0

    k, p = 3, 4
    factors = [
        _block_factor(op, alpha, beta, k=k, p=p, seed=400 + 7 * i)
        for i, (alpha, beta) in enumerate(pairs)
    ]

    rng = np.random.default_rng(45)
    psi = rng.standard_normal((op.shape[0], 5))

    peeled = peeled_rmatvec(op, psi, factors=factors)
    expected = op.rmatvec(psi) - apply_truncated_T(factors, psi, n_cols=op.shape[1])

    np.testing.assert_allclose(peeled, expected, atol=1e-12)


def test_peeled_matvec_on_admissible_block_approximates_zero() -> None:
    """For an admissible pair (alpha, beta) whose factors were compressed
    from a genuinely low-rank MockGF block, peeling that block's
    contribution should leave only a small residual restricted to
    alpha.row_indices when omega is concentrated on beta.col_indices."""
    mesh = _grid_mesh(8, 8)
    root = build_tree(mesh, m=2)
    op = MockGF(mesh)

    level = 2
    pairs = _admissible_pairs(root, level)
    alpha, beta = pairs[0]

    k, p = 4, 6
    factor = _block_factor(op, alpha, beta, k=k, p=p, seed=500)

    rng = np.random.default_rng(46)
    omega = np.zeros((op.shape[1], 3))
    omega[beta.col_indices, :] = rng.standard_normal((len(beta.col_indices), 3))

    peeled = peeled_matvec(op, omega, factors=[factor])

    block = op.block(alpha.patch_indices, beta.patch_indices)
    full_block_contribution = block @ omega[beta.col_indices, :]
    rel_err = np.linalg.norm(peeled[alpha.row_indices, :] - 0.0 * full_block_contribution)
    rel_err /= np.linalg.norm(full_block_contribution)

    # The peeled residual on alpha's rows should be small relative to the
    # full block contribution, since U B V* captures most of the low-rank
    # block.
    assert rel_err < 1e-2
