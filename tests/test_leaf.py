"""Tests for leaf/inadmissible dense-block extraction through the residual
operator `A - A^{(L)}` (Task 5.4).

`extract_leaves(operator, root, lists, mesh, level, factors)` applies Task
4.3's shared-slot periodic leaf test matrices
(`gfcompress.fixed_pattern.build_leaf_test_matrices`) through
`gfcompress.peeling.peeled_matvec` with the full set of admissible-block
`Factors` for levels `2, ..., level`, and reads off one `DenseLeaf(alpha,
beta, block)` per same-level neighbor pair `(alpha, beta)`, `beta in
lists.nei[alpha]`.

Per CLAUDE.md's testing rule, all blocks below come from the smooth `MockGF`
kernel (genuine spectral decay), never a random dense matrix. Test (b) below
is the sharp one: it uses ground-truth *exact* factors (full-rank SVDs of
every admissible block, test-only) so the residual `A - A^{(L)}` is peeled to
round-off and any leftover error is purely an indexing bug in the extractor,
not far-field approximation error. Test (a) instead uses real `compress_level`
factors (genuine `k < rank` truncation); the achievable tolerance there was
verified empirically against several seeds and `k` values (errors shrink
monotonically as `k` grows, confirming the residual error tracks the
admissible-block truncation and is not a bug) before picking the threshold
below, per CLAUDE.md's "no seed-tuned thresholds" rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.build_tree import build_tree
from gfcompress.fixed_pattern import LEAF_PERIOD
from gfcompress.geometry import FaultMesh
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.leaf import DenseLeaf, extract_leaves
from gfcompress.mockgf import MockGF
from gfcompress.operators import MatVecOperator
from gfcompress.peeling import BlockFactor, Factors
from gfcompress.row_basis import compress_level
from gfcompress.tree import TreeNode


def _grid_mesh(*shape: int, spacing: float = 1.0) -> FaultMesh:
    """Build a `FaultMesh` whose centroids form a regular grid of the given
    `shape` (length `d`, `d in (2, 3)`), with unit spacing along each axis."""
    axes = [np.arange(n, dtype=float) * spacing for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    lengths = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=lengths)


def _deepest_level(root: TreeNode) -> int:
    deepest = 0
    for level_nodes in root.iter_levels():
        deepest = level_nodes[0].level
    return deepest


def _all_neighbor_pairs(
    root: TreeNode, level: int, lists: TreeLists
) -> list[tuple[TreeNode, TreeNode]]:
    """All inadmissible `(alpha, beta)` pairs at `level`, from `L^nei`."""
    level_nodes = root.nodes_at_level(level)
    pairs: list[tuple[TreeNode, TreeNode]] = []
    for alpha in level_nodes:
        for beta in lists.nei[alpha]:
            pairs.append((alpha, beta))
    return pairs


def _exact_factors(gf: MockGF, root: TreeNode, lists: TreeLists) -> Factors:
    """Ground-truth (test-only) factors: an exact full-rank SVD of every
    admissible block at every level, so `A - A^{(L)}` peels the whole far
    field to round-off and only the leaf-level neighbor blocks survive."""
    factors: Factors = []
    for level_nodes in root.iter_levels():
        for alpha in level_nodes:
            for beta in lists.interaction[alpha]:
                block = gf.block(alpha.patch_indices, beta.patch_indices)
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                factors.append(
                    BlockFactor(alpha=alpha, beta=beta, u=u, b=np.diag(s), v=vt.conj().T)
                )
    return factors


def _compress_all_levels(
    op: MockGF,
    root: TreeNode,
    lists: TreeLists,
    mesh: FaultMesh,
    deepest: int,
    k: int,
    p: int,
    seed: int,
) -> Factors:
    """`compress_level` over every level `2, ..., deepest`, accumulating
    `Factors` (including `deepest`'s own admissible pairs, per Algorithm
    4.1/Task 5.6's driver loop)."""
    factors: Factors = []
    for level in range(2, deepest + 1):
        level_factors = compress_level(op, root, lists, mesh, level, factors, k, p, seed=seed)
        factors = factors + level_factors
    return factors


class _CountingOperator(MatVecOperator):
    """Local shim counting matvec columns issued (mirrors
    `tests/test_row_basis.py`'s `_CountingOperator`)."""

    def __init__(self, inner: MatVecOperator) -> None:
        self.inner = inner
        self.matvec_calls = 0
        self.matvec_columns = 0

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        self.matvec_calls += 1
        self.matvec_columns += omega.shape[1] if omega.ndim == 2 else 1
        return self.inner.matvec(omega)

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.inner.rmatvec(psi)

    @property
    def shape(self) -> tuple[int, int]:
        return self.inner.shape


# ---------------------------------------------------------------------------
# (a) Real compress_level factors: recovered leaf blocks match MockGF.block to
#     the far-field approximation tolerance, in 2D and 3D.
# ---------------------------------------------------------------------------


def test_extract_leaves_matches_mockgf_2d() -> None:
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    op = MockGF(mesh)
    L = _deepest_level(root)
    k, p = 4, 4

    factors = _compress_all_levels(op, root, lists, mesh, L, k, p, seed=0)
    leaves = extract_leaves(op, root, lists, mesh, L, factors)

    assert len(leaves) == len(_all_neighbor_pairs(root, L, lists))
    assert len(leaves) > 0

    for leaf in leaves:
        assert isinstance(leaf, DenseLeaf)
        expected = op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices)
        assert leaf.block.shape == expected.shape
        rel_err = np.linalg.norm(leaf.block - expected) / np.linalg.norm(expected)
        assert rel_err < 5e-2, f"rel_err={rel_err} for leaf block shape {leaf.block.shape}"


def test_extract_leaves_matches_mockgf_3d() -> None:
    mesh = _grid_mesh(8, 8, 8)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    op = MockGF(mesh)
    L = _deepest_level(root)
    k, p = 8, 8

    factors = _compress_all_levels(op, root, lists, mesh, L, k, p, seed=0)
    leaves = extract_leaves(op, root, lists, mesh, L, factors)

    assert len(leaves) == len(_all_neighbor_pairs(root, L, lists))
    assert len(leaves) > 0

    for leaf in leaves:
        expected = op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices)
        rel_err = np.linalg.norm(leaf.block - expected) / np.linalg.norm(expected)
        assert rel_err < 5e-2, f"rel_err={rel_err} for leaf block shape {leaf.block.shape}"


# ---------------------------------------------------------------------------
# (b) Ground-truth exact factors: recovered leaf blocks match to 1e-8. The
#     sharp test -- isolates the extractor's indexing from approximation
#     error.
# ---------------------------------------------------------------------------


def test_extract_leaves_exact_factors_2d() -> None:
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    op = MockGF(mesh)
    L = _deepest_level(root)

    factors = _exact_factors(op, root, lists)
    assert factors, "expected admissible blocks to peel"

    leaves = extract_leaves(op, root, lists, mesh, L, factors)
    assert len(leaves) == len(_all_neighbor_pairs(root, L, lists))
    assert len(leaves) > 0

    for leaf in leaves:
        expected = op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices)
        np.testing.assert_allclose(leaf.block, expected, rtol=1e-8, atol=1e-8)


def test_extract_leaves_exact_factors_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    op = MockGF(mesh)
    L = _deepest_level(root)

    factors = _exact_factors(op, root, lists)
    assert factors, "expected admissible blocks to peel"

    leaves = extract_leaves(op, root, lists, mesh, L, factors)
    assert len(leaves) == len(_all_neighbor_pairs(root, L, lists))
    assert len(leaves) > 0

    for leaf in leaves:
        expected = op.block(leaf.alpha.patch_indices, leaf.beta.patch_indices)
        np.testing.assert_allclose(leaf.block, expected, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# (c) Total probe width <= 3^d * w_max, independent of N, for both dims.
# ---------------------------------------------------------------------------


def test_extract_leaves_probe_width_bounded_2d() -> None:
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    inner = MockGF(mesh)
    counting_op = _CountingOperator(inner)
    L = _deepest_level(root)

    level_nodes = root.nodes_at_level(L)
    w_max = max(len(box.col_indices) for box in level_nodes)
    bound = LEAF_PERIOD**mesh.d * w_max

    extract_leaves(counting_op, root, lists, mesh, L, factors=[])

    assert counting_op.matvec_calls <= LEAF_PERIOD**mesh.d
    assert counting_op.matvec_columns <= bound
    # The bound must not degrade to O(N): far below one column per box.
    assert counting_op.matvec_columns < mesh.n_cols


def test_extract_leaves_probe_width_bounded_3d() -> None:
    mesh = _grid_mesh(8, 8, 8)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    inner = MockGF(mesh)
    counting_op = _CountingOperator(inner)
    L = _deepest_level(root)

    level_nodes = root.nodes_at_level(L)
    w_max = max(len(box.col_indices) for box in level_nodes)
    bound = LEAF_PERIOD**mesh.d * w_max

    extract_leaves(counting_op, root, lists, mesh, L, factors=[])

    assert counting_op.matvec_calls <= LEAF_PERIOD**mesh.d
    assert counting_op.matvec_columns <= bound
    assert counting_op.matvec_columns < mesh.n_cols


# ---------------------------------------------------------------------------
# DenseLeaf shapes and full coverage of L^nei (alpha included).
# ---------------------------------------------------------------------------


def test_extract_leaves_covers_every_neighbor_pair_including_self() -> None:
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    op = MockGF(mesh)
    L = _deepest_level(root)

    leaves = extract_leaves(op, root, lists, mesh, L, factors=[])
    pairs = {(leaf.alpha, leaf.beta) for leaf in leaves}

    expected_pairs = set(_all_neighbor_pairs(root, L, lists))
    assert pairs == expected_pairs

    # Every box is its own neighbor, so the diagonal (self) block must appear.
    assert any(leaf.alpha is leaf.beta for leaf in leaves)

    for leaf in leaves:
        assert leaf.block.shape == (len(leaf.alpha.row_indices), len(leaf.beta.col_indices))
