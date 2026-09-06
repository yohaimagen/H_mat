"""Tests for the `HMatrix` container: `dot`/`rdot`, `block_partition`
(Task 5.5).

Per CLAUDE.md's testing rule, all admissible-block factors below come from
the smooth `MockGF` kernel (genuine spectral decay), never a random dense
matrix. `_exact_factors`/`_exact_leaves` use ground-truth (test-only) exact
values -- full-rank SVDs for admissible blocks, the true dense sub-block for
leaves -- so `dot`/`rdot` are checked against `MockGF.A`/`MockGF.A.conj().T`
to round-off, isolating `HMatrix`'s own indexing/association from far-field
approximation error (the sharp test the task calls for). A second pair of
tests instead builds a `HMatrix` from real `compress_level` factors, to
confirm the real, non-exact pipeline reaches a sane accuracy too. Truncation
(`k < rank`) is only genuinely exercised at the coarser admissible levels
(e.g. level 2); finer levels can have blocks small enough that `k` already
equals the full rank.
"""

from __future__ import annotations

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import TreeLists, build_lists
from gfcompress.leaf import DenseLeaf, extract_leaves
from gfcompress.mockgf import MockGF
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


def _clustered_mesh_2d() -> FaultMesh:
    """Non-uniform mesh: two well-separated clusters, so leaf/admissible
    boxes have varying sizes -- the branch a uniform grid never exercises."""
    rng = np.random.default_rng(0)
    cluster_a = rng.uniform(0.0, 0.3, size=(60, 2))
    cluster_b = rng.uniform(0.7, 1.0, size=(60, 2))
    centroids = np.vstack([cluster_a, cluster_b])
    lengths = np.full(centroids.shape[0], 0.01)
    return FaultMesh(centroids=centroids, L=lengths)


def _deepest_level(root: TreeNode) -> int:
    deepest = 0
    for level_nodes in root.iter_levels():
        deepest = level_nodes[0].level
    return deepest


def _exact_factors(gf: MockGF, root: TreeNode, lists: TreeLists) -> Factors:
    """Ground-truth (test-only) factors: an exact full-rank SVD of every
    admissible block at every level."""
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


def _exact_leaves(gf: MockGF, root: TreeNode, level: int, lists: TreeLists) -> list[DenseLeaf]:
    """Ground-truth (test-only) leaves: the true dense sub-block for every
    inadmissible neighbor pair at `level`."""
    leaves: list[DenseLeaf] = []
    for alpha in root.nodes_at_level(level):
        for beta in lists.nei[alpha]:
            block = gf.block(alpha.patch_indices, beta.patch_indices)
            leaves.append(DenseLeaf(alpha=alpha, beta=beta, block=block))
    return leaves


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
    factors: Factors = []
    for level in range(2, deepest + 1):
        level_factors = compress_level(op, root, lists, mesh, level, factors, k, p, seed=seed)
        factors = factors + level_factors
    return factors


def _build_exact_hmatrix(gf: MockGF, root: TreeNode, lists: TreeLists, level: int) -> HMatrix:
    factors = _exact_factors(gf, root, lists)
    leaves = _exact_leaves(gf, root, level, lists)
    return HMatrix(root=root, mesh=gf.mesh, factors=factors, leaves=leaves)


# ---------------------------------------------------------------------------
# block_partition: complete, disjoint cover of every (i, j) patch pair.
# ---------------------------------------------------------------------------


def _assert_complete_disjoint_cover(hmat: HMatrix, n_patches: int) -> None:
    pair_sets = hmat.block_partition()
    assert len(pair_sets) == len(hmat.factors) + len(hmat.leaves)

    seen: set[tuple[int, int]] = set()
    for pair_set in pair_sets:
        assert seen.isdisjoint(pair_set), "a patch pair was stored by more than one block"
        seen.update(pair_set)

    expected = {(i, j) for i in range(n_patches) for j in range(n_patches)}
    assert seen == expected, "block_partition does not cover every (i, j) patch pair exactly once"


def test_block_partition_covers_every_pair_2d() -> None:
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    hmat = _build_exact_hmatrix(gf, root, lists, L)
    _assert_complete_disjoint_cover(hmat, mesh.n_patches)


def test_block_partition_covers_every_pair_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    hmat = _build_exact_hmatrix(gf, root, lists, L)
    _assert_complete_disjoint_cover(hmat, mesh.n_patches)


def test_block_partition_covers_every_pair_clustered_2d() -> None:
    mesh = _clustered_mesh_2d()
    root = build_tree(mesh, m=6)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    # Self-verifying: this fixture must actually exercise varying box sizes,
    # or it is no better than the uniform-grid fixtures above.
    widths = {len(box.col_indices) for box in root.nodes_at_level(L)}
    assert len(widths) > 1, "fixture must produce leaf boxes of varying width"

    hmat = _build_exact_hmatrix(gf, root, lists, L)
    _assert_complete_disjoint_cover(hmat, mesh.n_patches)


# ---------------------------------------------------------------------------
# dot/rdot against exact factors: matches MockGF.A / MockGF.A.conj().T to
# round-off. rdot is checked against the true adjoint A.conj().T directly,
# not against dot() of a transposed input.
# ---------------------------------------------------------------------------


def test_dot_rdot_match_dense_reference_exact_2d() -> None:
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    hmat = _build_exact_hmatrix(gf, root, lists, L)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((mesh.n_cols, 3))
    y = rng.standard_normal((mesh.n_rows, 3))

    np.testing.assert_allclose(hmat.dot(x), gf.A @ x, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hmat.rdot(y), gf.A.conj().T @ y, rtol=1e-8, atol=1e-8)

    # Single-vector inputs too.
    x1 = rng.standard_normal(mesh.n_cols)
    y1 = rng.standard_normal(mesh.n_rows)
    np.testing.assert_allclose(hmat.dot(x1), gf.A @ x1, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hmat.rdot(y1), gf.A.conj().T @ y1, rtol=1e-8, atol=1e-8)


def test_dot_rdot_match_dense_reference_exact_3d() -> None:
    mesh = _grid_mesh(4, 4, 4)
    root = build_tree(mesh, m=2)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    hmat = _build_exact_hmatrix(gf, root, lists, L)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((mesh.n_cols, 3))
    y = rng.standard_normal((mesh.n_rows, 3))

    np.testing.assert_allclose(hmat.dot(x), gf.A @ x, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hmat.rdot(y), gf.A.conj().T @ y, rtol=1e-8, atol=1e-8)


def test_dot_rdot_match_dense_reference_exact_clustered_2d() -> None:
    mesh = _clustered_mesh_2d()
    root = build_tree(mesh, m=6)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    hmat = _build_exact_hmatrix(gf, root, lists, L)

    rng = np.random.default_rng(3)
    x = rng.standard_normal((mesh.n_cols, 3))
    y = rng.standard_normal((mesh.n_rows, 3))

    np.testing.assert_allclose(hmat.dot(x), gf.A @ x, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hmat.rdot(y), gf.A.conj().T @ y, rtol=1e-8, atol=1e-8)


def test_rdot_is_the_true_adjoint_not_dot_of_transpose() -> None:
    """`<A_H x, y> == <x, A_H* y>` against random vectors -- catches a bug
    that would swap `alpha`/`beta` roles symmetrically enough to still pass
    `rdot(y) == hmat.dot(y.T).T`-style checks on a square-shaped mistake."""
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    hmat = _build_exact_hmatrix(gf, root, lists, L)

    rng = np.random.default_rng(4)
    x = rng.standard_normal(mesh.n_cols)
    y = rng.standard_normal(mesh.n_rows)

    lhs = float(y @ hmat.dot(x))
    rhs = float(hmat.rdot(y) @ x)
    assert np.isclose(lhs, rhs, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# dot/rdot with real compress_level factors (k < rank): sane, non-tuned
# accuracy against the dense reference.
# ---------------------------------------------------------------------------


def test_dot_matches_dense_reference_real_factors_2d() -> None:
    # Same mesh/m/k/p as test_leaf.py's test_extract_leaves_matches_mockgf_2d
    # (validated there: k < rank at level 2, per-block rel_err < 5e-2).
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    op = MockGF(mesh)
    L = _deepest_level(root)
    k, p = 4, 4

    factors = _compress_all_levels(op, root, lists, mesh, L, k, p, seed=11)
    leaves = extract_leaves(op, root, lists, mesh, L, factors)
    hmat = HMatrix(root=root, mesh=mesh, factors=factors, leaves=leaves)
    _assert_complete_disjoint_cover(hmat, mesh.n_patches)

    rng = np.random.default_rng(12)
    x = rng.standard_normal(mesh.n_cols)
    y = rng.standard_normal(mesh.n_rows)

    rel_err_dot = np.linalg.norm(hmat.dot(x) - op.A @ x) / np.linalg.norm(op.A @ x)
    rel_err_rdot = np.linalg.norm(hmat.rdot(y) - op.A.conj().T @ y) / np.linalg.norm(
        op.A.conj().T @ y
    )
    assert rel_err_dot < 1e-7, f"rel_err_dot={rel_err_dot}"
    assert rel_err_rdot < 1e-7, f"rel_err_rdot={rel_err_rdot}"

    # MockGF's near-diagonal self-interaction dominates ‖A x‖ (~1/eps larger
    # than the far field), so an absolute error threshold alone cannot
    # certify that the compressed far field (the admissible factors) is
    # doing anything: a leaves-only HMatrix would also pass it. Require the
    # full HMatrix to beat a leaves-only approximation by a wide margin.
    leaves_only = HMatrix(root=root, mesh=mesh, factors=[], leaves=leaves)
    rel_err_leaves_only = np.linalg.norm(leaves_only.dot(x) - op.A @ x) / np.linalg.norm(op.A @ x)
    assert (
        rel_err_dot * 10 < rel_err_leaves_only
    ), f"rel_err_dot={rel_err_dot}, rel_err_leaves_only={rel_err_leaves_only}"


def test_dot_matches_dense_reference_real_factors_3d() -> None:
    # Same mesh/m/k/p as test_leaf.py's test_extract_leaves_matches_mockgf_3d
    # (validated there: k < rank at level 2, per-block rel_err < 5e-2).
    mesh = _grid_mesh(8, 8, 8)
    root = build_tree(mesh, m=8)
    lists = build_lists(root)
    op = MockGF(mesh)
    L = _deepest_level(root)
    k, p = 8, 8

    factors = _compress_all_levels(op, root, lists, mesh, L, k, p, seed=13)
    leaves = extract_leaves(op, root, lists, mesh, L, factors)
    hmat = HMatrix(root=root, mesh=mesh, factors=factors, leaves=leaves)
    _assert_complete_disjoint_cover(hmat, mesh.n_patches)

    rng = np.random.default_rng(14)
    x = rng.standard_normal(mesh.n_cols)
    y = rng.standard_normal(mesh.n_rows)

    rel_err_dot = np.linalg.norm(hmat.dot(x) - op.A @ x) / np.linalg.norm(op.A @ x)
    rel_err_rdot = np.linalg.norm(hmat.rdot(y) - op.A.conj().T @ y) / np.linalg.norm(
        op.A.conj().T @ y
    )
    assert rel_err_dot < 1e-9, f"rel_err_dot={rel_err_dot}"
    assert rel_err_rdot < 1e-9, f"rel_err_rdot={rel_err_rdot}"

    # MockGF's near-diagonal self-interaction dominates ‖A x‖ (~1/eps larger
    # than the far field), so an absolute error threshold alone cannot
    # certify that the compressed far field (the admissible factors) is
    # doing anything: a leaves-only HMatrix would also pass it. Require the
    # full HMatrix to beat a leaves-only approximation by a wide margin.
    leaves_only = HMatrix(root=root, mesh=mesh, factors=[], leaves=leaves)
    rel_err_leaves_only = np.linalg.norm(leaves_only.dot(x) - op.A @ x) / np.linalg.norm(op.A @ x)
    assert (
        rel_err_dot * 10 < rel_err_leaves_only
    ), f"rel_err_dot={rel_err_dot}, rel_err_leaves_only={rel_err_leaves_only}"


# ---------------------------------------------------------------------------
# HMatrix is a MatVecOperator: matvec/rmatvec/shape delegate to dot/rdot.
# ---------------------------------------------------------------------------


def test_hmatrix_matvec_interface_matches_dot_rdot() -> None:
    mesh = _grid_mesh(16, 16)
    root = build_tree(mesh, m=4)
    lists = build_lists(root)
    gf = MockGF(mesh)
    L = _deepest_level(root)

    hmat = _build_exact_hmatrix(gf, root, lists, L)
    assert hmat.shape == (mesh.n_rows, mesh.n_cols)

    rng = np.random.default_rng(5)
    x = rng.standard_normal(mesh.n_cols)
    y = rng.standard_normal(mesh.n_rows)

    np.testing.assert_allclose(hmat.matvec(x), hmat.dot(x))
    np.testing.assert_allclose(hmat.rmatvec(y), hmat.rdot(y))
