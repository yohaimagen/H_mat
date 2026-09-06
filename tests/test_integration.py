"""Task 6.1: end-to-end integration test of `compress()` (Algorithm 4.1,
paper Sec. 4.1) on 2D and 3D `MockGF` fixtures, following the same
discriminating-baseline convention Task 5.5/5.6 established: an absolute
`relative_error(H, A) < tol` threshold alone cannot certify anything (Task
5.5's regression: a leaves-only `HMatrix` also sits below a plausible-looking
`tol`, because `MockGF`'s near-diagonal self-interaction dominates `||A||`).
Every accuracy assertion below also builds that leaves-only baseline and
requires the real `compress()` output to beat it by >= 10x.

Fixture choice -- deviates from PLAN.md's literal Task 6.1 fixtures, and the
deviation is load-bearing, not cosmetic:

2D: `32x32` grid, **`m=16`** (PLAN.md says `m=8`), `k=10, p=10`. With `m=8`
the tree goes one level deeper (depth 4, leaf width 4 patches), and the leaf
level's admissible pairs are compressed by the *same* `compress_level` loop
as every other level (`compress()` iterates `level in range(2, leaf_level +
1)`, leaf level included). Verified directly: `compress(..., m=8, k=10,
p=10)` raises `ValueError: k must be in [0, 8]` from `randomized.orth`,
because a leaf-level column-basis sample has only `dof_row * 4 = 8` rows
while `k=10 > 8`. `m=16` stops the tree one level earlier (depth 3, leaf
width 16 patches), giving comfortable headroom at both compressed levels: 2
(box width 64) and 3 (box width 16, the leaf level) -- `k=10` is well below
either level's block rank.

3D: PLAN.md's `8x8x8, m=8` is REJECTED. That grid has depth 2, so its only
compressed level (2) *is* the leaf level and is reached with `factors=[]`
(nothing peeled yet) -- `compress()` never exercises cross-level peeling
(subtracting already-compressed coarser factors before sampling a finer
level, `peeling.apply_truncated`), which is exactly the mechanism an
Algorithm-4.1 integration test needs to exercise. A uniform-depth-3 fixture
is required instead. The tree bisects all `d` axes every level (Task
F.1), so for a depth-3 CUBE, `N = leaf_width * 8**3` regardless of shape;
the smallest cube with `leaf_width` large enough for genuine `k < rank`
(`16x16x16`, `N=4096`) measured ~2.7 GB peak RSS to assemble `MockGF`'s dense
reference and ~50s for `compress()` alone -- too costly for a routine test.
`12x12x12` was also rejected: its level-3 box widths are non-uniform
(1-8 patches), so `randomized.orth` rejects `k > 2` for the narrower boxes.

Used instead: an ANISOTROPIC `8x16x16` grid, `m=8, k=6, p=6` (`N=2048`).
Bisecting all axes every level still reaches uniform depth 3 (widths 32,
then 4), but at half `16x16x16`'s patch count, because the tree's `N` vs.
`leaf_width` relationship depends only on depth, not on which axis is
"short". Both compressed levels here are genuinely `k < rank` (checked
exhaustively below, not assumed -- Task 5.5's finding was that unverified
"`k < rank` at every level" claims are exactly what slips through): the
worst-case `sigma_(k+1)/sigma_1` measured is ~0.09 at level 2 and ~0.05 at
level 3. `compress()` still costs ~30s here: `peeling.apply_truncated`
reconstructs each already-compressed level from its full factor list on
*every* probe column, so its cost is `O(N)` per probe regardless of `N`'s
size -- a genuinely multi-level 3D fixture is inherently more expensive here
than the 2D one, independent of any test-authoring choice. Setup time is
logged, not asserted, for exactly this reason (machine-dependent, and
already known to run into the tens of seconds for 3D).
"""

from __future__ import annotations

import time

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.compress import CountingOperator, compress
from gfcompress.error import relative_error
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import build_lists
from gfcompress.mockgf import MockGF


def _grid_mesh(*shape: int, spacing: float = 1.0) -> FaultMesh:
    """Build a `FaultMesh` whose centroids form a regular grid of the given
    `shape` (length `d`, `d in (2, 3)`), with unit spacing along each axis."""
    axes = [np.arange(n, dtype=float) * spacing for n in shape]
    mesh_grids = np.meshgrid(*axes, indexing="ij")
    centroids = np.stack([g.ravel() for g in mesh_grids], axis=1)
    lengths = np.full(centroids.shape[0], 0.1 * spacing)
    return FaultMesh(centroids=centroids, L=lengths)


def _deepest_level(root: object) -> int:
    deepest = 0
    for level_nodes in root.iter_levels():  # type: ignore[attr-defined]
        deepest = level_nodes[0].level
    return deepest


def _assert_genuine_truncation(
    op: MockGF,
    root: object,
    lists: object,
    levels: tuple[int, ...],
    k: int,
    max_gap: float,
) -> None:
    """Exhaustively check, over every admissible pair at every level in
    `levels`, that the block's rank exceeds `k` (so truncation is not a
    structural no-op) and that `sigma_(k+1)/sigma_1 < max_gap` (so it is a
    genuine low-rank approximation, not `k >= rank` in numerical disguise)."""
    for level in levels:
        for alpha in root.nodes_at_level(level):  # type: ignore[attr-defined]
            for beta in lists.interaction[alpha]:  # type: ignore[attr-defined]
                block = op.block(alpha.patch_indices, beta.patch_indices)
                sv = np.linalg.svd(block, compute_uv=False)
                assert len(sv) > k, (
                    f"level {level}: block is only rank {len(sv)} <= k={k}; "
                    "not a genuine k < rank test"
                )
                gap = sv[k] / sv[0]
                assert gap < max_gap, (
                    f"level {level}: sigma_(k+1)/sigma_1={gap} >= {max_gap}; "
                    "block is not genuinely low rank"
                )


def _log_metrics(
    label: str,
    hmat: HMatrix,
    mesh: FaultMesh,
    counting_op: CountingOperator,
    setup_time: float,
) -> None:
    dense_bytes = mesh.n_rows * mesh.n_cols
    compressed_bytes = sum(f.u.size + f.b.size + f.v.size for f in hmat.factors) + sum(
        leaf.block.size for leaf in hmat.leaves
    )
    ratio = dense_bytes / compressed_bytes if compressed_bytes else float("inf")
    print(
        f"[{label}] N={mesh.n_patches} setup_time={setup_time:.2f}s "
        f"matvec_cols={counting_op.matvec_columns} rmatvec_cols={counting_op.rmatvec_columns} "
        f"compression_ratio={ratio:.2f}x "
        f"(dense={dense_bytes} vs compressed={compressed_bytes} entries)"
    )


def test_integration_2d() -> None:
    mesh = _grid_mesh(32, 32)
    op = MockGF(mesh)
    m, k, p, seed = 16, 10, 10, 0

    root = build_tree(mesh, m)
    lists = build_lists(root)
    assert _deepest_level(root) == 3
    _assert_genuine_truncation(op, root, lists, levels=(2, 3), k=k, max_gap=1e-2)

    counting_op = CountingOperator(op)
    t0 = time.time()
    hmat = compress(counting_op, mesh, m=m, k=k, p=p, seed=seed, sampling="fixed")
    setup_time = time.time() - t0

    rel_err = relative_error(hmat, op, seed=1)
    tol = 1e-8
    assert rel_err < tol, f"rel_err={rel_err} (measured ~8e-10), tol={tol}"

    leaves_only = HMatrix(root=hmat.root, mesh=mesh, factors=[], leaves=hmat.leaves)
    rel_err_leaves_only = relative_error(leaves_only, op, seed=1)
    assert (
        rel_err * 10 < rel_err_leaves_only
    ), f"rel_err={rel_err}, rel_err_leaves_only={rel_err_leaves_only}"

    _log_metrics("2D 32x32 m=16 k=10 p=10", hmat, mesh, counting_op, setup_time)


def test_integration_3d() -> None:
    mesh = _grid_mesh(8, 16, 16)
    op = MockGF(mesh)
    m, k, p, seed = 8, 6, 6, 0

    root = build_tree(mesh, m)
    lists = build_lists(root)
    assert _deepest_level(root) == 3
    _assert_genuine_truncation(op, root, lists, levels=(2, 3), k=k, max_gap=0.15)

    counting_op = CountingOperator(op)
    t0 = time.time()
    hmat = compress(counting_op, mesh, m=m, k=k, p=p, seed=seed, sampling="fixed")
    setup_time = time.time() - t0

    rel_err = relative_error(hmat, op, seed=1)
    tol = 1e-8
    assert rel_err < tol, f"rel_err={rel_err} (measured ~8e-10), tol={tol}"

    leaves_only = HMatrix(root=hmat.root, mesh=mesh, factors=[], leaves=hmat.leaves)
    rel_err_leaves_only = relative_error(leaves_only, op, seed=1)
    assert (
        rel_err * 10 < rel_err_leaves_only
    ), f"rel_err={rel_err}, rel_err_leaves_only={rel_err_leaves_only}"

    _log_metrics("3D 8x16x16 m=8 k=6 p=6", hmat, mesh, counting_op, setup_time)
