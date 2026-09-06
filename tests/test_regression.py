"""Task 6.2: parameter regression fixtures -- small sweeps over rank `k`,
oversampling `p`, leaf size `m`, and the admissibility separation `eta`,
recording error and matvec/rmatvec column counts as regression baselines.

Design choices (per CLAUDE.md and Task 6.1's established conventions):

- `eta` cannot be threaded through `compress()`/`build_lists` (both build the
  neighbor/interaction split combinatorially, never from a threshold -- see
  `gfcompress.interactions` module docstring and CLAUDE.md). So the `eta`
  "sweep" here is a *consistency* check: `gfcompress.interactions.
  is_admissible` is evaluated directly at several `eta` values (spanning
  `DEFAULT_ETA=0.5` up to the `1/sqrt(d)` bound the module documents) against
  the combinatorial `nei`/`interaction` split from `build_lists`, in both 2D
  and 3D. At the *exact* value `1/sqrt(d)` a handful of interaction-list pairs
  fail by a few ULPs (measured: ratio `0.7071067811865471` vs.
  `1/sqrt(2)=0.7071067811865475`, a `~4e-16` gap) -- pure floating-point
  rounding in `box_dist`/`diam`, not a real violation of the documented bound
  (verified directly: the true minimum ratio over all interaction pairs
  coincides with `1/sqrt(d)` to machine precision). The swept value is
  therefore `1/sqrt(d) * (1 - 1e-9)`, a hair below the true bound, and this is
  noted rather than silently worked around.

- `k`/`p`/`m` sweeps use `compress()` end to end on `MockGF` (per CLAUDE.md's
  testing rule: only a kernel with genuine spectral decay can validate a
  compressor). All three reuse one 2D fixture (`32x32` grid, `m=16` for the
  `k`/`p` sweeps) chosen for depth-3 headroom (`dof_col=1`, leaf width 16, so
  `k` up to 16 is valid -- see the `k`-precondition test below for the exact
  boundary) while staying cheap (`N=1024`, each `compress()` call ~0.1-0.3s).
  Trends are verified across seeds 0-3 while designing each sweep (not
  asserted from one lucky seed, per Task 6.1's review finding); each test
  asserts the trend on 2 of those seeds to bound runtime.

- Every accuracy assertion follows Task 5.5/6.1's discriminating-baseline
  rule: an absolute `rel_err < tol` alone cannot certify anything here
  (`MockGF`'s near-diagonal self-interaction dominates `||A||`), so the real
  check is the sweep's worst point still beating a leaves-only (`factors=[]`)
  baseline by a wide margin, plus the sweep's monotonic trend end-to-end.

- Matvec/rmatvec column counts are asserted as *exact* equalities against an
  independent recomputation from `gfcompress.fixed_pattern.
  build_admissible_test_matrices`/`build_leaf_test_matrices` (the same
  approach `tests/test_compress.py` uses), not copied from the driver's own
  bookkeeping.

Runtime: the `k`/`p`/`m` sweeps together cost ~20s wall-clock (logged, not
asserted -- machine-dependent); the `eta` and `k`-precondition checks are
near-instant (no `compress()` calls).
"""

from __future__ import annotations

import time

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.compress import CountingOperator, compress
from gfcompress.error import relative_error
from gfcompress.fixed_pattern import build_admissible_test_matrices, build_leaf_test_matrices
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import DEFAULT_ETA, build_lists, is_admissible
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


def _predicted_counts(mesh: FaultMesh, m: int, k: int, p: int) -> int:
    """Independently recompute the exact predicted matvec-column total (same
    approach as `tests/test_compress.py`'s `_predicted_counts`, calling the
    actual per-level test-matrix builders rather than trusting `compress`'s
    own bookkeeping)."""
    root = build_tree(mesh, m)
    leaf_level = _deepest_level(root)

    d = mesh.d
    total = 0
    for level in range(2, leaf_level + 1):
        omegas = build_admissible_test_matrices(root, level, mesh, k, p, side="col")
        psis = build_admissible_test_matrices(root, level, mesh, k, p, side="row")
        assert len(omegas) <= 6**d
        assert len(psis) <= 6**d
        total += (len(omegas) + len(psis)) * (k + p)

    leaf_tms = build_leaf_test_matrices(root, leaf_level, mesh)
    w_max = max(len(box.col_indices) for box in root.nodes_at_level(leaf_level))
    assert len(leaf_tms) * w_max <= 3**d * w_max
    total += len(leaf_tms) * w_max

    return total


def _observed_counts(mesh: FaultMesh, m: int, k: int, p: int, seed: int) -> tuple[int, float]:
    """Run `compress()` through a `CountingOperator`, returning
    `(observed matvec+rmatvec columns, relative_error)`."""
    op = MockGF(mesh)
    counting_op = CountingOperator(op)
    hmat = compress(counting_op, mesh, m=m, k=k, p=p, seed=seed, sampling="fixed")
    rel_err = relative_error(hmat, op, seed=1)
    observed = counting_op.matvec_columns + counting_op.rmatvec_columns
    return observed, rel_err


# ---------------------------------------------------------------------------
# eta sweep: a consistency check, not a threaded parameter (see module
# docstring). Spans DEFAULT_ETA=0.5 and the 1/sqrt(d) bound `interactions`
# documents, in both 2D and 3D.
# ---------------------------------------------------------------------------


def _assert_eta_consistent(mesh: FaultMesh, m: int, etas: list[float]) -> None:
    root = build_tree(mesh, m)
    lists = build_lists(root)
    for eta in etas:
        for level_nodes in root.iter_levels():
            level = level_nodes[0].level
            if level == 0:
                continue
            for alpha in level_nodes:
                for beta in lists.interaction[alpha]:
                    assert is_admissible(alpha, beta, eta), (
                        f"eta={eta}: interaction-list pair ({alpha}, {beta}) at "
                        f"level {level} tested inadmissible"
                    )
                for beta in lists.nei[alpha]:
                    assert not is_admissible(alpha, beta, eta), (
                        f"eta={eta}: neighbor-list pair ({alpha}, {beta}) at "
                        f"level {level} tested admissible"
                    )


def test_eta_sweep_consistency_2d() -> None:
    d = 2
    bound = (1.0 / np.sqrt(d)) * (1.0 - 1e-9)  # true bound minus float-noise margin
    etas = [0.1, 0.3, DEFAULT_ETA, bound]
    _assert_eta_consistent(_grid_mesh(16, 16), m=4, etas=etas)


def test_eta_sweep_consistency_3d() -> None:
    d = 3
    bound = (1.0 / np.sqrt(d)) * (1.0 - 1e-9)
    etas = [0.1, 0.3, DEFAULT_ETA, bound]
    _assert_eta_consistent(_grid_mesh(8, 8, 8), m=8, etas=etas)


# ---------------------------------------------------------------------------
# k precondition boundary: k <= dof_col * min_leaf_patches (fold-in docstring
# claim, verified directly here). 16x16/m=4 has leaf width 4, dof_col=1.
# ---------------------------------------------------------------------------


def test_k_precondition_boundary_16x16_m4() -> None:
    mesh = _grid_mesh(16, 16)
    op = MockGF(mesh)
    root = build_tree(mesh, m=4)
    min_leaf_patches = min(len(n.patch_indices) for n in root.nodes_at_level(_deepest_level(root)))
    assert min_leaf_patches == 4
    bound = mesh.dof_col * min_leaf_patches
    assert bound == 4

    compress(op, mesh, m=4, k=bound, p=0, seed=0, sampling="fixed")  # must not raise

    try:
        compress(op, mesh, m=4, k=bound + 1, p=0, seed=0, sampling="fixed")
        raise AssertionError(f"expected ValueError for k={bound + 1} > bound={bound}")
    except ValueError as exc:
        assert f"[0, {bound}]" in str(exc), f"expected bound {bound} in error message: {exc}"


# ---------------------------------------------------------------------------
# k sweep: 32x32/m=16 (depth 3, dof_col=1 so k up to 16 valid), p=4 fixed.
# Measured (seeds 0-3, p=4): rel_err ~2.7e-7-3.6e-7 (k=2) down to ~5.6e-11-
# 1.25e-10 (k=16), monotonically non-increasing at every seed tried (0-3;
# 2 seeds asserted here to bound runtime).
# ---------------------------------------------------------------------------


def test_k_sweep_monotonic_error_and_exact_matvec_counts() -> None:
    mesh = _grid_mesh(32, 32)
    m, p = 16, 4
    ks = [2, 4, 8, 12, 16]
    seeds = [0, 1]  # trend verified across seeds 0-3 during design; 2 asserted here

    t0 = time.time()
    for seed in seeds:
        errs = []
        for k in ks:
            observed, rel_err = _observed_counts(mesh, m, k, p, seed)
            predicted = _predicted_counts(mesh, m, k, p)
            assert (
                observed == predicted
            ), f"seed={seed} k={k}: observed={observed}, predicted={predicted}"
            errs.append(rel_err)

        for i in range(len(errs) - 1):
            assert errs[i] >= errs[i + 1] - 1e-14, (
                f"seed={seed}: error increased going k={ks[i]} -> k={ks[i + 1]}: "
                f"{errs[i]} -> {errs[i + 1]}"
            )
        # Wide end-to-end spread: k=16's error is orders of magnitude below
        # k=2's (measured ratio ~4e3-6e3 across seeds); 100x is a safe floor.
        assert errs[0] > errs[-1] * 100, f"seed={seed}: errs={errs}"

    # Discriminating check (the real work, per CLAUDE.md's near-field-
    # dominance warning): even the sweep's *worst* point (smallest k) must
    # still beat a leaves-only (far field dropped) baseline by a wide margin.
    # `leaves_only` is independent of k/p (leaf blocks are dense/exact,
    # untouched by any admissible-block factorization), so it is computed
    # once here rather than once per (seed, k).
    op = MockGF(mesh)
    hmat = compress(op, mesh, m=m, k=ks[0], p=p, seed=seeds[0], sampling="fixed")
    rel_err_worst = relative_error(hmat, op, seed=1)
    leaves_only = HMatrix(root=hmat.root, mesh=mesh, factors=[], leaves=hmat.leaves)
    leaves_only_rel_err = relative_error(leaves_only, op, seed=1)
    assert (
        rel_err_worst * 10 < leaves_only_rel_err
    ), f"rel_err_worst={rel_err_worst}, leaves_only={leaves_only_rel_err}"

    print(f"[k-sweep] setup_time={time.time() - t0:.2f}s leaves_only={leaves_only_rel_err:.3e}")


def test_p_sweep_monotonic_error_and_exact_matvec_counts() -> None:
    """Same 32x32/m=16 fixture, k=8 fixed, sweeping oversampling p. Measured
    (seeds 0-9, k=8): rel_err ~1.7e-6-1.1e-4 (p=0, not discriminating -- see
    the in-test comment below) down to ~1.3-1.4e-9 (p=16), monotonically
    non-increasing at every seed tried (0-9; 2 seeds asserted here to bound
    runtime)."""
    mesh = _grid_mesh(32, 32)
    m, k = 16, 8
    ps = [0, 2, 4, 8, 16]
    seeds = [0, 1]

    t0 = time.time()
    for seed in seeds:
        errs = []
        for p in ps:
            observed, rel_err = _observed_counts(mesh, m, k, p, seed)
            predicted = _predicted_counts(mesh, m, k, p)
            assert (
                observed == predicted
            ), f"seed={seed} p={p}: observed={observed}, predicted={predicted}"
            errs.append(rel_err)

        for i in range(len(errs) - 1):
            assert errs[i] >= errs[i + 1] - 1e-14, (
                f"seed={seed}: error increased going p={ps[i]} -> p={ps[i + 1]}: "
                f"{errs[i]} -> {errs[i + 1]}"
            )
        # p=0 is NOT a discriminating point: measured rel_err 1.7e-6-1.1e-4
        # across seeds 0-9 vs. leaves_only 7.2e-6, i.e. at p=0 the compressed
        # far field can be *worse* than dropping it entirely. The
        # discriminating comparison is therefore p=16 (~1.4e-9, three orders
        # below leaves-only); p=0 contributes only the start of the monotone
        # trend and its exact matvec count.
        op = MockGF(mesh)
        hmat = compress(op, mesh, m=m, k=k, p=ps[-1], seed=seed, sampling="fixed")
        leaves_only = HMatrix(root=hmat.root, mesh=mesh, factors=[], leaves=hmat.leaves)
        leaves_only_rel_err = relative_error(leaves_only, op, seed=1)
        assert (
            errs[-1] * 100 < leaves_only_rel_err
        ), f"seed={seed}: errs[-1]={errs[-1]}, leaves_only={leaves_only_rel_err}"

    print(f"[p-sweep] setup_time={time.time() - t0:.2f}s")


# ---------------------------------------------------------------------------
# m sweep: leaf size controls tree depth (quantized by the dyadic 4x-per-level
# split in 2D), which is exactly the k<->m coupling documented on
# `compress()`. 32x32 grid, k=4/p=4 fixed (valid at every m below, since the
# narrowest m=8 leaf still has width 4 -> bound 4).
# ---------------------------------------------------------------------------


def test_m_sweep_leaf_size_changes_depth_error_and_matvec_counts() -> None:
    """Measured (seeds 0-3, k=4, p=4): rel_err decreases monotonically as m
    grows (~9e-8 at m=8 down to ~4-5e-8 at m=100) at every seed tried -- an
    empirical trend on this fixture, not a general claim (coarser leaves here
    both shrink the tree and grow the exact dense near-field blocks, and both
    measured effects point the same way; not derived from a general
    property). 2 of the 4 measured seeds are asserted here to bound
    runtime."""
    mesh = _grid_mesh(32, 32)
    k, p = 4, 4
    ms = [8, 20, 100]
    expected_depths = {8: 4, 20: 3, 100: 2}
    seeds = [0, 1]

    t0 = time.time()
    for m in ms:
        root = build_tree(mesh, m)
        assert _deepest_level(root) == expected_depths[m], f"m={m}"

    for seed in seeds:
        errs = []
        for m in ms:
            observed, rel_err = _observed_counts(mesh, m, k, p, seed)
            predicted = _predicted_counts(mesh, m, k, p)
            assert (
                observed == predicted
            ), f"seed={seed} m={m}: observed={observed}, predicted={predicted}"
            errs.append(rel_err)

        for i in range(len(errs) - 1):
            assert errs[i] >= errs[i + 1] - 1e-14, (
                f"seed={seed}: error increased going m={ms[i]} -> m={ms[i + 1]}: "
                f"{errs[i]} -> {errs[i + 1]}"
            )

    # Discriminating check at the shallowest tree (m=100, fewest compressed
    # levels -- the point most at risk of a vacuous pass): still beats
    # leaves-only by a wide margin (measured leaves_only ~2.65e-6 vs.
    # rel_err ~4-5e-8, ~50-65x).
    m = ms[-1]
    op = MockGF(mesh)
    hmat = compress(op, mesh, m=m, k=k, p=p, seed=0, sampling="fixed")
    rel_err = relative_error(hmat, op, seed=1)
    leaves_only = HMatrix(root=hmat.root, mesh=mesh, factors=[], leaves=hmat.leaves)
    leaves_only_rel_err = relative_error(leaves_only, op, seed=1)
    assert (
        rel_err * 20 < leaves_only_rel_err
    ), f"rel_err={rel_err}, leaves_only={leaves_only_rel_err}"

    print(f"[m-sweep] setup_time={time.time() - t0:.2f}s")
