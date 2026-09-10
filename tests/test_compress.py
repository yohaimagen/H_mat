"""Tests for the `compress()` driver (Algorithm 4.1, end to end) and
`CountingOperator` (Task 5.6).

Per CLAUDE.md's testing rule, all accuracy checks use the smooth `MockGF`
kernel (genuine spectral decay on admissible blocks), never a random dense
matrix, and are compared against the reference operator only through
`matvec`/`rmatvec` (`gfcompress.error.relative_error`'s power method).

Two defects sent earlier tasks back are guarded against explicitly here:

1. (Task 5.5's defect) An absolute `relative_error(H, A) < tol` threshold
   alone cannot certify the compressed far field matters, because `MockGF`'s
   near-diagonal self-interaction dominates `||A||` -- a leaves-only `HMatrix`
   (`factors=[]`) can also sit below a plausible-looking `tol`. Every accuracy
   test below also builds that leaves-only baseline and requires the real
   `compress()` output to beat it by a wide (10x) margin.
2. (Task 5.5's defect) `k < rank` must be verified, not assumed at every
   level: the mesh/level combination used here (2D `16x16`/`m=4`/`k=4`) is
   checked directly against `MockGF.block`'s singular values at level 2 (the
   coarsest admissible level), where the gap `sigma_{k+1}/sigma_1` is
   genuinely small (blocks are 32x16 or larger, well above `k=4`). Level 3's
   boxes hold only `m=4` patches, so `beta.col_indices` has width `<= 4 == k`:
   there `k >= rank` and truncation is a no-op (matches the same finding
   already documented in `tests/test_hmatrix.py`).

The matvec-count test computes its "predicted" total independently, straight
from `gfcompress.fixed_pattern.build_admissible_test_matrices`/
`build_leaf_test_matrices` (the actual per-level test-matrix counts), not by
copying a formula that mirrors the driver's internals.
"""

from __future__ import annotations

import inspect

import numpy as np

from gfcompress.build_tree import build_tree
from gfcompress.column_basis import column_bases
from gfcompress.compress import CountingOperator, compress, compress_level
from gfcompress.error import relative_error
from gfcompress.fixed_pattern import build_admissible_test_matrices, build_leaf_test_matrices
from gfcompress.geometry import FaultMesh
from gfcompress.hmatrix import HMatrix
from gfcompress.interactions import build_lists
from gfcompress.mockgf import MockGF
from gfcompress.operators import MatVecOperator
from gfcompress.randomized import gaussian
from gfcompress.row_basis import row_bases


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


# ---------------------------------------------------------------------------
# sampling="coloring" is Stage 7, not yet implemented -- rejected, not silently
# ignored.
# ---------------------------------------------------------------------------


def test_compress_rejects_unsupported_sampling() -> None:
    mesh = _grid_mesh(8, 8)
    op = MockGF(mesh)
    try:
        compress(op, mesh, m=4, k=2, p=2, seed=0, sampling="coloring")
        raise AssertionError("expected ValueError for sampling='coloring'")
    except ValueError as exc:
        assert "coloring" in str(exc)


def test_public_oversampling_defaults_are_consistently_positive() -> None:
    """`p=0` remains explicit, but no public sampling path defaults to it."""
    for function in (
        gaussian,
        build_admissible_test_matrices,
        column_bases,
        row_bases,
        compress_level,
        compress,
    ):
        assert inspect.signature(function).parameters["p"].default == 10


class _NoSamplingOperator(MatVecOperator):
    """Operator double that records any accidental pre-validation sampling."""

    def __init__(self, shape: tuple[object, object]) -> None:
        self._shape = shape
        self.calls = 0

    def matvec(self, omega: np.ndarray) -> np.ndarray:
        self.calls += 1
        raise AssertionError("compress sampled an invalid input")

    def rmatvec(self, psi: np.ndarray) -> np.ndarray:
        self.calls += 1
        raise AssertionError("compress sampled an invalid input")

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape


class _MissingProductsOperator:
    """Shape-correct object with no callable black-box products."""

    def __init__(self, shape: tuple[int, int], *, noncallable: bool) -> None:
        self.shape = shape
        self.calls = 0
        if noncallable:
            self.matvec = None
            self.rmatvec = None


def test_compress_validates_inputs_before_sampling() -> None:
    mesh = _grid_mesh(4, 4)
    cases = [
        (_NoSamplingOperator((mesh.n_rows - 1, mesh.n_cols)), dict(m=1, k=1, p=0)),
        (_NoSamplingOperator((mesh.n_rows, mesh.n_cols)), dict(m=0, k=1, p=0)),
        (_NoSamplingOperator((mesh.n_rows, mesh.n_cols)), dict(m=1, k=0, p=0)),
        (_NoSamplingOperator((mesh.n_rows, mesh.n_cols)), dict(m=1, k=1, p=-1)),
        (_NoSamplingOperator((float(mesh.n_rows), mesh.n_cols)), dict(m=1, k=1, p=0)),
        (_NoSamplingOperator((True, mesh.n_cols)), dict(m=1, k=1, p=0)),
        (_NoSamplingOperator(("rows", mesh.n_cols)), dict(m=1, k=1, p=0)),
        (
            _MissingProductsOperator((mesh.n_rows, mesh.n_cols), noncallable=False),
            dict(m=1, k=1, p=0),
        ),
        (
            _MissingProductsOperator((mesh.n_rows, mesh.n_cols), noncallable=True),
            dict(m=1, k=1, p=0),
        ),
    ]
    for operator, options in cases:
        try:
            compress(operator, mesh, seed=0, sampling="fixed", **options)
            raise AssertionError("expected invalid compression input to fail")
        except ValueError:
            assert operator.calls == 0


# ---------------------------------------------------------------------------
# End-to-end accuracy: relative_error(H, A) < tol, with a discriminating
# leaves-only baseline showing the compressed far field measurably matters.
# ---------------------------------------------------------------------------


def test_compress_relative_error_2d() -> None:
    mesh = _grid_mesh(16, 16)
    op = MockGF(mesh)
    m, k, p, seed = 4, 4, 4, 0

    # k < rank check at level 2 (the only level here with blocks bigger than
    # k -- see module docstring): genuine truncation, not a degenerate no-op.
    root = build_tree(mesh, m)
    lists = build_lists(root)
    for alpha in root.nodes_at_level(2):
        for beta in lists.interaction[alpha]:
            block = op.block(alpha.patch_indices, beta.patch_indices)
            sv = np.linalg.svd(block, compute_uv=False)
            assert len(sv) > k, "level-2 blocks must be bigger than k for a genuine k < rank test"
            gap = sv[k] / sv[0]
            assert gap < 1e-2, f"block is not genuinely low rank, sigma_(k+1)/sigma_1={gap}"

    hmat = compress(op, mesh, m=m, k=k, p=p, seed=seed, sampling="fixed")

    rel_err = relative_error(hmat, op, seed=1)
    tol = 1e-6
    assert rel_err < tol, f"rel_err={rel_err} (measured), tol={tol}"

    # Discriminating: a leaves-only HMatrix (far field dropped entirely) must
    # be measurably worse, or tol alone would be vacuous (Task 5.5's defect).
    leaves_only = HMatrix(root=hmat.root, mesh=mesh, factors=[], leaves=hmat.leaves)
    rel_err_leaves_only = relative_error(leaves_only, op, seed=1)
    assert (
        rel_err * 10 < rel_err_leaves_only
    ), f"rel_err={rel_err}, rel_err_leaves_only={rel_err_leaves_only}"


def test_compress_relative_error_3d() -> None:
    mesh = _grid_mesh(8, 8, 8)
    op = MockGF(mesh)
    m, k, p, seed = 8, 8, 8, 1

    hmat = compress(op, mesh, m=m, k=k, p=p, seed=seed, sampling="fixed")

    rel_err = relative_error(hmat, op, seed=2)
    tol = 1e-9
    assert rel_err < tol, f"rel_err={rel_err} (measured), tol={tol}"

    leaves_only = HMatrix(root=hmat.root, mesh=mesh, factors=[], leaves=hmat.leaves)
    rel_err_leaves_only = relative_error(leaves_only, op, seed=2)
    assert (
        rel_err * 10 < rel_err_leaves_only
    ), f"rel_err={rel_err}, rel_err_leaves_only={rel_err_leaves_only}"


def test_repeated_seed_oversampling_diagnostic() -> None:
    """Keep construction and validation streams separate in this diagnostic.

    This records both the public default (`p=10`) and the explicit `p=0`
    experiment across several construction seeds. Randomized sketches do not
    support a useful per-seed monotonic-error assertion, so this only guards
    that every measured configuration remains finite; the measured values are
    tracked in ``MEASUREMENTS_C4.md``.
    """
    mesh = _grid_mesh(16, 16)
    op = MockGF(mesh)
    errors: dict[int, list[float]] = {0: [], 10: []}
    for p in errors:
        for construction_seed in (0, 1, 2):
            hmat = compress(op, mesh, m=4, k=4, p=p, seed=construction_seed)
            errors[p].append(relative_error(hmat, op, seed=1000 + construction_seed))

    assert all(np.isfinite(error) for values in errors.values() for error in values)


# ---------------------------------------------------------------------------
# Matvec count: exact prediction from the actual per-level test-matrix
# counts, plus the structural bounds and the N-independence claim.
# ---------------------------------------------------------------------------


def _predicted_counts(
    mesh: FaultMesh, m: int, k: int, p: int
) -> tuple[int, dict[int, tuple[int, int]], int, int]:
    """Independently recompute the exact predicted matvec-column total.

    Returns `(total, per_level_counts, leaf_probe_width, w_max)` where
    `per_level_counts[level] = (n_Omega(level), n_Psi(level))`, from directly
    calling `build_admissible_test_matrices`/`build_leaf_test_matrices` (the
    same builders `compress` uses internally, but counted independently here
    rather than trusting the driver's own bookkeeping).
    """
    root = build_tree(mesh, m)
    leaf_level = _deepest_level(root)

    total = 0
    per_level: dict[int, tuple[int, int]] = {}
    for level in range(2, leaf_level + 1):
        omegas = build_admissible_test_matrices(root, level, mesh, k, p, side="col")
        psis = build_admissible_test_matrices(root, level, mesh, k, p, side="row")
        per_level[level] = (len(omegas), len(psis))
        total += (len(omegas) + len(psis)) * (k + p)

    leaf_tms = build_leaf_test_matrices(root, leaf_level, mesh)
    w_max = max(len(box.col_indices) for box in root.nodes_at_level(leaf_level))
    leaf_probe_width = len(leaf_tms) * w_max
    total += leaf_probe_width

    return total, per_level, leaf_probe_width, w_max


def test_matvec_count_matches_exact_prediction_2d() -> None:
    mesh = _grid_mesh(16, 16)
    m, k, p, seed = 4, 4, 4, 3
    d = mesh.tree_dim

    predicted_total, per_level, leaf_probe_width, w_max = _predicted_counts(mesh, m, k, p)

    for n_omega, n_psi in per_level.values():
        assert n_omega <= 6**d
        assert n_psi <= 6**d
    assert leaf_probe_width <= 3**d * w_max

    counting_op = CountingOperator(MockGF(mesh))
    compress(counting_op, mesh, m=m, k=k, p=p, seed=seed, sampling="fixed")

    observed_total = counting_op.matvec_columns + counting_op.rmatvec_columns
    assert observed_total == predicted_total, (
        f"observed={observed_total}, predicted={predicted_total}, "
        f"per_level={per_level}, leaf_probe_width={leaf_probe_width}"
    )
    # Call counts fall out for free.
    assert counting_op.matvec_calls > 0
    assert counting_op.rmatvec_calls > 0


def test_matvec_count_matches_exact_prediction_3d() -> None:
    mesh = _grid_mesh(8, 8, 8)
    m, k, p, seed = 8, 8, 8, 4
    d = mesh.tree_dim

    predicted_total, per_level, leaf_probe_width, w_max = _predicted_counts(mesh, m, k, p)

    for n_omega, n_psi in per_level.values():
        assert n_omega <= 6**d
        assert n_psi <= 6**d
    assert leaf_probe_width <= 3**d * w_max

    counting_op = CountingOperator(MockGF(mesh))
    compress(counting_op, mesh, m=m, k=k, p=p, seed=seed, sampling="fixed")

    observed_total = counting_op.matvec_columns + counting_op.rmatvec_columns
    assert observed_total == predicted_total, (
        f"observed={observed_total}, predicted={predicted_total}, "
        f"per_level={per_level}, leaf_probe_width={leaf_probe_width}"
    )


def test_matvec_count_independent_of_n_at_fixed_depth() -> None:
    """Two 2D grids of very different patch count `N` (144 vs. 256), same
    leaf threshold `m=4` and hence the same tree depth `L=3`: the per-level
    admissible test-matrix counts -- and so the whole predicted total -- are
    *exactly* equal, because `build_admissible_test_matrices` groups boxes by
    a fixed-size (`6^d`) periodic pattern that depends only on grid
    coordinates mod 6, never on how many boxes/patches exist. This is the
    concrete sense in which Sec. 4.1's matvec count "does not scale with N":
    it scales with the number of levels and the (bounded) probe pattern, and
    is literally unchanged here despite `N` growing by 78%."""
    mesh_small = _grid_mesh(12, 12)
    mesh_large = _grid_mesh(16, 16)
    m, k, p = 4, 4, 4

    root_small = build_tree(mesh_small, m)
    root_large = build_tree(mesh_large, m)
    assert _deepest_level(root_small) == _deepest_level(root_large) == 3
    assert mesh_small.n_patches != mesh_large.n_patches

    total_small, per_level_small, _, _ = _predicted_counts(mesh_small, m, k, p)
    total_large, per_level_large, _, _ = _predicted_counts(mesh_large, m, k, p)

    assert per_level_small == per_level_large
    assert total_small == total_large


# ---------------------------------------------------------------------------
# CountingOperator as a standalone unit: counts columns (not calls), forwards
# results, exposes call counts, and reports the wrapped operator's shape.
# ---------------------------------------------------------------------------


def test_counting_operator_counts_columns_not_calls() -> None:
    mesh = _grid_mesh(4, 4)
    op = MockGF(mesh)
    counting_op = CountingOperator(op)
    assert counting_op.shape == op.shape

    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(mesh.n_cols)
    x_thick = rng.standard_normal((mesh.n_cols, 5))
    y1 = rng.standard_normal(mesh.n_rows)
    y_thick = rng.standard_normal((mesh.n_rows, 3))

    np.testing.assert_allclose(counting_op.matvec(x1), op.matvec(x1))
    np.testing.assert_allclose(counting_op.matvec(x_thick), op.matvec(x_thick))
    np.testing.assert_allclose(counting_op.rmatvec(y1), op.rmatvec(y1))
    np.testing.assert_allclose(counting_op.rmatvec(y_thick), op.rmatvec(y_thick))

    assert counting_op.matvec_calls == 2
    assert counting_op.matvec_columns == 1 + 5
    assert counting_op.rmatvec_calls == 2
    assert counting_op.rmatvec_columns == 1 + 3
