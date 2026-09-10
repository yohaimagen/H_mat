"""Tests for FaultMesh / Patch geometry helpers (Task 1.1), tree/dof
decoupling and PCA alignment (Task R.2)."""

import warnings

import numpy as np
import pytest

from gfcompress.build_tree import build_tree
from gfcompress.geometry import FaultMesh, Patch, pairwise_distances, pca_align


def _random_mesh(d: int, n: int, seed: int = 0) -> FaultMesh:
    rng = np.random.default_rng(seed)
    centroids = rng.uniform(-1.0, 1.0, size=(n, d))
    L = rng.uniform(0.1, 0.5, size=n)
    return FaultMesh(centroids=centroids, L=L)


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------


def test_patch_2d() -> None:
    p = Patch(centroid=np.array([1.0, 2.0]), L=0.5)
    assert p.centroid.shape == (2,)
    assert p.L == 0.5


def test_patch_3d() -> None:
    p = Patch(centroid=np.array([1.0, 2.0, 3.0]), L=0.25)
    assert p.centroid.shape == (3,)


def test_patch_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        Patch(centroid=np.array([1.0]), L=0.5)
    with pytest.raises(ValueError):
        Patch(centroid=np.array([1.0, 2.0, 3.0, 4.0]), L=0.5)


# ---------------------------------------------------------------------------
# FaultMesh construction & derived dof
# ---------------------------------------------------------------------------


def test_faultmesh_2d_dof() -> None:
    mesh = _random_mesh(d=2, n=5)
    assert mesh.tree_dim == 2
    assert mesh.dof_row == 2
    assert mesh.dof_col == 1
    assert mesh.n_patches == 5
    assert mesh.n_rows == 2 * 5
    assert mesh.n_cols == 1 * 5


def test_faultmesh_3d_dof() -> None:
    mesh = _random_mesh(d=3, n=7)
    assert mesh.tree_dim == 3
    assert mesh.dof_row == 3
    assert mesh.dof_col == 2
    assert mesh.n_patches == 7
    assert mesh.n_rows == 3 * 7
    assert mesh.n_cols == 2 * 7


def test_faultmesh_bad_centroid_shape_raises() -> None:
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros((5, 4)), L=np.ones(5))
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros(5), L=np.ones(5))


def test_faultmesh_mismatched_L_raises() -> None:
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros((5, 2)), L=np.ones(4))


# ---------------------------------------------------------------------------
# tree_dim / dof_row decoupling (Task R.2, part 1): BP3 is 2D elasticity
# (dof_row=2) on a 1D fault (tree_dim=1) -- FaultMesh must accept that
# combination while defaulting to today's pre-R.2 behavior when dof_row is
# omitted.
# ---------------------------------------------------------------------------


def test_faultmesh_1d_tree_requires_explicit_dof_row() -> None:
    # A bare 1D centroid array has no elasticity dimension to infer: 1 is not
    # a valid dof_row (elasticity is 2D or 3D only), so this must raise
    # rather than silently picking one.
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros((5, 1)), L=np.ones(5))


def test_faultmesh_1d_tree_with_explicit_dof_row() -> None:
    n = 10
    rng = np.random.default_rng(0)
    centroids = rng.uniform(-1.0, 1.0, size=(n, 1))
    mesh = FaultMesh(centroids=centroids, L=np.full(n, 0.1), dof_row=2)
    assert mesh.tree_dim == 1
    assert mesh.dof_row == 2
    assert mesh.dof_col == 1
    assert mesh.n_rows == 2 * n
    assert mesh.n_cols == 1 * n


def test_faultmesh_2d_tree_with_explicit_dof_row_3() -> None:
    # BP7-like: a 2D (PCA-reduced) tree embedded in 3D elasticity.
    n = 8
    rng = np.random.default_rng(1)
    centroids = rng.uniform(-1.0, 1.0, size=(n, 2))
    mesh = FaultMesh(centroids=centroids, L=np.full(n, 0.1), dof_row=3)
    assert mesh.tree_dim == 2
    assert mesh.dof_row == 3
    assert mesh.dof_col == 2


def test_faultmesh_dof_row_defaults_to_tree_dim() -> None:
    # Pre-R.2 behavior, byte-for-byte: omitting dof_row ties it to tree_dim.
    mesh = FaultMesh(centroids=np.zeros((5, 2)), L=np.ones(5))
    assert mesh.dof_row == 2
    mesh3 = FaultMesh(centroids=np.zeros((5, 3)), L=np.ones(5))
    assert mesh3.dof_row == 3


def test_faultmesh_tree_dim_exceeds_dof_row_raises() -> None:
    # The tree can only live in a subspace of the elastic ambient space, not
    # a larger one.
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros((5, 3)), L=np.ones(5), dof_row=2)


def test_faultmesh_invalid_dof_row_raises() -> None:
    with pytest.raises(ValueError):
        FaultMesh(centroids=np.zeros((5, 2)), L=np.ones(5), dof_row=4)


# ---------------------------------------------------------------------------
# patch_to_rows / patch_to_cols
# ---------------------------------------------------------------------------


def test_patch_to_rows_2d_full_set_size_and_partition() -> None:
    n = 6
    mesh = _random_mesh(d=2, n=n)
    all_patches = np.arange(n)
    rows = mesh.patch_to_rows(all_patches)
    assert rows.shape == (mesh.dof_row * n,)
    assert rows.shape[0] == 2 * n
    # Full set must be a permutation of 0..2N-1 (complete, disjoint cover).
    assert sorted(rows.tolist()) == list(range(2 * n))


def test_patch_to_cols_2d_full_set_size_and_partition() -> None:
    n = 6
    mesh = _random_mesh(d=2, n=n)
    all_patches = np.arange(n)
    cols = mesh.patch_to_cols(all_patches)
    assert cols.shape == (mesh.dof_col * n,)
    assert cols.shape[0] == 1 * n
    assert sorted(cols.tolist()) == list(range(n))


def test_patch_to_rows_3d_full_set_size_and_partition() -> None:
    n = 5
    mesh = _random_mesh(d=3, n=n)
    all_patches = np.arange(n)
    rows = mesh.patch_to_rows(all_patches)
    assert rows.shape == (3 * n,)
    assert sorted(rows.tolist()) == list(range(3 * n))


def test_patch_to_cols_3d_full_set_size_and_partition() -> None:
    n = 5
    mesh = _random_mesh(d=3, n=n)
    all_patches = np.arange(n)
    cols = mesh.patch_to_cols(all_patches)
    assert cols.shape == (2 * n,)
    assert sorted(cols.tolist()) == list(range(2 * n))


def test_patch_to_rows_block_interleaved_2d() -> None:
    mesh = _random_mesh(d=2, n=4)
    # Patch-major, block-interleaved: patch i -> rows [2i, 2i+1].
    rows = mesh.patch_to_rows(np.array([0, 2, 3]))
    np.testing.assert_array_equal(rows, np.array([0, 1, 4, 5, 6, 7]))


def test_patch_to_cols_block_interleaved_3d() -> None:
    mesh = _random_mesh(d=3, n=4)
    # Patch-major, block-interleaved: patch i -> cols [2i, 2i+1] (dof_col=2).
    cols = mesh.patch_to_cols(np.array([1, 3]))
    np.testing.assert_array_equal(cols, np.array([2, 3, 6, 7]))


def test_patch_to_rows_3d_block_interleaved() -> None:
    mesh = _random_mesh(d=3, n=4)
    # patch i -> rows [3i, 3i+1, 3i+2] (dof_row=3).
    rows = mesh.patch_to_rows(np.array([0, 2]))
    np.testing.assert_array_equal(rows, np.array([0, 1, 2, 6, 7, 8]))


def test_disjoint_subsets_partition_full_index_set_2d() -> None:
    n = 8
    mesh = _random_mesh(d=2, n=n)
    left = np.arange(0, n // 2)
    right = np.arange(n // 2, n)

    rows_left = mesh.patch_to_rows(left)
    rows_right = mesh.patch_to_rows(right)

    assert set(rows_left.tolist()).isdisjoint(set(rows_right.tolist()))
    union = sorted(rows_left.tolist() + rows_right.tolist())
    assert union == list(range(mesh.n_rows))

    cols_left = mesh.patch_to_cols(left)
    cols_right = mesh.patch_to_cols(right)
    assert set(cols_left.tolist()).isdisjoint(set(cols_right.tolist()))
    union_cols = sorted(cols_left.tolist() + cols_right.tolist())
    assert union_cols == list(range(mesh.n_cols))


# ---------------------------------------------------------------------------
# pairwise_distances
# ---------------------------------------------------------------------------


def test_pairwise_distances_shape_and_zero_diagonal() -> None:
    mesh = _random_mesh(d=2, n=10)
    dist = pairwise_distances(mesh.centroids)
    assert dist.shape == (10, 10)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-12)
    # Symmetric.
    np.testing.assert_allclose(dist, dist.T, atol=1e-12)


def test_pairwise_distances_known_values_2d() -> None:
    centroids = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    dist = pairwise_distances(centroids)
    expected = np.array(
        [
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]
    )
    np.testing.assert_allclose(dist, expected, atol=1e-12)


def test_pairwise_distances_known_values_3d() -> None:
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    dist = pairwise_distances(centroids)
    np.testing.assert_allclose(dist[0, 1], 3.0, atol=1e-12)
    np.testing.assert_allclose(dist[1, 0], 3.0, atol=1e-12)


# ---------------------------------------------------------------------------
# pca_align (Task R.2, parts 2-3): PCA alignment + degenerate-axis dropping.
# ---------------------------------------------------------------------------


def test_pca_align_line_in_2d_drops_to_one_axis() -> None:
    # A line, embedded diagonally in 2D (BP3-like): exactly rank-1 up to
    # float roundoff, so the second axis' variance ratio must land at the
    # float64 noise floor (~1e-30), vastly below DEFAULT_PCA_VAR_TOL=1e-12.
    n = 50
    t = np.linspace(0.0, 40.0, n)
    theta = np.deg2rad(60.0)
    centroids = t[:, None] * np.array([[np.cos(theta), np.sin(theta)]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no axis here is remotely ambiguous
        result = pca_align(centroids)

    assert result.kept_axes == (0,)
    assert result.dropped_axes == (1,)
    assert result.variance_ratio[0] == pytest.approx(1.0, abs=1e-9)
    assert result.variance_ratio[1] < 1e-20
    assert result.centroids.shape == (n, 1)


def test_pca_align_plane_in_3d_drops_to_two_axes() -> None:
    # A plane in 3D (BP7-like): z is an exact linear combination of x, y, so
    # the cloud is exactly rank-2 up to float roundoff.
    rng = np.random.default_rng(2)
    n = 60
    xy = rng.uniform(-5.0, 5.0, size=(n, 2))
    z = 0.3 * xy[:, 0] - 0.7 * xy[:, 1] + 2.0
    centroids = np.column_stack([xy, z])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pca_align(centroids)

    assert result.kept_axes == (0, 1)
    assert result.dropped_axes == (2,)
    assert result.variance_ratio[2] < 1e-20
    assert result.centroids.shape == (n, 2)


def test_pca_align_is_translation_and_scale_invariant() -> None:
    t = np.linspace(-3.0, 4.0, 40)
    cloud = np.column_stack([t, 2.0 * t, -t])
    base = pca_align(cloud)
    moved = pca_align(1e3 * cloud + np.array([7e3, -3e3, 9e3]))

    assert base.kept_axes == moved.kept_axes == (0,)
    assert base.variance_ratio[0] == pytest.approx(moved.variance_ratio[0])
    assert moved.relative_projection_residual < 1e-14


def test_pca_align_slightly_curved_cloud_keeps_real_second_axis() -> None:
    t = np.linspace(-1.0, 1.0, 100)
    result = pca_align(np.column_stack([t, 1e-4 * t * t, np.zeros_like(t)]))

    assert result.kept_axes == (0, 1)
    assert result.dropped_axes == (2,)


def test_pca_align_plane_in_3d_improves_leaf_occupancy() -> None:
    # BP7-like (REALGF_PLAN.md Finding B's table): a near-planar 3D point
    # cloud whose out-of-plane thickness is at the true float64 roundoff
    # floor (~1e-17, matching BP7's measured y-extent of ~6e-17), not an
    # exaggerated jitter. Building the tree at full ambient dimension (3D)
    # wastes a whole axis bisecting noise the data does not have; the
    # PCA-reduced 2D tree does not. Effect is large and robust here (unlike
    # the marginal 2D line case below): min leaf occupancy 1 -> 4, far fewer
    # leaves, at the same `m`.
    rng = np.random.default_rng(5)
    nx = ny = 40
    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1)
    n = xy.shape[0]
    z = rng.uniform(-3e-17, 3e-17, size=n)
    centroids = np.column_stack([xy, z])
    lengths = np.full(n, 0.05)

    result = pca_align(centroids)
    assert result.kept_axes == (0, 1)
    assert result.dropped_axes == (2,)
    assert result.variance_ratio[2] < result.variance_threshold

    m = 16
    leaves_3d = build_tree(FaultMesh(centroids=centroids, L=lengths), m).leaves()
    occ_3d = np.array([len(box.patch_indices) for box in leaves_3d])

    leaves_2d = build_tree(FaultMesh(centroids=result.centroids, L=lengths, dof_row=3), m).leaves()
    occ_2d = np.array([len(box.patch_indices) for box in leaves_2d])

    assert len(leaves_2d) < len(leaves_3d)
    assert occ_2d.min() > occ_3d.min()


def test_pca_align_genuine_3d_cloud_keeps_all_axes() -> None:
    rng = np.random.default_rng(3)
    centroids = rng.uniform(-1.0, 1.0, size=(80, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pca_align(centroids)

    assert result.kept_axes == (0, 1, 2)
    assert result.dropped_axes == ()
    assert result.centroids.shape == (80, 3)
    assert np.all(result.variance_ratio > result.variance_threshold)


def test_pca_align_never_reduces_below_one_dimension() -> None:
    # All points numerically coincident: every ratio is (at best) noise, but
    # at least one axis must survive.
    centroids = np.zeros((10, 3))
    result = pca_align(centroids)
    assert result.kept_axes == (0,)
    assert result.centroids.shape == (10, 1)
    assert result.absolute_projection_residual == 0.0
    assert result.relative_projection_residual == 0.0


def test_pca_align_small_but_real_variation_is_kept() -> None:
    # The prior 1e-12 cutoff incorrectly discarded this genuine ~1e-16
    # variance ratio.  It is enormously above the float64 numerical-rank
    # cutoff and must therefore remain a tree axis by default.
    rng = np.random.default_rng(1)
    n = 100
    x = np.linspace(0.0, 100.0, n)
    y = rng.normal(0.0, 4e-7, n)
    centroids = np.stack([x, y], axis=1)

    result = pca_align(centroids)

    assert 1e-16 <= result.variance_ratio[1] < 1e-12
    assert result.kept_axes == (0, 1)
    assert result.dropped_axes == ()


def test_pca_align_rejects_invalid_tolerances_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        pca_align(np.zeros((5, 2)), var_tol=-1.0)
    with pytest.raises(ValueError):
        pca_align(np.zeros((5, 2)), var_tol=np.inf)
    with pytest.raises(ValueError):
        pca_align(np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError):
        pca_align(np.zeros((5, 2)), tree_dim=0)


def test_pca_align_rejects_empty_or_1d_input() -> None:
    with pytest.raises(ValueError):
        pca_align(np.zeros((0, 2)))
    with pytest.raises(ValueError):
        pca_align(np.zeros(5))


def test_pca_align_does_not_run_automatically_no_fixture_moves() -> None:
    # Hard constraint: FaultMesh/build_tree never call pca_align on their
    # own, so a mesh built directly from raw (non-degenerate) centroids is
    # completely unaffected by this module's existence.
    rng = np.random.default_rng(4)
    centroids = rng.uniform(-1.0, 1.0, size=(20, 2))
    mesh_a = FaultMesh(centroids=centroids, L=np.full(20, 0.1))
    mesh_b = FaultMesh(centroids=centroids, L=np.full(20, 0.1))
    np.testing.assert_array_equal(mesh_a.centroids, mesh_b.centroids)
    assert mesh_a.tree_dim == 2


def test_pca_align_override_can_drop_small_real_variation_with_diagnostics() -> None:
    # Reproduces REALGF_PLAN.md Finding A/B's mechanism synthetically, at
    # small scale: a diagonal line in 2D with Gauss-Lobatto-like node
    # clustering per "element" plus a perpendicular jitter of relative size
    # ~1.7e-8 (sigma=1e-6 on coordinates spanning ~60) -- NOT true float64
    # roundoff (~1e-15 relative): variance ratio here is ~3.4e-15, three
    # orders above BP3's measured 4.7e-30.  Automatic PCA keeps it; an
    # explicit model-reduction choice may drop it, with its real residual
    # reported rather than mislabeled as roundoff.
    rng = np.random.default_rng(0)
    n_elements = 60
    # Degree-3 Gauss-Lobatto-Legendre nodes on a unit element, clustered at
    # the edges.
    local = np.array([0.0, 0.5 - 0.5 / np.sqrt(5.0), 0.5 + 0.5 / np.sqrt(5.0), 1.0])
    t = np.concatenate([e + local for e in range(n_elements)])
    n = len(t)
    theta = np.deg2rad(58.0)
    direction = np.array([np.cos(theta), np.sin(theta)])
    noise = rng.normal(0.0, 1e-6, size=(n, 2))
    centroids = t[:, None] * direction[None, :] + noise
    lengths = np.full(n, 0.05)

    result = pca_align(centroids, tree_dim=1)
    assert result.kept_axes == (0,)
    assert result.dropped_axes == (1,)
    assert result.automatic_kept_axes == (0, 1)
    assert result.relative_projection_residual == pytest.approx(
        np.sqrt(result.variance_ratio[1]), rel=1e-6
    )

    m = 4
    mesh_2d = FaultMesh(centroids=centroids, L=lengths)
    leaves_2d = build_tree(mesh_2d, m).leaves()
    occ_2d = np.array([len(box.patch_indices) for box in leaves_2d])

    mesh_1d = FaultMesh(centroids=result.centroids, L=lengths, dof_row=2)
    leaves_1d = build_tree(mesh_1d, m).leaves()
    occ_1d = np.array([len(box.patch_indices) for box in leaves_1d])

    # The PCA-reduced 1D tree strictly improves on all three occupancy
    # measures (measured for this fixed seed/configuration).
    assert len(leaves_1d) < len(leaves_2d)
    assert occ_1d.min() > occ_2d.min()
    assert int((occ_1d <= 2).sum()) < int((occ_2d <= 2).sum())


def test_pca_align_reports_transform_and_roundoff_cutoff_uncertainty() -> None:
    # Explicit variance threshold makes the kept and dropped sides of a
    # borderline decision observable.  The cloud has axes just above and
    # below 1e-12, both inside the factor-ten warning band.
    rng = np.random.default_rng(8)
    x = np.linspace(-1.0, 1.0, 500)
    y = rng.normal(scale=1.5e-6, size=len(x))
    z = rng.normal(scale=4e-7, size=len(x))
    original = np.column_stack([x, y, z]) + np.array([1000.0, -20.0, 7.0])
    with pytest.warns(UserWarning, match="both retained and dropped"):
        result = pca_align(original, var_tol=1e-12)

    np.testing.assert_array_equal(result.original_centroids, original)
    np.testing.assert_allclose(
        result.centroids,
        (original - result.mean) @ result.rotation[list(result.kept_axes)].T,
    )
    assert result.kept_axes == (0, 1)
    assert result.dropped_axes == (2,)
    assert result.absolute_projection_residual > 0
    assert result.relative_projection_residual > 0
