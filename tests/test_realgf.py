"""Tests for the real-GF reader `RealGF` (Task R.1).

Two groups:

- Synthetic tests (always run, no real data needed): a hand-crafted tiny
  PETSc `Mat` file with a known patch-major permutation, used to validate
  `read_petsc_mat_header`'s error paths and `RealGF`'s permutation/matvec
  logic against an independently-derived reference -- this is what actually
  proves the Finding-C permutation formula is implemented correctly.
- Real-data tests (`real_gfs/`, skipped cleanly when absent): shapes, header
  offset arithmetic, permutation round-trips, the falsifiable row-layout
  evidence check, and the `L`-candidate sensitivity measurement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from gfcompress.build_tree import build_tree
from gfcompress.interactions import DEFAULT_ETA, build_lists
from gfcompress.operators import DenseOperator
from gfcompress.realgf import (
    PETSC_MAT_CLASSID,
    RealGF,
    RealGFSubset,
    _load_coords_csv,
    _row_pm_to_raw,
    patch_length_candidates,
    read_petsc_mat_header,
    representative_patch_ids,
    representative_subset,
)

REAL_GFS = Path(__file__).resolve().parent.parent / "real_gfs"
BP3_DIR = REAL_GFS / "gf_bp3"
BP7_DIR = REAL_GFS / "gf_bp7"

_needs_bp3 = pytest.mark.skipif(
    not (BP3_DIR / "gf_mat.bin").exists(), reason="real_gfs/gf_bp3 not present"
)
_needs_bp7 = pytest.mark.skipif(
    not (BP7_DIR / "gf_mat.bin").exists(), reason="real_gfs/gf_bp7 not present"
)


# ------------------------------------------------------------------------
# Synthetic PETSc Mat writer
# ------------------------------------------------------------------------


def _write_petsc_mat(
    path: Path,
    values: NDArray[np.float64],
    classid: int = PETSC_MAT_CLASSID,
    nnz_override: int | None = None,
    corrupt_colidx_row: int | None = None,
    truncate_bytes: int = 0,
) -> None:
    """Write a minimal, fully-dense PETSc binary `Mat` file (big-endian)."""
    rows, cols = values.shape
    nnz = nnz_override if nnz_override is not None else rows * cols
    with open(path, "wb") as fh:
        np.array([40, cols], dtype=">i4").tofile(fh)  # application preamble
        np.array([classid, rows, cols, nnz], dtype=">i4").tofile(fh)
        np.full(rows, cols, dtype=">i4").tofile(fh)
        colidx = np.tile(np.arange(cols, dtype=">i4"), rows).reshape(rows, cols)
        if corrupt_colidx_row is not None:
            colidx[corrupt_colidx_row, 0] = cols  # out of contiguous range
        colidx.tofile(fh)
        values.astype(">f8").tofile(fh)
    if truncate_bytes:
        with open(path, "r+b") as fh:
            fh.truncate(fh.seek(0, 2) - truncate_bytes)


# ------------------------------------------------------------------------
# `read_petsc_mat_header`: synthetic, always run
# ------------------------------------------------------------------------


def test_header_valid_roundtrip(tmp_path: Path) -> None:
    values = np.arange(6 * 4, dtype=np.float64).reshape(6, 4)
    path = tmp_path / "mat.bin"
    _write_petsc_mat(path, values)

    header = read_petsc_mat_header(path)

    assert header.rows == 6
    assert header.cols == 4
    assert header.nnz == 24
    assert header.predicted_end == path.stat().st_size


def test_header_bad_classid_raises(tmp_path: Path) -> None:
    path = tmp_path / "mat.bin"
    _write_petsc_mat(path, np.zeros((3, 2)), classid=999)
    with pytest.raises(ValueError, match="classid"):
        read_petsc_mat_header(path)


def test_header_nnz_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "mat.bin"
    _write_petsc_mat(path, np.zeros((3, 2)), nnz_override=5)
    with pytest.raises(ValueError, match="nnz"):
        read_petsc_mat_header(path)


def test_header_noncontiguous_colidx_raises(tmp_path: Path) -> None:
    path = tmp_path / "mat.bin"
    _write_petsc_mat(path, np.zeros((6, 4)), corrupt_colidx_row=3)
    with pytest.raises(ValueError, match="column indices"):
        read_petsc_mat_header(path, n_samples=6, seed=1)


def test_header_size_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "mat.bin"
    _write_petsc_mat(path, np.zeros((6, 4)), truncate_bytes=8)
    with pytest.raises(ValueError, match="file size"):
        read_petsc_mat_header(path)


# ------------------------------------------------------------------------
# `RealGF` correctness on a tiny hand-derived synthetic dataset
# ------------------------------------------------------------------------
#
# 3D case (BP7-like): n_elements=2, nbf=2, dof_row=3, dof_col=2, N=4 patches.
# File col order: element*(comps*nbf) + comp*nbf + bf. Hand-derived below
# independently of gfcompress.realgf's formula, to actually catch a bug in
# it rather than just checking self-consistency.


def _tiny_synthetic_dataset(
    tmp_path: Path,
) -> tuple[Path, Path, NDArray[np.intp], NDArray[np.intp]]:
    # nbf must differ from both dof_row and dof_col: the within-element
    # (comp, bf) permutation is a dof x nbf transpose, which is self-inverse
    # (and so fails to catch a swapped permutation direction) exactly when
    # nbf == dof_row or nbf == dof_col.
    n_elements, nbf, dof_row, dof_col = 2, 4, 3, 2
    n_patches = n_elements * nbf  # 8
    n_rows = dof_row * n_patches  # 24
    n_cols = dof_col * n_patches  # 16

    # CSV: raw col c -> (element, comp, bf) via c = e*(comps*nbf) + comp*nbf + bf.
    rows_csv = []
    for e in range(n_elements):
        for comp in range(dof_col):
            for bf in range(nbf):
                c = e * (dof_col * nbf) + comp * nbf + bf
                patch = e * nbf + bf
                x, y, z = float(patch), float(patch) * 2, float(patch) * 3
                rows_csv.append((c, e, comp, bf, x, y, z))
    rows_csv.sort(key=lambda r: r[0])
    csv_path = tmp_path / "coords.csv"
    with open(csv_path, "w") as fh:
        fh.write("col_idx,element,slip_comp,basis_func,coord_0,coord_1,coord_2\n")
        for c, e, comp, bf, x, y, z in rows_csv:
            fh.write(f"{c},{e},{comp},{bf},{x},{y},{z}\n")

    # Hand-derived raw->patch-major target maps (independent of realgf.py).
    col_target = np.empty(n_cols, dtype=np.intp)
    for c, e, comp, bf, *_ in rows_csv:
        patch = e * nbf + bf
        col_target[c] = patch * dof_col + comp
    col_pm_to_raw_expected = np.empty(n_cols, dtype=np.intp)
    col_pm_to_raw_expected[col_target] = np.arange(n_cols)

    row_target = np.empty(n_rows, dtype=np.intp)
    for r in range(n_rows):
        e = r // (dof_row * nbf)
        rem = r % (dof_row * nbf)
        comp = rem // nbf
        bf = rem % nbf
        patch = e * nbf + bf
        row_target[r] = patch * dof_row + comp
    row_pm_to_raw_expected = np.empty(n_rows, dtype=np.intp)
    row_pm_to_raw_expected[row_target] = np.arange(n_rows)

    # Guard: a self-inverse permutation can't distinguish forward from
    # inverse, so it can't catch a swapped matvec/rmatvec direction (see the
    # comment above `nbf`). Fail loudly at collection time if either
    # permutation is accidentally self-inverse again.
    assert not np.array_equal(col_pm_to_raw_expected, np.argsort(col_pm_to_raw_expected))
    assert not np.array_equal(row_pm_to_raw_expected, np.argsort(row_pm_to_raw_expected))

    values = (100 * np.arange(n_rows)[:, None] + np.arange(n_cols)[None, :]).astype(np.float64)
    mat_path = tmp_path / "mat.bin"
    _write_petsc_mat(mat_path, values)

    return mat_path, csv_path, row_pm_to_raw_expected, col_pm_to_raw_expected


def test_realgf_tiny_synthetic_permutation_matches_hand_derived(tmp_path: Path) -> None:
    mat_path, csv_path, row_expected, col_expected = _tiny_synthetic_dataset(tmp_path)

    gf = RealGF(mat_path, csv_path)

    assert gf.shape == (24, 16)
    assert gf.dof_row == 3
    assert gf.dof_col == 2
    np.testing.assert_array_equal(gf.row_pm_to_raw, row_expected)
    np.testing.assert_array_equal(gf.col_pm_to_raw, col_expected)


def test_realgf_tiny_synthetic_matvec_rmatvec_match_dense(tmp_path: Path) -> None:
    mat_path, csv_path, row_expected, col_expected = _tiny_synthetic_dataset(tmp_path)
    gf = RealGF(mat_path, csv_path)

    raw_values = (100 * np.arange(24)[:, None] + np.arange(16)[None, :]).astype(np.float64)
    dense_pm = raw_values[np.ix_(row_expected, col_expected)]
    reference = DenseOperator(dense_pm)

    rng = np.random.default_rng(0)
    omega = rng.standard_normal((16, 3))
    psi = rng.standard_normal((24, 3))

    np.testing.assert_allclose(gf.matvec(omega), reference.matvec(omega))
    np.testing.assert_allclose(gf.rmatvec(psi), reference.rmatvec(psi))

    # Single-vector form too.
    omega1 = rng.standard_normal(16)
    np.testing.assert_allclose(gf.matvec(omega1), reference.matvec(omega1))


def test_realgf_component_ids_are_ordered_but_not_physical_labels(tmp_path: Path) -> None:
    """CSV component ids are verified; the row layout is an explicit inference."""
    mat_path, csv_path, _, _ = _tiny_synthetic_dataset(tmp_path)
    gf = RealGF(mat_path, csv_path)

    np.testing.assert_array_equal(gf.column_component_ids.reshape(-1, 2), [[0, 1]] * 8)
    np.testing.assert_array_equal(gf.row_component_ids.reshape(-1, 3), [[0, 1, 2]] * 8)
    assert not hasattr(gf, "column_component_labels")
    assert not hasattr(gf, "row_component_labels")


def test_realgf_tiny_synthetic_adjoint_and_component_sensitive_order(tmp_path: Path) -> None:
    mat_path, csv_path, row_expected, col_expected = _tiny_synthetic_dataset(tmp_path)
    gf = RealGF(mat_path, csv_path)
    raw = (100 * np.arange(24)[:, None] + np.arange(16)[None, :]).astype(float)
    dense_pm = raw[np.ix_(row_expected, col_expected)]

    # A component-specific impulse catches an otherwise easy-to-miss swap of
    # the two BP7 column components or any of the three row components.
    omega = np.zeros(16)
    omega[2 * 3 + 1] = 1.0
    psi = np.zeros(24)
    psi[3 * 5 + 2] = 1.0
    np.testing.assert_allclose(gf.matvec(omega), dense_pm[:, 7])
    np.testing.assert_allclose(gf.rmatvec(psi), dense_pm[17, :])
    assert np.vdot(gf.matvec(omega), psi) == pytest.approx(np.vdot(omega, gf.rmatvec(psi)))


def test_representative_subset_keeps_patch_major_maps_and_separated_regions(tmp_path: Path) -> None:
    mat_path, csv_path, _, _ = _tiny_synthetic_dataset(tmp_path)
    gf = RealGF(mat_path, csv_path)
    subset = representative_subset(gf, max_patches=6, regions=3)

    np.testing.assert_array_equal(
        subset.patch_ids, representative_patch_ids(gf.original_centroids, 6, 3)
    )
    np.testing.assert_array_equal(subset.row_map, gf.mesh.patch_to_rows(subset.patch_ids))
    np.testing.assert_array_equal(subset.col_map, gf.mesh.patch_to_cols(subset.patch_ids))
    assert subset.shape == (18, 12)
    assert np.unique(subset.patch_ids // gf.nbf).size == 2
    omega = np.arange(subset.shape[1], dtype=float)
    psi = np.arange(subset.shape[0], dtype=float)
    assert np.vdot(subset.matvec(omega), psi) == pytest.approx(np.vdot(omega, subset.rmatvec(psi)))


def test_load_coords_csv_col_permutation_is_bijection(tmp_path: Path) -> None:
    _, csv_path, _, col_expected = _tiny_synthetic_dataset(tmp_path)
    coords = _load_coords_csv(csv_path)
    np.testing.assert_array_equal(coords.col_pm_to_raw, col_expected)
    assert coords.centroids.shape == (8, 3)


def test_row_pm_to_raw_permutation_is_bijection() -> None:
    perm = _row_pm_to_raw(n_elements=2, nbf=2, dof_row=3)
    assert sorted(perm.tolist()) == list(range(12))


# ------------------------------------------------------------------------
# Real-data tests (skip cleanly if `real_gfs/` is absent)
# ------------------------------------------------------------------------

_BP3_MAT = BP3_DIR / "gf_mat.bin"
_BP3_CSV = BP3_DIR / "bp3_fbf_coords.csv"
_BP7_MAT = BP7_DIR / "gf_mat.bin"
_BP7_CSV = BP7_DIR / "bp7_fbf_coords.csv"


@_needs_bp3
def test_bp3_shape_and_header_offset() -> None:
    header = read_petsc_mat_header(_BP3_MAT)
    assert (header.rows, header.cols) == (28000, 14000)
    assert header.predicted_end == _BP3_MAT.stat().st_size

    gf = RealGF(_BP3_MAT, _BP3_CSV)
    assert gf.shape == (28000, 14000)
    assert gf.dof_row == 2
    assert gf.dof_col == 1
    assert gf.mesh.n_patches == 14000


@_needs_bp3
def test_bp3_pca_tree_preserves_operator_dofs_and_sizes() -> None:
    gf = RealGF(_BP3_MAT, _BP3_CSV)

    assert (gf.mesh.tree_dim, gf.mesh.dof_row, gf.mesh.dof_col) == (1, 2, 1)
    assert (gf.mesh.n_rows, gf.mesh.n_cols) == gf.shape
    assert gf.alignment.original_centroids.shape == (14000, 2)
    assert gf.alignment.centroids.shape == (14000, 1)


@_needs_bp7
def test_bp7_shape_and_header_offset() -> None:
    header = read_petsc_mat_header(_BP7_MAT)
    assert (header.rows, header.cols) == (16560, 11040)
    assert header.predicted_end == _BP7_MAT.stat().st_size

    gf = RealGF(_BP7_MAT, _BP7_CSV)
    assert gf.shape == (16560, 11040)
    assert gf.dof_row == 3
    assert gf.dof_col == 2
    assert gf.mesh.n_patches == 5520


@_needs_bp7
def test_bp7_pca_tree_preserves_operator_dofs_and_sizes() -> None:
    gf = RealGF(_BP7_MAT, _BP7_CSV)

    assert (gf.mesh.tree_dim, gf.mesh.dof_row, gf.mesh.dof_col) == (2, 3, 2)
    assert (gf.mesh.n_rows, gf.mesh.n_cols) == gf.shape
    assert gf.alignment.original_centroids.shape == (5520, 3)
    assert gf.alignment.centroids.shape == (5520, 2)


@_needs_bp3
def test_bp3_permutation_roundtrips() -> None:
    gf = RealGF(_BP3_MAT, _BP3_CSV)
    assert sorted(gf.row_pm_to_raw.tolist()) == list(range(gf.shape[0]))
    assert sorted(gf.col_pm_to_raw.tolist()) == list(range(gf.shape[1]))
    # BP3 has dof_col=1, so the column permutation must be the identity
    # (Finding C: BP3 is trivially patch-major already).
    np.testing.assert_array_equal(gf.col_pm_to_raw, np.arange(gf.shape[1]))


@_needs_bp7
def test_bp7_permutation_roundtrips() -> None:
    gf = RealGF(_BP7_MAT, _BP7_CSV)
    assert sorted(gf.row_pm_to_raw.tolist()) == list(range(gf.shape[0]))
    assert sorted(gf.col_pm_to_raw.tolist()) == list(range(gf.shape[1]))
    # BP7 columns are genuinely blocked (dof_col=2): patch-major requires a
    # non-trivial permutation.
    assert not np.array_equal(gf.col_pm_to_raw, np.arange(gf.shape[1]))


def _assert_adjoint(gf: RealGF) -> None:
    rng = np.random.default_rng(7)
    omega = rng.standard_normal((gf.shape[1], 2))
    psi = rng.standard_normal((gf.shape[0], 2))
    np.testing.assert_allclose(np.vdot(gf.matvec(omega), psi), np.vdot(omega, gf.rmatvec(psi)))


@_needs_bp3
def test_bp3_adjoint_consistency() -> None:
    _assert_adjoint(RealGF(_BP3_MAT, _BP3_CSV))


@_needs_bp7
def test_bp7_adjoint_consistency() -> None:
    _assert_adjoint(RealGF(_BP7_MAT, _BP7_CSV))


def _row_layout_probe(gf: RealGF, patch: int) -> tuple[float, float, float, int, int, int]:
    """Discriminating check (Finding C): |A| decay with centroid distance,
    and near-diagonal dominance, under the applied (correct) row
    permutation vs. two wrong hypotheses: "file order is already
    patch-major" (`wrong`), and "file order is comp-major globally", i.e.
    all patches of component 0, then all of component 1, ... (`compmajor`).

    Returns `(rho_correct, rho_wrong, rho_compmajor, rank_correct,
    rank_wrong, rank_compmajor)` where `rank` is how many patches have a
    *larger* block-norm than the probed patch's own block (0 = exact
    argmax).
    """
    n_patches = gf.mesh.n_patches
    raw_col = gf.col_pm_to_raw[patch * gf.dof_col]
    col_raw = np.asarray(gf.mat[:, raw_col])
    col_correct = col_raw[gf.row_pm_to_raw]
    block_correct = np.abs(col_correct.reshape(-1, gf.dof_row)).max(axis=1)
    block_wrong = np.abs(col_raw.reshape(-1, gf.dof_row)).max(axis=1)
    block_compmajor = np.abs(col_raw.reshape(gf.dof_row, n_patches)).max(axis=0)

    dist = np.linalg.norm(gf.mesh.centroids - gf.mesh.centroids[patch], axis=1)
    rho_correct, _ = spearmanr(dist, block_correct)
    rho_wrong, _ = spearmanr(dist, block_wrong)
    rho_compmajor, _ = spearmanr(dist, block_compmajor)
    rank_correct = int((block_correct > block_correct[patch]).sum())
    rank_wrong = int((block_wrong > block_wrong[patch]).sum())
    rank_compmajor = int((block_compmajor > block_compmajor[patch]).sum())
    return (
        float(rho_correct),
        float(rho_wrong),
        float(rho_compmajor),
        rank_correct,
        rank_wrong,
        rank_compmajor,
    )


@_needs_bp3
def test_row_layout_evidence_bp3() -> None:
    gf = RealGF(_BP3_MAT, _BP3_CSV)
    probes = [0, 7000, 13999]
    for patch in probes:
        rho_c, rho_w, rho_cm, rank_c, _rank_w, _rank_cm = _row_layout_probe(gf, patch)
        # Correct permutation: strong decay, near-diagonal dominant.
        assert rho_c < -0.85, f"patch {patch}: rho_correct={rho_c}"
        assert rank_c <= 5, f"patch {patch}: rank_correct={rank_c}"
        # Wrong ("already patch-major") permutation: measurably weaker.
        assert rho_w > rho_c + 0.3, f"patch {patch}: rho_wrong={rho_w} not weaker than {rho_c}"
        # Wrong ("comp-major globally") permutation: no measurable decay
        # (rank alone is not a reliable discriminator here: with only
        # `dof_row` groups, the near-diagonal block can land back near rank
        # 0 by chance under this null on some probes).
        assert abs(rho_cm) < 0.15, f"patch {patch}: rho_compmajor={rho_cm}"


@_needs_bp7
def test_row_layout_evidence_bp7() -> None:
    gf = RealGF(_BP7_MAT, _BP7_CSV)
    probes = [0, 2760, 5519]
    for patch in probes:
        rho_c, rho_w, rho_cm, rank_c, _rank_w, _rank_cm = _row_layout_probe(gf, patch)
        assert rho_c < -0.85, f"patch {patch}: rho_correct={rho_c}"
        assert rank_c <= 5, f"patch {patch}: rank_correct={rank_c}"
        assert rho_w > rho_c + 0.2, f"patch {patch}: rho_wrong={rho_w} not weaker than {rho_c}"
        # Wrong ("comp-major globally") permutation: no measurable decay
        # (rank alone is not a reliable discriminator here, see BP3 above).
        assert abs(rho_cm) < 0.15, f"patch {patch}: rho_compmajor={rho_cm}"


def _near_neighbor_admissible_fraction(
    centroids: NDArray[np.float64],
    length: NDArray[np.float64],
    k: int = 30,
    eta: float = DEFAULT_ETA,
) -> float:
    """Fraction of a point's `k` nearest neighbours classified admissible
    (`dist >= eta * max(L_i, L_j)`) -- where the choice of `L` actually
    matters, unlike a global random sample of pairs (almost all admissible
    regardless of `L`)."""
    tree = cKDTree(centroids)
    dist, idx = tree.query(centroids, k=k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    max_l = np.maximum(length[:, None], length[idx])
    return float((dist >= eta * max_l).mean())


@_needs_bp3
def test_L_candidate_sensitivity_bp3() -> None:
    coords = _load_coords_csv(_BP3_CSV)
    candidates = patch_length_candidates(
        coords.centroids, coords.element_of_patch, coords.n_elements, coords.nbf
    )
    fracs = {
        name: _near_neighbor_admissible_fraction(coords.centroids, length)
        for name, length in candidates.items()
    }
    # representative and voronoi agree closely; element differs materially.
    assert abs(fracs["representative"] - fracs["voronoi"]) < 0.05
    assert abs(fracs["representative"] - fracs["element"]) > 0.1


@_needs_bp7
def test_L_candidate_sensitivity_bp7() -> None:
    coords = _load_coords_csv(_BP7_CSV)
    candidates = patch_length_candidates(
        coords.centroids, coords.element_of_patch, coords.n_elements, coords.nbf
    )
    fracs = {
        name: _near_neighbor_admissible_fraction(coords.centroids, length)
        for name, length in candidates.items()
    }
    assert abs(fracs["representative"] - fracs["voronoi"]) < 0.05
    assert abs(fracs["representative"] - fracs["element"]) > 0.3


def _subset_diagnostics(gf: RealGF, regions: int) -> tuple[RealGFSubset, int, int, int, int, int]:
    """Fast, bounded real-data checks; never form the full source matrix."""
    subset = representative_subset(gf, max_patches=192, regions=regions)
    root = build_tree(subset.mesh, m=12)
    lists = build_lists(root)
    levels = list(root.iter_levels())
    leaves = levels[-1]
    interactions = sum(len(lists.interaction[node]) for nodes in levels[2:] for node in nodes)
    occupancies = [len(node.patch_indices) for node in leaves]
    # One genuine far-field block, recovered through a narrow matvec only.
    alpha = next(node for nodes in levels[2:] for node in nodes if lists.interaction[node])
    beta = lists.interaction[alpha][0]
    probe = np.zeros((subset.shape[1], len(beta.col_indices)))
    probe[beta.col_indices] = np.eye(len(beta.col_indices))
    block = subset.matvec(probe)[alpha.row_indices]
    singular_values = np.linalg.svd(block, compute_uv=False)
    rank = int((singular_values > singular_values[0] * 1e-4).sum())
    return subset, len(levels) - 1, interactions, max(occupancies), rank, min(block.shape)


@_needs_bp3
def test_bp3_subset_bp3_diagnostics_and_l_independence() -> None:
    gf = RealGF(_BP3_MAT, _BP3_CSV)
    subset, depth, interactions, occupancy, rank, block_width = _subset_diagnostics(gf, regions=4)
    alternate, *_ = _subset_diagnostics(gf, regions=6)
    assert subset.original_centroids.shape[1] == 2
    assert subset.reduced_centroids.shape[1] == 1
    assert depth >= 2 and interactions > 0 and occupancy <= 12
    assert 0 < rank < block_width  # numerical far-field rank at 1e-4, not an exact rank claim
    assert not np.array_equal(subset.patch_ids, alternate.patch_ids)
    # L is metadata only; it cannot change the dyadic partition.
    roots = [
        build_tree(RealGF(_BP3_MAT, _BP3_CSV, l_method=name).mesh, m=12)
        for name in ("representative", "voronoi", "element")
    ]
    assert [[node.cell_coords for node in list(root.iter_levels())[-1]] for root in roots].count(
        [node.cell_coords for node in list(roots[0].iter_levels())[-1]]
    ) == 3


@_needs_bp7
def test_bp7_subset_bp7_diagnostics_and_subset_sensitivity() -> None:
    gf = RealGF(_BP7_MAT, _BP7_CSV)
    first, depth, interactions, occupancy, rank, block_width = _subset_diagnostics(gf, regions=4)
    second, *_ = _subset_diagnostics(gf, regions=6)
    assert first.original_centroids.shape[1] == 3
    assert first.reduced_centroids.shape[1] == 2
    assert depth >= 2 and interactions > 0 and occupancy <= 12
    assert 0 < rank < block_width
    assert not np.array_equal(first.patch_ids, second.patch_ids)
