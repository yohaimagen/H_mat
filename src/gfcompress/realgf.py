"""Reader for the real SCEC/SEAS Green's-function datasets (Task R.1).

See `REALGF_PLAN.md` for the full background; this module implements the
parts of that plan marked R.1: a PETSc-binary `Mat` reader, a `bp{3,7}_fbf_
coords.csv` reader, and a `RealGF` `MatVecOperator` that exposes the two
real operators (BP3, 2D elasticity; BP7, 3D elasticity) in the patch-major,
`dof_row`/`dof_col`-flattened layout `CLAUDE.md` requires -- **without ever
materialising the dense array**. The `nnz * 8` values block is `np.memmap`ped
and touched only through `matvec`/`rmatvec`.

File format
-----------
`gf_mat.bin` is a big-endian PETSc binary `Mat`, preceded by an 8-byte
application preamble (2 int32, ignored):

    offset 0  : int32 x2  preamble -- skip
    offset 8  : int32 x4  classid=1211216, rows, cols, nnz
    offset 24 : int32 x rows  row lengths (all == cols; the matrix is dense)
              : int32 x nnz   column indices (contiguous 0..cols-1 per row)
              : float64 x nnz values, row-major

`read_petsc_mat_header` validates `nnz == rows * cols`, sample-validates row
lengths and column-index contiguity, and checks the predicted end offset
`24 + rows*4 + nnz*4 + nnz*8` against the actual file size. The values block
is then `np.memmap`ped directly as `(rows, cols)`; the column-index block
(pure redundancy for a dense matrix) is never read beyond the sample.

Finding C -- patch-major layout (gate)
---------------------------------------
The file stores dofs blocked by component within each element:
`file_index = element*(comps*nbf) + comp*nbf + bf`. `CLAUDE.md` requires
patch-major flattening: `element*(comps*nbf) + bf*comps + comp`. For columns
this is confirmed directly from the CSV (BP7: `slip_comp` changes every `nbf`
columns; BP3 is trivially patch-major already, `dof_col = 1`). Rows have no
CSV, so the same blocked layout is *assumed* for rows and validated here by a
falsifiable check (see `tests/test_realgf.py::test_row_layout_evidence_*` and
the measurement below): for a probed patch, the block-norm of `A[:, j]` over
patch-sized row blocks should (a) decay with centroid distance from the
probed patch -- measured as the Spearman rank correlation `rho` between
distance and block-norm, which should be strongly negative -- and (b) be
largest on the probed patch's own (near-diagonal) block. Two wrong hypotheses
are checked as nulls: "file order is already patch-major" (no row
permutation), and "file order is comp-major globally" (all patches of
component 0, then all of component 1, ...).

Measured evidence (2026-09-07, `real_gfs/`, both datasets, several probed
patches): under the blocked-row hypothesis, `rho` in `[-0.996, -0.96]` and
the probed patch's own block is the argmax or within rank 2 of it, at every
probe on both datasets. Under the "already patch-major" null, `rho` is
markedly weaker, `[-0.51, -0.25]`, and the near-diagonal rank is far worse
(up to ~3965 of 5520 on BP7). Under the "comp-major globally" null, `rho` is
essentially zero (`|rho| < 0.1` at every probe on both datasets) -- the
scrambling destroys the decay signal entirely, even though the near-diagonal
rank under this null is noisy rather than uniformly bad (it can land near 0
by chance on some probes, since with only `dof_row` groups there are few ways
to scramble). `rho` is therefore the reliable discriminator for this null;
both wrong hypotheses are clearly distinguishable from the blocked hypothesis
by `rho` alone on both datasets; no probe found a mismatch under the blocked
hypothesis. This is evidence *for* the blocked hypothesis, gathered by direct
measurement, not proof for every one of the 14000/16560 columns -- treat any
future compression result on real data whose accuracy looks anomalously poor
as reason to re-run this check first.

`L` (patch characteristic length)
----------------------------------
`FaultMesh.L` is per-patch mesh metadata. It does not control the production
block partition: that partition is defined by the dyadic interaction lists
and leaf neighbors. `TreeNode.diam`/`bounding_box` are computed from patch
centroids and then overwritten by the dyadic cell; nothing outside
`geometry.py` reads `mesh.L`, so `l_method` has no effect on near/far blocks.
It is not a deferred eta-split control. GLL nodes cluster at element edges, so true
nearest-neighbour spacing varies by an order of magnitude *within one
element*. Three
candidates are implemented (`patch_length_candidates`):

- `"representative"` (**default**): `element_diam / nbf`, uniform across an
  element's nodes.
- `"voronoi"`: half the distance to the node's nearest neighbour within the
  same element -- the most locally faithful, but node-dependent.
- `"element"`: the full element diameter, assigned to every node it carries.

`"representative"` is the default: it agrees closely with `"voronoi"` (both
give each node a length tied to local mesh resolution, not to where within
the element it happens to sit) while being cheaper to compute and stable
under GLL clustering, and it avoids `"element"`'s over-conservatism.

Measured sensitivity is an **independent proxy experiment**, not a
measurement of the compressor's actual partition: it counts the fraction of
a patch's `k=30` nearest neighbours that a direct patch-pair test
`dist >= eta * max(L_i, L_j)` (eta=0.5) would call admissible, independently
of the tree (see `tests/test_realgf.py::test_L_candidate_sensitivity_*`).
`"representative"` and `"voronoi"` agree to
within a few percent (BP3: 0.991 vs 0.991; BP7: 0.931 vs 0.931), while
`"element"` differs sharply (BP3: 0.801; BP7: 0.302) -- using the full
element size for every node would classify far more near-neighbour pairs as
inadmissible, especially on BP7 where element sizes vary ~6x across the
mesh. The proxy result is materially sensitive to this choice; it does not
change or predict a production partition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh, PCAAlignment, pca_align
from gfcompress.operators import MatVecOperator

PETSC_MAT_CLASSID = 1211216
_HEADER_INT_COUNT = 6  # 2 preamble + classid, rows, cols, nnz
_HEADER_BYTES = _HEADER_INT_COUNT * 4

L_METHODS = ("representative", "voronoi", "element")


@dataclass(frozen=True)
class PetscMatHeader:
    """Parsed and validated header of a PETSc binary `Mat` file.

    Attributes:
        preamble: The 8-byte application preamble, as two int32s (unused,
            kept only for introspection).
        classid: PETSc `MAT_FILE_CLASSID`; must equal `PETSC_MAT_CLASSID`.
        rows: Number of matrix rows.
        cols: Number of matrix columns.
        nnz: Number of stored entries; validated `== rows * cols` (the
            matrix is stored fully dense).
        val_offset: Byte offset of the `float64 x nnz` values block.
    """

    preamble: tuple[int, int]
    classid: int
    rows: int
    cols: int
    nnz: int
    val_offset: int

    @property
    def predicted_end(self) -> int:
        """Predicted file size implied by the header: `val_offset + nnz*8`."""
        return self.val_offset + self.nnz * 8


def read_petsc_mat_header(path: str | Path, n_samples: int = 16, seed: int = 0) -> PetscMatHeader:
    """Parse and validate a PETSc binary `Mat` header.

    Reads the 8-byte preamble and the `classid, rows, cols, nnz` quadruple,
    validates `nnz == rows * cols`, sample-validates that row lengths all
    equal `cols` and that each sampled row's column indices are `0..cols-1`
    contiguous, and checks the predicted end offset against the file's
    actual size. Never reads the (large) values block.

    Args:
        path: Path to the `gf_mat.bin` file.
        n_samples: Number of additional randomly-sampled rows to validate,
            beyond the first, middle, and last row (which are always
            checked).
        seed: Seed for the random sample of rows.

    Returns:
        The parsed, validated header.

    Raises:
        ValueError: On any validation failure (bad classid, `nnz != rows *
            cols`, a sampled row length `!= cols`, non-contiguous column
            indices in a sampled row, or a predicted end offset that does
            not match the file's actual size).
    """
    path = Path(path)
    header_ints = np.fromfile(path, dtype=">i4", count=_HEADER_INT_COUNT)
    if header_ints.size != _HEADER_INT_COUNT:
        raise ValueError(f"{path}: file too short for a PETSc Mat header")

    preamble = (int(header_ints[0]), int(header_ints[1]))
    classid, rows, cols, nnz = (int(x) for x in header_ints[2:6])

    if classid != PETSC_MAT_CLASSID:
        raise ValueError(f"{path}: classid {classid} != expected {PETSC_MAT_CLASSID}")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{path}: invalid rows={rows}, cols={cols}")
    if nnz != rows * cols:
        raise ValueError(f"{path}: nnz={nnz} != rows*cols={rows * cols} (matrix must be dense)")

    rowlen_off = _HEADER_BYTES
    colidx_off = rowlen_off + rows * 4
    val_off = colidx_off + nnz * 4

    rowlens = np.memmap(path, dtype=">i4", mode="r", offset=rowlen_off, shape=(rows,))
    colidx = np.memmap(path, dtype=">i4", mode="r", offset=colidx_off, shape=(rows, cols))

    rng = np.random.default_rng(seed)
    extra = rng.integers(0, rows, size=min(n_samples, rows)).tolist()
    sample_rows = sorted({0, rows // 2, rows - 1, *extra})
    expected_row = np.arange(cols, dtype=">i4")
    for r in sample_rows:
        if int(rowlens[r]) != cols:
            raise ValueError(f"{path}: row {r} length {int(rowlens[r])} != cols={cols}")
        if not np.array_equal(colidx[r], expected_row):
            raise ValueError(f"{path}: row {r} column indices are not 0..{cols - 1} contiguous")

    predicted_end = val_off + nnz * 8
    actual_size = os.path.getsize(path)
    if predicted_end != actual_size:
        raise ValueError(
            f"{path}: predicted end offset {predicted_end} != actual file size {actual_size}"
        )

    return PetscMatHeader(
        preamble=preamble, classid=classid, rows=rows, cols=cols, nnz=nnz, val_offset=val_off
    )


@dataclass(frozen=True)
class _CoordsData:
    """Parsed `bp{3,7}_fbf_coords.csv`, reduced to patch-level quantities."""

    centroids: NDArray[np.float64]  # (N, d), indexed by patch_id
    n_elements: int
    nbf: int
    d: int
    dof_col: int
    element_of_patch: NDArray[np.intp]  # (N,)
    col_pm_to_raw: NDArray[np.intp]  # (dof_col*N,) patch-major col -> raw file col


def _load_coords_csv(path: str | Path, coord_tol: float = 1e-6) -> _CoordsData:
    """Load and validate `bp{3,7}_fbf_coords.csv`.

    Args:
        path: Path to the coordinates CSV.
        coord_tol: Absolute tolerance for cross-checking that a patch's
            coordinate is identical across its `slip_comp` entries (BP7).

    Returns:
        Parsed patch centroids and the column patch-major permutation.

    Raises:
        ValueError: If `col_idx` is not contiguous `0..len-1`, the
            `(element, basis_func)` pairs do not cover a complete
            `0..N-1` patch numbering, or a patch's coordinate disagrees
            across its `slip_comp` entries by more than `coord_tol`.
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw[None, :]

    col_idx = raw[:, 0].astype(np.int64)
    element = raw[:, 1].astype(np.int64)
    slip_comp = raw[:, 2].astype(np.int64)
    basis_func = raw[:, 3].astype(np.int64)
    coords = raw[:, 4:]
    d = coords.shape[1]

    n = len(raw)
    if not np.array_equal(col_idx, np.arange(n)):
        raise ValueError(f"{path}: col_idx is not contiguous 0..{n - 1}")

    n_elements = int(element.max()) + 1
    nbf = int(basis_func.max()) + 1
    dof_col = int(slip_comp.max()) + 1
    n_patches = n_elements * nbf
    if n != dof_col * n_patches:
        raise ValueError(
            f"{path}: {n} rows != dof_col*n_elements*nbf = "
            f"{dof_col}*{n_elements}*{nbf} = {dof_col * n_patches}"
        )

    patch_id = element * nbf + basis_func

    # Patch-major permutation (Finding C): file col = element*(comps*nbf) +
    # comp*nbf + bf; required = patch_id*dof_col + comp.
    target_col = patch_id * dof_col + slip_comp
    if not np.array_equal(np.sort(target_col), np.arange(n)):
        raise ValueError(f"{path}: patch-major column targets are not a permutation of 0..{n - 1}")
    col_pm_to_raw = np.empty(n, dtype=np.intp)
    col_pm_to_raw[target_col] = col_idx

    mask0 = slip_comp == 0
    patch_id0 = patch_id[mask0]
    order = np.argsort(patch_id0)
    if not np.array_equal(patch_id0[order], np.arange(n_patches)):
        raise ValueError(f"{path}: (element, basis_func) pairs do not cover 0..{n_patches - 1}")
    centroids = coords[mask0][order]
    element_of_patch = element[mask0][order]

    for comp in range(1, dof_col):
        mask_c = slip_comp == comp
        pid_c = patch_id[mask_c]
        order_c = np.argsort(pid_c)
        coords_c = coords[mask_c][order_c]
        if not np.allclose(coords_c, centroids, atol=coord_tol):
            raise ValueError(
                f"{path}: slip_comp={comp} coordinates disagree with slip_comp=0 "
                f"by more than {coord_tol} for some patch"
            )

    return _CoordsData(
        centroids=centroids,
        n_elements=n_elements,
        nbf=nbf,
        d=d,
        dof_col=dof_col,
        element_of_patch=element_of_patch,
        col_pm_to_raw=col_pm_to_raw,
    )


def _row_pm_to_raw(n_elements: int, nbf: int, dof_row: int) -> NDArray[np.intp]:
    """Row patch-major permutation, under the blocked-row-layout assumption.

    See the module docstring ("Finding C") for the evidence supporting this
    assumption. File row = `element*(dof_row*nbf) + comp*nbf + bf`; required
    = `patch_id*dof_row + comp` with `patch_id = element*nbf + bf`.

    Returns:
        `row_pm_to_raw`, shape `(dof_row*n_elements*nbf,)`: `row_pm_to_raw[i]`
        is the raw file row index of patch-major row `i`.
    """
    n_rows = dof_row * n_elements * nbf
    raw = np.arange(n_rows)
    element_r = raw // (dof_row * nbf)
    rem = raw % (dof_row * nbf)
    comp_r = rem // nbf
    bf_r = rem % nbf
    patch_id_r = element_r * nbf + bf_r
    target_row = patch_id_r * dof_row + comp_r
    if not np.array_equal(np.sort(target_row), np.arange(n_rows)):
        raise ValueError("row patch-major targets are not a permutation (unexpected shape)")
    row_pm_to_raw = np.empty(n_rows, dtype=np.intp)
    row_pm_to_raw[target_row] = raw
    return row_pm_to_raw


def element_diameters(
    centroids: NDArray[np.float64], element_of_patch: NDArray[np.intp], n_elements: int
) -> NDArray[np.float64]:
    """Per-element diameter: the max pairwise distance among its nodes.

    Args:
        centroids: Patch centroids, shape `(N, d)`.
        element_of_patch: Element index owning each patch, shape `(N,)`.
        n_elements: Number of elements.

    Returns:
        Array of shape `(n_elements,)`; `0.0` for a single-node element.
    """
    diam = np.zeros(n_elements)
    for e in range(n_elements):
        pts = centroids[element_of_patch == e]
        if len(pts) > 1:
            diff = pts[:, None, :] - pts[None, :, :]
            diam[e] = float(np.sqrt((diff * diff).sum(-1)).max())
    return diam


def patch_length_voronoi(
    centroids: NDArray[np.float64], element_of_patch: NDArray[np.intp], n_elements: int
) -> NDArray[np.float64]:
    """Per-patch dual/Voronoi length: half the distance to the nearest node
    within the same element.

    Args:
        centroids: Patch centroids, shape `(N, d)`.
        element_of_patch: Element index owning each patch, shape `(N,)`.
        n_elements: Number of elements.

    Returns:
        Array of shape `(N,)`; `0.0` for a single-node element.
    """
    n = len(centroids)
    length = np.zeros(n)
    for e in range(n_elements):
        idx = np.where(element_of_patch == e)[0]
        pts = centroids[idx]
        if len(pts) > 1:
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt((diff * diff).sum(-1))
            np.fill_diagonal(dist, np.inf)
            length[idx] = dist.min(axis=1) / 2.0
    return length


def patch_length_candidates(
    centroids: NDArray[np.float64], element_of_patch: NDArray[np.intp], n_elements: int, nbf: int
) -> dict[str, NDArray[np.float64]]:
    """The three candidate per-patch characteristic lengths `L` (see module
    docstring), keyed by `L_METHODS`.
    """
    diam = element_diameters(centroids, element_of_patch, n_elements)
    return {
        "representative": diam[element_of_patch] / nbf,
        "element": diam[element_of_patch],
        "voronoi": patch_length_voronoi(centroids, element_of_patch, n_elements),
    }


class RealGF(MatVecOperator):
    """Real elastostatic Green's-function operator, read from PETSc/CSV.

    Wraps `gf_mat.bin` (memory-mapped, never materialised) and `*_fbf_
    coords.csv`, exposing `matvec`/`rmatvec`/`shape` in the patch-major
    layout `CLAUDE.md` requires, plus a `FaultMesh` built from the CSV
    centroids. See the module docstring for the file format, the patch-major
    permutation (Finding C), and the `L` choice.

    Attributes:
        header: The validated `PetscMatHeader`.
        mat: The memory-mapped raw (file-order) values, shape `(header.rows,
            header.cols)`, big-endian `float64`. Read only via `matvec`/
            `rmatvec`.
        mesh: `FaultMesh` of the PCA-aligned retained tree coordinates, with
            `L` from `l_method`. This geometric transform does not alter the
            operator's patch-major component ordering or shape.
        alignment: The `PCAAlignment` retaining the original coordinates and
            tree-coordinate transform.
        dof_row: Row dofs per patch (`= mesh.dof_row`, the elastic problem
            dimension). It is independent of `mesh.tree_dim`: BP3 uses a
            1D tree with 2 row dofs and BP7 a 2D tree with 3 row dofs.
        dof_col: Column dofs per patch (`= mesh.dof_row - 1`).
        row_pm_to_raw: Patch-major row index -> raw file row index, shape
            `(n_rows,)`.
        col_pm_to_raw: Patch-major column index -> raw file column index,
            shape `(n_cols,)`.
        patch_length_candidates: The three `L` candidates (see module
            docstring), keyed by `L_METHODS`.
    """

    def __init__(
        self,
        mat_path: str | Path,
        csv_path: str | Path,
        l_method: str = "representative",
        header_sample_seed: int = 0,
    ) -> None:
        """Load a real GF operator.

        Args:
            mat_path: Path to `gf_mat.bin`.
            csv_path: Path to `bp{3,7}_fbf_coords.csv`.
            l_method: Which `L` candidate to use for `mesh.L`; one of
                `L_METHODS`. Defaults to `"representative"` (see module
                docstring for why).
            header_sample_seed: Seed for the header's row-sample validation.

        Raises:
            ValueError: If `l_method` is not in `L_METHODS`, if the header
                and CSV disagree on `N`/`dof_row`/`dof_col`, or on any
                validation failure in `read_petsc_mat_header` /
                `_load_coords_csv`.
        """
        if l_method not in L_METHODS:
            raise ValueError(f"l_method must be one of {L_METHODS}, got {l_method!r}")

        self.header = read_petsc_mat_header(mat_path, seed=header_sample_seed)
        coords = _load_coords_csv(csv_path)

        dof_row = coords.d
        dof_col = dof_row - 1
        n_patches = coords.n_elements * coords.nbf
        if dof_col != coords.dof_col:
            raise ValueError(
                f"CSV slip_comp count {coords.dof_col} != dof_col={dof_col} implied by "
                f"coordinate dimension d={dof_row}"
            )
        if self.header.rows != dof_row * n_patches:
            raise ValueError(
                f"header rows={self.header.rows} != dof_row*N = {dof_row}*{n_patches} "
                f"= {dof_row * n_patches}"
            )
        if self.header.cols != dof_col * n_patches:
            raise ValueError(
                f"header cols={self.header.cols} != dof_col*N = {dof_col}*{n_patches} "
                f"= {dof_col * n_patches}"
            )

        self.dof_row = dof_row
        self.dof_col = dof_col
        self.n_elements = coords.n_elements
        self.nbf = coords.nbf

        self.col_pm_to_raw = coords.col_pm_to_raw
        self.row_pm_to_raw = _row_pm_to_raw(coords.n_elements, coords.nbf, dof_row)
        self._inv_col = _invert_permutation(self.col_pm_to_raw)
        self._inv_row = _invert_permutation(self.row_pm_to_raw)

        self.patch_length_candidates = patch_length_candidates(
            coords.centroids, coords.element_of_patch, coords.n_elements, coords.nbf
        )
        # Geometry may be numerically lower dimensional, but operator components
        # remain in the original patch-major elasticity layout.
        self.alignment: PCAAlignment = pca_align(coords.centroids)
        self.mesh = FaultMesh(
            centroids=self.alignment.centroids,
            L=self.patch_length_candidates[l_method],
            dof_row=dof_row,
        )

        self.mat: NDArray[np.float64] = np.memmap(
            mat_path,
            dtype=">f8",
            mode="r",
            offset=self.header.val_offset,
            shape=(self.header.rows, self.header.cols),
        )

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `A @ omega` in patch-major order. See `MatVecOperator.matvec`."""
        omega_raw = omega[self._inv_col]
        y_raw = self.mat @ omega_raw
        result: NDArray[np.floating] = y_raw[self.row_pm_to_raw]
        return result

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `A* @ psi` in patch-major order. See `MatVecOperator.rmatvec`."""
        psi_raw = psi[self._inv_row]
        z_raw = self.mat.conj().T @ psi_raw
        result: NDArray[np.floating] = z_raw[self.col_pm_to_raw]
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """`(n_rows, n_cols) = (header.rows, header.cols)`."""
        return (self.header.rows, self.header.cols)


def _invert_permutation(perm: NDArray[np.intp]) -> NDArray[np.intp]:
    """Return `inv` such that `inv[perm[i]] == i` for all `i`."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv
