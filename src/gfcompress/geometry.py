"""Mesh and patch data structures for the dual-index cluster tree.

A `FaultMesh` holds the `N` patch centroids that drive the geometric cluster
tree shared by the row and column index sets. Per CLAUDE.md, the flattened
operator `A` maps `R^{dof_col * N} -> R^{dof_row * N}` with patch-major
flattening: each patch contributes `dof_row` consecutive scalar row indices
and `dof_col` consecutive scalar column indices, block-interleaved across
patches.

Tree dimension vs. elasticity dimension
----------------------------------------
`FaultMesh.tree_dim` (`= centroids.shape[1]`, in `{1, 2, 3}`) is the
dimensionality of the geometric cluster tree -- the space `build_tree` and
all the neighbor/interaction/pattern-cell combinatorics operate in.
`FaultMesh.dof_row`/`dof_col` (`in {(2, 1), (3, 2)}`) are fixed by the
*elastic* problem and drive nothing about tree geometry. The two coincide for
a fault mesh embedded at full ambient dimension (the only case this codebase
supported before Task R.2), but they need not: a 1D fault embedded in 2D
elasticity (SCEC/SEAS BP3) has `tree_dim=1`, `dof_row=2`. Pass `dof_row`
explicitly whenever `tree_dim` does not already equal the desired `dof_row`;
otherwise it defaults to `tree_dim` (preserving the pre-R.2 behavior exactly).
`pca_align` (below) is the tool that produces a reduced-`tree_dim` centroid
array from a full-ambient one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Patch:
    """A single mesh patch.

    Attributes:
        centroid: Patch centroid, shape `(d,)`.
        L: Characteristic length of the patch.
    """

    centroid: NDArray[np.float64]
    L: float

    def __post_init__(self) -> None:
        centroid = np.asarray(self.centroid, dtype=np.float64)
        if centroid.ndim != 1 or centroid.shape[0] not in (2, 3):
            raise ValueError(f"centroid must have shape (2,) or (3,), got {centroid.shape}")
        object.__setattr__(self, "centroid", centroid)


@dataclass
class FaultMesh:
    """Collection of `N` patches forming the geometric mesh.

    Attributes:
        centroids: Patch centroids, shape `(N, tree_dim)` with `tree_dim in
            {1, 2, 3}` -- the dimensionality of the geometric cluster tree,
            not necessarily the elasticity problem (see module docstring).
        L: Characteristic length per patch, shape `(N,)`.
        dof_row: Row degrees of freedom per patch (elastic problem
            dimension `d`, `in {2, 3}`). Defaults to `tree_dim` if not given
            explicitly, preserving pre-R.2 behavior when the tree is built at
            full ambient dimension.
        tree_dim: Dimensionality of the cluster tree (`= centroids.shape[1]`),
            `in {1, 2, 3}`, `<= dof_row`.
        dof_col: Column degrees of freedom per patch, equal to `dof_row - 1`.
    """

    centroids: NDArray[np.float64]
    L: NDArray[np.float64]
    dof_row: int | None = None
    tree_dim: int = field(init=False)
    dof_col: int = field(init=False)

    def __post_init__(self) -> None:
        centroids = np.asarray(self.centroids, dtype=np.float64)
        L = np.asarray(self.L, dtype=np.float64)

        if centroids.ndim != 2 or centroids.shape[1] not in (1, 2, 3):
            raise ValueError(
                f"centroids must have shape (N, 1), (N, 2) or (N, 3), got {centroids.shape}"
            )
        if L.ndim != 1 or L.shape[0] != centroids.shape[0]:
            raise ValueError(
                f"L must have shape (N,) matching centroids, got {L.shape} "
                f"vs centroids {centroids.shape}"
            )

        tree_dim = centroids.shape[1]
        dof_row = tree_dim if self.dof_row is None else self.dof_row
        if dof_row not in (2, 3):
            raise ValueError(
                f"dof_row must be 2 or 3 (the elastic problem's spatial dimension), "
                f"got {dof_row}"
            )
        if tree_dim > dof_row:
            raise ValueError(
                f"tree dimension {tree_dim} (centroids.shape[1]) cannot exceed "
                f"dof_row={dof_row}: the tree lives in a subspace of (or equal to) "
                "the elastic ambient space, never a larger one"
            )

        self.centroids = centroids
        self.L = L
        self.tree_dim = tree_dim
        self.dof_row = dof_row
        self.dof_col = dof_row - 1

    @property
    def n_patches(self) -> int:
        """Number of patches `N`."""
        return int(self.centroids.shape[0])

    @property
    def n_rows(self) -> int:
        """Total row dimension `dof_row * N`."""
        # __post_init__ always resolves dof_row to an int; the field stays
        # typed `int | None` only to accept the optional constructor override.
        assert self.dof_row is not None
        return self.dof_row * self.n_patches

    @property
    def n_cols(self) -> int:
        """Total column dimension `dof_col * N`."""
        return self.dof_col * self.n_patches

    def patch_to_rows(self, patch_ids: NDArray[np.integer]) -> NDArray[np.intp]:
        """Expand a set of patch indices to their flattened row indices.

        Patch-major flattening: patch `i` owns rows
        `[dof_row * i, dof_row * i + dof_row)`.

        Args:
            patch_ids: Integer array of patch indices, any shape.

        Returns:
            1D array of scalar row indices, length `dof_row * len(patch_ids)`,
            ordered patch-major (block-interleaved) following the input order.
        """
        assert self.dof_row is not None
        return _expand_indices(patch_ids, self.dof_row)

    def patch_to_cols(self, patch_ids: NDArray[np.integer]) -> NDArray[np.intp]:
        """Expand a set of patch indices to their flattened column indices.

        Patch-major flattening: patch `i` owns columns
        `[dof_col * i, dof_col * i + dof_col)`.

        Args:
            patch_ids: Integer array of patch indices, any shape.

        Returns:
            1D array of scalar column indices, length `dof_col * len(patch_ids)`,
            ordered patch-major (block-interleaved) following the input order.
        """
        return _expand_indices(patch_ids, self.dof_col)


def _expand_indices(patch_ids: NDArray[np.integer], dof: int) -> NDArray[np.intp]:
    """Expand patch indices into `dof` consecutive scalar indices each.

    For input `patch_ids = [i_0, i_1, ...]`, returns
    `[dof*i_0, dof*i_0+1, ..., dof*i_0+dof-1, dof*i_1, ...]`, i.e. patch-major
    block-interleaved flattening.
    """
    patch_ids = np.asarray(patch_ids, dtype=np.intp).reshape(-1)
    base = dof * patch_ids[:, None]
    offsets = np.arange(dof, dtype=np.intp)[None, :]
    return (base + offsets).reshape(-1)


def pairwise_distances(centroids: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized pairwise Euclidean distances between centroids.

    Args:
        centroids: Array of shape `(N, d)`.

    Returns:
        Array of shape `(N, N)` with `dist[i, j] = ||centroids[i] - centroids[j]||_2`.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    diff = centroids[:, None, :] - centroids[None, :, :]
    result: NDArray[np.float64] = np.sqrt(np.sum(diff * diff, axis=-1))
    return result


#: Default explained-variance-ratio threshold below which a principal axis is
#: dropped by `pca_align`. Argued (REALGF_PLAN.md S2, Finding B) from two
#: floors that sit ~18 orders of magnitude apart: float64 roundoff on
#: O(1)-O(10) coordinates gives a variance-ratio noise floor around 1e-30 to
#: 1e-32, while any genuine geometric feature (fault roughness, curvature) is
#: at least ~1e-16. `1e-12` sits comfortably above the noise floor and well
#: below the real-feature floor, biased conservative (dropping a genuine axis
#: silently corrupts the geometry; keeping a noise axis only costs a factor
#: of ~2 in tree cost).
DEFAULT_PCA_VAR_TOL = 1e-12

#: Explained-variance-ratio band in which whether an axis is "real" or
#: "noise" is not decidable from the data alone; `pca_align` warns loudly
#: (never decides silently) whenever any ratio falls in this half-open band.
_AMBIGUITY_BAND = (1e-12, 1e-6)


@dataclass(frozen=True)
class PCAAlignment:
    """Result of PCA-aligning a centroid cloud and dropping degenerate axes.

    `pca_align` is a pure utility: it never runs automatically inside
    `FaultMesh`/`build_tree`, so building a `FaultMesh` directly from raw
    centroids (the pre-Task-R.2 path) is completely unaffected by this
    module -- existing trees stay byte-identical. Callers opt in explicitly
    by calling `pca_align` and feeding `.centroids` to `FaultMesh`.

    Attributes:
        centroids: Centered, rotated centroids restricted to the retained
            principal axes, shape `(N, len(kept_axes))`.
        mean: Centroid-cloud mean subtracted before rotation, shape
            `(d_orig,)`.
        rotation: Rows are the principal axes (right singular vectors of the
            centered cloud, descending singular value), shape
            `(d_orig, d_orig)`. `centered @ rotation.T` gives the full
            (unreduced) rotated cloud; `.centroids` is that restricted to
            `kept_axes`.
        variance_ratio: Explained-variance ratio `sigma_i^2 / sum(sigma_j^2)`
            per principal axis, shape `(d_orig,)`, descending.
        kept_axes: Indices into `variance_ratio`/`rotation` of the retained
            axes, in descending-variance order. Never empty.
        dropped_axes: Indices of the dropped axes, in descending-variance
            order.
    """

    centroids: NDArray[np.float64]
    mean: NDArray[np.float64]
    rotation: NDArray[np.float64]
    variance_ratio: NDArray[np.float64]
    kept_axes: tuple[int, ...]
    dropped_axes: tuple[int, ...]


def pca_align(centroids: NDArray[np.float64], var_tol: float = DEFAULT_PCA_VAR_TOL) -> PCAAlignment:
    """Center a centroid cloud, rotate into principal axes, and drop axes
    whose explained-variance ratio falls below `var_tol` (Task R.2, resolving
    REALGF_PLAN.md Finding B).

    Centering and rotation alone are not enough and can actively hurt: on a
    perfectly straight line embedded in 2D (e.g. BP3), rotating without
    dropping manufactures a ~1e-13-ratio second axis out of pure float
    roundoff, which a subsequent `build_tree` then shatters on (measured:
    83 leaves -> 2075). Rotation is only safe paired with the drop, which is
    why this function always does both together.

    Three safeguards (REALGF_PLAN.md S2):
      1. Any ratio in the ambiguity band `_AMBIGUITY_BAND` (`[1e-12, 1e-6)`)
         triggers a `UserWarning` -- such a mesh is *nearly* degenerate and
         silently deciding either way risks a wrong answer.
      2. At least one axis is always kept, even if every ratio is below
         `var_tol` (e.g. a single point, or a cloud degenerate to numerical
         noise in every direction).
      3. The full `variance_ratio` and the `kept_axes`/`dropped_axes` split
         are always returned, never silently discarded.

    Args:
        centroids: Raw centroids, shape `(N, d)`, `N >= 1`.
        var_tol: Explained-variance-ratio threshold; axes with
            `sigma_i^2 / sum(sigma_j^2) < var_tol` are dropped (subject to
            safeguard 2). Defaults to `DEFAULT_PCA_VAR_TOL`.

    Returns:
        A `PCAAlignment` with the reduced, rotated centroids and the full
        report of what was kept/dropped and why.

    Raises:
        ValueError: If `centroids` is not `(N, d)` with `N >= 1`, or
            `var_tol < 0`.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[0] == 0:
        raise ValueError(f"centroids must have shape (N, d) with N >= 1, got {centroids.shape}")
    if var_tol < 0:
        raise ValueError(f"var_tol must be >= 0, got {var_tol}")

    mean = centroids.mean(axis=0)
    centered = centroids - mean

    # centered = U @ diag(s) @ vt; vt's rows are the principal axes, sorted by
    # descending singular value (numpy guarantees this ordering).
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s * s
    total = float(variance.sum())
    ratio = variance / total if total > 0 else np.zeros_like(variance)

    lo, hi = _AMBIGUITY_BAND
    ambiguous = (ratio >= lo) & (ratio < hi)
    if np.any(ambiguous):
        warnings.warn(
            f"pca_align: explained-variance ratio(s) {ratio[ambiguous].tolist()} fall in the "
            f"ambiguity band [{lo}, {hi}) -- whether these axes are genuine geometric "
            f"features or noise is not decidable from the data alone. Proceeding with "
            f"var_tol={var_tol}; inspect `variance_ratio` before trusting the result.",
            stacklevel=2,
        )

    keep_mask = ratio >= var_tol
    if not np.any(keep_mask):
        # Safeguard: never reduce below one dimension, even if every ratio is
        # below var_tol (e.g. a single distinct point among near-duplicates).
        keep_mask[int(np.argmax(ratio))] = True

    kept_axes = tuple(int(i) for i in np.flatnonzero(keep_mask))
    dropped_axes = tuple(int(i) for i in np.flatnonzero(~keep_mask))

    rotated = centered @ vt.T
    reduced = rotated[:, kept_axes]

    return PCAAlignment(
        centroids=reduced,
        mean=mean,
        rotation=vt,
        variance_ratio=ratio,
        kept_axes=kept_axes,
        dropped_axes=dropped_axes,
    )
