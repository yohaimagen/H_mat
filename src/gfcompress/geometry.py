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
            dimension `d`, `in {2, 3}`). Pass `0` (the default) to infer it
            from `tree_dim`, preserving pre-R.2 behavior when the tree is
            built at full ambient dimension. Always a real `int` (`2` or
            `3`) after construction -- never `0` -- so downstream arithmetic
            on `mesh.dof_row` needs no `None`/`0` check.
        tree_dim: Dimensionality of the cluster tree (`= centroids.shape[1]`),
            `in {1, 2, 3}`, `<= dof_row`.
        dof_col: Column degrees of freedom per patch, equal to `dof_row - 1`.
    """

    centroids: NDArray[np.float64]
    L: NDArray[np.float64]
    dof_row: int = 0
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
        if not np.isfinite(centroids).all() or not np.isfinite(L).all():
            raise ValueError("centroids and L must be finite")

        tree_dim = centroids.shape[1]
        dof_row = tree_dim if self.dof_row == 0 else self.dof_row
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


#: Numerical rank criterion used by `pca_align`: an axis is roundoff-degenerate
#: when ``sigma_i / sigma_0 < eps * max(N, d)``.  This is NumPy's standard SVD
#: matrix-rank scale.  In variance-ratio terms the cutoff is data dependent:
#: ``(eps * max(N, d))**2 * sigma_0**2 / sum(sigma**2)``; it is returned as
#: `PCAAlignment.variance_threshold`.  This keeps genuine 1e-16-scale variance
#: rather than treating it as noise via the former arbitrary 1e-12 cutoff.
DEFAULT_PCA_VAR_TOL: float | None = None
_CUTOFF_UNCERTAINTY_FACTOR = 10.0


@dataclass(frozen=True)
class PCAAlignment:
    """Result of PCA-aligning a centroid cloud and dropping degenerate axes.

    `pca_align` is a pure utility: it never runs automatically inside
    `FaultMesh`/`build_tree`, so building a `FaultMesh` directly from raw
    centroids (the pre-Task-R.2 path) is completely unaffected by this
    module -- existing trees stay byte-identical. Callers opt in explicitly
    by calling `pca_align` and feeding `.centroids` to `FaultMesh`.

    Attributes:
        original_centroids: Original, unmodified input coordinates.
        centroids: Centered, rotated centroids restricted to the retained
            principal axes, shape `(N, len(kept_axes))`.
        mean: Centroid-cloud mean subtracted before rotation, shape
            `(d_orig,)`.
        rotation: Rows are the principal axes (right singular vectors of the
            centered cloud, descending singular value), shape
            `(d_r, d_orig)` with `d_r = min(N, d_orig)` (`np.linalg.svd(...,
            full_matrices=False)`). `centered @ rotation.T` gives the full
            (unreduced) rotated cloud; `.centroids` is that restricted to
            `kept_axes`. `d_r == d_orig` for every mesh this codebase
            actually builds (patches always vastly outnumber `d_orig <= 3`);
            the `N < d_orig` case (needs `N <= 3` to trigger) is a
            pathological edge this function does not specially handle: `d_r`
            axes are ranked/reported exactly as usual, but the
            `d_orig - d_r` axes the cloud is too small to even define are
            absent from `variance_ratio`/`kept_axes`/`dropped_axes`
            altogether, not counted as "dropped".
        variance_ratio: Explained-variance ratio `sigma_i^2 / sum(sigma_j^2)`
            per principal axis, shape `(d_r,)` (see `rotation`), descending.
        kept_axes: Indices into `variance_ratio`/`rotation` of the retained
            axes, in descending-variance order. Never empty.
        dropped_axes: Indices of the dropped axes, in descending-variance
            order.
        singular_values: Singular values of the centered cloud.
        singular_value_threshold: Roundoff singular-value cutoff used for the
            automatic decision.
        variance_threshold: Equivalent explained-variance-ratio cutoff.
        automatic_kept_axes: Axes retained by the automatic rank criterion,
            before an optional `tree_dim` override.
        absolute_projection_residual: Frobenius norm discarded by projection.
        relative_projection_residual: That residual divided by the centered
            cloud's Frobenius norm (zero for a zero-spread cloud).
    """

    original_centroids: NDArray[np.float64]
    centroids: NDArray[np.float64]
    mean: NDArray[np.float64]
    rotation: NDArray[np.float64]
    variance_ratio: NDArray[np.float64]
    kept_axes: tuple[int, ...]
    dropped_axes: tuple[int, ...]
    singular_values: NDArray[np.float64]
    singular_value_threshold: float
    variance_threshold: float
    automatic_kept_axes: tuple[int, ...]
    absolute_projection_residual: float
    relative_projection_residual: float


def pca_align(
    centroids: NDArray[np.float64],
    var_tol: float | None = DEFAULT_PCA_VAR_TOL,
    *,
    tree_dim: int | None = None,
) -> PCAAlignment:
    """Center a centroid cloud, rotate into principal axes, and drop axes
    whose singular value is numerically zero (Task C.2).

    Centering and rotation alone are not enough and can actively hurt: on a
    perfectly straight line embedded in 2D (e.g. BP3), rotating without
    dropping manufactures a ~1e-13-ratio second axis out of pure float
    roundoff, which a subsequent `build_tree` then shatters on (measured:
    83 leaves -> 2075). Rotation is only safe paired with the drop, which is
    why this function always does both together.

    A tenfold band around the cutoff warns for axes on either side of the
    decision.  At least one axis is always kept.  `tree_dim` is an explicit
    override for deliberate model reduction; its residual diagnostics make
    that choice visible rather than silently treating it as numerical rank.

    Args:
        centroids: Raw centroids, shape `(N, d)`, `N >= 1`.
        var_tol: Optional explained-variance-ratio override. `None` (the
            default) uses the documented roundoff singular-value criterion.
        tree_dim: Explicit number of leading principal axes to retain.

    Returns:
        A `PCAAlignment` with the reduced, rotated centroids and the full
        report of what was kept/dropped and why.

    Raises:
        ValueError: If inputs are malformed, nonfinite, or tolerances are
            invalid.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[0] == 0 or centroids.shape[1] == 0:
        raise ValueError(f"centroids must have shape (N, d) with N >= 1, got {centroids.shape}")
    if not np.isfinite(centroids).all():
        raise ValueError("centroids must be finite")
    if var_tol is not None and (not np.isfinite(var_tol) or not 0.0 <= var_tol <= 1.0):
        raise ValueError(f"var_tol must be finite and in [0, 1], got {var_tol}")

    mean = centroids.mean(axis=0)
    centered = centroids - mean

    # centered = U @ diag(s) @ vt; vt's rows are the principal axes, sorted by
    # descending singular value (numpy guarantees this ordering).
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s * s
    total = float(variance.sum())
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    sigma0 = float(s[0]) if len(s) else 0.0
    singular_threshold = np.finfo(np.float64).eps * max(centered.shape) * sigma0
    variance_threshold = singular_threshold**2 / total if total else 0.0
    threshold = float(var_tol) if var_tol is not None else variance_threshold
    automatic_mask = ratio >= threshold if total else np.zeros_like(ratio, dtype=bool)
    if not np.any(automatic_mask):
        automatic_mask[0] = True
    automatic_kept_axes = tuple(int(i) for i in np.flatnonzero(automatic_mask))

    uncertainty = (ratio >= threshold / _CUTOFF_UNCERTAINTY_FACTOR) & (
        ratio <= threshold * _CUTOFF_UNCERTAINTY_FACTOR
    )
    if np.any(uncertainty) and threshold > 0:
        warnings.warn(
            "pca_align: explained-variance ratio(s) "
            f"{ratio[uncertainty].tolist()} are within a factor of "
            f"{_CUTOFF_UNCERTAINTY_FACTOR:g} of cutoff {threshold:.3e}; inspect "
            "both retained and dropped axes before relying on automatic reduction.",
            stacklevel=2,
        )

    if tree_dim is None:
        keep_mask = automatic_mask
    else:
        if not isinstance(tree_dim, int) or not 1 <= tree_dim <= len(s):
            raise ValueError(f"tree_dim must be an integer in [1, {len(s)}], got {tree_dim}")
        keep_mask = np.arange(len(s)) < tree_dim

    kept_axes = tuple(int(i) for i in np.flatnonzero(keep_mask))
    dropped_axes = tuple(int(i) for i in np.flatnonzero(~keep_mask))

    rotated = centered @ vt.T
    reduced = rotated[:, kept_axes]
    projected = reduced @ vt[np.array(kept_axes)].reshape(len(kept_axes), -1)
    absolute_residual = float(np.linalg.norm(centered - projected))
    relative_residual = absolute_residual / np.sqrt(total) if total else 0.0

    return PCAAlignment(
        original_centroids=centroids.copy(),
        centroids=reduced,
        mean=mean,
        rotation=vt,
        variance_ratio=ratio,
        kept_axes=kept_axes,
        dropped_axes=dropped_axes,
        singular_values=s,
        singular_value_threshold=singular_threshold,
        variance_threshold=variance_threshold,
        automatic_kept_axes=automatic_kept_axes,
        absolute_projection_residual=absolute_residual,
        relative_projection_residual=relative_residual,
    )
