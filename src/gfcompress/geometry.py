"""Mesh and patch data structures for the dual-index cluster tree.

A `FaultMesh` holds the `N` patch centroids that drive the geometric cluster
tree shared by the row and column index sets. Per CLAUDE.md, the flattened
operator `A` maps `R^{dof_col * N} -> R^{dof_row * N}` with patch-major
flattening: each patch contributes `dof_row` consecutive scalar row indices
and `dof_col` consecutive scalar column indices, block-interleaved across
patches.
"""

from __future__ import annotations

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
        centroids: Patch centroids, shape `(N, d)` with `d in {2, 3}`.
        L: Characteristic length per patch, shape `(N,)`.
        d: Spatial dimension (2 or 3).
        dof_row: Row degrees of freedom per patch, equal to `d`.
        dof_col: Column degrees of freedom per patch, equal to `d - 1`.
    """

    centroids: NDArray[np.float64]
    L: NDArray[np.float64]
    d: int = field(init=False)
    dof_row: int = field(init=False)
    dof_col: int = field(init=False)

    def __post_init__(self) -> None:
        centroids = np.asarray(self.centroids, dtype=np.float64)
        L = np.asarray(self.L, dtype=np.float64)

        if centroids.ndim != 2 or centroids.shape[1] not in (2, 3):
            raise ValueError(f"centroids must have shape (N, 2) or (N, 3), got {centroids.shape}")
        if L.ndim != 1 or L.shape[0] != centroids.shape[0]:
            raise ValueError(
                f"L must have shape (N,) matching centroids, got {L.shape} "
                f"vs centroids {centroids.shape}"
            )

        self.centroids = centroids
        self.L = L
        d = centroids.shape[1]
        self.d = d
        self.dof_row = d
        self.dof_col = d - 1

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
