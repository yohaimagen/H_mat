"""Level-truncation operator `(A - A^{(l-1)})` for peeling (Task 5.1).

The non-uniform H1 peeling driver (Stage 5) processes tree levels
coarse-to-fine. By the time it reaches level `l`, it has already compressed
every admissible block `(alpha, beta)` at levels `2, ..., l-1` into a
low-rank factor triple `U_{alpha,beta} B_{alpha,beta} V_{alpha,beta}*`. Before
sampling level `l`, those already-captured contributions must be subtracted
off so the level-`l` samples see only the *remaining* operator
`A - A^{(l-1)}`, where

    A^{(l-1)} = sum_{l'=2}^{l-1} sum_{(alpha,beta) admissible at l'}
                    P_alpha U_{alpha,beta} B_{alpha,beta} V_{alpha,beta}*
                    P_beta*

with `P_alpha`/`P_beta` the (patch-major, dof-expanded) inclusion maps from
box `alpha`'s row-index set `I_alpha` / box `beta`'s col-index set `I_beta`
into the global row/col index ranges `{0, ..., n_rows-1}` /
`{0, ..., n_cols-1}`.

Factors representation
-----------------------
A single admissible block's stored factors are a `BlockFactor`: the row box
`alpha`, the col box `beta`, and the dense low-rank factors `u` (`U_{alpha,
beta}`, shape `(dof_row * |alpha|, k)`), `b` (`B_{alpha,beta}`, shape `(k,
k)`), `v` (`V_{alpha,beta}`, shape `(dof_col * |beta|, k)`), such that

    A(I_alpha, I_beta) ~= u @ b @ v.conj().T.

The full set of factors accumulated for levels `2, ..., l-1` is simply a flat
list of `BlockFactor`s (`Factors = list[BlockFactor]`) -- one entry per
admissible pair across all of those levels. Reconstruction
(`apply_truncated`/`apply_truncated_T`) does not need to know which level a
factor came from; it only needs each block's row/col index sets and its `U,
B, V` factors. This is the minimal contract that Tasks 5.2-5.6 must populate:
whatever per-level bookkeeping those tasks use internally, they hand
`apply_truncated`/`apply_truncated_T` (via `peeled_matvec`/`peeled_rmatvec`) a
flat `Factors` list covering exactly the admissible pairs of levels
`2, ..., l-1`.

Shapes
------
Per CLAUDE.md, the flattened operator `A` maps `R^{dof_col * N} ->
R^{dof_row * N}` (`2N x N` in 2D, `3N x 2N` in 3D); `A` and `A*` act on
different spaces:

- `apply_truncated(factors, omega)` approximates `A^{(l-1)} @ omega`: `omega`
  has shape `(n_cols,)` or `(n_cols, k)` (domain of `A`, indexed by
  `beta.col_indices`), the result has shape `(n_rows,)` or `(n_rows, k)`
  (range of `A`, indexed by `alpha.row_indices`).
- `apply_truncated_T(factors, psi)` approximates `(A^{(l-1)})* @ psi`: `psi`
  has shape `(n_rows,)` or `(n_rows, k)`, the result has shape `(n_cols,)` or
  `(n_cols, k)`.

`peeled_matvec`/`peeled_rmatvec` combine these with the black-box
`operator.matvec`/`operator.rmatvec` (Task 2.1) to give `(A - A^{(l-1)}) @
omega` and `(A - A^{(l-1)})* @ psi` respectively. "Level `l`" only matters
insofar as `factors` must contain exactly the blocks from levels
`2, ..., l-1`; for the coarsest level with admissible blocks (no earlier
levels to peel), `factors` is empty and both reduce to the raw
`matvec`/`rmatvec`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gfcompress.operators import MatVecOperator
from gfcompress.tree import TreeNode


@dataclass(frozen=True)
class BlockFactor:
    """Stored low-rank factors for one admissible block `(alpha, beta)`.

    Attributes:
        alpha: The row box. `alpha.row_indices` (length `dof_row * |alpha|`)
            indexes the global row space `{0, ..., n_rows - 1}`.
        beta: The col box. `beta.col_indices` (length `dof_col * |beta|`)
            indexes the global col space `{0, ..., n_cols - 1}`.
        u: Column-space basis `U_{alpha,beta}`, shape
            `(len(alpha.row_indices), k)`.
        b: Core matrix `B_{alpha,beta}`, shape `(k, k)`.
        v: Row-space basis `V_{alpha,beta}`, shape
            `(len(beta.col_indices), k)`.

    The block's approximation is `A(I_alpha, I_beta) ~= u @ b @ v.conj().T`,
    with `I_alpha = alpha.row_indices` and `I_beta = beta.col_indices`.
    """

    alpha: TreeNode
    beta: TreeNode
    u: NDArray[np.floating]
    b: NDArray[np.floating]
    v: NDArray[np.floating]


#: Flat collection of admissible-block factors, one entry per admissible pair
#: across all levels already peeled (i.e. levels `2, ..., l-1` for the
#: level-`l` truncation operator). Order is irrelevant; reconstruction sums
#: over all entries.
Factors = list[BlockFactor]


def apply_truncated(
    factors: Factors,
    omega: NDArray[np.floating],
    n_rows: int,
) -> NDArray[np.floating]:
    """Apply the truncated operator `A^{(l-1)} @ omega` from stored factors.

    Computes

        A^{(l-1)} @ omega = sum_{(alpha,beta) in factors}
            P_alpha @ (u_{alpha,beta} @ (b_{alpha,beta} @ (v_{alpha,beta}.conj().T
                @ omega[beta.col_indices, ...])))

    i.e. for each stored block, restrict `omega` to `beta.col_indices`, apply
    `u @ b @ v.conj().T`, and scatter-add the result into the rows
    `alpha.row_indices` of the output.

    Args:
        factors: Flat list of `BlockFactor`s for levels `2, ..., l-1`. May be
            empty (e.g. at the coarsest level), in which case the zero vector
            is returned.
        omega: Array of shape `(n_cols,)` or `(n_cols, k)`, in `A`'s domain
            (indexed by `beta.col_indices` for every stored block).
        n_rows: The output's leading dimension (`A`'s `n_rows`), needed to
            allocate the zero result when `factors` is empty or when no
            factor's `alpha.row_indices` cover the full range.

    Returns:
        Array of shape `(n_rows,)` or `(n_rows, k)` matching `omega`'s
        trailing shape, approximating `A^{(l-1)} @ omega`.
    """
    omega = np.asarray(omega)
    out_shape: tuple[int, ...] = (n_rows,) if omega.ndim == 1 else (n_rows, omega.shape[1])
    result = np.zeros(out_shape, dtype=omega.dtype)

    for factor in factors:
        omega_beta = omega[factor.beta.col_indices, ...]
        contribution = factor.u @ (factor.b @ (factor.v.conj().T @ omega_beta))
        result[factor.alpha.row_indices, ...] += contribution

    return result


def apply_truncated_T(
    factors: Factors,
    psi: NDArray[np.floating],
    n_cols: int,
) -> NDArray[np.floating]:
    """Apply the truncated adjoint operator `(A^{(l-1)})* @ psi` from stored
    factors.

    Computes

        (A^{(l-1)})* @ psi = sum_{(alpha,beta) in factors}
            P_beta @ (v_{alpha,beta} @ (b_{alpha,beta}.conj().T @ (u_{alpha,beta}.conj().T
                @ psi[alpha.row_indices, ...])))

    the conjugate-transpose counterpart of `apply_truncated`: for each stored
    block, restrict `psi` to `alpha.row_indices`, apply
    `v @ b.conj().T @ u.conj().T`, and scatter-add the result into the columns
    `beta.col_indices` of the output.

    Args:
        factors: Flat list of `BlockFactor`s for levels `2, ..., l-1`. May be
            empty, in which case the zero vector is returned.
        psi: Array of shape `(n_rows,)` or `(n_rows, k)`, in `A`'s range
            (indexed by `alpha.row_indices` for every stored block).
        n_cols: The output's leading dimension (`A`'s `n_cols`).

    Returns:
        Array of shape `(n_cols,)` or `(n_cols, k)` matching `psi`'s trailing
        shape, approximating `(A^{(l-1)})* @ psi`.
    """
    psi = np.asarray(psi)
    out_shape: tuple[int, ...] = (n_cols,) if psi.ndim == 1 else (n_cols, psi.shape[1])
    result = np.zeros(out_shape, dtype=psi.dtype)

    for factor in factors:
        psi_alpha = psi[factor.alpha.row_indices, ...]
        contribution = factor.v @ (factor.b.conj().T @ (factor.u.conj().T @ psi_alpha))
        result[factor.beta.col_indices, ...] += contribution

    return result


def peeled_matvec(
    operator: MatVecOperator,
    omega: NDArray[np.floating],
    factors: Factors,
) -> NDArray[np.floating]:
    """Compute `(A - A^{(l-1)}) @ omega = operator.matvec(omega) -
    apply_truncated(factors, omega)`.

    Args:
        operator: The black-box operator `A` (accessed only via `matvec`).
        omega: Array of shape `(n_cols,)` or `(n_cols, k)`.
        factors: Flat list of `BlockFactor`s for levels `2, ..., l-1`. If
            empty, this reduces exactly to `operator.matvec(omega)` (the
            coarsest-level case).

    Returns:
        Array of shape `(n_rows,)` or `(n_rows, k)` matching `omega`'s
        trailing shape.
    """
    n_rows = operator.shape[0]
    full: NDArray[np.floating] = operator.matvec(omega)
    if not factors:
        return full
    truncated = apply_truncated(factors, omega, n_rows)
    result: NDArray[np.floating] = full - truncated
    return result


def peeled_rmatvec(
    operator: MatVecOperator,
    psi: NDArray[np.floating],
    factors: Factors,
) -> NDArray[np.floating]:
    """Compute `(A - A^{(l-1)})* @ psi = operator.rmatvec(psi) -
    apply_truncated_T(factors, psi)`.

    Args:
        operator: The black-box operator `A` (accessed only via `rmatvec`).
        psi: Array of shape `(n_rows,)` or `(n_rows, k)`.
        factors: Flat list of `BlockFactor`s for levels `2, ..., l-1`. If
            empty, this reduces exactly to `operator.rmatvec(psi)` (the
            coarsest-level case).

    Returns:
        Array of shape `(n_cols,)` or `(n_cols, k)` matching `psi`'s trailing
        shape.
    """
    n_cols = operator.shape[1]
    full: NDArray[np.floating] = operator.rmatvec(psi)
    if not factors:
        return full
    truncated = apply_truncated_T(factors, psi, n_cols)
    result: NDArray[np.floating] = full - truncated
    return result


__all__ = [
    "BlockFactor",
    "Factors",
    "apply_truncated",
    "apply_truncated_T",
    "peeled_matvec",
    "peeled_rmatvec",
]
