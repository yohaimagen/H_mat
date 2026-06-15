"""Randomized low-rank primitives: Gaussian sampling, orthonormalization,
two-sample compression, and the core-matrix solve (Tasks 3.1-3.3).

These are the basic building blocks for the randomized SVD machinery used
throughout Stage 3 and beyond (Algorithm 2.1 two-sample compression, the core
matrix solve, etc.). Per CLAUDE.md, the matrix `A` itself is never assembled
here -- this module only provides:

- `gaussian`: draw an `n x (k + p)` standard-normal sketch matrix `Omega`
  (seedable, via `numpy.random.default_rng`), used as `Omega` in
  `matvec(Omega) = A @ Omega`.
- `orth`: a thin QR wrapper around `scipy.linalg.qr` returning an
  orthonormal basis `Q` for `range(Y)` (economy/"thin" mode, `Q` has shape
  `(n, min(n, m))`).
- A rank-`k`-truncated variant of `orth` that returns only the first `k`
  columns of `Q` -- an orthonormal basis for an approximate rank-`k` subspace
  of `range(Y)`.
- `two_sample_compress` (Algorithm 2.1): given a column sample `Y` (rows of
  `A @ Omega` restricted to a block's row indices) and a row sample `Z` (rows
  of `A* @ Psi` restricted to the block's column indices), returns `(U, V) =
  (qr(Y, k), qr(Z, k))` -- orthonormal bases for the block's column and row
  spaces.
- `core_matrix_solve` (Eq. 4.3): given the bases `U`, `V` from
  `two_sample_compress`, the column sample restricted to block-`alpha` rows
  `Y(I_alpha, :)`, and the Gaussian sketches `G_alpha`, `G_beta` used to
  generate the row/column samples, returns the `k x k` core matrix `B` such
  that `A_{alpha,beta} ~= U @ B @ V*`. `B` is formed entirely from these
  sample/basis quantities via two least-squares (pseudoinverse) solves --
  `A_{alpha,beta}` is never assembled.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray


def gaussian(n: int, k: int, p: int = 0, seed: int | None = None) -> NDArray[np.float64]:
    """Draw an `n x (k + p)` standard-normal Gaussian sketch matrix.

    Args:
        n: Number of rows.
        k: Target rank.
        p: Oversampling parameter (extra columns beyond `k`). Defaults to
            `0`.
        seed: Optional seed for `numpy.random.default_rng`, for
            reproducibility.

    Returns:
        An `(n, k + p)` array of i.i.d. standard-normal entries.

    Raises:
        ValueError: If `n < 0`, `k < 0`, or `p < 0`.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if p < 0:
        raise ValueError(f"p must be >= 0, got {p}")

    rng = np.random.default_rng(seed)
    result: NDArray[np.float64] = rng.standard_normal((n, k + p))
    return result


def orth(y: NDArray[np.floating], k: int | None = None) -> NDArray[np.float64]:
    """Return an orthonormal basis for (an approximation of) `range(Y)`.

    Thin wrapper around `scipy.linalg.qr` in economy ("thin") mode: for `Y`
    of shape `(n, m)`, `Q` has shape `(n, min(n, m))` and satisfies `Q.T @ Q =
    I`.

    Args:
        y: The matrix whose column space is orthonormalized, shape `(n, m)`.
        k: If given, truncate to the first `k` columns of `Q` -- an
            orthonormal basis for an approximate rank-`k` subspace of
            `range(Y)`. Must satisfy `0 <= k <= min(n, m)`.

    Returns:
        `Q`, shape `(n, min(n, m))` if `k` is `None`, otherwise `(n, k)`.
        Columns are orthonormal: `Q.T @ Q = I`.

    Raises:
        ValueError: If `k` is given and `k < 0` or `k > min(Y.shape)`.
    """
    y = np.asarray(y, dtype=np.float64)
    q, _ = scipy.linalg.qr(y, mode="economic")

    if k is None:
        result: NDArray[np.float64] = q
        return result

    max_rank = min(y.shape)
    if k < 0 or k > max_rank:
        raise ValueError(f"k must be in [0, {max_rank}], got {k}")

    truncated: NDArray[np.float64] = q[:, :k]
    return truncated


def two_sample_compress(
    y: NDArray[np.floating], z: NDArray[np.floating], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Algorithm 2.1 two-sample compression of a single block `A_{alpha,beta}`.

    Given a *column sample* `Y` and a *row sample* `Z` of one block, return
    orthonormal bases `U = qr(Y, k)` and `V = qr(Z, k)` for the block's
    (approximate rank-`k`) column and row spaces, respectively. Reuses
    `orth(., k)` (Task 3.1) -- no QR is reimplemented here.

    Per CLAUDE.md's shape conventions, `A_{alpha,beta}` maps `R^{dof_col *
    |beta|} -> R^{dof_row * |alpha|}`, so:

    - `Y` is the row-restriction `(A @ Omega)(I_alpha, :)`, shape
      `(dof_row * |alpha|, k + p)` -- samples the block's column space (the
      space spanned by its columns, i.e. `range(A_{alpha,beta})`).
    - `Z` is the column-restriction `(A* @ Psi)(I_beta, :)`, shape
      `(dof_col * |beta|, k + p)` -- samples the block's row space (`range(
      A_{alpha,beta}*)`).

    `Y` and `Z` in general have *different* numbers of rows (`Y` has
    `dof_row * |alpha|` rows, `Z` has `dof_col * |beta|` rows); they are not
    interchangeable.

    Args:
        y: Column sample of the block, shape `(dof_row * |alpha|, k + p)`.
        z: Row sample of the block, shape `(dof_col * |beta|, k + p)`.
        k: Target rank. Must satisfy `0 <= k <= min(y.shape)` and
            `0 <= k <= min(z.shape)`.

    Returns:
        A tuple `(U, V)`:
            - `U`, shape `(dof_row * |alpha|, k)`, orthonormal basis for the
              block's (approximate) column space.
            - `V`, shape `(dof_col * |beta|, k)`, orthonormal basis for the
              block's (approximate) row space.

    Raises:
        ValueError: If `k` is out of range for `y` or `z` (see `orth`).
    """
    u = orth(y, k)
    v = orth(z, k)
    return u, v


def core_matrix_solve(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    y_alpha: NDArray[np.floating],
    g_alpha: NDArray[np.floating],
    g_beta: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Core-matrix solve for a single admissible block (Eq. 4.3).

    Computes the `k x k` core matrix `B` of the rank-`k` factorization
    `A_{alpha,beta} ~= U @ B @ V*`, via

        B = (G_alpha* U)^+ (G_alpha* Y(I_alpha, :)) (V* G_beta)^+

    where `^+` denotes the Moore-Penrose pseudoinverse, implemented as
    least-squares solves (`scipy.linalg.lstsq`) rather than by inverting or
    assembling `A_{alpha,beta}`. Everything on the right-hand side is a
    sample or basis quantity:

    - `Y(I_alpha, :) = A_{alpha,beta} @ G_beta` is the column sample of the
      block (rows of the global column sample `Y = A @ Omega` restricted to
      box `alpha`'s row indices `I_alpha`).
    - `G_alpha` is the Gaussian sketch (restricted to `I_alpha`) used to
      generate the block's row sample `Z(I_beta, :) = A_{alpha,beta}* @
      G_alpha`, from which `V` was computed.
    - `G_beta` is the Gaussian sketch (restricted to `I_beta`) used to
      generate `Y(I_alpha, :)` above, from which `U` was computed.

    Per CLAUDE.md's shape conventions, `A_{alpha,beta}` maps
    `R^{dof_col * |beta|} -> R^{dof_row * |alpha|}`, so `U` and `V` have
    different numbers of rows, and `G_alpha`/`G_beta` correspondingly have
    different numbers of rows (`U`/`G_alpha` index over box `alpha`'s rows,
    `V`/`G_beta` over box `beta`'s columns).

    Args:
        u: Column-space basis from `two_sample_compress`, shape
            `(dof_row * |alpha|, k)`, orthonormal columns.
        v: Row-space basis from `two_sample_compress`, shape
            `(dof_col * |beta|, k)`, orthonormal columns.
        y_alpha: Column sample restricted to box `alpha`'s rows,
            `Y(I_alpha, :) = A_{alpha,beta} @ G_beta`, shape
            `(dof_row * |alpha|, k + p)`.
        g_alpha: Gaussian sketch used for the row sample, restricted to box
            `alpha`'s rows, shape `(dof_row * |alpha|, k + p)`. Same number
            of rows as `u` and `y_alpha`.
        g_beta: Gaussian sketch used for the column sample, restricted to
            box `beta`'s columns, shape `(dof_col * |beta|, k + p)`. Same
            number of rows as `v`.

    Returns:
        `B`, shape `(k, k)`, the core matrix such that
        `U @ B @ V.conj().T` approximates `A_{alpha,beta}`.

    Raises:
        ValueError: If the shapes of `u`, `v`, `y_alpha`, `g_alpha`, and
            `g_beta` are inconsistent (row counts of `u`/`y_alpha`/`g_alpha`
            must agree, as must those of `v`/`g_beta`, and `u`/`v` must have
            the same number of columns `k`).
    """
    u = np.asarray(u)
    v = np.asarray(v)
    y_alpha = np.asarray(y_alpha)
    g_alpha = np.asarray(g_alpha)
    g_beta = np.asarray(g_beta)

    if u.shape[1] != v.shape[1]:
        raise ValueError(
            "u and v must have the same number of columns k, got "
            f"u.shape={u.shape}, v.shape={v.shape}"
        )
    if not (u.shape[0] == y_alpha.shape[0] == g_alpha.shape[0]):
        raise ValueError(
            "u, y_alpha, and g_alpha must have the same number of rows "
            f"(dof_row * |alpha|), got u.shape={u.shape}, "
            f"y_alpha.shape={y_alpha.shape}, g_alpha.shape={g_alpha.shape}"
        )
    if v.shape[0] != g_beta.shape[0]:
        raise ValueError(
            "v and g_beta must have the same number of rows (dof_col * "
            f"|beta|), got v.shape={v.shape}, g_beta.shape={g_beta.shape}"
        )

    # left = (G_alpha* U)^+ (G_alpha* Y(I_alpha, :)), shape (k, k+p).
    lhs = g_alpha.conj().T @ u  # (k+p, k)
    rhs = g_alpha.conj().T @ y_alpha  # (k+p, k+p)
    left, *_ = scipy.linalg.lstsq(lhs, rhs)  # (k, k+p)

    # right = (V* G_beta)^+, shape (k+p, k).
    vg = v.conj().T @ g_beta  # (k, k+p)
    right = np.linalg.pinv(vg)  # (k+p, k)

    b: NDArray[np.float64] = left @ right
    return b
