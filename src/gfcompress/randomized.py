"""Randomized low-rank primitives: Gaussian sampling and orthonormalization
(Task 3.1).

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
