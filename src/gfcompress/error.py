"""Relative-error utility via the power method (Task 2.3).

Per CLAUDE.md, the only way to compare two operators (e.g. an `HMatrix`
approximation `A_approx` against a reference `A`) is through their
`matvec`/`rmatvec` -- never by materializing a dense difference. This module
provides:

- `DifferenceOperator`: a `MatVecOperator` representing `A_approx - A`,
  computed lazily as `A_approx.matvec(x) - A.matvec(x)` (and similarly for
  `rmatvec`).
- `spectral_norm_power_method`: a power-method estimate of the spectral
  (operator 2-) norm `||A||_2` of a (possibly rectangular) `MatVecOperator`,
  using ~20 iterations of the symmetric power method on `A* A` (equivalently
  `A A*`), which only requires `matvec`/`rmatvec`.
- `relative_error`: the ratio `||A_approx - A||_2 / ||A||_2`, the standard
  measure of compression accuracy used throughout this package (e.g. Task
  6.1's `||A_H - A|| / ||A|| < tol`).

Shape conventions
------------------
`A` (and `A_approx`) map `R^{n_cols} -> R^{n_rows}`, with `n_rows != n_cols`
in general (`2N x N` in 2D, `3N x 2N` in 3D, per CLAUDE.md). The power method
below iterates on the square Gram operator `A* A` (shape `n_cols x n_cols`),
applying `matvec` then `rmatvec` each step; its leading eigenvalue is
`||A||_2^2`. `A` and `A_approx` must share the same `shape` for the
difference and ratio to be meaningful.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gfcompress.operators import MatVecOperator

DEFAULT_ITERS = 20


class DifferenceOperator(MatVecOperator):
    """`MatVecOperator` representing `A_approx - A`.

    Computed lazily: `matvec`/`rmatvec` apply both operators and subtract the
    results, never materializing a dense difference.

    Attributes:
        a_approx: The approximation operator `A_approx`.
        a: The reference operator `A`.
    """

    def __init__(self, a_approx: MatVecOperator, a: MatVecOperator) -> None:
        """Wrap `A_approx - A` as a `MatVecOperator`.

        Args:
            a_approx: The approximation operator `A_approx`.
            a: The reference operator `A`.

        Raises:
            ValueError: If `a_approx.shape != a.shape`.
        """
        if a_approx.shape != a.shape:
            raise ValueError(
                f"a_approx.shape {a_approx.shape} != a.shape {a.shape}; "
                "the difference A_approx - A requires matching shapes"
            )
        self.a_approx = a_approx
        self.a = a

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `(A_approx - A) @ omega`. See `MatVecOperator.matvec`."""
        result: NDArray[np.floating] = self.a_approx.matvec(omega) - self.a.matvec(omega)
        return result

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `(A_approx - A)* @ psi`. See `MatVecOperator.rmatvec`."""
        result: NDArray[np.floating] = self.a_approx.rmatvec(psi) - self.a.rmatvec(psi)
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """`(n_rows, n_cols)`, shared by `a_approx` and `a`."""
        return self.a.shape


def spectral_norm_power_method(
    op: MatVecOperator,
    n_iters: int = DEFAULT_ITERS,
    seed: int | None = None,
) -> float:
    """Estimate `||op||_2` (the spectral/operator 2-norm) via the power method.

    Runs the symmetric power method on the Gram operator `op* op` (shape
    `n_cols x n_cols`), applying `op.matvec` then `op.rmatvec` each
    iteration -- touching `op` only through these two methods, per CLAUDE.md.
    The square root of the resulting Rayleigh quotient `x* (A* A) x / (x* x)`
    converges to `||op||_2 = sigma_max(op)`.

    Args:
        op: The operator whose spectral norm is estimated. May be
            rectangular (`n_rows != n_cols`).
        n_iters: Number of power-method iterations. Defaults to `20`.
        seed: Optional seed for the random initial vector (`numpy.random`,
            reproducible).

    Returns:
        An estimate of `||op||_2 >= 0`. Returns `0.0` if `op`'s domain has
        zero dimension or the initial vector converges to the zero vector
        (i.e. `op` is the zero operator).

    Raises:
        ValueError: If `n_iters < 1`.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")

    n_cols = op.shape[1]
    rng = np.random.default_rng(seed)
    x: NDArray[np.float64] = rng.standard_normal(n_cols)
    x_norm = np.linalg.norm(x)
    if x_norm == 0.0:
        return 0.0
    x = x / x_norm

    eigenvalue = 0.0
    for _ in range(n_iters):
        y = op.matvec(x)
        gram_x = op.rmatvec(y)
        gram_x_norm = np.linalg.norm(gram_x)
        if gram_x_norm == 0.0:
            return 0.0
        eigenvalue = float(np.vdot(x, gram_x).real)
        x = gram_x / gram_x_norm

    norm: float = float(np.sqrt(max(eigenvalue, 0.0)))
    return norm


def relative_error(
    a_approx: MatVecOperator,
    a: MatVecOperator,
    n_iters: int = DEFAULT_ITERS,
    seed: int | None = None,
) -> float:
    """Estimate `||A_approx - A||_2 / ||A||_2` via the power method.

    Both the numerator and the denominator are estimated with
    `spectral_norm_power_method`, touching `a_approx` and `a` only through
    their `matvec`/`rmatvec`.

    Args:
        a_approx: The approximation operator `A_approx`.
        a: The reference operator `A`. Must satisfy `a_approx.shape ==
            a.shape`.
        n_iters: Number of power-method iterations for each spectral-norm
            estimate. Defaults to `20`.
        seed: Optional seed for the random initial vectors (`numpy.random`,
            reproducible). The same seed is used for both norm estimates.

    Returns:
        The relative error `||A_approx - A||_2 / ||A||_2`. Returns `0.0` if
        both `A_approx` and `A` are (numerically) the zero operator. Returns
        `inf` if `A` is the zero operator but `A_approx` is not.

    Raises:
        ValueError: If `a_approx.shape != a.shape`.
    """
    diff = DifferenceOperator(a_approx, a)
    diff_norm = spectral_norm_power_method(diff, n_iters=n_iters, seed=seed)
    a_norm = spectral_norm_power_method(a, n_iters=n_iters, seed=seed)

    if a_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")

    return diff_norm / a_norm
