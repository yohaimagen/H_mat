"""Black-box matvec operator interface (Task 2.1).

Per CLAUDE.md, the matrix `A` is accessed *exclusively* through
`matvec(Omega) = A @ Omega` and `rmatvec(Psi) = A.conj().T @ Psi`. The
compressor never assembles a dense `A`; the only dense arrays in this
package belong to test doubles (`DenseOperator` here, and `MockGF`'s
near-field leaf blocks in Task 2.2).

`A` is in general **rectangular**: it maps `R^{dof_col * N} -> R^{dof_row *
N}`, i.e. shape `(dof_row * N, dof_col * N)` -- `2N x N` in 2D (`dof_row = 2`,
`dof_col = 1`) and `3N x 2N` in 3D (`dof_row = 3`, `dof_col = 2`). `A` and
`A*` therefore act on different spaces:

- `matvec`: `R^{n_cols} -> R^{n_rows}` (or `C^{n_cols} -> C^{n_rows}`).
- `rmatvec`: `R^{n_rows} -> R^{n_cols}` (or `C^{n_rows} -> C^{n_cols}`),
  applying the conjugate transpose `A*`.

Both methods accept either a single vector, shape `(n,)`, or a "thick"
matrix of stacked column vectors, shape `(n, k)`, returning shape `(m,)` or
`(m, k)` respectively -- this is what the randomized-SVD primitives in
Stage 3 need (`Omega`/`Psi` are `n x (k+p)` Gaussian sketch matrices).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class MatVecOperator(ABC):
    """Abstract black-box linear operator, accessed only via matvecs.

    A `MatVecOperator` represents a (possibly rectangular) linear map `A` of
    shape `(n_rows, n_cols)`, i.e. `A: R^{n_cols} -> R^{n_rows}`. Concrete
    subclasses (e.g. `DenseOperator`, and `MockGF` in Task 2.2) must implement
    `matvec`, `rmatvec`, and `shape`; nothing else may assume access to a
    dense representation of `A`.

    Contract:
        - `matvec(Omega)` returns `A @ Omega`. `Omega` has shape `(n_cols,)`
          or `(n_cols, k)`; the result has shape `(n_rows,)` or
          `(n_rows, k)` respectively.
        - `rmatvec(Psi)` returns `A* @ Psi` (conjugate transpose). `Psi` has
          shape `(n_rows,)` or `(n_rows, k)`; the result has shape
          `(n_cols,)` or `(n_cols, k)` respectively.
        - `shape` is the tuple `(n_rows, n_cols)`. For the flattened
          elastostatic operator this is `(dof_row * N, dof_col * N)`, i.e.
          `2N x N` in 2D and `3N x 2N` in 3D -- in general `n_rows != n_cols`,
          so `matvec` and `rmatvec` act on different spaces and must not be
          assumed interchangeable.
    """

    @abstractmethod
    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply `A` to `omega`: return `A @ omega`.

        Args:
            omega: Array of shape `(n_cols,)` or `(n_cols, k)`.

        Returns:
            `A @ omega`, of shape `(n_rows,)` or `(n_rows, k)` matching the
            input's trailing shape.
        """
        raise NotImplementedError

    @abstractmethod
    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply `A*` (conjugate transpose) to `psi`: return `A* @ psi`.

        Args:
            psi: Array of shape `(n_rows,)` or `(n_rows, k)`.

        Returns:
            `A* @ psi`, of shape `(n_cols,)` or `(n_cols, k)` matching the
            input's trailing shape.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """The operator's shape `(n_rows, n_cols)`.

        For the flattened elastostatic operator, `n_rows = dof_row * N` and
        `n_cols = dof_col * N` (`2N x N` in 2D, `3N x 2N` in 3D); in general
        `n_rows != n_cols`.
        """
        raise NotImplementedError


class DenseOperator(MatVecOperator):
    """Concrete `MatVecOperator` backed by an explicit dense array.

    Trivial test double: wraps a dense `(n_rows, n_cols)` array `A` and
    implements `matvec`/`rmatvec` via direct matrix multiplication. Used to
    validate the `MatVecOperator` contract and as ground truth in tests for
    operators that are otherwise accessed only through matvecs.

    Attributes:
        A: The wrapped dense array, shape `(n_rows, n_cols)`.
    """

    def __init__(self, a: NDArray[np.floating]) -> None:
        """Wrap a dense array as a `MatVecOperator`.

        Args:
            a: Dense array of shape `(n_rows, n_cols)`.

        Raises:
            ValueError: If `a` is not 2-dimensional.
        """
        if a.ndim != 2:
            raise ValueError(f"a must be 2-dimensional, got shape {a.shape}")
        self.A: NDArray[np.floating] = a

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `A @ omega`. See `MatVecOperator.matvec`."""
        return self.A @ omega

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return `A.conj().T @ psi`. See `MatVecOperator.rmatvec`."""
        a_adj: NDArray[np.floating] = self.A.conj().T
        return a_adj @ psi

    @property
    def shape(self) -> tuple[int, int]:
        """`(n_rows, n_cols)` of the wrapped dense array."""
        return (self.A.shape[0], self.A.shape[1])
