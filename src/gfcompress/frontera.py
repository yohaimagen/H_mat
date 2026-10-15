"""Bounded-memory reader for Tandem's native-dense PETSc checkpoints.

The Frontera checkpoint differs from :mod:`gfcompress.realgf`: its ``nz``
header field is ``-1`` and the big-endian float64 values begin immediately
after the 24-byte Tandem/PETSc header.  Products stream contiguous row chunks
and convert only the current chunk to native endian; the dense matrix is never
memory-mapped or materialized in RAM.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from gfcompress.geometry import FaultMesh, PCAAlignment, pca_align
from gfcompress.operators import MatVecOperator

PETSC_MAT_CLASSID = 1211216
NATIVE_DENSE_FORMAT = -1
HEADER_BYTES = 24


@dataclass(frozen=True)
class NativeDenseHeader:
    """Validated Tandem/PETSc native-dense header."""

    comm_size: int
    assembled_columns: int
    classid: int
    rows: int
    cols: int
    format_code: int
    values_offset: int = HEADER_BYTES

    @property
    def value_bytes(self) -> int:
        """Bytes occupied by the dense float64 values."""
        return self.rows * self.cols * 8

    @property
    def predicted_size(self) -> int:
        """Exact file size implied by this header."""
        return self.values_offset + self.value_bytes


def read_native_dense_header(path: str | Path) -> NativeDenseHeader:
    """Read and validate a complete native-dense Tandem checkpoint."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(HEADER_BYTES)
    if len(raw) != HEADER_BYTES:
        raise ValueError(f"{path}: file is too short for the 24-byte header")
    comm_size, assembled, classid, rows, cols, format_code = struct.unpack(">6i", raw)
    if classid != PETSC_MAT_CLASSID:
        raise ValueError(f"{path}: classid {classid} != {PETSC_MAT_CLASSID}")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{path}: invalid matrix shape ({rows}, {cols})")
    if format_code != NATIVE_DENSE_FORMAT:
        raise ValueError(
            f"{path}: format code {format_code} is not native dense ({NATIVE_DENSE_FORMAT})"
        )
    if assembled != cols:
        raise ValueError(f"{path}: checkpoint is incomplete ({assembled}/{cols} columns)")
    header = NativeDenseHeader(comm_size, assembled, classid, rows, cols, format_code)
    actual_size = os.path.getsize(path)
    if actual_size != header.predicted_size:
        raise ValueError(f"{path}: file size {actual_size} != predicted {header.predicted_size}")
    return header


def read_native_coordinates(
    path: str | Path, n_patches: int, *, dimension: int = 3
) -> NDArray[np.float64]:
    """Read raw big-endian patch-major coordinates as native float64."""
    path = Path(path)
    expected_size = n_patches * dimension * 8
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise ValueError(f"{path}: file size {actual_size} != expected {expected_size}")
    coords = np.fromfile(path, dtype=">f8").astype(np.float64).reshape(n_patches, dimension)
    if not np.isfinite(coords).all():
        raise ValueError(f"{path}: coordinates must be finite")
    return coords


def _read_values(
    stream: BinaryIO, header: NativeDenseHeader, start: int, stop: int
) -> NDArray[np.float64]:
    """Read one contiguous row interval and convert just that interval."""
    count = (stop - start) * header.cols
    stream.seek(header.values_offset + start * header.cols * 8)
    raw = np.fromfile(stream, dtype=">f8", count=count)
    if raw.size != count:
        raise OSError(f"short matrix read at rows [{start}, {stop})")
    return raw.astype(np.float64, copy=False).reshape(stop - start, header.cols)


class NativeDenseGF(MatVecOperator):
    """Raw-file-order 3D operator backed by a streamed checkpoint.

    Products apply the stored matrix exactly and make no scalar permutation.
    ``mesh`` is a conditional patch-major interpretation of the coordinates;
    the Frontera preflight marks that physical mapping unresolved, so it must
    not be used for production compression until writer metadata establishes
    the row and column layouts.
    """

    def __init__(
        self,
        mat_path: str | Path,
        coordinate_path: str | Path,
        *,
        dimension: int = 3,
        basis_functions_per_element: int = 10,
        chunk_rows: int = 32,
    ) -> None:
        if dimension != 3:
            raise ValueError("the Frontera native-dense reader currently supports 3D only")
        if basis_functions_per_element < 1:
            raise ValueError("basis_functions_per_element must be positive")
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")

        self.mat_path = Path(mat_path)
        self.coordinate_path = Path(coordinate_path)
        self.header = read_native_dense_header(self.mat_path)
        self.dof_row = dimension
        self.dof_col = dimension - 1
        self.basis_functions_per_element = basis_functions_per_element
        self.ordering_status = "unresolved"
        if self.header.rows % self.dof_row or self.header.cols % self.dof_col:
            raise ValueError("matrix dimensions are not complete patch-major dof blocks")
        n_patches = self.header.rows // self.dof_row
        if self.header.cols // self.dof_col != n_patches:
            raise ValueError("row and column dimensions imply different patch counts")
        if n_patches % basis_functions_per_element:
            raise ValueError("patch count is not divisible by basis_functions_per_element")

        self.chunk_rows = chunk_rows
        self.coordinates = read_native_coordinates(
            self.coordinate_path, n_patches, dimension=dimension
        )
        grouped = self.coordinates.reshape(-1, basis_functions_per_element, dimension)
        delta = grouped[:, :, None, :] - grouped[:, None, :, :]
        element_diameter = np.sqrt(np.sum(delta * delta, axis=-1)).max(axis=(1, 2))
        lengths = np.repeat(
            element_diameter / basis_functions_per_element, basis_functions_per_element
        )
        self.alignment: PCAAlignment = pca_align(self.coordinates)
        self.mesh = FaultMesh(self.alignment.centroids, lengths, dof_row=dimension)

    def _rhs(
        self, value: NDArray[np.floating], expected_rows: int, name: str
    ) -> tuple[NDArray[np.float64], bool]:
        array = np.asarray(value)
        vector = array.ndim == 1
        if array.ndim not in (1, 2) or array.shape[0] != expected_rows:
            raise ValueError(
                f"{name} must have shape ({expected_rows},) or ({expected_rows}, k), "
                f"got {array.shape}"
            )
        return np.asarray(array, dtype=np.float64).reshape(expected_rows, -1), vector

    def _chunks(self, stream: BinaryIO) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
        for start in range(0, self.header.rows, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.header.rows)
            yield start, stop, _read_values(stream, self.header, start, stop)

    def matvec(self, omega: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return ``A @ omega`` after one sequential, bounded-memory pass."""
        rhs, vector = self._rhs(omega, self.header.cols, "omega")
        result = np.empty((self.header.rows, rhs.shape[1]), dtype=np.float64)
        with self.mat_path.open("rb") as stream:
            for start, stop, block in self._chunks(stream):
                result[start:stop] = block @ rhs
        return result[:, 0] if vector else result

    def rmatvec(self, psi: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return ``A.T @ psi`` after one sequential, bounded-memory pass."""
        rhs, vector = self._rhs(psi, self.header.rows, "psi")
        result = np.zeros((self.header.cols, rhs.shape[1]), dtype=np.float64)
        with self.mat_path.open("rb") as stream:
            for start, stop, block in self._chunks(stream):
                result += block.T @ rhs[start:stop]
        return result[:, 0] if vector else result

    def read_block(
        self, row_indices: NDArray[np.integer], col_indices: NDArray[np.integer]
    ) -> NDArray[np.float64]:
        """Read a bounded diagnostic block without exposing it to compression."""
        rows = np.asarray(row_indices, dtype=np.intp).reshape(-1)
        cols = np.asarray(col_indices, dtype=np.intp).reshape(-1)
        if rows.size == 0 or cols.size == 0:
            raise ValueError("diagnostic blocks must have at least one row and column")
        if rows.min() < 0 or rows.max() >= self.header.rows:
            raise ValueError("row index outside matrix")
        if cols.min() < 0 or cols.max() >= self.header.cols:
            raise ValueError("column index outside matrix")
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        result = np.empty((rows.size, cols.size), dtype=np.float64)
        with self.mat_path.open("rb") as stream:
            for position, row in enumerate(sorted_rows):
                values = _read_values(stream, self.header, int(row), int(row) + 1)
                result[order[position]] = values[0, cols]
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape ``(3*N, 2*N)``."""
        return (self.header.rows, self.header.cols)

    @property
    def maximum_chunk_bytes(self) -> int:
        """Maximum simultaneous row-buffer bytes during chunk iteration.

        A consumer can retain the previous native-endian block while the generator
        allocates both the next big-endian read buffer and its native-endian copy.
        """
        rows = min(self.chunk_rows, self.header.rows)
        return 3 * rows * self.header.cols * 8
