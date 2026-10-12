# Frontera native-dense feasibility preflight

This is bounded evidence for the `frontera_np20k_3d` checkpoint, not a C.11
baseline and not a BP7 correction. The checkpoint is a complete PETSc native
dense matrix with shape `60,420 x 40,280` (`3 x 20,140` rows and
`2 x 20,140` columns). Its 19.47 GB value block begins directly after the
24-byte Tandem/PETSc header.

The streamed operator made one forward and one adjoint pass, each with three
right-hand sides. Those two passes jointly check forward linearity, adjoint
linearity, and the adjoint identity. They sustained 0.943 and 0.942 GB/s,
respectively; process peak RSS was 195 MB, with at most 41.2 MB of simultaneous
big/native-endian row buffers. Relative residuals were between `1.5e-15` and
`9.5e-15`.

The physical scalar ordering is unresolved. Five spatial probes were evaluated
under all nine row/column combinations of patch-major, component-blocked within
each 10-basis-function element, and globally component-major layouts. Several
hypotheses retain strong decay: patch-major/patch-major has median Spearman
correlation `-0.871`, patch-major rows with element-blocked columns is even
stronger at `-0.889`, and element-blocked/element-blocked is `-0.847`. Spatial
decay therefore does not identify the writer layout, and no authoritative local
writer metadata was available. The coordinate cloud is planar after PCA
(`94.25%`, `5.75%`, and effectively zero variance), but no physical tree/dof
mapping is accepted until the ordering is established independently.

The earlier `64 x 64` patch spectra are retained in the JSON only as
`conditional_block_preflight`: they assume patch-major rows and columns, are
marked untrusted, and are not used by the gate or as evidence for a production
rank. Their rank fields now state the boundary convention explicitly: rank
`r` retains the first `r` singular values and tests the next singular value
against the inclusive tolerance.

For the diagnostic candidate `m=512, k=47, p=10`, the fixed scheduler predicts
73 forward and 64 adjoint calls, 13,704 combined construction columns, and
2.67 TB read across 137 full matrix passes. Finalized factors plus dense leaves
are upper-bounded at 3.16 GB of numerical storage. The array-only peak-memory
model includes simultaneous probe, full product, truncated product, subtraction
result, retained factors/leaves, and finalization overlap, giving 4.49 GB of
known-live arrays. It excludes Python/BLAS overhead, allocator fragmentation,
and unmodeled LAPACK workspace, so it is not an RSS upper bound. The measured
width-three passes imply a 47-minute I/O floor. Charging one third of that
width-three time to every scheduled column gives 26.2 hours, but this is a
deliberately pessimistic upper extrapolation that ignores wider-BLAS efficiency,
not a compute-time estimate or full-run timing.

Full compression was not run. It is gated because there is no frozen
Frontera-specific operator-wide accuracy tolerance or independent validation
plan, the physical ordering is unresolved, conditional block samples cannot
certify every full block, and predicted construction traffic exceeds the
predeclared 1 TB read ceiling. No large or temporary data were written.

Correction round 1 reused the original two-pass timings without another full
matrix scan. It read 45 selected rows (14,500,800 bytes) to expand the ordering
hypotheses, corrected deterministic derived fields, and records the original
measurement invocation, current source hashes, SciPy/CPU details, and the hash
of the original measured JSON. Git state is explicitly unavailable because
this task prohibited Git commands.

Reproduce the bounded measurement with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m benchmarks.frontera \
  --revision <source-revision> --run
```

The machine-readable record is
[`results/frontera_np20k_preflight.json`](results/frontera_np20k_preflight.json).
