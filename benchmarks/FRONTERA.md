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

Five spatial probes support the expected patch-major order. The median
Spearman correlation between coordinate distance and patch-block magnitude
was `-0.871`; the component-major falsification null was `-0.029`. The
coordinate cloud is planar after PCA (`94.25%`, `5.75%`, and effectively zero
variance), so the production tree is two-dimensional while the operator keeps
three row and two column components per patch.

Bounded `64 x 64` patch samples from the closest and farthest interaction-list
blocks at levels 2--4 reached relative singular value `1e-4` by ranks 9--32.
At `1e-6`, they required ranks 62--120. These samples show useful far-field
structure around `1e-4`; they do not certify a global rank or operator error.

For the diagnostic candidate `m=512, k=47, p=10`, the fixed scheduler predicts
73 forward and 64 adjoint calls, 13,704 combined construction columns, and
2.67 TB read across 137 full matrix passes. Finalized factors plus dense leaves
are upper-bounded at 3.16 GB of numerical storage. The array-only peak-memory
model is 4.15 GB, excluding Python/BLAS overhead and allocator fragmentation.
Extrapolating the measured width-three products gives a 47-minute I/O floor
but a 26.2-hour linear-RHS compute estimate; neither is a full-run timing.

Full compression was not run. It is gated because there is no frozen
Frontera-specific operator-wide accuracy tolerance or independent validation
plan, bounded block samples cannot certify every full block, and predicted
construction traffic exceeds the predeclared 1 TB read ceiling. No large or
temporary data were written.

Reproduce the bounded measurement with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m benchmarks.frontera --run
```

The machine-readable record is
[`results/frontera_np20k_preflight.json`](results/frontera_np20k_preflight.json).
