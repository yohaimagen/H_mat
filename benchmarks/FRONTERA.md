# Frontera native-dense feasibility preflight

This is bounded evidence for the `frontera_np20k_3d` checkpoint, not a C.11
baseline and not a BP7 correction. The checkpoint is a complete PETSc native
dense matrix with shape `60,420 x 40,280` (`3 x 20,140` rows and
`2 x 20,140` columns). Its 19.47 GB value block begins directly after the
24-byte Tandem/PETSc header.

Physical ordering is now resolved from executable Tandem source on `hise_jump`:
checkout `/home/ymagen/tandem_h_matrix/tandem`, branch
`ymagen/strumpack-production`, commit `c4eb1b4615b0`, function
`app/form/StrumpackGFFullOperator.cpp::build_index_maps`, corroborated by the
blocked assembly code. The native-dense and `gf_coords` features match this
dataset. For logical patch `i=e*nbf+n`, the raw row is
`e*(3*nbf)+alpha*nbf+n` and raw column is `e*(2*nbf)+beta*nbf+n`, with
`nbf=10`. The later `dump_fault_node_coords` comment and downloaded
`analyze_gf.py` patch-major helpers contradict this executable mapping and are
not ordering authority.

`NativeDenseGF` exposes logical patch-major products and diagnostic blocks.
It scatters input vectors to raw Tandem order, streams the original matrix,
then gathers outputs to logical order. Two O(N) maps occupy 805,600 bytes;
there is no matrix permutation, rewrite, or full-matrix cache. Tests encode
independent tiny logical matrices into element-blocked files and cover both
products, vector and multiple-RHS shapes, adjoint identity, repeated diagnostic
indices, valid basis counts including one, and invalid basis counts.

The fresh corrected-order measurement made exactly one forward and one adjoint
pass, each with three right-hand sides, reading 38,939,481,600 value bytes.
They took 20.878 and 20.895 seconds (0.933 and 0.932 GB/s), respectively.
Forward linearity, adjoint linearity, and adjoint-identity relative residuals
were `9.42e-15`, `1.51e-15`, and `1.19e-15`. Process peak RSS at the streaming
checkpoint was 183.2 MB. The chunk-lifetime allowance remains 61.9 MB: the
previous native-endian block, next raw read buffer, and native-endian copy.

Five bounded spatial probes give median Spearman correlation `-0.847` and
median same-location strength rank zero. These are sanity checks, not the
basis for identifying ordering. The coordinate cloud is planar after PCA
(`94.25%`, `5.75%`, and effectively zero variance).

Corrected physical block spectra sample closest and farthest interaction-list
pairs at levels 2–4, using at most 64 patches per side. The smallest retained
ranks for a next-singular-value ratio at most `1e-4` are respectively
`29, 9, 26, 7, 34, 10`; for `1e-6` they are `118, 59, 99, 50, 88, 67`.
These bounded spectra do not certify every full block or operator-wide error.
Rank `r` retains the first `r` singular values and tests the next singular
value against the inclusive tolerance. The diagnostic candidate remains
`m=512, k=47, p=10`; it is not a validated production rank.

The fixed scheduler predicts 73 forward and 64 adjoint calls, 13,704 combined
construction columns, and 2.67 TB read across 137 matrix passes. Finalized
factors plus dense leaves are upper-bounded at 3,161,579,392 numerical bytes.
The conservative known-live-array model includes the probe, full product,
truncated product, subtraction result, raw permuted input/output, persistent
maps, retained factors/leaves, and finalization overlap. Its peak is
5,084,245,744 bytes, including maximum forward and adjoint workspaces of
1,897,349,120 and 190,443,840 bytes. It excludes Python/BLAS overhead, allocator
fragmentation, and unmodeled LAPACK workspace; it is not an RSS upper bound.
The new width-three timings imply a 47.7-minute I/O floor and a deliberately
pessimistic 26.5-hour linear-RHS upper extrapolation. Neither is a full-run
timing or a reliable compute-time estimate.

Full compression remains gated: predicted traffic exceeds the declared 1 TB
read ceiling, no frozen Frontera operator-wide tolerance or independent
validation plan exists, and bounded block spectra cannot certify full-block
ranks. No full compression, matrix rewrite, cache, or large output was made.

The JSON `timing_history` explicitly preserves the earlier raw-order timing
record, provenance, correction metadata, and record hash. Those historical
passes sustained about 0.943/0.942 GB/s with 195 MB peak RSS; correction rounds
reused their timings and only revised ordering hypotheses or lifetime
accounting. They are distinct from this fresh corrected-order measurement.
Current provenance includes source hashes, the supplied base revision
`16646f8`, command, thread settings, and source-revision evidence. Git state
was not collected because the implementer is prohibited from using Git.

Reproduce the bounded measurement with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m benchmarks.frontera \
  --revision <source-revision> --run
```

The machine-readable record is
[`results/frontera_np20k_preflight.json`](results/frontera_np20k_preflight.json).
