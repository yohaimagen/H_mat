# Frontera native-dense feasibility preflight

The original preflight below is bounded evidence for the `frontera_np20k_3d` checkpoint, not a C.11
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

## Formal C.11 confirmation

The PI revised the acceptance contract on 2026-10-22: BP3 and the larger
Frontera 3D GF must succeed; BP7 remains negative evidence, with all failed
records and attempted remedies preserved in [RESULTS.md](RESULTS.md). Its
small-fault geometry is the PI's reason for expecting insufficient meaningful
far field, not a new claim that BP7 passed. Coloring remains closed until the
formal Frontera confirmation passes alongside the accepted BP3 baseline.

The external pilot record
`benchmark_results/full_construction_seed20261012.json` is calibration only.
It constructed `m=512, k=47, p=10` in 48.7 minutes, matched 137 predicted and
observed passes / 13,704 columns, retained 3.162 GB numerical storage (16.2%
of dense), and reached 4.33 GiB lifetime peak RSS. Its single forward and
adjoint errors near `9.3e-8` are response checks, not power-method norm
estimates. The pilot seed is excluded from formal confirmation.

[`configs/frontera_full_fixed.json`](configs/frontera_full_fixed.json) freezes
the fixed path and construction seeds **11, 23, 37**, with independent
validation starts **101, 103**, before any confirmatory measurement. Review
this runner/configuration/contract before using `--run`. Implementation tests
use bounded fixtures only; no three-seed result is claimed here.

| Check, required for every construction seed and validation start | Limit |
| --- | --- |
| Global relative spectral-error estimate, eight power iterations | `1e-5` |
| Random, localized, smooth response error, forward and adjoint | `1e-5` |
| Component-index-group response error, forward and adjoint | `1e-4` |
| Selected genuinely truncated block unit-drive response error | `1e-3` |
| Leaves-only error / compressed error for every input and direction | at least `100` |
| Numerical storage and combined construction columns | strictly below dense entries and `n_cols` |
| Predicted versus observed construction calls and columns | exact match |

The `1e-5` global/input target gives roughly 100-fold margin over the pilot
random responses; it is an engineering target, not a norm guarantee inferred
from them. The component target still requires 0.01% accuracy and allows
margin over the pilot's weakest output error (`2.5e-6`). C.10 component groups
isolate inputs in both directions; they are not separate spectral norms of
every component suboperator. The block threshold is 0.1% relative to a unit
source-box response, with no absolute-floor loophole; a zero predicted block
has error one. Corrected bounded spectra near `1e-5` at rank 47 motivate this
target but cannot guarantee full-block reconstruction. A hundredfold advantage
over leaves-only prevents a near-field-dominated global error from being the
only success criterion. Response floors are `1e-12` in native operator units;
absolute error and reference response norms are also retained.

The runner reuses C.10 diagnostics unchanged: both-direction random/localized/
smooth inputs, all input-component groups, one largest genuinely truncated
admissible block per active level where available, and the power-method ratio.
No selected truncated block makes acceptance fail rather than vacuously pass.
Two independent starts reduce sensitivity to the power initialization, but
eight iterations are finite estimates, not certified upper bounds. Selected
unit drives are not exhaustive block spectral tests. No best seed is chosen.
Any failed formal gate remains failed; a revised configuration requires a
separately documented exploratory round and fresh confirmation, never a silent
threshold edit based on these seeds.

`benchmarks.frontera_full` uses a fresh process per seed and streams the
authoritatively mapped `NativeDenseGF`; it never calls native dense preparation
and saves no factors. Small atomic JSON checkpoints retain each completed
seed and an aggregate report outside the repository. `--resume` reuses both
passing and failing completed seeds only with matching source, configuration,
environment, data, and BP3/BP7 record identities. Interrupted seeds restart
from construction; their prior checkpoint metadata is preserved. This is
result checkpointing, not a factor checkpoint or mid-construction restart.

The formal resource protocol explicitly supersedes the old **preflight-only
1 TB ceiling** after review. It allows at most **300 passes / 6 TB per seed**,
including construction, validation, and repeated dense timing. The current
schedule bounds these at 253 passes (137 construction, at most 104 validation,
12 timing), approximately 4.93 TB per seed, or 14.78 TB for three seeds. One
additional sequential matrix SHA-256 scan per invocation uses 8 MiB buffers
and is recorded separately. This substantial I/O budget is intentional; it
must not be confused with the much smaller bounded preflight.

Guards require 100 GiB external free space, at most 8 GiB modeled live arrays,
12 GiB process lifetime RSS, and at most 1,024 RHS columns. The predictor
includes permutation buffers and maps. Runtime RSS/free-space checks occur
before/after products and at row-chunk boundaries; checks also run after seed
validation. These are cooperative stop guards, not an OS allocation limit:
an allocation can cross the RSS threshold before the next check. The modeled
array bound excludes Python/BLAS/LAPACK overhead, so measured RSS is retained.
One seed per fresh process prevents earlier seeds' allocator state from
contaminating its lifetime high-water measurement.

Provenance records the orchestrator-supplied reviewed revision, SHA-256 of all
compressor sources and benchmark helpers, exact configuration hash, full matrix
and coordinate hashes, data size/mtime, pilot identity, and BP3/BP7 record hashes.
Source hashes and data metadata are checked before and after every seed. The
matrix is assumed immutable during confirmation; it is not rehashed per product.
Construction, validation, and timing costs remain separate. Each direction has
one warmup and five timed single-vector products, using the same streamed dense
reference (including disk reads, endian conversion, and permutations). Times
are means, not compute-only dense BLAS comparisons. Break-even includes
construction, reader/preflight preparation, and one full-data hashing cost
charged to each independently usable seed; it is null if compressed applies
are no faster. Filesystem cache state is uncontrolled and no cache is created.

After review, from the C.11 worktree, the formal command is:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/Users/ymagen/.codex/worktrees/c11-frontera/H_mat/src \
/Users/ymagen/work/H_mat/.venv/bin/python -m benchmarks.frontera_full \
  --dataset '/Volumes/Extreme SSD/H_mat/frontera_np20k_3d' \
  --config benchmarks/configs/frontera_full_fixed.json \
  --revision '<reviewed-source-commit>' --run
```

Without `--run`, it prints the frozen plan and source provenance without
reading the external matrix or creating output. For an interrupted invocation,
repeat the same command with `--resume`. Results go to the external dataset's
`benchmark_results/formal_fixed/{frontera_fixed,seed_11,seed_23,seed_37}.json`.
Only reviewed small summaries may later be copied into the repository.
