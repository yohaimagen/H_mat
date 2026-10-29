# Frontera fixed baseline and historical preflight

**The C.11 scientific gate has passed for BP3 and Frontera formal v2. Coloring
is eligible only after human merge; no coloring result exists yet.** All three
fresh Frontera seeds (41, 43, 47) pass the unchanged frozen checks after the
versioned screen/exploratory correction path. The failed formal v1 results and
BP7 negative evidence remain preserved. The current result is documented in
[Accepted formal v2 result](#accepted-formal-v2-result), with a mechanically
derived [tracked summary](results/frontera_fixed_v2_summary.json).

## Historical native-dense feasibility preflight

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

This section preserves the original v1 protocol and its rationale; the later
versioned v2 result and fresh seeds are recorded below.

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

## Correction round 1: bounded v1 screen

The external formal v1 record remains failed: all three seeds passed the global,
component/input, storage, and count checks, but leaves-only improvement was only
76.84--79.94 (required: 100), and seed 11's selected L4 response was 0.001306
(required: 0.001). Those thresholds and seeds are preserved; this screen does
not rerun confirmation.

Before any new confirmation, the versioned screen compares clean randomized
reconstructions of the failed L4 `16,2` interaction and the predeclared L4
`4,16` representative. Their whole physical shapes are `1038 x 712` and
`1038 x 692`; both have minimum dimension above rank 128. Each block is read
once in row orientation and reused for all candidates and seeds, totaling
668,970,240 matrix-value bytes. The comparator is `(512,64,20)`;
the only alternatives are `(512,96,32)` and `(512,128,32)`. Their exact predicted
construction columns are 17,160, 22,792, and 26,888; reviewed storage/working-set
bounds are 3.54/4.25/4.96 GB and 5.49/6.25/7.04 GB, respectively.

Promotion is predeclared: at most one candidate may proceed only when its worst
clean L4 reconstruction improves the comparator by at least 20% and no bounded
representative block worsens. This is a local screen rule, not a frozen C.11
acceptance threshold. Candidates are considered in predeclared cost order:
rank 96 wins when both qualify. If only rank 128 qualifies, no full construction
is authorized by the current v2 configuration.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=/Users/ymagen/.codex/worktrees/c11-frontera/H_mat/src \
/Users/ymagen/work/H_mat/.venv/bin/python -m benchmarks.frontera_screen \
  --revision '<reviewed-source-commit>' --run
```

## Correction round 2: exploratory gate and evidence validation

The screen uses independent left/right PCG64 streams per block and seed. It
generates the maximum width once, slices prefixes for every candidate, and
records the hashes of both sliced probes. The locked plan includes full block
dimensions, exact row/column index hashes, occupied cell counts, and expected
row-read bytes. Runtime checks enforce those identities, actual bytes returned,
eight million maximum block entries, 4 GiB RSS, 100 GiB free disk, and an explicit
20 GB diagnostic matrix-read ceiling. No external matrix values were read in
this implementation round; coordinates and the 24-byte header established the
predeclared block identities.

Only promotion of `(512,96,32)` unlocks the one-construction exploratory stage
in [frontera_exploratory_v2.json](configs/frontera_exploratory_v2.json), using
construction seed 431 and independent validation seed 433. Both random forward
and adjoint responses must improve on leaves-only by at least 150x. Every
preselected largest genuinely truncated block per level must have response
error at most `5e-4`. This stage runs no power-method diagnostics or timing,
whether it passes or fails. A failure records the early evidence and stops.

```sh
# Use the same single-thread settings and reviewed revision as the screen.
PYTHONPATH=src .venv/bin/python -m benchmarks.frontera_full --exploratory \
  --screen-report '<dataset>/benchmark_results/frontera_screen_v1/frontera_screen.json' \
  --revision '<reviewed-source-commit>' --run

PYTHONPATH=src .venv/bin/python -m benchmarks.frontera_full \
  --config benchmarks/configs/frontera_full_fixed_v2.json \
  --screen-report '<dataset>/benchmark_results/frontera_screen_v1/frontera_screen.json' \
  --exploratory-report '<dataset>/benchmark_results/exploratory_v2/frontera_exploratory.json' \
  --revision '<reviewed-source-commit>' --run
```

Fresh formal seeds `41,43,47` and starts `107,109` remain locked until both
reports validate. The formal accuracy thresholds are unchanged. Outputs use
separate `exploratory_v2` and `formal_fixed_v2` directories; failed v1 evidence
is retained. Screen and exploratory evidence is published atomically and cannot
be overwritten, including with `--resume`.

Version-2 evidence binds canonical content hashes to the exact configurations,
plan, runner/helper/compressor source hashes, supplied reviewed revision,
coordinate digest, matrix stat identity, and ordering authority. Consumers
recompute the geometric identities and probe hashes and require complete seeds,
candidates, results, and matching aggregates; a bare promotion flag is rejected.
Exploration additionally hashes the full matrix once, and formal execution
independently verifies that digest. The screen does not scan the matrix for a
hash. Hashes provide reproducible content/provenance checks, not a cryptographic
signature from an independent authority: trusted orchestrator execution and an
immutable dataset during each stage remain assumptions. At this implementation
checkpoint no new Frontera accuracy or timing result was claimed; subsequent
execution completed the gates and formal v2 confirmation below.

## Accepted formal v2 result

The accepted external record is
`/Volumes/Extreme SSD/H_mat/frontera_np20k_3d/benchmark_results/formal_fixed_v2/frontera_fixed.json`.
It is 60,123 bytes with SHA-256
`a852e7a26cf9c19bff6387dc8ec2e938c54d8c56830d1a69087a84c5170ba3c7` and supplied
reviewed revision `ec2ec56`. The compact
[frontera_fixed_v2_summary.json](results/frontera_fixed_v2_summary.json) preserves
the report's configuration, source and data provenance, per-seed acceptance and
costs, and aggregate ranges/medians without copying the raw report or factors.
Its predecessor evidence is bound by canonical content hashes:

- Screen: `f7955af4de10e68a81ade092ececd6a44e1615ec9d75d605fc7e41473165ccb5`.
- Exploratory: `27202972d86cbaa51b038a1d3d9a19a6939995b1220c6f28ca4e15530efdf713`.

Formal v2 uses `(m,k,p)=(512,96,32)`, construction seeds **41/43/47**, and
independent starts **107/109**. All three seeds report `all_pass: true`.
Across the six seed/start records, the worst global estimate is `9.90953e-8`,
component response error `3.60097e-7`, input response error `3.98044e-7`, and
selected-block error `9.53540e-5`; minimum leaves-only improvement is `612.141`.
The unchanged limits are `1e-5`, `1e-4`, `1e-5`, `1e-3`, and `100`, respectively.
The pilot and exploratory seed remain outside formal confirmation. Failed v1
seeds 11/23/37 retain their original failed status and thresholds.

Numerical storage is 4,127,117,312 bytes, or 21.1976% of the 19,469,740,800-byte
dense value block (4.7175× compression). Retained allocations are
4,132,439,888 bytes; peak RSS ranges from 4,655,185,920 to 5,350,817,792 bytes.
Every seed exactly matches its predicted 137 construction passes and 22,792
combined construction columns, below `n_cols=40,280`. Forward counts are
73 calls / 14,600 columns; transpose counts are 64 calls / 8,192 columns.

Construction takes 2931.750–2973.207 seconds per seed. Preparation, including
the hashing charge, is 23.8595–23.8629 seconds; total setup is
2955.613–2997.066 seconds. Forward apply takes 0.07244–0.07599 seconds versus
20.77775–20.92235 for streamed dense, and adjoint apply takes
0.07696–0.08810 versus 20.78689–20.90702 seconds. Setup break-even is
142.667–143.769 forward applies and 142.343–143.957 adjoint applies. The
comparison includes dense I/O, conversion, and permutations; it is not a claim
about resident dense BLAS. Cache state is uncontrolled, and each mean uses five
single-vector repeats after one warmup. Exact per-seed values are in the summary.

Formal v2 recorded **14,797,003,491,384 total read bytes**: three seeds ×
253 passes × 19,469,740,800 value bytes = 14,777,533,267,200 streamed bytes,
plus 19,470,224,184 hashing bytes. Hashing comprises one full matrix-file scan
(including its 24-byte header) and 483,360 coordinate bytes. It is counted once
in total traffic, although its time is charged to each independently usable
seed's setup. Earlier stages and uninstrumented metadata/report reads are
excluded. This substantial traffic remains part of the cost of the baseline.

The exact command, single-thread environment, matrix digest/stat identity,
coordinate digest/stat identity, and compressor/helper hashes are retained in
the summary. Reproduction uses the versioned formal command above with
`--revision ec2ec56` and the recorded screen/exploratory paths; it requires the
external dataset and reviewed runner/configuration, not just the summary file.
The `coloring_unlocked` flag in the source report denotes scientific eligibility:
human merge remains required before coloring work, and **no coloring result
exists yet**. The finite power estimates and selected unit drives retain the
limitations of the frozen protocol.
