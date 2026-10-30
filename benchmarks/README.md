# Benchmark reports

## C.16 paired bounded regression

**Acceptance is blocked**, not completed: BP3 passes; BP7's unchanged primary
fails paired/near-field checks and the colored leaves-only gate. See
[RESULTS.md](RESULTS.md#c16-public-coloring-bounded-regression). The C.15 rank-4
negative test remains. No thresholds, seeds, primary parameters, or paired
formula were tuned after the C.16 outcomes.

Run datasets **serially**, using the frozen C.10 primary settings, three
construction seeds and two validation starts. The runner reads a representative
192-patch reference; it never compresses the full matrix:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m benchmarks.coloring --dataset bp3 \
  --data-root /path/to/real_gfs --revision COMMITTED_BASE --run \
  --output /tmp/c16_bp3_coloring.json
# After BP3 completes, repeat with --dataset bp7 and a different output path.
```

Omitting `--run` produces `not_run`; unavailable data produces `blocked`.
The supplied revision identifies the base; actual source/config SHA-256 hashes
bind the working implementation. Reports retain all seeds, raw paired errors
and limits, worst near-field block identities, permutation/adjoint/coverage
checks, and per-pass predicted/observed products. They make no RSS or speedup
claim. Tracked JSON files are byte-identical copies of the reported `/tmp`
outputs; no matrices or factors are retained.

The predeclared comparison is
`colored_absolute <= 1.1 * max(fixed_absolute, absolute_response_floor)` for
response errors, with C.10's floor of 1.0. Global errors require
`colored_global <= 1.1 * fixed_global + 64 * float64_eps`. All absolute C.10
gates remain independent and unchanged, as does the 5% near-field criterion.
Structural checks require complete/disjoint coverage, adjoint discrepancy below
`1e-12`, and permuted global errors within the dataset's gate.

The existing synthetic and subset benchmark CLIs also accept
`--sampling fixed|coloring` (default `fixed`) through the shared public pipeline.
`products.budget` describes the selected schedule; `products.fixed_budget`
retains the counterfactual fixed budget. `sampling_report.passes` records
requested/selected strategy, fallback reason, graph sizes/build time, coloring
time, and fixed/selected probe/column counts. Fixed requests skip graphs and
report null sizes, not fabricated measurements. Validation products stay separate.

The Frontera native-dense feasibility measurement is documented in
[`FRONTERA.md`](FRONTERA.md). It streams the 19.47 GB checkpoint without a
native full-matrix copy and stops at a documented preflight gate. Physical
row/column ordering is resolved from Tandem's executable index map, and the
reader exposes logical patch-major products. Bounded physical block spectra
do not replace operator-wide validation. This is not a C.11 baseline or a BP7 correction.

The reviewed-before-run [formal Frontera protocol](FRONTERA.md#formal-c11-confirmation)
uses three fixed-path seeds and the revised BP3-plus-Frontera C.11 acceptance
contract. BP7 remains documented negative evidence. Formal confirmation is
opt-in and has not been run as part of implementing the runner.

Run the reproducible synthetic report with one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python benchmarks/synthetic.py
```

The report is JSON and records automatic Git provenance (commit, dirty flag,
and dirty-diff hash), data identity, environment,
seeds, parameters, product counts, storage, and timing protocol. Construction
counts are snapped before application/error work. `budget` counts use the
occupied fixed-pattern groups and actual leaf widths; construction and leaves
are exact, while validation/total budgets are upper bounds because the power
method may converge early. Actual validation and total observations are
recorded separately.

`numerical_bytes` is the H-matrix's factor plus leaf allocations;
`retained_allocation_bytes` additionally includes tree/index allocations.
Both exclude the dense reference. `compressed_peak_rss_bytes` and
`dense_peak_rss_bytes` are collected in separate spawned processes; the
compressed process does include the reference operator, which is why
`dense_storage_ratio` and `rss_ratio` are intentionally different quantities.
Disk cache size is zero unless a caller passes a cache path to
`representation_storage`.

The reported dense comparison uses a native-endian reference, identical RHS
width, thread settings, and one warm product before repeated timings. Sampled
columns are not a wall-clock speedup claim. Local factorization timing is
measured by benchmark-only wrappers around QR/core routines; sampling/peeling
is reported after subtracting that measured subtotal.

Run `benchmarks/subset.py --dataset bp3` (or `bp7`) for the C.7 subset
contract. With data absent it writes an explicit JSON `status: "skipped"`
record. With data present it records the representative subset's IDs/maps
hashes, region choice, source file size/mtime identity, and a native-endian
dense reference constructed from that bounded subset only. It then runs the
same construction, error-validation, storage, and apply-time accounting core
as the synthetic report; it never fabricates BP3/BP7 metrics.

`benchmarks/c9.py` records the C.9 before/after measurement for core
absorption.  It also records the fraction of low-rank factors whose stored
entry count is not less than their dense block's entry count.  That fraction
is a diagnostic only: C.9 deliberately leaves dense far-field fallback,
tree-order permutation, and a native-endian cache out of production unless
real-data measurements justify them.

`benchmarks/c10.py --dataset bp3` (or `bp7`) writes a blocked record if the
real matrix is missing. With the matrix present, add `--run` to opt into the
bounded fixed sweep; full runs remain opt-in. The tracked
`benchmarks/configs/*_fixed.json` files now contain measured, frozen subset
operating points and criteria. Completed reports evaluate every primary
construction-seed/independent-validation-start record; they do not select a
favorable seed or turn the old subset report into an operating point.

Full C.11 runs are explicitly opt-in and must run one dataset at a time:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m benchmarks.full --dataset bp3 \
  --config benchmarks/configs/bp3_full_fixed.json --revision <source-revision> --run
```

Use `bp7` and its corresponding full configuration for the second run. Omit
`--run` to write a `not_run` record without loading data. `--revision` is supplied
by the orchestrator; the report additionally hashes the runner, all production
Python sources, and the exact config, so the revision label does not stand in
for the measured source identity. No Git command is executed by this runner.

The full runner converts the real reference's values to one native-endian
file-order array in bounded row chunks, preserving its patch-major input/output
permutations. Preparation time is charged separately and in each seed's total
setup time. The compressor continues to use only products. Every seed is built
once, validated with both frozen C.10 starts, and timed against this same dense
operator. Construction counts include both directions and dense leaf extraction;
all diagnostic products are counted separately. Break-even is reported only
when that direction's compressed apply is faster. RSS is a process lifetime
high-water mark, including the dense reference and preparation, not factor size.

The full configuration files preserve the subset configurations and link to
failed initial full reports. Accuracy thresholds are unchanged. See
[`RESULTS.md`](RESULTS.md) for outcomes and the gate to coloring. Monitor memory
pressure externally while running; the runner checkpoints each seed but does
not autonomously terminate on operating-system memory pressure.

C.11 correction diagnostics are explicitly opt-in:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m benchmarks.diagnose --revision <source-revision> --run
```

This serial BP7 diagnostic selects the largest and first interaction pair at
each level for construction seed 11. It compares exact bounded reference blocks
(at most eight million entries each) with the sampled bases and Eq. 4.3 core.
SVD tails, projection/reconstruction errors, sketch condition numbers, and
clean-versus-peeled sample errors are Frobenius-relative block diagnostics;
they do not replace any frozen operator-wide check. No dense reference block
is exposed to the compressor. The full runner now rejects a configuration
whose predicted combined construction columns are not strictly below `n_cols`
before preparing the native operator or constructing factors.

The corrected diagnostic requires an explicit revision and records matrix and
coordinate identities plus hashes for its imported benchmark helpers. Its
default output is `bp7_block_diagnostics_v2.json`; the original diagnostic
record is retained with its historical, incomplete provenance.

The finite correction-round screen is declared in `configs/bp7_screen.json`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m benchmarks.screen --revision <source-revision> --run
```

It compares deterministic largest/first bounded blocks at every level of each
declared tree across all three construction seeds, using clean block-only
sketches from the fixed schedules. Each declared width must be the largest whose
predicted combined construction columns remain strictly below `n_cols`; it
rejects a nonmaximum width before sampling. It excludes peeling and full
validation; its predeclared promotion rule is a screening decision, not a
replacement for any frozen threshold. It reads bounded reference blocks without
preparing a native copy of the entire dense operator.

Frontera correction round 2 follows a separate three-stage protocol: bounded
whole-block screen, one exploratory construction at `(m=512,k=96,p=32)`, then
fresh formal v2 confirmation only after complete passing evidence validates.
The exploratory early gates are 150x random leaves-only improvement and `5e-4`
selected-block error; formal thresholds and failed v1 evidence are preserved.
See [the commands, resource bounds, and provenance contract](FRONTERA.md#correction-round-2-exploratory-gate-and-evidence-validation).
