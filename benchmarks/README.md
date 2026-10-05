# Benchmark reports

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
