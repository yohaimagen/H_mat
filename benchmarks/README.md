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
