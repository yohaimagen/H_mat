# Benchmark reports

Run the reproducible synthetic report with one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GFCOMPRESS_REVISION=$(git rev-parse HEAD) \
  .venv/bin/python benchmarks/synthetic.py
```

The report is JSON and records the revision, data identity, environment,
seeds, parameters, product counts, storage, and timing protocol. Construction
counts are snapped before application/error work. `predicted` counts use the
occupied fixed-pattern groups and actual leaf widths; they include `k + p` and
have separate validation fields. The current synthetic entry point has no
validation probes, so those fields are zero.

`numerical_bytes` is the H-matrix's factor plus leaf allocations;
`retained_allocation_bytes` additionally includes tree/index allocations.
Both exclude the dense reference. `peak_rss_bytes` is collected in a spawned
process and does include the reference operator, which is why
`dense_storage_ratio` and `rss_ratio` are intentionally different quantities.
Disk cache size is zero unless a caller passes a cache path to
`representation_storage`.

The reported dense comparison uses a native-endian reference, identical RHS
width, thread settings, and one warm product before repeated timings. Sampled
columns are not a wall-clock speedup claim. Local factorization timing is
`null`: it is not separately observable without changing compressor math.

BP3/BP7 data are intentionally not fabricated. Their C.7 tests continue to
skip when the source files are absent; a future data-aware runner should call
the same helpers on `representative_subset` and preserve this schema.
