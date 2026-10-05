# C.11 full fixed-pattern baselines

**C.11 is not accepted: BP3 passes, but BP7 still fails frozen discriminating
accuracy checks. The gate to coloring remains closed.** Both full datasets were
run with construction seeds 11, 23, 37 and independent validation starts 101,
103. No tolerance was relaxed and no seed was discarded.

The full runner reuses C.10's global power estimate, genuinely truncated
selected-block unit drives, component-index groups, random/localized/smooth
inputs in both directions, and leaves-only comparisons. It constructs once per
seed and snapshots construction products before any validation. Only benchmark
accounting and configurations changed; the compressor equations are unchanged.

## Results and mandatory gates

Values below are worst errors / minimum improvement over all six seed/start
records. Numerical entries and product costs are identical across construction
seeds for each final configuration.

| Measure | BP3 | BP7 | Requirement |
|---|---:|---:|---|
| Shape | 28000 × 14000 | 16560 × 11040 | full operators |
| `(m,k,p)` | `(8,16,10)` | `(512,47,10)` | configurations linked below |
| Tree dimension / depth / leaves | 1 / 11 / 2048 | 2 / 3 / 64 | measured; no oversized leaves |
| Global relative error | 8.0080e-14 | 1.8356e-5 | ≤1e-8 / ≤2e-5: pass / pass |
| Selected-block response error | 1.7150e-10 | 0.01424 | ≤0.0003 / ≤0.4: pass / pass |
| Component response error | 1.2763e-12 | 0.41019 | ≤5e-8 / ≤0.08: pass / **fail** |
| Input response error | 5.0662e-11 | 4.8857e-5 | ≤1e-8 / ≤3e-6: pass / **fail** |
| Leaves-only improvement | 1.3773e12 | 467.48 | ≥1000: pass / **fail** |
| Numerical entries | 17,276,498 | 82,840,000 | <392,000,000 / <182,822,400: pass |
| Numerical bytes | 138,211,984 | 662,720,000 | factors plus leaves |
| Retained allocation bytes | 143,910,264 | 663,916,400 | includes tree/index/mesh arrays |
| Forward calls / columns | 61 / 1529 | 61 / 8004 | includes leaf extraction |
| Transpose calls / columns | 58 / 1508 | 52 / 2964 | construction only |
| Total construction columns | 3037 | 10968 | <14000 / <11040: pass |
| Predicted vs observed counts | exact | exact | both calls and columns |
| Process peak RSS, maximum bytes | 4,789,125,120 | 3,365,666,816 | reference included |

[BP3 final record](results/bp3_fixed.json) and
[configuration](configs/bp3_full_fixed.json);
[BP7 final failed record](results/bp7_fixed.json) and
[configuration](configs/bp7_full_fixed.json).

## Timing

One warm product precedes five timed products of one RHS, with identical thread
caps and native-endian dense reference for both representations. Ranges span
construction seeds. Times are seconds; preparation includes reader/geometry
loading and bounded-chunk native-endian conversion. OS file caches are not
flushed, so preparation is not a cold-disk benchmark. Preparation is charged to
each independently usable seed's setup even though the process reuses its dense
reference across the three seeds.

| Measure | BP3 | BP7 |
|---|---:|---:|
| Operator preparation | 4.689 | 0.216 |
| Construction | 20.613–21.102 | 22.951–23.268 |
| Total setup | 25.301–25.791 | 23.167–23.484 |
| Forward compressed / dense apply | 0.05494–0.05604 / 0.02880–0.03006 | 0.02489–0.02505 / 0.01341–0.01413 |
| Adjoint compressed / dense apply | 0.05522–0.05662 / 0.08708–0.09026 | 0.02792–0.02807 / 0.02697–0.02809 |

BP3 forward apply is slower; adjoint setup break-even is about 767–798 applies.
BP7 forward apply is slower. Its adjoint timing is effectively tied: seed 11's
tiny positive difference produces an arithmetic break-even of about 133,402
applies, while seeds 23 and 37 are slower and have no break-even. This is not
evidence of a robust BP7 speedup, and BP7 fails accuracy regardless.

## Visible configuration history and diagnosis

| Dataset / attempt | `(m,k,p)` | Outcome |
|---|---|---|
| BP3 original C.10 config | `(8,4,2)` | all three seeds fail all five accuracy checks; storage/count pass |
| BP7 original C.10 config | `(8,6,2)` | all three seeds fail all five accuracy checks; storage/count pass |
| BP3 full revision | `(8,16,10)` | all gates pass |
| BP7 full attempt 1 | `(128,40,10)` | selected-block/storage/count pass; other accuracy checks fail |
| BP7 full attempt 2 | `(512,47,10)` | global/selected-block/storage/count pass; component/input/improvement fail |

The original subset configs remain unchanged. Failed reports are retained as
[BP3 initial](results/bp3_fixed_initial.json),
[BP7 initial](results/bp7_fixed_initial.json), and
[BP7 attempt 1](results/bp7_fixed_attempt1.json), with its
[archived config](configs/bp7_full_fixed_attempt1.json). Each full config carries
history links; each report embeds its complete measured config and its hash.
The attempt-1 report's original config path was subsequently revised; its
embedded config/hash matches the archived attempt-1 file.

Higher rank and oversampling materially improve the full-data approximation;
the subset-selected low ranks did not transfer to the full hierarchy. This is
a measured configuration limitation, not evidence of a changed reconstruction
equation. BP7's small adjoint component remains discriminating: in attempt 1,
seed 11/start 101, its reference norm is 4.858 and its absolute error is 5.843,
while the other two component responses are near ten million. Global error
alone therefore conceals an important failure.

Geometry-only BP7 budget checks give maximum feasible `k+p` of 23, 33, 42, 53,
and 57 for `m=8,16,32/64,128/256,512`, respectively. At `m=1024`, leaf probes
alone require 17,226 columns, already above 11,040. The final candidate spends
10,968 columns, leaving only 71 columns under the strict bound. This bounded
search does **not** prove that no fixed configuration can pass; it identifies
the unresolved accuracy-versus-budget constraint for review. No coloring was
implemented or used to conceal the failure.

## Reproduction and resource observations

Runs used macOS 26.7.1 arm64, Python 3.12.9, NumPy 2.4.6, and
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`. The
orchestrator supplied revision `a3d2cba` (based on merged `cc3c048`); reports
additionally hash the exact runner, C.10 helper, production sources, and config,
and record source file identities and exact commands. Full source/dataset paths
are local provenance, not portable requirements. The virtualenv import was
verified at `/private/tmp/H_mat-c11/src/gfcompress/__init__.py`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m benchmarks.full --dataset bp3 --config benchmarks/configs/bp3_full_fixed.json --revision a3d2cba --run
.venv/bin/python -m benchmarks.full --dataset bp7 --config benchmarks/configs/bp7_full_fixed.json --revision a3d2cba --run
```

Original runs omitted `--config`; the first BP7 revision used the now-archived
attempt-1 config. Dataset processes ran strictly serially. Repeated `vm_stat`
checks showed zero throttled pages and unchanged cumulative swap-outs
(2,810,527); compression and small page-out increases were observed. `ps` was
denied by the sandbox, so process RSS comes from `resource.getrusage` checkpoint
high-water marks. These include reference storage/preparation and retained
allocator memory across seeds, not just the compressed representation. Retained
allocation bytes count NumPy backing allocations, not Python object overhead.
No matrices, factors, or operator caches were written.

Validation after the benchmark runs: with the same three thread caps,
`.venv/bin/pytest -q -o addopts=''` passed **402 tests in 97.68 s**;
`.venv/bin/ruff check .` passed; `.venv/bin/black --check .` reported 55 files
unchanged; `.venv/bin/mypy` reported no issues in 21 source files. The focused
C.10/C.11 run initially passed 7 tests, and the final C.11 contracts passed 3
tests before the revised full runs. The copied virtualenv's pytest/Black/mypy
entrypoint shebangs were repaired locally after an initial collection failure
pointed at the original checkout; those environment files are untracked.
