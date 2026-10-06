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

## Correction round 1: bounded block diagnosis

The reviewer requested evidence separating rank loss, sampled-basis error,
core amplification, and peeling contamination. The
[diagnostic record](results/bp7_block_diagnostics.json) uses seed 11 and the
incumbent `(512,47,10)` configuration. It chooses the largest scalar-area pair
and the first interaction pair at each level, with a hard eight-million-entry
block limit. The four actual blocks contain at most 1,177,110 entries. These
are reference-only diagnostic blocks; no entry access was added to compression.

The following errors are **Frobenius-relative block errors**, not the frozen
operator power estimates. The two-sided projection uses the measured bases
with the optimal projected core; reconstruction uses Eq. 4.3. The SVD column
uses the best rank-`k_eff` approximation of the entire rectangular block.

| Level / shape | Best-rank SVD error | Two-sided projection error | Reconstruction error | Relative column-sample contamination |
|---|---:|---:|---:|---:|
| 2 / 615×1914 | 0.005757 | 0.012942 | 0.029232 | 1.21e-15 |
| 2 / 186×410 | 0.000670 | 0.002344 | 0.005014 | 3.71e-16 |
| 3 / 840×550 | 0.003461 | 0.006967 | 0.013893 | 8.58e-16 |
| 3 / 21×20 (effective rank 20) | 1.37e-15 | 8.39e-8 | 5.80e-7 | 8.10e-7 |

For the three genuinely truncated representatives, sketch condition numbers
are 16.2–19.9 and the core increases the projection error by about 2.0–2.3×.
Replacing the measured column sketch by the exact block sketch, or rebuilding
both bases from exact block sketches, gives essentially the same reconstruction.
Thus their main limitation is real singular-value decay and sampled low-rank
approximation, not contamination from earlier levels. At rank 48, their
singular-value ratios to the leading singular value are 1.22e-3, 3.89e-4, and
6.53e-4. The first two are level 2, with no preceding admissible level to peel.

The weak component is poorly represented by the shared dominant subspace:
on the 186×410 block, the globally optimal rank-47 approximation still has
component-0 error 0.532, compared with 1.478 for reconstruction. Its own
component-only rank-47 SVD tail is 0.0241. On the 840×550 block, even the
component-only tail is 0.0932. These are local diagnostics, not lower bounds
on the full operator acceptance metrics and not proof of infeasibility.

The small full-rank level-3 representative isolates a different effect:
production reconstruction error is 5.80e-7, exact column samples with the same
measured bases reduce it to 1.20e-7, and clean two-sided sampling recovers the
block to 1.73e-15. Earlier approximation error can therefore contaminate even
an otherwise exact small block. No solver formula was changed to conceal this.

Based on these observations, exactly two further full configurations were
declared before their runs: more rank `(512,51,6)` and more oversampling
`(512,37,20)`. Both keep sketch width 57 and predict 10,968 construction columns.
The runner now enforces the strict predicted budget before native preparation
or compression; both candidates' observed counts match the prediction.
Every construction seed 11/23/37 and validation start 101/103 is retained.

| Candidate | Worst global | Worst component | Worst input | Minimum leaves improvement | Numerical entries |
|---|---:|---:|---:|---:|---:|
| More rank, `(512,51,6)` | 1.6984e-5 | 0.52692 | 5.2539e-5 | 434.71 | 85,457,835 |
| More oversampling, `(512,37,20)` | 3.8925e-5 | 0.38126 | 8.3102e-5 | 274.81 | 76,044,284 |

Both fail component/input/improvement on every seed; the oversampling candidate
also fails global error. Both pass selected-block, numerical storage, and
construction-budget checks. The original BP7 final record is retained as the
incumbent failed baseline, not relabelled as successful. The candidate
[rank config](configs/bp7_full_rank51.json) / [report](results/bp7_fixed_rank51.json)
and [oversampling config](configs/bp7_full_stable37.json) /
[report](results/bp7_fixed_stable37.json) preserve this additional history.

Correction runs used the same thread caps, serial execution, and
orchestrator-supplied revision `eaba6dc` plus exact source/config hashes.
The commands were:

```sh
.venv/bin/python -m benchmarks.diagnose --run
.venv/bin/python -m benchmarks.full --dataset bp7 --config benchmarks/configs/bp7_full_rank51.json --output benchmarks/results/bp7_fixed_rank51.json --revision eaba6dc --run
.venv/bin/python -m benchmarks.full --dataset bp7 --config benchmarks/configs/bp7_full_stable37.json --output benchmarks/results/bp7_fixed_stable37.json --revision eaba6dc --run
```

All commands used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1`. Setup ranges were 23.35–27.79 s and 23.99–24.19 s;
maximum RSS was 3.478 GB and 3.183 GB, respectively. Repeated memory checks
again found zero throttled pages and unchanged swap-outs (2,810,527), with
some page-out activity. No full runs exceeded the predicted sampling budget.
The unresolved issue is now supported by local approximation diagnostics and
two controlled rank/oversampling tradeoffs, but still does not establish that
all fixed configurations fail. **C.11 remains unaccepted; coloring stays closed.**

Correction-round validation: focused C.11 contracts passed 5 tests in 0.21 s;
the serial full command `.venv/bin/pytest -q -o addopts=''` passed **404 tests
in 99.22 s**. `.venv/bin/ruff check .` passed; `.venv/bin/black --check .`
reported 56 files unchanged; `.venv/bin/mypy` found no issues in 21 source files.
The three new reports' recorded source/config hashes, both full candidates'
seed/start completeness, and predicted/observed counts were independently
checked against the saved files.

## Correction round 2: provenance repair and finite screening

The original `bp7_block_diagnostics.json` is preserved unchanged. Its hard-coded
revision label, absent coordinate identity, and missing benchmark-helper hashes
limit its provenance. A fresh rerun is saved separately as
[bp7_block_diagnostics_v2.json](results/bp7_block_diagnostics_v2.json). It requires
an explicit `--revision fb9646b`, records both matrix and coordinate file
identities, and hashes the executed diagnostic, `benchmarks/c10.py`,
`benchmarks/full.py`, production sources, and exact configuration. The numerical
block records reproduce the historical diagnostic; metadata was not silently
patched into the old measurement.

Before screening, [bp7_screen.json](configs/bp7_screen.json) declared six remaining
splits at `m=512`, `k+p=57`: `(41,16)`, `(44,13)`, `(49,8)`, `(53,4)`, `(55,2)`,
and `(57,0)`, with incumbent `(47,10)` as comparator. Each uses construction
seeds 11/23/37 and the same four deterministic representative blocks. The screen
uses clean block-only Gaussian sketches matching the production fixed schedules;
it excludes peeling contamination and is not full validation. It evaluates 84
block/seed/configuration combinations without preparing a full native dense
operator. Every configuration's predicted full construction cost is 10,968
columns, below 11,040.

The predeclared promotion rule required at least a twofold reduction in worst
representative component-0 reconstruction error, with no worsening of worst
whole-block reconstruction error; at most two candidates could be promoted.
This was a decision rule for allocating full runs, not a new accuracy tolerance:
the incumbent still misses full component and input thresholds by factors of
about 5.1 and 16.3. The raw worst errors across all three seeds and four blocks
are shown below; they are Frobenius-relative block errors.

| `(k,p)` | Worst component-0 error | Worst whole-block error |
|---|---:|---:|
| `(47,10)`, incumbent | 1.609005 | 0.029232 |
| `(41,16)` | 1.959434 | 0.025303 |
| `(44,13)` | 1.944188 | 0.026350 |
| `(49,8)` | 2.089095 | 0.038881 |
| `(53,4)` | 2.442957 | 0.064161 |
| `(55,2)` | 4.316748 | 0.124521 |
| `(57,0)` | 13.587588 | 0.290165 |

No candidate improves the worst component error at all, so none meets the
promotion rule. More oversampling improves the worst whole-block error slightly
but sacrifices component quality; reducing oversampling progressively amplifies
both errors. The complete [screen report](results/bp7_screen.json) includes
singular tails, projections, reconstructions, sketch conditioning, exact source
hashes, and all seeds. No new full runs were promoted, no frozen threshold was
changed, and no previously failing baseline was relabelled. The evidence applies
to this finite set and these representatives, not every possible fixed
configuration. **The BP7 accuracy gate remains unresolved.**

With `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`, commands
ran serially:

```sh
.venv/bin/python -m benchmarks.diagnose --revision fb9646b --run
.venv/bin/python -m benchmarks.screen --revision fb9646b --run
```

The screen report hashes its own runner in addition to the shared provenance
sources. Memory checks again recorded zero throttled pages and unchanged
swap-outs (2,810,527). The screening run did not increase the observed cumulative
page-out counter (301,606). Historical records retain their original provenance;
the new records identify the corrected execution explicitly.

Correction-round validation: focused C.11 contracts passed 7 tests in 0.37 s.
The serial `.venv/bin/pytest -q -o addopts=''` run passed **406 tests in
101.68 s**; `.venv/bin/ruff check .` passed; `.venv/bin/black --check .`
reported 57 files unchanged; `.venv/bin/mypy` found no issues in 21 source
files. Both new reports' hashes and coordinate provenance were verified,
all 84 screen combinations and their budgets were checked, and the v2
diagnostic's numerical records match the historical record exactly.

## Correction round 3: alternative leaf-size screen

This final bounded action keeps the frozen accuracy thresholds and equations
unchanged. Before measurement, the screen declared three alternative leaf
sizes and required each one to use the largest sketch width whose predicted
combined forward-plus-transpose construction count is strictly below BP7's
11,040 input columns. Based on the previous full runs, which gave better weak-
component error to the oversampling-heavy `(37,20)` split than to `(51,6)`, and
on the `m=512` screen, which worsened below `p=10`, the declared splits retain
about 30% of their smaller widths for oversampling: `(m,k,p)=(8,17,6)`,
`(16,23,10)`, and `(128,37,16)`. This rationale was stored in the configuration
before the run.

The same deterministic largest-area and first interaction pair per level was
selected independently for each candidate tree, subject to the existing
eight-million-entry block limit, and each was screened with seeds 11, 23, and
37. The clean block-only screen remains a local Frobenius diagnostic, not a
replacement for frozen full-operator validation. The incumbent remains
`(512,47,10)`.

| `(m,k,p)` | Width | Combined predicted construction columns | Representative records | Worst component-0 error | Worst whole-block error |
|---|---:|---:|---:|---:|---:|
| `(512,47,10)`, incumbent | 57 | 10,968 | 12 | 1.609005 | 0.029232 |
| `(8,17,6)` | 23 | 10,816 | 42 | 2.840119 | 0.220433 |
| `(16,23,10)` | 33 | 10,830 | 30 | 2.518317 | 0.091848 |
| `(128,37,16)` | 53 | 11,020 | 18 | 1.684613 | 0.028393 |

All candidates meet the strict construction-count bound and are verified to be
at their maximum feasible widths: raising the width by one reaches or exceeds
11,040. None reduces the incumbent's worst component-0 error, much less by the
required factor of two. The `m=128` candidate modestly improves whole-block
error but has worse component error; the two smaller leaf sizes worsen both.
The promotion list is therefore empty. No full BP7 run was authorized, no
threshold changed, and no coloring/C.12 work was started. This is a precise
blocker for these three declared candidates and representatives, not an
infeasibility claim about every fixed configuration.

The run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1` and was serial. A post-run `vm_stat` check reported
zero throttled pages; the bounded reference blocks avoid full native-reference
preparation. Focused C.11 tests passed 9 tests, and the serialized full suite
passed 408 tests in 108.16 s; `ruff`, Black check, and mypy also passed. The
complete machine-readable evidence is in
[bp7_screen.json](results/bp7_screen.json), including exact source/config
hashes and every representative record.
