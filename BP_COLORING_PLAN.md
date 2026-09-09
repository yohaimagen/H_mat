# BP_COLORING_PLAN — Validated BP3/BP7 compression, then graph coloring

## Objective

Deliver a running non-uniform H¹ compression package for the full BP3 and BP7
Green's-function datasets with both `sampling="fixed"` and
`sampling="coloring"`. First repair the necessary foundations, validate the
fixed-pattern algorithm on both datasets, and freeze reproducible baselines.
Only then implement coloring, rerun the same experiments, and compare results.

The algorithmic reference is Levitt & Martinsson (2024), *Randomized compression
of rank-structured matrices accelerated with graph coloring*, especially
Algorithm 2.1 and §4.1 / Algorithm 4.1. The local reference is named
`Levitt_N_Martinsson_2018.pdf`, but its publication year is 2024.

**Status:** implementation plan, not a report of completed work. Existing code
is reused and repaired; completed historical tasks are not reimplemented.
Current work includes the fixed-pattern compressor, the R.1 real-data reader,
and an R.2 branch with PCA/tree-dimensionality changes and uncommitted edits.
Reconcile that work before starting the tasks below; do not overwrite it.

**Relationship to earlier plans.** Once activated by Task C.0, this plan replaces
the remaining execution sequence in `REALGF_PLAN.md` and Stage 7 of `plan.md`,
and adds the necessary corrections to Stages 1–6. Earlier plans remain the
historical record. Do not execute both sequences in parallel or duplicate R.2.
This document's creation alone does not change `tasks.txt` or task dispatch.

---

## 1. Scope and scientific contract

### Operator and matrix format

- Treat dense `G @ omega` and `G.T @ psi` as the reference black-box products.
  Building a live Tandem adapter, proving savings in its elasticity solves,
  and integrating a time-stepping simulation are out of scope.
- The compressor accesses the operator exclusively through `matvec` and
  `rmatvec`. It never assembles the full matrix or requests arbitrary entries.
- Use non-uniform H¹: each admissible pair has its own `U`, `B`, and `V`.
  Uniform H¹ and H² are not part of this plan.
- Preserve patch-major rectangular indexing: BP3 has shape `28000 × 14000`
  with `(dof_row, dof_col) = (2, 1)`; BP7 has shape `16560 × 11040` with
  `(dof_row, dof_col) = (3, 2)`.
- Tree dimension and elasticity dimension are distinct. Expected retained
  tree dimensions are 1 for BP3 and 2 for BP7; measure and report them.
- Externally, products retain the input operator's ordering, even if internal
  factor storage is reordered. Component ordering and geometric rotation are
  separate: rotating tree coordinates must not rotate traction/slip values.

### Geometry and admissibility

- Adopt the paper's dyadic interaction-list partition as the production block
  partition. A configurable geometric `eta` filter is not introduced here.
- Use a padded hypercube root in the retained geometric coordinates, with one
  physical side length for all axes. Do not independently rescale axes: that
  changes the distance metric. Integer dyadic cell indices define adjacency.
- With equal-sided cells, interaction-list pairs satisfy the separation bound
  `dist/max(diam) >= 1/sqrt(tree_dim)`, subject to numerical rounding. Test
  that property as a diagnostic. Do not claim that every geometrically distant
  same-level pair belongs to the interaction list: many are represented at
  coarser levels already.
- Coverage means all admissible blocks over levels `2..L`, together with
  neighboring leaf blocks at `L`, cover every patch pair exactly once. At an
  intermediate level, neighbors plus interactions cover the children of the
  parent-neighborhood; they do not cover the entire matrix by themselves.
- Retain level-synchronous subdivision for a common leaf depth. Document it as
  this implementation's simplifying choice, not an explicit universal mandate
  of the paper. Keep termination guards and report leaves exceeding `m`.
- Drop only numerically degenerate PCA directions automatically. Preserve
  original physical coordinates and report discarded residual distances.
  Meaningful dimension reduction beyond roundoff requires an explicit choice.
- `L` remains documented mesh metadata, not an unimplemented admissibility
  control. Do not use its proxy sensitivity study as evidence about the actual
  partition. Basis-support geometry is a limitation to document and investigate
  if the real block-rank measurements reveal a problem.

### Numerical and testing rules

- Use one effective rank per block:
  `k_eff = min(k, len(alpha.row_indices), len(beta.col_indices))`.
- Use positive default oversampling, initially `p=10`, with explicit overrides
  including `p=0` for experiments. Confirm the operating values by measurement.
- Derive randomness from seed, level, side, and stable geometric box identity.
  Keep validation randomness independent of construction randomness.
- Preserve the two-sided core solve of Eq. 4.3. Use matching sketches from the
  probes actually selected for each block.
- Smooth `MockGF` and real GF subsets establish approximation quality. Random
  dense arrays may test parsing/indexing, but cannot establish compressibility.
- Include both genuine rank-truncation tests and exact-recovery/small-block
  boundary tests. Do not require every test to truncate a nonzero singular value.
- Operator-wide errors use `matvec`/`rmatvec` only. Bounded test-only `MockGF`
  blocks and real-data reference blocks may use SVD for diagnostics; do not
  introduce entry access into the compressor or materialize a full difference.
- Dense leaf blocks are extracted from an approximately peeled residual.
  They may inherit far-field approximation error; do not call them exact unless
  the preceding factors are exact in the relevant test.
- No universal monotonic-error assertions for randomized parameter sweeps.
  Use measured accuracy thresholds and repeated-seed evidence instead.

### Execution conventions

Each task is one reviewable task branch and draft PR, with scope, tests, and
acceptance below. Follow the existing implement/review workflow; humans merge.
Run all Python tooling through `.venv/`. Production changes must pass pytest,
ruff, black, and mypy. Use check-only formatting when validating a task.

Keep datasets and generated operator caches out of git. Small configuration
files, benchmark summaries, and measured result records belong in git. Tests
requiring data skip clearly when it is absent. Full runs require an explicit
slow-test or benchmark invocation, including when the data is present.

---

## Stage A: Repair the fixed-pattern foundations

**Objective:** retain the paper's algorithm while removing geometry, rank,
randomness, and sampling-interface weaknesses before real-data baselines.

### Task C.0 — Activate the plan and reconcile existing work

* **Scope:** Documentation/task routing only. Inventory merged work and the
  current R.2 branch; identify reusable completed changes and outstanding
  fixes. Preserve uncommitted work. Update project conventions and active task
  dispatch to point to this document and the C-series execution order.
* **Steps:**
  1. Record the starting revision and the disposition of existing R.2 work.
  2. Reconcile `AGENTS.md` and any other active convention file with §1,
     particularly admissibility, coverage, test requirements, and coloring ties.
  3. Update `tasks.txt` and plan lookup so `/task C.x` resolves these task blocks.
     Keep historical completion information; do not repeat completed tasks.
* **Output:** Documentation/routing changes and a task-resolution check. No
  compressor behavior changes. Explain superseded requirements explicitly.

### Task C.1 — Scale-independent dyadic geometry and neighbor lists

* **Scope:** `tree.py`, `build_tree.py`, `neighbors.py`, `interactions.py`, and
  relevant geometry/pattern tests. Implement the hypercube and integer-adjacency
  contract; reuse one source of cell coordinates across neighbors and probes.
* **Steps:**
  1. Carry integer cell coordinates through dyadic subdivision.
  2. Look up occupied neighboring cells using the `{-1,0,1}^tree_dim` offsets.
     Remove production all-pairs neighbor arrays and fixed absolute tolerance.
  3. Retain interaction construction from parent-neighbors and exact index
     coverage; skip empty cells without changing grid coordinates.
  4. Clarify `is_admissible` as a geometric diagnostic rather than an unused
     alleged runtime gate; document the supported partition.
* **Output:** Tests for translation, coordinate scaling, elongated domains,
  boundaries, sparse occupancy, and clustered meshes in 1D/2D/3D. Assert
  neighbor bounds, local cover, global block cover, separation, and period-6 /
  period-3 isolation on grids large enough to wrap the patterns.
* **Acceptance:** The same scaled geometry has the same combinatorial
  partition within the representable precision of the input. No sampling
  collisions. Explain any changed historical fixture geometry rather than
  silently replacing expected results.

### Task C.2 — Conservative PCA and complete geometry/dof separation

* **Scope:** Complete and correct the existing R.2 work in `geometry.py` and
  its consumers. Reuse existing dimension separation rather than rebuilding it.
* **Steps:**
  1. Choose a documented roundoff-based rank criterion on singular values and
     report its corresponding variance threshold; remove the inverted
     `1e-12` versus `1e-16` argument.
  2. Return original coordinates, transform information, retained/dropped axes,
     and absolute/relative projection residuals through the simplest existing
     data structures. Cover zero-spread and very small input clouds.
  3. Detect uncertain dimension decisions around the chosen cutoff, including
     the dropped side. Keep genuine dimensions by default; allow an explicit
     override with diagnostics. Reject nonfinite inputs/invalid tolerances.
  4. Audit all pattern/count/index code for tree dimension versus dof use.
* **Output:** Synthetic line, plane, full-dimensional, slightly curved,
  translated, and rescaled cases. In particular, preserve the real variation
  in the previously failing approximately `1e-16` variance-ratio example.
  Re-measure BP3/BP7 dimensions and leaf occupancy over several `m` values.
* **Acceptance:** BP3/BP7 numerical degeneracy is removed without altering
  operator components. Occupancy measurements precede performance renumbering;
  do not assume older PCA/tree measurements still apply after C.1.

### Task C.3 — Shared per-block rank and input validation

* **Scope:** Basis passes, core assembly, and `compress()` validation.
* **Steps:**
  1. Compute the same `k_eff` for a pair on both sides. Keep probe width `k+p`
     in this task; do not add adaptive sampling.
  2. Preserve compatible core dimensions and handle zero/small-rank blocks
     deliberately. Document whether `k=0` is supported; reject it early if not.
  3. Validate operator/mesh shapes and rank/oversampling/leaf parameters before
     expensive operator calls.
* **Output:** Smooth clustered tests with one-patch boxes, asymmetric block
  dimensions, exact-recovery cases, and ordinary truncation cases.
* **Acceptance:** Small boxes do not abort valid compression; malformed inputs
  fail before sampling. Existing well-sized blocks retain their formulas.

### Task C.4 — Oversampling defaults and independent random streams

* **Scope:** Random generation, fixed-pattern builders, and public defaults.
* **Steps:**
  1. Set consistent documented positive oversampling defaults.
  2. Derive reproducible streams from level, side, and cell identity, independent
     of group traversal. Keep per-box sketches usable by later coloring probes.
  3. Keep explicit seeds/configuration in benchmark outputs.
* **Output:** Tests for reproducibility, distinct levels/sides, and deterministic
  traversal-independent sketch assignment. Repeated-seed accuracy tests on
  smooth blocks, including `p=0` as a diagnostic comparison.
* **Acceptance:** Do not preserve old random values at the expense of fixing
  stream reuse. Record changed synthetic measurements before freezing new ones.

### Task C.5 — Stream sampling with explicit block-to-probe ownership

* **Scope:** Fixed-pattern builders, `column_basis.py`, `row_basis.py`, and
  `leaf.py`. Prepare the consumption contract now; implement no coloring yet.
* **Steps:**
  1. Separate lightweight probe descriptions from dense probe realization.
  2. Associate each block pair with the probe satisfying its constraints.
     Consumers must not infer ownership solely from an active source box.
  3. Process one probe at a time; extract required samples/bases immediately.
     Retain only the sample/sketch quantities needed by Eq. 4.3.
  4. Avoid retaining unused QR backing columns. Release full probes/samples
     when their assigned blocks have been processed.
* **Output:** Fixed-path equivalence tests, unchanged sampled-column counts,
  and bounded-memory evidence. Include a synthetic valid schedule in which the
  same source box appears in multiple probes serving different target blocks.
* **Acceptance:** Each pair uses its designated column and row sketches;
  later probes cannot overwrite a pair's samples. No global matrix access.

---

## Stage B: Establish useful fixed-pattern compression

**Objective:** validate numerical quality and costs, choose operating points,
then freeze full BP3/BP7 results before any graph-coloring implementation.

### Task C.6 — Discriminating synthetic accuracy and error estimation

* **Scope:** Existing synthetic tests and `error.py` where necessary.
* **Steps:**
  1. Retain multilevel rectangular coverage and leaves-only comparisons.
  2. For selected smooth admissible blocks, compare reconstruction error with
     the best rank-`k_eff` SVD error, using absolute floors near exact rank.
     Measure errors at multiple levels to expose peeling contamination.
  3. Validate power estimates with independent starts and an iteration-count
     convergence check; test the estimator against exact small reference norms.
  4. Add normalized localized and smooth inputs, plus forward/adjoint inner-
     product consistency. Keep these validation vectors independent of sketches.
  5. Replace universal monotonic sweep assertions with measured tolerance and
     robustness checks. Keep exact and genuinely truncated cases distinct.
* **Output:** Synthetic regression suite and a short accuracy table stating
  construction seeds, validation seeds, rank gaps, and the near-field baseline.
* **Acceptance:** Both 2D and 3D exercise more than one compressed level. Small
  global errors alone cannot pass the suite while dropping the far field.

### Task C.7 — Real-data reference contract and representative subsets

* **Scope:** Extend the existing R.1 reader and real-data tests as needed.
* **Steps:**
  1. Recheck headers, CSV permutations, shapes, and forward/transpose behavior.
     Add component-sensitive known-order synthetic checks and a real-data
     adjoint test; retain the existing falsifiable row-grouping evidence.
  2. Clearly separate verified component labels from inferred row layout.
     Exact physical component names need not block comparing the same dense
     reference operator, but block norms alone must not be called proof of them.
  3. Extract bounded real suboperators from several spatial regions and mesh
     resolutions, including subsets spanning separated regions. Keep explicit
     index maps; contiguous file indices are not a representativeness criterion.
  4. Use original physical coordinates for data diagnostics, reduced coordinates
     for the tree, and both row/column patch expansions for restriction.
* **Output:** Fast BP3/BP7 subset tests with real rank structure, leaf occupancy,
  block-rank diagnostics, and sensitivity to subset choice. Tests skip without
  data. Document that `L` choices do not change the current partition.
* **Acceptance:** Subsets include genuine far-field compression and multilevel
  peeling where feasible; a shallow all-dense subset is not an accuracy result.

### Task C.8 — Reproducible benchmark and cost accounting

* **Scope:** A minimal reusable benchmark entry point and result schema, shared
  by fixed and coloring runs. Keep it separate from compressor mathematics.
* **Steps:**
  1. Count forward/transpose calls and columns separately; snapshot/reset counts
     around construction so error estimation is not charged as construction.
  2. Predict counts from actual occupied groups and leaf widths, not only the
     upper bounds. Include leaf probes, oversampling, and validation work.
  3. Record numerical factor/leaf bytes, retained allocation accounting without
     double-counting views, index/tree costs, peak RSS, and disk-cache size.
  4. Time loading/conversion, geometry, sampling/peeling, local factorizations,
     total setup, and repeated forward/transpose applications where practical.
  5. Compare with a native-endian dense reference under the same hardware,
     thread settings, right-hand-side widths, and warm/cold-cache convention.
     Do not claim an algorithmic speedup from comparing against byte swapping.
* **Output:** Reproducible synthetic and subset reports plus tests for count
  accounting. Record code revision, dataset identity, environment, seeds,
  parameters, and timing protocol. Peak-RSS runs use isolated processes or an
  equivalent method that does not inherit an earlier run's high-water mark.
* **Acceptance:** Distinguish a dense-storage ratio from an RSS ratio; distinguish
  sampled-column savings from wall-clock speedup. Exclude reference-operator
  storage from compressed-representation size but include it in process RSS.

### Task C.9 — Measured fixed-path storage/application improvements

* **Scope:** Resolve material bottlenecks identified by C.8 before baseline
  freezing. No speculative abstractions or graph coloring.
* **Steps:**
  1. Measure the fraction of blocks for which `k_eff*(r+c)+k_eff**2 >= r*c`.
  2. Evaluate absorbing the core into one factor at finalization, preserving
     the represented approximation. Introduce no new approximation tolerance.
  3. Benchmark contiguous tree-order factor slices and vector permutations if
     gathers/scatters are material. Preserve external ordering and all maps.
  4. Use a streamed native-endian operator cache only if justified; a separate
     full permuted matrix per tree is not mandatory. Any cache key must include
     data identity and the effective transform/permutation, not just `m`.
* **Output:** Before/after measurements, equivalence tests for products and
  adjoints, and a documented decision on which changes are worthwhile.
* **Acceptance:** Keep the simple implementation when an optimization has no
  demonstrated benefit. Do not materialize full far-field blocks in production.
  A dense fallback for uneconomical far-field factors is deferred unless the
  project matrix-access/storage conventions are explicitly revised first.

### Task C.10 — Fixed-path parameter study and locked acceptance criteria

* **Scope:** Run `sampling="fixed"` sweeps on the C.7 subsets using C.8.
* **Steps:**
  1. Sweep a bounded set of `m`, `k`, and `p` on both datasets, over at least
     three construction seeds and independent validation starts.
  2. Compare global error, selected block errors, component-index-group errors,
     localized/smooth input errors, and leaves-only performance. Near-zero
     reference responses use absolute floors rather than unstable ratios.
  3. Choose one primary operating configuration per dataset based on accuracy,
     compressed storage, sampled columns, and apply cost. Record alternatives.
  4. Write explicit numerical tolerances and comparison slack into configuration
     files before full-baseline acceptance and before any coloring experiment.
     Derive thresholds from measured behavior and state their limitations.
* **Output:** Tracked BP3/BP7 configurations, measured sweep records, and fixed
  acceptance thresholds. Record synthetic-versus-real differences honestly.
* **Acceptance:** Do not pick a favorable seed or subset. A tolerance that permits
  the leaves-only result does not qualify without a discriminating improvement
  factor and additional block/input checks. Mark full runs opt-in regardless
  of whether they happen to be fast on the development machine.

### Task C.11 — Full BP3/BP7 fixed baselines and gate to coloring

* **Scope:** Full fixed-pattern compression and validation of both datasets.
* **Steps:**
  1. Run the locked configurations, with at least three construction seeds per
     dataset. Use independent validation vectors and norm starts from C.10.
  2. Record full accuracy, storage, peak RSS, predicted/observed sampling costs,
     setup time, and repeated apply times against dense multiplication.
  3. Record any operator preparation cost separately and as part of total
     end-to-end setup. Report setup break-even when applies are faster.
  4. Save small structured result records and a human-readable baseline report;
     large matrices/factors remain gitignored if retained locally.
* **Output:** `benchmarks/results/bp3_fixed.json`, `bp7_fixed.json`, and
  `benchmarks/RESULTS.md` (or one consistent equivalent layout established in
  C.8), linked to exact configurations and revisions.
* **Acceptance — mandatory gate:** Both full datasets run successfully; pass
  the frozen accuracy/discriminating checks; store fewer numerical entries than
  dense; and use fewer construction columns than `n_cols`, counting both
  forward and transpose columns. Actual speedup is measured, not assumed.
  These are minimum feasibility checks, not grounds for claiming large savings.
  If a gate fails, diagnose and revise the fixed configuration or implementation,
  then rerun and freeze the baseline. Do not start coloring to conceal a failing
  fixed path. All parameter changes remain visible in the experiment record.

---

## Stage C: Implement graph coloring after the fixed baselines pass

**Objective:** change sampling schedules while preserving the block partition,
rank policy, operator interface, and numerical reconstruction equations.

### Task C.12 — Canonical constraints and incompatibility graphs

* **Prerequisite:** C.11 accepted for both datasets.
* **Scope:** Sampling descriptors and a new minimal graph-construction module,
  following Eq. 4.4 and Definition 4.1.
* **Steps:**
  1. Canonicalize each constraint as a random box plus its zero-box set.
     Deduplicate identical sets while preserving all owning block pairs.
  2. Create edges exactly for random-versus-zero conflicts.
  3. Use box-to-constraint incidence lists to enumerate conflicts. Report costs
     in terms of incidences and emitted edges; avoid an all-vertices pair scan
     and do not promise a bound stronger than the generated graph supports.
  4. Establish the explicit row-side mapping by swapping the pair roles. Reuse
     a graph/coloring only after testing that mapping under the shared tree.
* **Output:** Graph tests against a brute-force oracle on small trees, including
  irregular occupancy, deduplication, and both rectangular sampling sides.
* **Acceptance:** Every required pair has an owner; no missing conflicts or
  self-edges. Deduplication need not reduce every graph. The brute-force oracle
  is test-only and never used on full datasets.

### Task C.13 — Deterministic DSatur and fixed-pattern fallback

* **Scope:** Algorithm 2.3 coloring on C.12 graphs.
* **Steps:**
  1. Implement saturation-degree selection with deterministic tie-breaking and
     correct priority updates. Use existing standard-library data structures.
  2. Validate every resulting coloring against graph edges.
  3. Construct the fixed-pattern coloring as a known feasible alternative.
     Compare actual schedule costs, not a presumed DSatur optimality guarantee.
* **Output:** Tests on empty graphs, cliques, paths, disconnected graphs, and
  generated sampling graphs; record vertices, edges, colors, and coloring time.
* **Acceptance:** Proper coloring and reproducibility are mandatory. DSatur is
  heuristic; neither chromatic optimality nor strict improvement is required.
  Retain fixed sampling when it is cheaper; prefer fixed on ties to avoid
  needless schedule changes, and record the decision.

### Task C.14 — Colored admissible probes and pair-specific sample extraction

* **Scope:** Connect C.12/C.13 to the streaming contract from C.5, for both
  column and row sampling.
* **Steps:**
  1. For each color, activate the union of its random boxes and satisfy all
     zero constraints. Reuse C.4's per-box sketch streams and width `k+p`.
  2. Map every pair to a satisfying probe; retain its actual `G_beta` and
     `G_alpha` for Eq. 4.3. A box may be active in multiple probes.
  3. Run existing basis/core/peeling computations through that schedule.
* **Output:** Isolation tests, including repeated source boxes, direct sample
  checks on `MockGF`, and multilevel fixed-versus-colored compression tests.
* **Acceptance:** Both paths meet the same accuracy criteria and the colored
  selected schedule uses no more admissible sampling columns than fixed.
  Equal seeds do not imply bitwise-equal factors: residual contamination can
  differ between schedules. Compare quality, not exact factor entries.

### Task C.15 — Colored dense-leaf extraction

* **Scope:** The §4.1.3 neighbor-only constraint graph and shared identity
  probes. Reuse graph, coloring, and ownership code where applicable.
* **Steps:**
  1. Build constraints for every neighboring leaf pair and assign a satisfying
     probe explicitly, including repeated active boxes across colors.
  2. Use shared identity columns. For the first implementation, keep the same
     global `w_max` as fixed sampling; defer per-color width optimization.
  3. Extract from the fully peeled residual and apply the fixed fallback policy.
* **Output:** Exact-factor isolation tests and approximate-factor end-to-end
  tests on uneven leaf occupancies; predicted/observed leaf-column counts.
* **Acceptance:** Every leaf pair appears once in the output, with no mixing
  from another active neighbor. Retain the same near-field accuracy criteria;
  do not assume approximate peeling makes dense leaves exact.

### Task C.16 — Public coloring option and full-path regression

* **Scope:** Wire `sampling="fixed" | "coloring"` through `compress()` and
  the benchmark interface. Default remains `fixed`.
* **Steps:**
  1. Keep geometry, ranks, defaults, and reconstruction shared between paths.
  2. Report requested strategy, selected schedule per level/side/leaf pass,
     fallback decisions, graph costs, and operator counts.
  3. Run synthetic and BP3/BP7 subset regressions with both paths, including
     permutation/adjoint/coverage checks and multiple seeds.
* **Output:** Running public API and an example invocation for both strategies.
* **Acceptance:** Both meet C.10 quality requirements on subsets; selected
  coloring schedules never exceed fixed construction-column counts. Demonstrate
  strict reduction on at least one suitable synthetic or real-subset geometry;
  do not require it on a fully occupied grid where fixed can already be optimal.

---

## Stage D: Rerun full examples and compare

**Objective:** demonstrate the final package on BP3 and BP7 and quantify what
coloring changes, using the established fixed baselines as the reference.

### Task C.17 — Paired full BP3/BP7 fixed-versus-coloring experiments

* **Scope:** Run both strategies with the C.11 datasets, transformations,
  primary configurations, seed list, validation vectors, and acceptance limits.
* **Steps:**
  1. Rerun fixed on the final code revision and compare with the frozen C.11
     results before interpreting coloring. Diagnose any unexplained drift.
  2. Run coloring for each paired configuration/seed. Alternate or otherwise
     balance run order and use the same cache/thread protocol.
  3. Report accuracy distributions, raw calls and columns by side and pass,
     graph/build overhead, total setup, numerical storage, RSS, and apply times.
  4. Compute column savings and setup/apply speed ratios; distinguish results
     before fallback from the selected schedule. Include ties and regressions.
* **Output:** Tracked paired JSON records and a BP3/BP7 comparison table in
  `benchmarks/RESULTS.md`, with commands and environment details.
* **Acceptance:** Both strategies pass the frozen scientific tolerances for
  both full datasets. Colored errors must also remain within the comparison
  slack fixed in C.10, with absolute floors for near-zero errors. Selected
  coloring uses no more construction columns than fixed. Strict savings on
  BP3/BP7 are an experimental outcome, not a fabricated guarantee; investigate
  absent savings and report any fallback honestly. Do not claim speedup unless
  total measured time, including graph construction, supports it.
* **Baseline integrity:** If correctness fixes or shared algorithm changes are
  required after C.11, retain the historical baseline and rerun both methods
  with an explicitly versioned new baseline. Do not silently retune only coloring
  or replace failing thresholds after seeing its results.

### Task C.18 — Package examples, documentation, and final acceptance

* **Scope:** README, usage examples, benchmark instructions, and final checks.
* **Steps:**
  1. Document reading each dataset, preparing its mesh, compressing with either
     strategy, applying forward/transpose products, and reproducing comparison.
  2. Publish the measured recommended configuration for each dataset, with its
     accuracy/storage/time limits and the observed benefit or cost of coloring.
  3. Explain non-uniform H¹ scope, external ordering, inferred data semantics,
     geometry limitations, dataset/cache requirements, and data-free test skips.
  4. Run the complete default suite and required lint/format/type checks. Cite
     the successful full-run records separately from default-suite results.
* **Output:** A usable package with reproducible BP3/BP7 examples and a concise
  scientific comparison, not just implementation-pipeline documentation.
* **Acceptance:** A new user with the two datasets can reproduce both modes
  from documented commands. No live Tandem dependency, committed large dataset,
  unreported fallback, or unsupported performance claim.

---

## 2. Final comparison table — required fields

For each dataset and strategy, show the primary configuration and seed range,
plus median/range where repeated measurements apply:

| Quantity | Fixed | Coloring | Comparison |
|---|---|---|---|
| Estimated global relative error | measured | measured | frozen tolerance |
| Block/input/component-group diagnostics | measured | measured | same validation |
| Improvement over leaves-only | measured | measured | frozen factor |
| Forward / transpose construction columns | measured | measured | fraction saved |
| Leaf construction columns | measured | measured | fraction saved |
| Validation columns | measured | measured | reported separately |
| Probe/color counts and fallbacks | measured | measured | by level and pass |
| Numerical representation bytes | measured | measured | ratio to dense |
| Retained memory / peak process RSS | measured | measured | same protocol |
| Operator preparation / total setup time | measured | measured | includes graph cost |
| Forward / transpose apply time | measured | measured | same dense reference |
| Graph construction / coloring time | not applicable | measured | overhead |

Raw data identity, code revision, parameters, hardware, BLAS thread settings,
cache state, construction seeds, validation seeds, and commands accompany the
table. A theoretical bound or an old estimate is never labeled as a measurement.

## 3. Task order and gates

```text
C.0  activate/reconcile
  -> C.1 geometry -> C.2 PCA -> C.3 ranks -> C.4 randomness -> C.5 sampling
  -> C.6 synthetic validation -> C.7 real subsets -> C.8 accounting
  -> C.9 measured improvements -> C.10 operating points
  -> C.11 FULL FIXED BP3/BP7 BASELINES — mandatory gate
  -> C.12 graphs -> C.13 DSatur -> C.14 admissible probes
  -> C.15 leaf probes -> C.16 public coloring/regression
  -> C.17 FULL PAIRED BP3/BP7 COMPARISON -> C.18 delivery
```

Run sequentially in this order through the existing task workflow. Do not start
production coloring work before C.11 passes. C.5 prepares a general sampling
ownership contract but does not construct or color graphs.

Historical synthetic numbers may change during foundation repairs; changes
must be explained and validated. The fixed baseline is frozen only after those
repairs and selected optimizations are complete.

## 4. Deferred work

- Live Tandem integration, time stepping, and matrix-free elasticity adapters.
- Uniform H¹, H², and conversion to those formats.
- Adaptive rank/oversampling, automatic target-tolerance compression, and
  additional approximate core truncation.
- General adaptive-depth trees or configurable admissibility that changes the
  interaction partition and requires a new sampling proof.
- Basis-support-aware clustering unless measurements establish it is necessary
  to pass the fixed baseline gate; if so, revise the geometry task explicitly.
- Production dense storage of admissible blocks, per-color leaf widths,
  serialization APIs, GPU/MPI execution, and speculative batching frameworks.

These may follow the final comparison. They are not prerequisites for proving
the requested fixed-versus-coloring package works on BP3 and BP7.
