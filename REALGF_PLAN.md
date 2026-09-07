# REALGF_PLAN — Validating the compressor against real Green's functions

Companion to `plan.md` / `FIXPLAN.md` (Stage R). Everything below marked
*(measured)* was verified against the files in `real_gfs/` on 2026-09-07; the
probe commands are reproducible from the facts given. Run these tasks through
the normal `/task R.x` pipeline, in order.

**Objective.** Compress two real elastostatic Green's function operators —
SCEC/SEAS benchmark BP3 (2D) and BP7 (3D) — and show the compressor reproduces
them to a stated tolerance using far fewer operator applications than a dense
solve, on geometry we did not choose.

---

## 1. What the data is *(measured)*

| | **BP3** | **BP7** |
|---|---|---|
| dimension | 2D elasticity | 3D elasticity |
| matrix | `28000 × 14000` | `16560 × 11040` |
| `dof_row` / `dof_col` | 2 / 1 | 3 / 2 |
| patches `N` | 14000 | 5520 |
| discretisation | 2000 elements × 7 basis fns | 552 elements × 10 basis fns × 2 slip comps |
| centroid cloud rank | **1** (σ₂/σ₁ = 2.2e-15) | **2** (σ₃/σ₁ = 2.5e-32) |
| geometry | straight line in R², dipping ~60° | planar surface in the `y = 0` plane of R³ |
| `gf_mat.bin` | 4.38 GiB | 2.04 GiB |
| dense footprint | 2.92 GiB | 1.36 GiB |

The row/column dimensions match `CLAUDE.md`'s conventions exactly — `2N×N` in
2D, `3N×2N` in 3D — which is a good independent confirmation that the shape
model is right.

### File format *(measured, reader validated end to end)*

`gf_mat.bin` is a PETSc binary `Mat`, **big-endian**, preceded by an 8-byte
application preamble:

```
offset 0   : int32 ×2   application preamble (40, ncols) / (70, ncols) -- skip
offset 8   : int32 ×4   MAT_FILE_CLASSID=1211216, rows, cols, nnz
offset 24  : int32 ×rows  row lengths      -- all == cols (matrix is fully dense)
           : int32 ×nnz   column indices   -- contiguous 0..cols-1 per row; pure
                                              redundancy, 1.46 GiB in BP3
           : float64 ×nnz values           -- row-major dense
```

Verified: `nnz == rows*cols`, every row length `== cols`, column indices
contiguous, and the predicted end offset equals the file size **exactly** for
both datasets. Values are finite and physically plausible (BP3 row 0 spans
6.8e2 down to 7.2e-11).

Consequence: **the values block can be `np.memmap`ped directly as a
`(rows, cols)` array** at `offset = 24 + rows*4 + nnz*4`. The column-index
block is skipped entirely. Nothing needs to be materialised in RAM.

Also present: `gf_vec.bin` (PETSc `Vec`, classid 1211214, length `= rows`),
`gf_facet_labels.bin` (PETSc `IS`, classid 1211218; BP3 n=4000, BP7 n=1656).
Neither is needed for compression; they are recorded here so a reader task does
not have to rediscover them.

### Coordinates

`bp{3,7}_fbf_coords.csv` gives one row per **matrix column**: `col_idx,
element, slip_comp, basis_func, coord_0, coord_1[, coord_2]`. `col_idx` is
contiguous. For BP7 the two `slip_comp` entries of a given
`(element, basis_func)` carry identical coordinates, so the 5520 distinct
`(element, basis_func)` pairs are the patch centroids. There is **no CSV for
rows** — the row layout has to be inferred (see R.1).

---

## 2. Four findings that must be dealt with before any compression runs

### Finding A — no usable rank survives the current tree *(measured; largely dissolved by Finding B's fix — re-evaluated in R.4)*

Both meshes produce **1-patch leaf boxes at every leaf size tried**:

| | m=32 | m=64 | m=128 | m=256 | m=512 |
|---|---|---|---|---|---|
| BP3 depth / min leaf occupancy | 9 / **1** | 8 / **1** | 7 / **1** | 6 / **1** | 5 / **1** |
| BP7 depth / min leaf occupancy | 5 / **1** | 4 / **1** | 4 / **1** | 3 / **1** | 3 / **1** |

By the precondition documented in Task 6.2 (`k ≤ dof_col · min_leaf_patches`),
that caps the usable rank at **k ≤ 1 for BP3** and **k ≤ 2 for BP7** — i.e.
`compress()` raises `ValueError` for any useful rank, at every `m`. Raising `m`
does not help: `m` is a *stopping* threshold, so it stops splitting a box that
is already small but does nothing to prevent a split from producing a 1-patch
child on the way down. The nodes are Gauss–Lobatto-like and cluster toward
element edges, so a cut landing near an element boundary isolates one of them.

**Severity differs sharply between the datasets** *(measured)*:

| | leaves | median occupancy | leaves ≤4 patches | **share of patches** in leaves ≤4 |
|---|---|---|---|---|
| BP3 (m=256) | 83 | 218 | 22.9% | **0.20%** (28 of 14000) |
| BP7 (m=128) | 760 | 3 | 61.3% | **15.8%** (872 of 5520) |

BP3 is a stragglers problem — the tree is otherwise healthy and the tiny boxes
hold a fifth of a percent of the data. BP7 is genuinely shattered: median
occupancy 3, with a sixth of all patches in boxes too small to compress.

**Crucially, most of this is an artifact of the tree/geometry mismatch in
Finding B, not an intrinsic property of the meshes.** Once the cluster tree is
built in the fault's own dimensionality (R.2), BP3's tiny boxes vanish entirely
and BP7's shrink by ~6x — see R.4, which re-evaluates this finding on the fixed
tree before the clamp's scope is settled. The clamp is still expected to be
needed for BP7, but as a safety net rather than the main remedy.

### Finding B — the cluster tree is built in the wrong dimensionality *(measured)*

A fault is a codimension-1 manifold: BP3's centroids are a **line** in R², BP7's
a **plane** in R³. The cluster tree is built in the ambient space and bisects
all `d` axes, so it spends a dimension on structure the data does not have.

- **BP7**: the `y` extent is ±6e-17, yet the tree bisects it and level 1 has all
  8 children occupied (108–1268 patches each) — points at the *same physical
  location* separated by floating-point noise. Cost, not incorrectness (such
  boxes are near-coincident, hence neighbours, hence stored densely), but it is
  **2× the boxes and 2× the probes per level for nothing**.
- **BP3**: *no axis is degenerate* — the line is diagonal, with `x` extent 20 and
  `z` extent 34.64. The waste appears instead as empty off-diagonal cells, and
  the surviving slivers are exactly Finding A's 19 tiny leaves.

**Correction to an earlier draft of this plan.** "Skip axes with negligible
extent" does *not* work here, and PCA rotation *alone* actively makes BP3 worse —
rotating aligns the line with an axis and thereby manufactures a degenerate
second axis of ~1e-13 noise, which the tree then shatters on: **83 leaves → 2075,
median occupancy 218 → 5** *(measured)*. Rotation is only correct when paired
with dropping the degenerate directions.

**Decision: PCA-align the centroids and drop principal axes whose explained
variance falls below a threshold**, building the tree in the retained
dimensions. Measured effect:

| | tree dim | leaves | min occupancy | median | patches in leaves ≤4 |
|---|---|---|---|---|---|
| BP3 as-is (m=256) | 2 | 83 | 1 | 218 | 0.20% |
| **BP3 PCA-dropped** | **1** | **64** | **218** | **219** | **0.00%** |
| BP7 as-is (m=128) | 3 | 760 | 1 | 3 | 15.8% |
| **BP7 PCA-dropped** | **2** | **256** | **1** | **12** | **2.68%** |

BP3 becomes *perfectly balanced* (min 218, max 220 at depth 6) — Finding A
disappears for it completely. BP7 improves ~6x but retains 1-patch boxes, so the
clamp is still needed there.

#### Choosing the threshold

Explained-variance ratios `σᵢ²/Σσⱼ²` *(measured)*:

| | axis 0 | axis 1 | axis 2 |
|---|---|---|---|
| BP3 | 1.000 | **4.7e-30** | — |
| BP7 | 0.512 | 0.488 | **3.2e-64** |

The smallest *retained* ratio is 0.488 and the largest *dropped* is 4.7e-30 —
a gap of ~30 orders of magnitude, so no threshold in that range is
distinguishable on this data. The default must therefore be argued from
principle, not fitted:

- **Noise floor.** Float64 roundoff on O(1)–O(10) coordinates gives a relative
  RMS deviation ~1e-15 to 1e-16, i.e. a variance ratio ~1e-30 to 1e-32. BP3's
  4.7e-30 is exactly this.
- **Real-geometry floor.** Any genuine feature a mesh generator writes — fault
  roughness, listric curvature, a slight non-planarity — is at least ~1e-8 in
  relative RMS terms, i.e. a variance ratio ≥1e-16, and usually far larger.
- The two failure modes are **not symmetric**: keeping an axis you should have
  dropped costs a factor of two in boxes and probes, while dropping one you
  should have kept silently corrupts the geometry — points that are genuinely
  separated become coincident. Bias conservative.

**Recommended default: `pca_var_tol = 1e-12`** on the explained-variance ratio.
That is ~18 orders of magnitude above the float64 noise floor and ~4 below the
most pessimistic real feature, equivalently: drop an axis only when its RMS
extent is below one part in 10⁶ of the dominant axis. It cleanly drops both
datasets' degenerate directions with vast margin on either side.

Two safeguards to implement with it:
1. **Ambiguity band.** If any ratio lands in `[1e-12, 1e-6]` the mesh is
   *nearly* degenerate and the right answer is not obvious — warn loudly (or
   require an explicit opt-in) rather than deciding silently.
2. **Never reduce below 1 dimension**, and always report which axes were
   dropped with their ratios. Silent dimension reduction is exactly the kind of
   thing that produces a plausible-looking wrong answer.

#### A blocker inside this fix

`FaultMesh` rejects a 1-D tree outright: `centroids must have shape (N, 2) or
(N, 3)` *(measured)*. Tree dimensionality is currently hard-coupled to the
elasticity dimensionality, but they are different things — BP3 is 2D elasticity
(`dof_row=2, dof_col=1`) on a 1D fault. **R.2 must decouple them**: the tree
dimension becomes a property of the geometry, while `dof_row`/`dof_col` stay
tied to the elastic problem. This is the largest single piece of work in the
plan and the reason R.2 comes before the clamp.

### Finding C — the file dof layout is not patch-major *(measured)*

`CLAUDE.md` requires patch-major flattening: a patch's `dof_col` columns (and
`dof_row` rows) must be **consecutive**. Both files instead store dofs blocked
by component within each element:

```
file column = element·(comps·nbf) + comp·nbf + basis_func
required    = element·(comps·nbf) + basis_func·comps + comp
```

- **BP7 columns** — confirmed directly from the CSV: `slip_comp` changes every
  10 rows, `basis_func` every 1, so a node's two slip components sit 10 columns
  apart. A permutation is mandatory.
- **BP3 columns** — `dof_col = 1`, so trivially already patch-major.
- **Rows, both datasets** — evidence points to the same blocked layout
  (`element·14 + comp·7 + bf` for BP3, `element·30 + comp·10 + bf` for BP7).
  Probing `argmax_row |A[:, j]|` lands exactly on the blocked prediction at the
  domain ends (BP3 col 0 → row 7; col 13999 → row 27999; BP7 col 0 → row 10)
  and within 2–3 rows in the interior, where the kernel peak legitimately sits
  on an adjacent basis function rather than the self-node. **This is strong but
  not conclusive** and must be established rigorously in R.1 — a wrong row
  permutation silently destroys the block structure the compressor depends on,
  and would show up only as poor compression, not as an error.

### Finding D — patch numbering is not geometrically coherent *(measured)*

Independently of the dof layout in Finding C, the *patch* ordering in these files
does not track position on the fault. Element numbering comes from the mesh
generator and jumps around:

- **BP3**: correlation between column index and arc-length along the fault is
  **0.14** — essentially uncorrelated. 993 of 13999 consecutive index pairs jump
  more than 1% of the fault's extent, and the largest single jump is **30.7 of a
  40-unit fault** (77%).
- **BP7**: 116 of 5519 consecutive pairs jump more than 10% of the in-plane
  extent; largest jump 0.67 of a 1.13 extent (60%).

The compressor is *correct* under any numbering — it works with index sets, not
index ranges — so this is purely a cost problem. But it is a large one. Because
a tree box collects geometrically nearby patches, and nearby patches have
scattered indices, every box's index set is badly fragmented *(measured, at the
leaf level)*:

| | leaves | contiguous runs per leaf (median) | mean run length | vs. ideal |
|---|---|---|---|---|
| BP3 (m=256) | 83 | **19** | 10.8 patches | 168.7 → **16x fragmentation** |
| BP7 (m=128, degen axis dropped) | 256 | **8** | 2.2 patches | 21.6 → **10x fragmentation** |

Every `Y[alpha.row_indices, :]` gather and every scatter into
`Omega[beta.col_indices, :]` therefore touches ~10–16x more discontiguous memory
segments than necessary. This falls hardest on `peeling.apply_truncated`, which
Task 6.1's review measured as *the* dominant cost (~28 s of the 29 s 3D
integration test) and which performs exactly these gathers once per stored
factor per probe.

The standard remedy, and the one this plan adopts, is to **renumber patches into
cluster-tree order** (depth-first over the leaves) so that every box's patch set
is a contiguous range by construction — one run per box instead of 8–19. This is
routine in H-matrix libraries for precisely this reason. It is tree-dependent, so
it must happen after R.2 and before the re-evaluation in R.4.

---

## 3. Cost model *(measured)*

Cold-cache `memmap` matvec: **~5.0 s** (BP3), **~2.2 s** (BP7) per single
right-hand side. Once converted to native-endian and resident, expect memory-
bandwidth-bound behaviour (~0.3 s and ~0.15 s respectively).

The fixed-pattern path issues up to `2·6^dₜ` products of width `k+p` per level,
plus `3^dₜ` leaf probes of width `w_max` — where **`dₜ` is the tree dimension,
not the elasticity dimension** (see R.2). That distinction dominates the cost
here, because R.2 reduces `dₜ` on both datasets:

| | `dₜ` before | `dₜ` after | probes/level (`2·6^dₜ`) | leaf probes (`3^dₜ`) |
|---|---|---|---|---|
| BP3 | 2 | **1** | 72 → **12** | 9 → **3** |
| BP7 | 3 | **2** | 432 → **72** | 27 → **9** |

So an earlier estimate in this plan of "~360 wide products, order 10–30 minutes"
for BP3 was **~6x too pessimistic**: at `dₜ = 1` and depth 6 it is ~60 wide
products, plausibly **2–5 minutes**. BP7 sees a 6x reduction on the same grounds.
`apply_truncated` remains the term to watch — 6.1's review measured it as the
dominant cost at far smaller sizes — and R.3's renumbering targets exactly that.

**Because these estimates now straddle the "can it live in the default suite?"
line, do not fix the fast/slow split until R.7 has measured it.** The plan still
assumes small subsets in the default suite and full-scale runs opt-in
(`-m realgf_slow`, deselected by default), but if a full BP3 run really is ~2
minutes, promoting it is worth reconsidering.

**Storage budget.** The values block is memory-mapped, so a run's resident set is
driven by the probe blocks, not the operator. But R.3 writes a permuted,
native-endian copy per configuration: **2.92 GiB (BP3) + 1.36 GiB (BP7) ≈ 4.3
GiB** on top of the 6.5 GiB of source data, and one copy per distinct
`(m, pca)` key. Cache these under a gitignored scratch directory, make the key
explicit in the filename, and give R.7/R.8 a way to reuse rather than regenerate
them.

---

## 4. Task breakdown

### Task R.1 — PETSc/CSV reader and `RealGF` operator
* **Scope:** `src/gfcompress/realgf.py`. Parse the PETSc `Mat` header (skipping
  the 8-byte preamble), validate `nnz == rows*cols` and contiguous column
  indices on a sample, and `memmap` the values block. Load the CSV into patch
  centroids plus the row/column permutations to patch-major order. Expose a
  `MatVecOperator` (`matvec`/`rmatvec`) and a `FaultMesh`. Never materialise the
  dense array; never assemble inside the compressor.
* **Row layout must be established, not assumed.** Test the blocked hypothesis
  against a falsifiable prediction — e.g. that under the correct permutation
  `|A|` decays monotonically with centroid separation, and that the near-diagonal
  block is dominant, while a wrong permutation scrambles both. State the
  evidence in the module docstring.
* **`L` (patch length) is a deliberate decision with consequences, not a
  detail.** `FaultMesh` needs a per-patch length, and it feeds `diam`, which
  feeds the admissibility test `dist ≥ η·max(diam α, diam β)` — so the choice
  silently reclassifies blocks between near and far. Naive nearest-neighbour
  spacing is a poor default here: GLL nodes cluster hard at element edges, so
  spacing varies by an order of magnitude *within a single element*, which would
  hand wildly different lengths to physically equivalent patches. Candidates:
  the representative length `element_size / nbf`, the dual/Voronoi length of the
  node, or the element size itself for every node it carries. **Pick one, state
  why, and test sensitivity**: report how the admissible/inadmissible split
  changes across the candidates on both datasets. If the split is materially
  sensitive, that is itself a finding worth recording.
* **Resolving Finding C is a gate, not a detail.** The reader must emit the
  operator in patch-major order — a patch's `dof_col` columns and `dof_row` rows
  consecutive — via the permutation
  `element·(comps·nbf) + comp·nbf + bf  →  element·(comps·nbf) + bf·comps + comp`,
  applied to columns (BP7; BP3 is trivially fine at `dof_col=1`) and to rows
  (both datasets). Nothing downstream may consume the operator until this is
  verified, because every shape convention in `CLAUDE.md` assumes it and a wrong
  permutation produces a plausible-looking but meaningless compression.
* **Output:** reader + operator; tests on both datasets asserting shapes
  (`28000×14000`, `16560×11040`), header/offset arithmetic against actual file
  size, permutation round-trip, and the row-layout evidence. Include a
  **falsifiable check of the patch-major result** — e.g. that the near-diagonal
  block is dominant and `|A|` decays with centroid separation under the applied
  permutation, and visibly does not under a deliberately wrong one. Mark
  file-touching tests so they skip cleanly when `real_gfs/` is absent.

### Task R.2 — PCA alignment, degenerate-axis dropping, tree/dof decoupling (`src/`)
* **Scope:** Resolve Finding B. Three parts:
  1. **Decouple tree dimensionality from elasticity dimensionality.** `FaultMesh`
     must accept centroids of tree-dimension 1, 2 or 3 while `dof_row`/`dof_col`
     stay determined by the elastic problem (`d`, `d−1`).

     **The blast radius is larger than it looks, and this is the main risk in the
     plan.** `grid_coordinates` derives from `node.bounding_box`, which has one
     row per *geometric* axis — so `pattern_cell`, the period-6 tiling, the
     neighbour combinatorics and the `6^d` / `3^d` probe-count bounds all follow
     the **tree** dimension. They read `mesh.d` today only because the two
     dimensions currently coincide. Every one of those sites must be audited and
     switched to the tree dimension: `fixed_pattern` (`PERIOD`/`LEAF_PERIOD`
     tiling, `6**d`/`3**d` assertions), `neighbors`, `interactions`, and the
     count assertions in `tests/test_compress.py` / `test_regression.py`.
     Stage 7 is about to build on this same machinery, so getting the
     tree-vs-elasticity distinction named explicitly (not left implicit in a
     shared `d`) matters beyond this plan.
  1b. **Admissibility constants follow the tree dimension too.** `DEFAULT_ETA =
     0.5` is documented as safe because `0.5 ≤ 1/√d`; that bound is `1/√1 = 1.0`
     for a 1-D tree and `1/√2` for a 2-D one, so `η = 0.5` remains valid, but the
     justification must be restated against the tree dimension rather than left
     pointing at the elastic one. Re-run Task 6.2's η consistency test on the
     reduced-dimension trees.
  2. **PCA alignment.** Centre the centroids, rotate into principal axes.
  3. **Drop degenerate directions**: discard principal axes whose explained
     variance ratio `σᵢ²/Σσⱼ²` falls below `pca_var_tol`, **default `1e-12`**
     (justified in Finding B). Warn loudly on any ratio in the ambiguity band
     `[1e-12, 1e-6]`; never reduce below one dimension; always report which
     axes were dropped and their ratios.
* **Must not perturb existing fixtures.** On a non-degenerate mesh the tree must
  be identical to today's — the existing Stage 6 regression baselines are the
  guard, and none of them may move.
* **Output:** the change plus tests on *synthetic* meshes (cheap, no large
  files): a line-in-2D and a plane-in-3D, asserting the retained dimension, the
  dropped axes' ratios, the improvement in leaf occupancy, and that a genuinely
  3-D cloud keeps all three axes. Test the ambiguity band explicitly with a mesh
  built to land in it.

### Task R.3 — Renumber patches into cluster-tree order
* **Scope:** Resolve Finding D. After the tree is built (R.2), compute the
  depth-first leaf traversal order and renumber patches so that **every box's
  patch set is a contiguous range**. Expose the permutation and its inverse so
  results can be mapped back to the caller's original numbering — the compressed
  operator must still behave, from outside, as the operator the user handed in.
* **Where the win is:** `alpha.row_indices` / `beta.col_indices` become slices
  rather than fancy-index gathers, dense leaf blocks become contiguous
  sub-arrays, and `apply_truncated` — the measured dominant cost — stops paying
  a 10–16x fragmentation penalty on every factor of every probe.
* **Watch for:** applying the permutation lazily inside `matvec` would re-scatter
  on every call and give back the whole benefit. For the real datasets, write the
  permuted operator once (native-endian, tree order) and memory-map that. The
  permutation depends on the tree, hence on `m` and the PCA result, so it must be
  keyed to those.
* **Output:** the renumbering plus tests asserting one contiguous run per box on
  both a synthetic mesh and (skip-if-absent) the real ones; the permutation
  round-trips to identity; and compression results are **numerically unchanged**
  versus the un-renumbered path on an existing fixture — this is a pure
  performance change and must not move any Stage 6 baseline. Record the measured
  speed-up.

### Task R.4 — Re-evaluate Finding A on the fixed tree
* **Scope:** With R.2 merged, re-measure the leaf-occupancy statistics that
  motivated the clamp, on both real datasets across a range of `m`: tree depth,
  leaf count, min/median occupancy, and the **share of patches** sitting in
  leaves too small for the target rank. Compare against the pre-R.2 baselines
  recorded in Finding A.
* **Why it is its own step:** the predicted outcome is that BP3 no longer needs
  the clamp at all (min occupancy 218 vs 1) while BP7 still does, but with ~6x
  less exposure (2.68% of patches vs 15.8%). If that holds, R.5's scope shrinks
  from "the thing that unblocks everything" to "a safety net for a small
  fraction of BP7's blocks", which changes how much accuracy risk the clamp is
  allowed to carry. **Settle this with measurements before designing R.5.**
* **Output:** a short measured comparison table appended to this document (or to
  `real_gfs/RESULTS.md`), and an explicit recommendation for R.5's design. No
  production code. If the prediction fails — if BP3 still has tiny boxes, or
  BP7's exposure does not fall — say so and revise the plan rather than
  proceeding.

### Task R.5 — Per-block rank clamping, scoped by R.4
* **Scope:** Resolve whatever of Finding A survives R.2. Either clamp
  `k_eff = min(k, min(block dims))` at the `orth` call sites in
  `column_bases`/`row_bases`, or store admissible blocks smaller than the target
  rank densely. Pick one **on the evidence from R.4**, justify it against
  Algorithm 4.1, and say what it means for the Eq. 4.3 core solve when the two
  sides clamp to different ranks.
* **Output:** the fix, plus tests on a synthetic clustered mesh proving
  `compress()` runs where it previously raised, and that accuracy is unchanged
  on the existing uniform fixtures. Stage 6's regression baselines must not move.

### Task R.6 — Subset extraction and fast real-data tests
* **Scope:** A helper that restricts a `RealGF` to a contiguous patch subset —
  slicing rows and columns by the patch-major expansion of the subset — yielding
  a genuine sub-operator of the real GF with real rank structure, small enough
  for the default suite. Target a few hundred patches, runtime seconds.
* **Output:** subsetting helper + default-suite tests running `compress()` on
  BP3 and BP7 subsets, asserting `‖A_H − A‖/‖A‖` below a tolerance chosen from
  measurement, with a **discriminating guard** (per Stage 6: a leaves-only
  baseline the real result must beat by a stated factor — an absolute threshold
  alone certifies nothing on a diagonally dominant kernel).

### Task R.7 — Parameter behaviour on real geometry (choose the operating point)
* **Scope:** The Stage 6.2 sweeps (`k`, `p`, `m`) re-run on a BP3 and a BP7
  subset. Two questions: whether the accuracy/cost trends established on smooth
  synthetic grids survive on real, non-uniform, reduced-dimension geometry — in
  particular whether the rank needed for a given accuracy is materially higher —
  and **what `(m, k, p)` the full-scale runs should actually use**.
* **Why this comes before the full run:** R.8 commits minutes of compute per
  configuration. Choosing its parameters blind and discovering afterwards what
  they should have been wastes the expensive step and produces a headline number
  that does not represent the method at its sensible operating point.
* **Output:** regression fixtures with measured trends, an explicit written
  comparison against the synthetic baselines from 6.2, and a **recommended
  `(m, k, p)` per dataset, with the evidence** for R.8 to adopt. Also settle the
  fast/slow split question raised in the cost model, now that per-run timings are
  known.

### Task R.8 — Full-scale runs
* **Scope:** Full BP3 and BP7 compression at the operating point R.7
  recommends. Record, as fixtures: relative error, compression ratio,
  matvec/rmatvec column counts via `CountingOperator`, peak RSS, and wall-clock.
  Keep the deselected-by-default marker unless R.7's timings show the run is
  cheap enough to promote into the default suite.
* **Output:** tests plus a short `real_gfs/RESULTS.md` recording the numbers, the
  operating point, and the hardware they were measured on. This is the
  deliverable that answers "does it work on real data".

---

## 5. Acceptance

"Fewer operator columns than the matrix has columns" (BP3 < 14000, BP7 < 11040)
is necessary but far too weak to be the bar: at the post-R.2 tree dimensions the
fixed-pattern path clears it by two orders of magnitude while saying nothing
about whether the result is *useful*. All four of the following must be recorded
in `real_gfs/RESULTS.md`, at the operating point R.7 recommends, and reproducible
from the tests:

1. **Accuracy** — `‖A_H − A‖/‖A‖` below a stated tolerance, measured through the
   power-method utility (matvec-only), with the discriminating guard R.6
   establishes. An absolute threshold alone certifies nothing on a diagonally
   dominant kernel; this was true on synthetic fixtures and is more true here.
2. **Sampling cost** — total matvec/rmatvec **columns** via `CountingOperator`,
   against the `2·6^dₜ`-per-level prediction, and as a fraction of `n_cols`.
3. **Memory** — compressed storage (factors + dense leaves) as a ratio of the
   dense footprint (2.92 GiB / 1.36 GiB). A representation that is accurate and
   cheap to build but no smaller than the matrix has not achieved anything.
4. **Apply cost** — time for `H.dot(x)` against a dense `A @ x` on the same
   hardware. This is what a downstream solver actually pays.

If any of these turns out materially worse than the synthetic fixtures predict,
**that is a result, not a failure**: record it with the measured numbers and a
diagnosis rather than tuning it away. A negative result on real geometry, clearly
attributed, is more valuable than a flattering one on a fixture chosen to pass.

---

## 6. Out of scope

- Stage 7 (graph coloring). This plan runs entirely on `sampling="fixed"`. Once
  Stage 7 lands, R.8's recorded counts become its comparison baseline.
- Uniform H¹ (§4.2) and H² (§4.3) remain out of scope per `CLAUDE.md`.
- `gf_vec.bin` and `gf_facet_labels.bin` are documented but unused.
- Committing the `real_gfs/` data. It is 6.5 GB; it must stay out of git, and
  every test touching it skips cleanly when absent.

---

## 7. Ordering and dependencies

```
R.1 (reader, patch-major)  ──────────────────────────────────┐
                                                             ├─> R.6 (subset + fast tests) ─> R.7 (sweeps, pick operating point) ─> R.8 (full scale)
R.2 (PCA/tree) ─> R.3 (tree-order) ─> R.4 (re-evaluate) ─> R.5 (clamp) ─┘
```

**R.2 → R.3 → R.4 → R.5 is a strict chain, and the ordering is the point.**

- **R.2** is the structural fix — the tree gets built in the fault's own
  dimensionality.
- **R.3** renumbers patches into tree order. It must come *after* R.2, because
  the ordering is derived from the tree, and *before* R.4, because it changes
  nothing numerically but everything about cost — so the re-evaluation measures
  the configuration we will actually run.
- **R.4** measures what Finding A still looks like on the fixed tree.
- **R.5** is designed against that measurement rather than against the pre-R.2
  numbers. Doing the clamp first would size it for a problem R.2 largely removes
  — on BP3 entirely — and would bake unnecessary accuracy risk into every block.

**R.1 is independent of that chain and can run in parallel**, with one hard
requirement: it must resolve Finding C (patch-major dof layout) before anything
consumes the operator. Finding C is a *data-format* permutation, internal to a
patch; Finding D (R.3) is a *patch-ordering* permutation across the mesh. They
compose and are separate concerns — do not merge them into one step.

R.2 and R.3 are testable on synthetic meshes alone and need no large files.

**R.6 cannot start until R.1 and R.5 are merged.** Without R.5 `compress()` may
still raise on BP7's remaining 1-patch boxes, and without R.2 the recorded box
counts — and therefore R.8's baselines — would be inflated by a factor of two per
degenerate axis.

---

## Appendix — reproducing the measurements

Every *(measured)* claim above came from short probes against `real_gfs/`. Run
them with `.venv/bin/python3`. Shared preamble:

```python
import numpy as np, csv
from gfcompress.geometry import FaultMesh
from gfcompress.build_tree import build_tree

def load_csv(path):
    with open(path) as fh:
        r = csv.reader(fh); next(r)
        return np.array([row for row in r], dtype=float)

# BP3: dof_col=1, so every column is a patch. BP7: keep slip_comp==0 rows.
bp3 = load_csv("real_gfs/gf_bp3/bp3_fbf_coords.csv")[:, 4:6]
a7  = load_csv("real_gfs/gf_bp7/bp7_fbf_coords.csv")
bp7 = a7[a7[:, 2] == 0][:, 4:7]
```

**§1 file format / header offsets.** Big-endian int32; 8-byte preamble, then
`classid, rows, cols, nnz`; values at `24 + rows*4 + nnz*4`. `predicted_end`
must equal the file size exactly.

```python
h = np.fromfile(path, dtype=">i4", count=6)
rows, cols, nnz = int(h[3]), int(h[4]), int(h[5])
val_off = 24 + rows*4 + nnz*4
A = np.memmap(path, dtype=">f8", mode="r", offset=val_off, shape=(rows, cols))
```

**§2 Finding A — leaf occupancy.** Build the tree and read the deepest level:

```python
mesh = FaultMesh(centroids=c, L=np.full(len(c), 1e-3))
root = build_tree(mesh, m=m)
leaves = [lvl for lvl in root.iter_levels()][-1]
occ = np.array([len(b.patch_indices) for b in leaves])
```

**§2 Finding B — degeneracy and the PCA threshold.** Centroid-cloud spectrum and
explained-variance ratios; `ratio` is what `pca_var_tol` is compared against:

```python
v = c - c.mean(0)
s = np.linalg.svd(v, compute_uv=False)
ratio = s**2 / (s**2).sum()
```
Re-run the occupancy probe on `c[:, [0, 2]]` (BP7, degenerate axis dropped) and
on `(v @ vt.T)[:, :1]` (BP3, 1-D) to reproduce the improvement table. Note the
1-D case currently raises in `FaultMesh` — that is the blocker R.2 removes; the
BP3 1-D numbers in this document were obtained by dyadic binning of the rotated
first coordinate, which is what a 1-D tree would produce.

**§2 Finding C — row layout.** For a column `j`, `argmax_row |A[:, j]|` should
land on the blocked prediction (`element·(comps·nbf) + comp·nbf + bf`), exactly
at the domain ends and within a few rows in the interior.

**§2 Finding D — fragmentation.** Contiguous runs per leaf box:

```python
runs = [1 + int((np.diff(np.sort(b.patch_indices)) != 1).sum()) for b in leaves]
```

**§Task R.3 — tree invariance under renumbering.** Build the tree, take the
depth-first leaf order as a permutation, rebuild on the permuted mesh, and
confirm box membership is unchanged when mapped back through the permutation.
Verified: identical depth, identical boxes per level, identical membership, and
one contiguous run per leaf afterwards.
