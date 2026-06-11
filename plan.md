# Implementation Plan: Randomized GF Matrix Compression (2D & 3D)

## Overview
This package implements the black-box randomized compression algorithm of
Levitt & Martinsson (2024), *"Randomized compression of rank-structured matrices
accelerated with graph coloring"*, specialized to elastostatic Green's Function
(GF) matrices over fault meshes.

The matrix `A` is accessed **only** through matrix–vector products `Y = AΩ` and
`Z = A*Ψ`. From a small number of such products we reconstruct a hierarchical
(H-matrix) representation: dense near-field (inadmissible) leaf blocks plus
low-rank far-field (admissible) blocks.

### Scope and key design decisions
- **Format:** non-uniform **H¹** only. Each admissible block `(α,β)` gets its
  own factors `A(I_α, I_β) ≈ U_{α,β} B_{α,β} V_{α,β}*` (paper §4.1, Algorithm
  2.1 / 4.1). Uniform H¹ (§4.2) and H² (§4.3) are explicitly out of scope.
- **Admissibility:** **geometric strong admissibility**, paper-faithful.
  A block is admissible iff its boxes are in each other's interaction lists,
  equivalently `dist(α,β) ≥ η · max(diam α, diam β)`. The `1/r^d` physics decay
  is used *only* to choose/sanity-check the separation parameter `η`, never as a
  standalone block-norm threshold (a free threshold would break the level-nested
  structure that peeling depends on).
- **Block-matrix view over a single `N`-patch tree.** We model `A` as an `N×N`
  *block* matrix whose logical entry `GF[i,j]` is a small fixed tensor of shape
  `dof_row × dof_col` (`2×1` in 2D, `3×2` in 3D; `dof_row = d`, `dof_col = d-1`).
  This makes the operator "square at the patch/box level," so a **single
  geometric cluster tree over the `N` patch centroids** plays the role of both
  the row tree and the column tree — exactly as in the scalar paper. All
  combinatorics (neighbor lists, interaction lists, admissibility, the
  incompatibility graph, coloring, level peeling) operate on the `N` boxes and
  are *unchanged* by the `dof` multiplicity.
- **Numerics are rectangular via patch-major flattening.** The `dof`
  multiplicity appears only inside numerical blocks. Each box expands to scalar
  indices (`dof_row` consecutive rows / `dof_col` consecutive cols per patch,
  block-interleaved), so a far-field block is `(dof_row·|α|) × (dof_col·|β|)`,
  and the flattened operator maps `ℝ^{dof_col·N} → ℝ^{dof_row·N}` (i.e. `2N×N` /
  `3N×2N`). `A` and `A*` therefore have different domain/range sizes, which the
  separate row/column sampling already respects.
- **Build order de-risks the hardest piece.** The paper's §4.1.4 fixed periodic
  test-matrix patterns (≤ 6^d admissible, ≤ 3^d leaf) yield a *fully correct*
  algorithm with **no graph at all**. We implement that path first and validate
  end-to-end (Stages 0–6). Graph coloring (§4.1.2) is then added as a drop-in
  optimization with a clean regression target: identical accuracy, fewer matvecs
  (Stage 7).

Each task below is sized to a single PR: small, self-contained, independently
testable. Tasks within a stage are ordered by dependency.

---

## Stage 0: Project Setup
**Objective:** Installable package with tooling and CI.

* **Task 0.1: Scaffolding and tooling.**
    * **Scope:** `pyproject.toml` (Poetry or Flit). Configure `pytest`, `black`,
      `ruff`, `mypy`. Minimal package skeleton and a GitHub Actions (or equiv.)
      workflow running lint + tests.
    * **Output:** Empty installable package, green CI on an empty test.

---

## Stage 1: Geometry & Dual-Index Cluster Tree
**Objective:** Represent the mesh and build one geometric tree carrying row and
column index sets, plus the neighbor/interaction lists that define admissibility.

* **Task 1.1: `FaultMesh` / `Patch` data structures.**
    * **Scope:** Hold patch centroids (`(N,2)` or `(N,3)`), characteristic
      length `L` per patch, and dimension `d`. Derive `dof_row = d`,
      `dof_col = d-1`. Provide helpers mapping a set of patch indices to its row
      index set (`dof_row` consecutive rows per patch) and col index set
      (`dof_col` per patch). Vectorized pairwise centroid distances.
    * **Steps:**
        1. `Patch` / `FaultMesh` dataclasses (centroids, `L`, `d`, derived
           `dof_row`/`dof_col`).
        2. `patch_to_rows(patch_ids)` and `patch_to_cols(patch_ids)` index
           expansion helpers.
        3. `pairwise_distances(centroids)` vectorized helper.
    * **Output:** Mesh/patch classes; unit tests for 2D and 3D index mapping
      (sizes `2N×N`, `3N×2N`) and distance correctness.

* **Task 1.2: `TreeNode` class.**
    * **Scope:** Fields: `patch_indices`, `row_indices`, `col_indices`,
      `children`, `parent`, `level`, axis-aligned `bounding_box`, `center`,
      `diam`. Recursive traversal helpers (leaves, levels, by-level iteration).
    * **Steps:**
        1. `TreeNode` dataclass: fields + parent/child links.
        2. Geometry computation from a patch set: `bounding_box`, `center`,
           `diam`.
        3. Traversal helpers: `leaves()`, `nodes_at_level(l)`, by-level iterator.
    * **Output:** `TreeNode` with traversal utilities and tests.

* **Task 1.3: Geometric bisection tree builder.**
    * **Scope:** Recursively split a box by bisecting along the longest axis into
      `2^d` children (uniform-style partition over the centroids); stop when a
      box has fewer than `m` patches (`m` a parameter; tree depth `L ~ log N`).
      Populate each node's row/col index sets from its patch set.
    * **Steps:**
        1. Single-box split: bisect the longest axis into ≤ `2^d` child patch
           partitions.
        2. Recursive builder with the leaf stop condition (`< m` patches).
        3. Populate each node's `row_indices`/`col_indices` via the Task 1.1
           helpers.
    * **Output:** Builder returning the root. Tests: leaf row-index sets partition
      `{0..2N-1}` (2D) / `{0..3N-1}` (3D) exactly (disjoint + complete); same for
      cols; verified for both dimensions.

* **Task 1.4: Neighbor lists `L^nei`.**
    * **Scope:** For each box, compute same-level boxes whose bounding boxes touch
      or overlap (includes itself); ≤ 3^d entries on a regular grid.
    * **Steps:**
        1. Box-adjacency predicate (two bounding boxes touch or overlap).
        2. Per-level neighbor-list map built from the predicate.
    * **Output:** Neighbor-list map. Tests on a uniform grid checking expected
      counts (3 in 1D, 9 in 2D, 27 in 3D interior boxes).

* **Task 1.5: Interaction lists `L^int` and admissibility predicate.**
    * **Scope:** `L^int(α)` = children of α's parent's neighbors, excluding α's
      own neighbors (≤ 6^d − 3^d). Admissibility predicate
      `dist(α,β) ≥ η · max(diam α, diam β)`; expose a helper that uses the
      `1/(r+γL)^d` decay only to suggest a default `η`.
    * **Steps:**
        1. `interaction_list(α)`: children of α's parent's neighbors, minus α's
           own neighbors.
        2. `is_admissible(α, β)`: the `dist ≥ η · max(diam)` predicate.
        3. `suggest_eta(...)`: default `η` from the `1/(r+γL)^d` decay.
    * **Output:** Interaction-list map + predicate. Tests: at each level the
      admissible (interaction-list) and inadmissible (neighbor) blocks form a
      complete, disjoint cover; reproduce the small Fig. 3 tessellation.

---

## Stage 2: Reference Kernel & Black-Box Interface
**Objective:** A matvec interface and a *physically meaningful* mock with genuine
rank structure to test against. (A random dense matrix has full-rank blocks and
cannot validate a compressor — the mock must come from a smooth kernel.)

* **Task 2.1: `MatVecOperator` abstract base class.**
    * **Scope:** Abstract `matvec(Ω) -> AΩ`, `rmatvec(Ψ) -> A*Ψ`, and `shape`
      (rectangular `2N×N` / `3N×2N`).
    * **Steps:**
        1. `MatVecOperator` ABC declaring `matvec`, `rmatvec`, `shape`.
        2. `DenseOperator` concrete subclass wrapping an explicit array (test
           double).
    * **Output:** ABC with docstring contract and a trivial dense-backed test
      implementation.

* **Task 2.2: Analytic-kernel mock GF operator.**
    * **Scope:** Implement a smooth tensor-valued kernel `K(x_i, x_j)` returning a
      `dof_row × dof_col` block per patch pair (a simplified elastostatic /
      Kelvin-type kernel decaying like `1/r^d`). Assemble the dense rectangular
      `A`; implement `matvec`/`rmatvec`. Expose a direct `block(I_rows, J_cols)`
      accessor for ground-truth tests.
    * **Steps:**
        1. Tensor kernel `K(x_i, x_j) -> (dof_row, dof_col)` block.
        2. Dense assembler building the flattened patch-major `A`.
        3. `matvec` / `rmatvec` on the assembled operator.
        4. `block(I_rows, J_cols)` direct accessor for ground-truth tests.
    * **Output:** `MockGF` operator. Tests: off-diagonal admissible blocks are
      numerically low rank (singular values decay); near-diagonal blocks are not.

* **Task 2.3: Relative-error utility (power method).**
    * **Scope:** Estimate `‖A_approx − A‖ / ‖A‖` using ~20 power-method iterations
      that touch both operators only through `matvec`/`rmatvec`.
    * **Steps:**
        1. Difference operator `(A_approx − A)` exposed via the two operators'
           `matvec`/`rmatvec`.
        2. Power-method spectral-norm estimate (~20 iters) and the relative-error
           ratio.
    * **Output:** Error utility with tests against known small cases.

---

## Stage 3: Randomized Low-Rank Primitives
**Objective:** The RSVD building blocks, tested independently of the H-machinery.

* **Task 3.1: Gaussian sampling and orthonormalization helpers.**
    * **Scope:** Draw `n×(k+p)` standard-normal matrices (`numpy.random`,
      seedable). Thin wrappers `Q = qr(Y)` and rank-`k`-truncated `qr(Y, k)`
      around `scipy.linalg.qr`.
    * **Steps:**
        1. `gaussian(n, k, p, seed)` standard-normal draw.
        2. `orth(Y)` and rank-truncated `orth(Y, k)` QR wrappers.
    * **Output:** Helpers with tests verifying orthonormality and shapes.

* **Task 3.2: Algorithm 2.1 two-sample compression.**
    * **Scope:** Given column sample `Y` and row sample `Z` of a single block (and
      the Gaussian blocks used), produce `U = qr(Y, k)`, `V = qr(Z, k)`.
    * **Output:** Function returning `U, V`; test on a synthetic rank-`k` dense
      block: `U`,`V` span the correct subspaces.

* **Task 3.3: Core-matrix solve (Eq. 4.3).**
    * **Scope:** `B_{α,β} = (G_α* U)† (G_α* Y(I_α,:)) (V* G_β)†`, formed entirely
      from samples — never assembling `A_{α,β}`.
    * **Output:** Solver; test that `U B V*` reconstructs the synthetic block to
      tolerance.

---

## Stage 4: Fixed-Pattern Structured Test Matrices (no graph)
**Objective:** Build the structured test matrices via the paper's §4.1.4 fixed
periodic patterns — the simple, correct sampling scheme.

* **Task 4.1: Sampling-constraint descriptor (Eq. 4.4).**
    * **Scope:** For an admissible pair `(α,β)`: mark col-box `β` as *random*
      (`G_β`) and all boxes `γ ∈ L^nei_α ∪ L^int_α \ {β}` as *zero*. Represent as
      a per-block constraint object over col-index ranges.
    * **Output:** Constraint builder; tests on small trees.

* **Task 4.2: Fixed periodic admissible test matrices (≤ 6^d).**
    * **Scope:** Tile the boxes at a level into a `6×…×6` periodic pattern; emit at
      most `6^d` test matrices `Ω` of shape `(N_col, k+p)`, filling each active
      box's col rows with Gaussian blocks and others with zeros.
    * **Steps:**
        1. Periodic box → pattern-cell assignment (`6×…×6` tiling over the
           level's boxes).
        2. Fill one `Ω` per pattern offset: active col-boxes get Gaussian
           blocks, the rest zeros.
    * **Output:** Generator. Test: every admissible block at the level is sampled
      by exactly one `Ω` with all its required zeros satisfied.

* **Task 4.3: Fixed periodic leaf/inadmissible test matrices (≤ 3^d).**
    * **Scope:** Emit ≤ `3^d` test matrices with identity-like blocks on col
      ranges to extract dense neighbor (leaf) blocks while zeroing other
      neighbors (§4.1.3). Not full identity — must stay ≤ 3^d matrices.
    * **Steps:**
        1. Periodic box → pattern-cell assignment (`3×…×3` tiling over leaf
           boxes).
        2. Fill identity-like blocks per pattern offset on the active col ranges.
    * **Output:** Generator; test that each inadmissible leaf block is isolated by
      exactly one matrix.

---

## Stage 5: Peeling Driver & Block Compression (non-uniform H¹)
**Objective:** Drive levels coarse→fine with peeling and produce the H¹ factors.

* **Task 5.1: Level-truncation operator `(A − A^(l−1))`.**
    * **Scope:** Given the low-rank factors already computed for levels `2..l−1`,
      apply `A^(l−1)Ω` from those factors, and compute
      `(A − A^(l−1))Ω = operator.matvec(Ω) − A^(l−1)Ω`. Same for the transpose.
    * **Steps:**
        1. `apply_truncated(factors, Ω) -> A^(l−1)Ω` from the stored low-rank
           factors of coarser levels.
        2. `peeled_matvec(Ω, l) = operator.matvec(Ω) − apply_truncated(...)`.
        3. Transpose variants `apply_truncated_T` / `peeled_rmatvec` for `A*`.
    * **Output:** Wrapper; test that for the coarsest level it equals the raw
      matvec, and that subtracting all levels reproduces stored factors.

* **Task 5.2: Per-level column bases `U_{α,β}`.**
    * **Scope:** Apply the level's admissible test matrices through Task 5.1,
      extract `Y(I_α,:)` for each admissible pair, `U_{α,β} = qr(Y(I_α,:), k)`.
    * **Steps:**
        1. Apply the level's `Ω` test matrices through the peeled matvec
           (Task 5.1).
        2. Per admissible pair: extract `Y(I_α,:)` from the right sample and set
           `U_{α,β} = qr(·, k)`.
    * **Output:** Column-basis pass; tests verifying basis orthogonality and
      subspace accuracy against `MockGF` blocks.

* **Task 5.3: Per-level row bases `V_{α,β}`.**
    * **Scope:** Same as 5.2 but through `rmatvec` with `Ψ` test matrices, giving
      `V_{α,β} = qr(Z(I_β,:), k)`.
    * **Output:** Row-basis pass with tests.

* **Task 5.4: Per-level core matrices `B_{α,β}`.**
    * **Scope:** Combine the level's `U`, `V`, and stored samples via Task 3.3.
    * **Output:** Core pass; per-block reconstruction error test.

* **Task 5.5: Leaf inadmissible block extraction.**
    * **Scope:** At the finest level, apply Task 4.3 patterns and read off the
      dense neighbor blocks.
    * **Output:** Dense-leaf extractor with exactness test.

* **Task 5.6: `HMatrix` container and `dot`.**
    * **Scope:** Store all `U_{α,β}, B_{α,β}, V_{α,β}` and the dense leaf blocks.
      Implement `dot(x)` and `rdot(y)` respecting the rectangular shapes
      (`2N×N` / `3N×2N`).
    * **Steps:**
        1. `HMatrix` container storing the factors, dense leaves, and tree/index
           maps.
        2. `dot(x)`: dense leaf blocks plus admissible `U (B (V* x))`.
        3. `rdot(y)`: the transpose application.
    * **Output:** `HMatrix` with fast matvec; test against the dense reference
      product.

---

## Stage 6: End-to-End Validation (fixed-pattern path)
**Objective:** Prove the fixed-pattern pipeline is correct in 2D and 3D before
adding coloring.

* **Task 6.1: Integration test.**
    * **Scope:** Build small 2D (`2N×N`) and 3D (`3N×2N`) meshes from `MockGF`,
      run the full fixed-pattern compression, and check
      `‖A_H − A‖/‖A‖ < tol` via Task 2.3. Log compression ratio, matvec count,
      and setup time.
    * **Output:** Passing integration suite for both dimensions.

* **Task 6.2: Parameter regression fixtures.**
    * **Scope:** Small sweeps over rank `k`, oversampling `p`, leaf size `m`, and
      `η`; record error/matvec counts as regression baselines.
    * **Output:** Fixture-backed regression tests.

---

## Stage 7: Graph-Coloring Optimization
**Objective:** Replace fixed patterns with problem-tailored test matrices that
reproduce Stage 6 accuracy using fewer matvecs (paper §4.1.2).

* **Task 7.1: Constraint-set deduplication and incompatibility graph (Def. 4.1).**
    * **Scope:** Build, *per level*, the set of distinct sampling-constraint sets
      (multiple admissible blocks may share one — vertices < blocks). Add an edge
      between two sets iff incompatible (one requires a box random where the other
      requires it zero).
    * **Steps:**
        1. Canonicalize + deduplicate constraint sets (shared sets collapse to
           one vertex).
        2. Pairwise incompatibility test (random-vs-zero conflict on a shared
           box).
        3. Assemble the per-level adjacency list.
    * **Output:** Per-level graph (adjacency list); tests on the small Fig. 4 case
      (vertex count < block count; correct edges).

* **Task 7.2: DSatur coloring.**
    * **Scope:** Degree-of-saturation greedy coloring (Algorithm 2.3) returning a
      vertex→color map.
    * **Steps:**
        1. Saturation-degree priority structure over vertices.
        2. Greedy color-assignment loop with neighbor saturation updates.
    * **Output:** `dsatur` with tests on standard graphs verifying valid colorings.

* **Task 7.3: Color-group → test-matrix builder.**
    * **Scope:** For each color, assemble one test matrix satisfying all sampling
      constraints in that group. Drop-in replacement for Task 4.2's output.
    * **Output:** Builder; test that each admissible block is still sampled with
      its zeros satisfied, using ≤ 6^d (and typically far fewer) matrices.

* **Task 7.4: Leaf incompatibility graph + coloring.**
    * **Scope:** The `3^d` analog for inadmissible leaf extraction (Fig. 5),
      replacing Task 4.3's output.
    * **Output:** Leaf coloring path with tests.

* **Task 7.5: Wire coloring into the driver behind a flag.**
    * **Scope:** Add a `sampling="fixed" | "coloring"` option to the compression
      driver. Default stays `fixed`; `coloring` uses Stages 7.3/7.4.
    * **Output:** Regression test: `coloring` matches `fixed` accuracy within
      tolerance while issuing strictly fewer matvecs on the 2D/3D fixtures.
