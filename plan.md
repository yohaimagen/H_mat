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

> **Revision note (2026-09-02).** Tasks 0.1–5.2 were implemented under the
> previous revision of this plan. A review against the paper found deviations
> (leaf probes costing O(N) matvecs, unpivoted `qr(·,k)`, a tree builder that
> breaks the dyadic grid and yields non-uniform leaf depth, and a 5.2→5.4
> interface gap). Those are repaired by the remediation tasks in **Stage F**
> (`FIXPLAN.md`), which run *before* resuming at Task 5.3. Task blocks below
> describe the **target** state; where an already-done task's description
> changed, the change is delivered by the Stage F task named in brackets.

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
- **Uniform-depth dyadic tree (paper §3).** Level `l` boxes are the non-empty
  cells of the root box subdivided `2^l` times per axis; a cell is *never*
  bisected in place. Splitting is **level-synchronous**: a level is split iff
  some box on it holds `> m` patches, so **every leaf sits at the same level
  `L`**. Consequently the block tessellation is exactly {admissible pairs at
  levels `2..L`} ∪ {neighbor pairs at level `L`} (Fig. 3), and it is a
  complete, disjoint cover of all `N×N` patch pairs — asserted by a test.
- **Row/column duality.** `Ψ` test matrices use the *same* box grouping as `Ω`
  (Algorithm 4.1's "reuse `{Ω_i}`" remark), differing only in which index set
  (`row_indices` vs `col_indices`) and which length (`n_rows` vs `n_cols`) they
  fill. One builder, parametrized by side.
- **`qr(·, k)` means column-pivoted truncated QR (paper §2.4).** Unpivoted
  `Q[:, :k]` spans only the first `k` sample columns and silently discards the
  oversampling `p`.
- **The matvec budget is a deliverable.** Per level: `6^d·(k+p)` products with
  `A` and `6^d·(k+p)` with `A*`. Leaf pass: `≤ 3^d` probes of width
  `w_max = dof_col·m`. Anything scaling like `O(N)` matvecs is a bug, and the
  driver test asserts the count.
- **Build order de-risks the hardest piece.** The paper's §4.1.4 fixed periodic
  test-matrix patterns (≤ 6^d admissible, ≤ 3^d leaf) yield a *fully correct*
  algorithm with **no graph at all**. We implement that path first and validate
  end-to-end (Stages 0–6). Graph coloring (§4.1.2) is then added as a drop-in
  optimization with a clean regression target: identical accuracy, fewer matvecs
  (Stage 7).

### Testing rules (in addition to CLAUDE.md)
- **Low-rank tests must truncate.** Any test of a basis `U`/`V` or a factor
  `U B V*` must use `k` strictly below the numerical rank of the block (e.g. a
  32×32 2D grid with `m = 8` gives level-2 blocks of size `128×64`; use
  `k ≈ 10`). A test with `k ≥ rank` passes for *any* orthonormal basis and
  validates nothing.
- **Whole-matrix coverage.** The set of stored blocks must partition the `N×N`
  patch-pair set exactly once (dense leaf pairs ∪ admissible pairs over all
  levels).
- **Matvec count.** End-to-end tests wrap the operator in a counting wrapper
  and assert the budget above.

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
      (`dof_col` per patch).
    * **Steps:**
        1. `FaultMesh` dataclass (centroids, `L`, `d`, derived
           `dof_row`/`dof_col`).
        2. `patch_to_rows(patch_ids)` and `patch_to_cols(patch_ids)` index
           expansion helpers.
    * **Output:** Mesh class; unit tests for 2D and 3D index mapping
      (sizes `2N×N`, `3N×2N`).

* **Task 1.2: `TreeNode` class.** *[revised by F.1]*
    * **Scope:** Fields: `patch_indices`, `row_indices`, `col_indices`,
      `children`, `parent`, `level`, `index_in_level`, axis-aligned dyadic
      `bounding_box`, `center`, `diam`. The dataclass is declared with
      `eq=False` so nodes are **hashable by identity** and can key dicts/sets
      directly (no `id()` plumbing, no linear scans to recover a node's
      position). Recursive traversal helpers (leaves, levels, by-level
      iteration).
    * **Steps:**
        1. `TreeNode` dataclass (`eq=False`): fields + parent/child links.
        2. Geometry computation from a cell: `bounding_box`, `center`, `diam`.
        3. Traversal helpers: `leaves()`, `nodes_at_level(l)`, by-level iterator.
    * **Output:** `TreeNode` with traversal utilities and tests (including
      hashability and `index_in_level` consistency).

* **Task 1.3: Uniform-depth dyadic tree builder (paper §3).** *[revised by F.1]*
    * **Scope:** Level 0 is the root box enclosing all centroids. Level `l+1`
      is obtained by bisecting **every** level-`l` box at its geometric
      midpoint along every axis (`≤ 2^d` children; empty cells omitted). A cell
      is never bisected in place — a single non-empty child is a legitimate
      level-`(l+1)` box. Splitting is **level-synchronous**: proceed to level
      `l+1` iff some level-`l` box holds `> m` patches (paper: "boxes that
      contain more than `m` points"); otherwise every level-`l` box is a leaf,
      so all leaves share the depth `L`. Termination guard: stop when the cell
      width underflows (coincident centroids) or a `max_depth` is reached.
    * **Steps:**
        1. Single-cell bisection into `≤ 2^d` non-empty sub-cells.
        2. Level-synchronous builder with the `> m` continue condition.
        3. Populate each node's `row_indices`/`col_indices` and
           `index_in_level`.
    * **Output:** Builder returning the root. Tests: leaf row/col index sets
      partition `{0..2N-1}` / `{0..3N-1}` exactly; **all leaves at one level**;
      on a *clustered* (non-uniform) point set every node's cell has width
      `root_width / 2^level` and integer grid coordinates (the dyadic-grid
      invariant); verified in 2D and 3D.

* **Task 1.4: Neighbor lists `L^nei`.** *[revised by F.2]*
    * **Scope:** For each box, compute same-level boxes whose bounding boxes touch
      or overlap (includes itself); ≤ 3^d entries on a regular grid. Computed
      **once per tree** and handed to consumers; keyed by `TreeNode`.
    * **Steps:**
        1. Box-adjacency predicate (two bounding boxes touch or overlap).
        2. Per-level neighbor-list map built from the predicate.
    * **Output:** Neighbor-list map. Tests on a uniform grid checking expected
      counts (3 in 1D, 9 in 2D, 27 in 3D interior boxes).

* **Task 1.5: Interaction lists `L^int`, admissibility predicate, `TreeLists`.**
  *[revised by F.2]*
    * **Scope:** `L^int(α)` = children of α's parent's neighbors, excluding α's
      own neighbors (≤ 6^d − 3^d). Admissibility predicate
      `dist(α,β) ≥ η · max(diam α, diam β)` (cross-check only — the paper's
      admissibility *is* membership in the interaction list); `suggest_eta`
      from the `1/(r+γL)^d` decay. Bundle both maps in a `TreeLists(nei, int)`
      object built **once** by `build_lists(root)`; every downstream function
      (sampling constraints, test-matrix builders, basis passes, graph
      construction) takes `TreeLists` rather than recomputing.
    * **Steps:**
        1. `interaction_lists(root, nei)` from the precomputed neighbor map.
        2. `is_admissible(α, β)`: the `dist ≥ η · max(diam)` predicate.
        3. `suggest_eta(...)`; `TreeLists` + `build_lists(root)`.
    * **Output:** Interaction-list map + predicate. Tests: at each level the
      admissible (interaction-list) and inadmissible (neighbor) blocks form a
      complete, disjoint cover; reproduce the small Fig. 3 tessellation. Note
      the correct geometric window: `L^nei ∪ L^int` spans grid offsets
      `−3..+2` (a **6-wide** window, paper §4.1.4 "a square of 6×6 boxes"),
      not `±2`.

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

* **Task 2.2: Analytic-kernel mock GF operator.** *[revised by F.4]*
    * **Scope:** Implement a smooth tensor-valued kernel `K(x_i, x_j)` returning a
      `dof_row × dof_col` block per patch pair (a simplified elastostatic /
      Kelvin-type kernel decaying like `1/r^d`). Assemble the dense rectangular
      `A` **vectorized** (broadcast over all pairs; no Python double loop — the
      Stage 5/6 tests use meshes of `10^3` patches); implement
      `matvec`/`rmatvec`. Expose a direct `block(I_rows, J_cols)` accessor for
      ground-truth tests.
    * **Steps:**
        1. Tensor kernel `K(x_i, x_j) -> (dof_row, dof_col)` block.
        2. Vectorized dense assembler building the flattened patch-major `A`.
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

* **Task 3.1: Gaussian sampling and orthonormalization helpers.** *[revised by F.3]*
    * **Scope:** Draw `n×(k+p)` standard-normal matrices (`numpy.random`,
      seedable). `orth(Y)` = unpivoted economy QR (paper's `Q = qr(A)`).
      `orth(Y, k)` = **column-pivoted** economy QR truncated to `k` columns
      (paper §2.4 `qr(A, k)`): `scipy.linalg.qr(Y, mode="economic",
      pivoting=True)[0][:, :k]`.
    * **Steps:**
        1. `gaussian(n, k, p, seed)` standard-normal draw.
        2. `orth(Y)` and pivoted rank-truncated `orth(Y, k)` QR wrappers.
    * **Output:** Helpers with tests verifying orthonormality and shapes, and a
      test where `Y` has a small leading column: `orth(Y, k)` must span the
      dominant columns (unpivoted would not).

* **Task 3.2: Algorithm 2.1 two-sample compression.**
    * **Scope:** Given column sample `Y` and row sample `Z` of a single block (and
      the Gaussian blocks used), produce `U = qr(Y, k)`, `V = qr(Z, k)`.
    * **Output:** Function returning `U, V`; test on a synthetic block with
      `k < rank` and decaying tail: `U`,`V` span the dominant subspaces.

* **Task 3.3: Core-matrix solve (Eq. 4.3).**
    * **Scope:** `B_{α,β} = (G_α* U)† (G_α* Y(I_α,:)) (V* G_β)†`, formed entirely
      from samples — never assembling `A_{α,β}`. Here `G_β = Ω(I_β,:)` is the
      Gaussian block that produced `Y(I_α,:)` and `G_α = Ψ(I_α,:)` the one that
      produced `Z(I_β,:)`.
    * **Output:** Solver; test that `U B V*` reconstructs the synthetic block to
      tolerance.

---

## Stage 4: Fixed-Pattern Structured Test Matrices (no graph)
**Objective:** Build the structured test matrices via the paper's §4.1.4 fixed
periodic patterns — the simple, correct sampling scheme.

* **Task 4.1: Sampling-constraint descriptor (Eq. 4.4).** *[revised by F.2]*
    * **Scope:** For an admissible pair `(α,β)`: mark col-box `β` as *random*
      (`G_β`) and all boxes `γ ∈ L^nei_α ∪ L^int_α \ {β}` as *zero*. Represent as
      a per-block constraint object over col-index ranges. Takes `TreeLists`;
      never recomputes neighbor/interaction maps. Consumed by Stage 7.
    * **Output:** Constraint builder; tests on small trees.

* **Task 4.2: Fixed periodic admissible test matrices (≤ 6^d), both sides.**
  *[revised by F.4]*
    * **Scope:** Tile the level's dyadic grid with period 6 per axis; group boxes
      by pattern cell; emit one test matrix per non-empty cell (≤ `6^d`). The
      builder is parametrized by **side**: `side="col"` fills
      `Ω[β.col_indices, :]` with Gaussian blocks (shape `(n_cols, k+p)`);
      `side="row"` fills `Ψ[α.row_indices, :]` (shape `(n_rows, k+p)`). Each
      emitted matrix records its `active_boxes` and the per-box Gaussian block
      `G` so later passes can read `G_β`/`G_α` without re-deriving seeds.
      Independent seeds for the two sides.
    * **Steps:**
        1. Periodic box → pattern-cell assignment (`6×…×6` tiling over the
           level's boxes).
        2. Fill one matrix per pattern offset for the requested side.
    * **Output:** Generator. Tests: every admissible block at the level is
      sampled by exactly one `Ω` with all Eq. 4.4 zeros satisfied, and by
      exactly one `Ψ` with the dual zeros (`γ ∈ L^nei_β ∪ L^int_β \ {α}`)
      satisfied; the test grid must be large enough that pattern cells wrap
      (`≥ 12` boxes per axis) so the 6-wide window is actually exercised.

* **Task 4.3: Fixed periodic leaf/inadmissible test matrices (≤ 3^d, width `w_max`).**
  *[revised by F.4]*
    * **Scope:** Paper §4.1.3: `Ω(I_β,:) = I`, `Ω(I_γ,:) = 0` for
      `γ ∈ L^nei_α \ {β}`, applied to `A − A^(L)`. Tile the leaf level with period
      3; emit `≤ 3^d` matrices, **each of width `w_max = max_β |I_β|`
      (`= dof_col·m` on a full leaf)**. All active boxes in a matrix **share the
      same columns**: `Ω[β.col_indices, :w_β] = I_{w_β}`. Isolation of
      `A(I_α, I_β)` comes from (i) peeling `A^(L)` so far-field rows vanish and
      (ii) the period-3 zeros on `α`'s other neighbors — *not* from private
      column slots. Total probe width over all matrices is `≤ 3^d·w_max`,
      independent of `N`.
    * **Steps:**
        1. Periodic box → pattern-cell assignment (`3×…×3` tiling over leaf
           boxes).
        2. Fill shared identity blocks per pattern offset.
    * **Output:** Generator; tests: each inadmissible leaf pair `(α,β)` has
      exactly one matrix in which `β` is the only active member of `L^nei(α)`;
      `sum(width) ≤ 3^d·w_max`; and reading `(A − A^(L))Ω` (with `A^(L)` built
      from exact ground-truth block SVDs of `MockGF`, test-only) recovers every
      dense neighbor block to `1e-8`.

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

* **Task 5.2: Per-level column bases `U_{α,β}` (+ retained samples).**
  *[revised by F.5]*
    * **Scope:** Apply the level's `side="col"` test matrices through Task 5.1,
      extract `Y(I_α,:)` for each admissible pair, `U_{α,β} = qr(Y(I_α,:), k)`
      (pivoted). **Return, per pair, everything Eq. 4.3 needs later:**
      `ColumnBasis(alpha, beta, u, y_alpha, g_beta)` with
      `y_alpha = Y(I_α,:)` and `g_beta = Ω(I_β,:)`.
    * **Steps:**
        1. Apply the level's `Ω` test matrices through the peeled matvec
           (Task 5.1) — exactly `≤ 6^d` matvecs of width `k+p`.
        2. Per admissible pair: extract `Y(I_α,:)`, `G_β`, set
           `U_{α,β} = qr(·, k)`.
    * **Output:** Column-basis pass; tests verifying basis orthogonality and
      subspace accuracy against `MockGF` blocks with `k < rank` (32×32 grid,
      `m = 8`, `k ≈ 10`).

* **Task 5.3: Per-level row bases `V_{α,β}`, core matrices `B_{α,β}`, and
  `compress_level`.**
    * **Scope:** Mirror of 5.2 through `peeled_rmatvec` with the `side="row"`
      test matrices `Ψ`: `RowBasis(alpha, beta, v, g_alpha)` with
      `V_{α,β} = qr(Z(I_β,:), k)` and `g_alpha = Ψ(I_α,:)`. Then combine with
      the 5.2 output via Task 3.3 (Eq. 4.3) into a `BlockFactor(alpha, beta,
      u, b, v)` per pair. Expose `compress_level(operator, root, lists, mesh,
      level, factors, k, p, seed) -> list[BlockFactor]` — the body of
      Algorithm 4.1's level loop — which issues exactly `n_Ω + n_Ψ ≤ 2·6^d`
      products of width `k+p`.
    * **Steps:**
        1. `row_bases(...)` using the shared side-parametrized builder.
        2. `core_matrices(col_bases, row_bases)` via `core_matrix_solve`.
        3. `compress_level(...)` = 5.2 + steps 1–2.
    * **Output:** Row-basis pass, core pass, level driver. Tests: `V`
      orthonormal and captures the dominant row space with `k < rank`;
      per-block `‖A_{α,β} − U B V*‖/‖A_{α,β}‖` small on `MockGF`; a peeled
      second level (factors from level 2 fed into level 3) reaches the same
      per-block accuracy; matvec count per level asserted.

* **Task 5.4: Leaf inadmissible block extraction through `A − A^(L)`.**
    * **Scope:** At level `L`, apply the Task 4.3 shared-slot probes through
      `peeled_matvec(·, all factors)` and read
      `A(I_α, I_β) ≈ Y(I_α, :w_β)` for every neighbor pair. Store as
      `DenseLeaf(alpha, beta, block)`.
    * **Output:** Dense-leaf extractor. Test: on `MockGF` with factors from
      `compress_level` over levels `2..L`, recovered leaf blocks match
      `MockGF.block` to the far-field approximation tolerance; with
      ground-truth exact factors (test-only) they match to `1e-8`; total probe
      width `≤ 3^d·w_max`.

* **Task 5.5: `HMatrix` container, `dot`/`rdot`, coverage test.**
    * **Scope:** Store all `BlockFactor`s and `DenseLeaf`s plus the tree.
      Implement `dot(x)` and `rdot(y)` respecting the rectangular shapes
      (`2N×N` / `3N×2N`). Provide `block_partition()` yielding every stored
      block's patch-pair set.
    * **Steps:**
        1. `HMatrix` container storing the factors, dense leaves, and tree/index
           maps.
        2. `dot(x)`: dense leaf blocks plus admissible `U (B (V* x))`.
        3. `rdot(y)`: the transpose application.
    * **Output:** `HMatrix` with fast matvec; tests: matches the dense reference
      product; **`block_partition()` covers every `(i,j)` patch pair exactly
      once** (2D and 3D, including a non-uniform/clustered mesh).

* **Task 5.6: `compress()` driver (Algorithm 4.1) and `CountingOperator`.**
    * **Scope:** `compress(operator, mesh, m, k, p, seed, sampling="fixed")
      -> HMatrix`: build tree and `TreeLists`, loop `l = 2..L` calling
      `compress_level` and accumulating factors, then Task 5.4 leaf extraction.
      `CountingOperator(op)` wraps any `MatVecOperator` and counts
      `matvec`/`rmatvec` **columns** issued.
    * **Output:** Driver + counter. Tests: `relative_error(H, A) < tol` on small
      2D/3D `MockGF`; matvec count equals
      `Σ_l (n_Ω(l) + n_Ψ(l))·(k+p) + Σ_leaf-probes width`, with
      `n_Ω(l), n_Ψ(l) ≤ 6^d` and leaf width `≤ 3^d·w_max`.

---

## Stage 6: End-to-End Validation (fixed-pattern path)
**Objective:** Prove the fixed-pattern pipeline is correct in 2D and 3D before
adding coloring.

* **Task 6.1: Integration test.**
    * **Scope:** Build 2D (`2N×N`) and 3D (`3N×2N`) meshes from `MockGF` that
      are large enough for `k < rank` at every level (2D: `32×32`, `m = 8`;
      3D: `8×8×8`, `m = 8`), run `compress(..., sampling="fixed")`, and check
      `‖A_H − A‖/‖A‖ < tol` via Task 2.3. Log compression ratio, matvec count
      (via `CountingOperator`), and setup time.
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
      (multiple admissible blocks may share one — vertices < blocks) from the
      precomputed `TreeLists`. Add an edge between two sets iff incompatible
      (one requires a box random where the other requires it zero). The
      `Ψ`-side graph is the same construction with `(α,β)` swapped (constraints
      on `β`'s neighborhood with `α` random); for one shared tree it is the
      transpose of the `Ω`-side graph and its coloring can be reused.
    * **Steps:**
        1. Canonicalize + deduplicate constraint sets (shared sets collapse to
           one vertex).
        2. Pairwise incompatibility test (random-vs-zero conflict on a shared
           box).
        3. Assemble the per-level adjacency list; complexity must be
           `O(#pairs · 6^d)`, never `O(#pairs · N)`.
    * **Output:** Per-level graph (adjacency list); tests on the small Fig. 4 case
      (vertex count < block count; correct edges).

* **Task 7.2: DSatur coloring.**
    * **Scope:** Degree-of-saturation greedy coloring (Algorithm 2.3) returning a
      vertex→color map.
    * **Steps:**
        1. Saturation-degree priority structure over vertices.
        2. Greedy color-assignment loop with neighbor saturation updates.
    * **Output:** `dsatur` with tests on standard graphs verifying valid colorings.

* **Task 7.3: Color-group → test-matrix builder (both sides).**
    * **Scope:** For each color, assemble one test matrix satisfying all sampling
      constraints in that group, reusing the side-parametrized fill from Task
      4.2 (`side="col"` → `Ω`, `side="row"` → `Ψ`). Drop-in replacement for
      Task 4.2's output (same record type: `active_boxes` + per-box `G`).
    * **Output:** Builder; test that each admissible block is still sampled with
      its zeros satisfied, using ≤ 6^d (and typically far fewer) matrices.

* **Task 7.4: Leaf incompatibility graph + coloring.**
    * **Scope:** The `3^d` analog for inadmissible leaf extraction (Fig. 5),
      replacing Task 4.3's output with the same shared-slot, width-`w_max`
      record type.
    * **Output:** Leaf coloring path with tests.

* **Task 7.5: Wire coloring into the driver behind a flag.**
    * **Scope:** `compress(..., sampling="fixed" | "coloring")`. Default stays
      `fixed`; `coloring` uses Tasks 7.3/7.4.
    * **Output:** Regression test: `coloring` matches `fixed` accuracy within
      tolerance while issuing strictly fewer matvecs (via `CountingOperator`)
      on the 2D/3D fixtures.

---

## Stage F: Remediation of Tasks 1.2–5.2 (run before resuming at 5.3)
**Objective:** Bring the already-merged code up to the revised task descriptions
above. Full task blocks (scope, steps, tests, acceptance) live in
**`FIXPLAN.md`**; the entries here exist so the `/task` pipeline can resolve
the ids. Run in order; each is one PR off `main`.

* **Task F.1: Tree — uniform depth, no in-place bisection, hashable nodes.**
  Revises Tasks 1.2/1.3. Details: `FIXPLAN.md` §F.1.
* **Task F.2: `TreeLists` — compute neighbor/interaction maps once.**
  Revises Tasks 1.4/1.5/4.1. Details: `FIXPLAN.md` §F.2.
* **Task F.3: Column-pivoted `orth(Y, k)`.**
  Revises Task 3.1. Details: `FIXPLAN.md` §F.3.
* **Task F.4: Test-matrix builders — side parametrization, shared-slot leaf
  probes; vectorized `MockGF`.**
  Revises Tasks 4.2/4.3/2.2. Details: `FIXPLAN.md` §F.4.
* **Task F.5: `column_bases` retains `Y(I_α,:)` and `G_β`; real low-rank tests.**
  Revises Task 5.2. Details: `FIXPLAN.md` §F.5.
