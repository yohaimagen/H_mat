# FIXPLAN — Remediation of Tasks 1.2–5.2 before resuming at 5.3

Companion to `plan.md` (Stage F). Produced by the 2026-09-02 review of the
merged code against Levitt & Martinsson (2024). Every task here is one PR off
`main`, run through the normal `/task F.x` pipeline, in the order listed.
When F.5 is merged, resume `plan.md` at **Task 5.3** (revised).

## Why this exists

| # | Finding | Where | Paper reference | Fixed by |
|---|---------|-------|-----------------|----------|
| 1 | Leaf probes give each active box a private column slot, so total probe width = `n_cols` → **O(N) matvecs** (verified: 16×16 grid, 9 matrices, total width 256 = `n_cols`). | `fixed_pattern.build_leaf_test_matrices` | §4.1.3 (`Ω` is `N×m`; `I8, I11, I14` share one column block) | F.4 |
| 2 | `orth(Y, k)` uses unpivoted QR and keeps `Q[:, :k]`, which spans only `Y[:, :k]` — oversampling `p` is discarded. | `randomized.orth` | §2.4: `qr(A, k)` is the column-pivoted truncated QR | F.3 |
| 3a | `_split` bisects a cell **in place** when only one sub-cell is populated → cells leave the `2^l` dyadic grid (verified on a two-cluster point set: 63 nodes with non-integer grid coords, two cell sizes at one level) → `pattern_cell` collisions → Eq. 4.4 silently violated. | `build_tree._split` | §3 (level `l+1` = bisect every level-`l` box) | F.1 |
| 3b | Leaves at different depths on non-uniform meshes (same probe: leaves at levels 2–5). Downstream assumes all leaves at `L`; blocks between a shallow leaf and a deeper neighbor's descendants are stored by nobody. | `build_tree` | §3 / Alg. 4.1 ("neighbor pairs in level L") | F.1 |
| 3c | Splits boxes with `≥ m` patches; paper splits `> m`. | `build_tree._split` | §3 | F.1 |
| 4 | `column_bases`/`row_bases` discard `Y(I_α,:)`, `Ω(I_β,:)`, `Ψ(I_α,:)`, which Eq. 4.3 needs. | `column_basis`, `row_basis` | Eq. 4.3 | F.5 (+ revised 5.3) |
| 5 | Subspace tests use `k ≥ rank(block)`, so any orthonormal basis passes (this is why #2 was not caught). | `tests/test_column_basis.py`, `tests/test_row_basis.py` | — | F.5 |
| 6 | `row_basis._row_pattern_cell` / `_build_row_test_matrices` duplicate `fixed_pattern` with rows swapped in. | `row_basis` | Alg. 4.1 remark: reuse `{Ω_i}` pattern for `Ψ` | F.4 (+ revised 5.3) |
| 7 | Docstrings justify period 6 with "`L^int` within ±2 → 5×5 window". Wrong: children of the parent's neighbors span offsets `−3..+2` (6 wide); period 5 would break. | `fixed_pattern`, `row_basis` | §4.1.4 "a square of 6×6 boxes" | F.4 |
| 8 | `neighbor_lists`/`interaction_lists` rebuilt per call; `build_sampling_constraint` rebuilds both maps **per pair** → Stage 7.1 would be `O(pairs·N²)`. | `interactions`, `sampling` | — | F.2 |
| 9 | `TreeNode` unhashable → `id()`-keyed dicts and `next(idx for …)` linear scans throughout. | `tree`, `neighbors`, `interactions`, `sampling`, `*_basis` | — | F.1, F.2 |
| 10 | `MockGF._assemble` is a Python double loop; the corrected tests need `~10^3`-patch meshes. | `mockgf` | — | F.4 |

Not changed (reviewed, correct): `operators`, `error`, `two_sample_compress`,
`core_matrix_solve` (matches Eq. 4.3 with `G_α = Ψ(I_α,:)`, `G_β = Ω(I_β,:)`),
`peeling` (`A − A^(l−1)` and its adjoint), `PERIOD = 6`, `LEAF_PERIOD = 3`,
`interaction_lists` combinatorics, `is_admissible`/`DEFAULT_ETA` consistency.

## Handling the in-flight Task 5.3 branch

`task/5.3` has an open draft PR and **untracked** `src/gfcompress/row_basis.py`
and `tests/test_row_basis.py`. Do not commit them as-is: the revised 5.3 merges
row bases with core matrices into `compress_level` and consumes the
side-parametrized builder from F.4. Keep the files locally as a starting point
(the `row_bases` loop body and the orthogonality tests carry over), rebase
`task/5.3` onto `main` after F.5 merges, then run `/task 5.3` against the
revised block in `plan.md`. Update the 5.3 PR body to the revised scope.

---

## Task F.1: Tree — uniform depth, no in-place bisection, hashable nodes
*Revises plan.md Tasks 1.2 and 1.3. Paper §3.*

* **Scope:**
    * `TreeNode`: declare `@dataclass(eq=False)` so nodes hash by identity;
      add `index_in_level: int` (position in `nodes_at_level(level)`), set by
      the builder. Nothing else in the node changes.
    * `build_tree(mesh, m, max_depth=64)`: replace the recursive
      `_split` with a **level-synchronous** loop:
      ```
      level_nodes = [root]
      while any(len(n.patch_indices) > m for n in level_nodes)
            and depth < max_depth and cells have not underflowed:
          next = []
          for node in level_nodes:
              for (patches, cell) in _bisect_cell(node):   # ≤ 2^d non-empty
                  next.append(make child at level+1 with cell geometry)
          level_nodes = next
      ```
      Every node at a level is bisected — including nodes that hold `≤ m`
      patches — so all leaves are at the same depth `L`. A single non-empty
      child is a normal child; **delete the "bisect in place" `while True`
      loop entirely.** Continue condition is `> m` (paper), not `≥ m`.
    * Cell-width underflow guard: if `mid == lo` or `mid == hi` on any axis
      (coincident centroids), stop refining; the current level becomes `L`.
    * Remove `id()`-keyed dicts / `next(idx for …)` scans in `tree.py`
      callers that F.1 touches; the rest is F.2.
* **Files:** `src/gfcompress/tree.py`, `src/gfcompress/build_tree.py`,
  `tests/test_tree.py`, `tests/test_build_tree.py`.
* **Tests (add/adjust):**
    1. `TreeNode` is hashable; `{node}`/`dict[node]` work; two distinct nodes
       with identical fields are not equal.
    2. `node.index_in_level == root.nodes_at_level(node.level).index(node)`
       for every node.
    3. **Uniform depth:** `len({leaf.level for leaf in root.leaves()}) == 1`
       on (a) a uniform grid, (b) a two-cluster point set (`40` points in
       `[0, .1]²`, `40` in `[.9, 1]²`), (c) a 3D planar-fault set (points with
       `z = 0.3` inside a unit cube), for several `m`.
    4. **Dyadic-grid invariant** on the same three point sets: for every node,
       `(bounding_box[:,0] − root.bounding_box[:,0]) / (root_width / 2**level)`
       is an integer to `1e-9`, and all same-level cells have identical widths.
    5. `> m` rule: a set of exactly `m` points is a single leaf (root at level
       0); `m + 1` points split.
    6. Existing partition tests (leaf row/col index sets cover `{0..n_rows−1}`
       / `{0..n_cols−1}` exactly) still pass.
    7. Termination: `m` coincident centroids with `m` small terminates and
       yields a leaf level.
* **Acceptance:** all tests pass; existing `test_neighbors`,
  `test_interactions`, `test_fixed_pattern`, `test_column_basis` pass unchanged
  (they run on uniform grids, where behaviour is identical); `ruff`, `black`,
  `mypy` clean.

## Task F.2: `TreeLists` — compute neighbor/interaction maps once
*Revises plan.md Tasks 1.4, 1.5, 4.1.*

* **Scope:**
    * New `TreeLists` dataclass in `interactions.py`:
      `nei: dict[TreeNode, list[TreeNode]]`,
      `int: dict[TreeNode, list[TreeNode]]` (name it `interaction` to avoid
      shadowing the builtin), built once by `build_lists(root) -> TreeLists`.
      Keyed by node (hashable after F.1) — drop the
      `level -> index_in_level -> list` nesting; callers that need per-level
      iteration use `root.nodes_at_level(l)` and index the dict.
    * `neighbor_lists(root)` keeps its current O(n_l²) touch test but returns
      the node-keyed flat dict (or is folded into `build_lists`). Keep
      `boxes_adjacent` and `box_dist` as-is.
    * `interaction_lists(root, nei)` takes the precomputed neighbor map. Remove
      the single-node `interaction_list(alpha, root)` wrapper (it rebuilt the
      whole map per call) or make it index `TreeLists`.
    * `build_sampling_constraint(alpha, beta, lists: TreeLists)` — no `root`,
      no recomputation.
    * `column_bases(...)` (and the in-flight `row_bases`) take `lists`
      instead of calling `interaction_lists(root)`.
* **Files:** `src/gfcompress/neighbors.py`, `src/gfcompress/interactions.py`,
  `src/gfcompress/sampling.py`, `src/gfcompress/column_basis.py`, all
  corresponding tests.
* **Tests:**
    1. `build_lists(root).nei[node]` / `.interaction[node]` equal the previous
       per-level maps on the existing grid fixtures (port the existing count,
       disjointness, cover, and Fig. 3 tests to the new keying).
    2. Fix the docstring/test comment on the window: `L^nei ∪ L^int` spans
       grid offsets `−3..+2` along each axis. Add a test on a 1D-like grid
       (`16×1` in 2D) that the max offset of an interaction-list member is
       `3` on one side and `2` on the other.
    3. `build_sampling_constraint` results unchanged vs. current behaviour on
       the small-tree fixtures.
    4. A cheap perf guard: `build_lists` on a `64×64` grid with `m = 4`
       completes in `< 2 s`; `build_sampling_constraint` for all admissible
       pairs at the leaf level completes in `< 2 s` (it was quadratic per
       pair).
* **Acceptance:** tests green, lint/type clean, no remaining `id(` in
  `src/gfcompress/`.

## Task F.3: Column-pivoted `orth(Y, k)`
*Revises plan.md Task 3.1. Paper §2.4.*

* **Scope:** In `randomized.orth`, when `k is not None` use
  `q, _, _ = scipy.linalg.qr(y, mode="economic", pivoting=True)` and return
  `q[:, :k]`. `orth(Y)` (no `k`) stays unpivoted (paper's `Q = qr(A)`). Update
  the docstring: pivoted truncation approximates the dominant `k`-dimensional
  subspace of `range(Y)`; unpivoted `Q[:, :k]` would only span `Y[:, :k]`.
  `two_sample_compress` and `core_matrix_solve` need no change.
* **Files:** `src/gfcompress/randomized.py`, `tests/test_randomized.py`.
* **Tests:**
    1. `Y = [1e-8·u, Y_big]` where `Y_big` is `n×k` with well-conditioned
       columns: `orth(Y, k)` spans `range(Y_big)` to `1e-10` (projection
       residual). Also assert the *unpivoted* first-`k` `Q` does **not** (so the
       test is known to discriminate).
    2. `Y = A G` for a synthetic `A` with singular values `10^{-j}` and
       `k = 5`, `p = 5`: `‖A − Q Q* A‖ / ‖A‖` from `orth(Y, k)` is within a
       factor `10` of `σ_{k+1}`.
    3. Existing orthonormality/shape tests unchanged.
* **Acceptance:** green, lint/type clean.

## Task F.4: Test-matrix builders — side parametrization, shared-slot leaf probes; vectorized `MockGF`
*Revises plan.md Tasks 4.2, 4.3, 2.2. Paper §4.1.3, §4.1.4, Alg. 4.1.*

* **Scope:**
    * **Admissible builder, both sides.**
      `build_admissible_test_matrices(root, level, mesh, k, p, seed, side="col")`.
      `side="col"`: fills `omega[box.col_indices, :]`, shape `(n_cols, k+p)`.
      `side="row"`: fills `psi[box.row_indices, :]`, shape `(n_rows, k+p)`.
      Extend the record: `PeriodicTestMatrix(omega, pattern, active_boxes,
      blocks: dict[TreeNode, NDArray])` where `blocks[box]` is the Gaussian
      block written into that box's rows (so `G_β`/`G_α` are read back, not
      re-derived from seeds). Seeds for the two sides must be independent
      (e.g. derive `side` into the seed stream) so `Ω` and `Ψ` never share
      rows.
    * **Leaf builder, shared slots.** `build_leaf_test_matrices(root, level,
      mesh)`: `w_max = max(len(b.col_indices) for b in level nodes)`; each
      `omega` has shape `(n_cols, w_max)`; for every active `β`,
      `omega[β.col_indices, :w_β] = I_{w_β}`. Record type
      `PeriodicLeafTestMatrix(omega, pattern, active_boxes)` — **drop
      `col_slices`**. The consumer reads `Y[α.row_indices, :w_β]`.
    * **Docstrings:** replace the "±2 → 5×5 window" argument in
      `fixed_pattern.py` with the correct one: `L^int(α)` = children of the
      parent's neighbors, spanning grid offsets `−3..+2` relative to `α` (a
      6-wide window), hence period 6 is minimal; `L^nei` spans `−1..+1`, hence
      period 3.
    * **`MockGF._assemble` vectorized:** `diff = X[:, None, :] − X[None, :, :]`
      `(N,N,d)`, `r = ‖diff‖`, `denom = r + eps`,
      `T = (I + diff⊗diff / denom²) / denom^d` as `(N,N,d,d)`, keep
      `[:, :, :, :dof_col]`, transpose/reshape to `(dof_row·N, dof_col·N)`.
      Keep `kernel_block` as the scalar reference and assert equality.
* **Files:** `src/gfcompress/fixed_pattern.py`, `src/gfcompress/mockgf.py`,
  `tests/test_fixed_pattern.py`, `tests/test_mockgf.py`.
* **Tests:**
    1. Admissible, `side="col"`: existing Eq. 4.4 coverage tests, on a grid
       with `≥ 12` boxes per axis at the tested level (so pattern cells wrap
       and the 6-wide window is exercised).
    2. Admissible, `side="row"`: for every admissible `(α,β)`, the unique `Ψ`
       whose `active_boxes` contains `α` has Gaussian rows on `α.row_indices`
       and zeros on `γ.row_indices` for all `γ ∈ L^nei(β) ∪ L^int(β) \ {α}`.
    3. `blocks[box]` equals `omega[box.<side>_indices, :]`.
    4. Leaf: `len(matrices) ≤ 3^d`; `sum(width) ≤ 3^d · w_max` (assert it is
       `≪ n_cols` on a `16×16`, `m=4` grid); for every neighbor pair `(α,β)`,
       exactly one matrix has `β` active and in it **no other member of
       `L^nei(α)` is active**.
    5. Leaf recovery through `A − A^(L)`: build `factors` as *exact* truncated
       SVDs of every admissible block from `MockGF.block` (test-only
       ground truth, full rank so the far field is peeled to round-off), then
       `Y = peeled_matvec(gf, omega, factors)` and
       `Y[α.row_indices, :w_β] ≈ gf.block(α, β)` to `1e-8` for every neighbor
       pair. (Replace the old exact test that relied on private slots.)
    6. `MockGF`: vectorized `A` equals the loop assembly on a small mesh to
       `1e-14`; a `1024`-patch 2D mesh assembles in `< 1 s`.
* **Acceptance:** green, lint/type clean; `column_basis.py` adjusted to the
  new record (it already reads `active_boxes`).

## Task F.5: `column_bases` retains `Y(I_α,:)` and `G_β`; real low-rank tests
*Revises plan.md Task 5.2. Paper Eq. 4.3, Alg. 4.1.*

* **Scope:**
    * `ColumnBasis(alpha, beta, u, y_alpha, g_beta)`:
      `y_alpha = Y[α.row_indices, :]` (shape `(dof_row·|α|, k+p)`),
      `g_beta = tm.blocks[β]` (shape `(dof_col·|β|, k+p)`).
    * Signature: `column_bases(operator, root, lists, mesh, level, factors, k,
      p, seed)` — takes `TreeLists` (F.2), uses the `side="col"` builder (F.4)
      and pivoted `orth` (F.3). Keep the guarantee "exactly one peeled matvec
      of width `k+p` per emitted `Ω`".
    * Return order unchanged (outer loop over `nodes_at_level`, inner over
      `lists.interaction[α]`).
* **Files:** `src/gfcompress/column_basis.py`, `tests/test_column_basis.py`.
* **Tests (replace the trivial `k ≥ rank` ones):**
    1. 2D `32×32` grid, `m = 8` → level 2 has `4×4` boxes of `64` patches;
       blocks are `128×64`. Use `k = 10, p = 10`. For every admissible pair:
       `‖A_{αβ} − U U* A_{αβ}‖ / ‖A_{αβ}‖ < 1e-3` **and** the block's
       `σ_{k+1}/σ_1 > 1e-8` (proves truncation is real).
    2. 3D `8×8×8` grid, `m = 8`, `k = 6, p = 6`: same assertions at the first
       admissible level.
    3. `y_alpha` and `g_beta` shapes; `y_alpha ≈ A_{αβ} g_beta` on the
       coarsest level (no peeling) to `1e-10` — this is exactly the quantity
       Eq. 4.3 consumes.
    4. Matvec count: wrap `MockGF` in a local counting shim; assert
       `count == len(test_matrices) · (k+p)` and `len(test_matrices) ≤ 6^d`.
* **Acceptance:** green, lint/type clean. After merge, rebase `task/5.3` and
  resume `plan.md` at the revised Task 5.3.

---

## After F.5

Resume the main plan at **5.3 → 5.4 → 5.5 → 5.6 → 6.x → 7.x** as written in
`plan.md`. The revised Stage 5 ordering is: `compress_level` (5.3), leaf
extraction through `A − A^(L)` (5.4), `HMatrix` + coverage test (5.5),
`compress()` driver + `CountingOperator` (5.6).
