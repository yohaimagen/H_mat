# CLAUDE.md — project conventions

Black-box randomized H-matrix compression of elastostatic Green's Function
matrices, after Levitt & Martinsson (2024). The full task breakdown is in
`PLAN.md`; implement strictly the one task you are given, in the order in
`tasks.txt`.

## Tooling
- Python, packaged with `pyproject.toml`. A project virtualenv lives at `.venv/`
  (Python 3.12, `gfcompress` installed editable). **Always run tooling through
  it** — the bare shell `python`/`pytest` resolves to a different (conda)
  interpreter and will not see the project:
  - tests: `.venv/bin/pytest -q`
  - lint/format/type: `.venv/bin/ruff check .`, `.venv/bin/black .`, `.venv/bin/mypy`
  You may `source .venv/bin/activate` if you prefer, but never invoke a bare
  `python`/`pytest` that resolves outside `.venv`.
- **Sandbox execution**: All tasks, tests, and shell operations must remain entirely 
  within the repository root directory. Never write temporary files, artifacts, or 
  test outputs outside the project tree (e.g., do not use `/tmp/`). If a test or 
  tool requires writing to disk, use a local temporary file or directory within 
  the workspace root.
- Code must pass `ruff`, `black`, and `mypy`.
- Every task ships its own tests (see each task's "Output"). No untested code.

## The matrix is accessed ONLY through matvecs
`A` is reached exclusively via `matvec(Ω) = AΩ` and `rmatvec(Ψ) = A*Ψ`. Never
assemble a dense `A` inside the compressor. The only place a dense block is
formed is the `MockGF` test double and the dense near-field leaf blocks.

## Shape conventions — THE #1 source of bugs
- `dof_row = d`, `dof_col = d - 1`  (so 2D: 2×1 per patch pair; 3D: 3×2).
- A single geometric cluster tree over the `N` patch centroids serves as BOTH
  the row tree and the column tree. All combinatorics (neighbors, interaction
  lists, admissibility, coloring, peeling) run on the `N` boxes and are
  unchanged by dof multiplicity.
- **Patch-major flattening**: each box expands to scalar indices with `dof_row`
  consecutive rows / `dof_col` consecutive cols per patch, block-interleaved. A
  far-field block is `(dof_row·|α|) × (dof_col·|β|)`.
- The flattened operator maps `R^{dof_col·N} → R^{dof_row·N}`: **2N×N** in 2D,
  **3N×2N** in 3D. `A` and `A*` therefore have DIFFERENT domain/range sizes —
  row sampling and column sampling are not interchangeable. Respect this
  everywhere; do not assume square.

## Admissibility — geometric, paper-faithful
- Strong admissibility: a block `(α,β)` is admissible iff
  `dist(α,β) ≥ η · max(diam α, diam β)`, equivalently the boxes are in each
  other's interaction lists.
- The `1/r^d` physics decay is used ONLY to choose/sanity-check `η`. Never use a
  block-norm threshold as a standalone admissibility test — it breaks the
  level-nested structure peeling depends on.
- At every level the admissible (interaction-list) and inadmissible (neighbor)
  blocks must form a COMPLETE, DISJOINT cover. Tests should assert this.

## Scope
- Format is **non-uniform H¹ only**. Uniform H¹ (§4.2) and H² (§4.3) are OUT of
  scope — do not implement them.
- Build order: the fixed periodic test-matrix path (Stages 0–6) first and fully
  validated. Graph coloring (Stage 7) is a drop-in optimization behind a flag
  (`sampling="fixed" | "coloring"`); it must match fixed-path accuracy while
  issuing strictly fewer matvecs.

## Testing rule — do not fake rank structure
A random dense matrix has full-rank blocks and CANNOT validate a compressor.
Low-rank/admissible-block tests must use blocks from the smooth `MockGF` kernel
whose singular values genuinely decay. Near-diagonal blocks must NOT be low
rank. Error is measured with the power-method relative-error utility through
matvec/rmatvec only.

## Git & GitHub flow
- Each task runs on its own branch `task/<id>` and a **draft PR** opened off
  `main`; the PR body states what is going to be done.
- The implementer NEVER touches git. The committer makes **one commit per
  implement/review round** and pushes it to the task branch — never to `main`,
  never force-push.
- The reviewer reads the branch diff (`git diff main...HEAD`); it does not commit
  and does not post to GitHub.
- On approval the pipeline posts a final "what was implemented and how" comment
  and marks the PR **ready for review**. **A human merges the PR to `main`** — no
  agent merges, and nothing is pushed directly to `main`.
