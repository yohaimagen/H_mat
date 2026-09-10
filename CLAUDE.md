# CLAUDE.md — project conventions

Black-box randomized H-matrix compression of elastostatic Green's Function
matrices, after Levitt & Martinsson (2024). Resolve C.x in
`BP_COLORING_PLAN.md`, R.x in `REALGF_PLAN.md`, F.x in `FIXPLAN.md`, and
historical numeric ids in `plan.md`. C.0 activated `BP_COLORING_PLAN.md` as
the execution sequence; implement strictly one active task in `tasks.txt`.

## Tooling
- Python, packaged with `pyproject.toml`. A project virtualenv lives at `.venv/`
  (Python 3.12, `gfcompress` installed editable). **Always run tooling through
  it** — the bare shell `python`/`pytest` resolves to a different (conda)
  interpreter and will not see the project:
  - tests: `.venv/bin/pytest -q`
  - lint/format/type: `.venv/bin/ruff check .`, `.venv/bin/black .`, `.venv/bin/mypy`
  You may `source .venv/bin/activate` if you prefer, but never invoke a bare
  `python`/`pytest` that resolves outside `.venv`.
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

## Admissibility — dyadic, paper-faithful
- The dyadic interaction-list partition is the production far-field gate;
  neighbor lists define the near field. `is_admissible` and `eta` are
  diagnostics only and do not add or remove production pairs.
- The `1/r^d` physics decay may sanity-check `eta`; never use a block-norm
  threshold as an admissibility rule — it breaks the level-nested structure
  peeling depends on.
- At an intermediate level, neighbors plus interactions are a complete,
  disjoint cover only of children of the parent-neighborhood. The global exact
  cover is interactions across levels plus leaf neighbors. Tests should assert
  both the appropriate local and global properties.

## Scope
- Format is **non-uniform H¹ only**. Uniform H¹ (§4.2) and H² (§4.3) are OUT of
  scope — do not implement them.
- Build order: accept the C.11 fixed BP3/BP7 baselines before coloring work.
  Coloring is a schedule optimization behind `sampling="fixed" | "coloring"`:
  it must preserve the partition and match fixed-path quality. Compare measured
  costs; retain fixed whenever it is cheaper or tied. Strict savings are an
  experimental result, not a universal requirement.

## Testing rule — do not fake rank structure
A random dense matrix has full-rank blocks and CANNOT validate a compressor.
Low-rank/admissible-block tests must use blocks from the smooth `MockGF` kernel
whose singular values genuinely decay. Near-diagonal blocks must NOT be low
rank. Error is measured with the power-method relative-error utility through
matvec/rmatvec only.

## Git & GitHub flow
- Each task runs on its own branch `task/<id>` and a **draft PR** opened off
  `main`; the PR body states what is going to be done.
- The implementer NEVER touches git. The orchestrator (`/task`) makes **one commit
  per implement/review round** and pushes it to the task branch — never to `main`,
  never force-push. It stages only the files the implementer names, never
  `git add -A`, so unrelated in-progress edits are never swept into a task PR.
- The reviewer reads the branch diff (`git diff main...HEAD`); it does not commit
  and does not post to GitHub. On approval it also writes the final PR summary,
  which the orchestrator posts verbatim.
- `.claude/hooks/guard.sh` (PreToolUse on Bash) is the hard stop for "no merges,
  no writes to `main`", so agent separation is not what enforces that rule.
- On approval the pipeline posts a final "what was implemented and how" comment
  and marks the PR **ready for review**. **A human merges the PR to `main`** — no
  agent merges, and nothing is pushed directly to `main`.
- Every final PR comment **opens with a `## Plain-language summary`** section,
  before the technical breakdown. Audience: a scientist who knows linear algebra
  and geophysics but has not read Levitt & Martinsson and is not tracking this
  repo's task graph. Two or three short paragraphs, no bullet lists.
  - Say what this task actually *does* in the compression scheme, in words:
    what quantity it computes, and what that quantity is for. Prefer "the
    block's dominant row directions" over "`V_{α,β} = qr(Z(I_β,:), k)`". Introduce
    a symbol only when it earns its place, and gloss it when you do.
  - Put it in context: which part of the overall algorithm this is, what had to
    exist before it, and what it unlocks next. The reader should learn where the
    piece sits without reconstructing the plan.
  - State what is *not* yet true — the honest limits (accuracy achieved, what
    remains unimplemented, any assumption the tests lean on). A summary that
    reads as unqualified success when the numbers are marginal is a failure of
    this section, not a polish issue.
