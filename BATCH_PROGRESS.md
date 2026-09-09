# C.0-C.5 cumulative batch ledger

## Batch identity

- Authorized range: C.0 through C.5 only.
- Branch: `codex/c0-c5-fixed-foundations`.
- Aggregate draft PR: https://github.com/yohaimagen/H_mat/pull/28.
- Dataset path: `real_gfs` links to `/Users/ymagen/work/H_mat/real_gfs`, used read-only.
- Starting batch revision: `b0a67f678077083c6c4b416ea8022b5cb9c19c6c`.
- Main at batch start: `b034c33` (R.1 merged).
- Historical planning revision included by the batch base: `295ab81`.
- R.2 committed revision included by the batch base: `b0a67f6`.
- R.2 PR: https://github.com/yohaimagen/H_mat/pull/27 (open; unchanged by this batch).

## Inherited work and preservation boundary

R.2's committed tree-dimension/PCA changes are inherited prerequisite work.
Uncommitted R.2 fixes are intentionally preserved for C.2: at C.0 start they
are `src/gfcompress/geometry.py` (+21/-12 lines) and `tests/test_geometry.py`
(+49/-8 lines). They are not C.0 behavior changes and must not be staged as
C.0. The original inherited patch is retained at
`/private/tmp/hmat-c0-c5-inherited-r2.patch`.

The inherited `.codex/hooks.json` and `.codex/hooks/` are untracked imports and
outside C.0's staged scope. Their configuration points to
`/Users/ymagen/work/H_mat/.codex/hooks/guard.sh`, not this worktree, so hook
activation has not been established and is not claimed as enforcement.

## Active routing and supersession

- C.x resolves to `BP_COLORING_PLAN.md`; R.x to `REALGF_PLAN.md`; F.x to
  `FIXPLAN.md`; numeric historical ids to `plan.md`.
- C.0 activates the C-series ordering in `tasks.txt`.
- Remaining R.2-R.8 and historical Stage 7 entries remain recorded but are
  superseded execution paths. Do not run them alongside C.0-C.18.
- C.0 documentation reconciles the dyadic interaction partition, local versus
  global coverage, genuine truncation plus exact-recovery tests, and the
  deterministic coloring fallback that prefers fixed on equal measured cost.

## Task review ledger

| Task | Implementer revision | Sol review | Validation | Next action |
| --- | --- | --- | --- | --- |
| C.0 | 886dbb7 | APPROVED, no fixes | routing/TOML/ignore/diff validators passed | proceed C.1 |
| C.1-C.5 | not started | not started | not run | await preceding Sol approval |

## Baseline and usage record

Working-tree baseline (includes preserved dirty R.2 corrections): 305 passed in
109.23s with single-thread BLAS, NumPy 2.4.6 / SciPy 1.17.1. Ruff and Black pass;
mypy passes 19 source files. Initial NumPy 2.5.3 stubs failed against the
project Python 3.10 type target; environment aligned with source versions, no
project configuration workaround. Editable import verified from this checkout.
This is not committed-revision production evidence for C.0. Initial account-wide usage snapshot: 11%
primary and 23% weekly. Record subsequent snapshots as account-wide observations.
