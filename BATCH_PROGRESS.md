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
| C.1 | 93bee82 + fbe0ac8 + c0d375f + af5d0e7 + e3257d9 + c50f1a1 | APPROVED after 4 corrections | 118 focused plus full pytest; ruff/black/mypy pass | proceed C.2 |
| C.2 | 0c8df92 + b1a4b9f + 6604919 | APPROVED after 2 corrections | 340 tests; ruff/black/mypy independently pass | proceed C.3 |
| C.3 | 8113385 + 4eb126e | APPROVED after 1 correction | full suite; ruff/black/mypy independently pass | proceed C.4 |
| C.4 | fd88b43 + 9533703 + 94ef6b4 | APPROVED after 2 corrections | full suite; measurements reproduced; ruff/black/mypy independently pass | proceed C.5 |
| C.5 | not started | not started | not run | start Terra implementation |

## Baseline and usage record

Working-tree baseline (includes preserved dirty R.2 corrections): 305 passed in
109.23s with single-thread BLAS, NumPy 2.4.6 / SciPy 1.17.1. Ruff and Black pass;
mypy passes 19 source files. Initial NumPy 2.5.3 stubs failed against the
project Python 3.10 type target; environment aligned with source versions, no
project configuration workaround. Editable import verified from this checkout.
This is not committed-revision production evidence for C.0. Initial account-wide usage snapshot: 11%
primary and 23% weekly. Record subsequent snapshots as account-wide observations.

After C.0 cycle, account-wide usage was 57% primary / 30% weekly (same primary
reset timestamp 1788973035); change +46/+7 percentage points, not task billing.
C.0 approval summary posted verbatim to aggregate PR #28.

C.1 review findings at fbe0ac8: add committed wrapped period-6/period-3 isolation
in all 1D/2D/3D cases; assert exact 1/sqrt(tree_dim) separation; remove eta-gate
documentation contradictions; explicitly document common-depth adaptation and
correct obsolete hypercube/bounding-box descriptions. Reviewer independent
8^d diagnostics pass; these are acceptance-coverage and documentation gaps.

C.1 follow-up findings at c0d375f: make root-hypercube midpoint/padding finite
and enclosing for extreme and negative coordinates; reconcile the remaining
`realgf.py` and `CLAUDE.md` eta/coverage statements; and finish two helper
docstrings. Sol independently passed the focused/full tests, Ruff, Black, mypy,
and additional sparse-tree coverage diagnostics before requesting corrections.

C.1 follow-up findings at af5d0e7: use overflow-safe Euclidean norms so large
finite cells retain finite diameters and separation ratios; remove the final
deferred-eta implication from the historical `L` proxy discussion; and correct
`d` to `tree_dim` in the builder argument docs. All standard checks passed
before Sol identified these numerical and documentation gaps.

C.1 follow-up finding at e3257d9: the one-ULP root-side search does not
terminate when outward finite padding is impossible at `float64` maximum. After
three correction rounds, the diagnosed issue is the unbounded search strategy;
round 4 replaces it with bounded direct construction and immediate explicit
failure when no finite enclosing padded cell exists.

C.1 received Sol approval at `c50f1a12095c19445f3fad4bb195ee81f8669293`
after correction round 4, with no remaining in-scope findings, questions, or
suggestions. Sol independently passed 118 focused tests (including warnings as
errors), the complete suite, Ruff, Black, mypy, diff checks, and additional
extreme-coordinate probes. C.1's final summary was posted to PR #28.

C.2 review findings at `0c8df929b0eb32f86c90e7d61db2edf203ba6d00`:
normalize singular values/residuals so PCA decisions survive extreme finite
rescaling; return a complete ambient transform and axis report for `N < d`;
correct and interpret several BP occupancy medians/tiny-leaf counts; and add
direct RealGF geometry-versus-dof regression assertions while fixing its stale
public docs. Sol independently passed all 333 tests and static checks and
reproduced the real-data measurements before requesting corrections.

C.2 follow-up at `b1a4b9f4bc4b8415395434114c59fa0c1b572126`
closed all implementation and integration findings. Sol requested only that the
tracked BP3/BP7 projection-residual values be refreshed after the stable
discarded-singular-value calculation replaced the earlier reconstruction-based
measurement. All 340 tests and static checks passed independently.

C.2 received Sol approval at `660491909970e1df2f189a794da54d2d9bb65535`
after correction round 2, with no remaining findings, questions, or
suggestions. The approved evidence includes scale-safe PCA, complete `N < d`
transforms, direct BP3/BP7 geometry-versus-dof integration, all 340 tests and
static checks, and reproduced occupancy/residual measurements. Its final
summary was posted to PR #28.

C.3 review findings at `81133852bfcd5cf2546dfe32336e7e7d5038cf9e`:
reject non-integer operator shape components and missing/non-callable products
before sampling, and update public column/row basis docs from `k` to the shared
per-block `k_eff`. Sol confirmed the Eq. 4.3 implementation and full validation
otherwise pass.

C.4 review findings at `fd88b43`: replace finiteness-only repeated-seed checks
with measured accuracy bounds, remove the older forbidden per-seed monotonic
oversampling assertion, and prevent arbitrary integer seeds from aliasing
modulo `2**64`. Sol reproduced all six recorded measurements and confirmed the
remaining default/stream/traversal contracts and full validation pass.

C.4 follow-up findings at `9533703e7a5c389c88a8b6cbf6a9b6a865d40c26`:
the lossless signed-seed encoding changed the deterministic streams, so refresh
the tracked measurements, and update the repeated-seed test docstring to state
its new absolute-accuracy and aggregate-improvement contract. Sol confirmed
all three prior functional findings are resolved and all validation passes.

C.4 received Sol approval at `94ef6b48e60bc31a3fa503ade53c28cfdc47b5fd`
after correction round 2, with no remaining findings, questions, or
suggestions. Sol independently reproduced every tracked measurement, verified
the accuracy/default/seed/traversal contracts, and passed the full suite plus
Ruff, Black, mypy, and diff checks. Its final summary was posted to PR #28.

C.3 received Sol approval at `4eb126ed89267a017c879b39f769725a9fa4f62e`
after correction round 1, with no remaining findings, questions, or
suggestions. Sol independently verified the shared Eq. 4.3 rank, retained
`k+p` probe width, exact/smooth truncation coverage, pre-sampling validation,
the complete suite, and static checks. Its final summary was posted to PR #28.
