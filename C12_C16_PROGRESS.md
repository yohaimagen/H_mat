# C.12-C.16 cumulative batch ledger

## Batch identity

- Authorized range: C.12 through C.16 only.
- Branch: `task/C.12-C.16`.
- Starting revision: `81ec1a4b645bde8919ecd0b63dbfae7611db8731` (merged C.11 on `origin/main`).
- C.11 prerequisite: merged and scientifically accepted for BP3 and Frontera; BP7 remains negative evidence.
- Worktree: `/Users/ymagen/.codex/worktrees/c11-frontera/H_mat` (clean managed worktree repurposed after C.11 merge).
- Environment: `/Users/ymagen/work/H_mat/.venv` with `PYTHONPATH=src`, resolving the package from this worktree.

## Scope and preservation boundary

This batch implements graph construction, deterministic coloring, colored
admissible probes, colored dense-leaf extraction, and the public coloring path.
It does not run C.17 full-matrix paired experiments or C.18 delivery work.
The dirty primary checkout on branch `task/R.2` is outside this worktree and
must remain untouched.

## Baseline validation

- Pytest: 465 passed, 14 data-dependent skips.
- Ruff: passed.
- Black: 63 files unchanged.
- Mypy: 22 production files passed with incremental caching disabled.

## Task review ledger

| Task | Implementation revision(s) | Sol review | Validation | Next action |
| --- | --- | --- | --- | --- |
| C.12 | `f8f4018` | APPROVED, no fixes | 22 focused; 487 passed/14 skipped; Ruff/Black/mypy pass | proceed C.13 |
| C.13 | `35abe07` | APPROVED, no fixes | 32 focused; 519 passed/14 skipped; Ruff/Black/mypy pass | proceed C.14 |
| C.14 | `08c37a8` | APPROVED, no fixes | 15 focused; 532 passed/14 skipped; Ruff/Black/mypy pass | proceed C.15 |
| C.15 | `dae5dd3` | APPROVED, no fixes | 15 focused; 547 passed/14 skipped; Ruff/Black/mypy pass | proceed C.16 |
| C.16 | pending | pending | pending | gated on C.15 approval |

Final GPT-6 Astra batch review is gated on per-task approval for C.12-C.16.

### C.12 review record

Sol approved `f8f4018713810c28f291f20d69e29b37f1ef3b09` with no
findings. The review confirmed Eq. 4.4/Definition 4.1 semantics, incidence-based
conflict enumeration, complete owner retention, row-side role swapping,
deterministic ordering, honest complexity, and independence of the brute-force
test oracle. C.13 remains the first task that may color these graphs.

### C.13 review record

Sol approved `35abe07af08ed275968cf1e4f8136bcdecc6303c` with no
findings. The review confirmed deterministic Algorithm 2.3 priorities,
independent oracle agreement, proper graph/color validation, actual fixed-probe
accounting, common-width column comparison, and fixed selection on cheaper or
tied schedules. Colored numerical probes remain C.14 scope.

### C.14 review record

Sol approved `08c37a8c9cbd43d47c77154c634a5cf5b2023859` with no
findings. The review confirmed zero-box isolation, pair-specific sketches for
both rectangular sides and repeated unseeded boxes, streaming lifetime,
direct peeled MockGF samples, exact fixed fallback, and schedule/operator-count
agreement. The irregular synthetic path selected 1,624 admissible columns
versus 2,184 fixed with comparable three-seed quality. Dense leaves remain C.15.

### C.15 review record

Sol approved `dae5dd3d0e112fd2aadc98aaaa79015adfd8da3c` with no
findings. The review confirmed neighbor-only ownership, shared identity-slot
isolation, global fixed width, fully peeled residual extraction, exact count
accounting, and positive three-seed near-field quality. The retained rank-4
negative case (5.25046% colored versus 3.03594% fixed) demonstrates that a
cheaper schedule alone does not guarantee accuracy; C.16 must validate its
chosen complete public configurations without relaxing the 5% criterion.
