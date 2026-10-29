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
| C.12 | pending | pending | pending | implement C.12 |
| C.13 | pending | pending | pending | gated on C.12 approval |
| C.14 | pending | pending | pending | gated on C.13 approval |
| C.15 | pending | pending | pending | gated on C.14 approval |
| C.16 | pending | pending | pending | gated on C.15 approval |

Final GPT-6 Astra batch review is gated on per-task approval for C.12-C.16.
