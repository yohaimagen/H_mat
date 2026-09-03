# Agentic workflow log

One section per `/task` run. Purpose: measure what the pipeline actually costs
and trim it. Timings are from PR commit timestamps (wall clock) and subagent
telemetry (agent-busy time); the two differ by human-wait.

---

## Task F.1 — uniform-depth tree (PR #12, merged 2026-09-03)

Diff shipped: 5 files, +310/−96. `build_tree.py`, `tree.py`, `test_build_tree.py`,
`test_tree.py`, one fixture line in `test_neighbors.py`.

### Timeline

| UTC | Event | Agent-busy |
|---|---|---|
| 07:19 | pr-opener: branch + draft PR #12 | 1.5 min |
| 07:19–07:46 | implementer round 1 (+ 1-line fixture follow-up) | ~25 min |
| 07:46 | committer → `c961a9c` | 0.8 min |
| 07:46–07:53 | reviewer round 1 → **CHANGES REQUESTED** | 6.2 min |
| 07:53–07:59 | implementer fix round 1 | 5.1 min |
| 07:59 | committer → `095bf6a` | 0.8 min |
| 07:59–08:03 | reviewer round 2 → **CHANGES REQUESTED** | 3.6 min |
| 08:03–09:50 | *blocked on human*: `.claude` scope question | — (~1 h 40 min wait) |
| 09:50–09:55 | implementer nit + committer → `987ae99`, `a2268c3` | 6 min |
| 09:55–09:58 | reviewer round 3 → **APPROVED** | 2.1 min |
| 09:58–10:14 | pr-finalizer: tasks.txt + PR comment + ready | 16 min |
| 11:32 | human merge | — (78 min wait) |

**Agent-busy: ~68 min. Wall clock start→merge: 4 h 12 min.** The 3 h 20 min
delta is entirely human-wait (2 of the 3 touchpoints).

### Cost

| Agent | Invocations | Tokens | Tool calls |
|---|---|---|---|
| implementer | 4 (1 + 3 resumes) | ~114 k | 64 |
| reviewer | 3 | ~88 k | 30 |
| committer | 3 | ~58 k | 79 |
| pr-finalizer | 1 | ~22 k | 18 |
| pr-opener | 1 | ~18 k | 6 |
| **Total** | **12** | **~300 k** | **197** |

### What the pipeline caught

1. **Real regression, reviewer round 1.** The new cell-underflow guard aborted
   the build on any non-bisectable axis — including a zero-span axis, i.e. every
   3D planar fault, this project's primary geometry. At `z = 5000` the tree
   collapsed to root-only, 256 patches in one "leaf", silently, and worse than
   `main`. The implementer's own tests missed it because the 3D fixture used
   `z = 0.3`. **This alone justifies the reviewer step.**
2. **Scope leak, reviewer round 2.** The committer swept uncommitted `.claude/`
   permission+hook edits into the F.1 commit; a "tree builder" PR would have
   silently carried a permissions widening.
3. **Expected fallout, implementer round 1.** The `>= m` → `> m` fix broke a
   stale `test_neighbors` fixture. Correctly flagged rather than papered over.

### Where the time went that shouldn't have

- **pr-finalizer: 16 min** to edit one line of `tasks.txt`, post one comment, and
  flip a draft flag. Worst ratio in the run by far.
- **committer: 79 tool calls / 58 k tokens** across 3 rounds to run
  `git add && git commit && git push`. One round took 61 tool calls.
- **The 1 h 40 min stall** was a human question that a one-line rule would have
  prevented: never stage files outside the task's declared file list.
- **pr-opener: 1.5 min** for two shell commands.

### Proposed trim

Keep **implementer** and **reviewer**. They do the work and the reviewer paid
for itself twice in one run.

Fold **pr-opener**, **committer**, and **pr-finalizer** into the orchestrator —
they are 5–10 shell commands each, and delegating them costs a cold agent boot,
a context re-derivation, and (for the committer) an agent with enough latitude to
stage the wrong files. The safety argument for a separate committer ("implementer
never touches git") is now enforced at the tool level by `.claude/hooks/guard.sh`,
which is a stronger guarantee than agent separation ever was.

Add to the commit step: **stage only the task's declared files**; anything else in
the working tree is left alone and reported.

Expected: 12 agent invocations → ~5, ~300 k → ~200 k tokens, ~68 → ~45 min
agent-busy, 3 human touchpoints → 2 (merge PR, answer only genuine ambiguities).
