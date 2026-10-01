---
name: codex-task
description: Run an H_mat implementation-plan task or explicitly requested sequential batch through implementation, task review, and final batch review, with a draft PR and human merge. Use for task execution such as C.1 or C.0 through C.11; not for planning discussions or configuring the workflow.
---

# Codex task workflow

Run exactly the requested task or explicitly requested task range in this repository.
Example: `$codex-task C.0` or `C.0 through C.11`. Do not extend the requested range.
This is the Codex counterpart to the Claude `/task` command. Do not invoke
`run_plan.sh` (it launches Claude), or edit `.claude/` to run this workflow.

## Resolve and prepare

- Resolve the exact task: C.x -> `BP_COLORING_PLAN.md`; R.x -> `REALGF_PLAN.md`;
  F.x -> `FIXPLAN.md`; numeric historical ids -> `plan.md`. C.0 activates the
  C-series as the execution sequence; the remaining R and historical Stage 7
  entries are preserved for history and must not be dispatched in parallel.
  Read the task block, applicable stage gate, scope/contracts, and `AGENTS.md`.
- Verify prerequisites landed on main, rather than relying on a task-branch
  completion marker alone. C.0 is the activation task and may run before C-series
  ids appear in `tasks.txt`. For an explicitly requested cumulative batch, an
  approved prerequisite on the batch branch permits the next requested task;
  intermediate human merges are not required.
- Inspect the branch, existing PR, and dirty paths. Reuse the requested task's
  branch/PR when resuming. Otherwise use `task/<id>` off main, with an isolated
  checkout when necessary to preserve ongoing work. Do not discard or stash
  others' changes, and never silently stage overlapping unrelated edits.
- For an isolated checkout, make the project environment and required datasets
  available without copying multi-gigabyte data. Verify the editable package
  resolves to that checkout before testing. If this cannot be established,
  surface the concrete checkout/environment blocker rather than testing another
  checkout and claiming success.
- Check baseline tests for code tasks and record existing failures. Documentation
  tasks use relevant validation. Use the project `.venv/` throughout.

## Delegate with bounded context

Explicitly use the custom `implementer` and `reviewer` agents from
`.codex/agents/`. The implementer is GPT-5.6 Terra / medium; the per-task reviewer
is GPT-5.6 Sol / high. The final `batch-reviewer` is GPT-6 Astra / high and runs
after every task in the requested range has task-review approval. A single task
is a range of one. Do not silently substitute models. Agents must not delegate.

Give each agent a fresh, concise task message, not the whole conversation:
task id, exact plan path, relevant task text/contracts, working directory,
branch/base revision, allowed files or module responsibility, dirty paths to
preserve, and relevant paper sections. Read the role file if the custom role
has not loaded; use its instructions with explicit model/effort settings when
the runtime supports that fallback. Otherwise report the loading problem.
If a loaded role has stale hard-coded model settings, use a default agent with
the role file's instructions and explicit requested model/effort instead; do
not dispatch the stale role. Editing a role file does not reload a running agent.

Use no inherited history when the delegation API supports `fork_turns="none"`.
Include enough task context for the agent to work independently. Reuse the
implementer for corrections and the reviewer for follow-up reviews. Wait on
agent events; do not repeatedly poll unchanged progress or duplicate their work.

## Implementation, review, and git ownership

1. The orchestrator owns all git/GitHub mutations. Open a draft PR against main
   describing intended scope/acceptance; create an initial task-branch planning
   commit if needed to open it. Use structured arguments or body files for PR
   text. Follow existing configured git identity; do not invent an identity.
2. Dispatch the implementer. Check its named paths against task scope, then
   stage only those paths, inspect the staged diff, commit one implementation
   round, and push explicitly to the task branch. Never push to main, force-push,
   merge, or use `git add -A` / `git add .`.
3. Dispatch the independent reviewer against the committed task diff and exact
   plan. Supply test evidence but let the reviewer validate independently.
   Ensure unrelated working-tree edits cannot masquerade as reviewed code.
4. On CHANGES REQUESTED, send the findings to the implementer, commit the scoped
   fixes, and obtain a fresh reviewer verdict. Do not auto-approve test-only or
   configuration-only fixes. Repeat until explicit approval. After three
   unsuccessful correction rounds, diagnose the underlying issue and document
   a justified adjustment within the approved scope before continuing. Stop
   only for a real external blocker or a decision requiring user input; never
   waive an acceptance gate to move on.
5. On explicit APPROVED, use the reviewer's FINAL SUMMARY as the PR comment.
   Mark the task done in `tasks.txt` on its branch when the entry exists, with
   the reviewed commit id and date. Stage that bookkeeping file separately.
   A done marker records task-review approval, not final batch approval. For a
   requested batch, continue to the next task on one cumulative branch and one
   aggregate draft PR. Review each task from the prior approved revision and
   check its interaction with the cumulative changes. Preserve the starting
   revision, inherited edits, reviewed revisions, findings, checks, and next
   action in a resumable progress ledger. Keep the PR draft.
6. After the last requested task has Sol approval, dispatch `batch-reviewer`
   against the entire series from its recorded starting revision to current HEAD,
   also providing the cumulative diff against main and any inherited changes.
   Supply the exact plan/range, per-task approval ledger, relevant paper sections,
   acceptance evidence, and benchmark records. Astra independently reviews the
   integrated result and every required batch gate.
7. Route batch findings to the Terra implementer with bounded file/task ownership.
   Commit each fix round, obtain Sol approval for the affected task changes,
   then obtain Astra's fresh whole-batch verdict. Repeat until Astra explicitly
   approves; test/configuration fixes also require re-review. Any later change
   invalidates the prior approval for the affected scope. Only then post Astra's
   FINAL SUMMARY verbatim, mark the aggregate PR ready, and report it for human
   merge. Do not merge or push directly to main.

Respect actual sandbox/tool approvals. The imported `.codex/hooks.json` must
not be assumed to enforce git restrictions unless its runtime activation has
been verified. This skill preserves the no-merge/no-main-write workflow; it
does not claim to provide an OS-enforced security boundary.

## Usage accounting and completion

When a usage-limits tool is available, take a snapshot before delegation and
after the full implementation/review cycle. Report percentage-point changes
as account-wide observations, not exact per-task billing: other activity or a
reset can affect them. If a reset occurs, do not subtract across windows.

Finish with task id, configured agent models, correction rounds, PR link,
validation, preserved unrelated edits, usage observations when available, and
the fact that the human performs the merge. Include the final batch verdict and
reviewed revision. Do not launch tasks outside the requested range.
