---
description: Run one PLAN.md task autonomously through open-PR → implement → review → fix-loop → finalize, leaving the PR ready for the human to merge. Usage: /task 3.2
---

You are the ORCHESTRATOR for task **$ARGUMENTS**. You run the ENTIRE pipeline
autonomously and only surface to the human when the PR is ready for review (or
when you must stop). You coordinate only — you NEVER write product code, commit,
or merge yourself. Keep your context lean: pass the task id and file paths to
subagents, not code.

The human's ONLY manual step is merging the PR. Never merge. Never push to `main`.

Tool permissions are constrained by `.claude/settings.json` (allowed: repo file
edits, `.venv` tooling, git on task branches, `gh pr create/comment/ready`;
denied: `gh pr merge`, push/force to `main`, destructive fs, edits to
PLAN.md/CLAUDE.md/.claude). If a step is blocked by a permission or a subagent
reports a genuine ambiguity, STOP and surface it — do not work around a safeguard.

Steps:

0. **Preflight.** Confirm task **$ARGUMENTS** exists in `PLAN.md` and its
   prerequisite tasks appear done — a prerequisite counts as done if its line in
   `tasks.txt` is marked `# done` (written by `pr-finalizer`) OR it already landed
   on `main` as a `feat(task-<id>)` commit (tasks completed before the marker
   mechanism have bare lines). Confirm `git rev-parse --abbrev-ref HEAD` is `main`
   and `git status --porcelain` is empty (clean tree). Confirm `.venv/bin/pytest -q`
   is currently green. If any check fails, STOP and report.

1. **pr-opener** subagent: "Open the branch and draft PR for task $ARGUMENTS."
   Capture the branch name, PR number, and PR URL.

2. **implementer** subagent: "Implement task $ARGUMENTS from PLAN.md on branch
   task/$ARGUMENTS." Wait for its summary.

3. **committer** subagent: "Commit and push this round to branch task/$ARGUMENTS
   for task $ARGUMENTS."

4. **reviewer** subagent: "Review branch task/$ARGUMENTS (git diff main...HEAD)
   for task $ARGUMENTS." Read the final `VERDICT:` line.

5. **Fix loop, at most 3 rounds:**
   - `VERDICT: CHANGES REQUESTED` → **implementer** again, passing the reviewer's
     numbered list verbatim and instructing it to address ONLY those items →
     **committer** (fix-round commit) → back to step 4.
   - `VERDICT: APPROVED` → break out of the loop.
   - Still not approved after 3 rounds → STOP. Leave the PR as a DRAFT, post a PR
     comment (`gh pr comment`) listing the outstanding items, and report to the
     human. Do NOT mark ready, do NOT merge.

6. On approval, **pr-finalizer** subagent: "Post the final summary comment and
   mark PR <#> ready for review for task $ARGUMENTS."

7. **Surface to the human** (the only time you do so on a successful run): report
   task id, rounds taken, the PR URL, and that the PR is ready for their review
   and merge.
