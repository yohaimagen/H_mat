---
description: Run one PLAN.md task autonomously through open-PR → implement → review → fix-loop → finalize, leaving the PR ready for the human to merge. Usage: /task 3.2
---

You are the ORCHESTRATOR for task **$ARGUMENTS**. You run the ENTIRE pipeline
autonomously and surface to the human exactly ONCE — when the PR is ready for
their merge (or when you must stop).

Only TWO steps are subagents: **implementer** and **reviewer**. Every git and
`gh` command below you run YOURSELF, inline. Do not delegate them — a cold
subagent boot costs more than the three commands it would run. You never write
product code and never merge.

`.claude/settings.json` allows every tool, so the pipeline never stops for a
permission prompt. The hard stop is `.claude/hooks/guard.sh` (PreToolUse on
Bash): it blocks merges, any push touching `main`/`master`, force pushes, and
destructive filesystem commands, whatever shape the command takes. If the guard
blocks a step, or a subagent reports a genuine ambiguity, STOP and surface it —
never work around a safeguard. Otherwise never ask the human anything mid-run.

Git identity: this repo has no `user.name`/`user.email` configured. Commit with
`git -c user.name=yohaimagen -c user.email=mayochay@gmail.com commit ...`.

## 0. Preflight

Confirm task **$ARGUMENTS** exists in `PLAN.md` (Stage F tasks carry a one-line
entry in `plan.md` and their full block in `FIXPLAN.md` — read the full block)
and its prerequisites appear done: a prerequisite counts as done if its line in
`tasks.txt` is marked `# done` OR it landed on `main` as a `feat(task-<id>)`
commit. Confirm HEAD is `main` and `.venv/bin/pytest -q` is green.

The working tree does NOT have to be clean — you stage by explicit path (step 2),
so unrelated in-progress edits are simply left alone. Note them in your final
report. If a preflight check fails, STOP and report.

## 1. Open branch and draft PR (inline)

```
git checkout -b task/$ARGUMENTS
git -c user.name=yohaimagen -c user.email=mayochay@gmail.com commit --allow-empty -m "chore(task-$ARGUMENTS): open PR"
git push -u origin task/$ARGUMENTS
gh pr create --draft --base main --head task/$ARGUMENTS --title "task-$ARGUMENTS: <short title>" --body-file <plan>
```
Body = "what is going to be done": a 2–5 sentence scope summary, then a checklist
of the task's Steps and expected Output. Write it with `--body-file` (never a
heredoc with backticks inline). Capture the PR number and URL.

## 2. implementer → commit (inline)

Give the implementer the task id, the branch, and where the full task block lives.
It returns a summary naming **the files it changed**.

Stage EXACTLY those paths — never `git add -A`, never `git add .`:
```
git add -- <path> <path> ...
git status --porcelain    # verify nothing else is staged
git -c user.name=... -c user.email=... commit -m "feat(task-$ARGUMENTS): <subject>"
git push
```
Anything else in the working tree is someone else's in-progress work: leave it,
do not revert it, mention it in the final report. If the implementer names a file
outside the task's declared Files list, that is a scope question — ask the
implementer to justify or drop it; do not commit it silently.

Message: `feat` for new functionality, `test` for test-only, `chore` for
scaffolding; `fix(task-$ARGUMENTS): address review — <note>` for fix rounds.
Subject under ~72 chars.

## 3. reviewer

Give it the task id and branch. It reviews `git diff main...HEAD` against
`PLAN.md`/`FIXPLAN.md` AND the paper, and ends with a `VERDICT:` line.

## 4. Fix loop (at most 3 rounds)

- `CHANGES REQUESTED` → **implementer** again with the reviewer's numbered list
  verbatim, "address ONLY these items" → commit (step 2) → back to step 3,
  **except**: if the fix round's `git diff --name-only` touched no file under
  `src/`, skip the re-review — run `.venv/bin/pytest -q` yourself, and if green
  treat the prior verdict's remaining items as closed. A test-only or config-only
  fix does not need the paper re-read.
- `APPROVED` → leave the loop.
- Not approved after 3 rounds → STOP. Leave the PR a DRAFT, post a comment
  listing what is outstanding, report to the human. Do not mark ready.

## 5. Finalize (inline)

The reviewer's approval carries a `FINAL SUMMARY:` block — the PR comment body.
Do NOT re-derive it, do not re-run the suite, do not re-read the paper.

1. Post it verbatim: `gh pr comment <PR#> --body-file <file>`.
2. Mark the task done in `tasks.txt` on the task branch (it reaches `main` only
   through the human's merge). The `Edit`/`Write` tools are denied on `tasks.txt`
   by design and BSD `sed -i` is a trap — use Python:
   ```
   SHA=$(git rev-parse --short HEAD)
   .venv/bin/python3 - <<'PY'
   import re, subprocess
   tid, sha = "$ARGUMENTS", "SHA_HERE"
   date = subprocess.run(["git","show","-s","--format=%cd","--date=short","HEAD"],
                         capture_output=True, text=True).stdout.strip()
   src = open("tasks.txt").read()
   new, n = re.subn(rf"^(\s*){re.escape(tid)}\s*$", rf"\g<1>{tid}  # done {sha} {date}",
                    src, flags=re.M)
   assert n == 1, f"expected exactly one bare '{tid}' line, matched {n}"
   open("tasks.txt","w").write(new)
   PY
   ```
   Then `git add tasks.txt`, commit `chore(task-$ARGUMENTS): mark done in tasks.txt`,
   push.
3. `gh pr ready <PR#>`, then `git checkout main`.

## 6. Report to the human (the only time you surface)

Task id, rounds taken, PR URL, what the reviewer caught, any working-tree files
you deliberately left uncommitted, and that the PR awaits their merge.
