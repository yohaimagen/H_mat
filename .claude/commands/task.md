---
description: Run one PLAN.md task through implement → review → fix-loop → commit. Usage: /task 3.2
---

You are the ORCHESTRATOR. Run task **$ARGUMENTS** through the full pipeline. You coordinate only — you do NOT write product code or commit yourself. Keep your own context lean: pass the task id and file paths to subagents, not code.

Steps:

1. Confirm task **$ARGUMENTS** exists in `PLAN.md` and that its prerequisite tasks appear already done (the relevant files/tests exist and `pytest -q` is currently green). If a prerequisite is missing, stop and report it.

2. Delegate to the **implementer** subagent: "Implement task $ARGUMENTS from PLAN.md." Wait for its summary.

3. Delegate to the **reviewer** subagent: "Review the working tree for task $ARGUMENTS." Read the final `VERDICT:` line.

4. Loop, at most **3 rounds**:
   - If `VERDICT: CHANGES REQUESTED` → delegate to the **implementer** again, passing the reviewer's numbered list verbatim and instructing it to address only those items. Then go back to step 3.
   - If `VERDICT: APPROVED` → break out of the loop.
   - If still not approved after 3 rounds → STOP, do not commit, and report the outstanding review items to the human. Do not push broken or unreviewed work.

5. On approval, delegate to the **committer** subagent: "Commit and push the approved tree for task $ARGUMENTS." Report the commit hash back.

Output a 3-line final summary: task id, rounds taken, commit hash (or why it stopped).
