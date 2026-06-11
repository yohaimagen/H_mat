#!/usr/bin/env bash
# Drive the whole PLAN.md task-by-task in headless mode.
# Each task gets a FRESH orchestrator session (fresh context), which itself
# runs the implement -> review -> fix -> commit loop via the /task command.
#
# Usage:  ./run_plan.sh tasks.txt
# where tasks.txt is one task id per line, in dependency order, e.g.:
#   0.1
#   1.1
#   1.2
#   ...
set -euo pipefail

TASKS_FILE="${1:-tasks.txt}"

# Orchestrator + reviewer run on Opus; implementer/committer pinned in their
# own frontmatter. This env var is the reliable fallback for any subagent left
# on `inherit` (and a safety net for the model-field bug).
export CLAUDE_CODE_SUBAGENT_MODEL="claude-sonnet-4-6"

while IFS= read -r task; do
  [[ -z "$task" || "$task" == \#* ]] && continue
  echo "==================== TASK $task ===================="

  claude -p "/task $task" \
    --model claude-opus-4-8 \
    --permission-mode acceptEdits \
    --output-format text \
    | tee "logs/task_${task}.log"

  # Gate: only continue if the suite is green after this task.
  if ! pytest -q >/dev/null 2>&1; then
    echo "!! Suite is RED after task $task — stopping for human review." >&2
    exit 1
  fi
done < "$TASKS_FILE"

echo "All tasks complete."
