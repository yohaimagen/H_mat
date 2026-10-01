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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_FILE="${1:-$REPO_ROOT/tasks.txt}"
if [[ "$TASKS_FILE" != /* ]]; then
  TASKS_FILE="$(pwd)/$TASKS_FILE"
fi
cd "$REPO_ROOT"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
PYTEST_BIN="${PYTEST_BIN:-$REPO_ROOT/.venv/bin/pytest}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

# Orchestrator + reviewer run on Opus; implementer/committer pinned in their
# own frontmatter. This env var is the reliable fallback for any subagent left
# on `inherit` (and a safety net for the model-field bug).
export CLAUDE_CODE_SUBAGENT_MODEL="claude-sonnet-4-6"

mkdir -p "$LOG_DIR"

tasks=()
while IFS= read -r raw_task || [[ -n "$raw_task" ]]; do
  [[ "$raw_task" =~ \#[[:space:]]*done([[:space:]]|$) ]] && continue
  task="${raw_task%%#*}"
  task="${task#"${task%%[![:space:]]*}"}"
  task="${task%"${task##*[![:space:]]}"}"
  [[ -z "$task" ]] && continue
  [[ "$task" =~ ^([0-9]+|[CFR])\.[0-9]+$ ]] || {
    echo "Invalid task id: $task" >&2
    exit 2
  }
  tasks+=("$task")
done < "$TASKS_FILE"

for task in "${tasks[@]}"; do
  echo "==================== TASK $task ===================="

  "$CLAUDE_BIN" -p "/task $task" \
    --model claude-opus-4-8 \
    --permission-mode acceptEdits \
    --output-format text \
    | tee "$LOG_DIR/task_${task}.log"

  # Gate: only continue if the suite is green after this task.
  if ! "$PYTEST_BIN" -q >/dev/null 2>&1; then
    echo "!! Suite is RED after task $task — stopping for human review." >&2
    exit 1
  fi
done

echo "All tasks complete."
