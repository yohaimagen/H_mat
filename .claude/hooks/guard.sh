#!/usr/bin/env bash
# PreToolUse(Bash) guard for the /task pipeline.
#
# Permissions allow every tool so the loop never stops to ask. This hook is the
# hard stop instead: exit 2 blocks the call before permission rules are even
# evaluated, and it matches anywhere in the command string, so it survives
# compound commands, heredocs and `git -c ...` prefixes that a prefix-matching
# deny rule misses.
#
# Rule: agents never merge and never write to main. A human does that.

cmd=$(jq -r '.tool_input.command // ""')

block() {
  printf 'BLOCKED by .claude/hooks/guard.sh: %s\nAgents never merge or write to main; report to the human instead.\n' "$1" >&2
  exit 2
}

deny() { grep -Eq "$1" <<<"$cmd" && block "$2"; }

deny '\bgh\b.*\bpr\b.*\bmerge\b'                    'gh pr merge'
deny '\bgit\b.*\bpush\b.*(\bmain\b|\bmaster\b)'     'push targeting main/master'
deny '\bgit\b.*\bpush\b.*(\s-f\b|--force)'          'force push'
deny '\bgit\b.*\breset\b.*--hard'                   'git reset --hard'
deny '\bgit\b.*\bclean\b.*-[a-zA-Z]*[fd]'           'git clean'
deny '\brm\b[^|;&]*\s-[a-zA-Z]*r'                   'recursive rm'
deny '\bsudo\b'                                     'sudo'
deny '\bgh\b.*\brepo\b.*\bdelete\b'                 'gh repo delete'

exit 0
