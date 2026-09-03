#!/usr/bin/env bash
# Self-check for guard.sh. Run: bash .claude/hooks/guard_test.sh
G="$(cd "$(dirname "$0")" && pwd)/guard.sh"
fail=0
while IFS=$'\t' read -r want cmd; do
  [ -z "$cmd" ] && continue
  jq -Rn --arg c "$cmd" '{tool_name:"Bash",tool_input:{command:$c}}' | bash "$G" >/dev/null 2>&1
  rc=$?
  got=ALLOW; [ "$rc" -eq 2 ] && got=BLOCK
  if [ "$got" = "$want" ]; then printf 'ok   %-5s %s\n' "$got" "$cmd"
  else printf 'FAIL want=%s got=%s  %s\n' "$want" "$got" "$cmd"; fail=1; fi
done <<'CASES'
ALLOW	git status
ALLOW	.venv/bin/pytest -q 2>&1 | tail -5
ALLOW	git -c user.name=x -c user.email=y commit -m "wip"
ALLOW	wc -l src/*.py && echo done
ALLOW	git push -u origin task/F.1
ALLOW	gh pr create --draft --base main --head task/F.1 --title t
ALLOW	gh pr ready 12
ALLOW	git checkout main
ALLOW	rm build/tmp.o
BLOCK	git checkout main && git push origin main
BLOCK	git push origin HEAD:main
BLOCK	gh pr merge 12 --squash
BLOCK	git push --force origin task/F.1
BLOCK	git push -f origin task/F.1
BLOCK	rm -rf build
BLOCK	git reset --hard origin/main
BLOCK	sudo rm /etc/hosts
CASES
[ $fail -eq 0 ] && echo "guard.sh: all cases pass"
exit $fail
