# Tiered multi-agent pipeline for the GF compression plan

## Layout (drop into your repo root)
```
your-repo/
├── PLAN.md                    # your implementation plan (the doc you have)
├── CLAUDE.md                  # project conventions (see below)
├── tasks.txt                  # task ids in dependency order, one per line
├── run_plan.sh                # headless driver
└── .claude/
    ├── agents/
    │   ├── implementer.md      # model: sonnet — writes code+tests, no git
    │   ├── reviewer.md         # model: opus   — read-only, runs diff+tests
    │   └── committer.md        # model: haiku  — only agent that pushes
    └── commands/
        └── task.md             # /task <id> — orchestrator loop
```

## The loop (per task)
implementer (Sonnet) → reviewer (Opus) → [fix loop, max 3] → committer (Haiku)

- Subagents can't spawn subagents, so the loop lives in the orchestrator (the
  main Opus session driven by `/task`).
- Handoff is via the **working tree**, not context: implementer leaves changes
  dirty, reviewer reads `git diff HEAD`, committer makes ONE commit per task.
- Tool scoping is the safety layer: implementer can edit but not push, reviewer
  is read-only, committer is the only one with git-write.

## Run it
Interactive (one task at a time, you stay in control):
```
claude --model claude-opus-4-8
> /task 3.2
```

Headless (whole plan):
```
mkdir -p logs
./run_plan.sh tasks.txt
```

## CLAUDE.md should contain
Project conventions every agent inherits: package layout, that tests use
`pytest`, the patch-major flattening rule, rectangular shapes (2N×N / 3N×2N),
and "tests must use smooth-kernel low-rank blocks, never random dense matrices."
Keep it short — it's loaded into every agent.

## Verify model tiering actually works
There was a reported bug (#44385) where the frontmatter `model:` was ignored.
After your first `/task` run, check `/cost` or the per-subagent model indicator
to confirm the implementer really ran on Sonnet and the committer on Haiku. The
`CLAUDE_CODE_SUBAGENT_MODEL` env var in run_plan.sh is the fallback.

## Notes
- Model strings (`claude-opus-4-8`, `claude-sonnet-4-6`) — adjust to whatever is
  current when you run this.
- For independent tasks you could parallelize with `isolation: worktree` on the
  implementer, but the review loop is sequential per task, so start serial.
