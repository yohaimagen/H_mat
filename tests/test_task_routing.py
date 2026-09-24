"""The active task list must not redispatch superseded plans."""

from pathlib import Path

import pytest


def test_c_series_is_active_and_historical_r_and_stage_7_are_comments() -> None:
    lines = Path("tasks.txt").read_text().splitlines()
    active = {line.split()[0] for line in lines if line and not line.startswith("#")}

    assert {"C.0", "C.11"} <= active
    assert not active.intersection({"R.2", "R.3", "R.4", "R.5", "R.6", "R.7", "R.8"})
    assert not active.intersection({"7.1", "7.2", "7.3", "7.4", "7.5"})


@pytest.mark.parametrize(
    ("task_id", "plan"),
    [
        ("C.6", "BP_COLORING_PLAN.md"),
        ("R.3", "REALGF_PLAN.md"),
        ("F.2", "FIXPLAN.md"),
        ("3.2", "plan.md"),
    ],
)
def test_task_command_routes_each_task_series_to_its_authoritative_plan(
    task_id: str, plan: str
) -> None:
    command = Path(".claude/commands/task.md").read_text()
    normalized = " ".join(command.split())

    assert "as `task_plan`" in command
    assert f"`{task_id}` resolves `task_plan` to `{plan}`" in normalized
    assert "concrete task block's Steps and expected Output from `task_plan`" in command
    assert "full task block from\n`task_plan`" in command
    assert "against the concrete task block in `task_plan`" in command
    assert "`PLAN.md`/`FIXPLAN.md`" not in command
