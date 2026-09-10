"""The active task list must not redispatch superseded plans."""

from pathlib import Path


def test_c_series_is_active_and_historical_r_and_stage_7_are_comments() -> None:
    lines = Path("tasks.txt").read_text().splitlines()
    active = {line.split()[0] for line in lines if line and not line.startswith("#")}

    assert {"C.0", "C.11"} <= active
    assert not active.intersection({"R.2", "R.3", "R.4", "R.5", "R.6", "R.7", "R.8"})
    assert not active.intersection({"7.1", "7.2", "7.3", "7.4", "7.5"})
