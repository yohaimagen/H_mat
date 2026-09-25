"""The headless driver dispatches only clean, active task identifiers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_plan.sh"


def _commands(tmp_path: Path) -> tuple[Path, Path, Path]:
    calls = tmp_path / "calls"
    claude = tmp_path / "claude"
    claude.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$2\" >> {calls}\n")
    claude.chmod(0o755)
    pytest = tmp_path / "pytest"
    pytest.write_text("#!/usr/bin/env bash\nexit 0\n")
    pytest.chmod(0o755)
    return claude, pytest, calls


def _run(tmp_path: Path, task_text: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    tasks = tmp_path / "tasks.txt"
    tasks.write_text(task_text)
    claude, pytest, calls = _commands(tmp_path)
    env = os.environ | {
        "CLAUDE_BIN": str(claude),
        "PYTEST_BIN": str(pytest),
        "LOG_DIR": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        [str(RUNNER), str(tasks)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, calls


def test_repository_list_dispatches_c6_first(tmp_path: Path) -> None:
    claude, pytest, calls = _commands(tmp_path)
    env = os.environ | {
        "CLAUDE_BIN": str(claude),
        "PYTEST_BIN": str(pytest),
        "LOG_DIR": str(tmp_path / "logs"),
    }
    subprocess.run([str(RUNNER)], cwd=tmp_path, env=env, check=True)

    assert calls.read_text().splitlines()[0] == "/task C.6"


def test_completed_and_comment_lines_launch_nothing(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "# comment\nC.5 # done historical\n\n")

    assert result.returncode == 0
    assert not calls.exists()


def test_inline_active_annotation_dispatches_clean_id(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "  C.6  # ready after review\n")

    assert result.returncode == 0
    assert calls.read_text().splitlines() == ["/task C.6"]


def test_malformed_id_fails_before_any_launch(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "C.6\nnot-a-task\n")

    assert result.returncode == 2
    assert "Invalid task id: not-a-task" in result.stderr
    assert not calls.exists()
