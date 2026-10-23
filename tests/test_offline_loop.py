"""The offline loop wrapper: two real games end to end on CPU with replayed replies."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from arcagent.agent.scheduler import MIN_GAME_S, game_budget

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "offline_loop.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("offline_loop", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_loop_runs_two_games_end_to_end(
    env_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("pyarrow")
    offline_loop = load_script()
    out = tmp_path / "run"
    offline_loop.main(
        [
            "--games",
            "ls20",
            "ft09",
            "--max-actions",
            "200",
            "--env-dir",
            str(env_dir),
            "--replies",
            str(tmp_path / "no-replies"),
            "--out",
            str(out),
        ]
    )
    assert (out / "submission.parquet").is_file()
    for game in ("ls20", "ft09"):
        (log,) = (out / "logs").glob(f"{game}*.jsonl")  # file name carries the full game id
        lines = log.read_text(encoding="utf-8").splitlines()
        assert any(json.loads(line)["type"] == "action" for line in lines)
    printed = capsys.readouterr().out
    assert "ls20" in printed and "ft09" in printed


@pytest.mark.parametrize(("game_seconds", "n_games", "workers"), [(300, 25, 8), (150, 2, 2)])
def test_derived_total_gives_each_first_wave_game_its_seconds(
    game_seconds: float, n_games: int, workers: int
) -> None:
    offline_loop = load_script()
    total = offline_loop.derive_total_seconds(game_seconds, n_games, workers)
    assert game_budget(0.0, total, n_games, workers) == pytest.approx(game_seconds)


def test_derived_total_floors_at_min_game_seconds() -> None:
    """Below MIN_GAME_S the scheduler's floor wins: 20 s asked, 120 s given."""
    offline_loop = load_script()
    total = offline_loop.derive_total_seconds(20, 2, 2)
    assert game_budget(0.0, total, 2, 2) == MIN_GAME_S
