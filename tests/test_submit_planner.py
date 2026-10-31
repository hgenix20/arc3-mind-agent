"""``MindRun`` on the ``PLANNER`` config: the agent class, the hybrid wrap, the store."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from arcagent.agent import submit
from arcagent.agent.hybrid import HybridPolicy
from arcagent.core.types import GRID_SIZE, Observation, Phase
from arcagent.mind.config import CONFIGS
from arcagent.mind.longterm import LongTermStore
from arcagent.mind.planner_agent import PlannerAgent
from arcagent.mind.vchat import scripted


def _run(tmp_path: Path, **extra: object) -> submit.MindRun:
    now = time.monotonic()
    return submit.MindRun(
        CONFIGS["PLANNER"],
        chat=None,
        games=["g"],
        workers=1,
        log_dir=tmp_path / "logs",
        seed=0,
        run_deadline=now + 3600,
        vchat=scripted([]),
        **extra,  # pyright: ignore[reportArgumentType]
    )


def _play(game: submit.MindGame, moves: int) -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), np.int8)
    for i in range(moves):
        grid = grid.copy()
        grid[0, i] = 1
        game.choose(Observation(grid, Phase.NOT_FINISHED, 0, 1, (1, 2)))


def test_planner_config_is_fluid_llm_with_the_planner_on() -> None:
    cfg = CONFIGS["PLANNER"]
    assert cfg.planner and cfg.fluid and cfg.llm and not cfg.drive
    assert cfg == submit.replace(CONFIGS["FLUID-LLM"], name="PLANNER", planner=True)
    assert not any(c.planner for name, c in CONFIGS.items() if name != "PLANNER")


def test_mind_run_planner_config_builds_a_planner_agent_on_the_store(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert isinstance(run.store, LongTermStore)
    assert run.store.path == tmp_path / "logs" / "longterm.json"
    game = run.make_agent("g", time.monotonic() + 3600)
    assert isinstance(game.agent, PlannerAgent)
    _play(game, 3)
    run.pool.shutdown(wait=True)
    game.close()
    run.close()
    assert run.store.path.exists()
    recs = [json.loads(x) for x in (tmp_path / "logs" / "g.jsonl").read_text().splitlines()]
    assert not [r for r in recs if r["type"] == "agent_error"]
    assert [r for r in recs if r["type"] == "longterm"]
    assert recs[-1]["type"] == "game_end" and recs[-1]["config"] == "PLANNER"


def test_mind_run_planner_close_saves_the_store_at_the_given_path(tmp_path: Path) -> None:
    path = tmp_path / "elsewhere" / "store.json"
    run = _run(tmp_path, longterm=path)
    assert run.store is not None and run.store.path == path
    run.close()  # no game was played; the store is still written
    assert path.exists()
    fresh = LongTermStore(path)
    fresh.load()
    assert fresh.load_error is None and fresh.rules == {}


def test_hybrid_moves_wrap_the_planner_agent(tmp_path: Path) -> None:
    run = _run(tmp_path, hybrid_moves=5)
    game = run.make_agent("g", time.monotonic() + 3600)
    assert isinstance(game.agent, HybridPolicy)
    assert isinstance(game.agent.mind, PlannerAgent)
    _play(game, 2)
    run.pool.shutdown(wait=True)
    game.close()
    run.close()


def test_mind_run_without_the_planner_has_no_store(tmp_path: Path) -> None:
    run = submit.MindRun(
        CONFIGS["FLUID"],
        chat=None,
        games=["g"],
        workers=1,
        log_dir=tmp_path,
        seed=0,
        run_deadline=time.monotonic() + 3600,
    )
    assert run.store is None
    run.close()
    assert not (tmp_path / "longterm.json").exists()


def test_parse_args_and_the_results_record_carry_longterm(tmp_path: Path) -> None:
    args = submit.parse_args(["--policy", "mind", "--config", "PLANNER"])
    assert args.longterm is None
    args = submit.parse_args(
        ["--policy", "mind", "--config", "PLANNER", "--longterm", str(tmp_path / "lt.json")]
    )
    assert args.longterm == str(tmp_path / "lt.json")
