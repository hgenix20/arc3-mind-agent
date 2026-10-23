"""fluid.FluidAgent end to end on the tiny fake games: no LLM, no sandbox worker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from arcagent.core.types import Action, Observation, Phase
from arcagent.mind import fluid
from arcagent.mind.config import CONFIGS
from arcagent.mind.runlog import RunLog
from tests.mind.fakegames import FakeClickGame, FakeGame, play

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mind_metrics.py"
Rec = dict[str, Any]


def load_metrics() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mind_metrics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_agent(tmp_path: Path, name: str = "f", **kw: Any) -> fluid.FluidAgent:
    log = RunLog(tmp_path / f"{name}.jsonl", "fake", "FLUID", 0)
    return fluid.FluidAgent(
        "fake", CONFIGS["FLUID"], seed=0, log=log, game_deadline=1e12, run_deadline=1e12, **kw
    )


def records(tmp_path: Path, name: str = "f") -> list[Rec]:
    lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_fluid_config_exists_and_skips_llm() -> None:
    cfg = CONFIGS["FLUID"]
    assert cfg.fluid and not cfg.llm
    assert cfg.name == "FLUID"
    assert not (cfg.metacognition or cfg.inhibition or cfg.voi or cfg.transfer)
    assert not CONFIGS["FULL"].fluid


def test_agent_learns_moves_then_plans_to_the_target(tmp_path: Path) -> None:
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=30, until=lambda: game.levels >= 1)
    agent.close()
    assert game.levels == 1
    assert game.level_up_steps[0] <= 12
    branches = [r["branch"] for r in records(tmp_path) if r["type"] == "action"]
    assert "learn" in branches and "plan" in branches


def test_agent_plays_the_second_level_from_its_facts(tmp_path: Path) -> None:
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=60)
    agent.close()
    assert game.levels == 2
    first, second = game.level_up_steps
    assert second - first <= 6  # three moves right, no tests needed


def test_agent_resets_on_game_over_and_avoids_the_cause_next_time(tmp_path: Path) -> None:
    # Lava sits on the straight path down to the target: the first plan walks into
    # it, the death fact it leaves makes the next plan go around.
    game = FakeGame(player=(2, 2), target=(2, 8), walls=[], lava=[(2, 5)])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=60, until=lambda: game.levels >= 1 or game.game_overs >= 2)
    agent.close()
    assert game.game_overs <= 1
    assert game.levels == 1
    recs = records(tmp_path)
    acts = [r for r in recs if r["type"] == "action"]
    if game.game_overs:
        over = next(r for r in recs if r["type"] == "game_over")
        after = [a for a in acts if a["step"] == over["step"] + 1]
        assert after and after[0]["action"][0] == 0 and after[0]["branch"] == "reset"


def test_brief_lava_layout_is_never_entered_twice(tmp_path: Path) -> None:
    game = FakeGame(player=(2, 2), target=(2, 8), walls=[], lava=[(3, 2)])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=60, until=lambda: game.levels >= 1 or game.game_overs >= 2)
    agent.close()
    assert game.game_overs <= 1 and game.levels == 1


def test_level_change_learns_nothing_from_the_new_screen(tmp_path: Path) -> None:
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=40, until=lambda: game.levels >= 1)
    play(game, agent, max_steps=1)  # the agent sees the new level's first frame
    agent.close()
    assert game.levels == 1
    move = game.level_up_steps[0] - 1  # the agent's step index of the winning move
    for fact in agent.mem.facts.values():
        if move in fact.evidence:
            assert fact.key == ("event", "level_up", "any")
    recs = records(tmp_path)
    (up,) = [r for r in recs if r["type"] == "level_up"]
    assert up["step"] == move and up["level"] == 1
    for r in recs:
        if r["type"] == "fact" and r["step"] == move:
            assert r["key"] == ["event", "level_up", "any"]


def test_click_only_game_tests_each_object_once(tmp_path: Path) -> None:
    game = FakeClickGame(objects=3)
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=20, until=lambda: game.levels >= 1)
    agent.close()
    assert game.levels == 1
    assert sorted(game.clicks) == [0, 1, 2]


def test_log_scores_with_mind_metrics(tmp_path: Path) -> None:
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=30, until=lambda: game.levels >= 1)
    play(game, agent, max_steps=2)
    agent.close()
    mm = load_metrics()
    out = mm.compute(mm.load([tmp_path])[0], human={"fake": [10, 10]})
    assert out["games"]["fake"]["levels_completed"] == 1
    assert out["games"]["fake"]["prediction_accuracy"] is not None
    recs = records(tmp_path)
    keys = {"level", "step", "frame_hash", "action", "branch", "why", "programs", "vetoes"}
    keys |= {"voi", "entropy", "wake", "active"}
    for r in recs:
        if r["type"] == "action":
            assert keys <= set(r)
            assert r["active"] is None and r["voi"] == {} and r["wake"] is None
    end = recs[-1]
    assert end["type"] == "game_end"
    assert sum(end["facts"].values()) == len(agent.mem.facts)
    assert {"facts", "goals", "tests", "vetoes"} <= set(end)
    assert any(r["type"] == "fact" for r in recs)
    assert any(r["type"] == "goal" for r in recs)
    assert any(r["type"] == "test" for r in recs)


def test_not_played_returns_reset_and_close_is_idempotent(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    grid = np.zeros((64, 64), dtype=np.int8)
    obs = Observation(grid, Phase.NOT_PLAYED, 0, 1, (0, 1, 2, 3, 4))
    assert agent.choose(obs) == Action(0)
    agent.close()
    agent.close()
    assert sum(1 for r in records(tmp_path) if r["type"] == "game_end") == 1


def test_mind_side_errors_fall_back_to_b0(tmp_path: Path, monkeypatch: Any) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("x" * 1000)

    monkeypatch.setattr(fluid, "learn", boom)
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=5)
    agent.close()
    recs = records(tmp_path)
    errors = [r for r in recs if r["type"] == "agent_error"]
    assert errors and all(len(r["error"]) <= fluid.MAX_ERROR_CHARS for r in errors)
    acts = [r for r in recs if r["type"] == "action"]
    assert len(acts) == 5
    assert all(a["why"] == "agent_error" for a in acts[1:])


def test_past_the_deadline_skips_search_and_tests(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "f.jsonl", "fake", "FLUID", 0)
    agent = fluid.FluidAgent(
        "fake", CONFIGS["FLUID"], seed=0, log=log, game_deadline=0.0, run_deadline=1e12
    )
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    play(game, agent, max_steps=5)
    agent.close()
    acts = [r for r in records(tmp_path) if r["type"] == "action"]
    assert all(a["branch"] == "fallback" for a in acts)
