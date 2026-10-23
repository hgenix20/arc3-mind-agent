"""fluid.FluidAgent end to end on the tiny fake games: no LLM, no sandbox worker."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from arcagent.core.types import Action, Grid, Observation, Phase
from arcagent.mind import fluid
from arcagent.mind.config import CONFIGS
from arcagent.mind.facts import SPAWN, Effect, FactMemory
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.predictor import Prediction, predict
from arcagent.mind.runlog import RunLog
from tests.mind.fakegames import FakeClickGame, FakeGame, play

DOT = shape_of([(0, 0)])
PSIG, TSIG, SIDE = f"c1:{DOT}", f"c4:{DOT}", f"c7:{DOT}"
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


# --- plans stop only on misses that matter (Task 8c) ----------------------------------


class BlinkGame(FakeGame):
    """FakeGame plus a one-cell side object at (40, 40) that flips 6/7 on every second A4.

    Its facts keep missing (none, then recolor, under the same sig), and it is never
    noise: only one action changes it, and not on every step.
    """

    def __init__(self, *args: Any, **kw: Any) -> None:
        self.blink = False
        self.presses = 0
        super().__init__(*args, **kw)

    def observe(self) -> Observation:
        obs = super().observe()
        g = obs.grid.copy()
        g[40, 40] = 7 if self.blink else 6
        return replace(obs, grid=g)

    def step(self, action: Action) -> Observation:
        if action.id == 4 and self.phase == Phase.NOT_FINISHED:
            self.presses += 1
            if self.presses % 2 == 0:
                self.blink = not self.blink
        return super().step(action)


def test_plan_continues_past_a_side_object_miss(tmp_path: Path) -> None:
    game = BlinkGame(player=(2, 2), target=(16, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=60, until=lambda: game.levels >= 1)
    agent.close()
    assert game.levels == 1
    branches = [r["branch"] for r in records(tmp_path) if r["type"] == "action"]
    last_test = max(i for i, b in enumerate(branches) if b == "learn")
    after = branches[last_test + 1 :]
    # the side object misses on every second A4; the walk right is planned at most twice
    assert after.count("plan") <= 2 and after.count("follow") >= 8, branches


def _grid(cells: dict[tuple[int, int], int]) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def plan_case() -> tuple[Scene, Prediction, Tracker]:
    t = Tracker()
    before, _ = t.update(_grid({(2, 2): 1, (6, 2): 4, (30, 30): 7}), frozenset())
    mem = FactMemory()
    mem.record(("A4", PSIG, "any"), (Effect("move", dx=1),), 0)
    pred = predict(before, Action(4), mem, assume_still=frozenset({PSIG, TSIG}))
    return before, pred, t


def test_plan_breaks_only_on_misses_that_matter() -> None:
    before, pred, t = plan_case()
    after, _ = t.update(_grid({(3, 2): 1, (6, 2): 4, (30, 30): 5}), frozenset())
    movers, protected = frozenset({PSIG}), frozenset({PSIG, TSIG})
    side_key = ("A4", SIDE, "any")
    assert not fluid.plan_breaks(pred, before, after, [side_key], movers, protected)
    assert not fluid.plan_breaks(pred, before, after, [("A4", SPAWN, "any")], movers, protected)
    assert fluid.plan_breaks(pred, before, after, [("A4", PSIG, "any")], movers, protected)
    assert fluid.plan_breaks(pred, before, after, [("A4", TSIG, "any")], movers, protected)


def test_plan_breaks_when_the_mover_lands_elsewhere() -> None:
    before, pred, t = plan_case()
    after, _ = t.update(_grid({(2, 3): 1, (6, 2): 4, (30, 30): 7}), frozenset())
    assert fluid.plan_breaks(pred, before, after, [], frozenset({PSIG}), frozenset({PSIG}))


class ShyGame(FakeGame):
    """FakeGame whose target jumps 3 cells right when the player comes within distance 2."""

    def step(self, action: Action) -> Observation:
        obs = super().step(action)
        px, py = self.player
        tx, ty = self.target
        if self.phase == Phase.NOT_FINISHED and abs(px - tx) + abs(py - ty) == 2:
            self.target = (min(tx + 3, 63), ty)
            return self.observe()
        return obs


def test_goal_stalls_after_two_failed_attempts(tmp_path: Path) -> None:
    game = ShyGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=40)
    agent.close()
    events = [r for r in records(tmp_path) if r["type"] == "goal"]
    stalled = [e for e in events if e["status"] == "stalled"]
    assert stalled, events
    goal = stalled[0]["id"]
    plans = [
        r
        for r in records(tmp_path)
        if r["type"] == "action" and r["branch"] == "plan" and r["why"] == f"goal {goal}"
    ]
    assert 2 <= len(plans) <= 3
    assert agent.goals.status[next(g for g in agent.goals.status if g.id() == goal)] == "stalled"


def test_stalled_goals_reopen_on_level_change(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    g = fluid.Goal("reach", a=PSIG, b=TSIG)
    agent.goals.status[g] = "open"
    agent.note_failed(g)
    assert agent.goals.status[g] == "open"
    agent.note_failed(g)
    assert agent.goals.status[g] == "stalled"
    agent.reopen_goals()
    assert agent.goals.status[g] == "open"
    kinds = [(r["id"], r["status"]) for r in records(tmp_path) if r["type"] == "goal"]
    assert (g.id(), "stalled") in kinds and (g.id(), "open") in kinds


def test_assumed_still_mover_never_breaks_a_plan() -> None:
    bar = f"c7:{shape_of([(0, 0), (1, 0)])}"
    t = Tracker()
    before, _ = t.update(_grid({(2, 2): 1, (6, 2): 4, (30, 30): 7, (31, 30): 7}), frozenset())
    mem = FactMemory()
    mem.record(("A4", PSIG, "any"), (Effect("move", dx=1),), 0)
    pred = predict(before, Action(4), mem, assume_still=frozenset({PSIG, TSIG}))
    assert ("A4", bar) in pred.assumed
    after, _ = t.update(_grid({(3, 2): 1, (6, 2): 4, (31, 30): 7, (32, 30): 7}), frozenset())
    movers = frozenset({PSIG, bar})  # the bar's move fact was learned after the prediction
    assert not fluid.plan_breaks(pred, before, after, [], movers, movers | {TSIG})


def test_plan_ends_when_its_goal_leaves_the_open_list(tmp_path: Path) -> None:
    game = FakeGame(player=(2, 2), target=(20, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=40, until=lambda: len(agent._plan) >= 3)  # pyright: ignore[reportPrivateUsage]
    goal = agent._plan_goal  # pyright: ignore[reportPrivateUsage]
    assert goal is not None and len(agent._plan) >= 3  # pyright: ignore[reportPrivateUsage]
    other = fluid.Goal("reach", a=PSIG, b=SIDE)
    agent.goals.status[other] = "ruled_out"  # a goal-list change that isn't the plan's own
    obs = game.observe()
    a = agent.choose(obs)
    obs = game.step(a)
    agent.choose(obs)
    branches = [r["branch"] for r in records(tmp_path) if r["type"] == "action"]
    assert branches[-1] == "follow"
    agent.goals.status[goal] = "ruled_out"
    obs = game.step(agent._last_action or a)  # pyright: ignore[reportPrivateUsage]
    agent.choose(obs)
    branches = [r["branch"] for r in records(tmp_path) if r["type"] == "action"]
    assert branches[-1] != "follow"


def test_search_events_are_logged(tmp_path: Path) -> None:
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=30, until=lambda: game.levels >= 1)
    agent.close()
    events = [r for r in records(tmp_path) if r["type"] == "search"]
    assert events
    keys = {"step", "stop", "nodes", "depth", "goals", "frontier", "protected"}
    assert all(keys <= e.keys() for e in events)
    assert any(e["stop"] == "found" for e in events)
