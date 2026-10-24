"""fluid.FluidAgent end to end on the tiny fake games: no LLM, no sandbox worker."""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

from arcagent.core.types import Action, Grid, Observation, Phase
from arcagent.mind import fluid, fluid_modules
from arcagent.mind.config import CONFIGS
from arcagent.mind.facts import SPAWN, Effect, FactMemory
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.predictor import Prediction, predict
from arcagent.mind.runlog import RunLog
from arcagent.mind.vchat import ImagesRejected, VisionChat, VMessage, scripted
from arcagent.synth.run import Completion
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
    assert cfg.metacognition and cfg.inhibition and cfg.voi and cfg.transfer
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


# --- slice 2: the executive turn wired into the agent (Task 5) -------------------------

P = ParamSpec("P")
T = TypeVar("T")
_ID_RE = r"#(\d+) c{color} "


def _settle[R](fut: Future[R], fn: Callable[[], R]) -> None:
    try:
        fut.set_result(fn())
    except Exception as exc:
        fut.set_exception(exc)


class SyncPool(Executor):
    """Runs every job inside ``submit``: the turn is done before the next move."""

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        fut: Future[T] = Future()
        _settle(fut, lambda: fn(*args, **kwargs))
        return fut


class NeverPool(Executor):
    """Accepts every job and never runs it: each turn's future stays pending."""

    def __init__(self) -> None:
        self.futures: list[Future[Any]] = []

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        del fn, args, kwargs
        fut: Future[T] = Future()
        fut.set_running_or_notify_cancel()  # running, so cancel fails as on a pool thread
        self.futures.append(fut)
        return fut


def make_llm_agent(
    tmp_path: Path,
    chat: VisionChat,
    pool: Executor,
    name: str = "f",
    *,
    clock: Callable[[], float] | None = None,
    game_deadline: float = 1e12,
    seed: int = 0,
) -> fluid.FluidAgent:
    log = RunLog(tmp_path / f"{name}.jsonl", "fake", "FLUID-LLM", seed)
    extra: dict[str, Any] = {} if clock is None else {"clock": clock}
    return fluid.FluidAgent(
        "fake",
        CONFIGS["FLUID-LLM"],
        seed=seed,
        log=log,
        game_deadline=game_deadline,
        run_deadline=1e12,
        chat=chat,
        pool=pool,
        **extra,
    )


def _prompt_text(messages: list[VMessage]) -> str:
    content = messages[1]["content"]
    assert isinstance(content, list)
    part = cast("dict[str, object]", content[0])
    return str(part["text"])


def _completion(text: str) -> Completion:
    return Completion(text=text, prompt_tokens=0, completion_tokens=0, seconds=0.0)


def reach_chat(plan: list[str]) -> tuple[VisionChat, list[int]]:
    """First turn: reach(player, target) plus ``plan``, ids read from the prompt; then ``{}``."""
    seeds: list[int] = []

    def call(messages: list[VMessage], seed: int) -> Completion:
        seeds.append(seed)
        if len(seeds) > 1:
            return _completion("{}")
        prompt = _prompt_text(messages)
        a = re.search(_ID_RE.format(color=1), prompt)
        b = re.search(_ID_RE.format(color=4), prompt)
        assert a is not None and b is not None, prompt
        reply = {
            "goals": [{"kind": "reach", "a": int(a.group(1)), "b": int(b.group(1))}],
            "plan": [{"action": s, "expect": "player moves"} for s in plan],
        }
        return _completion(json.dumps(reply))

    return call, seeds


def counting_chat() -> tuple[VisionChat, list[int]]:
    seeds: list[int] = []

    def call(messages: list[VMessage], seed: int) -> Completion:
        del messages
        seeds.append(seed)
        return _completion("{}")

    return call, seeds


CLEAR7 = json.dumps({"goals": [{"kind": "clear", "color": 7}]})


def test_fluid_llm_config_exists() -> None:
    cfg = CONFIGS["FLUID-LLM"]
    assert cfg.fluid and cfg.llm and cfg.name == "FLUID-LLM"
    assert cfg.metacognition and cfg.inhibition and cfg.voi and cfg.transfer


def test_llm_reach_goal_and_plan_finish_the_level_faster(tmp_path: Path) -> None:
    base_game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    base = make_agent(tmp_path, "base")
    play(base_game, base, max_steps=30, until=lambda: base_game.levels >= 1)
    base.close()
    chat, seeds = reach_chat(["A4", "A4", "A4"])
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    play(game, agent, max_steps=30, until=lambda: game.levels >= 1)
    agent.close()
    assert game.levels == 1 and base_game.levels == 1
    assert game.level_up_steps[0] < base_game.level_up_steps[0]
    recs = records(tmp_path)
    acts = [r for r in recs if r["type"] == "action"]
    assert any(a["branch"] == "plan" and a["why"] == "llm plan" for a in acts)
    goals = [r for r in recs if r["type"] == "goal" and r["source"] == "llm"]
    assert goals and goals[0]["status"] == "open"
    turns = [r for r in recs if r["type"] == "exec_turn"]
    assert turns[0]["plan_len"] == 3 and len(turns[0]["goals"]) == 1
    assert seeds[0] == 0
    assert not [r for r in recs if r["type"] in ("agent_error", "exec_failed")]


def test_the_reply_is_applied_in_the_move_that_asked(tmp_path: Path) -> None:
    chat, _ = reach_chat(["A4", "A4", "A4"])
    game = FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    play(game, agent, max_steps=3)
    agent.close()
    recs = records(tmp_path)
    acts = [r for r in recs if r["type"] == "action"]
    # The level start wakes at step 0 and the plan's first step is that same move.
    assert acts[0]["wake"] == "level start"
    assert [(a["why"], a["action"][0]) for a in acts] == [("llm plan", 4)] * 3
    turn = next(r for r in recs if r["type"] == "exec_turn")
    assert turn["step"] == 0 and turn["plan_len"] == 3
    assert game.player == (5, 2)


def test_a_level_start_wake_survives_a_heuristic_plan(tmp_path: Path, monkeypatch: Any) -> None:
    chat, seeds = counting_chat()
    game = FakeGame(player=(2, 2), target=(40, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    real = agent._plan_choice  # pyright: ignore[reportPrivateUsage]
    calls = [0]

    def first_is_a_plan(scene: Scene) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            return fluid._Choice(Action(1), "plan", "goal fake")  # up: no route to the target  # pyright: ignore[reportPrivateUsage]
        return real(scene)

    monkeypatch.setattr(agent, "_plan_choice", first_is_a_plan)
    play(game, agent, max_steps=3)
    agent.close()
    recs = records(tmp_path)
    acts = [r for r in recs if r["type"] == "action"]
    assert acts[0]["branch"] == "plan" and acts[0]["wake"] is None
    wakes = [(r["step"], r["reason"]) for r in recs if r["type"] == "wake"]
    assert wakes[0] == (1, "level start") and len(seeds) >= 1


def test_pending_wakes_clear_on_a_new_epoch(tmp_path: Path) -> None:
    agent = make_llm_agent(tmp_path, counting_chat()[0], SyncPool())
    agent._pending.add("fact broken")  # pyright: ignore[reportPrivateUsage]
    epoch = agent._epoch  # pyright: ignore[reportPrivateUsage]
    blank = np.zeros((64, 64), dtype=np.int8)
    agent.choose(Observation(blank, Phase.NOT_PLAYED, 0, 2, (0, 1, 2, 3, 4)))
    agent.close()
    assert agent._epoch == epoch + 1  # pyright: ignore[reportPrivateUsage]
    assert not agent._pending  # pyright: ignore[reportPrivateUsage]


def test_a_turn_past_its_deadline_keeps_the_slice_1_move(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(fluid, "TURN_DEADLINE_S", 0.01)
    base_game = FakeGame(player=(2, 2), target=(40, 2), walls=[], lava=[])
    base_agent = make_agent(tmp_path, "base")
    base = play(base_game, base_agent, max_steps=20)
    base_agent.close()
    pool = NeverPool()
    game = FakeGame(player=(2, 2), target=(40, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, scripted([CLEAR7]), pool)
    assert play(game, agent, max_steps=20) == base
    agent.close()
    recs = records(tmp_path)
    failed = [r for r in recs if r["type"] == "exec_failed"]
    assert failed and failed[0] == {**failed[0], "step": 0, "call": 0, "reason": "timeout"}
    assert len(failed) == len(pool.futures)
    assert all(f.cancel() is False for f in pool.futures)  # left running, never read
    assert not [r for r in recs if r["type"] == "exec_turn"]


def test_match_goals_are_checked_on_real_frames(tmp_path: Path) -> None:
    # Two empty regions match on every frame: ruled out against the level's history.
    same = {"kind": "match", "region_a": [40, 40, 42, 42], "region_b": [50, 50, 52, 52]}
    # The player's start cell against an empty cell: true once the player has moved.
    moved = {"kind": "match", "region_a": [2, 2, 3, 3], "region_b": [50, 50, 51, 51]}
    reply = json.dumps({"goals": [same, moved]})
    game = FakeGame(player=(2, 2), target=(40, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, scripted([reply]), SyncPool())
    play(game, agent, max_steps=6)
    agent.close()
    status = {g.id(): s for g, s in agent.goals.status.items() if g.kind == "match"}
    assert status == {
        "match (40, 40, 42, 42)=(50, 50, 52, 52)": "ruled_out",
        "match (2, 2, 3, 3)=(50, 50, 51, 51)": "excluded",
    }


def test_llm_goals_are_searched_before_heuristic_ones(tmp_path: Path) -> None:
    # The wall is the shallower heuristic reach goal; the executive names the target.
    def make() -> FakeGame:
        return FakeGame(player=(2, 2), target=(20, 2), walls=[(4, 4)], lava=[])

    base_game = make()
    base = make_agent(tmp_path, "base")
    play(base_game, base, max_steps=40, until=lambda: base._plan_goal is not None)  # pyright: ignore[reportPrivateUsage]
    base.close()
    assert base._plan_goal is not None and base._plan_goal.b == f"c8:{DOT}"  # pyright: ignore[reportPrivateUsage]
    chat, _ = reach_chat([])
    game = make()
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    play(game, agent, max_steps=40, until=lambda: agent._plan_goal is not None)  # pyright: ignore[reportPrivateUsage]
    agent.close()
    assert agent._plan_goal == fluid.Goal("reach", a=PSIG, b=TSIG)  # pyright: ignore[reportPrivateUsage]


def test_rule_out_checks_only_the_added_goals(tmp_path: Path, monkeypatch: Any) -> None:
    seen: list[object] = []
    real = fluid.GoalBook.rule_out

    def spy(book: fluid.GoalBook, *args: Any, **kw: Any) -> list[fluid.Goal]:
        seen.append(kw.get("only"))
        return real(book, *args, **kw)

    monkeypatch.setattr(fluid.GoalBook, "rule_out", spy)
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=30, until=lambda: game.levels >= 1)
    agent.close()
    assert seen and all(isinstance(o, list) and o for o in seen)


def test_a_vetoed_first_plan_step_is_a_failed_attempt(tmp_path: Path, monkeypatch: Any) -> None:
    real = fluid_modules.veto

    def refuse_plans(scene: Scene, action: Action, mem: FactMemory, tried: Any) -> str | None:
        # Every move toward the target is a "predicted game over" once moves are known.
        if action.id == 4 and mem.get(("A4", PSIG, "any")) is not None:
            return fluid.GAME_OVER
        return real(scene, action, mem, tried)

    monkeypatch.setattr(fluid_modules, "veto", refuse_plans)
    game = FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=40)
    agent.close()
    recs = records(tmp_path)
    goal = fluid.Goal("reach", a=PSIG, b=TSIG)
    assert agent.goals.status[goal] == "stalled"
    assert (goal.id(), "stalled") in [(r["id"], r["status"]) for r in recs if r["type"] == "goal"]


def test_turn_reply_budget() -> None:
    assert fluid.TURN_MAX_TOKENS == 1536


def test_the_prompt_names_click_facts_by_on_screen_ids(tmp_path: Path) -> None:
    game = FakeClickGame(objects=3)
    agent = make_llm_agent(tmp_path, counting_chat()[0], SyncPool())
    play(game, agent, max_steps=2)  # the first click is learned on the second frame
    scene = agent._last_scene  # pyright: ignore[reportPrivateUsage]
    assert scene is not None
    inp = agent._turn_input(game.observe(), scene, "probe")  # pyright: ignore[reportPrivateUsage]
    agent.close()
    ids = {f"#{t.id}" for t in scene.things}
    clicks = [f for f in inp.facts if f.startswith("click on #")]
    assert clicks, inp.facts
    assert all(set(re.findall(r"#\d+", f)) <= ids for f in clicks)
    text = " ".join([*inp.facts, *(g for g, _ in inp.goals)])
    assert not re.search(r"c\d+:[0-9a-f]{4}", text), text  # no sig, full or abbreviated


def test_images_rejected_disables_turns_for_the_game(tmp_path: Path) -> None:
    calls: list[int] = []

    def chat(messages: list[VMessage], seed: int) -> Completion:
        del messages
        calls.append(seed)
        raise ImagesRejected("400 image input not supported")

    game = FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    play(game, agent, max_steps=40)
    agent.close()
    recs = records(tmp_path)
    failed = [r for r in recs if r["type"] == "exec_failed"]
    assert calls == [0]
    assert [r["reason"] for r in failed] == ["images rejected"]
    assert len([r for r in recs if r["type"] == "wake"]) == 1
    assert not [r for r in recs if r["type"] == "agent_error"]


def test_a_failing_chat_costs_only_the_turn(tmp_path: Path) -> None:
    def chat(messages: list[VMessage], seed: int) -> Completion:
        del messages, seed
        raise ConnectionError("refused")

    base_game = FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    base = play(base_game, make_agent(tmp_path, "base"), max_steps=40)
    game = FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    assert play(game, agent, max_steps=40) == base
    agent.close()
    recs = records(tmp_path)
    failed = [r for r in recs if r["type"] == "exec_failed"]
    assert failed and all("ConnectionError" in r["reason"] for r in failed)
    assert not [r for r in recs if r["type"] in ("agent_error", "exec_turn")]


def test_empty_replies_leave_slice_1_behavior_unchanged(tmp_path: Path) -> None:
    def move_game() -> FakeGame:
        return FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[(3, 2)])

    def click_game() -> FakeClickGame:
        return FakeClickGame(objects=3)

    for name, make in (("move", move_game), ("click", click_game)):
        base_agent = make_agent(tmp_path, f"{name}-base")
        base = play(make(), base_agent, max_steps=60)
        base_agent.close()
        llm_agent = make_llm_agent(tmp_path, scripted([]), SyncPool(), f"{name}-llm")
        assert play(make(), llm_agent, max_steps=60) == base, name
        llm_agent.close()
        recs = records(tmp_path, f"{name}-llm")
        assert [r for r in recs if r["type"] == "exec_turn"]
        assert not [r for r in recs if r["type"] in ("agent_error", "exec_failed")]


BOX = [(1, 2), (3, 2), (2, 1), (2, 3)]  # walls on every side of the player at (2, 2)


def test_wake_reasons_log_in_order(tmp_path: Path) -> None:
    # Boxed in: one LLM plan, then the tests run out and every move falls back to B0.
    plan = json.dumps({"plan": [{"action": "A1"}, {"action": "A2"}]})
    chat_seeds: list[int] = []
    inner = scripted([plan])

    def chat(messages: list[VMessage], seed: int) -> Completion:
        chat_seeds.append(seed)
        return inner(messages, seed)

    game = FakeGame(player=(2, 2), target=(30, 30), walls=BOX, lava=[])
    agent = make_llm_agent(tmp_path, chat, SyncPool(), seed=2)
    play(game, agent, max_steps=80)
    agent.close()
    recs = records(tmp_path)
    wakes = [(r["step"], r["reason"]) for r in recs if r["type"] == "wake"]
    # The reply's two plan steps are moves 0 and 1 (the turn runs inside move 0). "plan
    # finished" stays pending until the spacing window after that turn ends at move 6;
    # then B0 fallbacks wake every 6 moves.
    gap = fluid.MIN_MOVES_BETWEEN_TURNS
    expect = [(0, "level start"), (gap, "plan finished")]
    expect += [(gap * i, "no plan and no test") for i in range(2, fluid.MAX_TURNS_PER_LEVEL)]
    assert wakes == expect
    assert chat_seeds == [2000 + i for i in range(fluid.MAX_TURNS_PER_LEVEL)]
    acts = [r for r in recs if r["type"] == "action"]
    assert [a["why"] for a in acts[0:2]] == ["llm plan", "llm plan"]
    by_step = {a["step"]: a["wake"] for a in acts}
    assert all(by_step[step] == reason for step, reason in wakes)
    assert sum(1 for w in by_step.values() if w is not None) == len(wakes)
    kinds = [r["type"] for r in recs if r["type"] in ("wake", "exec_turn")]
    assert kinds[:2] == ["wake", "exec_turn"]
    assert recs[-1]["type"] == "game_end" and recs[-1]["turns"] == len(wakes)


def test_game_over_wakes_right_after_the_reset(tmp_path: Path) -> None:
    prompts: list[str] = []

    def chat(messages: list[VMessage], seed: int) -> Completion:
        del seed
        prompts.append(_prompt_text(messages))
        return _completion("{}")

    # Lava right above the player: the first test (A1) dies.
    game = FakeGame(player=(2, 2), target=(2, 8), walls=[], lava=[(2, 1)])
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    play(game, agent, max_steps=12)
    agent.close()
    assert game.game_overs == 1
    recs = records(tmp_path)
    wakes = [(r["step"], r["reason"]) for r in recs if r["type"] == "wake"]
    # step 0 dies, step 1 is the RESET, step 2 is the restart frame (spacing exempt)
    assert wakes[:2] == [(0, "level start"), (2, "game over")]
    assert "Wake: game over (last attempt ended in game over after 1 move)" in prompts[1]


def test_long_stretch_wakes_after_many_moves_without_a_turn(tmp_path: Path) -> None:
    chat, seeds = counting_chat()
    game = FakeGame(player=(2, 2), target=(40, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    play(game, agent, max_steps=2)  # level start: turn 0, applied in move 0
    # Pretend the level has gone on without turns; spacing is already satisfied.
    step = agent._action_no  # pyright: ignore[reportPrivateUsage]
    agent._stretch_from = step - fluid.LONG_STRETCH_MOVES + 1  # pyright: ignore[reportPrivateUsage]
    agent._turn_done = step - fluid.MIN_MOVES_BETWEEN_TURNS  # pyright: ignore[reportPrivateUsage]
    play(game, agent, max_steps=1)  # 39 moves since the last turn: no wake yet
    assert len(seeds) == 1
    step = agent._action_no  # pyright: ignore[reportPrivateUsage]
    agent._stretch_from = step - fluid.LONG_STRETCH_MOVES  # pyright: ignore[reportPrivateUsage]
    play(game, agent, max_steps=1)
    agent.close()
    recs = records(tmp_path)
    wakes = [(r["step"], r["reason"]) for r in recs if r["type"] == "wake"]
    assert wakes == [(0, "level start"), (step, "long stretch")]
    assert len(seeds) == 2
    acts = [r for r in recs if r["type"] == "action"]
    assert all(a["branch"] == "learn" for a in acts)  # no plan held on these moves


def test_wake_reason_takes_the_first_rule_in_order() -> None:
    assert fluid.wake_reason(set()) is None
    assert fluid.wake_reason({"fact broken", "plan finished"}) == "plan finished"
    assert fluid.wake_reason({"no plan and no test", "level start"}) == "level start"
    assert fluid.WAKE_ORDER[0] == "level start"
    assert fluid.WAKE_ORDER[1] == "game over"
    assert fluid.WAKE_ORDER[-1] == "long stretch"


def test_turn_budget_per_game(tmp_path: Path) -> None:
    chat, seeds = counting_chat()
    agent = make_llm_agent(tmp_path, chat, SyncPool())
    agent._calls = fluid.MAX_TURNS_PER_GAME  # pyright: ignore[reportPrivateUsage]
    game = FakeGame(player=(2, 2), target=(30, 30), walls=BOX, lava=[])
    play(game, agent, max_steps=10)
    agent.close()
    assert seeds == []


def test_turns_stop_past_three_quarters_of_the_game_budget(tmp_path: Path) -> None:
    now = [0.0]
    chat, seeds = counting_chat()
    agent = make_llm_agent(tmp_path, chat, SyncPool(), clock=lambda: now[0], game_deadline=100.0)
    now[0] = 76.0
    game = FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    play(game, agent, max_steps=10)
    agent.close()
    assert seeds == []


def test_an_llm_test_is_taken_in_the_waking_move(tmp_path: Path) -> None:
    reply = json.dumps({"test": {"action": "A2", "question": "does A2 move the player down?"}})
    game = FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[])
    agent = make_llm_agent(tmp_path, scripted([reply]), SyncPool())
    play(game, agent, max_steps=2)
    agent.close()
    recs = records(tmp_path)
    acts = [r for r in recs if r["type"] == "action"]
    assert acts[0]["action"][0] == 2 and acts[0]["branch"] == "learn"
    assert acts[0]["why"] == "llm test: does A2 move the player down?"
    tests = [r for r in recs if r["type"] == "test"]
    assert tests[0]["step"] == 0 and tests[0]["question"] == "does A2 move the player down?"


def test_click_steps_follow_the_clicked_thing_by_sig() -> None:
    asked, _ = Tracker().update(_grid({(4, 6): 3, (5, 6): 3, (20, 6): 5}), frozenset())
    now, _ = Tracker().update(_grid({(8, 9): 3, (9, 9): 3, (20, 6): 5}), frozenset())
    block = next(x for x in asked.things if x.color == 3)
    cell = min(block.cells)

    def corner(th: fluid.Thing) -> tuple[int, int]:
        return th.x, th.y

    assert fluid.remap_click(Action(6, *cell), asked, now, corner) == Action(6, 8, 9)
    gone, _ = Tracker().update(_grid({(20, 6): 5}), frozenset())
    assert fluid.remap_click(Action(6, *cell), asked, gone, corner) is None
    assert fluid.remap_click(Action(3), asked, gone, corner) == Action(3)
