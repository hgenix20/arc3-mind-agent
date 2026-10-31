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
from typing import Any, ClassVar, ParamSpec, TypeVar, cast

import numpy as np

from arcagent.core.types import Action, Grid, Observation, Phase
from arcagent.mind import fluid, fluid_modules
from arcagent.mind.config import CONFIGS
from arcagent.mind.facts import SPAWN, Effect, FactMemory
from arcagent.mind.objects import Scene, Thing, Tracker, shape_of
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


def test_novelty_explores_a_closed_room_before_b0(tmp_path: Path) -> None:
    # A 5x5 room walled in on every side; the target sits outside, out of reach.
    ring = [(x, y) for x in range(1, 8) for y in (1, 7)]
    ring += [(x, y) for y in range(2, 7) for x in (1, 7)]
    inside = {(x, y) for x in range(2, 7) for y in range(2, 7)}
    game = FakeGame(player=(4, 4), target=(40, 40), walls=ring, lava=[], win_levels=1)
    agent = make_agent(tmp_path)
    visited: set[tuple[int, int]] = set()
    before: list[set[tuple[int, int]]] = []
    obs = game.observe()
    for _ in range(80):
        visited.add(game.player)
        before.append(set(visited))
        obs = game.step(agent.choose(obs))
    agent.close()
    acts = [r for r in records(tmp_path) if r["type"] == "action"]
    novel = [i for i, a in enumerate(acts) if a["why"] == fluid.NOVEL_WHY]
    assert novel and all(acts[i]["branch"] == "plan" for i in novel)
    assert visited == inside
    for i, a in enumerate(acts):
        if a["why"] == "b0":
            assert before[i] == inside  # B0 only once every cell was reached


def test_novelty_without_movers_falls_to_b0(tmp_path: Path, monkeypatch: Any) -> None:
    game = FakeClickGame(objects=2)
    agent = make_agent(tmp_path)

    def no_test(*_a: object) -> None:
        return None

    monkeypatch.setattr(agent, "_test_choice", no_test)
    play(game, agent, max_steps=3)
    agent.close()
    recs = records(tmp_path)
    acts = [r for r in recs if r["type"] == "action"]
    assert acts[0]["branch"] == "fallback" and acts[0]["why"] == "b0"
    assert all(a["why"] != fluid.NOVEL_WHY for a in acts)
    assert not [r for r in recs if r["type"] == "search" and r.get("novel")]


def test_novelty_ignores_a_mover_the_player_does_not_control(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    enemy = f"c7:{DOT}"
    for i in (1, 2, 3, 4):
        agent.mem.record((f"A{i}", enemy, "any"), (Effect("move", dx=1),), 0)
    scene = Tracker().update(_grid({(10, 10): 7}), frozenset())[0]
    agent._simple = [Action(i) for i in (1, 2, 3, 4)]  # pyright: ignore[reportPrivateUsage]
    assert agent._novel_choice(scene) is None  # pyright: ignore[reportPrivateUsage]
    agent.mem.record(("A1", PSIG, "any"), (Effect("move", dy=-1),), 0)
    agent.mem.record(("A2", PSIG, "any"), (Effect("move", dy=1),), 0)
    both = Tracker().update(_grid({(10, 10): 7, (20, 20): 1}), frozenset())[0]
    agent._keep(both, both, _grid({(10, 10): 7, (20, 20): 1}))  # pyright: ignore[reportPrivateUsage]
    choice = agent._novel_choice(both)  # pyright: ignore[reportPrivateUsage]
    assert choice is not None and choice.why == fluid.NOVEL_WHY


def test_novelty_search_uses_half_the_caps(tmp_path: Path, monkeypatch: Any) -> None:
    seen: list[tuple[int, float, bool]] = []
    real = fluid.search

    def spy(*args: Any, **kw: Any) -> Any:
        seen.append((kw["max_nodes"], kw["time_s"], kw.get("goal_test") is not None))
        return real(*args, **kw)

    monkeypatch.setattr(fluid, "search", spy)
    ring = [(x, y) for x in range(1, 8) for y in (1, 7)]
    ring += [(x, y) for y in range(2, 7) for x in (1, 7)]
    game = FakeGame(player=(4, 4), target=(40, 40), walls=ring, lava=[], win_levels=1)
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=30)
    agent.close()
    novel = [s for s in seen if s[2]]
    assert novel and all(s[:2] == (1500, 0.5) for s in novel)
    assert all(s[:2] == (3000, 1.0) for s in seen if not s[2])


def test_a_sleep_error_at_level_up_still_resets_the_level(tmp_path: Path, monkeypatch: Any) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("sleep broke")

    monkeypatch.setattr(fluid, "fluid_sleep", boom)
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=40, until=lambda: game.levels >= 1)
    play(game, agent, max_steps=1)
    assert agent.mem.missed == set() and not agent.mem.landed  # new_level ran
    assert len(agent._history) == 1  # pyright: ignore[reportPrivateUsage]
    play(game, agent, max_steps=40)
    agent.close()
    assert game.levels == 2
    errors = [r for r in records(tmp_path) if r["type"] == "agent_error"]
    up = next(r for r in records(tmp_path) if r["type"] == "level_up")
    assert any(e["step"] == up["step"] and "sleep broke" in e["error"] for e in errors)


def test_test_moves_stop_after_the_attempt_cap(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(fluid, "MAX_TESTS_PER_ATTEMPT", 2)
    # Lava right under the start: a game over comes early and restarts the count.
    game = FakeGame(player=(4, 4), target=(9, 4), walls=[], lava=[(4, 5)])
    agent = make_agent(tmp_path)
    obs = game.observe()
    after_over: list[int] = []
    for _ in range(80):
        if obs.won:
            break
        over = obs.phase == Phase.GAME_OVER
        obs = game.step(agent.choose(obs))
        if over:
            after_over.append(agent._tests_attempt)  # pyright: ignore[reportPrivateUsage]
        assert agent._tests_attempt <= 2  # pyright: ignore[reportPrivateUsage]
    agent.close()
    recs = records(tmp_path)
    first_end = min(r["step"] for r in recs if r["type"] in ("level_up", "game_over"))
    assert len([r for r in recs if r["type"] == "test" and r["step"] <= first_end]) <= 2
    assert game.game_overs >= 1 and after_over and all(n == 0 for n in after_over)
    assert len([r for r in recs if r["type"] == "test"]) > 2  # later attempts test again


def test_search_actions_skip_things_predicted_off_the_board(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent._can_click = True  # pyright: ignore[reportPrivateUsage]
    on = Thing(1, 2, DOT, f"c2:{DOT}", 5, 5, frozenset({(5, 5)}))
    off = Thing(2, 3, DOT, f"c3:{DOT}", 0, 64, frozenset({(0, 64)}))  # pushed past the edge
    scene = Scene((on, off), 0, frozenset(), np.zeros((64, 64), dtype=np.int8))
    clicks = [a for a in agent._actions(scene) if a.id == 6]  # pyright: ignore[reportPrivateUsage]
    assert [(a.x, a.y) for a in clicks] == [(5, 5)]


def test_a_move_seen_to_do_nothing_is_refused_on_that_screen(tmp_path: Path) -> None:
    game = FakeGame(player=(3, 3), target=(10, 10), walls=[], lava=[])
    agent = make_agent(tmp_path)
    obs = game.observe()
    a = agent.choose(obs)
    agent.choose(obs)  # the same frame back: ``a`` changed nothing on this screen
    scene = agent._last_scene  # pyright: ignore[reportPrivateUsage]
    assert scene is not None
    assert agent._veto(scene, a) == fluid.SEEN_NO_EFFECT  # pyright: ignore[reportPrivateUsage]
    for b in (Action(i) for i in (1, 2, 3, 4)):
        if b.key() != a.key():
            assert agent._veto(scene, b) != fluid.SEEN_NO_EFFECT  # pyright: ignore[reportPrivateUsage]
    agent.close()


def test_the_null_inhibition_module_refuses_no_seen_noop(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "n.jsonl", "fake", "FLUID", 0)
    cfg = replace(CONFIGS["FLUID"], inhibition=False)
    agent = fluid.FluidAgent("fake", cfg, seed=0, log=log, game_deadline=1e12, run_deadline=1e12)
    obs = FakeGame(player=(3, 3), target=(10, 10), walls=[], lava=[]).observe()
    a = agent.choose(obs)
    agent.choose(obs)
    scene = agent._last_scene  # pyright: ignore[reportPrivateUsage]
    assert scene is not None
    assert agent._veto(scene, a) is None  # pyright: ignore[reportPrivateUsage]
    agent.close()


def _bar_frame(bar: int, phase: Phase = Phase.NOT_FINISHED, levels: int = 0) -> Observation:
    """A player dot and a 1-cell-thick bar of ``bar`` cells along the bottom edge."""
    g = np.zeros((64, 64), dtype=np.int8)
    g[10, 10] = 1
    g[63, :bar] = 5
    return Observation(g, phase, levels, 3, (0, 1, 2, 3, 4))


def test_counter_cells_survive_game_over(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    for bar in range(20, 14, -1):  # the bar shrinks each move until it's a counter
        agent.choose(_bar_frame(bar))
    noise = agent._noise  # pyright: ignore[reportPrivateUsage]
    assert (0, 63) in noise.cells
    agent.choose(_bar_frame(15, Phase.GAME_OVER))
    assert (0, 63) in noise.cells
    agent.choose(_bar_frame(20))  # the frame after the RESET
    assert (0, 63) in noise.cells
    for bar in range(19, 13, -1):
        agent.choose(_bar_frame(bar))
    agent.choose(_bar_frame(20, levels=1))  # a level-up drops the counter cells
    assert (0, 63) not in noise.cells
    agent.close()


def test_single_unvetoed_action_is_taken_without_search(tmp_path: Path, monkeypatch: Any) -> None:
    game = FakeGame(player=(3, 3), target=(10, 10), walls=[], lava=[])
    agent = make_agent(tmp_path)
    obs = game.observe()
    first = agent.choose(obs)
    scene = agent._last_scene  # pyright: ignore[reportPrivateUsage]
    assert scene is not None
    others = [Action(i) for i in (1, 2, 3, 4) if i != first.id]
    for b in others[:2]:  # seen to do nothing here; ``first`` joins them on the next frame
        agent._noop.add((scene.key(), b.key()))  # pyright: ignore[reportPrivateUsage]

    def no_search(*_a: object, **_k: object) -> object:
        raise AssertionError("a forced move must not search")

    monkeypatch.setattr(fluid, "search", no_search)
    forced = agent.choose(obs)  # the same frame back: ``first`` changed nothing either
    assert forced.key() == others[2].key()
    agent.close()
    last = [r for r in records(tmp_path) if r.get("type") == "action"][-1]
    assert last["branch"] == "forced"


def test_settled_actions_do_not_make_a_move_forced(tmp_path: Path) -> None:
    game = FakeGame(player=(3, 3), target=(10, 10), walls=[], lava=[])
    agent = make_agent(tmp_path)
    obs = game.observe()
    agent.choose(obs)
    obs = game.step(Action(1))
    agent.choose(obs)
    scene = agent._last_scene  # pyright: ignore[reportPrivateUsage]
    assert scene is not None
    agent._tried.update((scene.key(), Action(i).key()) for i in (1, 2, 3, 4))  # pyright: ignore[reportPrivateUsage]
    assert agent._forced(scene) is None  # pyright: ignore[reportPrivateUsage]
    agent.close()


def test_search_cap_falls_back_to_the_best_partial_plan(tmp_path: Path, monkeypatch: Any) -> None:
    real = fluid.search

    def breadth_first(*a: Any, **k: Any) -> Any:  # best-first would find the whole plan
        return real(*a, **{**k, "weight": 0.0})

    monkeypatch.setattr(fluid, "search", breadth_first)
    game = FakeGame(player=(2, 2), target=(60, 60), walls=[], lava=[], win_levels=1)
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=200)
    agent.close()
    acts = [r for r in records(tmp_path) if r["type"] == "action"]
    assert any(r["why"].startswith("toward reach") for r in acts)
    assert game.levels == 1
    assert not any(r["why"] == fluid.NOVEL_WHY for r in acts)


class CommitGame(FakeGame):
    """Standing next to the target doesn't win; A5 there does (A5 elsewhere does nothing).

    ``commits=False`` makes A5 do nothing anywhere."""

    def __init__(self, *, commits: bool = True) -> None:
        super().__init__(player=(2, 2), target=(6, 2), walls=[], lava=[], win_levels=1)
        self.commits = commits

    def observe(self) -> Observation:
        return replace(super().observe(), available_actions=(0, 1, 2, 3, 4, 5, 6))

    def _next_to_target(self) -> bool:
        return abs(self.player[0] - self.target[0]) + abs(self.player[1] - self.target[1]) == 1

    def step(self, action: Action) -> Observation:
        if action.id == 5 and self.commits and self._next_to_target():
            self.steps += 1
            self.levels = 1
            self.phase = Phase.WIN
            return self.observe()
        if action.id in (1, 2, 3, 4):
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action.id]
            nxt = (self.player[0] + dx, self.player[1] + dy)
            if nxt != self.target and 0 <= nxt[0] < 64 and 0 <= nxt[1] < 64:
                self.steps += 1
                self.player = nxt
                return self.observe()
        self.steps += 1
        return self.observe()


def a5_does_nothing(agent: fluid.FluidAgent) -> None:
    """Solid facts that A5 changes nothing: the usual state of a commit action that only
    works in the winning state, which the agent learned elsewhere."""
    for sig in (PSIG, TSIG):
        for move in range(5):
            agent.mem.record(("A5", sig, "any"), (Effect("none"),), move)


def test_goal_holding_without_level_up_triggers_commit_probe(tmp_path: Path) -> None:
    game = CommitGame()
    agent = make_agent(tmp_path)
    a5_does_nothing(agent)
    play(game, agent, max_steps=120)
    agent.choose(game.observe())  # the winning frame: the agent sees the level-up
    agent.close()
    assert game.levels == 1
    acts = [r for r in records(tmp_path) if r["type"] == "action"]
    win = acts[-2]  # the last action is the reply to the winning frame
    assert win["branch"] == "commit" and win["action"][0] == 5
    assert any(r["type"] == "commit_probe" for r in records(tmp_path))
    goal = fluid.Goal("reach", a=PSIG, b=TSIG)
    assert agent.goals.status[goal] == "confirmed"


def test_commit_probe_is_bounded(tmp_path: Path, monkeypatch: Any) -> None:
    game = CommitGame(commits=False)
    agent = make_agent(tmp_path)
    a5_does_nothing(agent)
    clicks = [Action(6, 40 + i, 40) for i in range(10)]  # empty cells: nothing changes

    def many(_scene: Scene) -> list[Action]:
        return [Action(5), *clicks]

    monkeypatch.setattr(agent, "_commit_candidates", many)
    play(game, agent, max_steps=80)
    agent.close()
    acts = [r for r in records(tmp_path) if r["type"] == "action"]
    probes = [i for i, r in enumerate(acts) if r["branch"] == "commit"]
    assert 0 < len(probes) <= fluid.COMMIT_PROBES
    assert probes == list(range(probes[0], probes[0] + len(probes)))  # one run, no repeat
    assert agent.goals.status[fluid.Goal("reach", a=PSIG, b=TSIG)] == "excluded"


class TurnGame:
    """A three-cell player that faces where it last moved sideways: A3 moves it one cell
    left and turns it left, A4 right and right, A1/A2 up/down without turning. Standing next
    to the target (a dot) wins."""

    RIGHT = ((0, 0), (1, 0), (0, 1))
    LEFT = ((0, 0), (1, 0), (1, 1))

    def __init__(self, target: tuple[int, int] = (30, 10)) -> None:
        self.x, self.y, self.face = 10, 10, self.RIGHT
        self.target = target
        self.levels = 0
        self.phase = Phase.NOT_FINISHED

    def cells(self) -> list[tuple[int, int]]:
        return [(self.x + dx, self.y + dy) for dx, dy in self.face]

    def observe(self) -> Observation:
        g = np.zeros((64, 64), dtype=np.int8)
        g[self.target[1], self.target[0]] = 4
        for x, y in self.cells():
            g[y, x] = 1
        return Observation(g, self.phase, self.levels, 1, (0, 1, 2, 3, 4))

    def step(self, action: Action) -> Observation:
        moves = {1: (0, -1, None), 2: (0, 1, None), 3: (-1, 0, self.LEFT), 4: (1, 0, self.RIGHT)}
        if action.id in moves and self.phase == Phase.NOT_FINISHED:
            dx, dy, face = moves[action.id]
            self.x, self.y = self.x + dx, self.y + dy
            self.face = face or self.face
            tx, ty = self.target
            if any(abs(x - tx) + abs(y - ty) == 1 for x, y in self.cells()):
                self.levels, self.phase = 1, Phase.WIN
        return self.observe()


def test_a_turning_player_becomes_one_kind_and_reaches_the_target(tmp_path: Path) -> None:
    game = TurnGame()
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=150)
    agent.close()
    assert game.levels == 1
    assert any(r["type"] == "kind" for r in records(tmp_path))
    kinds = {agent._kinds.kind_of(t.sig) for t in agent._last_full.things if t.color == 1}  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    assert all(k.startswith("k") for k in kinds)


class PegGame:
    """Pegs (color 2) and holes (color 3) in a row two cells apart. Clicking a peg selects it
    (an outline, color 8, around it); clicking a hole two slots from the selected peg with a
    peg between makes it jump, emptying both; any other click clears the selection. One peg
    left wins."""

    def __init__(self) -> None:
        self.slots = {10: 2, 12: 2, 14: 3, 16: 2, 18: 3}
        self.y = 20
        self.sel: int | None = None
        self.levels = 0
        self.phase = Phase.NOT_FINISHED

    def observe(self) -> Observation:
        g = np.zeros((64, 64), dtype=np.int8)
        for x, c in self.slots.items():
            g[self.y, x] = c
        if self.sel is not None:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        g[self.y + dy, self.sel + dx] = 8
        return Observation(g, self.phase, self.levels, 1, (0, 6))

    def step(self, action: Action) -> Observation:
        x = action.x if action.id == 6 and action.y == self.y else None
        kind = self.slots.get(x) if x is not None else None
        if x is not None and kind == 2:
            self.sel = x
        elif x is not None and kind == 3 and self.sel is not None and abs(x - self.sel) == 4:
            mid = (x + self.sel) // 2
            if self.slots.get(mid) == 2:
                self.slots[self.sel], self.slots[mid], self.slots[x] = 3, 3, 2
            self.sel = None
        else:
            self.sel = None
        if sum(1 for c in self.slots.values() if c == 2) == 1:
            self.levels, self.phase = 1, Phase.WIN
        return self.observe()


def test_the_agent_learns_a_selection_marker_and_files_clicks_under_it(tmp_path: Path) -> None:
    game = PegGame()
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=60)
    agent.close()
    assert agent._modes.known  # pyright: ignore[reportPrivateUsage]
    assert f"c2:{DOT}" in agent.mem.selectable
    assert any(k[2].startswith(("selected:", "sel:")) for k in agent.mem.facts)


def _peg_scenes() -> tuple[Scene, Scene, Any]:
    """The peg row after a jump, and the start row: a two-thing change back to the start."""
    start = {(10, 20): 2, (12, 20): 2, (14, 20): 3, (5, 20): 6}
    jumped = {(10, 20): 3, (12, 20): 3, (14, 20): 2, (5, 20): 6}
    t = Tracker()
    first, _ = t.update(fake_grid(start), frozenset())
    after_jump, _ = t.update(fake_grid(jumped), frozenset())
    back, diff = t.update(fake_grid(start), frozenset())
    return first, after_jump, (back, diff)


def fake_grid(cells: dict[tuple[int, int], int]) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def test_a_return_to_the_start_picture_is_learned_as_a_level_reset(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    first, jumped, (back, diff) = _peg_scenes()
    agent._start_key = first.key()  # pyright: ignore[reportPrivateUsage]
    button = Action(6, 5, 20)
    agent._note_reset(jumped, button, back, diff, 7)  # pyright: ignore[reportPrivateUsage]
    fact = agent.mem.get((fluid.action_label(button, jumped), fluid.LEVEL, "any"))
    assert fact is not None and fact.effects == (Effect("reset_level"),)
    agent.close()


def test_one_thing_stepping_back_to_the_start_is_not_a_reset(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    t = Tracker()
    start, _ = t.update(fake_grid({(10, 10): 1}), frozenset())
    moved, _ = t.update(fake_grid({(11, 10): 1}), frozenset())
    back, diff = t.update(fake_grid({(10, 10): 1}), frozenset())
    assert diff is not None
    agent._start_key = start.key()  # pyright: ignore[reportPrivateUsage]
    agent._note_reset(moved, Action(3), back, diff, 2)  # pyright: ignore[reportPrivateUsage]
    assert agent.mem.get(("A3", fluid.LEVEL, "any")) is None
    agent.close()


def test_a_known_reset_is_refused_only_while_this_life_has_progress(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    _, jumped, _ = _peg_scenes()
    button = Action(6, 5, 20)
    agent.mem.record(
        (fluid.action_label(button, jumped), fluid.LEVEL, "any"), (Effect("reset_level"),), 0
    )
    assert agent._veto(jumped, button) != fluid.UNDOES_PROGRESS  # pyright: ignore[reportPrivateUsage]
    goal = fluid.Goal("clear", color=2)
    monitor = agent._progress  # pyright: ignore[reportPrivateUsage]
    monitor.observe(("a",), {goal: 2}, active=goal)
    monitor.observe(("b",), {goal: 1}, active=goal)
    assert agent._veto(jumped, button) == fluid.UNDOES_PROGRESS  # pyright: ignore[reportPrivateUsage]
    agent.close()


class BudgetGame(FakeGame):
    """The move-to-the-target game with a 30-move budget: a bar along the top row loses a
    share of its 64 cells on every move, and the life ends when it is spent."""

    BUDGET = 30

    def __init__(self) -> None:
        super().__init__(player=(10, 10), target=(60, 60), walls=[], lava=[], win_levels=1)
        self.used = 0

    def observe(self) -> Observation:
        obs = super().observe()
        g = obs.grid.copy()
        g[0, :] = 0
        g[0, : round(64 * (self.BUDGET - self.used) / self.BUDGET)] = 7
        return replace(obs, grid=g)

    def step(self, action: Action) -> Observation:
        if action.id == 0:
            self.used = 0
            return super().step(action)
        super().step(action)
        self.used += 1
        if self.used >= self.BUDGET and self.phase == Phase.NOT_FINISHED:
            self.phase = Phase.GAME_OVER
            self.game_overs += 1
        return self.observe()


def test_the_agent_reads_its_budget_and_knows_it_exactly_after_a_budget_death(
    tmp_path: Path,
) -> None:
    game = BudgetGame()
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=50, until=lambda: game.game_overs >= 1 and game.used >= 8)
    agent.close()
    acts = [r for r in records(tmp_path) if r["type"] == "action"]
    assert game.game_overs == 1
    assert any(r["budget"] is not None for r in acts[:30])  # read during the first life
    last = acts[-1]
    assert last["budget"] == BudgetGame.BUDGET - game.used + 1  # exact: before its own move


def test_a_tight_budget_puts_a_fitting_partial_plan_before_tests() -> None:
    assert fluid.budget_tight(remaining=6, distance=5)
    assert fluid.budget_tight(remaining=7, distance=5)
    assert not fluid.budget_tight(remaining=8, distance=5)  # room for a test
    assert not fluid.budget_tight(remaining=4, distance=5)  # doesn't fit at all
    assert not fluid.budget_tight(remaining=None, distance=5)
    assert fluid.TEST_RESERVE == 2


class SequenceGame:
    """A player (color 1, one cell) and three 2x2 pads (colors 2, 3 and 6) that block it.
    Bumping a pad is a step: the next pad of a hidden order lights the next of three 2x2
    boxes along the bottom (color 7, lit 11); any other pad puts every box out (bumping
    the pad that lit the last box does nothing). Three lit boxes win. Arrows 1 to 4."""

    PADS: ClassVar[dict[int, tuple[int, int]]] = {2: (20, 20), 3: (40, 30), 6: (25, 44)}
    BOXES = (20, 24, 28)

    def __init__(self, order: tuple[int, ...] = (6, 2, 3)) -> None:
        self.order = order
        self.player = (10, 30)
        self.lit = 0
        self.bumps: list[int] = []
        self.levels = 0
        self.phase = Phase.NOT_FINISHED

    def _pad_at(self, x: int, y: int) -> int | None:
        for color, (px, py) in self.PADS.items():
            if px <= x < px + 2 and py <= y < py + 2:
                return color
        return None

    def observe(self) -> Observation:
        g = np.zeros((64, 64), dtype=np.int8)
        for color, (px, py) in self.PADS.items():
            g[py : py + 2, px : px + 2] = color
        for i, bx in enumerate(self.BOXES):
            g[60:62, bx : bx + 2] = 11 if i < self.lit else 7
        g[self.player[1], self.player[0]] = 1
        return Observation(g, self.phase, self.levels, 1, (0, 1, 2, 3, 4))

    def step(self, action: Action) -> Observation:
        moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        if self.phase != Phase.NOT_FINISHED or action.id not in moves:
            return self.observe()
        dx, dy = moves[action.id]
        x, y = self.player[0] + dx, self.player[1] + dy
        if not (0 <= x < 64 and 0 <= y < 58):
            return self.observe()
        pad = self._pad_at(x, y)
        if pad is None:
            self.player = (x, y)
            return self.observe()
        self.bumps.append(pad)
        if pad == self.order[self.lit]:
            self.lit += 1
        elif self.lit == 0 or pad != self.order[self.lit - 1]:
            self.lit = 0
        if self.lit == len(self.order):
            self.levels, self.phase = 1, Phase.WIN
        return self.observe()


def test_the_agent_finishes_a_hidden_order_sequence(tmp_path: Path) -> None:
    # (3, 6, 2) needs both kinds of learning: a wrong pad puts the boxes out, and the
    # agent's first bumps (2, then 6) don't start it. Without sequence steps only the
    # order its bumps happen to follow wins (one of the six).
    for order in ((6, 2, 3), (3, 6, 2)):
        game = SequenceGame(order)
        name = "".join(map(str, order))
        agent = make_agent(tmp_path, name)
        play(game, agent, max_steps=400)
        agent.close()
        assert game.phase == Phase.WIN, order
        assert agent._seq.row is not None  # pyright: ignore[reportPrivateUsage]
        assert any(r["type"] == "sequence" for r in records(tmp_path, name))


def test_a_reached_goal_is_probed_by_running_into_its_target(
    tmp_path: Path, monkeypatch: Any
) -> None:
    game = SequenceGame((2, 3, 6))
    agent = make_agent(tmp_path)

    def no_steps(_scene: Scene) -> list[fluid.Goal]:
        return []

    monkeypatch.setattr(agent, "_sequence_goals", no_steps)
    play(game, agent, max_steps=200, until=lambda: bool(game.bumps))
    agent.close()
    acts = [r for r in records(tmp_path) if r["type"] == "action"]
    assert game.bumps and acts[-1]["branch"] == "commit"


class RestyleGame:
    """A 2x2 piece (color 2) the arrows move and A5 recolors through 2, 3, 4, 9, 6 in turn,
    and a 2x2 reference that never changes: color 6 on level 1, 4 on level 2. The piece
    showing the reference's color completes the level, wherever it stands; each level
    starts with the piece at color 2 in its corner."""

    CYCLE = (2, 3, 4, 9, 6)
    REFS = (6, 4)

    def __init__(self) -> None:
        self.levels = 0
        self.level_moves: list[int] = []
        self.moves = 0
        self.phase = Phase.NOT_FINISHED
        self._load()

    def _load(self) -> None:
        self.piece = (10, 10)
        self.i = 0
        self.moves = 0

    def observe(self) -> Observation:
        g = np.zeros((64, 64), dtype=np.int8)
        g[10:12, 40:42] = self.REFS[min(self.levels, len(self.REFS) - 1)]
        x, y = self.piece
        g[y : y + 2, x : x + 2] = self.CYCLE[self.i]
        return Observation(g, self.phase, self.levels, len(self.REFS), (0, 1, 2, 3, 4, 5))

    def step(self, action: Action) -> Observation:
        moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        if self.phase != Phase.NOT_FINISHED:
            return self.observe()
        self.moves += 1
        if action.id == 5:
            self.i = (self.i + 1) % len(self.CYCLE)
        elif action.id in moves:
            dx, dy = moves[action.id]
            x, y = self.piece[0] + dx, self.piece[1] + dy
            if 0 <= x < 30 and 0 <= y < 60:
                self.piece = (x, y)
        if self.CYCLE[self.i] == self.REFS[self.levels]:
            self.levels += 1
            self.level_moves.append(self.moves)
            if self.levels == len(self.REFS):
                self.phase = Phase.WIN
            else:
                self._load()
        return self.observe()


def test_the_agent_restyles_a_piece_to_match_a_reference(tmp_path: Path) -> None:
    game = RestyleGame()
    agent = make_agent(tmp_path)
    play(game, agent, max_steps=300)
    agent.close()
    assert game.phase == Phase.WIN
    assert any(g.kind == "match_thing" for g in agent.goals.status)
    assert game.level_moves[1] <= 4  # the cycle is known: two presses, a few moves at most
