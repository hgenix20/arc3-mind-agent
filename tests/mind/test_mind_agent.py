"""The mind agent end to end on a fake 1-D corridor, a fake chat and a real sandbox."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arcagent.core.types import GRID_SIZE, RESET_ID, Action, Grid, Observation, Phase
from arcagent.mind.agent import SYNTH_CUTOFF_FRACTION, ActivityClock, MindAgent
from arcagent.mind.config import CONFIGS, MindConfig, build_modules
from arcagent.mind.modules.base import NullTransfer, NullVoi
from arcagent.mind.modules.inhibition import DisconfirmInhibition
from arcagent.mind.modules.metacognition import ConfidenceMetacognition
from arcagent.mind.modules.transfer import CarryTransfer
from arcagent.mind.modules.voi import DisagreementVoi
from arcagent.mind.perception import frame_hash
from arcagent.mind.runlog import RunLog
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.synthesizer import Synthesizer
from arcagent.mind.types import Program
from arcagent.synth.prompt import Message
from arcagent.synth.run import Chat, Completion

ROW = 32
FLOOR, PLAYER, GOAL, PIT = 1, 2, 3, 4
START, GOAL_X = 4, 60
FORK_X = 12
ACTION_KEYS = {"level", "step", "frame_hash", "action", "branch", "why", "programs"} | {
    "vetoes",
    "voi",
    "entropy",
    "wake",
    "active",
}

# The corridor's true movement rule and goal. It knows nothing about pits: the game
# draws the player on a pit cell exactly as on the floor, so the rule still predicts
# the GAME_OVER frame.
TRUE_RULES = f"""
def step(grid, action, x=-1, y=-1):
    g = [r[:] for r in grid]
    row = g[{ROW}]
    if {PLAYER} not in row:
        return g
    p = row.index({PLAYER})
    if action == 1:
        q = p - 1
    elif action == 2:
        q = p + 1
    else:
        return g
    if q < 0 or q > 63:
        return g
    row[p] = {FLOOR}
    row[q] = {PLAYER}
    return g

def goal(grid):
    return grid[{ROW}][{GOAL_X}] == {PLAYER}
"""


# Rules for the one-way corridor, where both actions move right.
ONE_WAY_RULES = TRUE_RULES.replace("q = p - 1", "q = p + 1")


class Corridor:
    """Row 32 is the corridor. Action 1 moves left, 2 right. The goal cell raises the
    level and shows the next level's start; the pit cell ends the game (RESET restarts
    the level). A hidden pit is drawn as floor. ``one_way`` makes action 1 move right
    too, so any policy reaches the goal in 56 actions."""

    def __init__(
        self,
        *,
        pit_x: int = 1,
        pit_visible: bool = True,
        win_levels: int = 2,
        one_way: bool = False,
    ) -> None:
        self.one_way = one_way
        self.pit_x = pit_x
        self.pit_visible = pit_visible
        self.win_levels = win_levels
        self.levels = 0
        self.pos = START
        self.phase = Phase.NOT_FINISHED

    def _grid(self) -> Grid:
        g = np.zeros((GRID_SIZE, GRID_SIZE), np.int8)
        g[ROW, :] = FLOOR
        g[ROW, GOAL_X] = GOAL
        if self.pit_visible:
            g[ROW, self.pit_x] = PIT
        g[ROW, self.pos] = PLAYER
        return g

    def obs(self) -> Observation:
        avail = (RESET_ID,) if self.phase == Phase.GAME_OVER else (1, 2)
        return Observation(self._grid(), self.phase, self.levels, self.win_levels, avail)

    def step(self, action: Action) -> Observation:
        if action.id == RESET_ID:
            self.pos, self.phase = START, Phase.NOT_FINISHED
            return self.obs()
        if self.phase != Phase.NOT_FINISHED:
            return self.obs()
        left = -1 if not self.one_way else 1
        q = self.pos + (left if action.id == 1 else 1 if action.id == 2 else 0)
        self.pos = min(max(q, 0), GRID_SIZE - 1)
        if self.pos == self.pit_x:
            self.phase = Phase.GAME_OVER
        elif self.pos == GOAL_X:
            self.levels += 1
            self.pos = START
            if self.levels >= self.win_levels:
                self.phase = Phase.WIN
        return self.obs()


class ImmediateExecutor(Executor):
    """Runs each submission on the caller's thread: deterministic synthesis timing."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        fut: Future[Any] = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # the agent must see the failure through the future
            fut.set_exception(exc)
        return fut


def reply(code: str) -> Completion:
    return Completion(f"```python\n{code}```\n", 100, 50, 0.1, "stop", 0)


def counting_chat(code: str) -> tuple[Chat, list[int]]:
    calls: list[int] = []

    def chat(messages: list[Message], seed: int) -> Completion:
        calls.append(seed)
        return reply(code)

    return chat, calls


def failing_chat(messages: list[Message], seed: int) -> Completion:
    raise ConnectionError("model server unreachable")


@pytest.fixture
def sandbox() -> Iterator[ProgramSandbox]:
    with ProgramSandbox(timeout_s=2.0) as sb:
        yield sb


def make_agent(
    tmp_path: Path,
    sandbox: ProgramSandbox,
    cfg: MindConfig,
    chat: Chat | None,
    *,
    pool: Executor | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    activity: ActivityClock | None = None,
    game_s: float = 3600.0,
) -> tuple[MindAgent, RunLog]:
    log = RunLog(tmp_path / "g.jsonl", "g", cfg.name, 0)
    synth = None if chat is None else Synthesizer(chat, k=cfg.k)
    now = clock()
    agent = MindAgent(
        "g",
        cfg,
        seed=0,
        synth=synth,
        pool=pool or ImmediateExecutor(),
        sandbox=sandbox,
        log=log,
        activity=activity or ActivityClock(clock),
        game_deadline=now + game_s,
        run_deadline=now + game_s + 3600.0,
        clock=clock,
        sleep=sleep,
    )
    return agent, log


def play(agent: MindAgent, game: Corridor, max_steps: int) -> Observation:
    obs = game.obs()
    for _ in range(max_steps):
        if obs.won:
            break
        obs = game.step(agent.choose(obs))
    return obs


def records(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "g.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def actions_of(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in recs if r["type"] == "action"]


def test_configs_and_module_factory() -> None:
    names = {"FULL", "B0", "EXPLORER"}
    for m in ("metacognition", "inhibition", "voi", "transfer"):
        names |= {f"B0+{m}", f"FULL-{m}"}
    assert set(CONFIGS) == names
    assert all(c.name == n for n, c in CONFIGS.items())
    assert not CONFIGS["EXPLORER"].llm and CONFIGS["B0"].llm
    b0 = CONFIGS["B0"]
    assert not (b0.metacognition or b0.inhibition or b0.voi or b0.transfer)
    assert CONFIGS["FULL-inhibition"].metacognition and not CONFIGS["FULL-inhibition"].inhibition
    full = build_modules(CONFIGS["FULL"])
    assert isinstance(full.metacognition, ConfidenceMetacognition)
    assert isinstance(full.inhibition, DisconfirmInhibition)
    assert isinstance(full.voi, DisagreementVoi) and isinstance(full.transfer, CarryTransfer)
    assert isinstance(build_modules(CONFIGS["FULL-transfer"]).transfer, NullTransfer)
    assert isinstance(build_modules(CONFIGS["B0"]).voi, NullVoi)
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIGS["FULL"].k = 2  # pyright: ignore[reportAttributeAccessIssue]  # frozen on purpose


def test_activity_clock_is_shared_idle_time() -> None:
    now = [100.0]
    act = ActivityClock(lambda: now[0])
    now[0] = 130.0
    assert act.idle_s(now[0]) == 30.0
    act.touch()
    assert act.idle_s(131.0) == 1.0


def test_explorer_config_never_calls_llm(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    chat, calls = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["EXPLORER"], chat)
    play(agent, Corridor(), 40)
    agent.close()
    recs = records(tmp_path)
    assert calls == []
    assert not [r for r in recs if r["type"] in ("llm_call", "plan")]
    acts = actions_of(recs)
    assert len(acts) == 40
    assert {a["branch"] for a in acts} <= {"fallback", "reset"}
    assert recs[-1]["type"] == "game_end"


def test_full_config_wakes_after_explore_steps_then_plans_to_goal(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    chat, calls = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat)
    game = Corridor()
    final = play(agent, game, 200)
    agent.close()
    recs = records(tmp_path)
    acts = actions_of(recs)
    first_wake = next(a for a in acts if a["wake"] is not None)
    assert first_wake["wake"] == "new_game"
    # The explore phase: 20 observed transitions before the first call (RESETs excluded).
    assert first_wake["step"] >= 20
    assert all(a["branch"] in ("fallback", "reset") for a in acts if a["step"] < first_wake["step"])
    calls_logged = [r for r in recs if r["type"] == "llm_call"]
    assert calls_logged[0]["admitted"] == 4 and calls_logged[0]["error"] is None
    assert len(calls) == 4 * len(calls_logged)
    assert any(a["branch"] == "plan" for a in acts)
    assert any(a["branch"] == "follow" for a in acts)
    plans = [r for r in recs if r["type"] == "plan"]
    assert plans and plans[0]["target"] == "goal" and plans[0]["stopped"] == "goal"
    assert game.levels >= 1
    assert final.won  # level 2 is solved the same way after a second call


def test_waits_for_synthesis_without_acting_until_idle_limit(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    now = [1000.0]
    released = threading.Event()
    slept: list[float] = []

    def blocked_chat(messages: list[Message], seed: int) -> Completion:
        released.wait(30.0)
        return reply(TRUE_RULES)

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds
        if released.is_set():
            time.sleep(0.01)  # let the pool thread finish

    cfg = CONFIGS["FULL"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        agent, _ = make_agent(
            tmp_path, sandbox, cfg, blocked_chat, pool=pool, clock=lambda: now[0], sleep=fake_sleep
        )
        game = Corridor(pit_x=63, pit_visible=False)  # the goal comes first: no game over
        obs = game.obs()
        for _ in range(20):
            obs = game.step(agent.choose(obs))
        assert slept == []
        before = now[0]
        obs = game.step(agent.choose(obs))  # wakes, submits, belief empty: waits
        acts = actions_of(records(tmp_path))
        assert acts[-1]["wake"] == "new_game"
        assert acts[-1]["branch"] == "wait"
        assert now[0] - before >= cfg.idle_act_s
        assert len(slept) == int(cfg.idle_act_s / cfg.synth_wait_poll_s)
        assert set(slept) == {cfg.synth_wait_poll_s}
        assert not any(a["branch"] == "wait" for a in acts[:-1])
        released.set()
        for _ in range(5):
            obs = game.step(agent.choose(obs))
        agent.close()
    recs = records(tmp_path)
    assert [r["admitted"] for r in recs if r["type"] == "llm_call"] == [4]
    assert any(a["branch"] == "plan" for a in actions_of(recs))


def test_game_over_mid_plan_resets_clears_plan_and_vetoes(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    chat, _ = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat)
    pit = 30
    game = Corridor(pit_x=pit, pit_visible=False)
    play(agent, game, 120)
    agent.close()
    recs = records(tmp_path)
    acts = actions_of(recs)
    pre_pit = Corridor(pit_x=pit, pit_visible=False)
    pre_pit.pos = pit - 1
    pre_hash = frame_hash(pre_pit.obs().grid)

    i_over = next(i for i, r in enumerate(recs) if r["type"] == "game_over")
    # The loss came from following a plan: the step before it was a plan action from pre_pit.
    losing = next(a for a in acts if a["frame_hash"] == pre_hash)
    assert losing["branch"] in ("plan", "follow") and losing["action"] == [2, -1, -1]
    vetoes = [r for r in recs if r["type"] == "veto" and r["reason"] == "game_over"]
    assert vetoes[0]["frame"] == pre_hash and vetoes[0]["action"] == [2, -1, -1]
    # Then RESET, and the next action is a fresh plan, not the rest of the old one.
    after = actions_of(recs[i_over:])
    assert after[0]["action"] == [0, -1, -1] and after[0]["branch"] == "reset"
    assert after[1]["branch"] == "plan"
    # Back at the pre-pit frame, the losing action is vetoed and not taken.
    again = [a for a in after if a["frame_hash"] == pre_hash]
    assert again
    assert [2, -1, -1] in again[0]["vetoes"]
    assert again[0]["action"] != [2, -1, -1]
    assert sum(r["type"] == "game_over" for r in recs) == 1


def test_server_failure_keeps_playing_on_b0(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    cfg = CONFIGS["FULL"]
    agent, _ = make_agent(tmp_path, sandbox, cfg, failing_chat)
    play(agent, Corridor(pit_x=63, pit_visible=False), 90)
    agent.close()
    recs = records(tmp_path)
    calls = [r for r in recs if r["type"] == "llm_call"]
    assert calls
    assert len(calls) <= cfg.max_calls_per_game
    assert all(c["admitted"] == 0 and "ConnectionError" in c["error"] for c in calls)
    acts = actions_of(recs)
    assert len(acts) == 90
    assert {a["branch"] for a in acts} <= {"fallback", "reset"}
    # The call budget is spent at most every min_new_steps; never more than the cap.
    assert [c["call"] for c in calls] == list(range(len(calls)))


def test_every_action_record_has_required_fields(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    chat, _ = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat)
    play(agent, Corridor(), 60)
    agent.close()
    recs = records(tmp_path)
    acts = actions_of(recs)
    assert len(acts) == 60
    for a in acts:
        assert set(a) >= ACTION_KEYS, ACTION_KEYS - set(a)
        assert len(a["action"]) == 3 and isinstance(a["voi"], dict)
        assert all(set(p) == {"id", "pred", "matched", "fit"} for p in a["programs"])
        assert all(len(v) == 3 for v in a["vetoes"])
    assert any(a["programs"] for a in acts)
    for c in (r for r in recs if r["type"] == "llm_call"):
        keys = {"call", "k", "seconds", "prompt_tokens", "completion_tokens"}
        assert keys | {"admitted", "held", "dropped", "error"} <= set(c)
    for d in (r for r in recs if r["type"] == "program_dropped"):
        assert {"pid", "reason", "step", "first_miss_step"} <= set(d)


def test_level_up_runs_sleep_and_transfer(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    chat, _ = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL-transfer"], chat)
    game = Corridor(win_levels=3)
    play(agent, game, 120)
    assert game.levels >= 1
    agent.close()
    kinds = [r["type"] for r in records(tmp_path)]
    i_up = kinds.index("level_up")
    assert kinds[i_up + 1 : i_up + 3] == ["sleep", "transfer"]
    recs = records(tmp_path)
    assert recs[i_up + 1]["reason"] == "level_up"
    # Null transfer carries nothing; the next level starts from an empty belief and wakes.
    assert recs[i_up + 2]["carried"] == []
    later = actions_of(recs[i_up:])
    assert any(a["wake"] == "belief_empty" for a in later)
    last_action = max(i for i, k in enumerate(kinds) if k == "action")
    assert "sleep" in kinds[last_action:] and kinds[-1] == "game_end"
    assert all(len(v) == 1 for v in drops_by_pid(recs).values())


def test_past_the_game_deadline_acts_on_b0_at_once(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    chat, calls = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat, game_s=-1.0)
    play(agent, Corridor(), 30)
    agent.close()
    recs = records(tmp_path)
    assert calls == [] and not [r for r in recs if r["type"] in ("llm_call", "plan")]
    acts = actions_of(recs)
    assert len(acts) == 30
    assert all(a["why"] == "deadline passed" for a in acts)


def drops_by_pid(recs: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in recs:
        if r["type"] == "program_dropped":
            out.setdefault(r["pid"], []).append(r["reason"])
    return out


def test_sleep_replay_drop_is_logged_once(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    # A belief program consolidated at the level-up and then dropped by replay used to be
    # logged twice (sleep_replay and level_end).
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], failing_chat)
    game = Corridor(pit_x=63, pit_visible=False, one_way=True)
    obs = game.obs()
    while game.levels == 0:
        obs = game.step(agent.choose(obs))
    frozen = "def step(grid, action, x=-1, y=-1):\n    return [r[:] for r in grid]\n"
    assert agent.belief.add(Program("frozen", frozen, False, "llm"), hits=9, trials=10) is None
    agent.choose(obs)  # observes the level-up step: 9/11 consolidates, replay finds 0 hits
    agent.close()
    drops = drops_by_pid(records(tmp_path))
    assert drops["frozen"] == ["sleep_replay"]
    assert all(len(v) == 1 for v in drops.values())


def test_stale_result_goes_to_held_not_belief(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    now = [1000.0]
    released = threading.Event()

    def blocked_chat(messages: list[Message], seed: int) -> Completion:
        released.wait(30.0)
        return reply(ONE_WAY_RULES)

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds
        if released.is_set():
            time.sleep(0.01)

    with ThreadPoolExecutor(max_workers=2) as pool:
        agent, _ = make_agent(
            tmp_path,
            sandbox,
            CONFIGS["FULL"],
            blocked_chat,
            pool=pool,
            clock=lambda: now[0],
            sleep=fake_sleep,
            game_s=1e6,  # each wait spends 600 fake seconds
        )
        game = Corridor(pit_x=63, pit_visible=False, one_way=True)
        obs = game.obs()
        # The call starts on level 0 and is still running at the level-up.
        while game.levels == 0:
            obs = game.step(agent.choose(obs))
        agent.choose(obs)  # sees the level-up while the call is in flight
        released.set()
        for _ in range(3):
            obs = game.step(agent.choose(obs))
        agent.close()
    recs = records(tmp_path)
    stale = [r for r in recs if r["type"] == "stale_result"]
    assert [r["pid"] for r in stale] == [f"g-c0-k{i}" for i in range(4)]
    assert all(r["submit_level"] == 0 and r["level"] == 1 for r in stale)
    first = next(r for r in recs if r["type"] == "llm_call")
    assert (first["call"], first["admitted"], first["held"]) == (0, 0, 4)
    assert not [r for r in recs if r["type"] == "program_admitted" and r["call"] == 0]
    # Held at collection; the game-end sleep's replay promotes them (they fit every step).
    assert {r["pid"] for r in stale} <= set(agent.store.held) | set(agent.store.procedural)
    assert not [r for r in recs if r["type"] == "program_dropped"]
    assert not {r["pid"] for r in stale} & set(agent.belief.programs)


def test_lone_program_below_min_trials_does_not_plan_to_goal(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    chat, _ = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat)
    assert agent.belief.add(Program("lone", TRUE_RULES, True, "llm"), hits=1, trials=1) is None
    agent.choose(Corridor().obs())
    agent.close()
    recs = records(tmp_path)
    assert [r["target"] for r in recs if r["type"] == "plan"] == ["novel"]
    act = actions_of(recs)[0]
    assert act["branch"] == "learn" and act["active"] == "lone"


def test_mind_side_error_falls_back_to_b0(
    tmp_path: Path, sandbox: ProgramSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat, _ = counting_chat(TRUE_RULES)
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat)
    game = Corridor(pit_x=63, pit_visible=False)
    obs = play(agent, game, 25)
    assert agent.belief.programs  # admitted after the first wake

    def broken_predict(*args: object, **kwargs: object) -> dict[str, str | None]:
        raise RuntimeError("worker pipe broke")

    monkeypatch.setattr(sandbox, "predict", broken_predict)
    for _ in range(5):
        obs = game.step(agent.choose(obs))
    monkeypatch.undo()
    agent.close()
    recs = records(tmp_path)
    errors = [r for r in recs if r["type"] == "agent_error"]
    assert len(errors) == 5 and all("RuntimeError: worker pipe broke" in e["error"] for e in errors)
    acts = actions_of(recs)
    assert len(acts) == 30
    assert all(a["why"] == "agent_error" and a["branch"] == "fallback" for a in acts[-5:])


class RefusingExecutor(Executor):
    """A pool that has been shut down."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        raise RuntimeError("cannot schedule new futures after shutdown")


def test_failed_submit_logs_no_wake(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    chat, calls = counting_chat(TRUE_RULES)
    pool = RefusingExecutor()
    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat, pool=pool)
    play(agent, Corridor(pit_x=63, pit_visible=False), 30)
    agent.close()
    recs = records(tmp_path)
    failed = [r for r in recs if r["type"] == "llm_call"]
    assert failed and all(r["error"].startswith("submit failed") for r in failed)
    assert calls == []
    assert all(a["wake"] is None for a in actions_of(recs))


def test_no_synthesis_call_starts_inside_the_cutoff(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    """With less than ``synth_cutoff_s`` left, the agent plays on without calling the model."""
    chat, calls = counting_chat(TRUE_RULES)
    cfg = CONFIGS["FULL"]
    budget = 4 * cfg.synth_cutoff_s  # long enough that the absolute cutoff binds
    now = [1000.0]
    agent, _ = make_agent(
        tmp_path, sandbox, cfg, chat, clock=lambda: now[0], sleep=lambda _s: None, game_s=budget
    )
    now[0] += budget - cfg.synth_cutoff_s + 10.0  # inside the last 300 s
    play(agent, Corridor(), 40)
    agent.close()
    recs = records(tmp_path)
    assert calls == [] and not [r for r in recs if r["type"] == "llm_call"]
    acts = actions_of(recs)
    assert len(acts) == 40 and all(a["why"] != "deadline passed" for a in acts)
    # Outside the cutoff the same game does call the model.
    chat2, calls2 = counting_chat(TRUE_RULES)
    with ProgramSandbox(timeout_s=2.0) as fresh:
        agent2, _ = make_agent(tmp_path / "b", fresh, cfg, chat2)
        play(agent2, Corridor(), 40)
        agent2.close()
    assert calls2


def test_cutoff_scales_to_a_short_game_budget(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    """A 120 s game still calls the model early; the last 25% of its budget is off limits."""
    cfg = CONFIGS["FULL"]
    budget = 120.0
    assert budget < cfg.synth_cutoff_s
    chat, calls = counting_chat(TRUE_RULES)
    now = [1000.0]
    agent, _ = make_agent(
        tmp_path, sandbox, cfg, chat, clock=lambda: now[0], sleep=lambda _s: None, game_s=budget
    )
    play(agent, Corridor(), 40)
    agent.close()
    assert calls, "a short game must still start a synthesis call"
    # Inside the last 25% (30 s) of the same budget no call starts.
    chat2, calls2 = counting_chat(TRUE_RULES)
    now2 = [1000.0]
    with ProgramSandbox(timeout_s=2.0) as fresh:
        agent2, _ = make_agent(
            tmp_path / "b",
            fresh,
            cfg,
            chat2,
            clock=lambda: now2[0],
            sleep=lambda _s: None,
            game_s=budget,
        )
        now2[0] += budget * (1.0 - SYNTH_CUTOFF_FRACTION) + 1.0
        play(agent2, Corridor(), 40)
        agent2.close()
    assert calls2 == []


def test_voi_drives_learn_actions_with_two_disagreeing_programs(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    # Both programs fit every explore step (the walk stays left of column 12); they
    # part ways only on action 2 from column 12 on, so VOI has something to separate
    # once the agent wanders there.
    # Neither has a goal(), so the executive cannot plan to the goal and VOI decides.
    no_goal = TRUE_RULES[: TRUE_RULES.index("def goal")]
    stalls = no_goal.replace("q = p + 1", f"q = p + 1 if p < {FORK_X} else p")

    def chat(messages: list[Message], seed: int) -> Completion:
        return reply(no_goal if seed % 2 == 0 else stalls)

    agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL"], chat)
    game = Corridor(pit_visible=False, win_levels=99)
    obs = game.obs()
    for _ in range(300):
        obs = game.step(agent.choose(obs))
        if any(a["branch"] == "learn" and a["voi"] for a in actions_of(records(tmp_path))):
            break
    agent.close()
    learn = [a for a in actions_of(records(tmp_path)) if a["branch"] == "learn" and a["voi"]]
    assert learn, "no learn action with a non-empty voi map"
    assert all(len(k.split(",")) == 3 for a in learn for k in a["voi"])
    assert any(v > 0 for a in learn for v in a["voi"].values())
