"""The concurrent scheduler: time budgets, error isolation, one shared scorecard."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arcagent.agent import scheduler, submit
from arcagent.core.types import Action, Observation
from arcagent.synth.trace import ProbePolicy


def test_game_budget_spreads_time_and_respects_floor() -> None:
    # 1000 s usable after the 300 s reserve, 4 workers, 8 games left: 500 s each.
    assert scheduler.game_budget(0.0, 1300.0, 8, 4) == pytest.approx(500.0)
    # One game left with many workers is capped so the reserve holds.
    assert scheduler.game_budget(0.0, 1300.0, 1, 8) == pytest.approx(1000.0)
    # Too little time per game: the floor applies, even inside the reserve.
    assert scheduler.game_budget(0.0, 1300.0, 1000, 1) == pytest.approx(120.0)
    assert scheduler.game_budget(1100.0, 1300.0, 1, 4) == pytest.approx(120.0)
    # The floor never reaches past the run deadline.
    assert scheduler.game_budget(1250.0, 1300.0, 5, 1) == pytest.approx(50.0)
    assert scheduler.game_budget(1400.0, 1300.0, 5, 1) == 0.0
    # games_left of 0 is treated as 1.
    assert scheduler.game_budget(0.0, 1300.0, 0, 1) == pytest.approx(1000.0)
    for now in (0.0, 500.0, 1000.0, 1200.0, 1299.0):
        for left in (1, 3, 30):
            assert now + scheduler.game_budget(now, 1300.0, left, 8) <= 1300.0


@dataclass
class FakeFrame:
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    win_levels: int = 1
    available_actions: list[int] = field(default_factory=lambda: [1, 2])
    frame: list[Any] = field(default_factory=lambda: [np.zeros((8, 8), dtype=np.int8)])


class WinningEnv:
    """Starts unfinished, wins on the first step."""

    @property
    def observation_space(self) -> FakeFrame | None:
        return FakeFrame()

    def reset(self) -> FakeFrame | None:
        return FakeFrame()

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> FakeFrame | None:
        return FakeFrame(state="WIN", levels_completed=1)


class FakeAgent:
    def __init__(self, game: str, crash: bool, closed: list[str]) -> None:
        self.game = game
        self.crash = crash
        self.closed = closed

    def choose(self, obs: Observation) -> Action:
        if self.crash:
            raise RuntimeError("mind exploded")
        return Action(1)

    def close(self) -> None:
        self.closed.append(self.game)


class FakeArcade:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.opened = 0
        self.closed: list[str] = []
        self.made: list[str] = []

    def open_scorecard(self, tags: list[str]) -> str:
        self.opened += 1
        return "card"

    def make(self, game_id: str, scorecard_id: str, save_recording: bool) -> WinningEnv | None:
        with self.lock:
            self.made.append(game_id)
        return None if game_id == "nomake" else WinningEnv()

    def close_scorecard(self, card: str) -> dict[str, Any]:
        self.closed.append(card)
        return {"score": 0.5, "environments": [{"id": "good", "score": 1.0}]}


def test_play_concurrent_isolates_a_crashing_game() -> None:
    pytest.importorskip("arcengine")
    arcade = FakeArcade()
    closed: list[str] = []
    deadlines: dict[str, float] = {}
    lock = threading.Lock()

    def make_agent(game: str, deadline: float) -> FakeAgent:
        with lock:
            deadlines[game] = deadline
        if game == "noagent":
            raise ValueError("factory failed")
        return FakeAgent(game, game == "crash", closed)

    lines: list[str] = []
    games = ["good", "crash", "noagent", "nomake", "good2"]
    results, errors, card = scheduler.play_concurrent(
        arcade,
        games,
        workers=3,
        make_agent=make_agent,
        total_seconds=1e6,
        max_actions=5,
        report=lines.append,
    )
    assert [r.game_id for r in results] == games
    assert [r.stop for r in results] == ["win", "error", "error", "error", "win"]
    by_game = {e["game_id"]: e["error"] for e in errors}
    assert by_game == {
        "crash": "RuntimeError: mind exploded",
        "noagent": "ValueError: factory failed",
        "nomake": "RuntimeError: arcade.make returned None",
    }
    # Every agent that was built was closed, the crashing one included.
    assert sorted(closed) == ["crash", "good", "good2"]
    assert arcade.opened == 1 and arcade.closed == ["card"]
    assert card is not None and card["score"] == 0.5
    assert len(lines) == len(games)
    assert set(deadlines) == {"good", "crash", "noagent", "good2"}


def test_play_concurrent_skips_games_past_the_run_deadline() -> None:
    pytest.importorskip("arcengine")
    arcade = FakeArcade()
    ticks = iter([0.0] + [100.0] * 50)
    results, errors, _ = scheduler.play_concurrent(
        arcade,
        ["a", "b"],
        workers=1,
        make_agent=lambda g, d: FakeAgent(g, False, []),
        total_seconds=50.0,
        clock=lambda: next(ticks),
        report=lambda s: None,
    )
    assert [r.stop for r in results] == ["skipped", "skipped"] and errors == []
    assert arcade.made == [] and arcade.closed == ["card"]


def test_play_concurrent_rejects_bad_workers() -> None:
    with pytest.raises(ValueError, match="workers"):
        scheduler.play_concurrent(
            FakeArcade(),
            ["a"],
            workers=0,
            make_agent=lambda g, d: FakeAgent(g, False, []),
            total_seconds=10.0,
        )


class BarrierProbe(ProbePolicy):
    """The probe policy, but the first choice waits until every game has started."""

    def __init__(self, barrier: threading.Barrier, seed: int) -> None:
        super().__init__(seed)
        self.barrier = barrier
        self.waited = False
        self.closed = False

    def choose(self, obs: Observation) -> Action:
        if not self.waited:
            self.waited = True
            self.barrier.wait()
        return super().choose(obs)

    def close(self) -> None:
        self.closed = True


def test_one_scorecard_accepts_concurrent_games(env_dir: Path, tmp_path: Path) -> None:
    """Two real public games in two threads on one OFFLINE scorecard, 30 actions each."""
    arcade = submit.make_arcade("offline", str(env_dir), tmp_path)
    games = submit.select_games(
        [e.game_id for e in arcade.available_environments], ["ls20", "ft09"]
    )
    barrier = threading.Barrier(len(games), timeout=60)
    agents: list[BarrierProbe] = []

    def make_agent(game: str, deadline: float) -> BarrierProbe:
        agent = BarrierProbe(barrier, len(agents))
        agents.append(agent)
        return agent

    results, errors, card = scheduler.play_concurrent(
        arcade,
        games,
        workers=2,
        make_agent=make_agent,
        total_seconds=600.0,
        max_actions=30,
        report=lambda s: None,
    )
    assert errors == []
    assert all(r.actions == 30 and r.stop == "max_actions" for r in results)
    assert all(a.closed for a in agents) and len(agents) == 2
    assert card is not None
    assert {e["id"] for e in card["environments"]} == set(games)


class HangingEnv(WinningEnv):
    """``step`` blocks until ``release`` is set: a stuck gateway call."""

    def __init__(self, release: threading.Event) -> None:
        self.release = release

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> FakeFrame | None:
        self.release.wait(30)
        return FakeFrame(state="WIN", levels_completed=1)


class HangingArcade(FakeArcade):
    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self.release = release

    def make(self, game_id: str, scorecard_id: str, save_recording: bool) -> WinningEnv | None:
        with self.lock:
            self.made.append(game_id)
        return HangingEnv(self.release) if game_id == "hang" else WinningEnv()


def test_play_concurrent_gives_up_on_a_hung_game() -> None:
    pytest.importorskip("arcengine")
    release = threading.Event()
    try:
        arcade = HangingArcade(release)
        t0 = time.monotonic()
        results, errors, card = scheduler.play_concurrent(
            arcade,
            ["good", "hang"],
            workers=2,
            make_agent=lambda g, d: FakeAgent(g, False, []),
            total_seconds=0.3,
            hang_grace_s=0.2,
            report=lambda s: None,
        )
        assert time.monotonic() - t0 < 5.0
        assert [r.stop for r in results] == ["win", "hung"]
        assert [e["game_id"] for e in errors] == ["hang"] and "hung" in errors[0]["error"]
        assert arcade.closed == ["card"] and card is not None

        # One worker: the hung game holds it, so the queued game never starts.
        arcade = HangingArcade(release)
        results, errors, _ = scheduler.play_concurrent(
            arcade,
            ["hang", "later"],
            workers=1,
            make_agent=lambda g, d: FakeAgent(g, False, []),
            total_seconds=0.3,
            hang_grace_s=0.2,
            report=lambda s: None,
        )
        assert [r.stop for r in results] == ["hung", "skipped"]
        assert [e["game_id"] for e in errors] == ["hang"]
        assert arcade.made == ["hang"] and arcade.closed == ["card"]
    finally:
        release.set()
