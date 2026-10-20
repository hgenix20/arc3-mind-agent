"""The playing agent: frame conversion, the play loop, the runner, the outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arcagent.agent import play, submit
from arcagent.agent.play import GameResult, play_game
from arcagent.core.frames import observation_from_frame
from arcagent.core.types import GRID_SIZE, Action, Observation, Phase
from arcagent.synth.trace import ProbePolicy


@dataclass
class FakeFrame:
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    win_levels: int = 2
    available_actions: list[int] = field(default_factory=lambda: [0, 1, 2])
    frame: list[Any] = field(default_factory=lambda: [np.zeros((8, 8), dtype=np.int8)])


class ScriptedEnv:
    """Returns the scripted frames in order; records every (action id, data)."""

    def __init__(self, frames: list[FakeFrame | None], first: FakeFrame | None = None) -> None:
        self.frames = list(frames)
        self.first = first
        self.calls: list[tuple[int, dict[str, Any] | None]] = []
        self.resets = 0

    @property
    def observation_space(self) -> FakeFrame | None:
        return self.first

    def reset(self) -> FakeFrame | None:
        self.resets += 1
        return self.first or FakeFrame()

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> FakeFrame | None:
        self.calls.append((int(action.value), data))
        return self.frames.pop(0) if self.frames else FakeFrame()


class FixedPolicy:
    def __init__(self, action: Action) -> None:
        self.action = action

    def choose(self, obs: Observation) -> Action:
        return self.action


def test_observation_from_frame_pads_and_maps_phase() -> None:
    obs = observation_from_frame(FakeFrame(state="GAME_OVER", levels_completed=1))
    assert obs.grid.shape == (GRID_SIZE, GRID_SIZE)
    assert obs.grid[0, 0] == 0 and obs.grid[63, 63] == -1
    assert obs.phase == Phase.GAME_OVER and obs.terminal
    assert obs.levels_completed == 1 and obs.available_actions == (0, 1, 2)
    assert observation_from_frame(FakeFrame(frame=[])).grid.min() == -1


def test_play_game_stops_on_win_and_counts_every_action() -> None:
    pytest.importorskip("arcengine")
    env = ScriptedEnv(
        [FakeFrame(), FakeFrame(levels_completed=1), FakeFrame(state="WIN", levels_completed=2)]
    )
    r = play_game(env, FixedPolicy(Action(1)), "g", max_actions=50)
    assert r.stop == "win" and r.actions == 3 and r.levels_completed == 2
    assert env.resets == 1 and [c[0] for c in env.calls] == [1, 1, 1]
    assert r.phase == "WIN"


def test_play_game_sends_click_coordinates_and_honors_budget() -> None:
    pytest.importorskip("arcengine")
    env = ScriptedEnv([])
    r = play_game(env, FixedPolicy(Action(6, 3, 4)), "g", max_actions=4)
    assert r.stop == "max_actions" and r.actions == 4
    assert env.calls[0] == (6, {"x": 3, "y": 4})


def test_play_game_deadline_and_step_failures() -> None:
    pytest.importorskip("arcengine")
    ticks = iter([0.0, 0.0, 10.0, 10.0, 10.0])
    r = play_game(
        ScriptedEnv([]),
        FixedPolicy(Action(1)),
        "g",
        max_actions=9,
        deadline=5.0,
        clock=lambda: next(ticks),
    )
    assert r.stop == "deadline" and r.actions == 1

    env = ScriptedEnv([None, None, None])
    r = play_game(env, FixedPolicy(Action(1)), "g", max_actions=9)
    assert (
        r.stop == "step_failures" and r.actions == play.MAX_STEP_FAILURES and r.step_failures == 3
    )

    env = ScriptedEnv([None, FakeFrame(), FakeFrame(state="WIN")])
    r = play_game(env, FixedPolicy(Action(1)), "g", max_actions=9)
    assert r.stop == "win" and r.step_failures == 0


def test_play_game_rejects_bad_budget_and_failed_reset() -> None:
    with pytest.raises(ValueError):
        play_game(ScriptedEnv([]), FixedPolicy(Action(1)), "g", max_actions=0)

    class DeadEnv(ScriptedEnv):
        def reset(self) -> None:
            return None

    with pytest.raises(RuntimeError):
        play_game(DeadEnv([]), FixedPolicy(Action(1)), "g", max_actions=1)


def test_probe_policy_is_deterministic_and_resets_on_terminal() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    grid[2:5, 2:5] = 3
    live = Observation(grid, Phase.NOT_FINISHED, 0, 1, (0, 1, 2, 6))
    dead = Observation(grid, Phase.GAME_OVER, 0, 1, (0, 1, 2, 6))
    a = [ProbePolicy(7).choose(live) for _ in range(1)] + [ProbePolicy(7).choose(live)]
    assert a[0] == a[1]
    p, q = ProbePolicy(3), ProbePolicy(3)
    assert [p.choose(live) for _ in range(20)] == [q.choose(live) for _ in range(20)]
    assert ProbePolicy(0).choose(dead) == Action(0)
    assert all(x.id in (1, 2, 6) for x in (ProbePolicy(1).choose(live) for _ in range(30)))


def test_resolve_mode() -> None:
    assert submit.resolve_mode("offline", {"KAGGLE_IS_COMPETITION_RERUN": "true"}) == "offline"
    assert submit.resolve_mode("auto", {"KAGGLE_IS_COMPETITION_RERUN": "true"}) == "competition"
    assert submit.resolve_mode("auto", {"KAGGLE_IS_COMPETITION_RERUN": "1"}) == "competition"
    assert submit.resolve_mode("auto", {}) == "offline"
    with pytest.raises(ValueError):
        submit.resolve_mode("live")


def test_wait_for_gateway_polls_then_times_out() -> None:
    attempts: list[str] = []

    class Ok:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: Any) -> None:
            return None

    def fetch(url: str) -> Ok:
        attempts.append(url)
        if len(attempts) < 3:
            raise OSError("refused")
        return Ok()

    ticks = iter(range(100))
    waited = submit.wait_for_gateway(
        "http://gw:1",
        50,
        poll_s=1,
        fetch=fetch,
        clock=lambda: float(next(ticks)),
        sleep=lambda s: None,
    )
    assert attempts == ["http://gw:1/api/games"] * 3 and waited >= 0
    with pytest.raises(TimeoutError, match="refused"):
        submit.wait_for_gateway(
            "http://gw:1",
            2,
            poll_s=1,
            fetch=lambda u: (_ for _ in ()).throw(OSError("refused")),
            clock=lambda: float(next(ticks)),
            sleep=lambda s: None,
        )


def test_select_games_filters_by_prefix_and_names_missing() -> None:
    avail = ["vc33-1", "ls20-2", "ft09-3"]
    assert submit.select_games(avail, None) == ["ft09-3", "ls20-2", "vc33-1"]
    assert submit.select_games(avail, ["ls20", "vc33"]) == ["ls20-2", "vc33-1"]
    with pytest.raises(ValueError, match="zz99"):
        submit.select_games(avail, ["ls20", "zz99"])


def test_submission_rows_take_scores_from_the_scorecard() -> None:
    results = [
        GameResult("a-1", 5, 1, 2, "NOT_FINISHED", 1.0, "max_actions"),
        GameResult("b-2", 0, 0, 0, "NOT_PLAYED", 0.0, "error"),
    ]
    card = {"score": 0.25, "environments": [{"id": "a-1", "score": 0.5}]}
    rows = submit.submission_rows(results, card)
    assert rows == [
        {"row_id": "0_0", "game_id": "a-1", "end_of_game": True, "score": 0.5},
        {"row_id": "1_0", "game_id": "b-2", "end_of_game": True, "score": 0.0},
    ]
    assert submit.submission_rows(results, None)[0]["score"] == 0.0


def test_write_parquet_schema(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    submit.write_parquet(
        [{"row_id": "0_0", "game_id": "x", "end_of_game": True, "score": 0.0}],
        tmp_path / "s.parquet",
    )
    t = pq.read_table(tmp_path / "s.parquet")
    assert t.column_names == list(submit.PARQUET_COLUMNS) and t.num_rows == 1
    assert str(t.schema.field("score").type) == "double"


def test_play_all_survives_a_broken_game_and_closes_the_scorecard() -> None:
    pytest.importorskip("arcengine")

    class FakeArcade:
        def __init__(self) -> None:
            self.closed: list[str] = []

        def open_scorecard(self, tags: list[str]) -> str:
            return "card"

        def make(self, game_id: str, scorecard_id: str, save_recording: bool) -> ScriptedEnv | None:
            if game_id == "bad":
                return None
            return ScriptedEnv([FakeFrame(state="WIN", levels_completed=2)])

        def close_scorecard(self, card: str) -> dict[str, Any]:
            self.closed.append(card)
            return {"score": 1.0, "environments": [{"id": "good", "score": 1.0}]}

    arcade = FakeArcade()
    lines: list[str] = []
    results, errors, card = submit.play_all(
        arcade,
        ["good", "bad", "late"],
        seed=0,
        max_actions=5,
        game_seconds=10,
        total_seconds=1e9,
        report=lines.append,
    )
    assert [r.stop for r in results] == ["win", "error", "win"]
    assert errors == [{"game_id": "bad", "error": "RuntimeError: arcade.make returned None"}]
    assert arcade.closed == ["card"] and card is not None and card["score"] == 1.0
    assert len(lines) == 3

    ticks = iter([0.0, 100.0, 100.0])
    results, _, _ = submit.play_all(
        arcade,
        ["good"],
        seed=0,
        max_actions=5,
        game_seconds=10,
        total_seconds=50,
        clock=lambda: next(ticks),
        report=lambda s: None,
    )
    assert results[0].stop == "skipped"


def test_offline_run_end_to_end(env_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    doc = submit.main(
        [
            "--mode",
            "offline",
            "--env-dir",
            str(env_dir),
            "--games",
            "ls20",
            "ft09",
            "--max-actions",
            "25",
            "--out",
            str(tmp_path),
        ]
    )
    assert doc["config"]["mode"] == "offline" and len(doc["games"]) == 2 and doc["errors"] == []
    assert all(g["actions"] == 25 and g["stop"] == "max_actions" for g in doc["games"])
    assert [g["game_id"][:4] for g in doc["games"]] == ["ft09", "ls20"]
    assert doc["scorecard"] is not None and {e["id"] for e in doc["scorecard"]["environments"]} == {
        g["game_id"] for g in doc["games"]
    }
    assert (tmp_path / "submission.parquet").exists()
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["submission_rows"] == doc["submission_rows"] and len(saved["submission_rows"]) == 2
    assert all(r["end_of_game"] is True for r in saved["submission_rows"])


def test_agent_and_trace_take_the_same_actions(env_dir: Path) -> None:
    """Same seed, same game: the agent's first actions equal the trace's."""
    from arcagent.synth.trace import collect_trace  # noqa: PLC0415  # needs the wheels

    trace = collect_trace(env_dir, "ls20", 12, seed=0)
    env = submit.make_arcade("offline", str(env_dir), Path("bench/agent-test")).make(
        "ls20", scorecard_id="t", save_recording=False
    )
    assert env is not None
    seen: list[Action] = []

    class Recording(ProbePolicy):
        def choose(self, obs: Observation) -> Action:
            a = super().choose(obs)
            seen.append(a)
            return a

    play_game(env, Recording(0), "ls20", max_actions=12)
    assert seen == [t.action for t in trace.transitions]
