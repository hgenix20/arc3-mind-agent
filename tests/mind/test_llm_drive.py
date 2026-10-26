"""DriveAgent: the LLM looks first, plans are played out in its model, experiments and fixes."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from arcagent.mind import fluid, llm_drive
from arcagent.mind.config import CONFIGS
from arcagent.mind.runlog import RunLog
from arcagent.mind.search import SearchResult
from arcagent.mind.tool_fix import FixResult
from arcagent.mind.toolbox import TOOLBOX
from arcagent.mind.vchat import VisionChat, VMessage
from arcagent.synth.run import Completion
from tests.mind.fakegames import FakeGame, play
from tests.mind.test_fluid_agent import SyncPool

Rec = dict[str, Any]
Replier = Callable[[str], dict[str, Any]]


def _prompt(messages: list[VMessage]) -> str:
    content = messages[1]["content"]
    assert isinstance(content, list)
    return str(content[0]["text"])


def ids(prompt: str) -> tuple[int, int]:
    """The player's (color 1) and the target's (color 4) ids in a prompt."""
    player = re.search(r"#(\d+) c1 ", prompt)
    target = re.search(r"#(\d+) c4 ", prompt)
    assert player is not None and target is not None
    return int(player.group(1)), int(target.group(1))


def chat_of(replier: Replier, log: list[str]) -> VisionChat:
    def call(messages: list[VMessage], seed: int) -> Completion:
        del seed
        text = _prompt(messages)
        log.append(text)
        return Completion(json.dumps(replier(text)), 0, 0, 0.0)

    return call


def make(tmp_path: Path, chat: VisionChat | None, name: str = "d") -> llm_drive.DriveAgent:
    log = RunLog(tmp_path / f"{name}.jsonl", "fake", "DRIVE", 0)
    return llm_drive.DriveAgent(
        "fake",
        CONFIGS["FLUID-LLM"],
        seed=0,
        log=log,
        game_deadline=1e12,
        run_deadline=1e12,
        chat=chat,
        pool=SyncPool() if chat is not None else None,
    )


def records(tmp_path: Path, name: str = "d") -> list[Rec]:
    lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def right_facts(p: int, t: int) -> list[dict[str, Any]]:
    return [
        {"action": "A4", "object": p, "effect": "move", "dx": 1, "dy": 0},
        {"action": "A4", "object": t, "effect": "none"},
    ]


def test_level_start_turn_before_first_move(tmp_path: Path) -> None:
    seen: list[str] = []
    agent = make(tmp_path, chat_of(lambda _p: {"model": "first look"}, seen))
    play(FakeGame(player=(2, 2), target=(9, 2), walls=[], lava=[]), agent, max_steps=1)
    agent.close()
    recs = records(tmp_path)
    wake = next(r for r in recs if r["type"] == "wake")
    assert wake["step"] == 0 and wake["reason"] == "level start"
    assert "Your model so far:\n(none yet)" in seen[0]


def test_model_text_carried_to_next_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_drive, "DRIVE_GAP_MOVES", 0)
    seen: list[str] = []
    agent = make(tmp_path, chat_of(lambda _p: {"model": "A4 moves me right"}, seen))
    play(FakeGame(player=(2, 2), target=(30, 2), walls=[], lava=[]), agent, max_steps=3)
    agent.close()
    assert len(seen) >= 2 and "Your model so far:\nA4 moves me right" in seen[1]


def test_unmodeled_choice_wakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_drive, "DRIVE_GAP_MOVES", 0)
    seen: list[str] = []
    agent = make(tmp_path, chat_of(lambda _p: {}, seen))
    play(FakeGame(player=(2, 2), target=(30, 2), walls=[], lava=[]), agent, max_steps=4)
    agent.close()
    reasons = [r["reason"] for r in records(tmp_path) if r["type"] == "wake"]
    assert reasons[0] == "level start"
    assert llm_drive.NO_MODELED_MOVE in reasons[1:]


def test_plan_played_out_runs_as_sim(tmp_path: Path) -> None:
    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {
            "facts": right_facts(p, t),
            "goals": [{"kind": "reach", "a": p, "b": t}],
            "plan": [{"action": "A4", "expect": "player right"}] * 6,
        }

    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make(tmp_path, chat_of(reply, []))
    play(game, agent, max_steps=10, until=lambda: game.levels >= 1)
    agent.close()
    assert game.levels == 1 and game.level_up_steps[0] == 3
    end = next(r for r in records(tmp_path) if r["type"] == "drive_end")
    assert end["modes"] == {"sim": 3} and end["claims"] == 2


def test_plan_runs_past_its_unknown_step(tmp_path: Path) -> None:
    # The first step the model can't predict is the experiment; the LLM's plan goes on
    # after it, and the next wake comes once no modeled move is left.
    calls: list[str] = []

    def reply(prompt: str) -> dict[str, Any]:
        if len(calls) > 1:
            return {}
        p, t = ids(prompt)
        plan = [{"action": a, "expect": "?"} for a in ("A4", "A1", "A4", "A4")]
        return {"facts": right_facts(p, t), "plan": plan}

    agent = make(tmp_path, chat_of(reply, calls))
    acts = play(FakeGame(player=(2, 9), target=(40, 2), walls=[], lava=[]), agent, max_steps=5)
    agent.close()
    assert [a.id for a in acts[:4]] == [4, 1, 4, 4]
    recs = records(tmp_path)
    wakes = [r["step"] for r in recs if r["type"] == "wake"]
    assert wakes[0] == 0 and not [w for w in wakes if 0 < w < 4]
    end = next(r for r in recs if r["type"] == "drive_end")
    assert end["modes"]["experiment"] == 1


def test_a_miss_waits_while_a_modeled_move_remains(tmp_path: Path) -> None:
    calls: list[str] = []

    def reply(prompt: str) -> dict[str, Any]:
        if len(calls) > 1:
            return {}
        p, t = ids(prompt)
        # A wrong claim: A4 is said to move the player 2 cells, it moves 1.
        facts = [
            {"action": "A4", "object": p, "effect": "move", "dx": 2, "dy": 0},
            {"action": "A4", "object": t, "effect": "none"},
        ]
        goals = [{"kind": "reach", "a": p, "b": t}]
        return {"facts": facts, "goals": goals, "plan": [{"action": "A4", "expect": "right"}] * 4}

    agent = make(tmp_path, chat_of(reply, calls))
    # After the miss the observed fact (1 cell) replaces the claim and search still
    # reaches the target, so no turn is needed for the next moves.
    play(FakeGame(player=(2, 2), target=(12, 2), walls=[], lava=[]), agent, max_steps=4)
    agent.close()
    wakes = [(r["step"], r["reason"]) for r in records(tmp_path) if r["type"] == "wake"]
    assert wakes == [(0, "level start")]


def test_search_failure_triggers_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fake_fix(*_a: Any, **_k: Any) -> FixResult:
        asked.append("fix")
        return FixResult(None, "no function in reply")

    monkeypatch.setattr(llm_drive, "try_fix", fake_fix)

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {
            "facts": right_facts(p, t),
            "goals": [{"kind": "reach", "a": p, "b": t}],
            "plan": [{"action": "A4", "expect": "player right"}] * 3,
        }

    agent = make(tmp_path, chat_of(reply, []))
    agent._search_fn = lambda *_a, **_k: SearchResult(None, (), (), 0, 0, "exhausted")  # pyright: ignore[reportPrivateUsage]
    play(FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[]), agent, max_steps=2)
    agent.close()
    recs = records(tmp_path)
    assert asked == ["fix"]
    fail = next(r for r in recs if r["type"] == "tool_fail")
    assert fail["tool"] == "search" and fail["got"] == "no plan"
    fix = next(r for r in recs if r["type"] == "tool_fix")
    assert fix["adopted"] is None and fix["reason"] == "no function in reply"


def test_no_chat_plays_like_fluid(tmp_path: Path) -> None:
    def game() -> FakeGame:
        return FakeGame(player=(2, 2), target=(12, 7), walls=[(5, 2), (5, 3)], lava=[(8, 7)])

    log = RunLog(tmp_path / "f.jsonl", "fake", "FLUID", 0)
    base = fluid.FluidAgent(
        "fake", CONFIGS["FLUID"], seed=0, log=log, game_deadline=1e12, run_deadline=1e12
    )
    drive = make(tmp_path, None)
    a = play(game(), base, max_steps=40)
    b = play(game(), drive, max_steps=40)
    base.close()
    drive.close()
    assert a == b


def test_failed_rule_is_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make(tmp_path, chat_of(lambda _p: {}, []))
    calls: list[str] = []
    worker = agent._worker  # pyright: ignore[reportPrivateUsage]
    assert worker is not None

    def hangs(*_a: Any) -> tuple[bool, Any]:
        calls.append("rule")
        return False, "timeout"

    monkeypatch.setattr(worker, "run_rule", hangs)
    agent._rule_src = "def rule(o, a):\n    while True: pass\n"  # pyright: ignore[reportPrivateUsage]
    rule = agent._rule_fn()  # pyright: ignore[reportPrivateUsage]
    assert rule is not None and rule([], {"id": 1, "x": 0, "y": 0}) is None
    assert agent._rule_fn() is None and calls == ["rule"]  # pyright: ignore[reportPrivateUsage]
    agent.close()


def test_search_plan_that_wins_becomes_a_regression_case(tmp_path: Path) -> None:
    TOOLBOX.clear()

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {"facts": right_facts(p, t), "goals": [{"kind": "reach", "a": p, "b": t}]}

    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make(tmp_path, chat_of(reply, []))
    play(game, agent, max_steps=10, until=lambda: game.levels >= 1)
    agent.choose(game.observe())  # the agent sees the level-up frame
    agent.close()
    assert game.levels == 1
    cases = TOOLBOX.cases("search")
    TOOLBOX.clear()
    assert len(cases) == 1 and cases[0].result.stop == "found"


def test_wake_right_after_an_llm_plan_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_drive, "DRIVE_GAP_MOVES", 0)
    calls: list[str] = []

    def reply(prompt: str) -> dict[str, Any]:
        if calls[1:]:
            return {}
        p, t = ids(prompt)
        return {"facts": right_facts(p, t), "plan": [{"action": "A4", "expect": "right"}]}

    agent = make(tmp_path, chat_of(reply, calls))
    play(FakeGame(player=(2, 9), target=(40, 2), walls=[], lava=[]), agent, max_steps=2)
    agent.close()
    steps = [r["step"] for r in records(tmp_path) if r["type"] == "wake"]
    assert steps[:2] == [0, 1]


def test_fix_replays_the_real_search_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def fake_fix(_chat: Any, failure: Any, *_a: Any, **_k: Any) -> FixResult:
        seen.append(failure)
        return FixResult(None, "x")

    monkeypatch.setattr(llm_drive, "try_fix", fake_fix)

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {
            "facts": right_facts(p, t),
            "goals": [{"kind": "reach", "a": p, "b": t}],
            "plan": [{"action": "A4", "expect": "right"}] * 3,
        }

    agent = make(tmp_path, chat_of(reply, []))
    agent._search_fn = lambda *_a, **_k: SearchResult(None, (), (), 0, 0, "exhausted")  # pyright: ignore[reportPrivateUsage]
    play(FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[]), agent, max_steps=1)
    agent.close()
    (failure,) = seen
    assert {"noop", "kin", "max_depth"} <= set(failure.kwargs)
    assert failure.kwargs["kin"].mem is failure.args[2]


def test_turn_timeout_backs_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.mind.test_fluid_agent import NeverPool  # noqa: PLC0415

    monkeypatch.setattr(llm_drive, "DRIVE_TURN_S", 0.01)
    log = RunLog(tmp_path / "d.jsonl", "fake", "DRIVE", 0)
    agent = llm_drive.DriveAgent(
        "fake",
        CONFIGS["FLUID-LLM"],
        seed=0,
        log=log,
        game_deadline=1e12,
        run_deadline=1e12,
        chat=chat_of(lambda _p: {}, []),
        pool=NeverPool(),
    )
    play(FakeGame(player=(2, 2), target=(30, 2), walls=[], lava=[]), agent, max_steps=4)
    agent.close()
    wakes = [r["step"] for r in records(tmp_path) if r["type"] == "wake"]
    assert wakes == [0]


def test_fix_turn_uses_the_fix_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    used: list[Any] = []

    def fake_fix(chat: Any, *_a: Any, **_k: Any) -> FixResult:
        used.append(chat)
        return FixResult(None, "x")

    monkeypatch.setattr(llm_drive, "try_fix", fake_fix)

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {
            "facts": right_facts(p, t),
            "goals": [{"kind": "reach", "a": p, "b": t}],
            "plan": [{"action": "A4", "expect": "right"}] * 3,
        }

    fix_chat = chat_of(lambda _p: {}, [])
    log = RunLog(tmp_path / "d.jsonl", "fake", "DRIVE", 0)
    agent = llm_drive.DriveAgent(
        "fake",
        CONFIGS["FLUID-LLM"],
        seed=0,
        log=log,
        game_deadline=1e12,
        run_deadline=1e12,
        chat=chat_of(reply, []),
        pool=SyncPool(),
        fix_chat=fix_chat,
    )
    agent._search_fn = lambda *_a, **_k: SearchResult(None, (), (), 0, 0, "exhausted")  # pyright: ignore[reportPrivateUsage]
    play(FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[]), agent, max_steps=1)
    agent.close()
    assert used == [fix_chat]


def test_unmodeled_moves_play_between_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After a turn with nothing in it the agent's own tests play; the next turn waits
    # for DRIVE_GAP_MOVES unmodeled moves (here the 4 "what does A_ do" tests).
    monkeypatch.setattr(llm_drive, "DRIVE_GAP_MOVES", 3)
    agent = make(tmp_path, chat_of(lambda _p: {}, []))
    play(FakeGame(player=(2, 9), target=(40, 2), walls=[], lava=[]), agent, max_steps=6)
    agent.close()
    wakes = [r["step"] for r in records(tmp_path) if r["type"] == "wake"]
    assert wakes == [0, 3]
