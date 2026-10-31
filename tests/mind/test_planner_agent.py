"""PlannerAgent (spec v2 section 3) end to end on the fake games with scripted planner replies.

The chat is a function that reads the briefing text, so each reply can name the player
and the target by the ids the briefing gives them. ``SyncPool`` runs the call inside the
move that asked, as ``tests/mind/test_llm_drive.py`` does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arcagent.mind import fluid, planner_agent
from arcagent.mind.budget import Counter
from arcagent.mind.config import CONFIGS
from arcagent.mind.longterm import LongTermStore
from arcagent.mind.planner_agent import PlannerAgent
from arcagent.mind.runlog import RunLog
from arcagent.mind.vchat import VisionChat, VMessage
from arcagent.synth.run import Completion
from tests.mind.fakegames import FakeGame, play
from tests.mind.test_fluid_agent import SyncPool

Rec = dict[str, Any]
Replier = Callable[[str], dict[str, Any]]
STEPS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _prompt(messages: list[VMessage]) -> str:
    content = messages[1]["content"]
    assert isinstance(content, list)
    return str(content[0]["text"])


def ids(prompt: str) -> tuple[int, int]:
    """The player's (color 1) and the target's (color 4) ids in a briefing."""
    player = re.search(r"#(\d+) c1 ", prompt)
    target = re.search(r"#(\d+) c4 ", prompt)
    assert player is not None and target is not None, prompt
    return int(player.group(1)), int(target.group(1))


def chat_of(
    replier: Replier,
    log: list[str],
    *,
    tokens: int = 0,
    tick: Callable[[], None] | None = None,
) -> VisionChat:
    """A chat that logs each briefing text, runs ``tick`` (a fake delay) and answers."""

    def call(messages: list[VMessage], seed: int) -> Completion:
        del seed
        text = _prompt(messages)
        log.append(text)
        if tick is not None:
            tick()
        return Completion(json.dumps(replier(text)), 0, tokens, 0.0)

    return call


def make(
    tmp_path: Path,
    chat: VisionChat | None,
    *,
    name: str = "p",
    store: LongTermStore | None = None,
    clock: Callable[[], float] | None = None,
) -> PlannerAgent:
    log = RunLog(tmp_path / f"{name}.jsonl", "fake", "PLANNER", 0)
    extra: dict[str, Any] = {} if clock is None else {"clock": clock}
    return PlannerAgent(
        "fake",
        CONFIGS["FLUID-LLM"],
        seed=0,
        log=log,
        game_deadline=1e12,
        run_deadline=1e12,
        chat=chat,
        pool=SyncPool() if chat is not None else None,
        store=store,
        **extra,
    )


def records(tmp_path: Path, name: str = "p") -> list[Rec]:
    lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def of_type(recs: list[Rec], kind: str) -> list[Rec]:
    return [r for r in recs if r["type"] == kind]


def move_claims(p: int, *, skip: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    """The four moves as ``move`` claims on the player, less the ids in ``skip``."""
    return [
        {"action": f"A{i}", "object": p, "effect": "move", "dx": dx, "dy": dy}
        for i, (dx, dy) in STEPS.items()
        if i not in skip
    ]


def still_claims(t: int) -> list[dict[str, Any]]:
    """The target stays put under every move. Without these the predictor lends the
    one-cell target the one-cell player's move fact (same shape) and walks it away."""
    return [{"action": f"A{i}", "object": t, "effect": "none"} for i in STEPS]


def claims(p: int, t: int, *, skip: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    return [*move_claims(p, skip=skip), *still_claims(t)]


def steps(*names: str) -> list[dict[str, str]]:
    return [{"action": n, "expect": "player moves"} for n in names]


def reach(p: int, t: int) -> dict[str, Any]:
    return {"kind": "reach", "a": p, "b": t}


def test_a_valid_planner_segment_wins_the_level(tmp_path: Path) -> None:
    seen: list[str] = []

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {"facts": claims(p, t), "goals": [reach(p, t)], "segment": steps("A4", "A4", "A4")}

    store = LongTermStore(tmp_path / "longterm.json")
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make(tmp_path, chat_of(reply, seen), store=store)
    play(game, agent, max_steps=6, until=lambda: game.levels >= 1)
    agent.choose(game.observe())  # the agent sees the level-up frame
    agent.close()
    assert game.levels == 1 and game.level_up_steps[0] <= 3 + 2
    recs = records(tmp_path)
    calls = of_type(recs, "planner_call")
    assert calls and calls[0]["step"] == 0 and calls[0]["segment_len"] == 3
    seg = of_type(recs, "segment")[0]
    assert seg["source"] == "planner" and seg["length"] == 3 and seg["step"] == 0
    acts = of_type(recs, "action")
    assert acts[0]["segment_id"] == seg["id"] and acts[0]["p_path"] == seg["p_path"]
    assert all(0.0 < a["p_step"] <= 1.0 for a in acts[:3])
    assert acts[0]["prev_matched"] is None and acts[1]["prev_matched"] is True
    assert acts[0]["gate"]["trip"] is None and acts[0]["self_rec"]["click"] is False
    assert "Wake: level start" in seen[0] and "Q-BLOCK" in seen[0]
    assert not [r for r in recs if r["type"] in ("agent_error", "exec_failed")]


def test_a_deadly_segment_is_rejected_once_and_the_second_reply_followed(tmp_path: Path) -> None:
    seen: list[str] = []

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        if len(seen) == 1:
            # The claim says A2 ends the game; the first segment starts with it.
            facts = [
                {"action": "A2", "object": p, "effect": "game_over"},
                *claims(p, t, skip=(2,)),
            ]
            return {
                "facts": facts,
                "goals": [reach(p, t)],
                "segment": steps("A2", "A4", "A4", "A4"),
            }
        return {"segment": steps("A4", "A4", "A4")}

    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[(2, 3)])
    agent = make(tmp_path, chat_of(reply, seen))
    play(game, agent, max_steps=6, until=lambda: game.levels >= 1)
    agent.close()
    assert game.levels == 1 and game.game_overs == 0
    recs = records(tmp_path)
    calls = of_type(recs, "planner_call")
    assert [c["step"] for c in calls] == [0, 0]
    assert any("rejected at step 0" in d and "death" in d for d in calls[0]["dropped"])
    assert "segment rejected at step 0" in seen[1]
    segs = of_type(recs, "segment")
    assert len(segs) == 1 and segs[0]["source"] == "planner" and segs[0]["length"] == 3


def test_a_broken_segment_hands_off_and_returns_to_the_fork(tmp_path: Path) -> None:
    seen: list[str] = []

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        plan = steps("A1", "A4", "A4", "A4", "A2")  # around the wall, above it
        return {"facts": claims(p, t), "goals": [reach(p, t)], "segment": plan}

    # The wall is unknown at first, so the code's search walks straight through it in
    # three moves; the planner's five-move path is the fork's other branch. Two cells
    # give the wall its own shape, so it borrows no move fact from the player.
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[(3, 2), (3, 3)], lava=[])
    agent = make(tmp_path, chat_of(reply, seen))
    play(game, agent, max_steps=12, until=lambda: game.levels >= 1)
    agent.close()
    recs = records(tmp_path)
    fork = of_type(recs, "fork_open")[0]
    assert fork["step"] == 0 and fork["paths"] == 2
    seg = of_type(recs, "segment")[0]
    assert seg["source"] == "search" and seg["length"] == 3
    acts = of_type(recs, "action")
    assert acts[0]["action"][0] == 4 and acts[0]["fork_id"] == fork["id"]
    hand = of_type(recs, "handoff")[0]
    assert hand["step"] == 1 and hand["rule"] == "H-b" and hand["method"] == "return_to_fork"
    failed = of_type(recs, "branch_failed")[0]
    assert failed["step"] == 1 and failed["fork"] == fork["id"] and failed["index"] == 0
    route = of_type(recs, "return_route")[0]
    assert route["step"] == 1 and route["fork"] == fork["id"] and route["length"] == 0
    assert [a["action"][0] for a in acts[1:6]] == [1, 4, 4, 4, 2]
    assert game.levels == 1


def test_a_planner_call_past_the_move_gate_trips_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(planner_agent, "T_MOVE", 10.0)
    now = [0.0]
    seen: list[str] = []

    def tick() -> None:
        now[0] += 30.0  # the call takes 30 seconds of the injected clock

    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {"facts": claims(p, t), "goals": [reach(p, t)], "segment": steps("A4", "A4", "A4")}

    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make(tmp_path, chat_of(reply, seen, tick=tick), clock=lambda: now[0])
    play(game, agent, max_steps=2)
    agent.close()
    recs = records(tmp_path)
    trip = of_type(recs, "gate_trip")[0]
    assert trip["step"] == 0 and trip["gate"] == "time" and trip["seconds"] >= 30.0
    acts = of_type(recs, "action")
    assert acts[0]["branch"] == "fallback" and acts[0]["gate"]["trip"] == "time"
    hand = of_type(recs, "handoff")[0]
    assert hand["step"] == 1 and hand["rule"] == "H-a" and hand["method"] == "fallback"
    assert acts[1]["branch"] == "fallback"
    assert len(seen) == 1


def test_a_reply_over_the_token_gate_trips(tmp_path: Path) -> None:
    def reply(prompt: str) -> dict[str, Any]:
        p, t = ids(prompt)
        return {"facts": claims(p, t), "goals": [reach(p, t)], "segment": steps("A4", "A4", "A4")}

    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = make(tmp_path, chat_of(reply, [], tokens=planner_agent.N_TOKEN + 1))
    play(game, agent, max_steps=1)
    agent.close()
    recs = records(tmp_path)
    trip = of_type(recs, "gate_trip")[0]
    assert trip["gate"] == "tokens" and trip["tokens"] == planner_agent.N_TOKEN + 1
    act = of_type(recs, "action")[0]
    assert act["branch"] == "fallback" and act["gate"]["tokens"] == planner_agent.N_TOKEN + 1


def test_no_chat_plays_like_fluid(tmp_path: Path) -> None:
    def maze() -> FakeGame:
        return FakeGame(player=(2, 2), target=(12, 7), walls=[(5, 2), (5, 3)], lava=[(8, 7)])

    log = RunLog(tmp_path / "f.jsonl", "fake", "FLUID", 0)
    base = fluid.FluidAgent(
        "fake", CONFIGS["FLUID"], seed=0, log=log, game_deadline=1e12, run_deadline=1e12
    )
    planner = make(tmp_path, None)
    a = play(maze(), base, max_steps=40)
    b = play(maze(), planner, max_steps=40)
    base.close()
    planner.close()
    assert a == b
    recs = records(tmp_path)
    acts = of_type(recs, "action")
    assert {"self_rec", "focus", "gate", "p_step", "p_path", "prev_matched", "segment_id"} <= set(
        acts[0]
    )
    assert not of_type(recs, "planner_call") and not of_type(recs, "handoff")

    open_game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    solo = make(tmp_path, None, name="solo")
    play(open_game, solo, max_steps=40, until=lambda: open_game.levels >= 1)
    solo.close()
    assert open_game.levels == 1


def test_health_bar_rule() -> None:
    assert planner_agent.is_health_bar(drops=2, moves=10)
    assert planner_agent.is_health_bar(drops=0, moves=10)
    assert not planner_agent.is_health_bar(drops=3, moves=10)  # a move counter drops on most moves
    assert not planner_agent.is_health_bar(drops=1, moves=9)  # too few moves watched


def test_bar_cells_reads_the_counter_band() -> None:
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[0, :20] = 7
    grid[5, :10] = 7  # outside the band, never counted
    row = Counter(color=7, axis="row", lines=frozenset({0}), shrinks=True)
    col = Counter(color=7, axis="col", lines=frozenset({0}), shrinks=True)
    assert planner_agent.bar_cells(grid, row) == 20
    assert planner_agent.bar_cells(grid, col) == 2
