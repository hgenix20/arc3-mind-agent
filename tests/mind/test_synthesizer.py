"""Synthesizer and reply store: a fake chat stands in for the model server."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import requests

from arcagent.core.types import GRID_SIZE, Action, Grid, Phase
from arcagent.mind.chat_store import REQUEST_META, recording_chat, replay_chat
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.synthesizer import GOAL_HINT, Synthesizer
from arcagent.mind.types import Program, Step
from arcagent.synth.prompt import Message
from arcagent.synth.render import render_grid
from arcagent.synth.run import Chat, Completion, request_seed
from arcagent.synth.verify import IDENTITY_CODE

# Writes the action id into cell (0, 0): the rule the steps below obey.
GOOD = "def step(grid, action, x=-1, y=-1):\n    g=[r[:] for r in grid]; g[0][0]=action; return g\n"
GOOD_GOAL = GOOD + "def goal(grid):\n    return grid[0][0] == 3\n"
# Right on actions 1 and 2 only, so 2 of 4 steps hit: held, not admitted.
PARTIAL = (
    "def step(grid, action, x=-1, y=-1):\n"
    "    g=[r[:] for r in grid]\n"
    "    if action in (1, 2):\n        g[0][0]=action\n"
    "    return g\n"
)
BROKEN = "def step(grid, action, x=-1, y=-1):\n    return None\n"


def fence(code: str) -> str:
    return f"Here it is.\n```python\n{code}```\n"


def completion(text: str) -> Completion:
    return Completion(text, 10, 20, 0.5, "stop", 0)


def zeros() -> Grid:
    return np.zeros((GRID_SIZE, GRID_SIZE), np.int8)


def make_steps(n: int = 4) -> list[Step]:
    out: list[Step] = []
    for i in range(n):
        after = zeros()
        action = i + 1
        after[0, 0] = action
        out.append(Step(i, zeros(), Action(action), after, 0, 0, Phase.NOT_FINISHED))
    return out


@pytest.fixture
def sandbox() -> Iterator[ProgramSandbox]:
    with ProgramSandbox() as sb:
        yield sb


def scripted(replies: list[str]) -> tuple[Chat, list[list[Message]]]:
    seen: list[list[Message]] = []

    def chat(messages: list[Message], seed: int) -> Completion:
        seen.append(messages)
        return completion(replies[(len(seen) - 1) % len(replies)])

    return chat, seen


def test_propose_admits_good_holds_partial_drops_broken(sandbox: ProgramSandbox) -> None:
    chat, _ = scripted([fence(GOOD_GOAL), fence(PARTIAL), fence(BROKEN)])
    res = Synthesizer(chat, k=3).propose("g1", 2, make_steps(), 3, [], None, sandbox, seed=7)
    assert res.error is None
    assert res.calls == 3
    assert res.prompt_tokens == 30 and res.completion_tokens == 60
    assert [(p.id, g.verdict) for p, g, _, _ in res.programs] == [
        ("g1-c2-k0", "pass"),
        ("g1-c2-k1", "hold"),
    ]
    assert [pid for pid, _ in res.dropped] == ["g1-c2-k2"]
    assert "below hold" in res.dropped[0][1]
    good, partial = res.programs[0], res.programs[1]
    assert good[0].has_goal and good[0].origin == "llm" and (good[2], good[3]) == (4, 4)
    assert not partial[0].has_goal and (partial[2], partial[3]) == (2, 4)
    # Admitted and held programs stay loaded; the dropped one is gone.
    assert sandbox.score("g1-c2-k0", make_steps()) == (4, 4)
    assert sandbox.score("g1-c2-k1", make_steps()) == (2, 4)
    assert "g1-c2-k2" not in sandbox._codes  # pyright: ignore[reportPrivateUsage]  # no public loaded-set


def test_server_error_yields_no_programs_and_error_text(sandbox: ProgramSandbox) -> None:
    def chat(messages: list[Message], seed: int) -> Completion:
        raise requests.ConnectionError("server down")

    res = Synthesizer(chat, k=2).propose("g", 1, make_steps(), 3, [], None, sandbox, seed=1)
    assert res.programs == []
    assert res.error is not None and "server down" in res.error
    assert res.calls == 2


def test_no_fence_reply_is_dropped(sandbox: ProgramSandbox) -> None:
    chat, _ = scripted(["I think the rule is to add one."])
    res = Synthesizer(chat, k=2).propose("g", 1, make_steps(), 3, [], None, sandbox, seed=1)
    assert res.programs == []
    assert res.error is not None and "no code" in res.error


def test_goal_hint_includes_evidence_grids(sandbox: ProgramSandbox) -> None:
    before, after = zeros(), zeros()
    before[5, 5] = 7
    after[5, 5] = 7
    after[9, 9] = 7
    chat, seen = scripted([fence(GOOD)])
    synth = Synthesizer(chat, k=1)
    synth.propose("g", 1, make_steps(), 3, [(before, after)], None, sandbox, seed=1)
    first_user = seen[0][1]["content"]
    assert GOAL_HINT in first_user
    assert render_grid(before) in first_user and render_grid(after) in first_user
    assert "goal(" in GOAL_HINT
    # No evidence: the hint is there, without grids.
    seen.clear()
    synth.propose("g", 2, make_steps(), 3, [], None, sandbox, seed=1)
    assert GOAL_HINT in seen[0][1]["content"]
    assert render_grid(before) not in seen[0][1]["content"]


def test_repair_turn_carries_prior_code_and_missed_steps(sandbox: ProgramSandbox) -> None:
    steps = make_steps()
    prior = Program("g-c1-k0", PARTIAL, False, "llm")
    assert sandbox.load(prior.id, prior.code) is None
    chat, seen = scripted([fence(GOOD)])
    res = Synthesizer(chat, k=1).propose("g", 2, steps, 3, [], (prior, steps[2:]), sandbox, seed=1)
    msgs = seen[0]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "train #2" in msgs[3]["content"] and "train #3" in msgs[3]["content"]
    assert res.programs and res.programs[0][1].verdict == "pass"


def test_record_then_replay_round_trip(tmp_path: Path, sandbox: ProgramSandbox) -> None:
    chat, _ = scripted([fence(GOOD)])
    rec = recording_chat(chat, tmp_path)
    Synthesizer(rec, k=2).propose("g", 3, make_steps(), 3, [], None, sandbox, seed=5)
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2
    assert not list(tmp_path.glob("*.tmp"))
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(doc) == {
        "messages_sha1",
        "seed",
        "game",
        "call",
        "k",
        "text",
        "prompt_tokens",
        "completion_tokens",
        "seconds",
        "finish_reason",
    }
    assert doc["game"] == "g" and doc["call"] == 3
    assert REQUEST_META.get() is None  # reset after each call
    replay = replay_chat(tmp_path)
    with ProgramSandbox() as sb2:
        res = Synthesizer(replay, k=2).propose("g", 3, make_steps(), 3, [], None, sb2, seed=5)
    assert [g.verdict for _, g, _, _ in res.programs] == ["pass", "pass"]


def test_replay_falls_back_to_game_call_k_then_identity(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    chat, _ = scripted([fence(GOOD)])
    rec = recording_chat(chat, tmp_path)
    Synthesizer(rec, k=1).propose("g", 3, make_steps(), 3, [], None, sandbox, seed=5)
    replay = replay_chat(tmp_path)
    msgs: list[Message] = [{"role": "user", "content": "different prompt"}]
    # Same (game, call, k) in the context, different hash: the recorded reply.
    token = REQUEST_META.set({"game": "g", "call": 3, "k": 0})
    try:
        assert "g[0][0]=action" in replay(msgs, 999).text
    finally:
        REQUEST_META.reset(token)
    # No match anywhere: the identity program.
    token = REQUEST_META.set({"game": "other", "call": 1, "k": 0})
    try:
        assert IDENTITY_CODE in replay(msgs, 999).text
    finally:
        REQUEST_META.reset(token)
    assert IDENTITY_CODE in replay(msgs, 999).text  # no metadata at all


def test_requests_run_concurrently_with_correct_metadata(
    tmp_path: Path, sandbox: ProgramSandbox
) -> None:
    threads: set[int] = set()

    def slow(messages: list[Message], seed: int) -> Completion:
        threads.add(threading.get_ident())
        time.sleep(0.3)
        return completion(fence(GOOD))

    rec = recording_chat(slow, tmp_path)
    t0 = time.perf_counter()
    res = Synthesizer(rec, k=4).propose("g", 5, make_steps(), 3, [], None, sandbox, seed=2)
    assert time.perf_counter() - t0 < 0.9
    assert len(threads) > 1
    assert [p.id for p, _, _, _ in res.programs] == [f"g-c5-k{i}" for i in range(4)]
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in tmp_path.glob("*.json")]
    assert sorted((d["game"], d["call"], d["k"]) for d in docs) == [("g", 5, i) for i in range(4)]
    assert all(d["seed"] == request_seed(2, 5, d["k"]) for d in docs)


def test_error_text_is_truncated(sandbox: ProgramSandbox) -> None:
    def chat(messages: list[Message], seed: int) -> Completion:
        raise RuntimeError("x" * 5000)

    res = Synthesizer(chat, k=2).propose("g", 1, make_steps(), 3, [], None, sandbox, seed=1)
    assert res.error is not None and len(res.error) <= 2 * 300 + 2
