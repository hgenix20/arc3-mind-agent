"""Executive turn: prompt content and size, image parts, salience cap, error handling."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from arcagent.core.types import Action
from arcagent.mind.objects import Scene, Thing, shape_of
from arcagent.mind.turn import (
    MAX_OBJECTS,
    MAX_PROMPT_CHARS,
    MOVE_LOG_LINES,
    SYSTEM_PROMPT,
    TurnInput,
    build_messages,
    take_turn,
)
from arcagent.mind.vchat import ImagesRejected, VMessage, scripted
from arcagent.synth.run import Completion

KNOWN_GAMES = [
    "tetris",
    "sokoban",
    "breakout",
    "snake",
    "pac-man",
    "pacman",
    "zelda",
    "mario",
    "pong",
    "minesweeper",
    "lights out",
    "2048",
    "chess",
]


def _thing(i: int, color: int, x: int, y: int, *, w: int = 2, h: int = 2) -> Thing:
    cells = frozenset((x + dx, y + dy) for dx in range(w) for dy in range(h))
    return Thing(i, color, shape_of(cells), f"c{color}:{i:010d}", x, y, cells)


def _scene(things: list[Thing]) -> Scene:
    return Scene(
        tuple(things),
        background=0,
        blocked=frozenset(),
        terrain=np.zeros((64, 64), dtype=np.int8),
    )


def _inp(
    n: int = 3,
    *,
    prev: bool = True,
    movers: frozenset[str] = frozenset(),
    changed: frozenset[int] = frozenset(),
    scene: Scene | None = None,
) -> TurnInput:
    grid = np.zeros((64, 64), dtype=np.int8)
    prev_grid = np.ones((64, 64), dtype=np.int8) if prev else None
    sc = scene or _scene(
        [_thing(i, 1 + i % 9, (i * 3) % 60, (i * 5) % 60) for i in range(1, n + 1)]
    )
    return TurnInput(
        grid=grid,
        prev_grid=prev_grid,
        scene=sc,
        available=(1, 2, 3, 4, 6),
        facts=("A1 moves c1 left",),
        goals=(("reach:a:b", "open"),),
        questions=("does clicking toggle?",),
        moves=("A1", "A2"),
        level=2,
        wake="level start",
        movers=movers,
        changed=changed,
    )


def _text(msgs: list[VMessage]) -> str:
    content = msgs[1]["content"]
    assert isinstance(content, list)
    return "".join(str(p["text"]) for p in content if p["type"] == "text")


def _images(msgs: list[VMessage]) -> int:
    content = msgs[1]["content"]
    assert isinstance(content, list)
    return sum(1 for p in content if p["type"] == "image_url")


def test_two_images_with_prev_grid_one_without() -> None:
    assert _images(build_messages(_inp(prev=True))) == 2
    assert _images(build_messages(_inp(prev=False))) == 1


def test_text_under_cap_with_300_objects() -> None:
    things = [_thing(i, i % 16, i % 60, (i * 7) % 60, w=1 + i % 3) for i in range(1, 301)]
    inp = _inp(scene=_scene(things))
    text = _text(build_messages(inp))
    assert len(text) < MAX_PROMPT_CHARS
    assert text.count("\n#") + text.startswith("#") <= MAX_OBJECTS + 1
    assert "and 260 more" in text


def test_text_under_cap_with_huge_facts_and_moves() -> None:
    inp = _inp()
    big = replace(
        inp,
        facts=tuple("f" * 500 for _ in range(100)),
        questions=tuple("q" * 500 for _ in range(100)),
        moves=tuple("m" * 500 for _ in range(100)),
        goals=tuple((f"g{i}" * 50, "open") for i in range(100)),
    )
    assert len(_text(build_messages(big))) <= MAX_PROMPT_CHARS


def test_system_prompt_names_no_known_game() -> None:
    low = SYSTEM_PROMPT.lower()
    for name in KNOWN_GAMES:
        assert name not in low, name
    for root in ("move", "block", "push", "collect", "toggle", "die", "win", "spawn"):
        assert root in low
    assert SYSTEM_PROMPT.count("\n") < 40
    assert "\u2014" not in SYSTEM_PROMPT


def test_salience_order_movers_changed_then_small() -> None:
    things = [
        _thing(1, 1, 0, 0, w=1, h=1),
        _thing(2, 2, 5, 5, w=6, h=6),  # big mover
        _thing(3, 3, 12, 12, w=5, h=5),  # big, changed
        _thing(4, 4, 20, 20, w=1, h=2),
        _thing(5, 5, 30, 30, w=1, h=1),
    ]
    inp = _inp(scene=_scene(things), movers=frozenset({things[1].sig}), changed=frozenset({3}))
    text = _text(build_messages(inp))
    order = [int(m) for m in re.findall(r"^#(\d+) ", text, re.MULTILINE)]
    assert order == [2, 3, 1, 5, 4]


def test_cap_keeps_movers_and_summarizes_rest_by_color() -> None:
    things = [_thing(i, 3 if i % 2 else 9, i % 60, (i * 7) % 60) for i in range(1, 101)]
    mover = things[99]  # largest id, would sort last otherwise
    inp = _inp(scene=_scene(things), movers=frozenset({mover.sig}))
    text = _text(build_messages(inp))
    assert f"#{mover.id} " in text
    assert "and 60 more" in text
    assert "c3 x" in text and "c9 x" in text


def test_move_log_limited() -> None:
    inp = _inp()
    long_moves = replace(
        inp,
        moves=tuple(f"MOVE{i:03d}" for i in range(30)),
    )
    text = _text(build_messages(long_moves))
    assert f"MOVE{29:03d}" in text
    assert f"MOVE{30 - MOVE_LOG_LINES - 1:03d}" not in text
    assert f"MOVE{30 - MOVE_LOG_LINES:03d}" in text


def test_take_turn_parses_valid_reply() -> None:
    inp = _inp()
    reply = json.dumps({"plan": [{"action": "A1", "expect": "moves"}], "notes": "n"})
    r = take_turn(scripted([reply]), inp, seed=3)
    assert [p.action for p in r.plan] == [Action(1)]
    assert r.notes == "n"


def test_take_turn_bad_json_gives_empty_reply() -> None:
    r = take_turn(scripted(["not json"]), _inp(), seed=3)
    assert r.plan == () and r.goals == () and r.test is None


def test_take_turn_passes_seed_and_messages() -> None:
    seen: list[tuple[int, int]] = []

    def chat(messages: list[VMessage], seed: int) -> Completion:
        seen.append((len(messages), seed))
        return Completion(text="{}", prompt_tokens=0, completion_tokens=0, seconds=0.0)

    take_turn(chat, _inp(), seed=77)
    assert seen == [(2, 77)]


def test_take_turn_chat_error_returns_empty_with_reason() -> None:
    def chat(messages: list[VMessage], seed: int) -> Completion:
        raise ConnectionError("boom")

    r = take_turn(chat, _inp(), seed=1)
    assert r.plan == () and any("boom" in d for d in r.dropped)


def test_take_turn_images_rejected_propagates() -> None:
    def chat(messages: list[VMessage], seed: int) -> Completion:
        raise ImagesRejected("no images")

    with pytest.raises(ImagesRejected):
        take_turn(chat, _inp(), seed=1)
