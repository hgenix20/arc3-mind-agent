"""Briefing: section order and titles, per-list caps, line clip, total cut, planner messages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arcagent.mind.briefing import (
    CUT_LINE,
    HORIZON_TOKEN,
    LINE_WIDTH,
    MAX_BRIEFING_CHARS,
    MODEL_TITLE,
    PLANNER_SYSTEM_PROMPT,
    SECTION_CAPS,
    SECTION_ORDER,
    SECTION_TITLES,
    Briefing,
    planner_messages,
    render,
    system_prompt,
)
from arcagent.mind.contract import MAX_MODEL_CHARS, QUESTION_IDS, SKILL_NAMES
from arcagent.mind.turn import DRIVE_SYSTEM_PROMPT
from arcagent.mind.vchat import VMessage

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

BASE = Briefing(
    state=("#1 kind k1 role controlled: A1 moves it up",),
    actions=("A1, A2, A3, A4, click #<id>",),
    goals=("level: reach #1 to #4 (open)", "game: unknown"),
    tasks=("1. reach #4",),
    level_rules=("A1 #1: moves 1 cell up [p 0.9]",),
    game_rules=("walls block movers [p 0.95]",),
    hypotheses=("H1 reach #4 via A1 A1 A3: p_path 0.81 ev 72.9",),
    sequence=("segment 2 of 3 steps, step 1 matched",),
    forks=("fork 1 at move 12: 2 untried paths",),
    flags=("check #7: value 0.6 cost 3",),
    self_focus=("self: agent, click no, character #1", "focus: level 2 goal g1 mode follow"),
    recent=("move 12 A1: predicted #1 up, observed #1 up",),
    lessons=("movers stop at walls (3 games, p 0.9)",),
    questions=(("Q-BLOCK", "nothing blocks the path"),),
    model="A1 moves the blue block up.",
    wake="level start",
    level=2,
    horizon=6,
)

LIST_FIELDS = [name for name in SECTION_ORDER if name != "questions"]


def _text(msgs: list[VMessage]) -> str:
    content = msgs[1]["content"]
    assert isinstance(content, list)
    return "".join(str(p["text"]) for p in content if p["type"] == "text")


def _images(msgs: list[VMessage]) -> int:
    content = msgs[1]["content"]
    assert isinstance(content, list)
    return sum(1 for p in content if p["type"] == "image_url")


def _section(text: str, title: str) -> str:
    """The lines under ``title`` up to the next section title."""
    starts = sorted(text.index(f"\n{t}:\n") for t in [*SECTION_TITLES.values(), MODEL_TITLE])
    begin = text.index(f"\n{title}:\n")
    later = [s for s in starts if s > begin]
    end = later[0] if later else len(text)
    return text[begin + len(f"\n{title}:\n") : end]


def test_section_caps_and_titles_cover_every_list_in_order() -> None:
    assert tuple(SECTION_CAPS) == SECTION_ORDER
    assert tuple(SECTION_TITLES) == SECTION_ORDER
    assert SECTION_ORDER == (
        "state",
        "actions",
        "goals",
        "tasks",
        "level_rules",
        "game_rules",
        "hypotheses",
        "sequence",
        "forks",
        "flags",
        "self_focus",
        "recent",
        "lessons",
        "questions",
    )
    assert SECTION_CAPS == {
        "state": 40,
        "actions": 1,
        "goals": 6,
        "tasks": 8,
        "level_rules": 16,
        "game_rules": 16,
        "hypotheses": 6,
        "sequence": 4,
        "forks": 5,
        "flags": 8,
        "self_focus": 2,
        "recent": 12,
        "lessons": 8,
        "questions": 7,
    }
    assert MODEL_TITLE == "Your notes from the last turn (unverified)"
    assert MAX_BRIEFING_CHARS == 12000 and LINE_WIDTH == 160


def test_render_keeps_header_then_sections_in_order_then_model_last() -> None:
    text = render(BASE)
    assert text.startswith("Wake: level start\nLevel: 2\nHorizon: 6")
    titles = [*SECTION_TITLES.values(), MODEL_TITLE]
    positions = [text.index(f"\n{t}:\n") for t in titles]
    assert positions == sorted(positions)
    assert _section(text, MODEL_TITLE).strip() == "A1 moves the blue block up."
    assert _section(text, SECTION_TITLES["state"]).strip() == BASE.state[0]
    assert not text.endswith(CUT_LINE)


@pytest.mark.parametrize("field", LIST_FIELDS)
def test_each_list_clips_at_its_cap_with_a_more_line(field: str) -> None:
    cap = SECTION_CAPS[field]
    lines = tuple(f"{field} line {i:03d}" for i in range(cap + 3))
    text = render(replace(BASE, **{field: lines}))
    body = _section(text, SECTION_TITLES[field]).strip().split("\n")
    assert body == [*lines[:cap], "(+3 more)"]


def test_list_at_its_cap_has_no_more_line() -> None:
    cap = SECTION_CAPS["tasks"]
    lines = tuple(f"task {i}" for i in range(cap))
    body = _section(render(replace(BASE, tasks=lines)), SECTION_TITLES["tasks"]).strip()
    assert body.split("\n") == list(lines)


def test_questions_render_as_id_colon_answer_and_clip_at_seven() -> None:
    qs = tuple((f"Q-{i}", f"answer {i}") for i in range(10))
    body = _section(render(replace(BASE, questions=qs)), SECTION_TITLES["questions"]).strip()
    assert body.split("\n") == [f"Q-{i}: answer {i}" for i in range(7)] + ["(+3 more)"]


def test_empty_lists_and_empty_model_render_none() -> None:
    empty = replace(BASE, forks=(), questions=(), model="")
    text = render(empty)
    assert _section(text, SECTION_TITLES["forks"]).strip() == "(none)"
    assert _section(text, SECTION_TITLES["questions"]).strip() == "(none)"
    assert _section(text, MODEL_TITLE).strip() == "(none)"


def test_each_line_is_clipped_to_the_width() -> None:
    long = "x" * 500
    text = render(replace(BASE, lessons=(long,), questions=(("Q-HARM", long),), wake=long))
    lesson = _section(text, SECTION_TITLES["lessons"]).strip()
    assert len(lesson) == LINE_WIDTH and lesson.endswith("...")
    question = _section(text, SECTION_TITLES["questions"]).strip()
    assert len(question) == LINE_WIDTH and question.startswith("Q-HARM: ")
    assert len(text.split("\n")[0]) <= len("Wake: ") + LINE_WIDTH


def test_model_text_is_kept_whole_up_to_its_cap() -> None:
    notes = "line one\nline two " + "m" * 3000
    body = _section(render(replace(BASE, model=notes)), MODEL_TITLE).strip()
    assert body.startswith("line one\nline two")
    assert len(body) == MAX_MODEL_CHARS


def test_whole_text_is_cut_at_the_cap_with_a_final_line() -> None:
    full = {name: tuple("w" * 200 for _ in range(SECTION_CAPS[name])) for name in LIST_FIELDS}
    big = replace(BASE, **full, questions=tuple((f"Q-{i}", "a" * 200) for i in range(7)))
    text = render(big)
    assert len(text) <= MAX_BRIEFING_CHARS
    assert text.endswith(f"\n{CUT_LINE}")
    assert text.startswith("Wake: level start")
    assert "\n" + "w" * 157 + "...\n" in text  # whole lines survive the cut


def test_cut_falls_on_a_line_boundary_and_a_small_cap_works() -> None:
    text = render(BASE, cap_chars=300)
    assert len(text) <= 300 and text.endswith(f"\n{CUT_LINE}")
    before_cut = text[: -len(f"\n{CUT_LINE}")]
    assert before_cut.split("\n")[-1] in render(BASE).split("\n")


def test_planner_messages_two_images_with_prev_grid_one_without() -> None:
    grid = np.zeros((64, 64), dtype=np.int8)
    prev = np.ones((64, 64), dtype=np.int8)
    with_prev = planner_messages(BASE, grid, prev)
    without = planner_messages(BASE, grid, None)
    assert _images(with_prev) == 2 and _images(without) == 1
    assert [m["role"] for m in with_prev] == ["system", "user"]
    assert _text(with_prev) == render(BASE)


def test_planner_messages_substitute_the_horizon() -> None:
    grid = np.zeros((64, 64), dtype=np.int8)
    msgs = planner_messages(replace(BASE, horizon=9), grid, None)
    system = msgs[0]["content"]
    assert isinstance(system, str)
    assert HORIZON_TOKEN not in system and "up to 9 steps" in system
    assert system == system_prompt(9)
    assert HORIZON_TOKEN in PLANNER_SYSTEM_PROMPT


def test_planner_prompt_states_the_contract_and_names_no_known_game() -> None:
    low = PLANNER_SYSTEM_PROMPT.lower()
    for name in KNOWN_GAMES:
        assert name not in low, name
    assert len(PLANNER_SYSTEM_PROMPT) < 2200
    em_dash, en_dash = chr(0x2014), chr(0x2013)
    assert em_dash not in PLANNER_SYSTEM_PROMPT and en_dash not in PLANNER_SYSTEM_PROMPT
    for key in ("model", "facts", "goals", "segment", "skill", "question", "labels", "notes"):
        assert f'"{key}"' in PLANNER_SYSTEM_PROMPT, key
    for name in SKILL_NAMES:
        assert name in PLANNER_SYSTEM_PROMPT, name
    for qid in QUESTION_IDS:
        assert qid in PLANNER_SYSTEM_PROMPT, qid
    for title in SECTION_TITLES.values():
        assert title.lower() in low, title
    assert "unverified" in low and "squared" in low
    assert "this turn only" in low and "before any real move" in low
    assert "failing step" in low and "not both" in low


def test_planner_prompt_keeps_the_drive_fact_and_goal_shapes() -> None:
    for fragment in (
        '"action": "A1" or "click #12", "object": <id> or "any"',
        '"effect": "move" (with "dx", "dy" in cells; needs a real object id), "none", "disappear",',
        '"recolor" (with "color"), "game_over" or "level_up"}',
        '{"kind": "reach", "a": <id>, "b": <id>}',
        '{"kind": "clear", "color": c}, {"kind": "count", "color": c, "n": n}',
        '{"kind": "match", "region_a": [x0,y0,x1,y1], "region_b": [x0,y0,x1,y1]}',
    ):
        assert fragment in PLANNER_SYSTEM_PROMPT, fragment
        assert fragment in DRIVE_SYSTEM_PROMPT, fragment
