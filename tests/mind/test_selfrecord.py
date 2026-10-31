"""Tests for the self record and the focus record (spec 4.5a and 4.5b)."""

from __future__ import annotations

import dataclasses

from arcagent.mind.selfrecord import LINE_CHARS, FocusRecord, SelfRecord

ALL_ACTIONS = (1, 2, 3, 4, 5, 6)
MOVES_ONLY = (1, 2, 3, 4)
MOVER = "kc3:ab12cdef01"
GOAL = "reach c1:ab12cdef01>c4:cd34ef5678"
FOCUS_FIELDS = {
    "level",
    "goal_id",
    "goal_status",
    "target",
    "mode",
    "budget_left",
    "verdict",
    "fallbacks_in_row",
    "segment_left",
    "tokens_this_move",
    "seconds_this_move",
}


def focus(
    *,
    goal_id: str | None = GOAL,
    goal_status: str | None = "open",
    target: str | None = "kc4:cd34ef5678",
    budget_left: int | None = 41,
    tokens_this_move: int = 812,
    seconds_this_move: float = 1.26,
) -> FocusRecord:
    return FocusRecord(
        level=2,
        goal_id=goal_id,
        goal_status=goal_status,
        target=target,
        mode="follow",
        budget_left=budget_left,
        verdict="progress",
        fallbacks_in_row=0,
        segment_left=4,
        tokens_this_move=tokens_this_move,
        seconds_this_move=seconds_this_move,
    )


# Self record: the four fixtures of spec 4.5a.


def test_click_and_character() -> None:
    rec = SelfRecord.from_state(ALL_ACTIONS, MOVER)
    assert (rec.i, rec.click, rec.my_character) == ("agent", True, MOVER)
    assert rec.line() == f"I am the agent. Click: yes. My character: {MOVER}."


def test_character_only() -> None:
    rec = SelfRecord.from_state(MOVES_ONLY, MOVER)
    assert (rec.i, rec.click, rec.my_character) == ("agent", False, MOVER)
    assert rec.line() == f"I am the agent. Click: no. My character: {MOVER}."


def test_click_only() -> None:
    rec = SelfRecord.from_state((6,), None)
    assert (rec.i, rec.click, rec.my_character) == ("agent", True, None)
    assert rec.line() == "I am the agent. Click: yes. My character: none."


def test_neither() -> None:
    rec = SelfRecord.from_state(MOVES_ONLY, None)
    assert (rec.i, rec.click, rec.my_character) == ("agent", False, None)
    assert rec.line() == "I am the agent. Click: no. My character: none."


def test_i_is_the_agent_by_default() -> None:
    assert SelfRecord(click=False, my_character=None).i == "agent"


# Focus record.


def test_focus_line_is_one_line_under_the_cap_with_real_values() -> None:
    line = focus().line()
    assert "\n" not in line
    assert len(line) < LINE_CHARS
    for part in (GOAL, "(open)", "kc4:cd34ef5678", "follow", "budget 41", "progress", "1.3s"):
        assert part in line, (part, line)


def test_focus_line_stays_under_the_cap_with_absurd_values() -> None:
    line = focus(
        goal_id="sequence " + "x" * 300,
        goal_status="s" * 50,
        target="t" * 120,
        budget_left=10**9,
        tokens_this_move=10**12,
        seconds_this_move=1e9,
    ).line()
    assert "\n" not in line
    assert len(line) < LINE_CHARS


def test_focus_line_names_an_unknown_budget_and_a_missing_goal() -> None:
    line = focus(goal_id=None, goal_status=None, target=None, budget_left=None).line()
    assert "goal none;" in line
    assert "target none" in line
    assert "budget unknown" in line


def test_focus_as_dict_has_every_field() -> None:
    rec = focus()
    out = rec.as_dict()
    assert set(out) == FOCUS_FIELDS
    assert set(out) == {f.name for f in dataclasses.fields(FocusRecord)}
    assert out["budget_left"] == 41
    assert out["goal_id"] == GOAL
    assert out["seconds_this_move"] == 1.26
