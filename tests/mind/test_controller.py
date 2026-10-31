"""Tests for the controller's ordered handoff rules (spec 4.5c)."""

from __future__ import annotations

import pytest

from arcagent.mind.controller import (
    DEFAULT_ORDER,
    NO_PROGRESS_MOVES,
    RULES,
    STALL_MOVES,
    Controller,
    Decision,
    Signals,
)


def quiet(
    *,
    gate_tripped: bool = False,
    segment_broke: bool = False,
    fallbacks_in_row: int = 0,
    no_progress_moves: int = 0,
    budget_left: int | None = None,
    segment_left: int = 3,
    deadline_passed: bool = False,
) -> Signals:
    """Signals on which no rule fires unless an argument says otherwise."""
    return Signals(
        gate_tripped=gate_tripped,
        segment_broke=segment_broke,
        fallbacks_in_row=fallbacks_in_row,
        no_progress_moves=no_progress_moves,
        budget_left=budget_left,
        segment_left=segment_left,
        deadline_passed=deadline_passed,
    )


def test_nothing_fires_keeps_the_method() -> None:
    assert Controller().decide(quiet()) == Decision("keep", "none", "no rule fired")


def test_gate_trip_fires_h_a() -> None:
    got = Controller().decide(quiet(gate_tripped=True))
    assert (got.method, got.rule) == ("fallback", "H-a")
    assert "gate_tripped" in got.reason


def test_segment_broke_fires_h_b() -> None:
    got = Controller().decide(quiet(segment_broke=True))
    assert (got.method, got.rule) == ("return_to_fork", "H-b")
    assert "segment_broke" in got.reason


def test_stall_fires_h_e_at_the_limit_and_not_below() -> None:
    ctl = Controller()
    assert ctl.decide(quiet(fallbacks_in_row=STALL_MOVES - 1)).rule == "none"
    got = ctl.decide(quiet(fallbacks_in_row=STALL_MOVES))
    assert (got.method, got.rule) == ("explore", "H-e")
    assert str(STALL_MOVES) in got.reason


def test_no_progress_fires_h_c_at_the_limit_and_not_below() -> None:
    ctl = Controller()
    assert ctl.decide(quiet(no_progress_moves=NO_PROGRESS_MOVES - 1)).rule == "none"
    got = ctl.decide(quiet(no_progress_moves=NO_PROGRESS_MOVES))
    assert (got.method, got.rule) == ("replan", "H-c")
    assert str(NO_PROGRESS_MOVES) in got.reason


def test_low_budget_fires_h_d_below_segment_plus_two() -> None:
    ctl = Controller()
    got = ctl.decide(quiet(budget_left=4, segment_left=3))
    assert (got.method, got.rule) == ("replan", "H-d")
    assert "budget_left=4" in got.reason
    assert "segment_left=3" in got.reason
    assert ctl.decide(quiet(budget_left=5, segment_left=3)).rule == "none"


def test_unknown_budget_never_fires_h_d() -> None:
    assert Controller().decide(quiet(budget_left=None, segment_left=50)).rule == "none"


def everything() -> Signals:
    return quiet(
        gate_tripped=True,
        segment_broke=True,
        fallbacks_in_row=STALL_MOVES,
        no_progress_moves=NO_PROGRESS_MOVES,
        budget_left=0,
        segment_left=3,
        deadline_passed=True,
    )


def test_deadline_wins_over_every_rule_and_every_order() -> None:
    for order in (DEFAULT_ORDER, tuple(reversed(DEFAULT_ORDER)), ()):
        got = Controller(order).decide(everything())
        assert (got.method, got.rule) == ("fallback", "deadline"), order
        assert "deadline_passed" in got.reason


def test_order_change_changes_the_winner_when_two_rules_fire() -> None:
    both = quiet(gate_tripped=True, segment_broke=True)
    assert Controller().decide(both).rule == "H-a"
    assert Controller(("H-b", "H-a")).decide(both).rule == "H-b"


def test_a_rule_left_out_of_the_order_never_fires() -> None:
    assert Controller(("H-a",)).decide(quiet(segment_broke=True)).rule == "none"


def test_default_order_names_every_rule_once() -> None:
    assert DEFAULT_ORDER == ("H-a", "H-b", "H-e", "H-c", "H-d")
    assert sorted(DEFAULT_ORDER) == sorted(RULES)
    assert len(set(RULES)) == len(RULES)


def test_unknown_rule_name_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="H-z"):
        Controller(("H-a", "H-z"))


def test_duplicate_rule_name_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="H-a"):
        Controller(("H-a", "H-b", "H-a"))


def test_thresholds_below_one_raise() -> None:
    with pytest.raises(ValueError, match="stall_moves"):
        Controller(stall_moves=0)
    with pytest.raises(ValueError, match="no_progress_moves"):
        Controller(no_progress_moves=0)


def test_custom_thresholds_move_the_limits() -> None:
    ctl = Controller(stall_moves=3, no_progress_moves=5)
    assert ctl.decide(quiet(fallbacks_in_row=3)).rule == "H-e"
    assert ctl.decide(quiet(fallbacks_in_row=2)).rule == "none"
    assert ctl.decide(quiet(no_progress_moves=5)).rule == "H-c"
    assert ctl.decide(quiet(no_progress_moves=4)).rule == "none"


def test_controller_reports_its_order_and_thresholds() -> None:
    ctl = Controller(("H-b", "H-a"), stall_moves=7, no_progress_moves=9)
    assert ctl.order == ("H-b", "H-a")
    assert (ctl.stall_moves, ctl.no_progress_moves) == (7, 9)
