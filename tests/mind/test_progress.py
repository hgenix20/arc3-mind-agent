"""Tests for the progress monitor: progress, motion, milestones, regress, settling."""

from __future__ import annotations

from arcagent.mind.goals import Goal
from arcagent.mind.progress import MOTION_WINDOW, SETTLE_MOVES, ProgressMonitor

REACH = Goal("reach", a="c1:p", b="c4:t")
CLEAR = Goal("clear", color=4)


def key(n: int) -> tuple[tuple[str, int, int], ...]:
    return (("c1:p", n, 0),)


def test_closer_to_target_is_progress() -> None:
    m = ProgressMonitor()
    assert m.observe(key(0), {REACH: 10}, active=REACH) == "none"  # the first look sets the bar
    assert m.observe(key(1), {REACH: 9}, active=REACH) == "progress"
    assert m.best(REACH) == 9 and m.progressed()


def test_back_and_forth_is_motion() -> None:
    m = ProgressMonitor()
    m.observe(key(0), {REACH: 10}, active=REACH)
    m.observe(key(1), {REACH: 9}, active=REACH)
    assert m.observe(key(0), {REACH: 10}, active=REACH) == "motion"
    assert m.observe(key(1), {REACH: 9}, active=REACH) == "motion"  # 9 isn't new any more
    assert MOTION_WINDOW == 12


def test_fewer_things_of_a_clear_color_is_a_milestone() -> None:
    m = ProgressMonitor()
    m.observe(key(0), {CLEAR: 3}, active=None)
    assert m.observe(key(1), {CLEAR: 2}, active=None) == "milestone"
    assert m.observe(key(2), {}, active=None, level_up=True) == "milestone"


def test_moving_away_from_the_active_goal_is_regress() -> None:
    m = ProgressMonitor()
    m.observe(key(0), {REACH: 5}, active=REACH)
    m.observe(key(1), {REACH: 4}, active=REACH)
    assert m.observe(key(9), {REACH: 7}, active=REACH) == "regress"


def test_settled_after_quiet_moves_or_at_zero() -> None:
    m = ProgressMonitor()
    for _ in range(SETTLE_MOVES):
        m.observe(key(0), {REACH: 3}, active=REACH)
    assert m.settled(REACH)
    m.reset_life()
    m.observe(key(5), {REACH: 0}, active=REACH)
    assert m.settled(REACH)


def test_a_new_life_starts_its_own_best() -> None:
    m = ProgressMonitor()
    m.observe(key(0), {REACH: 10}, active=REACH)
    m.observe(key(1), {REACH: 4}, active=REACH)
    m.reset_life()
    assert m.observe(key(0), {REACH: 10}, active=REACH) == "none"
    assert m.observe(key(1), {REACH: 8}, active=REACH) == "progress"  # beats this life only
    assert m.progressed() and m.best(REACH) == 8  # the life best
    m.reset_level()
    assert m.best(REACH) is None and not m.progressed()
