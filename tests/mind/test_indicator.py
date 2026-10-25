"""Tests for progress indicators and the sequence goal."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.goals import Goal
from arcagent.mind.indicator import SLOT_MIN, SequenceTracker, lit, slot_rows
from arcagent.mind.objects import Scene, Tracker

Cells = dict[tuple[int, int], int]


def grid_of(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def boxes(xs: list[int], y: int, color: int, size: int = 2) -> Cells:
    return {(x + dx, y + dy): color for x in xs for dx in range(size) for dy in range(size)}


def scene_of(cells: Cells) -> tuple[Scene, Grid]:
    g = grid_of(cells)
    return Tracker().update(g, frozenset())[0], g


ROW = [20, 24, 28]


def test_slot_rows_finds_three_equal_boxes_in_a_row() -> None:
    scene, g = scene_of(boxes(ROW, 60, 9) | {(5, 5): 1})
    rows = slot_rows(scene, g)
    assert len(rows) == 1 and SLOT_MIN == 3
    assert rows[0].boxes == ((20, 60, 2, 2), (24, 60, 2, 2), (28, 60, 2, 2))
    assert lit(rows[0], g) == 0
    lit_one = grid_of(boxes(ROW, 60, 9) | boxes([20], 60, 11))
    assert lit(rows[0], lit_one) == 1


def test_slot_rows_skips_pairs_and_mixed_shapes() -> None:
    pair, g = scene_of(boxes([20, 24], 60, 9))
    assert slot_rows(pair, g) == []
    mixed, g = scene_of(boxes([20, 24], 60, 9) | boxes([28], 60, 9, size=3))
    assert slot_rows(mixed, g) == []
    uneven, g = scene_of(boxes([20, 24, 30], 60, 9))
    assert slot_rows(uneven, g) == []


def test_tracker_learns_the_next_step_and_a_wrong_one() -> None:
    base = boxes(ROW, 60, 9)
    scene, g = scene_of(base)
    t = SequenceTracker()
    t.start_level(scene, g)
    t.start_life(g)
    assert t.row is None
    assert t.observe(grid_of(base | boxes([20], 60, 11)), frozenset({"pad5"}), None) == "up"
    assert t.row is not None and not t.by_click
    assert t.next_kinds() == frozenset()  # nothing known at one lit box yet
    assert t.observe(grid_of(base), frozenset({"pad3"}), None) == "reset"
    assert t.next_kinds() == frozenset({"pad5"})  # back at zero: the first step is known
    t.observe(grid_of(base | boxes([20], 60, 11)), frozenset({"pad5"}), None)
    assert "pad3" in t.avoid() and "pad5" in t.avoid()  # wrong here, and already used
    assert t.wrong_now() == frozenset({"pad3"})
    goal = t.goal()
    assert goal is not None and goal.kind == "sequence"


def test_a_clicked_row_is_an_input_not_an_indicator() -> None:
    base = boxes(ROW, 60, 9)
    scene, g = scene_of(base)
    t = SequenceTracker()
    t.start_level(scene, g)
    t.start_life(g)
    assert t.observe(grid_of(base | boxes([20], 60, 11)), frozenset({"k"}), (20, 60)) == "none"
    assert t.row is None
    assert t.observe(grid_of(base | boxes([20, 24], 60, 11)), frozenset({"k"}), (5, 5)) == "none"


def test_an_indicator_raised_by_a_click_elsewhere_is_learned_by_click() -> None:
    base = boxes(ROW, 60, 9)
    scene, g = scene_of(base)
    t = SequenceTracker()
    t.start_level(scene, g)
    t.start_life(g)
    assert t.observe(grid_of(base | boxes([20], 60, 11)), frozenset({"k"}), (5, 5)) == "up"
    assert t.by_click


def test_sequence_goal_holds_when_every_box_is_lit() -> None:
    base = boxes(ROW, 60, 9)
    scene, g = scene_of(base)
    t = SequenceTracker()
    t.start_level(scene, g)
    t.start_life(g)
    t.observe(grid_of(base | boxes([20], 60, 11)), frozenset({"k"}), None)
    goal = t.goal()
    assert goal is not None
    one = grid_of(base | boxes([20], 60, 11))
    full = grid_of(boxes(ROW, 60, 11))
    assert goal.distance(scene, one) == 2 and not goal.holds(scene, one)
    assert goal.distance(scene, full) == 0 and goal.holds(scene, full)
    assert goal.distance(scene, None) is None and not goal.holds(scene, None)
    assert goal == Goal("sequence", a=goal.a, b=goal.b) and goal.id().startswith("sequence ")
