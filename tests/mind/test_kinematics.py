"""Tests for the kinematic model: gaits from facts, walks, and distance fields."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.facts import Effect, FactMemory
from arcagent.mind.goals import Goal
from arcagent.mind.kinematics import KIN_LIMIT, Gait, Kinematics
from arcagent.mind.objects import Scene, Tracker, shape_of

Cells = dict[tuple[int, int], int]
DOT = shape_of([(0, 0)])
PSIG, TSIG = f"c1:{DOT}", f"c4:{DOT}"
WALL = 5


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(cells: Cells) -> Scene:
    return Tracker().update(grid_with(cells), frozenset())[0]


def walker(step: int = 1, slide_right: bool = False) -> FactMemory:
    mem = FactMemory()
    for i, (dx, dy) in {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.items():
        effect = (
            Effect("slide", dx=1)
            if i == 4 and slide_right
            else Effect("move", dx=dx * step, dy=dy * step)
        )
        mem.record((f"A{i}", PSIG, "any"), (effect,), 0)
    mem.blockers[PSIG] = {WALL}
    return mem


def wall_column(x: int, gap: int | None = None) -> Cells:
    return {(x, y): WALL for y in range(64) if y != gap}


def test_gaits_come_from_move_and_slide_facts() -> None:
    gaits = Kinematics(walker(slide_right=True)).gaits(PSIG)
    assert Gait("A4", 1, 0, slides=True) in gaits
    assert Gait("A3", -1, 0, slides=False) in gaits
    assert Kinematics(FactMemory()).gaits(PSIG) == ()


def test_reachable_stops_at_blocker_colors() -> None:
    s = scene_of({(10, 10): 1} | wall_column(14))
    reach = Kinematics(walker()).reachable(s, PSIG)
    assert (13, 10) in reach and (14, 10) not in reach and (15, 10) not in reach
    assert reach[(13, 10)] == 3


def test_slide_gait_stops_where_it_would_be_blocked() -> None:
    s = scene_of({(10, 10): 1} | wall_column(20))
    reach = Kinematics(walker(slide_right=True)).reachable(s, PSIG)
    assert reach[(19, 10)] == 1  # one slide to the cell before the wall
    assert reach[(11, 10)] == 9  # the slide, then eight single steps back left


def test_steps_to_goes_around_a_wall_and_unreachable_is_none() -> None:
    open_wall = scene_of({(10, 10): 1, (20, 10): 4} | wall_column(15, gap=30))
    kin = Kinematics(walker())
    steps = kin.steps_to(open_wall, PSIG, frozenset({(20, 10)}))
    assert steps is not None and steps > 10  # 9 cells straight, far more around the gap
    shut = scene_of({(10, 10): 1, (20, 10): 4} | wall_column(15))
    assert kin.steps_to(shut, PSIG, frozenset({(20, 10)})) is None


def test_limit_bounds_the_walk() -> None:
    s = scene_of({(10, 10): 1})
    reach = Kinematics(walker()).reachable(s, PSIG, limit=50)
    assert len(reach) <= 50 and KIN_LIMIT >= 4096


def test_distance_field_counts_moves_to_touch_the_target() -> None:
    root = scene_of({(10, 10): 1, (20, 10): 4} | wall_column(15, gap=30))
    kin = Kinematics(walker())
    field = kin.field(root, PSIG, TSIG)
    straight = kin.steps_to(root, PSIG, frozenset({(20, 10)}))
    assert field[(10, 10)] == straight
    assert field[(19, 10)] == 0 and field[(21, 10)] == 0
    assert kin.field(root, PSIG, TSIG) is field  # cached for the search


def test_goal_distance_with_kinematics_counts_moves_around_a_wall() -> None:
    s = scene_of({(10, 10): 1, (20, 10): 4} | wall_column(15, gap=30))
    goal = Goal("reach", a=PSIG, b=TSIG)
    plain = goal.distance(s)
    moved = goal.distance(s, None, Kinematics(walker()))
    assert plain == 9 and moved is not None and moved > 30  # down to the gap and back up
    near = scene_of({(19, 10): 1, (20, 10): 4} | wall_column(15, gap=30))
    assert goal.distance(near, None, Kinematics(walker())) == 0
