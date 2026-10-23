"""Perception: hashing, object extraction, change measurement, candidate actions."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import GRID_SIZE, Action, Grid, Observation, Phase
from arcagent.mind.perception import (
    candidate_actions,
    changed_bbox,
    changed_cells,
    frame_hash,
    perceive,
)
from arcagent.mind.types import Step


def grid_with_two_objects() -> Grid:
    g = np.zeros((GRID_SIZE, GRID_SIZE), np.int8)
    g[2:4, 10:13] = 5
    g[30:32, 40:42] = 7
    return g


def test_frame_hash_is_stable_and_sensitive() -> None:
    g = np.zeros((64, 64), np.int8)
    h = frame_hash(g)
    assert len(h) == 16 and h == frame_hash(g.copy())
    g2 = g.copy()
    g2[5, 7] = 3
    assert frame_hash(g2) != h


def test_perceive_finds_objects_with_bbox() -> None:
    g = np.zeros((64, 64), np.int8)
    g[2:4, 10:13] = 5
    p = perceive(g)
    assert [(o.color, o.size, o.bbox) for o in p.objects] == [(5, 6, (10, 2, 12, 3))]
    assert p.objects[0].centroid == (11.0, 2.5)
    assert p.hash == frame_hash(g)


def test_changed_bbox_none_when_equal() -> None:
    g = np.zeros((64, 64), np.int8)
    assert changed_bbox(g, g) is None and changed_cells(g, g) == 0


def test_changed_bbox_covers_differences() -> None:
    a = np.zeros((64, 64), np.int8)
    b = a.copy()
    b[3, 4] = 1
    b[8, 6] = 2
    assert changed_cells(a, b) == 2
    assert changed_bbox(a, b) == (4, 3, 6, 8)


def test_candidate_actions_excludes_reset_and_expands_clicks() -> None:
    obs = Observation(grid_with_two_objects(), Phase.NOT_FINISHED, 0, 3, (0, 1, 6))
    acts = candidate_actions(obs)
    assert Action(1) in acts and all(a.id != 0 for a in acts)
    assert sum(a.id == 6 for a in acts) == 2


def test_step_level_up() -> None:
    g = grid_with_two_objects()
    up = Step(0, g, Action(1), g, 0, 1, Phase.NOT_FINISHED)
    flat = Step(1, g, Action(1), g, 1, 1, Phase.NOT_FINISHED)
    assert up.level_up and not flat.level_up
