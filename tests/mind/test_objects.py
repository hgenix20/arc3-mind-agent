"""Tests for objects, scenes, tracking and diffs."""

from __future__ import annotations

import time

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.objects import Tracker, extract


def grid_with(cells: dict[tuple[int, int], int], bg: int = 0) -> Grid:
    g = np.full((64, 64), bg, dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def test_sig_is_position_invariant_and_color_aware() -> None:
    a = extract(grid_with({(1, 1): 3, (2, 1): 3}))[0]
    b = extract(grid_with({(10, 7): 3, (11, 7): 3}))[0]
    c = extract(grid_with({(10, 7): 4, (11, 7): 4}))[0]
    assert a.sig == b.sig and a.shape == c.shape and a.sig != c.sig
    assert a.sig.startswith("c3:")


def test_tracker_reports_move_with_stable_id() -> None:
    t = Tracker()
    s0, d0 = t.update(grid_with({(5, 5): 2, (20, 20): 7}), frozenset())
    s1, d1 = t.update(grid_with({(6, 5): 2, (20, 20): 7}), frozenset())
    assert d0 is None and d1 is not None
    mover = next(c for c in d1.changes if c.kind == "moved")
    assert (mover.dx, mover.dy) == (1, 0)
    assert s0.by_id()[mover.id].color == 2 and s1.by_id()[mover.id].x == 6


def test_tracker_recolor_and_disappear_and_appear() -> None:
    t = Tracker()
    t.update(grid_with({(5, 5): 2, (9, 9): 4}), frozenset())
    _, d = t.update(grid_with({(5, 5): 3, (30, 30): 6}), frozenset())
    assert d is not None
    kinds = sorted(c.kind for c in d.changes)
    assert kinds == ["appeared", "disappeared", "recolored"]


def test_mask_hides_cells_from_extraction() -> None:
    g = grid_with({(5, 5): 2, (40, 1): 9})
    assert {t.color for t in extract(g, frozenset({(40, 1)}))} == {2}


def test_padding_is_blocked_not_a_thing() -> None:
    g = np.full((64, 64), -1, dtype=np.int8)
    g[:32, :32] = 0
    g[3, 3] = 5
    s, _ = Tracker().update(g, frozenset())
    assert [t.color for t in s.things] == [5] and (40, 40) in s.blocked


def test_unchanged_frame_gives_empty_diff() -> None:
    t = Tracker()
    g = grid_with({(5, 5): 2})
    t.update(g, frozenset())
    d = t.update(g.copy(), frozenset())[1]
    assert d is not None and d.empty


def test_many_small_objects_extract_and_track_fast() -> None:
    g = grid_with({(x, y): 1 + (x + y) % 3 for x in range(0, 64, 2) for y in range(0, 64, 2)})
    t = Tracker()
    t.update(g, frozenset())
    start = time.perf_counter()
    t.update(g.copy(), frozenset())
    assert time.perf_counter() - start < 0.2


def test_reset_restarts_ids_and_drops_the_diff() -> None:
    t = Tracker()
    t.update(grid_with({(5, 5): 2}), frozenset())
    t.reset()
    s, d = t.update(grid_with({(9, 9): 4}), frozenset())
    assert d is None and [x.id for x in s.things] == [1]


def test_reshape_is_reported_for_same_color_overlap() -> None:
    t = Tracker()
    t.update(grid_with({(5, 5): 2, (6, 5): 2}), frozenset())
    _, d = t.update(grid_with({(5, 5): 2, (6, 5): 2, (7, 5): 2}), frozenset())
    assert d is not None and [c.kind for c in d.changes] == ["reshaped"]


def test_stray_cells_count_changes_outside_things() -> None:
    t = Tracker()
    t.update(grid_with({(5, 5): 2}), frozenset())
    g = grid_with({(5, 5): 2})
    g[40, 40] = 0  # no change: same as background
    _, same = t.update(g, frozenset())
    assert same is not None and same.stray_cells == 0
    # a change under the mask is never a stray cell
    h = grid_with({(5, 5): 2, (1, 1): 9})
    _, masked = t.update(h, frozenset({(1, 1)}))
    assert masked is not None and masked.stray_cells == 0 and masked.empty
