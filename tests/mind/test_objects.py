"""Tests for objects, scenes, tracking and diffs."""

from __future__ import annotations

import time

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.facts import cell_owner
from arcagent.mind.objects import Scene, Thing, Tracker, color_at, extract, shape_of
from arcagent.mind.regions import render


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


def test_many_identical_objects_track_fast() -> None:
    g = grid_with({(x, y): 1 for x in range(0, 64, 2) for y in range(0, 64, 2)})
    t = Tracker()
    t.update(g, frozenset())
    start = time.perf_counter()
    _, d = t.update(g.copy(), frozenset())
    assert time.perf_counter() - start < 0.2
    assert d is not None and d.empty


def test_ids_are_stable_and_never_reused() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with({(5, 5): 2, (20, 20): 7}), frozenset())
    ids0 = {th.color: th.id for th in s0.things}
    t.update(grid_with({(6, 5): 2, (20, 20): 7}), frozenset())
    t.update(grid_with({(6, 5): 2}), frozenset())  # color 7 disappears
    s3, _ = t.update(grid_with({(7, 5): 2, (30, 30): 7}), frozenset())  # a new 7 appears
    by_color = {th.color: th.id for th in s3.things}
    assert by_color[2] == ids0[2]
    assert by_color[7] not in ids0.values()
    assert by_color[7] > max(ids0.values())


def test_recolor_keeps_the_id() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with({(5, 5): 2, (6, 5): 2}), frozenset())
    s1, d = t.update(grid_with({(5, 5): 3, (6, 5): 3}), frozenset())
    assert d is not None and [c.kind for c in d.changes] == ["recolored"]
    assert s0.things[0].id == s1.things[0].id and d.changes[0].color == 3


def test_move_beyond_radius_is_disappear_plus_appear() -> None:
    t = Tracker()
    t.update(grid_with({(5, 5): 2}), frozenset())
    _, near = t.update(grid_with({(21, 5): 2}), frozenset())  # 16 cells: still a move
    assert near is not None and [c.kind for c in near.changes] == ["moved"]
    _, far = t.update(grid_with({(38, 5): 2}), frozenset())  # 17 cells
    assert far is not None and sorted(c.kind for c in far.changes) == ["appeared", "disappeared"]


def test_scene_key_ignores_ids() -> None:
    a, _ = Tracker().update(grid_with({(5, 5): 2, (9, 9): 4}), frozenset())
    t = Tracker()
    t.update(grid_with({(30, 30): 6}), frozenset())
    b, _ = t.update(grid_with({(5, 5): 2, (9, 9): 4}), frozenset())
    assert {x.id for x in a.things} != {x.id for x in b.things}
    assert a.key() == b.key()


# --- terrain (Task 8b) ----------------------------------------------------------------


def floor_grid(player: tuple[int, int] | None) -> Grid:
    """A 30x30 floor of color 3 (900 cells) with a one-cell spur at (30, 5); bg 0."""
    g = np.zeros((64, 64), dtype=np.int8)
    g[0:30, 0:30] = 3
    g[5, 30] = 3
    if player is not None:
        g[player[1], player[0]] = 1
    return g


def test_large_component_is_terrain_not_a_thing() -> None:
    s, _ = Tracker().update(floor_grid((10, 10)), frozenset())
    assert [t.color for t in s.things] == [1]
    assert s.terrain.shape == (64, 64) and s.terrain.dtype == np.int8
    assert s.terrain[0, 0] == 3 and s.terrain[40, 40] == 0
    # under the player with no memory: the most common non-thing neighbor color
    assert s.terrain[10, 10] == 3
    assert len(extract(floor_grid(None))) == 0
    assert hash(s) == hash(s)


def test_terrain_under_a_thing_comes_from_memory() -> None:
    t = Tracker()
    t.update(floor_grid((31, 5)), frozenset())
    s, d = t.update(floor_grid((30, 5)), frozenset())
    # three of the spur's four neighbors are background; memory still says floor
    assert s.terrain[5, 30] == 3 and s.terrain[5, 31] == 0
    assert d is not None and [c.kind for c in d.changes] == ["moved"] and d.stray_cells == 0


def test_terrain_memory_resets_with_the_tracker() -> None:
    t = Tracker()
    t.update(floor_grid((31, 5)), frozenset())
    t.reset()
    s, _ = t.update(floor_grid((30, 5)), frozenset())
    assert s.terrain[5, 30] == 0


def test_terrain_keeps_padding_and_masked_cells_read_background() -> None:
    g = floor_grid((40, 40))
    g[60:, :] = -1
    g[50, 50] = 9
    s, _ = Tracker().update(g, frozenset({(50, 50)}))
    assert s.terrain[62, 3] == -1 and s.terrain[50, 50] == 0
    assert {t.color for t in s.things} == {1}


def test_color_at_reads_owner_then_terrain() -> None:
    s, _ = Tracker().update(floor_grid((10, 10)), frozenset())
    owner = {c: (t.id, t.color) for t in s.things for c in t.cells}
    assert color_at(s, owner, (10, 10)) == 1
    assert color_at(s, owner, (11, 10)) == 3
    assert color_at(s, owner, (45, 45)) == 0
    assert color_at(s, owner, (-1, 0)) == -1 and color_at(s, owner, (0, 64)) == -1


def test_change_carries_the_new_size() -> None:
    t = Tracker()
    t.update(grid_with({(5, 5): 2, (6, 5): 2, (7, 5): 2}), frozenset())
    _, d = t.update(grid_with({(6, 5): 2, (7, 5): 2}), frozenset())
    assert d is not None and [(c.kind, c.size) for c in d.changes] == [("reshaped", 2)]
    _, d2 = t.update(grid_with({(7, 5): 2, (8, 5): 2}), frozenset())
    assert d2 is not None and [(c.kind, c.size) for c in d2.changes] == [("moved", 2)]


def test_terrain_split_by_a_thing_stays_terrain() -> None:
    """A floor cut in two by a thing standing across it doesn't turn into things."""
    g = np.zeros((64, 64), dtype=np.int8)
    g[10, 0:60] = 3  # a 3-wide corridor of 180 cells
    g[11, 0:60] = 3
    g[12, 0:60] = 3
    t = Tracker()
    t.update(g, frozenset())
    h = g.copy()
    h[10:13, 50] = 1  # a 1x3 sprite cuts off a 27-cell piece and leaves 150 cells
    s, d = t.update(h, frozenset())
    assert [x.color for x in s.things] == [1]
    assert s.terrain[11, 55] == 3 and s.terrain[11, 50] == 3
    assert d is not None and [c.kind for c in d.changes] == ["appeared"]


def _placed(
    scene: Scene, tid: int, x: int, y: int, *, cells: set[tuple[int, int]], color: int
) -> Scene:
    """``scene`` with thing ``tid`` predicted at top-left (x, y) with ``cells`` and ``color``."""
    shape = shape_of(cells)
    things = tuple(
        Thing(t.id, color, shape, f"c{color}:{shape}", x, y, frozenset(cells)) if t.id == tid else t
        for t in scene.things
    )
    return Scene(things, scene.background, scene.blocked)


def test_turned_sprite_that_moved_keeps_the_id() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with({(10, 10): 9, (11, 10): 9, (10, 11): 9}), frozenset())
    s1, d = t.update(grid_with({(13, 10): 9, (14, 10): 9, (14, 11): 9}), frozenset())
    assert d is not None and [c.kind for c in d.changes] == ["reshaped"]
    assert s1.things[0].id == s0.things[0].id
    assert t.how[s0.things[0].id] == "near"


def test_recolor_and_move_keeps_the_id_only_with_a_prior() -> None:
    cells0 = {(5, 5): 2, (6, 5): 2}
    after = grid_with({(8, 5): 3, (9, 5): 3})
    plain = Tracker()
    plain.update(grid_with(cells0), frozenset())
    _, d = plain.update(after, frozenset())
    assert d is not None and sorted(c.kind for c in d.changes) == ["appeared", "disappeared"]
    t = Tracker()
    s0, _ = t.update(grid_with(cells0), frozenset())
    tid = s0.things[0].id
    prior = _placed(s0, tid, 8, 5, cells={(8, 5), (9, 5)}, color=3)
    s1, d = t.update(after, frozenset(), prior=prior)
    assert d is not None and [c.kind for c in d.changes] == ["recolored"]
    assert s1.things[0].id == tid and t.how[tid] == "prior"


def test_predicted_teleport_keeps_the_id() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with({(5, 5): 2}), frozenset())
    tid = s0.things[0].id
    prior = _placed(s0, tid, 50, 40, cells={(50, 40)}, color=2)
    s1, d = t.update(grid_with({(50, 40): 2}), frozenset(), prior=prior)
    assert d is not None and [(c.kind, c.dx, c.dy) for c in d.changes] == [("moved", 45, 35)]
    assert s1.things[0].id == tid


def test_two_same_sig_things_crossing_keep_their_ids_with_a_prior() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with({(10, 5): 2, (14, 5): 2}), frozenset())
    left, right = sorted(s0.things, key=lambda th: th.x)
    prior = _placed(s0, left.id, 13, 5, cells={(13, 5)}, color=2)
    prior = _placed(prior, right.id, 11, 5, cells={(11, 5)}, color=2)
    s1, _ = t.update(grid_with({(11, 5): 2, (13, 5): 2}), frozenset(), prior=prior)
    assert s1.by_id()[left.id].x == 13 and s1.by_id()[right.id].x == 11


def test_new_thing_on_a_vacated_spot_is_not_a_reshape() -> None:
    t = Tracker()
    t.update(grid_with({(5, 5): 2, (6, 5): 2}), frozenset())
    _, d = t.update(grid_with({(5, 5): 6, (5, 6): 6, (6, 6): 6}), frozenset())
    assert d is not None and sorted(c.kind for c in d.changes) == ["appeared", "disappeared"]


def test_how_records_still_and_same_sig_matches() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with({(5, 5): 2, (30, 30): 4}), frozenset())
    t.update(grid_with({(7, 5): 2, (30, 30): 4}), frozenset())
    by_color = {th.color: th.id for th in s0.things}
    assert t.how[by_color[4]] == "still" and t.how[by_color[2]] == "sig"


def _piece(x: int, y: int, body: int = 14, corner: int = 1) -> dict[tuple[int, int], int]:
    """A 4x4 sprite: ``body`` everywhere but its four corner cells, which are ``corner``."""
    cells = {(x + i, y + j): body for i in range(4) for j in range(4)}
    for i, j in ((0, 0), (3, 0), (0, 3), (3, 3)):
        cells[(x + i, y + j)] = corner
    return cells


def test_corner_decorations_join_their_sprite() -> None:
    t = extract(grid_with(_piece(10, 10) | _piece(40, 40)))[0]
    plain = extract(grid_with({(10 + i, 10 + j): 14 for i in range(4) for j in range(4)}))[0]
    assert t.color == 14 and len(t.cells) == 16 and (t.x, t.y) == (10, 10)
    assert sorted(t.paint) == [(0, 0, 1), (0, 3, 1), (3, 0, 1), (3, 3, 1)]
    assert t.sig.startswith("c14:") and t.sig != plain.sig and plain.paint == ()
    assert t.sig == extract(grid_with(_piece(30, 40) | _piece(10, 50)))[0].sig


def test_a_thing_inside_a_hollow_frame_stays_its_own() -> None:
    frame = {(x, y): 5 for x in range(10, 20) for y in (10, 19)}
    frame |= {(x, y): 5 for y in range(10, 20) for x in (10, 19)}
    things = extract(grid_with(frame | {(11, 11): 3}))
    assert sorted(t.color for t in things) == [3, 5] and all(t.paint == () for t in things)


def test_a_part_over_a_quarter_of_its_host_stays_its_own() -> None:
    cells = {(10 + i, 10 + j): 14 for i in range(4) for j in range(4)}
    cells |= {(11, 11): 2, (12, 11): 2, (11, 12): 2, (12, 12): 2, (13, 12): 2}
    assert sorted(t.color for t in extract(grid_with(cells))) == [2, 14]


def test_a_piece_paints_and_owns_its_part_colors() -> None:
    s, _ = Tracker().update(grid_with(_piece(10, 10) | _piece(40, 40)), frozenset())
    g = render(s)
    assert g[10, 10] == 1 and g[10, 11] == 14 and g[13, 13] == 1
    owner = cell_owner(s)
    assert owner[(10, 10)][1] == 1 and owner[(11, 10)][1] == 14


def test_a_moving_piece_keeps_its_id_and_paint() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with(_piece(10, 10) | _piece(40, 40)), frozenset())
    s1, d = t.update(grid_with(_piece(22, 10) | _piece(40, 40)), frozenset())
    assert d is not None
    (c,) = [c for c in d.changes if c.kind != "same"]
    assert c.kind == "moved" and (c.dx, c.dy) == (12, 0)
    assert s1.by_id()[c.id].paint == s0.by_id()[c.id].paint != ()


def test_a_selection_ring_does_not_own_the_thing_it_rings() -> None:
    cells = {(10 + dx, 10 + dy): 8 for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy}
    things = extract(grid_with(cells | {(10, 10): 2}))
    assert sorted(t.color for t in things) == [2, 8] and all(t.paint == () for t in things)


def test_a_lone_decorated_sprite_keeps_its_parts_apart() -> None:
    things = extract(grid_with(_piece(10, 10)))
    assert sorted(t.color for t in things) == [1, 1, 1, 1, 14]


def test_a_piece_look_seen_twice_stays_a_piece_when_alone() -> None:
    t = Tracker()
    s0, _ = t.update(grid_with(_piece(10, 10) | _piece(30, 10)), frozenset())
    assert [len(x.cells) for x in s0.things] == [16, 16]
    s1, _ = t.update(grid_with(_piece(10, 10)), frozenset())
    assert [len(x.cells) for x in s1.things] == [16]
