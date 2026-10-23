"""Tests for the predictor: facts applied to a scene and an action, then checked."""

from __future__ import annotations

import time

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.facts import SPAWN, Effect, FactKey, FactMemory, learn
from arcagent.mind.objects import Change, Scene, SceneDiff, Thing, Tracker, shape_of
from arcagent.mind.predictor import Prediction, check, predict

Cells = dict[tuple[int, int], int]

DOT = shape_of([(0, 0)])
BAR = shape_of([(0, y) for y in range(5)])
PSIG = f"c1:{DOT}"  # the player: one color-1 cell
WSIG = f"c5:{BAR}"  # a wall: a 5-cell vertical color-5 bar
LSIG = f"c9:{DOT}"  # lava: one color-9 cell
BSIG = f"c3:{DOT}"  # a box: one color-3 cell


def grid_with(cells: Cells, bg: int = 0) -> Grid:
    g = np.full((64, 64), bg, dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(cells: Cells) -> Scene:
    return Tracker().update(grid_with(cells), frozenset())[0]


def transition(before: Cells, after: Cells) -> tuple[Scene, SceneDiff]:
    t = Tracker()
    s0, _ = t.update(grid_with(before), frozenset())
    _, d = t.update(grid_with(after), frozenset())
    assert d is not None
    return s0, d


def wall_cells(x: int, y0: int = 3) -> Cells:
    return {(x, y): 5 for y in range(y0, y0 + 5)}


def mem_with(*facts: tuple[FactKey, Effect | tuple[Effect, ...]]) -> FactMemory:
    mem = FactMemory()
    for key, eff in facts:
        mem.record(key, eff if isinstance(eff, tuple) else (eff,), 0)
    return mem


def player_in(scene: Scene | None) -> Thing:
    assert scene is not None
    return next(t for t in scene.things if t.sig == PSIG)


def at(scene: Scene | None, sig: str) -> list[tuple[int, int]]:
    assert scene is not None
    return sorted((t.x, t.y) for t in scene.things if t.sig == sig)


def at_id(scene: Scene, sig: str) -> int:
    return next(t.id for t in scene.things if t.sig == sig)


def break_fact(mem: FactMemory, key: FactKey) -> None:
    mem.record(key, (Effect("disappear"),), 1)
    mem.record(key, (Effect("disappear"),), 2)
    fact = mem.get(key)
    assert fact is not None and fact.status == "broken"


# --- brief tests ----------------------------------------------------------------------


def test_predicts_move_and_new_scene() -> None:
    p = predict(
        scene_of({(5, 5): 1}), Action(3), mem_with((("A3", PSIG, "any"), Effect("move", dx=-1)))
    )
    assert p.scene is not None and player_in(p.scene).x == 4 and p.unknown == ()
    assert p.used == (("A3", PSIG, "any"),) and not p.game_over and not p.borrowed


def test_wall_predicts_no_change() -> None:
    before = scene_of({(5, 5): 1, **wall_cells(6)})
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", WSIG, "any"), Effect("none")),
    )
    mem.blockers[PSIG] = {5}
    p = predict(before, Action(4), mem)
    assert player_in(p.scene).x == player_in(before).x
    assert p.effects[player_in(before).id] == (Effect("none"),)


def test_unknown_subject_gives_no_scene() -> None:
    p = predict(scene_of({(5, 5): 1}), Action(1), FactMemory())
    assert p.scene is None and ("A1", PSIG) in p.unknown


def test_slide_stops_at_blocker() -> None:
    corridor = scene_of({(2, 5): 1, **wall_cells(9)})
    slide_mem = mem_with(
        (("A4", PSIG, "any"), Effect("slide", dx=1)),
        (("A4", WSIG, "any"), Effect("none")),
    )
    slide_mem.blockers[PSIG] = {5}
    p = predict(corridor, Action(4), slide_mem)
    assert player_in(p.scene).x == 8
    assert p.effects[player_in(corridor).id] == (Effect("move", dx=6),)


def test_predicted_touch_of_deadly_color_flags_game_over() -> None:
    next_to_lava = scene_of({(5, 5): 1, (6, 5): 9})
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", LSIG, "any"), Effect("none")),
        (("any", PSIG, "touch:9"), Effect("game_over")),
    )
    p = predict(next_to_lava, Action(4), mem)
    assert p.game_over and ("any", PSIG, "touch:9") in p.used
    # moving away does not touch it
    mem.record(("A3", PSIG, "any"), (Effect("move", dx=-1),), 0)
    mem.record(("A3", LSIG, "any"), (Effect("none"),), 0)
    assert not predict(next_to_lava, Action(3), mem).game_over


def test_check_hit_and_miss() -> None:
    before, diff_moved_left = transition({(5, 5): 1}, {(4, 5): 1})
    _, diff_recolored = transition({(5, 5): 1}, {(5, 5): 2})
    pred_move_left = predict(
        before, Action(3), mem_with((("A3", PSIG, "any"), Effect("move", dx=-1)))
    )
    hit, missed = check(pred_move_left, before, diff_moved_left)
    assert hit and missed == []
    hit, missed = check(pred_move_left, before, diff_recolored)
    assert not hit and missed == [("A3", PSIG, "any")]


# --- blocking -------------------------------------------------------------------------


def test_grid_edge_and_padding_block() -> None:
    mem = mem_with((("A3", PSIG, "any"), Effect("move", dx=-1)))
    assert player_in(predict(scene_of({(0, 5): 1}), Action(3), mem).scene).x == 0
    g = grid_with({(5, 5): 1})
    g[5, 4] = -1
    padded = Tracker().update(g, frozenset())[0]
    assert player_in(predict(padded, Action(3), mem).scene).x == 5


def test_non_blocker_color_is_passed_over() -> None:
    before = scene_of({(5, 5): 1, (6, 5): 7})
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", f"c7:{DOT}", "any"), Effect("none")),
    )
    assert player_in(predict(before, Action(4), mem).scene).x == 6


def test_blocked_fact_is_used_when_stored() -> None:
    before = scene_of({(5, 5): 1, **wall_cells(6)})
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", PSIG, "blocked"), Effect("none")),
        (("A4", WSIG, "any"), Effect("none")),
    )
    mem.blockers[PSIG] = {5}
    p = predict(before, Action(4), mem)
    assert ("A4", PSIG, "blocked") in p.used and p.scene is not None
    _, still = transition({(5, 5): 1, **wall_cells(6)}, {(5, 5): 1, **wall_cells(6)})
    assert check(p, before, still) == (True, [])


def test_rigid_sprite_moves_as_one_body() -> None:
    # two-color sprite: head color 1 at (5,5), body color 4 at (6,5), moving right
    before = scene_of({(5, 5): 1, (6, 5): 4})
    body = f"c4:{DOT}"
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", body, "any"), Effect("move", dx=1)),
    )
    mem.blockers[PSIG] = {4}  # the head was once stopped by color 4 elsewhere
    p = predict(before, Action(4), mem)
    assert at(p.scene, PSIG) == [(6, 5)] and at(p.scene, body) == [(7, 5)]


def test_blocked_sprite_stays_whole() -> None:
    before = scene_of({(5, 5): 1, (6, 5): 4, **wall_cells(7)})
    body = f"c4:{DOT}"
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", body, "any"), Effect("move", dx=1)),
        (("A4", WSIG, "any"), Effect("none")),
    )
    mem.blockers[body] = {5}
    p = predict(before, Action(4), mem)
    assert at(p.scene, PSIG) == [(5, 5)] and at(p.scene, body) == [(6, 5)]


def test_sprite_parts_never_touch_each_other() -> None:
    before = scene_of({(5, 5): 1, (6, 5): 4})
    body = f"c4:{DOT}"
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", body, "any"), Effect("move", dx=1)),
        (("any", PSIG, "touch:4"), Effect("game_over")),
    )
    assert not predict(before, Action(4), mem).game_over


# --- pushing --------------------------------------------------------------------------


def push_mem() -> FactMemory:
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", BSIG, "any"), Effect("none")),
        (("A4", BSIG, "pushed"), Effect("move", dx=1)),
        (("A4", WSIG, "any"), Effect("none")),
    )
    mem.blockers[BSIG] = {5}
    return mem


def test_push_moves_the_box() -> None:
    p = predict(scene_of({(5, 5): 1, (6, 5): 3}), Action(4), push_mem())
    assert at(p.scene, PSIG) == [(6, 5)] and at(p.scene, BSIG) == [(7, 5)]
    assert ("A4", BSIG, "pushed") in p.used


def test_blocked_box_blocks_the_pusher() -> None:
    p = predict(scene_of({(5, 5): 1, (6, 5): 3, **wall_cells(7)}), Action(4), push_mem())
    assert at(p.scene, PSIG) == [(5, 5)] and at(p.scene, BSIG) == [(6, 5)]


def test_pushed_box_without_any_fact_is_known() -> None:
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", BSIG, "pushed"), Effect("move", dx=1)),
    )
    p = predict(scene_of({(5, 5): 1, (6, 5): 3}), Action(4), mem)
    assert p.unknown == () and at(p.scene, BSIG) == [(7, 5)]


def learn_step(mem: FactMemory, before: Cells, after: Cells, action: int, move: int) -> None:
    tr = Tracker()
    s0, _ = tr.update(grid_with(before), frozenset())
    s1, d = tr.update(grid_with(after), frozenset())
    assert d is not None
    learn(mem, s0, Action(action), d, move=move, level_up=False, game_over=False, after=s1)


def test_push_on_free_floor_survives_a_box_blocked_into_a_wall() -> None:
    walls = wall_cells(9)
    mem = FactMemory()
    learn_step(mem, {(2, 5): 1, (6, 5): 3, **walls}, {(3, 5): 1, (6, 5): 3, **walls}, 4, 0)
    learn_step(mem, {(3, 5): 1, (6, 5): 3, **walls}, {(4, 5): 1, (6, 5): 3, **walls}, 4, 1)
    learn_step(mem, {(5, 5): 1, (6, 5): 3, **walls}, {(6, 5): 1, (7, 5): 3, **walls}, 4, 2)
    learn_step(mem, {(7, 5): 1, (8, 5): 3, **walls}, {(7, 5): 1, (8, 5): 3, **walls}, 4, 3)
    assert 3 in mem.blockers[PSIG]  # learn blamed the box color for stopping the player
    p = predict(scene_of({(5, 5): 1, (6, 5): 3, **walls}), Action(4), mem)
    assert at(p.scene, PSIG) == [(6, 5)] and at(p.scene, BSIG) == [(7, 5)]
    p = predict(scene_of({(7, 5): 1, (8, 5): 3, **walls}), Action(4), mem)
    assert at(p.scene, PSIG) == [(7, 5)] and at(p.scene, BSIG) == [(8, 5)]


def test_box_pushed_into_any_thing_blocks_the_push() -> None:
    mem = mem_with(
        (("A4", PSIG, "any"), Effect("move", dx=1)),
        (("A4", BSIG, "any"), Effect("none")),
        (("A4", BSIG, "pushed"), Effect("move", dx=1)),
        (("A4", WSIG, "any"), Effect("none")),
    )  # no blockers recorded for the box at all
    p = predict(scene_of({(5, 5): 1, (6, 5): 3, **wall_cells(7)}), Action(4), mem)
    assert at(p.scene, PSIG) == [(5, 5)] and at(p.scene, BSIG) == [(6, 5)]


def test_check_skips_ignored_ids_and_appears_on_noise() -> None:
    noise_before = {(30, 30): 7, (31, 30): 7}
    noise_after = {(30, 30): 8}  # the noise thing changes shape and color: a new thing
    before_c = {(5, 5): 1, **noise_before}
    after_c = {(4, 5): 1, **noise_after}
    tr = Tracker()
    before, _ = tr.update(grid_with(before_c), frozenset())
    after, diff = tr.update(grid_with(after_c), frozenset())
    assert diff is not None
    noise_id = next(t.id for t in before.things if t.color == 7)
    mem = mem_with(
        (("A3", PSIG, "any"), Effect("move", dx=-1)),
        (("A3", f"c7:{shape_of([(0, 0), (1, 0)])}", "any"), Effect("none")),
    )
    p = predict(before, Action(3), mem)
    assert check(p, before, diff)[0] is False
    ignore = frozenset({noise_id})
    assert check(p, before, diff, ignore=ignore, after=after) == (True, [])
    assert check(p, before, diff, ignore=ignore) == (False, [])  # appear not placed


# --- facts of other kinds -------------------------------------------------------------


def test_broken_fact_is_unknown() -> None:
    mem = mem_with((("A3", PSIG, "any"), Effect("move", dx=-1)))
    break_fact(mem, ("A3", PSIG, "any"))
    p = predict(scene_of({(5, 5): 1}), Action(3), mem)
    assert p.scene is None and p.unknown == (("A3", PSIG),)


def test_borrowed_fact_sets_borrowed() -> None:
    mem = mem_with((("A3", f"c2:{DOT}", "any"), Effect("move", dx=-1)))
    p = predict(scene_of({(5, 5): 1}), Action(3), mem)
    assert p.borrowed and player_in(p.scene).x == 4


def test_borrowed_fact_miss_fails_check_without_blaming_the_sibling() -> None:
    before, diff = transition({(5, 5): 1}, {(5, 5): 1})  # it stayed put
    mem = mem_with((("A3", f"c2:{DOT}", "any"), Effect("move", dx=-1)))
    p = predict(before, Action(3), mem)
    assert p.borrowed and p.lent == {player_in(before).id}
    assert check(p, before, diff) == (False, [])
    _, moved = transition({(5, 5): 1}, {(4, 5): 1})
    assert check(p, before, moved) == (True, [])


def test_click_uses_self_subject() -> None:
    before = scene_of({(5, 5): 1, (9, 9): 2})
    mem = mem_with(
        ((f"A6:{PSIG}@0", "self", "any"), Effect("recolor", color=6)),
        ((f"A6:{PSIG}@0", f"c2:{DOT}", "any"), Effect("none")),
    )
    p = predict(before, Action(6, 5, 5), mem)
    assert at(p.scene, f"c6:{DOT}") == [(5, 5)]
    assert check(p, before, transition({(5, 5): 1, (9, 9): 2}, {(5, 5): 6, (9, 9): 2})[1]) == (
        True,
        [],
    )


def test_toggle_group_recolors_every_thing_of_that_sig() -> None:
    before = scene_of({(5, 5): 1, (20, 20): 2, (30, 30): 2})
    two = f"c2:{DOT}"
    mem = mem_with(
        (("A5", PSIG, "any"), Effect("toggle_group", sig=two, color=8)),
    )
    p = predict(before, Action(5), mem)
    assert p.unknown == () and at(p.scene, f"c8:{DOT}") == [(20, 20), (30, 30)]


def test_disappear_and_spawn() -> None:
    before = scene_of({(5, 5): 1, (9, 9): 2})
    two = f"c2:{DOT}"
    mem = mem_with(
        (("A5", PSIG, "any"), Effect("none")),
        (("A5", two, "any"), Effect("disappear")),
        (("A5", SPAWN, "any"), Effect("appear", sig=two, at=(40, 40))),
    )
    mem.record(("A5", SPAWN, "any"), (Effect("appear", sig=two, at=(40, 40)),), 1)
    p = predict(before, Action(5), mem)
    assert at(p.scene, two) == [(40, 40)] and ("A5", SPAWN, "any") in p.used
    pid, gid = player_in(before).id, at_id(before, two)
    diff_ok = SceneDiff(
        (
            Change(pid, "same", color=1, sig=PSIG),
            Change(gid, "disappeared"),
            Change(99, "appeared", color=2, sig=two),
        ),
        0,
    )
    assert check(p, before, diff_ok) == (True, [])
    no_spawn = SceneDiff(tuple(c for c in diff_ok.changes if c.kind != "appeared"), 0)
    assert check(p, before, no_spawn) == (False, [("A5", SPAWN, "any")])


def test_unpredicted_thing_never_counts_as_a_miss() -> None:
    before, diff = transition(
        {(5, 5): 1, (9, 9): 2, (10, 9): 2}, {(4, 5): 1, (9, 9): 7, (10, 9): 7}
    )
    p = predict(before, Action(3), mem_with((("A3", PSIG, "any"), Effect("move", dx=-1))))
    assert p.scene is None
    assert check(p, before, diff) == (True, [])


def test_teleport_lands_at_its_spot() -> None:
    before, diff = transition({(5, 5): 1}, {(15, 2): 1})
    p = predict(before, Action(1), mem_with((("A1", PSIG, "any"), Effect("teleport", at=(15, 2)))))
    assert player_in(p.scene).x == 15 and check(p, before, diff) == (True, [])


def test_prediction_is_a_frozen_value() -> None:
    p = predict(scene_of({(5, 5): 1}), Action(1), FactMemory())
    assert isinstance(p, Prediction) and p.effects == {}


# --- speed ----------------------------------------------------------------------------


def test_thousand_predictions_on_thirty_things_under_half_a_second() -> None:
    cells: Cells = {}
    for i in range(30):
        x, y = 2 + (i % 6) * 9, 2 + (i // 6) * 9
        color = 1 + i % 9
        cells.update({(x, y): color, (x + 1, y): color, (x, y + 1): color})
    scene = scene_of(cells)
    assert len(scene.things) == 30
    mem = FactMemory()
    for t in scene.things:
        eff = Effect("move", dx=1) if t.color in (1, 2) else Effect("none")
        mem.record(("A4", t.sig, "any"), (eff,), 0)
        mem.record(("any", t.sig, "touch:9"), (Effect("game_over"),), 0)
    mem.blockers = {t.sig: {5, 6} for t in scene.things}
    action = Action(4)
    p = predict(scene, action, mem)
    start = time.perf_counter()
    for _ in range(1000):
        p = predict(scene, action, mem)
    elapsed = time.perf_counter() - start
    assert p.scene is not None
    assert elapsed < 0.5, elapsed


# --- terrain and spawn (Task 8b) ------------------------------------------------------


def corridor(cells: Cells) -> Grid:
    """Background 4 (the walls); a 48x4 floor of color 3 on rows 10..13 (terrain)."""
    g = np.full((64, 64), 4, dtype=np.int8)
    g[10:14, 0:48] = 3
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def test_terrain_blocker_color_blocks_the_move() -> None:
    scene = Tracker().update(corridor({(5, 10): 1}), frozenset())[0]
    mem = mem_with(
        (("A1", PSIG, "any"), Effect("move", dy=-1)),
        (("A2", PSIG, "any"), Effect("move", dy=1)),
    )
    mem.blockers[PSIG] = {4}
    up = predict(scene, Action(1), mem)
    assert player_in(up.scene).y == 10 and up.effects[player_in(scene).id] == (Effect("none"),)
    down = predict(scene, Action(2), mem)
    assert player_in(down.scene).y == 11
    assert down.scene is not None and down.scene.terrain is scene.terrain


def test_spawn_is_never_an_unknown_pair() -> None:
    mem = mem_with(
        (("A5", PSIG, "any"), Effect("none")),
        (("A5", SPAWN, "any"), Effect("appear", sig=PSIG, at=(1, 1))),
    )
    break_fact(mem, ("A5", SPAWN, "any"))
    p = predict(scene_of({(5, 5): 1}), Action(5), mem)
    assert ("A5", SPAWN) not in p.unknown and p.scene is not None
    assert at(p.scene, PSIG) == [(5, 5)]
    q = predict(scene_of({(5, 5): 1}), Action(5), mem_with((("A5", PSIG, "any"), Effect("none"))))
    assert q.unknown == () and q.scene is not None


# --- side objects and spawn evidence (Task 8c) ----------------------------------------


def test_one_hit_spawn_fact_predicts_nothing() -> None:
    key = ("A5", SPAWN, "any")
    appear = Effect("appear", sig=PSIG, at=(1, 1))
    mem = mem_with((("A5", PSIG, "any"), Effect("none")), (key, appear))
    p = predict(scene_of({(5, 5): 1}), Action(5), mem)
    assert at(p.scene, PSIG) == [(5, 5)] and key not in p.used
    mem.record(key, (appear,), 1)
    q = predict(scene_of({(5, 5): 1}), Action(5), mem)
    assert at(q.scene, PSIG) == [(1, 1), (5, 5)] and key in q.used


def test_assume_still_predicts_side_objects_none() -> None:
    two = f"c2:{shape_of([(0, 0), (1, 0)])}"  # its own shape: nothing to borrow
    scene = scene_of({(5, 5): 1, (30, 30): 2, (31, 30): 2})
    mem = mem_with((("A3", PSIG, "any"), Effect("move", dx=-1)))
    plain = predict(scene, Action(3), mem)
    assert plain.scene is None and ("A3", two) in plain.unknown and plain.assumed == ()
    still = predict(scene, Action(3), mem, assume_still=frozenset({PSIG}))
    assert still.unknown == () and still.assumed == (("A3", two),)
    assert at(still.scene, two) == [(30, 30)] and player_in(still.scene).x == 4
    assert at_id(scene, two) not in still.effects
    guarded = predict(scene, Action(3), mem, assume_still=frozenset({PSIG, two}))
    assert guarded.scene is None and ("A3", two) in guarded.unknown


def test_assume_still_covers_broken_side_facts_but_not_protected_ones() -> None:
    two = f"c2:{shape_of([(0, 0), (1, 0)])}"  # its own shape: nothing to borrow
    scene = scene_of({(5, 5): 1, (30, 30): 2, (31, 30): 2})
    mem = mem_with(
        (("A3", PSIG, "any"), Effect("move", dx=-1)),
        (("A3", two, "any"), Effect("recolor", color=6)),
    )
    break_fact(mem, ("A3", two, "any"))
    still = predict(scene, Action(3), mem, assume_still=frozenset({PSIG}))
    assert still.scene is not None and still.assumed == (("A3", two),)
    break_fact(mem, ("A3", PSIG, "any"))
    lost = predict(scene, Action(3), mem, assume_still=frozenset({PSIG}))
    assert lost.scene is None and ("A3", PSIG) in lost.unknown


def test_spawns_false_predicts_no_spawn() -> None:
    key = ("A5", SPAWN, "any")
    appear = Effect("appear", sig=PSIG, at=(1, 1))
    mem = mem_with((("A5", PSIG, "any"), Effect("none")), (key, appear))
    mem.record(key, (appear,), 1)
    p = predict(scene_of({(5, 5): 1}), Action(5), mem, spawns=False)
    assert at(p.scene, PSIG) == [(5, 5)] and key not in p.used
