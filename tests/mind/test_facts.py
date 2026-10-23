"""Tests for the fact memory, its table of contents, and learning from a transition."""

from __future__ import annotations

import time

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.facts import (
    SPAWN,
    Effect,
    Fact,
    FactKey,
    FactMemory,
    action_label,
    learn,
    touch_contexts,
)
from arcagent.mind.objects import Scene, SceneDiff, Thing, Tracker

Cells = dict[tuple[int, int], int]


def grid_with(cells: Cells, bg: int = 0) -> Grid:
    g = np.full((64, 64), bg, dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def transition(before: Cells, after: Cells) -> tuple[Scene, Scene, SceneDiff]:
    t = Tracker()
    s0, _ = t.update(grid_with(before), frozenset())
    s1, d = t.update(grid_with(after), frozenset())
    assert d is not None
    return s0, s1, d


def thing_at(scene: Scene, x: int, y: int) -> Thing:
    return next(t for t in scene.things if (x, y) in t.cells)


def fact(mem: FactMemory, key: FactKey) -> Fact:
    f = mem.get(key)
    assert f is not None, key
    return f


def learn_plain(
    mem: FactMemory, before: Cells, after: Cells, action: Action, move: int
) -> tuple[Scene, Scene, list[FactKey]]:
    s0, s1, d = transition(before, after)
    keys = learn(mem, s0, action, d, move=move, level_up=False, game_over=False, after=s1)
    return s0, s1, keys


# --- record, status, text -------------------------------------------------------------


def test_new_fact_then_hits_until_solid() -> None:
    mem = FactMemory()
    key = ("A3", "c1:aaaa", "any")
    assert mem.record(key, (Effect("move", dx=-1),), 0) == "new"
    for i in range(1, 5):
        assert fact(mem, key).status == "thin"
        assert mem.record(key, (Effect("move", dx=-1),), i) == "hit"
    assert fact(mem, key).status == "solid" and fact(mem, key).evidence == [0, 1, 2, 3, 4]


def test_two_unabsorbed_misses_break_a_fact() -> None:
    mem = FactMemory()
    key = ("A1", "c1:aaaa", "any")
    mem.record(key, (Effect("move", dy=-1),), 0)
    assert mem.record(key, (Effect("recolor", color=3),), 1) == "miss"
    assert fact(mem, key).status == "thin"
    mem.record(key, (Effect("disappear"),), 2)
    f = fact(mem, key)
    assert f.status == "broken" and f.effects == (Effect("move", dy=-1),) and f.misses == 2


def test_effect_text_and_fact_line() -> None:
    assert Effect("move", dx=-1).text() == "moves 1 cell left"
    assert Effect("move", dy=3).text() == "moves 3 cells down"
    assert Effect("none").text() == "no change"
    assert Effect("slide", dx=1).text() == "slides right until blocked"
    assert Effect("teleport", at=(4, 7)).text() == "teleports to (4, 7)"
    mem = FactMemory()
    key = ("A3", "c1:ab12cdef01", "any")
    for i in range(6):
        mem.record(key, (Effect("move", dx=-1),), i)
    assert fact(mem, key).line() == "A3 c1:ab12.. any: moves 1 cell left [solid 6/0]"


# --- table of contents and lookup -----------------------------------------------------


def test_toc_finds_facts_by_behavior_and_borrows_for_siblings() -> None:
    mem = FactMemory()
    mem.record(("A1", "c2:shape1", "any"), (Effect("move", dy=-1),), 0)
    assert [f.key for f in mem.by_tag("move")] == [("A1", "c2:shape1", "any")]
    found = mem.lookup("A1", "c5:shape1", "any")  # same shape, new color
    assert found is not None
    f, borrowed = found
    assert borrowed and f.effects[0].kind == "move"
    assert mem.lookup("A1", "c2:other", "any") is None  # same color, new shape: no borrow


def test_lookup_order_exact_then_any_then_sibling_then_action_wide() -> None:
    mem = FactMemory()
    assert mem.lookup("A2", "c1:s", "blocked") is None
    mem.record(("A2", "any", "any"), (Effect("none"),), 0)
    found = mem.lookup("A2", "c1:s", "blocked")
    assert found is not None and found[0].key == ("A2", "any", "any") and not found[1]
    mem.record(("A2", "c7:s", "any"), (Effect("none"),), 1)
    found = mem.lookup("A2", "c1:s", "blocked")
    assert found is not None and found[0].key == ("A2", "c7:s", "any") and found[1]
    mem.record(("A2", "c1:s", "any"), (Effect("move", dx=1),), 2)
    found = mem.lookup("A2", "c1:s", "blocked")
    assert found is not None and found[0].key == ("A2", "c1:s", "any") and not found[1]
    mem.record(("A2", "c1:s", "blocked"), (Effect("none"),), 3)
    found = mem.lookup("A2", "c1:s", "blocked")
    assert found is not None and found[0].key == ("A2", "c1:s", "blocked")


def test_broken_facts_are_not_borrowed() -> None:
    mem = FactMemory()
    key = ("A1", "c2:shape1", "any")
    mem.record(key, (Effect("move", dy=-1),), 0)
    mem.record(key, (Effect("none"),), 1)
    mem.record(key, (Effect("none"),), 2)
    assert mem.lookup("A1", "c5:shape1", "any") is None


def test_tags_follow_effects() -> None:
    mem = FactMemory()
    mem.record(("A1", "c1:a", "any"), (Effect("disappear"),), 0)
    mem.record(("A1", "c2:a", "any"), (Effect("recolor", color=4),), 0)
    mem.record(("A1", "any", "any"), (Effect("appear", sig="c3:b"),), 0)
    mem.record(("A1", "c3:a", "blocked"), (Effect("none"),), 0)
    mem.record(("A1", "c4:a", "any"), (Effect("slide", dx=1),), 0)
    assert fact(mem, ("A1", "c1:a", "any")).tags == {"collect"}
    assert fact(mem, ("A1", "c2:a", "any")).tags == {"toggle"}
    assert fact(mem, ("A1", "any", "any")).tags == {"spawn"}
    assert fact(mem, ("A1", "c3:a", "blocked")).tags == {"no_effect", "block"}
    assert fact(mem, ("A1", "c4:a", "any")).tags == {"move", "moves until blocked"}
    assert [f.key for f in mem.by_tag("block")] == [("A1", "c3:a", "blocked")]


# --- action labels --------------------------------------------------------------------


def test_click_labels_use_the_clicked_sig_and_self() -> None:
    s0, _, _ = transition({(5, 5): 2}, {(5, 5): 2})
    thing = thing_at(s0, 5, 5)
    assert action_label(Action(6, 5, 5), s0) == f"A6:{thing.sig}"
    assert action_label(Action(6, 60, 60), s0) == "A6:none"
    assert action_label(Action(3), s0) == "A3"


# --- learn ----------------------------------------------------------------------------


def test_plain_transition_learns_move_and_none_for_unchanged() -> None:
    mem = FactMemory()
    s0, _, keys = learn_plain(mem, {(6, 5): 1, (30, 30): 8}, {(5, 5): 1, (30, 30): 8}, Action(3), 0)
    player, wall = thing_at(s0, 6, 5), thing_at(s0, 30, 30)
    assert keys == [("A3", player.sig, "any"), ("A3", wall.sig, "any")]
    assert fact(mem, keys[0]).effects == (Effect("move", dx=-1),)
    assert fact(mem, keys[1]).effects == (Effect("none"),)
    assert "no_effect" in fact(mem, keys[1]).tags


def test_wall_makes_blocked_fact_not_a_miss() -> None:
    mem = FactMemory()
    learn_plain(mem, {(6, 5): 1, (4, 5): 8}, {(5, 5): 1, (4, 5): 8}, Action(3), 6)
    before, _, same_diff = transition({(5, 5): 1, (4, 5): 8}, {(5, 5): 1, (4, 5): 8})
    player = thing_at(before, 5, 5)
    keys = learn(mem, before, Action(3), same_diff, move=7, level_up=False, game_over=False)
    assert ("A3", player.sig, "blocked") in keys
    assert fact(mem, ("A3", player.sig, "any")).misses == 0 and 8 in mem.blockers[player.sig]
    assert fact(mem, ("A3", player.sig, "blocked")).tags == {"no_effect", "block"}


def test_grid_edge_blocks_without_adding_a_color() -> None:
    mem = FactMemory()
    learn_plain(mem, {(1, 5): 1}, {(0, 5): 1}, Action(3), 0)
    s0, _, keys = learn_plain(mem, {(0, 5): 1}, {(0, 5): 1}, Action(3), 1)
    sig = thing_at(s0, 0, 5).sig
    assert keys == [("A3", sig, "blocked")] and mem.blockers[sig] == set()


def test_no_change_in_open_space_is_a_miss() -> None:
    mem = FactMemory()
    learn_plain(mem, {(6, 5): 1}, {(5, 5): 1}, Action(3), 0)
    s0, _, keys = learn_plain(mem, {(5, 5): 1}, {(5, 5): 1}, Action(3), 1)
    sig = thing_at(s0, 5, 5).sig
    assert keys == [("A3", sig, "any")] and fact(mem, keys[0]).misses == 1


def test_same_sig_disagreement_splits_blocked_and_free_once_each() -> None:
    mem = FactMemory()
    before = {(5, 5): 1, (4, 5): 8, (10, 10): 1, (20, 10): 1}
    after = {(5, 5): 1, (4, 5): 8, (9, 10): 1, (19, 10): 1}
    s0, _, keys = learn_plain(mem, before, after, Action(3), 0)
    sig = thing_at(s0, 5, 5).sig
    assert keys.count(("A3", sig, "any")) == 1 and keys.count(("A3", sig, "blocked")) == 1
    assert fact(mem, ("A3", sig, "any")).effects == (Effect("move", dx=-1),)
    assert 8 in mem.blockers[sig]


def test_mover_entering_another_objects_cells_records_pushed() -> None:
    # not adjacent before the move, so the two are not one co-moving sprite
    mem = FactMemory()
    s0, _, keys = learn_plain(mem, {(6, 5): 1, (4, 5): 3}, {(4, 5): 1, (2, 5): 3}, Action(3), 0)
    player, box = thing_at(s0, 6, 5), thing_at(s0, 4, 5)
    assert set(keys) == {("A3", player.sig, "any"), ("A3", box.sig, "pushed")}
    pushed = fact(mem, ("A3", box.sig, "pushed"))
    assert pushed.effects == (Effect("move", dx=-2),) and "push" in pushed.tags


def test_longer_move_ending_blocked_rewrites_to_slide_as_a_hit() -> None:
    mem = FactMemory()
    s0, _, _ = learn_plain(mem, {(12, 5): 1, (2, 5): 8}, {(11, 5): 1, (2, 5): 8}, Action(3), 0)
    key = ("A3", thing_at(s0, 12, 5).sig, "any")
    learn_plain(mem, {(10, 5): 1, (2, 5): 8}, {(3, 5): 1, (2, 5): 8}, Action(3), 1)
    f = fact(mem, key)
    assert f.effects == (Effect("slide", dx=-1),) and (f.hits, f.misses) == (2, 0)
    assert "moves until blocked" in f.tags
    learn_plain(mem, {(9, 5): 1, (2, 5): 8}, {(3, 5): 1, (2, 5): 8}, Action(3), 2)
    assert (fact(mem, key).hits, fact(mem, key).misses) == (3, 0)
    # pressed against the wall: the slide fact is absorbed as blocked
    learn_plain(mem, {(3, 5): 1, (2, 5): 8}, {(3, 5): 1, (2, 5): 8}, Action(3), 3)
    assert fact(mem, key).misses == 0 and fact(mem, ("A3", key[1], "blocked")).hits == 1


def test_longer_move_in_open_space_is_a_miss_not_a_slide() -> None:
    mem = FactMemory()
    s0, _, _ = learn_plain(mem, {(20, 5): 1}, {(19, 5): 1}, Action(3), 0)
    key = ("A3", thing_at(s0, 20, 5).sig, "any")
    learn_plain(mem, {(20, 5): 1}, {(15, 5): 1}, Action(3), 1)
    assert fact(mem, key).effects == (Effect("move", dx=-1),) and fact(mem, key).misses == 1


def test_same_landing_from_different_starts_rewrites_to_teleport() -> None:
    mem = FactMemory()
    s0, _, _ = learn_plain(mem, {(5, 5): 1}, {(5, 4): 1}, Action(1), 0)
    key = ("A1", thing_at(s0, 5, 5).sig, "any")
    learn_plain(mem, {(9, 9): 1}, {(5, 4): 1}, Action(1), 1)
    f = fact(mem, key)
    assert f.effects == (Effect("teleport", at=(5, 4)),) and (f.hits, f.misses) == (2, 0)
    learn_plain(mem, {(12, 1): 1}, {(5, 4): 1}, Action(1), 2)
    assert (fact(mem, key).hits, fact(mem, key).misses) == (3, 0)


def test_appear_records_position_under_any() -> None:
    mem = FactMemory()
    _, s1, keys = learn_plain(mem, {(5, 5): 1}, {(5, 5): 1, (30, 31): 4}, Action(5), 0)
    new = thing_at(s1, 30, 31)
    assert ("A5", SPAWN, "any") in keys
    assert fact(mem, ("A5", SPAWN, "any")).effects == (Effect("appear", sig=new.sig, at=(30, 31)),)
    assert mem.lookup("A5", "c3:unknown", "any") is None
    assert mem.lookup("A5", "any", "any") is None


def test_click_subject_is_self_and_others_keep_their_sig() -> None:
    mem = FactMemory()
    s0, _, keys = learn_plain(
        mem, {(5, 5): 2, (20, 20): 7}, {(5, 5): 3, (20, 20): 7}, Action(6, 5, 5), 0
    )
    label = f"A6:{thing_at(s0, 5, 5).sig}"
    other = thing_at(s0, 20, 20)
    assert keys == [(label, "self", "any"), (label, other.sig, "any")]
    assert fact(mem, (label, "self", "any")).effects == (Effect("recolor", color=3),)


def test_reshape_with_same_cell_count_is_tagged_rotate() -> None:
    mem = FactMemory()
    s0, s1, keys = learn_plain(
        mem, {(5, 5): 2, (5, 6): 2, (6, 6): 2}, {(5, 5): 2, (6, 5): 2, (5, 6): 2}, Action(1), 0
    )
    new_sig = thing_at(s1, 6, 5).sig
    f = fact(mem, ("A1", thing_at(s0, 5, 5).sig, "any"))
    assert keys == [f.key] and f.effects == (Effect("reshape", sig=new_sig),)
    assert "rotate" in f.tags


def test_disappear_is_collect() -> None:
    mem = FactMemory()
    s0, _, keys = learn_plain(mem, {(5, 5): 1, (20, 20): 6}, {(5, 5): 1}, Action(2), 0)
    key = ("A2", thing_at(s0, 20, 20).sig, "any")
    assert key in keys and fact(mem, key).tags == {"collect"}


def test_game_over_records_touch_fact_only() -> None:
    mem = FactMemory()
    before, _, diff = transition({(5, 5): 1, (3, 5): 9}, {(4, 5): 1, (3, 5): 9})
    player = thing_at(before, 5, 5)
    keys = learn(mem, before, Action(4), diff, move=3, level_up=False, game_over=True)
    assert keys == [("any", player.sig, "near:9")]
    assert list(mem.facts) == keys
    f = fact(mem, keys[0])
    assert f.effects == (Effect("game_over"),) and f.tags == {"die"}


def test_level_up_records_event_only() -> None:
    mem = FactMemory()
    before, _, diff = transition({(5, 5): 1}, {(4, 5): 1})
    keys = learn(mem, before, Action(3), diff, move=9, level_up=True, game_over=False)
    assert keys == [("event", "level_up", "any")] and list(mem.facts) == keys
    assert fact(mem, keys[0]).tags == {"win"}


def test_lookup_never_returns_a_spawn_fact_for_a_subject() -> None:
    mem = FactMemory()
    mem.record(("A2", "any", "any"), (Effect("appear", sig="c4:x", at=(1, 1)),), 0)
    assert mem.lookup("A2", "c1:s", "any") is None


def test_game_over_prefers_the_overlapped_color() -> None:
    mem = FactMemory()
    before, _, diff = transition({(6, 5): 1, (5, 5): 9, (5, 4): 8}, {(5, 5): 1, (5, 4): 8})
    player = thing_at(before, 6, 5)
    keys = learn(mem, before, Action(3), diff, move=0, level_up=False, game_over=True)
    assert keys == [("any", player.sig, "touch:9")]


def test_wall_beside_lava_is_spared_only_after_a_safe_touch() -> None:
    lava_row = {(5, 4): 8, (5, 6): 9}
    before, _, diff = transition({(6, 5): 1, **lava_row}, {(5, 5): 1, **lava_row})
    sig = thing_at(before, 6, 5).sig
    # no safe history: both adjacent colors are suspects
    mem = FactMemory()
    keys = learn(mem, before, Action(3), diff, move=1, level_up=False, game_over=True)
    assert keys == [("any", sig, "near:8"), ("any", sig, "near:9")]
    # the wall was touched safely first: only lava gets the death fact
    mem = FactMemory()
    learn_plain(mem, {(6, 3): 1, **lava_row}, {(6, 4): 1, **lava_row}, Action(2), 0)
    assert 8 in mem.safe_near[sig] and 9 not in mem.safe_near[sig]
    keys = learn(mem, before, Action(3), diff, move=1, level_up=False, game_over=True)
    assert keys == [("any", sig, "near:9")]


def test_two_safe_passes_beside_break_a_near_death_fact() -> None:
    mem = FactMemory()
    before, _, diff = transition({(6, 5): 1, (4, 5): 9}, {(5, 5): 1, (4, 5): 9})
    key = ("any", thing_at(before, 6, 5).sig, "near:9")
    learn(mem, before, Action(3), diff, move=0, level_up=False, game_over=True)
    _, _, keys = learn_plain(mem, {(3, 6): 1, (4, 5): 9}, {(3, 5): 1, (4, 5): 9}, Action(1), 1)
    assert key in keys and fact(mem, key).status == "thin"
    learn_plain(mem, {(4, 7): 1, (4, 5): 9}, {(4, 6): 1, (4, 5): 9}, Action(1), 2)
    assert fact(mem, key).status == "broken"


def test_touch_contexts_split_overlap_and_adjacency() -> None:
    s0, _, _ = transition({(5, 5): 9, (5, 4): 8, (9, 9): 7}, {(5, 5): 9, (5, 4): 8, (9, 9): 7})
    cells = frozenset({(5, 5), (6, 5)})
    assert touch_contexts(cells, s0, -1) == ["touch:9", "near:8"]
    assert touch_contexts(cells, s0, thing_at(s0, 5, 5).id) == ["near:8"]


def test_passing_beside_a_stepped_on_killer_leaves_its_touch_fact_solid() -> None:
    mem = FactMemory()
    before, _, diff = transition({(6, 5): 1, (5, 5): 9}, {(5, 5): 1})
    key = ("any", thing_at(before, 6, 5).sig, "touch:9")
    for i in range(5):
        assert learn(mem, before, Action(3), diff, move=i, level_up=False, game_over=True) == [key]
    assert fact(mem, key).status == "solid"
    learn_plain(mem, {(5, 7): 1, (5, 5): 9}, {(5, 6): 1, (5, 5): 9}, Action(1), 5)
    learn_plain(mem, {(4, 4): 1, (5, 5): 9}, {(4, 5): 1, (5, 5): 9}, Action(2), 6)
    assert 9 in mem.safe_near[key[1]] and 9 not in mem.safe_touch.get(key[1], set())
    assert fact(mem, key).status == "solid" and fact(mem, key).misses == 0


def test_safe_overlap_charges_only_the_touch_fact() -> None:
    mem = FactMemory()
    s0, _, _ = learn_plain(mem, {(6, 5): 1, (5, 5): 6}, {(5, 5): 1}, Action(3), 0)
    sig = thing_at(s0, 6, 5).sig
    assert 6 in mem.safe_touch[sig]
    mem.record(("any", sig, "touch:6"), (Effect("game_over"),), 1)
    mem.record(("any", sig, "near:6"), (Effect("game_over"),), 1)
    learn_plain(mem, {(6, 5): 1, (5, 5): 6}, {(5, 5): 1}, Action(3), 2)
    assert fact(mem, ("any", sig, "touch:6")).misses == 1
    assert fact(mem, ("any", sig, "near:6")).misses == 0


def test_rigid_two_color_sprite_moves_as_one_and_stops_at_a_wall() -> None:
    mem = FactMemory()
    wall = {(10, 14): 8}
    for i in range(5):
        y = 20 - i
        learn_plain(
            mem,
            {(10, y): 1, (10, y + 1): 2, **wall},
            {(10, y - 1): 1, (10, y): 2, **wall},
            Action(1),
            i,
        )
    s0, _, keys = learn_plain(
        mem, {(10, 15): 1, (10, 16): 2, **wall}, {(10, 15): 1, (10, 16): 2, **wall}, Action(1), 5
    )
    head, body = thing_at(s0, 10, 15), thing_at(s0, 10, 16)
    head_fact = fact(mem, ("A1", head.sig, "any"))
    assert head_fact.effects == (Effect("move", dy=-1),) and head_fact.status == "solid"
    assert not [k for k in mem.facts if k[2] == "pushed"]
    assert 1 not in mem.blockers.get(body.sig, set()) and mem.blockers[head.sig] == {8}
    assert not [f for f in mem.facts.values() if f.effects[0].kind == "slide"]
    assert ("A1", head.sig, "blocked") in keys and ("A1", body.sig, "blocked") in keys


def test_broken_fact_reseeds_from_the_next_observation() -> None:
    mem = FactMemory()
    wall = {(4, 5): 8}
    s0, _, _ = learn_plain(mem, {(5, 5): 1, **wall}, {(5, 5): 1, **wall}, Action(3), 0)
    key = ("A3", thing_at(s0, 5, 5).sig, "any")
    assert fact(mem, key).effects == (Effect("none"),)
    for i in range(1, 8):
        learn_plain(mem, {(31, 30): 1, **wall}, {(30, 30): 1, **wall}, Action(3), i)
        if i == 2:
            assert fact(mem, key).status == "broken"
    f = fact(mem, key)
    assert f.effects == (Effect("move", dx=-1),) and f.status == "solid"
    assert (f.hits, f.misses, f.streak) == (5, 2, 0)


def test_isolated_misses_do_not_break_a_solid_fact() -> None:
    mem = FactMemory()
    key = ("A3", "c1:a", "any")
    good, bad = (Effect("move", dx=-1),), (Effect("none"),)
    for i, effects in enumerate([good] * 6 + [bad, good, bad]):
        mem.record(key, effects, i)
    f = fact(mem, key)
    assert f.status != "broken" and f.misses == 2 and f.streak == 1 and f.effects == good


def test_reseed_returns_new() -> None:
    mem = FactMemory()
    key = ("A3", "c1:a", "any")
    mem.record(key, (Effect("none"),), 0)
    mem.record(key, (Effect("move", dx=1),), 1)
    mem.record(key, (Effect("move", dx=1),), 2)
    assert mem.record(key, (Effect("move", dx=1),), 3) == "new"
    assert fact(mem, key).status == "thin" and fact(mem, key).hits == 1


def test_evidence_keeps_the_last_20_moves() -> None:
    mem = FactMemory()
    key = ("A3", "c1:a", "any")
    for i in range(30):
        mem.record(key, (Effect("move", dx=-1),), i)
    assert fact(mem, key).evidence == list(range(10, 30)) and fact(mem, key).hits == 30


def test_fixed_step_cut_short_by_a_wall_is_not_a_slide() -> None:
    mem = FactMemory()
    s0, _, _ = learn_plain(mem, {(20, 5): 1, (3, 5): 8}, {(18, 5): 1, (3, 5): 8}, Action(3), 0)
    key = ("A3", thing_at(s0, 20, 5).sig, "any")
    learn_plain(mem, {(5, 5): 1, (3, 5): 8}, {(4, 5): 1, (3, 5): 8}, Action(3), 1)
    f = fact(mem, key)
    assert f.effects == (Effect("move", dx=-2),) and f.misses == 1


def test_new_level_forgets_landings() -> None:
    mem = FactMemory()
    s0, _, _ = learn_plain(mem, {(5, 5): 1}, {(5, 4): 1}, Action(1), 0)
    key = ("A1", thing_at(s0, 5, 5).sig, "any")
    assert key in mem.landed and key in mem.last_move
    mem.new_level()
    assert not mem.landed and not mem.last_move
    learn_plain(mem, {(9, 9): 1}, {(5, 4): 1}, Action(1), 1)
    assert fact(mem, key).effects == (Effect("move", dy=-1),)  # no teleport across levels


def test_learn_is_fast_on_300_static_things() -> None:
    statics = {(2 * (i % 30) + 2, 2 * (i // 30) + 2): 1 + i % 9 for i in range(300)}
    s0, s1, d = transition({**statics, (10, 40): 10}, {**statics, (9, 40): 10})
    assert len(s0.things) == 301
    mem = FactMemory()
    t0 = time.perf_counter()
    for i in range(20):
        learn(mem, s0, Action(3), d, move=i, level_up=False, game_over=False, after=s1)
    assert (time.perf_counter() - t0) / 20 < 0.010


def test_box_seen_standing_still_is_pushed_when_shoved() -> None:
    mem = FactMemory()
    for i, x in enumerate((5, 6, 7)):
        learn_plain(mem, {(x, 5): 1, (9, 5): 3}, {(x + 1, 5): 1, (9, 5): 3}, Action(4), i)
    box_sig = transition({(9, 5): 3}, {(9, 5): 3})[0].things[0].sig
    assert fact(mem, ("A4", box_sig, "any")).effects == (Effect("none"),)
    _, _, keys = learn_plain(mem, {(8, 5): 1, (9, 5): 3}, {(9, 5): 1, (10, 5): 3}, Action(4), 3)
    assert ("A4", box_sig, "pushed") in keys
    assert fact(mem, ("A4", box_sig, "pushed")).effects == (Effect("move", dx=1),)
    stay = fact(mem, ("A4", box_sig, "any"))
    assert stay.effects == (Effect("none"),) and stay.misses == 0 and stay.hits == 3


# --- terrain colors block (Task 8b) ---------------------------------------------------


def corridor(cells: Cells) -> Grid:
    """Background 4 (the walls); a 48x4 floor of color 3 on rows 10..13 (terrain)."""
    g = np.full((64, 64), 4, dtype=np.int8)
    g[10:14, 0:48] = 3
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def corridor_walk(mem: FactMemory, path: list[tuple[int, tuple[int, int]]]) -> Scene:
    """Learn each (action, player cell after) step in one tracker; return the last scene."""
    t = Tracker()
    scene, _ = t.update(corridor({path[0][1]: 1}), frozenset())
    for move, (action, cell) in enumerate(path[1:]):
        after, d = t.update(corridor({cell: 1}), frozenset())
        assert d is not None
        learn(
            mem, scene, Action(action), d, move=move, level_up=False, game_over=False, after=after
        )
        scene = after
    return scene


def test_background_wall_blocks_and_floor_does_not() -> None:
    mem = FactMemory()
    start = (5, 11)
    path = [(0, start), (2, (5, 12)), (1, (5, 11)), (1, (5, 10)), (1, (5, 10))]
    scene = corridor_walk(mem, path)
    psig = scene.things[0].sig
    assert [t.color for t in scene.things] == [1]
    assert 4 in mem.blockers[psig] and 3 not in mem.blockers[psig]
    assert 3 in mem.walkable[psig]
    blocked = fact(mem, ("A1", psig, "blocked"))
    assert blocked.effects == (Effect("none"),) and fact(mem, ("A1", psig, "any")).misses == 0


def test_walkable_color_never_becomes_a_blocker() -> None:
    mem = FactMemory()
    psig = Tracker().update(corridor({(5, 11): 1}), frozenset())[0].things[0].sig
    mem.blockers[psig] = {3}
    path = [(0, (5, 11)), (3, (4, 11)), (3, (4, 11))]
    corridor_walk(mem, path)
    assert 3 not in mem.blockers.get(psig, set()) and 3 in mem.walkable[psig]
    # the no-change onto walkable floor is a miss of the move fact, never a blocked fact
    assert fact(mem, ("A3", psig, "any")).misses == 1
    assert mem.get(("A3", psig, "blocked")) is None
