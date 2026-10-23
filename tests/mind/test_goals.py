"""Tests for goal guesses: predicates, proposal heuristics and status transitions."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.facts import Effect, FactMemory
from arcagent.mind.goals import Goal, GoalBook
from arcagent.mind.objects import Scene, Tracker

Cells = dict[tuple[int, int], int]

PLAYER = {(10, 10): 1}
T1 = {(30, 10): 4, (31, 10): 4}
T2 = {(10, 40): 5, (11, 40): 5, (10, 41): 5}


def grid_with(cells: Cells, bg: int = 0) -> Grid:
    g = np.full((64, 64), bg, dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(*parts: Cells) -> Scene:
    merged: Cells = {}
    for p in parts:
        merged.update(p)
    return Tracker().update(grid_with(merged), frozenset())[0]


def sig(scene: Scene, color: int) -> str:
    return next(t.sig for t in scene.things if t.color == color)


BASE = scene_of(PLAYER, T1, T2)
ON_T1 = scene_of({(29, 10): 1}, T1, T2)
ON_T2 = scene_of({(9, 40): 1}, T1, T2)
T1SIG, T2SIG, PSIG = sig(BASE, 4), sig(BASE, 5), sig(BASE, 1)


def mover_memory() -> FactMemory:
    mem = FactMemory()
    mem.record(("A3", PSIG, "any"), (Effect("move", dx=-1),), 1)
    return mem


def test_reach_proposed_for_movers_against_other_sigs() -> None:
    goals = GoalBook().propose(BASE, mover_memory(), source="heuristic")
    assert {g.kind for g in goals} == {"reach"} and len(goals) == 2
    assert {g.b for g in goals} == {T1SIG, T2SIG} and {g.a for g in goals} == {PSIG}


def test_goal_true_on_logged_frame_without_level_up_is_ruled_out() -> None:
    book = GoalBook()
    book.propose(BASE, mover_memory(), source="heuristic")
    out = book.rule_out([ON_T1])
    assert [g.b for g in out] == [T1SIG] and book.status[out[0]] == "ruled_out"


def test_reached_without_level_up_is_excluded_and_never_reproposed() -> None:
    book = GoalBook()
    book.propose(BASE, mover_memory(), source="heuristic")
    book.note_reached(ON_T1)
    assert book.propose(BASE, mover_memory(), source="heuristic") == []
    assert [g.b for g in book.open()] == [T2SIG]


def test_level_up_confirms_the_true_goal_and_puts_its_form_first() -> None:
    book = GoalBook()
    book.propose(BASE, mover_memory(), source="heuristic")
    book.level_up(ON_T2)
    assert book.confirmed_forms[0].b == T2SIG and book.open()[0] == book.confirmed_forms[0]
    assert book.status[book.confirmed_forms[0]] == "confirmed"


def test_confirmed_form_survives_later_reach_and_excluded_stays_excluded() -> None:
    book = GoalBook()
    book.propose(BASE, mover_memory(), source="heuristic")
    book.note_reached(ON_T1)
    book.level_up(ON_T2)
    book.note_reached(ON_T2)
    assert [g.b for g in book.confirmed_forms] == [T2SIG]
    assert sorted(book.status.values()) == ["confirmed", "excluded"]


def test_clear_proposed_from_recolor_facts_in_click_games() -> None:
    click = scene_of({(5, 5): 4, (6, 5): 4})
    mem = FactMemory()
    mem.record(("A6:" + sig(click, 4), "self", "any"), (Effect("recolor", color=9),), 1)
    goals = GoalBook().propose(click, mem, source="heuristic")
    assert {(g.kind, g.color) for g in goals} == {("clear", 4), ("clear", 9)}


def test_clear_proposed_for_disappear_with_sig_subject() -> None:
    mem = FactMemory()
    mem.record(("A3", T1SIG, "any"), (Effect("disappear"),), 1)
    goals = GoalBook().propose(BASE, mem, source="heuristic")
    assert [(g.kind, g.color) for g in goals] == [("clear", 4)]


def test_no_reach_against_walls_or_own_sig() -> None:
    wall = {(x, 0): 7 for x in range(64)} | {(x, 63): 7 for x in range(64)}
    wall |= {(0, y): 7 for y in range(64)} | {(63, y): 7 for y in range(64)}
    scene = scene_of(PLAYER, wall)
    assert GoalBook().propose(scene, mover_memory(), source="heuristic") == []
    twin = scene_of(PLAYER, {(40, 40): 1}, T1)
    assert {g.b for g in GoalBook().propose(twin, mover_memory(), source="x")} == {T1SIG}


def test_broken_move_fact_does_not_make_a_mover() -> None:
    mem = mover_memory()
    key = ("A3", PSIG, "any")
    mem.record(key, (Effect("move", dy=1),), 2)
    mem.record(key, (Effect("move", dy=1),), 3)
    assert mem.facts[key].status == "broken"
    assert GoalBook().propose(BASE, mem, source="heuristic") == []


def test_holds_predicates() -> None:
    assert Goal("reach", a=PSIG, b=T1SIG).holds(ON_T1)
    assert not Goal("reach", a=PSIG, b=T1SIG).holds(BASE)
    adjacent = scene_of({(29, 10): 1}, {(30, 10): 4, (31, 10): 4})
    assert Goal("reach", a=PSIG, b=T1SIG).holds(adjacent)
    diagonal = scene_of({(29, 9): 1}, {(30, 10): 4, (31, 10): 4})
    assert not Goal("reach", a=PSIG, b=T1SIG).holds(diagonal)
    assert Goal("clear", color=8).holds(BASE) and not Goal("clear", color=4).holds(BASE)
    assert Goal("count", color=4, n=1).holds(BASE) and not Goal("count", color=4, n=2).holds(BASE)


def test_match_needs_grid_and_compares_regions() -> None:
    g = grid_with({(0, 0): 3, (10, 10): 3})
    goal = Goal("match", region_a=(0, 0, 2, 2), region_b=(10, 10, 12, 12))
    assert goal.holds(BASE, g) and not goal.holds(BASE)
    g[11, 11] = 5
    assert not goal.holds(BASE, g)
    assert not Goal("match", region_a=(0, 0, 2, 2), region_b=(63, 63, 65, 65)).holds(BASE, g)


def test_ids_are_deterministic_text() -> None:
    assert Goal("reach", a="c1:ab", b="c4:cd").id() == "reach c1:ab>c4:cd"
    assert Goal("clear", color=4).id() == "clear c4"
    assert Goal("count", color=4, n=3).id() == "count c4=3"
    assert Goal("match", region_a=(0, 0, 2, 2), region_b=(5, 5, 7, 7)).id() == (
        "match (0, 0, 2, 2)=(5, 5, 7, 7)"
    )
