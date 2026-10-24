"""Tests for goal guesses: predicates, proposal heuristics and status transitions."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.facts import Effect, FactMemory
from arcagent.mind.goals import (
    MAX_OPEN_GOALS,
    MAX_REACH_TARGETS,
    Goal,
    GoalBook,
    controlled_sigs,
)
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


def test_reach_needs_target_cells_not_bbox() -> None:
    ring = {(x, y): 4 for x in range(20, 25) for y in range(20, 25)}
    for x in range(21, 24):
        for y in range(21, 24):
            del ring[(x, y)]
    scene = scene_of({(22, 22): 1}, ring)
    assert not Goal("reach", a=sig(scene, 1), b=sig(scene, 4)).holds(scene)
    outside = scene_of({(19, 22): 1}, ring)
    assert Goal("reach", a=sig(outside, 1), b=sig(outside, 4)).holds(outside)


def test_proposals_are_capped_and_dropped_goals_can_return() -> None:
    cells: Cells = {}
    for i in range(100):
        color, length = i % 15 + 1, i // 15 + 1
        x, y = (i % 16) * 4, (i // 16) * 9
        cells |= {(x, y + k): color for k in range(length)}
    scene = scene_of(cells)
    assert len(scene.things) == 100
    mem = FactMemory()
    for t in scene.things:
        mem.record(("A3", t.sig, "any"), (Effect("move", dx=1),), 1)
    book = GoalBook()
    first = book.propose(scene, mem, source="heuristic")
    assert len(first) == MAX_OPEN_GOALS == len(book.open())
    movers = {g.a for g in first}
    assert all(sum(1 for g in first if g.a == a) <= MAX_REACH_TARGETS for a in movers)
    assert book.propose(scene, mem, source="heuristic") == []
    for g in first[:10]:
        book.status[g] = "excluded"
    again = book.propose(scene, mem, source="heuristic")
    assert len(again) == 10 and not set(again) & set(first)


def test_negative_recolor_and_non_ascii_sigs_are_ignored() -> None:
    mem = FactMemory()
    mem.record(("A6:c4:ab", "self", "any"), (Effect("recolor", color=-1),), 1)
    mem.record(("A3", "c\u0663:ab", "any"), (Effect("disappear"),), 1)
    assert GoalBook().propose(BASE, mem, source="heuristic") == []


# --- stalled goals (Task 8d) ----------------------------------------------------------


def test_stalled_goal_leaves_open_and_reopens() -> None:
    book = GoalBook()
    g = Goal("reach", a=PSIG, b=T1SIG)
    h = Goal("reach", a=PSIG, b=T2SIG)
    book.status[g] = "open"
    book.status[h] = "open"
    assert book.stall(g) and book.status[g] == "stalled"
    assert book.open() == [h]
    assert not book.stall(g)  # already stalled
    assert book.propose(BASE, FactMemory(), source="x") == [] or g not in book.open()
    assert book.reopen_stalled() == [g]
    assert book.status[g] == "open" and g in book.open()


def test_stalled_confirmed_form_is_skipped_until_reopened() -> None:
    book = GoalBook()
    g = Goal("reach", a=PSIG, b=T1SIG)
    book.status[g] = "open"
    book.level_up(ON_T1)
    assert book.open() == [g]
    assert book.stall(g) and book.open() == []
    assert book.reopen_stalled() == [g] and book.open() == [g]


def test_rule_out_checks_only_the_given_goals() -> None:
    book = GoalBook()
    added = book.propose(BASE, mover_memory(), source="heuristic")
    on_t1 = next(g for g in added if g.b == T1SIG)
    on_t2 = next(g for g in added if g.b == T2SIG)
    assert book.rule_out([ON_T1], only=[on_t2]) == []
    assert book.status[on_t1] == "open"  # true on the frame, but not among the checked goals
    assert book.rule_out([ON_T1], only=[on_t1]) == [on_t1]


def test_match_goals_use_the_real_grid_on_every_check() -> None:
    same = np.zeros((64, 64), dtype=np.int8)
    differ = same.copy()
    differ[0, 0] = 3
    match = Goal("match", region_a=(0, 0, 2, 2), region_b=(10, 10, 12, 12))
    for check in ("rule_out", "note_reached", "level_up"):
        book = GoalBook()
        book.status[match] = "open"
        if check == "rule_out":
            assert book.rule_out([BASE, BASE], [differ]) == []  # second frame has no grid
            assert book.rule_out([BASE, BASE], [differ, same]) == [match]
        elif check == "note_reached":
            assert book.note_reached(BASE) == [] and book.note_reached(BASE, differ) == []
            assert book.note_reached(BASE, same) == [match]
        else:
            assert book.level_up(BASE, differ) == [] and book.level_up(BASE, same) == [match]


def test_controlled_sigs_need_different_effects_under_two_labels() -> None:
    mem = FactMemory()
    player, enemy, still = "c1:p", "c7:e", "c3:s"
    mem.record(("A1", player, "any"), (Effect("move", dy=-1),), 0)
    mem.record(("A2", player, "any"), (Effect("move", dy=1),), 0)
    for i in (1, 2, 3, 4):
        mem.record((f"A{i}", enemy, "any"), (Effect("move", dx=1),), 0)
        mem.record((f"A{i}", still, "any"), (Effect("none"),), 0)
    assert controlled_sigs(mem) == frozenset({player})
    mem.record(("A6:c7:e@0", "self", "any"), (Effect("recolor", color=2),), 0)
    assert enemy in controlled_sigs(mem)  # a click changes it differently


def test_level_up_confirms_a_goal_that_holds_only_in_the_predicted_winning_scene() -> None:
    # the frame after a winning move is already the next level, so the goal state is
    # never observed; the agent's prediction of that move's outcome stands in for it
    book = GoalBook()
    book.propose(BASE, mover_memory(), source="heuristic")
    assert book.level_up(BASE) == []
    out = book.level_up(BASE, after=ON_T2)
    assert [g.b for g in out] == [T2SIG] and book.confirmed_forms[0].b == T2SIG
