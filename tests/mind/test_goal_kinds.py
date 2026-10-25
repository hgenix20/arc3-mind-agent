"""Tests for the goal rule-book and the cover and match goal kinds."""

from __future__ import annotations

from typing import get_args

import numpy as np

from arcagent.core.types import COORD_ACTION_ID, Action, Grid
from arcagent.mind.facts import Effect, FactMemory
from arcagent.mind.goal_rules import GOAL_RULES, prior_of
from arcagent.mind.goals import Goal, GoalBook, GoalKind
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.search import search

Cells = dict[tuple[int, int], int]
DOT = shape_of([(0, 0)])
BOX, PAD = f"c3:{DOT}", f"c8:{DOT}"


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(cells: Cells) -> tuple[Scene, Grid]:
    g = grid_with(cells)
    return Tracker().update(g, frozenset())[0], g


def frame(x0: int, y0: int, w: int, h: int, color: int = 5) -> Cells:
    out: Cells = {}
    for x in range(x0, x0 + w):
        out[(x, y0)] = out[(x, y0 + h - 1)] = color
    for y in range(y0, y0 + h):
        out[(x0, y)] = out[(x0 + w - 1, y)] = color
    return out


def test_rule_book_names_only_known_kinds() -> None:
    kinds = set(get_args(GoalKind))
    assert {r.kind for r in GOAL_RULES} <= kinds
    assert prior_of("reach") > 0 and prior_of("cover") > 0 and prior_of("match") > 0


def test_cover_holds_when_every_target_has_its_own_box_beside_or_on_it() -> None:
    goal = Goal("cover", a=BOX, b=PAD)
    near, _ = scene_of({(10, 10): 3, (11, 10): 8, (20, 10): 3, (20, 11): 8})
    assert goal.holds(near) and goal.distance(near) == 0
    one_box, _ = scene_of({(10, 10): 3, (11, 10): 8, (30, 30): 8})  # one box, two pads
    assert not goal.holds(one_box)
    far, _ = scene_of({(10, 10): 3, (15, 10): 8, (20, 10): 3, (20, 11): 8})
    assert not goal.holds(far) and goal.distance(far) == 4  # five cells off, less the contact


def test_cover_needs_both_kinds_on_screen() -> None:
    goal = Goal("cover", a=BOX, b=PAD)
    boxes_only, _ = scene_of({(10, 10): 3})
    assert goal.distance(boxes_only) == 0 and goal.holds(boxes_only)  # every pad is covered
    pads_only, _ = scene_of({(10, 10): 8})
    assert goal.distance(pads_only) is None and not goal.holds(pads_only)


def test_cover_and_match_are_proposed() -> None:
    mem = FactMemory()
    mem.record(("A4", BOX, "any"), (Effect("move", dx=1),), 0)
    cells = {(10, 10): 3, (30, 10): 8, (30, 20): 8}
    cells |= frame(40, 40, 5, 5) | {(41, 41): 6} | frame(50, 40, 5, 5)
    scene, grid = scene_of(cells)
    book = GoalBook()
    added = book.propose(scene, mem, source="t", grid=grid)
    assert Goal("cover", a=BOX, b=PAD) in added
    assert Goal("match", region_a=(41, 41, 44, 44), region_b=(51, 41, 54, 44)) in added
    assert book.propose(scene, mem, source="t", grid=grid) == []  # nothing proposed twice


def test_search_plans_toward_a_match_goal() -> None:
    # The left box shows a color-6 cell; clicking the color-2 cell in the right box turns it 6.
    cells = frame(10, 10, 5, 5) | {(11, 11): 6} | frame(30, 10, 5, 5) | {(31, 11): 2}
    scene, _ = scene_of(cells)
    mem = FactMemory()
    label = f"A6:c2:{DOT}@5"  # the ring around it is mostly the frame's color
    mem.record((label, "self", "any"), (Effect("recolor", color=6),), 0)
    mem.record((label, "any", "any"), (Effect("none"),), 0)
    goal = Goal("match", region_a=(11, 11, 14, 14), region_b=(31, 11, 34, 14))

    def clicks(s: Scene) -> list[Action]:
        return [Action(COORD_ACTION_ID, x, y) for t in s.things for x, y in sorted(t.cells)[:1]]

    r = search(scene, clicks, mem, [goal])
    assert r.stop == "found" and r.goal == goal
    assert [(x.action.x, x.action.y) for x in r.plan] == [(31, 11)]


def ranked_book(scene: Scene) -> tuple[GoalBook, Goal, Goal]:
    book = GoalBook()
    near, far = Goal("reach", a=BOX, b=PAD), Goal("clear", color=9)
    for g in (far, near):
        book.status[g] = "open"
    return book, near, far


def test_ranked_puts_cheap_salient_goals_first() -> None:
    scene, _ = scene_of({(10, 10): 3, (13, 10): 8, (40, 40): 9, (44, 40): 9, (48, 40): 9})
    book, near, far = ranked_book(scene)
    assert book.ranked(scene) == [near, far]  # distance 2 beats three things left
    assert book.ranked(scene, salient=frozenset({f"c9:{DOT}"}))[0] == far  # salience doubles it
    assert book.open() == [far, near]  # the open list keeps insertion order


def test_confirmed_form_stays_first_and_search_gets_every_open_goal() -> None:
    scene, _ = scene_of({(10, 10): 3, (13, 10): 8})
    book = GoalBook()
    form = Goal("clear", color=7)
    book.confirmed_forms.append(form)
    for c in range(20):
        book.status[Goal("count", color=c, n=1)] = "open"
    ranked = book.ranked(scene)
    assert ranked[0] == form and len(ranked) == 21
