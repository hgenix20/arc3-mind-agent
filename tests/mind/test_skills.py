"""Tests for the strategy skills (spec 4.9d): S-AROUND, S-COLLECT, S-EXPLORE, the names
kept for later versions, and ``Kinematics.reachable_avoiding``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.facts import Effect, FactMemory
from arcagent.mind.goals import Goal
from arcagent.mind.kinematics import Kinematics
from arcagent.mind.objects import Cell, Scene, Thing, Tracker, click_cell, shape_of
from arcagent.mind.skills import (
    NOT_BUILT,
    SKILL_NAMES,
    UNKNOWN_SKILL,
    Flag,
    Role,
    SkillContext,
    run_skill,
)

Cells = dict[Cell, int]
DOT = shape_of([(0, 0)])
PSIG, TSIG = f"c1:{DOT}", f"c4:{DOT}"
PLAYER, TARGET, FRUIT, HARM, WALL, OTHER = 1, 4, 3, 7, 5, 2
STEPS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
REACH = Goal("reach", a=PSIG, b=TSIG)
BASE: Cells = {(10, 10): PLAYER, (20, 10): TARGET}
HARM_BAR = {(15, 9), (15, 10), (15, 11)}


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(cells: Cells) -> Scene:
    return Tracker().update(grid_with(cells), frozenset())[0]


def walker(slide_right: bool = False) -> FactMemory:
    """A one-cell player that steps one cell per direction (A4 slides when asked)."""
    mem = FactMemory()
    for i, (dx, dy) in STEPS.items():
        effect = Effect("slide", dx=1) if i == 4 and slide_right else Effect("move", dx=dx, dy=dy)
        mem.record((f"A{i}", PSIG, "any"), (effect,), 0)
    mem.blockers[PSIG] = {WALL}
    return mem


def wall_column(x: int, gap: int | None = None) -> Cells:
    return {(x, y): WALL for y in range(64) if y != gap}


def bar(x: int, ys: Sequence[int], color: int) -> Cells:
    return {(x, y): color for y in ys}


def thing_at(scene: Scene, cell: Cell) -> Thing:
    hits = [t for t in scene.things if cell in t.cells]
    assert len(hits) == 1, cell
    return hits[0]


def roles_for(scene: Scene, by_color: Mapping[int, Role]) -> dict[str, Role]:
    return {t.kind: by_color[t.color] for t in scene.things if t.color in by_color}


def context(
    scene: Scene,
    mem: FactMemory,
    *,
    goals: Sequence[Goal] = (),
    roles: Mapping[str, Role] | None = None,
    flags: Sequence[Flag] = (),
    can_click: bool = False,
    mover: str | None = PSIG,
    max_steps: int = 50,
) -> SkillContext:
    return SkillContext(
        scene=scene,
        mem=mem,
        kin=Kinematics(mem, mover),
        goals=tuple(goals),
        roles=roles or {},
        flags=tuple(flags),
        can_click=can_click,
        mover=mover,
        click_cell=click_cell,
        max_steps=max_steps,
    )


def replay(cells: Cells, mem: FactMemory, start: Cell, actions: Sequence[Action]) -> list[Cell]:
    """Every cell the one-cell walker occupies after each action, a slide cell by cell."""
    gaits = {g.action: g for g in Kinematics(mem).gaits(PSIG)}
    pos = start
    out: list[Cell] = []
    for a in actions:
        g = gaits[f"A{a.id}"]
        if not g.slides:
            pos = (pos[0] + g.dx, pos[1] + g.dy)
            out.append(pos)
            continue
        for _ in range(64):
            nxt = (pos[0] + g.dx, pos[1] + g.dy)
            if not (0 <= nxt[0] < 64 and 0 <= nxt[1] < 64) or cells.get(nxt) == WALL:
                break
            pos = nxt
            out.append(pos)
    return out


def touches(pos: Cell, cells: set[Cell]) -> bool:
    x, y = pos
    return any(c in cells for c in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))


def test_around_walks_around_a_harm_thing() -> None:
    cells = BASE | bar(15, (9, 10, 11), HARM)
    s = scene_of(cells)
    ctx = context(s, walker(), goals=[REACH], roles=roles_for(s, {HARM: "harm"}))
    res = run_skill("S-AROUND", {}, ctx)
    assert res.applies, res.why
    path = replay(cells, walker(), (10, 10), res.actions)
    assert len(res.actions) == 13 and touches(path[-1], {(20, 10)})
    assert HARM_BAR.isdisjoint(path)
    # Without the role the straight line through the harm cells is the shortest path.
    plain = run_skill("S-AROUND", {}, context(s, walker(), goals=[REACH]))
    assert plain.applies and len(plain.actions) == 9
    assert not HARM_BAR.isdisjoint(replay(cells, walker(), (10, 10), plain.actions))


def test_around_finds_no_path_when_harm_blocks_the_only_gap() -> None:
    cells = BASE | wall_column(15, gap=30) | {(15, 30): HARM}
    s = scene_of(cells)
    open_way = run_skill("S-AROUND", {}, context(s, walker(), goals=[REACH]))
    assert open_way.applies and len(open_way.actions) > 9
    shut = run_skill(
        "S-AROUND", {}, context(s, walker(), goals=[REACH], roles=roles_for(s, {HARM: "harm"}))
    )
    assert not shut.applies and shut.actions == () and "no path" in shut.why


def test_around_takes_the_target_and_the_avoid_cells_from_args() -> None:
    cells = BASE | {(10, 20): OTHER}
    s = scene_of(cells)
    other = thing_at(s, (10, 20))
    ctx = context(s, walker(), goals=[REACH])
    straight = run_skill("S-AROUND", {"target": other.id}, ctx)
    assert straight.applies and len(straight.actions) == 9
    assert touches(replay(cells, walker(), (10, 10), straight.actions)[-1], {(10, 20)})
    bent = run_skill("S-AROUND", {"target": other.id, "avoid_cells": [[10, 15]]}, ctx)
    assert bent.applies and len(bent.actions) == 11
    path = replay(cells, walker(), (10, 10), bent.actions)
    assert (10, 15) not in path and touches(path[-1], {(10, 20)})
    # A target id that is not on screen falls back to the goal's target.
    fallback = run_skill("S-AROUND", {"target": 999, "avoid_cells": "junk"}, ctx)
    assert fallback.applies
    assert touches(replay(cells, walker(), (10, 10), fallback.actions)[-1], {(20, 10)})


def test_around_needs_a_mover_a_target_and_distance_to_cover() -> None:
    s = scene_of(BASE)
    no_mover = run_skill("S-AROUND", {}, context(s, walker(), goals=[REACH], mover=None))
    assert (no_mover.applies, no_mover.actions, no_mover.why) == (False, (), "no mover")
    no_goal = run_skill("S-AROUND", {}, context(s, walker()))
    assert not no_goal.applies and no_goal.why.startswith("no target")
    gaitless = run_skill("S-AROUND", {}, context(s, FactMemory(), goals=[REACH]))
    assert not gaitless.applies and "gaits" in gaitless.why
    close = scene_of({(19, 10): PLAYER, (20, 10): TARGET})
    touching = run_skill("S-AROUND", {}, context(close, walker(), goals=[REACH]))
    assert not touching.applies and "already touching" in touching.why


def collect_ctx(cells: Cells, *, harm: bool = False) -> SkillContext:
    s = scene_of(cells)
    by_color: dict[int, Role] = {FRUIT: "collectible"}
    if harm:
        by_color[HARM] = "harm"
    return context(s, walker(), roles=roles_for(s, by_color))


def test_collect_walks_onto_the_nearer_of_two_collectibles() -> None:
    cells = {(10, 10): PLAYER, (14, 10): FRUIT, (20, 10): FRUIT}
    res = run_skill("S-COLLECT", {}, collect_ctx(cells))
    assert res.applies, res.why
    assert res.actions == (Action(4),) * 4  # onto (14, 10), the collectible itself
    assert replay(cells, walker(), (10, 10), res.actions)[-1] == (14, 10)


def test_collect_follows_the_route_order_over_the_nearest_target() -> None:
    # Nearest first (right, 2 moves) makes a 20-move tour; left first makes 16, so the
    # route starts with the farther collectible on the left.
    cells = {(10, 10): PLAYER, (7, 10): FRUIT, (12, 10): FRUIT, (20, 10): FRUIT}
    res = run_skill("S-COLLECT", {}, collect_ctx(cells))
    assert res.applies and res.actions == (Action(3),) * 3


def test_collect_skips_a_collectible_it_cannot_reach() -> None:
    box = {(13, 10): WALL, (15, 10): WALL, (14, 9): WALL, (14, 11): WALL}
    cells = {(10, 10): PLAYER, (14, 10): FRUIT, (20, 10): FRUIT} | box
    res = run_skill("S-COLLECT", {}, collect_ctx(cells))
    assert res.applies and replay(cells, walker(), (10, 10), res.actions)[-1] == (20, 10)


def test_collect_avoids_harm_and_needs_a_collectible() -> None:
    cells = {(10, 10): PLAYER, (20, 10): FRUIT} | bar(15, (9, 10, 11), HARM)
    res = run_skill("S-COLLECT", {}, collect_ctx(cells, harm=True))
    assert res.applies, res.why
    path = replay(cells, walker(), (10, 10), res.actions)
    assert len(res.actions) == 14 and path[-1] == (20, 10) and HARM_BAR.isdisjoint(path)
    none = run_skill("S-COLLECT", {}, context(scene_of(cells), walker()))
    assert (none.applies, none.actions, none.why) == (False, (), "no collectibles on screen")


def test_explore_clicks_when_it_can_and_walks_when_it_cannot() -> None:
    s = scene_of(BASE)
    flags: list[Flag] = [("c9:nothing", 5.0, 1.0), (TSIG, 1.0, 2.0)]
    clicked = run_skill("S-EXPLORE", {}, context(s, walker(), flags=flags, can_click=True))
    assert clicked.applies and clicked.actions == (Action(6, 20, 10),)
    walked = run_skill("S-EXPLORE", {}, context(s, walker(), flags=flags))
    assert walked.applies and len(walked.actions) == 9
    assert touches(replay(BASE, walker(), (10, 10), walked.actions)[-1], {(20, 10)})
    gone = run_skill("S-EXPLORE", {}, context(s, walker(), flags=flags[:1]))
    assert not gone.applies and gone.actions == ()
    assert not run_skill("S-EXPLORE", {}, context(s, walker())).applies


def test_unknown_and_unbuilt_skills_do_not_apply() -> None:
    ctx = context(scene_of(BASE), walker(), goals=[REACH])
    for name in ("S-ATTACK", "S-WAIT"):
        res = run_skill(name, {}, ctx)
        assert (res.applies, res.actions, res.why) == (False, (), NOT_BUILT)
    unknown = run_skill("S-FLY", {}, ctx)
    assert (unknown.applies, unknown.actions, unknown.why) == (False, (), UNKNOWN_SKILL)
    assert set(SKILL_NAMES) == {"S-AROUND", "S-COLLECT", "S-EXPLORE", "S-ATTACK", "S-WAIT"}


def test_max_steps_cuts_the_segment() -> None:
    s = scene_of(BASE)
    full = run_skill("S-AROUND", {}, context(s, walker(), goals=[REACH]))
    cut = run_skill("S-AROUND", {}, context(s, walker(), goals=[REACH], max_steps=3))
    assert cut.applies and cut.actions == full.actions[:3] and "cut to 3" in cut.why
    none = run_skill("S-AROUND", {}, context(s, walker(), goals=[REACH], max_steps=0))
    assert not none.applies and none.actions == ()


def chain(
    reach: Mapping[Cell, tuple[int, Cell | None, Action | None]], tl: Cell
) -> tuple[Cell, int]:
    """Follow the previous top-lefts from ``tl``: the cell they end at and the steps taken."""
    here, steps = tl, 0
    for _ in range(len(reach)):
        prev = reach[here][1]
        if prev is None:
            break
        here, steps = prev, steps + 1
    return here, steps


def test_reachable_avoiding_matches_reachable_with_nothing_to_avoid() -> None:
    s = scene_of({(10, 10): PLAYER} | wall_column(20) | wall_column(5, gap=12))
    kin = Kinematics(walker(slide_right=True))
    avoiding = kin.reachable_avoiding(s, PSIG, frozenset())
    assert {tl: m for tl, (m, _, _) in avoiding.items()} == kin.reachable(s, PSIG)
    assert avoiding[(10, 10)] == (0, None, None)
    assert avoiding[(19, 10)] == (1, (10, 10), Action(4))
    for tl, (m, _, _) in avoiding.items():
        assert chain(avoiding, tl) == ((10, 10), m)
    assert len(kin.reachable_avoiding(s, PSIG, frozenset(), limit=50)) <= 50


def test_reachable_avoiding_refuses_a_slide_through_an_avoided_cell() -> None:
    s = scene_of({(10, 10): PLAYER} | wall_column(20))
    kin = Kinematics(walker(slide_right=True))
    reach = kin.reachable_avoiding(s, PSIG, frozenset({(15, 10)}))
    assert (15, 10) not in reach
    assert reach[(19, 10)][0] == 3  # down, slide along the next row, up
    assert kin.reachable_avoiding(s, PSIG, frozenset())[(19, 10)][0] == 1
