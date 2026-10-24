"""Tests for the short search over predicted scenes."""

from __future__ import annotations

import itertools

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.facts import SPAWN, Effect, FactMemory, learn
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.search import SEARCH_MAX_NODES, SEARCH_TIME_S, search

Cells = dict[tuple[int, int], int]

DOT = shape_of([(0, 0)])
PSIG = f"c1:{DOT}"
TSIG = f"c4:{DOT}"
LSIG = f"c9:{DOT}"
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(cells: Cells) -> Scene:
    return Tracker().update(grid_with(cells), frozenset())[0]


def simple_actions(_scene: Scene) -> list[Action]:
    return [Action(i) for i in (1, 2, 3, 4)]


def moves(ids: tuple[int, ...]) -> FactMemory:
    mem = FactMemory()
    for i in ids:
        dx, dy = DIRS[i]
        mem.record((f"A{i}", PSIG, "any"), (Effect("move", dx=dx, dy=dy),), 0)
        # Without this, the same-shape target would borrow the player's move fact.
        mem.record((f"A{i}", TSIG, "any"), (Effect("none"),), 0)
    return mem


def reach(a: str, b: str) -> Goal:
    return Goal("reach", a=a, b=b)


def test_finds_shortest_plan_to_reach() -> None:
    s = scene_of({(2, 2): 1, (6, 2): 4})
    r = search(s, simple_actions, moves((1, 2, 3, 4)), [reach(PSIG, TSIG)])
    assert r.stop == "found" and [x.action.id for x in r.plan] == [4, 4, 4]
    assert r.depth == 3 and r.goal == reach(PSIG, TSIG)


def test_root_goal_returns_empty_plan() -> None:
    s = scene_of({(2, 2): 1, (3, 2): 4})
    r = search(s, simple_actions, moves((4,)), [reach(PSIG, TSIG)])
    assert r.stop == "found" and r.plan == () and r.nodes == 0


def test_avoids_predicted_game_over_route() -> None:
    mem = moves((1, 2, 3, 4))
    for i in DIRS:
        mem.record((f"A{i}", LSIG, "any"), (Effect("none"),), 0)
    mem.record(("any", PSIG, "touch:9"), (Effect("game_over"),), 0)
    s = scene_of({(2, 2): 1, (3, 2): 9, (6, 2): 4})
    r = search(s, simple_actions, mem, [reach(PSIG, TSIG)])
    assert r.stop == "found" and r.plan
    assert all(not x.expect.game_over for x in r.plan)
    assert r.plan[0].action.id != 4


def test_unknown_actions_end_branches_and_become_frontier() -> None:
    s = scene_of({(2, 2): 1, (0, 2): 4})
    mem = FactMemory()
    mem.record(("A4", PSIG, "any"), (Effect("move", dx=1),), 0)
    mem.record(("A4", TSIG, "any"), (Effect("none"),), 0)
    r = search(s, simple_actions, mem, [reach(PSIG, TSIG)])
    assert r.plan == () and r.goal is None and ("A3", PSIG) in r.frontier
    assert len(set(r.frontier)) == len(r.frontier)


def test_respects_time_cap_on_large_open_board() -> None:
    ticks = itertools.count()
    r = search(
        scene_of({(30, 30): 1, (0, 0): 4}),
        simple_actions,
        moves((1, 2, 3, 4)),
        [reach(PSIG, f"c7:{DOT}")],
        time_s=0.25,
        clock=lambda: next(ticks) * 0.01,
    )
    assert r.stop == "time" and r.goal is None


def test_node_cap_and_exhaustion() -> None:
    s = scene_of({(30, 30): 1, (0, 0): 4})
    g = [reach(PSIG, f"c7:{DOT}")]
    assert search(s, simple_actions, moves((1, 2, 3, 4)), g, max_nodes=50).stop == "nodes"
    assert search(s, simple_actions, FactMemory(), g).stop == "exhausted"


def test_node_count_is_the_default_bound() -> None:
    # A frozen clock: only the node cap can stop the search on an open board.
    assert (SEARCH_MAX_NODES, SEARCH_TIME_S) == (3000, 1.0)
    r = search(
        scene_of({(30, 30): 1, (0, 0): 4}),
        simple_actions,
        moves((1, 2, 3, 4)),
        [reach(PSIG, f"c7:{DOT}")],
        clock=lambda: 0.0,
    )
    assert r.stop == "nodes" and r.nodes == SEARCH_MAX_NODES


def test_match_goals_skipped_and_no_change_not_expanded() -> None:
    s = scene_of({(0, 0): 1, (5, 5): 4})
    g = Goal("match", region_a=(0, 0, 1, 1), region_b=(0, 0, 1, 1))
    r = search(s, lambda _s: [Action(3)], moves((3,)), [g])
    assert r.stop == "exhausted" and r.nodes == 1 and r.goal is None


def corridor(cells: Cells) -> Grid:
    """Background 4 (the walls); a 48x4 floor of color 3 on rows 10..13 (terrain)."""
    g = np.full((64, 64), 4, dtype=np.int8)
    g[10:14, 0:48] = 3
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def test_search_crosses_a_terrain_floor_end_to_end() -> None:
    target = (6, 11)
    path = [(0, (2, 11)), (4, (3, 11)), (3, (2, 11)), (1, (2, 10)), (2, (2, 11)), (1, (2, 10))]
    path += [(1, (2, 10)), (2, (2, 11))]
    mem = FactMemory()
    t = Tracker()
    scene, _ = t.update(corridor({path[0][1]: 1, target: 6}), frozenset())
    for move, (action, cell) in enumerate(path[1:]):
        after, d = t.update(corridor({cell: 1, target: 6}), frozenset())
        assert d is not None
        learn(
            mem, scene, Action(action), d, move=move, level_up=False, game_over=False, after=after
        )
        scene = after
    assert sorted(t.color for t in scene.things) == [1, 6]
    goal = reach(PSIG, f"c6:{DOT}")
    r = search(scene, simple_actions, mem, [goal])
    assert r.stop == "found" and [x.action.id for x in r.plan] == [4, 4, 4]


def test_search_assumes_side_objects_with_broken_facts_stay_still() -> None:
    side = f"c7:{DOT}"
    s = scene_of({(2, 2): 1, (6, 2): 4, (30, 30): 7})
    mem = moves((1, 2, 3, 4))
    for i in (1, 2, 3, 4):
        key = (f"A{i}", side, "any")
        for move, color in enumerate((5, 6, 7)):
            mem.record(key, (Effect("recolor", color=color),), move)
        f = mem.get(key)
        assert f is not None and f.status == "broken"
    r = search(s, simple_actions, mem, [reach(PSIG, TSIG)])
    assert r.stop == "found" and [x.action.id for x in r.plan] == [4, 4, 4]
    assert r.frontier == ()
    assert r.plan[0].expect.assumed == (("A4", side),)


def test_search_still_stops_on_an_unknown_mover() -> None:
    s = scene_of({(2, 2): 1, (6, 2): 4, (30, 30): 7})
    mem = moves((1, 2, 3))
    r = search(s, simple_actions, mem, [reach(PSIG, TSIG)])
    assert r.goal is None and ("A4", PSIG) in r.frontier


def test_side_objects_that_change_do_not_multiply_states() -> None:
    side = f"c5:{DOT}"
    plain = scene_of({(2, 2): 1, (40, 2): 4})
    busy = scene_of({(2, 2): 1, (40, 2): 4, (30, 30): 5})
    mem = moves((1, 2, 3, 4))
    for i in (1, 2, 3, 4):  # every action flips the side object 5 <-> 6
        mem.record((f"A{i}", side, "any"), (Effect("recolor", color=6),), 0)
        mem.record((f"A{i}", f"c6:{DOT}", "any"), (Effect("recolor", color=5),), 0)
    goal = [reach(PSIG, TSIG)]
    a = search(plain, simple_actions, mem, goal, time_s=60, max_nodes=20000)
    b = search(busy, simple_actions, mem, goal, time_s=60, max_nodes=20000)
    assert a.stop == b.stop == "found" and len(a.plan) == len(b.plan) == 37
    assert b.nodes == a.nodes


def test_search_never_predicts_spawns() -> None:
    s = scene_of({(2, 2): 1, (6, 2): 4})
    mem = moves((1, 2, 3, 4))
    for move in range(3):
        mem.record(("A4", SPAWN, "any"), (Effect("appear", sig=TSIG, at=(3, 3)),), move)
    r = search(s, simple_actions, mem, [reach(PSIG, TSIG)])
    assert r.stop == "found" and len(r.plan) == 3
    assert all(("A4", SPAWN, "any") not in step.expect.used for step in r.plan)


def test_goal_test_finds_the_nearest_novel_cell() -> None:
    s = scene_of({(2, 2): 1, (40, 40): 4})
    seen = {(2, 2), (2, 1), (2, 3), (1, 2), (3, 2)}

    def novel(scene: Scene) -> bool:
        return any(t.sig == PSIG and (t.x, t.y) not in seen for t in scene.things)

    r = search(s, simple_actions, moves((1, 2, 3, 4)), [], goal_test=novel)
    assert r.stop == "found" and r.goal is None
    assert [x.action.id for x in r.plan] == [1, 1]  # breadth first, actions in order
    assert r.plan[-1].expect.scene is not None
    end = next(t for t in r.plan[-1].expect.scene.things if t.sig == PSIG)
    assert (end.x, end.y) == (2, 0)


def test_goal_test_at_the_root_returns_an_empty_plan() -> None:
    s = scene_of({(2, 2): 1})
    r = search(s, simple_actions, moves((1,)), [], goal_test=lambda _s: True)
    assert r.stop == "found" and r.plan == () and r.goal is None
