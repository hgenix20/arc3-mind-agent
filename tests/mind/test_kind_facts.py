"""Facts, predictions, goals and search through kinds: one sprite, two looks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.facts import Effect, FactMemory, learn
from arcagent.mind.goals import Goal, mover_sigs
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.predictor import predict
from arcagent.mind.search import search

Cells = dict[tuple[int, int], int]

FACE_R = [(0, 0), (1, 0), (0, 1)]
FACE_L = [(0, 0), (1, 0), (1, 1)]
RSIG, LSIG = f"c9:{shape_of(FACE_R)}", f"c9:{shape_of(FACE_L)}"
KIND = "k" + RSIG
TSIG = f"c4:{shape_of([(0, 0)])}"


def sprite(x: int, y: int, face: list[tuple[int, int]]) -> Cells:
    return {(x + dx, y + dy): 9 for dx, dy in face}


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def kinded(scene: Scene) -> Scene:
    """The color-9 sprite labeled with KIND, as the agent's KindBook would."""
    things = tuple(replace(t, kind=KIND) if t.color == 9 else t for t in scene.things)
    return Scene(things, scene.background, scene.blocked, scene.terrain)


def step(t: Tracker, cells: Cells) -> tuple[Scene, object]:
    scene, diff = t.update(grid_with(cells), frozenset())
    return kinded(scene), diff


def test_move_fact_is_also_learned_under_the_kind_with_the_new_look() -> None:
    mem = FactMemory()
    t = Tracker()
    s0, _ = step(t, sprite(10, 10, FACE_R))
    s1, d1 = step(t, sprite(8, 10, FACE_L))  # A3: two cells left, turned to face left
    assert d1 is not None
    learn(mem, s0, Action(3), d1, move=0, level_up=False, game_over=False, after=s1)  # pyright: ignore[reportArgumentType]
    s2, d2 = step(t, sprite(6, 10, FACE_L))  # A3 again: already facing left
    learn(mem, s1, Action(3), d2, move=1, level_up=False, game_over=False, after=s2)  # pyright: ignore[reportArgumentType]
    kind_fact = mem.get(("A3", KIND, "any"))
    assert kind_fact is not None
    assert kind_fact.effects == (Effect("move", dx=-2), Effect("reshape", sig=LSIG))
    assert kind_fact.hits == 2 and kind_fact.status != "broken"
    assert mem.get(("A3", LSIG, "any")) is not None  # look facts are kept as before


def kind_memory() -> FactMemory:
    mem = FactMemory()
    mem.record(("A4", KIND, "any"), (Effect("move", dx=1), Effect("reshape", sig=RSIG)), 0)
    mem.record(("A3", KIND, "any"), (Effect("move", dx=-1), Effect("reshape", sig=LSIG)), 0)
    for i in (3, 4):
        mem.record((f"A{i}", TSIG, "any"), (Effect("none"),), 0)
    return mem


def test_kind_fact_predicts_a_look_never_seen_with_that_action() -> None:
    scene = kinded(Tracker().update(grid_with(sprite(10, 10, FACE_L)), frozenset())[0])
    pred = predict(scene, Action(4), kind_memory())
    assert pred.scene is not None and not pred.unknown
    moved = pred.scene.things[0]
    assert (moved.x, moved.sig, moved.kind) == (11, RSIG, KIND)


def test_look_fact_beats_the_kind_fact_until_it_breaks() -> None:
    scene = kinded(Tracker().update(grid_with(sprite(10, 10, FACE_L)), frozenset())[0])
    mem = kind_memory()
    mem.record(("A4", LSIG, "any"), (Effect("none"),), 0)
    pred = predict(scene, Action(4), mem)
    assert pred.scene is not None and pred.scene.things[0].x == 10
    for move in (1, 2):  # two misses in a row break the look fact
        mem.record(("A4", LSIG, "any"), (Effect("move", dx=3),), move)
    pred = predict(scene, Action(4), mem)
    assert pred.scene is not None and pred.scene.things[0].x == 11


def test_search_chains_moves_across_looks() -> None:
    cells = sprite(10, 10, FACE_L) | {(15, 10): 4}
    scene = kinded(Tracker().update(grid_with(cells), frozenset())[0])

    def lr(_s: Scene) -> list[Action]:
        return [Action(3), Action(4)]

    r = search(scene, lr, kind_memory(), [Goal("reach", a=KIND, b=TSIG)])
    assert r.stop == "found" and [x.action.id for x in r.plan] == [4, 4, 4]


def test_goal_on_a_kind_holds_whatever_the_look() -> None:
    goal = Goal("reach", a=KIND, b=TSIG)
    for face in (FACE_R, FACE_L):
        scene = kinded(
            Tracker().update(grid_with(sprite(10, 10, face) | {(12, 10): 4}), frozenset())[0]
        )
        assert goal.holds(scene) and goal.distance(scene) == 0


def test_kinds_are_movers() -> None:
    assert KIND in mover_sigs(kind_memory())
