"""Facts and predictions that read the mode: clicks relative to a selection, the active thing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from arcagent.core.types import COORD_ACTION_ID, Action, Grid
from arcagent.mind.facts import Effect, FactMemory, learn
from arcagent.mind.modes import selection_contexts
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.predictor import predict

Cells = dict[tuple[int, int], int]
DOT = shape_of([(0, 0)])
PEG, HOLE = f"c2:{DOT}", f"c3:{DOT}"


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def row(x: int, y: int) -> Cells:
    """Peg, peg, hole two cells apart, starting at (x, y)."""
    return {(x, y): 2, (x + 2, y): 2, (x + 4, y): 3}


def jumped(x: int, y: int) -> Cells:
    """The first peg jumped over the second into the hole; both start cells are holes."""
    return {(x, y): 3, (x + 2, y): 3, (x + 4, y): 2}


def at(scene: Scene, x: int, y: int) -> int:
    return next(t.id for t in scene.things if (x, y) in t.cells)


def click(x: int, y: int) -> Action:
    return Action(COORD_ACTION_ID, x, y)


def selected_jump(mem: FactMemory, x: int, y: int) -> None:
    t = Tracker()
    s0, _ = t.update(grid_with(row(x, y)), frozenset())
    s0 = replace(s0, selected=frozenset({at(s0, x, y)}))
    s1, d = t.update(grid_with(jumped(x, y)), frozenset())
    assert d is not None
    learn(mem, s0, click(x + 4, y), d, move=0, level_up=False, game_over=False, after=s1)


def test_contexts_name_the_selected_the_between_and_the_clicked() -> None:
    s = Tracker().update(grid_with(row(10, 10)), frozenset())[0]
    s = replace(s, selected=frozenset({at(s, 10, 10)}))
    ctx = selection_contexts(s, click(14, 10))
    assert ctx == {
        at(s, 10, 10): "selected:4,0",
        at(s, 12, 10): "between:4,0",
        at(s, 14, 10): "sel:4,0",
    }
    assert selection_contexts(replace(s, selected=frozenset()), click(14, 10)) == {}


def test_jump_two_away_is_learned_relative_to_selection_and_predicted_elsewhere() -> None:
    # Identical pegs make the tracker's reading ambiguous (a jump, or a recolor plus two
    # slides); either way the facts sit under the selection contexts and predict the same
    # picture on another row.
    mem = FactMemory()
    selected_jump(mem, 10, 10)
    contexts = {k[2] for k in mem.facts}
    assert {"selected:4,0", "between:4,0", "sel:4,0"} <= contexts
    s = Tracker().update(grid_with(row(30, 40)), frozenset())[0]
    s = replace(s, selected=frozenset({at(s, 30, 40)}))
    pred = predict(s, click(34, 40), mem)
    assert pred.scene is not None and not pred.unknown
    picture = sorted((t.color, t.x, t.y) for t in pred.scene.things)
    assert picture == [(2, 34, 40), (3, 30, 40), (3, 32, 40)]


def test_same_click_without_selection_is_a_different_fact() -> None:
    mem = FactMemory()
    selected_jump(mem, 10, 10)
    t = Tracker()
    s0, _ = t.update(grid_with(row(30, 40)), frozenset())
    s1, d = t.update(grid_with(row(30, 40)), frozenset())  # nothing selected: nothing happens
    assert d is not None
    keys = learn(mem, s0, click(34, 40), d, move=1, level_up=False, game_over=False, after=s1)
    assert all(not k[2].startswith(("selected", "between", "sel:")) for k in keys)
    assert mem.get((keys[0][0], PEG, "selected:4,0")) is not None  # the jump fact is untouched


def test_click_on_a_selectable_thing_predicts_it_selected() -> None:
    mem = FactMemory()
    mem.selectable.add(PEG)
    s = Tracker().update(grid_with(row(10, 10)), frozenset())[0]
    for k in (
        (f"A6:{PEG}@0", "self", "any"),
        (f"A6:{PEG}@0", HOLE, "any"),
        (f"A6:{PEG}@0", PEG, "any"),
    ):
        mem.record(k, (Effect("none"),), 0)
    pred = predict(s, click(12, 10), mem)
    assert pred.scene is not None and pred.scene.selected == frozenset({at(s, 12, 10)})
    assert pred.scene.key() != s.key()  # the selection is part of the state


def test_direction_moves_only_the_active_thing() -> None:
    mem = FactMemory()
    mem.record(("A4", PEG, "active"), (Effect("move", dx=1),), 0)
    mem.record(("A4", PEG, "any"), (Effect("none"),), 0)
    mem.record(("A4", HOLE, "any"), (Effect("none"),), 0)
    s = Tracker().update(grid_with(row(10, 10)), frozenset())[0]
    s = replace(s, active=at(s, 12, 10))
    pred = predict(s, Action(4), mem)
    assert pred.scene is not None
    assert sorted(t.x for t in pred.scene.things if t.sig == PEG) == [10, 13]


def test_learning_with_an_active_thing_files_its_move_under_active() -> None:
    mem = FactMemory()
    t = Tracker()
    s0, _ = t.update(grid_with({(10, 10): 2, (20, 10): 2}), frozenset())
    s0 = replace(s0, active=at(s0, 10, 10))
    s1, d = t.update(grid_with({(11, 10): 2, (20, 10): 2}), frozenset())
    assert d is not None
    learn(mem, s0, Action(4), d, move=0, level_up=False, game_over=False, after=s1)
    moved, still = mem.get(("A4", PEG, "active")), mem.get(("A4", PEG, "any"))
    assert moved is not None and moved.effects == (Effect("move", dx=1),)
    assert still is not None and still.effects == (Effect("none"),)
