"""Tests for the targeted test chooser and the wisdom vetoes."""

from __future__ import annotations

import time

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.facts import Effect, FactMemory, action_label
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.tester import choose_test, veto

Cells = dict[tuple[int, int], int]

DOT = shape_of([(0, 0)])
PSIG = f"c1:{DOT}"
SIG_A = f"c2:{DOT}"
SIG_B = f"c3:{DOT}"
WSIG = f"c5:{DOT}"
LSIG = f"c9:{DOT}"
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(cells: Cells) -> Scene:
    return Tracker().update(grid_with(cells), frozenset())[0]


def know(mem: FactMemory, label: str, sig: str, effect: Effect, times: int = 5) -> None:
    for i in range(times):
        mem.record((label, sig, "any"), (effect,), i)


def mem_moves(ids: tuple[int, ...], times: int = 5) -> FactMemory:
    mem = FactMemory()
    for i in ids:
        dx, dy = DIRS[i]
        know(mem, f"A{i}", PSIG, Effect("move", dx=dx, dy=dy), times)
    return mem


PLAYER = scene_of({(10, 10): 1})
ACTS = [Action(i) for i in (1, 2, 3, 4)]


def test_untried_simple_action_beats_known_ones() -> None:
    t = choose_test(PLAYER, ACTS, mem_moves((1, 2)), [], set(), None)
    assert t is not None and t.action.id == 3 and t.settles == 1
    assert "A3" in t.question


def test_click_game_tests_each_untried_clicked_sig_once() -> None:
    s = scene_of({(5, 5): 2, (20, 20): 3})
    clicks = [Action(6, 5, 5), Action(6, 20, 20)]
    mem = FactMemory()
    know(mem, f"A6:{SIG_A}@0", "self", Effect("none"))
    t = choose_test(s, clicks, mem, [], set(), None)
    assert t is not None and action_label(t.action, s) == f"A6:{SIG_B}@0"


def test_never_repeats_a_settled_test() -> None:
    tried = {(PLAYER.key(), Action(3).key())}
    assert choose_test(PLAYER, [Action(3)], mem_moves((1, 2)), [], tried, None) is None


def test_nothing_open_returns_none() -> None:
    assert choose_test(PLAYER, ACTS, mem_moves((1, 2, 3, 4)), [], set(), None) is None


def test_most_unknown_subjects_first() -> None:
    s = scene_of({(10, 10): 1, (30, 30): 3})
    mem = FactMemory()
    know(mem, "A1", PSIG, Effect("move", dy=-1))
    t = choose_test(s, [Action(1), Action(2)], mem, [], set(), None)
    assert t is not None and t.action.id == 2 and t.settles == 2


def test_thin_fact_ranks_above_solid_on_a_tie() -> None:
    s = scene_of({(10, 10): 1, (30, 30): 3, (31, 30): 3})
    mem = FactMemory()
    know(mem, "A1", PSIG, Effect("move", dy=-1), times=1)
    know(mem, "A2", PSIG, Effect("move", dy=1), times=5)
    t = choose_test(s, [Action(2), Action(1)], mem, [], set(), None)
    assert t is not None and t.action.id == 1


def test_frontier_then_toward_then_lowest_key() -> None:
    mem = FactMemory()
    front = [("A4", PSIG)]
    t = choose_test(PLAYER, ACTS, mem, front, set(), Action(2))
    assert t is not None and t.action.id == 4
    t = choose_test(PLAYER, ACTS, mem, [], set(), Action(2))
    assert t is not None and t.action.id == 2
    t = choose_test(PLAYER, ACTS, mem, [], set(), None)
    assert t is not None and t.action.id == 1


def test_tried_action_skipped_for_the_next() -> None:
    tried = {(PLAYER.key(), Action(1).key())}
    t = choose_test(PLAYER, ACTS, FactMemory(), [], tried, None)
    assert t is not None and t.action.id == 2


def test_predicted_game_over_is_not_chosen() -> None:
    s = scene_of({(10, 10): 1, (11, 10): 9})
    mem = FactMemory()
    know(mem, "A4", PSIG, Effect("move", dx=1))
    mem.record(("any", PSIG, "touch:9"), (Effect("game_over"),), 0)
    t = choose_test(s, [Action(4)], mem, [], set(), None)
    assert t is None or t.action.id != 4


def test_veto_reasons() -> None:
    lava = scene_of({(10, 10): 1, (11, 10): 9})
    mem = mem_moves((4,))
    know(mem, "A4", LSIG, Effect("none"))
    mem.record(("any", PSIG, "touch:9"), (Effect("game_over"),), 0)
    assert veto(lava, Action(4), mem, set()) == "predicted game_over"

    wall = scene_of({(10, 10): 1, (11, 10): 5})
    mem2 = mem_moves((4,))
    know(mem2, "A4", WSIG, Effect("none"))
    mem2.blockers[PSIG] = {5}
    assert veto(wall, Action(4), mem2, set()) == "known no effect here"


def test_veto_settled_test_and_clean_move() -> None:
    mem = mem_moves((4,))
    assert veto(PLAYER, Action(4), mem, set()) is None
    assert veto(PLAYER, Action(4), mem, {(PLAYER.key(), Action(4).key())}) == "settled test"


def test_thin_no_effect_is_not_vetoed() -> None:
    mem = FactMemory()
    know(mem, "A1", PSIG, Effect("none"), times=2)
    assert veto(PLAYER, Action(1), mem, set()) is None


def test_unknown_action_is_never_vetoed_as_settled() -> None:
    tried = {(PLAYER.key(), Action(1).key())}
    assert veto(PLAYER, Action(1), FactMemory(), tried) is None


def test_many_clicks_on_many_things_is_fast() -> None:
    cells: Cells = {(2 * (i % 30) + 1, 2 * (i // 30) + 1): 2 + i % 3 for i in range(300)}
    s = scene_of(cells)
    clicks = [Action(6, t.x, t.y) for t in s.things[:64]]
    start = time.monotonic()
    t = choose_test(s, clicks, FactMemory(), [], set(), None)
    assert t is not None and time.monotonic() - start < 2.0


# --- clicks by place, borrowed click predictions (Task 8d) ----------------------------


def test_identical_tiles_in_different_places_get_their_own_tests() -> None:
    g = np.zeros((64, 64), dtype=np.int8)
    g[10:30, 10:30] = 2
    g[15, 15] = 3
    g[50, 50] = 3
    s = Tracker().update(g, frozenset())[0]
    inert = action_label(Action(6, 50, 50), s)
    mem = FactMemory()
    know(mem, inert, "self", Effect("none"))
    know(mem, inert, s.things[0].sig, Effect("none"))  # the other tile, under that click
    t = choose_test(s, [Action(6, 50, 50), Action(6, 15, 15)], mem, [], set(), None)
    assert t is not None and t.action == Action(6, 15, 15)


def test_borrowed_click_prediction_counts_as_unknown_for_the_tester() -> None:
    s = scene_of({(5, 5): 3, (20, 20): 4, (30, 30): 5})
    label = action_label(Action(6, 5, 5), s)
    mem = FactMemory()
    know(mem, label, "self", Effect("recolor", color=6))
    know(mem, label, f"c4:{DOT}", Effect("none"))  # c5:DOT borrows this sibling's fact
    t = choose_test(s, [Action(6, 5, 5)], mem, [], set(), None)
    assert t is not None and t.action == Action(6, 5, 5)
    assert "c5:" in t.question
    know(mem, label, f"c5:{DOT}", Effect("none"))
    assert choose_test(s, [Action(6, 5, 5)], mem, [], set(), None) is None
