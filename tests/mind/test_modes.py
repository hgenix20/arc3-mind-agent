"""Tests for hidden mode state: selection and the active controller."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import COORD_ACTION_ID, Action, Grid
from arcagent.mind.modes import Mode, ModeTracker
from arcagent.mind.objects import Tracker

Cells = dict[tuple[int, int], int]

P1, P2 = (10, 10), (20, 10)


def ring(x: int, y: int, color: int = 8) -> Cells:
    return {(x + dx, y + dy): color for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy}


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


class Run:
    """A tracker and a mode tracker fed frame by frame."""

    def __init__(self, first: Cells) -> None:
        self.t = Tracker()
        self.modes = ModeTracker()
        self.scene, _ = self.t.update(grid_with(first), frozenset())

    def step(
        self, action: Action, cells: Cells, masked: frozenset[tuple[int, int]] = frozenset()
    ) -> Mode:
        before = self.scene
        self.scene, diff = self.t.update(grid_with(cells), frozenset())
        assert diff is not None
        return self.modes.observe(before, action, self.scene, diff, masked)

    def id_at(self, cell: tuple[int, int]) -> int:
        return next(t.id for t in self.scene.things if cell in t.cells)


def click(cell: tuple[int, int]) -> Action:
    return Action(COORD_ACTION_ID, cell[0], cell[1])


PEGS: Cells = {P1: 2, P2: 2}


def test_one_click_alone_is_not_yet_selection() -> None:
    run = Run(PEGS)
    assert run.step(click(P1), PEGS | ring(*P1)).selected == frozenset()


def test_outline_moving_with_clicks_marks_selection() -> None:
    run = Run(PEGS)
    run.step(click(P1), PEGS | ring(*P1))
    mode = run.step(click(P2), PEGS | ring(*P2))
    assert mode.selected == frozenset({run.id_at(P2)})
    mode = run.step(click(P1), PEGS | ring(*P1))  # known now: follows at once
    assert mode.selected == frozenset({run.id_at(P1)})


def test_recolor_on_click_then_undo_marks_selection() -> None:
    tile, lit = {(30, 30): 3, (31, 30): 3}, {(30, 30): 7, (31, 30): 7}
    run = Run(tile)
    run.step(click((30, 30)), lit)
    assert run.step(click((30, 30)), tile).selected == frozenset()  # undone: known, not shown
    mode = run.step(click((30, 30)), lit)
    assert mode.selected == frozenset({run.id_at((30, 30))})


def test_masked_change_is_not_a_selection_marker() -> None:
    run = Run(PEGS)
    masked = frozenset(ring(*P1)) | frozenset(ring(*P2))
    run.step(click(P1), PEGS | ring(*P1), masked)
    mode = run.step(click(P2), PEGS | ring(*P2), masked)
    assert mode.selected == frozenset() and not run.modes.known


def test_reset_clears_mode() -> None:
    run = Run(PEGS)
    run.step(click(P1), PEGS | ring(*P1))
    run.step(click(P2), PEGS | ring(*P2))
    run.modes.reset()
    assert run.modes.mode == Mode() and not run.modes.known


def test_a_move_is_never_a_marker() -> None:
    run = Run(PEGS)
    run.step(click(P1), {(11, 10): 2, P2: 2})  # the clicked peg moved: not a mark
    mode = run.step(click(P2), {(11, 10): 2, (21, 10): 2})
    assert mode.selected == frozenset() and not run.modes.known


A, B, ENEMY = (5, 30), (40, 30), (20, 50)
BODIES: Cells = {A: 1, B: 5, ENEMY: 6}


def controlled(run: Run) -> frozenset[str]:
    return frozenset(t.sig for t in run.scene.things if t.color in (1, 5))


def shifted(cells: Cells, at: tuple[int, int], dx: int) -> Cells:
    out = {k: v for k, v in cells.items() if k != at}
    out[(at[0] + dx, at[1])] = cells[at]
    return out


def test_active_is_the_controlled_thing_that_moved() -> None:
    run = Run(BODIES)
    ctl = controlled(run)
    nxt = shifted(BODIES, A, 1)
    before = run.scene
    run.scene, diff = run.t.update(grid_with(nxt), frozenset())
    assert diff is not None
    mode = run.modes.observe(before, Action(4), run.scene, diff, controlled=ctl)
    assert mode.active == run.id_at((A[0] + 1, A[1]))


def test_switch_action_changes_active_and_enemies_never_count() -> None:
    run = Run(BODIES)
    ctl = controlled(run)
    frames = [
        (Action(4), shifted(BODIES, A, 1)),
        (Action(5), shifted(shifted(BODIES, A, 1), ENEMY, 1)),  # only the enemy moved
        (Action(4), shifted(shifted(shifted(BODIES, A, 1), ENEMY, 1), B, 1)),
    ]
    modes = []
    for action, cells in frames:
        before = run.scene
        run.scene, diff = run.t.update(grid_with(cells), frozenset())
        assert diff is not None
        modes.append(run.modes.observe(before, action, run.scene, diff, controlled=ctl))
    a_id, b_id = run.id_at((A[0] + 1, A[1])), run.id_at((B[0] + 1, B[1]))
    assert [m.active for m in modes] == [a_id, a_id, b_id]


def test_click_moves_never_set_active() -> None:
    run = Run(BODIES)
    ctl = controlled(run)
    before = run.scene
    run.scene, diff = run.t.update(grid_with(shifted(BODIES, A, 1)), frozenset())
    assert diff is not None
    mode = run.modes.observe(before, click(A), run.scene, diff, controlled=ctl)
    assert mode.active is None


def test_a_lone_controlled_thing_is_never_tagged_active() -> None:
    run = Run({A: 1})
    ctl = controlled(run)
    before = run.scene
    run.scene, diff = run.t.update(grid_with({(A[0] + 1, A[1]): 1}), frozenset())
    assert diff is not None
    assert run.modes.observe(before, Action(4), run.scene, diff, controlled=ctl).active is None
