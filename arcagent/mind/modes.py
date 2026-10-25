"""Hidden mode state as tags on tracked things: which thing is selected, which is active.

Nothing on screen says "selected". ``ModeTracker`` infers it from what clicks change. After
a click on a thing T, a non-move change confined to T's bbox grown by ``MARK_RING`` cells is
a candidate mark: a thing that appears there (an outline), described by its color and
shape, or T itself recoloring, described by the new color. A candidate becomes a known
selection marker when a later transition shows it was a selection: the same mark appears
on another clicked thing while it vanishes from the one it was on, or a second click on the
thing it is on makes it vanish (an undo). From then on, a thing showing a known mark is
selected, and stops being selected when the mark vanishes from it.

A mark vanishes when the thing that drew it disappears, or when its owner recolors away
from the mark color. A drawn mark can also move (the tracker keeps one outline's id as it
jumps to another piece): moving to surround the clicked thing is the same as vanishing
from the old owner and appearing on the clicked one; moving anywhere else is vanishing.
Changes on ``masked`` cells (counters, noise) are never marks, and the clicked thing's own
move is never a mark. ``Mode.active`` is filled by the active-controller rule (Task 4.2).
Pure apart from the tracker's own state; ids are the object tracker's.
"""

from __future__ import annotations

from dataclasses import dataclass

from arcagent.core.types import COORD_ACTION_ID, Action
from arcagent.mind.objects import Cell, Scene, SceneDiff, Thing

MARK_RING = 2  # a mark lies within the clicked thing's bbox grown by this many cells

Mark = tuple[str, int, str]  # ("appear", color, shape) or ("recolor", color, "")


@dataclass(frozen=True, slots=True)
class Mode:
    """Selected track ids and the active controller's track id (None when unknown)."""

    selected: frozenset[int] = frozenset()
    active: int | None = None


def _box(t: Thing, grow: int) -> tuple[int, int, int, int]:
    xs = [x for x, _ in t.cells]
    ys = [y for _, y in t.cells]
    return min(xs) - grow, min(ys) - grow, max(xs) + grow, max(ys) + grow


def _inside(cells: frozenset[Cell], box: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    return all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in cells)


def clicked_thing(scene: Scene, action: Action) -> Thing | None:
    """The thing under a click, or None (not a click, or empty space)."""
    if action.id != COORD_ACTION_ID:
        return None
    point = (action.x, action.y)
    return next((t for t in scene.things if point in t.cells), None)


class ModeTracker:
    """Selection inferred from clicks (module docstring)."""

    def __init__(self) -> None:
        self.known: set[Mark] = set()
        self._drawn: dict[int, tuple[Mark, int]] = {}  # mark thing id -> (mark, owner id)
        self._tinted: dict[int, Mark] = {}  # owner id -> recolor mark it shows
        self._last_on: dict[Mark, int] = {}  # mark -> owner it last appeared on
        self._selected: set[int] = set()
        self._active: int | None = None

    def reset(self) -> None:
        """Forget marks, selection and the active thing (a new level or attempt)."""
        self.__init__()

    @property
    def mode(self) -> Mode:
        return Mode(frozenset(self._selected), self._active)

    def observe(
        self,
        before: Scene,
        action: Action,
        after: Scene,
        diff: SceneDiff,
        masked: frozenset[Cell] = frozenset(),
    ) -> Mode:
        """Update from one transition; return the mode after it."""
        clicked = clicked_thing(before, action)
        vanished, fresh = self._moved(clicked, after, diff)
        vanished += self._vanished(diff)
        if clicked is not None:
            fresh += self._fresh(clicked, after, diff, masked)
        for mark, owner in vanished:
            if clicked is not None and owner == clicked.id and self._last_on.get(mark) == owner:
                self.known.add(mark)  # a second click on it took it away: an undo
        for mark in fresh:
            assert clicked is not None
            prev = self._last_on.get(mark)
            if prev is not None and prev != clicked.id and (mark, prev) in vanished:
                self.known.add(mark)  # it moved from one clicked thing to another
            self._last_on[mark] = clicked.id
        for mark, owner in vanished:
            if mark in self.known:
                self._selected.discard(owner)
        if clicked is not None and any(m in self.known for m in fresh):
            self._selected.add(clicked.id)
        alive = {t.id for t in after.things}
        self._selected &= alive
        return self.mode

    def _moved(
        self, clicked: Thing | None, after: Scene, diff: SceneDiff
    ) -> tuple[list[tuple[Mark, int]], list[Mark]]:
        """Drawn marks that moved: each leaves its owner; one now around ``clicked`` lands
        on it. Returns (vanished, fresh)."""
        now = after.by_id()
        box = _box(clicked, MARK_RING) if clicked is not None else None
        gone: list[tuple[Mark, int]] = []
        landed: list[Mark] = []
        for c in diff.changes:
            if c.kind not in ("moved", "reshaped") or c.id not in self._drawn:
                continue
            mark, owner = self._drawn.pop(c.id)
            gone.append((mark, owner))
            t = now.get(c.id)
            if clicked is not None and box is not None and t is not None and _inside(t.cells, box):
                self._drawn[c.id] = (mark, clicked.id)
                landed.append(mark)
        return gone, landed

    def _vanished(self, diff: SceneDiff) -> list[tuple[Mark, int]]:
        """Marks gone in this transition, with the owner each was on."""
        out: list[tuple[Mark, int]] = []
        for c in diff.changes:
            if c.kind == "disappeared" and c.id in self._drawn:
                out.append(self._drawn.pop(c.id))
            elif c.kind == "recolored" and c.id in self._tinted:
                mark = self._tinted[c.id]
                if c.color != mark[1]:
                    out.append((self._tinted.pop(c.id), c.id))
        return out

    def _fresh(
        self, clicked: Thing, after: Scene, diff: SceneDiff, masked: frozenset[Cell]
    ) -> list[Mark]:
        """Marks this click put on ``clicked``: things appearing around it, or its recolor."""
        box = _box(clicked, MARK_RING)
        now = after.by_id()
        out: list[Mark] = []
        for c in diff.changes:
            if c.kind == "appeared":
                t = now.get(c.id)
                if t is None or t.cells <= masked or not _inside(t.cells, box):
                    continue
                mark: Mark = ("appear", t.color, t.shape)
                self._drawn[t.id] = (mark, clicked.id)
                out.append(mark)
            elif c.kind == "recolored" and c.id == clicked.id and not clicked.cells <= masked:
                mark = ("recolor", c.color, "")
                self._tinted[clicked.id] = mark
                out.append(mark)
        return out
