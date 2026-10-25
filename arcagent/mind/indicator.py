"""Progress indicators: rows of equal boxes that light one at a time, and what lights them.

A progress indicator is a row (or column) of at least ``SLOT_MIN`` equal boxes whose looks
change one box at a time when something right happens, and fall back when something wrong
happens. ``slot_rows`` finds the candidates on a level's first frame: things of at most
``SLOT_MAX_CELLS`` cells with one shape, on one line, evenly spaced (their colors may
differ). ``lit`` counts the boxes whose cells differ from their level-start look.

``SequenceTracker`` watches the candidates move by move. The first time a candidate's lit
count rises by one on a move that didn't click inside it, that candidate is the
indicator; a candidate a click lands in is an input row (the player sets it directly) and
is dropped. From then on, every rise records the move's trigger (what the agent clicked or
touched, as kinds) as the next step at the old count, and every fall records it as a
wrong step at that count. A new life rereads the count with no event: the game resets
the indicator, the agent did nothing wrong. ``goal`` is the ``sequence`` goal (every box
lit). Pure apart from the tracker's own state; nothing here names a particular game.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from arcagent.core.types import Grid
from arcagent.mind.goals import Box, Goal, box_look, encode_boxes
from arcagent.mind.objects import Scene, Thing

Cell = tuple[int, int]
Event = Literal["up", "reset", "none"]

SLOT_MIN = 3  # boxes in a run before it can be an indicator
SLOT_MAX_CELLS = 64  # bigger things aren't slots
MAX_ROWS = 8  # candidate rows kept per level


@dataclass(frozen=True, slots=True)
class SlotRow:
    """Boxes ``(x, y, w, h)`` in line order and their looks at level start, joined by ``|``."""

    boxes: tuple[Box, ...]
    base: str


def _box(t: Thing) -> Box:
    xs = [x for x, _ in t.cells]
    ys = [y for _, y in t.cells]
    return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _runs(boxes: list[Box], axis: int) -> list[list[Box]]:
    """Maximal evenly spaced runs of at least ``SLOT_MIN`` boxes along ``axis`` (0 is x),
    boxes sorted along it and not overlapping."""
    out: list[list[Box]] = []
    run: list[Box] = []
    size = 2 + axis  # w for x, h for y
    for b in boxes:
        if run and b[axis] - run[-1][axis] < run[-1][size]:
            continue  # overlaps the last box: not a slot of this run
        if len(run) >= 2 and b[axis] - run[-1][axis] != run[1][axis] - run[0][axis]:
            if len(run) >= SLOT_MIN:
                out.append(run)
            run = [run[-1]]
        run.append(b)
    if len(run) >= SLOT_MIN:
        out.append(run)
    return out


def slot_rows(scene: Scene, grid: Grid) -> list[SlotRow]:
    """Candidate indicators in ``scene`` (module docstring), at most ``MAX_ROWS``, in a
    deterministic order: rows before columns, by shape, then position."""
    groups: dict[tuple[str, int, int], list[Box]] = defaultdict(list)
    for t in scene.things:
        if len(t.cells) <= SLOT_MAX_CELLS:
            b = _box(t)
            groups[(t.shape, b[2], b[3])].append(b)
    found: list[tuple[Box, ...]] = []
    for axis in (0, 1):
        across = 1 - axis
        for key in sorted(groups):
            lines: dict[int, list[Box]] = defaultdict(list)
            for b in groups[key]:
                lines[b[across]].append(b)
            for line in sorted(lines):
                ordered = sorted(lines[line], key=lambda b: b[axis])
                found += [tuple(run) for run in _runs(ordered, axis)]
    rows: list[SlotRow] = []
    for boxes in found[:MAX_ROWS]:
        looks = [box_look(grid, b) for b in boxes]
        if all(look is not None for look in looks):
            rows.append(SlotRow(boxes, "|".join(look or "" for look in looks)))
    return rows


def lit(row: SlotRow, grid: Grid) -> int:
    """Boxes of ``row`` whose cells differ from their level-start look."""
    base = row.base.split("|")
    return sum(1 for b, look in zip(row.boxes, base, strict=True) if box_look(grid, b) != look)


def _inside(cell: Cell, row: SlotRow) -> bool:
    x, y = cell
    return any(bx <= x < bx + w and by <= y < by + h for bx, by, w, h in row.boxes)


class SequenceTracker:
    """Finds the indicator and learns its steps (module docstring)."""

    def __init__(self) -> None:
        self._candidates: list[SlotRow] = []
        self._counts: list[int] = []
        self.row: SlotRow | None = None
        self.by_click = False  # the indicator first rose on a click
        self._k = 0  # lit boxes now
        self._steps: dict[int, frozenset[str]] = {}
        self._wrong: dict[int, set[str]] = defaultdict(set)

    def start_level(self, scene: Scene, grid: Grid) -> None:
        """A new level: forget the indicator and its steps, find new candidates."""
        self.__init__()
        self._candidates = slot_rows(scene, grid)
        self._counts = [0] * len(self._candidates)

    def start_life(self, grid: Grid) -> None:
        """The first frame of a life: reread the counts, no event."""
        self._counts = [lit(r, grid) for r in self._candidates]
        if self.row is not None:
            self._k = lit(self.row, grid)

    def observe(self, grid: Grid, trigger: frozenset[str], clicked: Cell | None) -> Event:
        """The frame after a move; ``trigger`` the kinds it clicked or touched, ``clicked``
        the cell of a click (None for any other action)."""
        if self.row is None:
            return self._find(grid, trigger, clicked)
        k0, k1 = self._k, lit(self.row, grid)
        self._k = k1
        if k1 > k0:
            self._learn(k0, trigger)
            return "up"
        if k1 < k0:
            self._wrong[k0].update(trigger - self._steps.get(k0, frozenset()))
            return "reset"
        return "none"

    def _find(self, grid: Grid, trigger: frozenset[str], clicked: Cell | None) -> Event:
        keep: list[SlotRow] = []
        counts: list[int] = []
        found: tuple[SlotRow, int] | None = None
        for row, k0 in zip(self._candidates, self._counts, strict=True):
            if clicked is not None and _inside(clicked, row):
                continue  # an input row: the player sets it directly
            k1 = lit(row, grid)
            if found is None and k1 == k0 + 1:
                found = (row, k0)
            keep.append(row)
            counts.append(k1)
        self._candidates, self._counts = keep, counts
        if found is None:
            return "none"
        self.row, k0 = found
        self.by_click = clicked is not None
        self._learn(k0, trigger)
        self._k = k0 + 1
        return "up"

    def _learn(self, k: int, trigger: frozenset[str]) -> None:
        if not trigger:
            return
        old = self._steps.get(k)
        self._steps[k] = old & trigger if old is not None and old & trigger else trigger

    def next_kinds(self) -> frozenset[str]:
        """The kinds that lit the next box before (empty when not known yet)."""
        return self._steps.get(self._k, frozenset())

    def wrong_now(self) -> frozenset[str]:
        """Kinds that put the indicator out at this count."""
        return frozenset(self._wrong.get(self._k, set()))

    def avoid(self) -> frozenset[str]:
        """Kinds that were wrong at this count, and kinds used at other counts."""
        used = {kind for k, kinds in self._steps.items() if k != self._k for kind in kinds}
        return self.wrong_now() | used

    def goal(self) -> Goal | None:
        """The ``sequence`` goal for the indicator, once found."""
        if self.row is None:
            return None
        return Goal("sequence", a=self.row.base, b=encode_boxes(self.row.boxes))
