"""Objects, scenes, tracking and diffs: the perception layer of the fluid agent.

Pure and deterministic. A ``Thing`` is a 4-connected single-color component with a
position-invariant signature. A ``Tracker`` gives things stable ids across frames and
reports what each one did. Padding cells (value -1) are never things: they land in
``Scene.blocked``. Callers must call ``Tracker.reset`` on a level change or RESET so a
whole-screen replacement is not reported as every object moving.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from arcagent.core.types import Grid
from arcagent.synth.trace import background_color, components

MATCH_RADIUS = 16  # Manhattan distance between top-left corners for a same-sig match

Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Thing:
    """A single-color component. ``x, y`` is the bbox top-left; ``cells`` are absolute (x, y)."""

    id: int
    color: int
    shape: str
    sig: str
    x: int
    y: int
    cells: frozenset[Cell]


@dataclass(frozen=True, slots=True)
class Scene:
    """Things sorted by id, the background color, and padding cells (the grid edge is implicit)."""

    things: tuple[Thing, ...]
    background: int
    blocked: frozenset[Cell]

    def key(self) -> tuple[tuple[str, int, int], ...]:
        """Sorted (sig, x, y); ids are ignored so equal layouts dedupe."""
        return tuple(sorted((t.sig, t.x, t.y) for t in self.things))

    def by_id(self) -> dict[int, Thing]:
        return {t.id: t for t in self.things}

    def occupancy(self) -> dict[Cell, int]:
        """Cell to the id of the thing covering it."""
        return {c: t.id for t in self.things for c in t.cells}


ChangeKind = Literal["moved", "recolored", "reshaped", "disappeared", "appeared", "same"]


@dataclass(frozen=True, slots=True)
class Change:
    """What happened to one id: top-left shift ``dx, dy``; ``color``/``sig`` are the new state."""

    id: int
    kind: ChangeKind
    dx: int = 0
    dy: int = 0
    color: int = -1
    sig: str = ""


@dataclass(frozen=True, slots=True)
class SceneDiff:
    """One change per id present before or after, sorted by id."""

    changes: tuple[Change, ...]
    stray_cells: int  # changed cells outside any thing

    @property
    def empty(self) -> bool:
        return self.stray_cells == 0 and all(c.kind == "same" for c in self.changes)


def shape_of(cells: Iterable[Cell]) -> str:
    """First 10 hex of SHA-1 over the sorted cell offsets from the bbox top-left."""
    pts = list(cells)
    if not pts:
        return ""
    x0 = min(x for x, _ in pts)
    y0 = min(y for _, y in pts)
    offs = sorted((x - x0, y - y0) for x, y in pts)
    return hashlib.sha1(repr(offs).encode()).hexdigest()[:10]


def _visible(grid: Grid, mask: frozenset[Cell]) -> tuple[Grid, int]:
    """Grid with masked cells read as background, and the background color."""
    bg = background_color(grid)
    if not mask or bg < 0:
        return grid, bg
    g = grid.copy()
    for x, y in mask:
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and g[y, x] >= 0:
            g[y, x] = bg
    return g, bg


def _things(grid: Grid) -> list[Thing]:
    out: list[Thing] = []
    if background_color(grid) < 0:
        return out
    for yx in components(grid):
        cells = frozenset((x, y) for y, x in yx)
        color = int(grid[yx[0]])
        shape = shape_of(cells)
        out.append(
            Thing(
                id=-1,
                color=color,
                shape=shape,
                sig=f"c{color}:{shape}",
                x=min(x for x, _ in cells),
                y=min(y for _, y in cells),
                cells=cells,
            )
        )
    return out


def extract(grid: Grid, mask: frozenset[Cell] = frozenset()) -> list[Thing]:
    """Things in components order with id -1. Masked cells read as background."""
    return _things(_visible(grid, mask)[0])


def _with_id(t: Thing, tid: int) -> Thing:
    return Thing(tid, t.color, t.shape, t.sig, t.x, t.y, t.cells)


Pairs = dict[int, int]


def _match_same_sig(before: list[Thing], after: list[Thing], pairs: Pairs, used: set[int]) -> None:
    """Same sig: stationary pairs first, then nearest top-left within the radius.

    Stationary pairs (same sig, same top-left) are unique, so a dict finds them in
    linear time. Only the leftovers pay for the greedy nearest search.
    """
    here = {(a.sig, a.x, a.y): j for j, a in enumerate(after)}
    for i, b in enumerate(before):
        j = here.get((b.sig, b.x, b.y))
        if j is not None:
            pairs[i] = j
            used.add(j)
    by_sig: dict[str, list[int]] = defaultdict(list)
    for j, a in enumerate(after):
        if j not in used:
            by_sig[a.sig].append(j)
    cands: list[tuple[int, int, int]] = []
    for i, b in enumerate(before):
        if i in pairs:
            continue
        for j in by_sig.get(b.sig, ()):
            d = abs(after[j].x - b.x) + abs(after[j].y - b.y)
            if d <= MATCH_RADIUS:
                cands.append((d, i, j))
    _take(sorted(cands), pairs, used)


def _match_recolor(before: list[Thing], after: list[Thing], pairs: Pairs, used: set[int]) -> None:
    """Same shape at the same top-left."""
    at: dict[tuple[str, int, int], int] = {}
    for j, a in enumerate(after):
        if j not in used:
            at.setdefault((a.shape, a.x, a.y), j)
    cands = [
        (0, i, at[(b.shape, b.x, b.y)])
        for i, b in enumerate(before)
        if i not in pairs and (b.shape, b.x, b.y) in at
    ]
    _take(cands, pairs, used)


def _match_reshape(before: list[Thing], after: list[Thing], pairs: Pairs, used: set[int]) -> None:
    """Same color with overlapping cells, largest overlap first."""
    cands: list[tuple[int, int, int]] = []
    for i, b in enumerate(before):
        if i in pairs:
            continue
        for j, a in enumerate(after):
            ov = len(b.cells & a.cells) if j not in used and a.color == b.color else 0
            if ov:
                cands.append((-ov, i, j))
    _take(sorted(cands), pairs, used)


def _take(cands: list[tuple[int, int, int]], pairs: Pairs, used: set[int]) -> None:
    for _, i, j in cands:
        if i not in pairs and j not in used:
            pairs[i] = j
            used.add(j)


def _match(before: list[Thing], after: list[Thing]) -> Pairs:
    """Map index in ``before`` to index in ``after``: same sig, then recolor, then reshape."""
    pairs: Pairs = {}
    used: set[int] = set()
    _match_same_sig(before, after, pairs, used)
    _match_recolor(before, after, pairs, used)
    _match_reshape(before, after, pairs, used)
    return pairs


def _classify(b: Thing, a: Thing) -> Change:
    dx, dy = a.x - b.x, a.y - b.y
    if a.sig == b.sig:
        kind: ChangeKind = "moved" if dx or dy else "same"
        return Change(b.id, kind, dx, dy, a.color, a.sig)
    if a.shape == b.shape:
        return Change(b.id, "recolored", dx, dy, a.color, a.sig)
    return Change(b.id, "reshaped", dx, dy, a.color, a.sig)


class Tracker:
    """Stable ids across frames. The first frame after ``reset`` yields no diff."""

    def __init__(self) -> None:
        self._next_id = 1
        self._scene: Scene | None = None
        self._grid: Grid | None = None

    def reset(self) -> None:
        """Forget the previous frame and restart ids at 1."""
        self._next_id = 1
        self._scene = None
        self._grid = None

    def update(self, grid: Grid, mask: frozenset[Cell]) -> tuple[Scene, SceneDiff | None]:
        visible, bg = _visible(grid, mask)
        fresh = _things(visible)
        ys, xs = np.nonzero(grid < 0)
        blocked = frozenset(zip(xs.tolist(), ys.tolist(), strict=True))
        prev = self._scene
        if prev is None:
            first = [_with_id(t, self._take_id()) for t in fresh]
            scene = Scene(tuple(first), bg, blocked)
            self._scene, self._grid = scene, grid.copy()
            return scene, None

        pairs = _match(list(prev.things), fresh)
        matched_after = set(pairs.values())
        things: list[Thing] = []
        changes: list[Change] = []
        for i, b in enumerate(prev.things):
            j = pairs.get(i)
            if j is None:
                changes.append(Change(b.id, "disappeared"))
                continue
            things.append(_with_id(fresh[j], b.id))
            changes.append(_classify(b, fresh[j]))
        for j, a in enumerate(fresh):
            if j not in matched_after:
                t = _with_id(a, self._take_id())
                things.append(t)
                changes.append(Change(t.id, "appeared", color=t.color, sig=t.sig))
        things.sort(key=lambda t: t.id)
        changes.sort(key=lambda c: c.id)
        scene = Scene(tuple(things), bg, blocked)
        stray = self._stray(prev, scene, grid, mask)
        self._scene, self._grid = scene, grid.copy()
        return scene, SceneDiff(tuple(changes), stray)

    def _take_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _stray(self, prev: Scene, now: Scene, grid: Grid, mask: frozenset[Cell]) -> int:
        """Changed cells that no thing covered before or after, ignoring masked cells."""
        old = self._grid
        if old is None or old.shape != grid.shape:
            return 0
        ys, xs = np.nonzero(old != grid)
        covered: set[Cell] = set(mask)
        for t in prev.things:
            covered |= t.cells
        for t in now.things:
            covered |= t.cells
        return sum(1 for c in zip(xs.tolist(), ys.tolist(), strict=True) if c not in covered)
