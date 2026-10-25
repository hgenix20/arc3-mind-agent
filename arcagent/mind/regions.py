"""Region primitives for goal kinds that compare parts of the screen.

``framed_regions`` finds the inside of every thing whose cells are exactly the one-cell
outline of its bbox (a box drawn around something), as ``(x0, y0, x1, y1)`` regions with
exclusive ends, like ``goals.Region``. ``twin_regions`` pairs framed regions of equal size
(at least ``TWIN_MIN`` cells on each side): a reference pattern and the area meant to match
it are drawn that way. Pairs are ranked by how many non-background cells the two insides
hold together (an empty pair says little), then by position, and at most ``MAX_TWINS`` are
kept. ``static_cells`` gives the cells equal on every grid of a list (this level's frames).
``render`` paints a scene back into a grid (terrain, then things), so a predicted scene can
be checked against a region goal. Pure.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.objects import Cell, Scene, Thing

Region = tuple[int, int, int, int]  # (x0, y0, x1, y1), end exclusive

TWIN_MIN = 3  # a twin region is at least this many cells on each side
MAX_TWINS = 20  # twin pairs kept, richest first


def _inside_of_frame(t: Thing) -> Region | None:
    """The inside of ``t`` when its cells are exactly its bbox outline, else None."""
    xs = [x for x, _ in t.cells]
    ys = [y for _, y in t.cells]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    border = {(x, y) for x in range(x0, x1 + 1) for y in (y0, y1)}
    border |= {(x, y) for y in range(y0, y1 + 1) for x in (x0, x1)}
    return (x0 + 1, y0 + 1, x1, y1) if t.cells == frozenset(border) else None


def framed_regions(scene: Scene) -> list[Region]:
    """Insides of box outlines, in thing order."""
    return [r for t in scene.things if (r := _inside_of_frame(t)) is not None]


def _size(r: Region) -> tuple[int, int]:
    return r[2] - r[0], r[3] - r[1]


def _content(grid: Grid, r: Region, background: int) -> int:
    x0, y0, x1, y1 = r
    return int((grid[y0:y1, x0:x1] != background).sum())


def twin_regions(scene: Scene, grid: Grid) -> list[tuple[Region, Region]]:
    """Pairs of equal-size framed regions, richest first, at most ``MAX_TWINS``."""
    regions = [r for r in framed_regions(scene) if min(_size(r)) >= TWIN_MIN]
    pairs = [
        (a, b) if a <= b else (b, a) for a, b in combinations(regions, 2) if _size(a) == _size(b)
    ]
    bg = scene.background
    pairs.sort(key=lambda p: (-(_content(grid, p[0], bg) + _content(grid, p[1], bg)), p))
    return pairs[:MAX_TWINS]


def static_cells(grids: Sequence[Grid]) -> frozenset[Cell]:
    """Cells with the same value on every grid; empty for no grids."""
    if not grids:
        return frozenset()
    first = grids[0]
    same = np.ones(first.shape, dtype=bool)
    for g in grids[1:]:
        same &= g == first
    ys, xs = np.nonzero(same)
    return frozenset(zip(xs.tolist(), ys.tolist(), strict=True))


def render(scene: Scene) -> Grid:
    """The scene as a grid: its terrain with every thing painted on."""
    g = np.array(scene.terrain, dtype=np.int8, copy=True)
    for t in scene.things:
        for (x, y), color in t.colored():
            g[y, x] = color
    return g
