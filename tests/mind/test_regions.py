"""Tests for region primitives: framed regions, twin regions, static cells."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.objects import Scene, Tracker
from arcagent.mind.regions import MAX_TWINS, framed_regions, render, static_cells, twin_regions

Cells = dict[tuple[int, int], int]


def frame(x0: int, y0: int, w: int, h: int, color: int = 5) -> Cells:
    """A one-cell border of a w x h box (outer size) with top-left (x0, y0)."""
    out: Cells = {}
    for x in range(x0, x0 + w):
        out[(x, y0)] = out[(x, y0 + h - 1)] = color
    for y in range(y0, y0 + h):
        out[(x0, y)] = out[(x0 + w - 1, y)] = color
    return out


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def scene_of(cells: Cells) -> tuple[Scene, Grid]:
    g = grid_with(cells)
    return Tracker().update(g, frozenset())[0], g


def test_framed_region_is_the_inside_of_a_box_outline() -> None:
    scene, _ = scene_of(frame(10, 10, 5, 4))
    assert framed_regions(scene) == [(11, 11, 14, 13)]


def test_a_filled_rectangle_is_not_a_frame() -> None:
    scene, _ = scene_of({(x, y): 5 for x in range(10, 15) for y in range(10, 14)})
    assert framed_regions(scene) == []


def test_twin_regions_pair_equal_sizes_ranked_by_content() -> None:
    pattern = {(11, 11): 3, (12, 12): 3}
    cells = frame(10, 10, 5, 5) | pattern | frame(30, 10, 5, 5) | {(31, 11): 3}
    cells |= frame(10, 40, 6, 5)  # a different size: never a twin
    scene, grid = scene_of(cells)
    twins = twin_regions(scene, grid)
    assert twins == [((11, 11, 14, 14), (31, 11, 34, 14))]


def test_twins_are_capped() -> None:
    cells: Cells = {}
    for i in range(8):
        cells |= frame(2 + 7 * i, 5, 5, 5) | {(3 + 7 * i, 6): 2}
    scene, grid = scene_of(cells)
    assert len(twin_regions(scene, grid)) == MAX_TWINS == 20  # 28 pairs, kept to 20


def test_static_cells_are_the_ones_that_never_changed() -> None:
    a = grid_with({(1, 1): 2, (5, 5): 3})
    b = grid_with({(1, 1): 2, (6, 5): 3})
    still = static_cells([a, b])
    assert (1, 1) in still and (5, 5) not in still and (6, 5) not in still
    assert static_cells([]) == frozenset()


def test_render_paints_things_over_terrain() -> None:
    scene, grid = scene_of(frame(10, 10, 5, 4) | {(30, 30): 7})
    assert (render(scene) == grid).all()
