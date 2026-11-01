"""Perception: frame hashing, object extraction, change measurement, action list.

Everything here is pure and deterministic. ``candidate_actions`` is the single
action list the planner, value-of-information and the B0 fallback all use, so
they cannot disagree about what the agent may try.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from arcagent.agent.explore import click_candidates
from arcagent.core.types import COORD_ACTION_ID, RESET_ID, Action, Grid, Observation
from arcagent.synth.trace import background_color, components


def frame_hash(grid: Grid) -> str:
    """First 16 hex chars of the SHA-1 of the grid's int8 bytes."""
    return hashlib.sha1(grid.astype(np.int8).tobytes()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Obj:
    """A 4-connected single-color component. ``bbox`` is ``(x0, y0, x1, y1)`` inclusive."""

    color: int
    size: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]  # (x, y)


@dataclass(frozen=True, slots=True)
class Percept:
    grid: Grid
    hash: str
    objects: tuple[Obj, ...]


def perceive(grid: Grid) -> Percept:
    """Objects are the non-background components, in ``components`` order."""
    objs: list[Obj] = []
    if background_color(grid) >= 0:
        for cells in components(grid):
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            objs.append(
                Obj(
                    color=int(grid[cells[0]]),
                    size=len(cells),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
                )
            )
    return Percept(grid=grid, hash=frame_hash(grid), objects=tuple(objs))


def changed_cells(a: Grid, b: Grid) -> int:
    """Number of cells that differ. Grids must have the same shape."""
    return int(np.count_nonzero(a != b))


def changed_bbox(a: Grid, b: Grid) -> tuple[int, int, int, int] | None:
    """Inclusive ``(x0, y0, x1, y1)`` box around differing cells, or None if equal."""
    ys, xs = np.nonzero(a != b)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def candidate_actions(obs: Observation) -> list[Action]:
    """Available simple actions (never RESET or 6), then one click per click target."""
    acts = [
        Action(a)
        for a in obs.available_actions
        if a not in (RESET_ID, COORD_ACTION_ID) and 1 <= a <= 7
    ]
    if COORD_ACTION_ID in obs.available_actions:
        acts.extend(Action(COORD_ACTION_ID, x, y) for x, y in click_candidates(obs.grid))
    return acts
