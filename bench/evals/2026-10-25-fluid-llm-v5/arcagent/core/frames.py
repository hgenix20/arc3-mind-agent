"""Conversion from the toolkit's frame objects to the agent's ``Observation``.

Shared by the oracle (local engine) and the playing agent (public wrapper, local
or gateway): both receive a ``FrameDataRaw``-like object whose ``frame`` is a list
of animation layers, ``state`` a ``GameState`` enum, and the rest plain ints.
This module has no toolkit import so the package loads without the wheels.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from arcagent.core.types import GRID_SIZE, Grid, Observation, Phase


class FrameLike(Protocol):
    """The fields of ``arcengine.FrameDataRaw`` the agent reads."""

    @property
    def frame(self) -> list[Any]: ...

    @property
    def state(self) -> Any: ...

    @property
    def levels_completed(self) -> int: ...

    @property
    def win_levels(self) -> int: ...

    @property
    def available_actions(self) -> list[int]: ...


def grid_from_frame(frame: list[Any]) -> Grid:
    """Last animation layer of a step as a padded (64, 64) int8 grid; all -1 if empty."""
    g = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
    if not frame:
        return g
    arr = np.asarray(frame[-1], dtype=np.int8)
    h, w = arr.shape[:2]
    g[:h, :w] = arr
    return g


def phase_from_state(state: Any) -> Phase:
    """``GameState.WIN`` (enum or its string form) -> ``Phase.WIN``."""
    name = getattr(state, "name", str(state))
    return Phase(name.split(".")[-1])


def observation_from_frame(fd: FrameLike) -> Observation:
    """What the agent may see of a frame: grid, phase, progress, available actions."""
    return Observation(
        grid=grid_from_frame(list(fd.frame)),
        phase=phase_from_state(fd.state),
        levels_completed=int(fd.levels_completed),
        win_levels=int(fd.win_levels),
        available_actions=tuple(int(a) for a in fd.available_actions),
    )
