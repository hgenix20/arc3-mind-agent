"""Shared value types for the mind loops.

Invariants: every type is frozen. ``Step`` is one observed transition and is
the unit that programs are verified against; ``Event`` is the only signal the
predictive loop emits (it never chooses an action).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from arcagent.core.types import Action, Grid, Phase

EventKind = Literal[
    "new_game",
    "prediction_error",
    "level_up",
    "game_over",
    "belief_empty",
    "transfer_failed",
]
Branch = Literal["follow", "plan", "learn", "fallback", "wait", "reset"]
ActionKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Step:
    """One transition: ``action`` taken on ``before`` produced ``after``."""

    index: int
    before: Grid
    action: Action
    after: Grid
    levels_before: int
    levels_after: int
    phase_after: Phase

    @property
    def level_up(self) -> bool:
        return self.levels_after > self.levels_before


@dataclass(frozen=True, slots=True)
class Program:
    """A candidate world-model program. ``origin`` records where it came from."""

    id: str
    code: str
    has_goal: bool
    origin: Literal["llm", "transfer"]


@dataclass(frozen=True, slots=True)
class Event:
    """A logged occurrence. ``magnitude`` is in [0, 1] by convention of the emitter."""

    kind: EventKind
    step: int
    magnitude: float
    detail: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gate:
    """The wisdom function's verdict on a candidate; ``reason`` is always logged."""

    verdict: Literal["pass", "caveat", "hold", "drop"]
    reason: str


@dataclass(frozen=True, slots=True)
class Decision:
    """The action chosen for this tick and the branch that chose it.

    ``program`` is the id of the program whose plan produced the action (a
    follow, plan-to-goal or plan-to-novel step), else None. The agent uses it as
    the active program, which drives inhibition and the follow gate.
    """

    action: Action
    branch: Branch
    why: str
    program: str | None = None
