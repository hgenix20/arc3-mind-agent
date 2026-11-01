"""Core contracts shared by the world models, evaluators and searchers.

Invariants:
- A grid is a (64, 64) int8 array with values 0..15. Smaller games are padded
  with -1 by the model that produced them.
- ``Action.id`` follows the ARC-AGI-3 numbering: 0 RESET, 1..5 and 7 simple
  actions, 6 the coordinate action (needs ``x`` and ``y`` in 0..63).
- ``WorldModel.step`` mutates the model; ``clone`` is the only way to branch.
- Evaluators return a scalar where larger is better for the agent (MAX).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Grid = npt.NDArray[np.int8]
GRID_SIZE = 64
NUM_COLORS = 16
RESET_ID = 0
COORD_ACTION_ID = 6


class Phase(StrEnum):
    """Game phase as reported by the environment."""

    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


@dataclass(frozen=True, slots=True)
class Action:
    """One environment action. ``x``/``y`` are -1 unless ``id == 6``."""

    id: int
    x: int = -1
    y: int = -1

    def __post_init__(self) -> None:
        if not 0 <= self.id <= 7:
            raise ValueError(f"action id out of range: {self.id}")
        if self.id == COORD_ACTION_ID and not (0 <= self.x < GRID_SIZE and 0 <= self.y < GRID_SIZE):
            raise ValueError(f"ACTION6 needs x,y in 0..63, got {self.x},{self.y}")

    def key(self) -> tuple[int, int, int]:
        """Hashable identity used for move ordering and transposition tables."""
        return (self.id, self.x, self.y)


@dataclass(frozen=True, slots=True)
class Observation:
    """What the agent is allowed to see after a step."""

    grid: Grid
    phase: Phase
    levels_completed: int
    win_levels: int
    available_actions: tuple[int, ...]

    @property
    def terminal(self) -> bool:
        return self.phase in (Phase.WIN, Phase.GAME_OVER)

    @property
    def won(self) -> bool:
        return self.phase == Phase.WIN


@runtime_checkable
class WorldModel(Protocol):
    """A simulator the search can branch on.

    The oracle implementation wraps the real engine (test and benchmark use
    only). The learned implementation is the agent's own rule model.
    """

    def observe(self) -> Observation: ...

    def legal_actions(self) -> list[Action]:
        """Actions worth expanding, RESET excluded unless the game is terminal."""
        ...

    def step(self, action: Action) -> Observation:
        """Apply ``action`` in place and return the new observation."""
        ...

    def clone(self) -> WorldModel: ...

    def key(self) -> bytes:
        """State identity for transposition tables (grid bytes + progress)."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Static evaluation of an observation, larger is better for the agent.

    ``push``/``pop`` let an implementation keep an incrementally updated
    accumulator along the search path (the NNUE idea). Implementations that
    do not need it may make them no-ops. ``stats`` reports work done so the
    benchmark can compare full recomputes against incremental updates.
    """

    name: str

    def evaluate(self, obs: Observation) -> float: ...

    def push(self, before: Observation, after: Observation) -> None: ...

    def pop(self) -> None: ...

    def stats(self) -> EvalStats: ...

    def reset_stats(self) -> None: ...


@dataclass(slots=True)
class EvalStats:
    """Work counters an evaluator maintains for the benchmark."""

    evaluations: int = 0
    full_recomputes: int = 0
    incremental_updates: int = 0
    cells_touched: int = 0
    eval_seconds: float = 0.0


@dataclass(slots=True)
class SearchResult:
    """What a search returns for one root position."""

    best_action: Action | None
    value: float
    nodes: int
    max_depth_reached: int
    cutoffs: int
    evaluations: int
    elapsed_s: float
    principal_variation: tuple[Action, ...] = field(default_factory=tuple)
    transposition_hits: int = 0
    depth_nodes: dict[int, int] = field(default_factory=dict)


@runtime_checkable
class Searcher(Protocol):
    """A root search: minimax with alpha-beta is the first implementation."""

    name: str

    def search(
        self,
        model: WorldModel,
        evaluator: Evaluator,
        *,
        max_depth: int,
        node_budget: int | None = None,
        time_budget_s: float | None = None,
    ) -> SearchResult: ...


def grid_diff_count(a: Grid, b: Grid) -> int:
    """Number of cells that differ between two grids."""
    return int(np.count_nonzero(a != b))
