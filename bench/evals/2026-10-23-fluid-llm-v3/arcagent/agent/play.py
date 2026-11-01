"""Play one game to completion, a budget, or a failure.

Contract:
- ``play_game`` only calls ``env.reset`` / ``env.step`` and reads
  ``env.observation_space``: the interface ``arc_agi.EnvironmentWrapper``
  exposes for local and gateway games alike.
- Every executed action, RESET included, counts toward ``max_actions``.
- The loop stops when the game reports WIN (all levels), when ``max_actions``
  is reached, when ``deadline`` (a ``time.monotonic`` value) passes, or after
  ``MAX_STEP_FAILURES`` consecutive steps return no frame (a gateway or engine
  error). The reason is recorded in ``GameResult.stop``.
- A policy sees only ``Observation`` values; it never touches the wrapper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from arcagent.core.frames import FrameLike, observation_from_frame
from arcagent.core.types import COORD_ACTION_ID, Action, Observation

MAX_STEP_FAILURES = 3


class Policy(Protocol):
    """Chooses the next action from what the agent may see."""

    def choose(self, obs: Observation) -> Action: ...


class Env(Protocol):
    """The part of ``arc_agi.EnvironmentWrapper`` the agent uses."""

    @property
    def observation_space(self) -> FrameLike | None: ...

    def reset(self) -> FrameLike | None: ...

    def step(
        self,
        action: Any,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> FrameLike | None: ...


@dataclass(frozen=True, slots=True)
class GameResult:
    """Outcome of one ``play_game`` call, as the agent itself saw it."""

    game_id: str
    actions: int
    levels_completed: int
    win_levels: int
    phase: str
    seconds: float
    stop: str
    step_failures: int = 0


def to_game_action(action: Action) -> tuple[Any, dict[str, int] | None]:
    """Map an ``Action`` to the toolkit's ``GameAction`` plus its data dict."""
    from arcengine import GameAction  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    data = {"x": action.x, "y": action.y} if action.id == COORD_ACTION_ID else None
    return GameAction.from_id(action.id), data


def play_game(
    env: Env,
    policy: Policy,
    game_id: str,
    *,
    max_actions: int,
    deadline: float | None = None,
    clock: Any = time.monotonic,
) -> GameResult:
    """Drive ``env`` with ``policy`` until WIN, the budget, the deadline, or failure.

    Raises ``ValueError`` for ``max_actions < 1`` and ``RuntimeError`` if the
    game cannot even be reset.
    """
    if max_actions < 1:
        raise ValueError(f"max_actions must be >= 1, got {max_actions}")
    t0 = clock()
    frame = env.observation_space
    if frame is None or not list(frame.frame):
        frame = env.reset()
    if frame is None:
        raise RuntimeError(f"{game_id}: the environment did not reset")
    obs = observation_from_frame(frame)
    actions = 0
    failures = 0
    stop = "max_actions"
    while True:
        if obs.won:
            stop = "win"
            break
        if actions >= max_actions:
            stop = "max_actions"
            break
        if deadline is not None and clock() >= deadline:
            stop = "deadline"
            break
        action = policy.choose(obs)
        game_action, data = to_game_action(action)
        frame = env.step(game_action, data=data)
        actions += 1
        if frame is None:
            failures += 1
            if failures >= MAX_STEP_FAILURES:
                stop = "step_failures"
                break
            continue
        failures = 0
        obs = observation_from_frame(frame)
    return GameResult(
        game_id=game_id,
        actions=actions,
        levels_completed=obs.levels_completed,
        win_levels=obs.win_levels,
        phase=str(obs.phase),
        seconds=round(clock() - t0, 3),
        stop=stop,
        step_failures=failures,
    )
