"""Oracle world model: the real ARC-AGI-3 engine run locally.

ONLY for tests and benchmarks on the 25 public games. The competition's
private games are served through a gateway, so an agent never has engine
access there; nothing in the playing agent may import this module.

The oracle talks to the engine through the same public action interface the
gateway exposes (``perform_action``). It never reads hidden state.

Cloning uses ``copy.deepcopy`` on the engine object (about 12 ms on ls20),
so searches that branch heavily should clone only at branch points.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

from arcagent.core.types import (
    COORD_ACTION_ID,
    GRID_SIZE,
    RESET_ID,
    Action,
    Grid,
    Observation,
    Phase,
)

try:
    from arc_agi import Arcade, OperationMode  # pyright: ignore[reportMissingImports]
    from arcengine import ActionInput as _ActionInput  # pyright: ignore[reportMissingImports]
    from arcengine import GameAction as _GameAction  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover - the agent never imports this module
    raise ImportError("the oracle needs the arc_agi and arcengine wheels") from exc

_LOG = logging.getLogger(__name__)
# The engine enum's value is a tuple, so look members up by name.
_GAME_ACTIONS: dict[int, Any] = {
    i: getattr(_GameAction, "RESET" if i == 0 else f"ACTION{i}") for i in range(8)
}


def _to_grid(frame: list[Any]) -> Grid:
    """Last animation frame of a step as a padded (64, 64) int8 grid."""
    g = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
    if not frame:
        return g
    arr = np.asarray(frame[-1], dtype=np.int8)
    h, w = arr.shape[:2]
    g[:h, :w] = arr
    return g


def _phase(state: Any) -> Phase:
    name = getattr(state, "name", str(state))
    return Phase(name.split(".")[-1])


class OracleModel:
    """``WorldModel`` over a live local engine game object."""

    def __init__(self, game: Any, game_id: str) -> None:
        self._game = game
        self.game_id = game_id
        self._obs: Observation | None = None
        self.steps = 0

    @classmethod
    def load(cls, env_dir: str | Path, game: str, seed: int = 0) -> OracleModel:
        """Load a public game (4-char id such as ``ls20``) from ``env_dir`` and reset it."""
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(env_dir),
            logger=_LOG,
        )
        env = arcade.make(game, seed=seed, scorecard_id="oracle", save_recording=False)
        if env is None:
            raise FileNotFoundError(f"game {game!r} not found under {env_dir}")
        game_obj = getattr(env, "_game")  # noqa: B009  # the wrapper keeps the engine here
        model = cls(game_obj, game)
        model.step(Action(RESET_ID))
        model.steps = 0
        return model

    def _apply(self, action: Action) -> Observation:
        data: dict[str, int] = {}
        if action.id == COORD_ACTION_ID:
            data = {"x": action.x, "y": action.y}
        game_action = _GAME_ACTIONS[action.id]
        fd = self._game.perform_action(_ActionInput(id=game_action, data=data), raw=True)
        obs = Observation(
            grid=_to_grid(list(fd.frame)),
            phase=_phase(fd.state),
            levels_completed=int(fd.levels_completed),
            win_levels=int(fd.win_levels),
            available_actions=tuple(int(a) for a in fd.available_actions),
        )
        self._obs = obs
        return obs

    def observe(self) -> Observation:
        if self._obs is None:
            return self._apply(Action(RESET_ID))
        return self._obs

    def legal_actions(self) -> list[Action]:
        obs = self.observe()
        if obs.terminal:
            return [Action(RESET_ID)]
        acts: list[Action] = []
        for a in obs.available_actions:
            if a == RESET_ID:
                continue
            if a == COORD_ACTION_ID:
                # Coordinate actions are expanded on a coarse lattice; finer
                # targeting is the job of the object-aware move generator.
                acts.extend(
                    Action(COORD_ACTION_ID, x, y)
                    for y in range(4, GRID_SIZE, 8)
                    for x in range(4, GRID_SIZE, 8)
                )
            else:
                acts.append(Action(a))
        return acts

    def step(self, action: Action) -> Observation:
        self.steps += 1
        return self._apply(action)

    def clone(self) -> OracleModel:
        c = OracleModel(copy.deepcopy(self._game), self.game_id)
        c._obs = self._obs
        c.steps = self.steps
        return c

    def key(self) -> bytes:
        obs = self.observe()
        return obs.grid.tobytes() + bytes([obs.levels_completed, obs.phase == Phase.WIN])
