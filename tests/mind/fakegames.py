"""Tiny deterministic games that hand ``Observation`` values straight to an agent.

No toolkit and no environment: each game keeps its own state and renders a 64x64
int8 grid. ``play`` drives an agent through ``choose`` the way ``play_game`` does
(stop on WIN, on the step budget, or when ``until`` says so).

``FakeGame``: background 0, a one-cell player (color 1), a one-cell target (4),
walls (8) and lava (9). Actions 1 to 4 move the player up, down, left and right by
one cell; the grid edge, walls and the target block. Moving onto lava ends the game
with the player drawn on the lava cell. The level completes when the player is
4-adjacent to the target; the next level shifts every object by ``shift``, so the
whole screen changes at once.

``FakeClickGame``: ``objects`` 2x2 blocks of color 3 on background 0, clicks only.
Clicking a block toggles its color between 3 and 5; the level completes when every
block is color 5, and the next level shifts every block.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from arcagent.core.types import GRID_SIZE, RESET_ID, Action, Grid, Observation, Phase

Cell = tuple[int, int]

PLAYER, TARGET, WALL, LAVA = 1, 4, 8, 9
OFF, ON = 3, 5
_MOVES: dict[int, Cell] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Agent(Protocol):
    def choose(self, obs: Observation) -> Action: ...


class Game(Protocol):
    def observe(self) -> Observation: ...

    def step(self, action: Action) -> Observation: ...


def _blank() -> Grid:
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)


def _shift(cells: Sequence[Cell], by: Cell) -> list[Cell]:
    return [(x + by[0], y + by[1]) for x, y in cells]


class FakeGame:
    """The move-to-the-target game; see the module docstring."""

    def __init__(
        self,
        player: Cell,
        target: Cell,
        walls: Sequence[Cell],
        lava: Sequence[Cell],
        *,
        win_levels: int = 2,
        shift: Cell = (20, 30),
    ) -> None:
        self._start = (player, target, list(walls), list(lava))
        self._shift = shift
        self.win_levels = win_levels
        self.levels = 0
        self.game_overs = 0
        self.level_up_steps: list[int] = []
        self.steps = 0
        self.phase = Phase.NOT_FINISHED
        self._load()

    def _load(self) -> None:
        player, target, walls, lava = self._start
        by = (self._shift[0] * self.levels, self._shift[1] * self.levels)
        self.player = _shift([player], by)[0]
        self.target = _shift([target], by)[0]
        self.walls = set(_shift(walls, by))
        self.lava = set(_shift(lava, by))

    def observe(self) -> Observation:
        g = _blank()
        for x, y in self.walls:
            g[y, x] = WALL
        for x, y in self.lava:
            g[y, x] = LAVA
        g[self.target[1], self.target[0]] = TARGET
        g[self.player[1], self.player[0]] = PLAYER
        return Observation(g, self.phase, self.levels, self.win_levels, (0, 1, 2, 3, 4))

    def step(self, action: Action) -> Observation:
        self.steps += 1
        if action.id == RESET_ID:
            self.phase = Phase.NOT_FINISHED
            self._load()
            return self.observe()
        if self.phase != Phase.NOT_FINISHED or action.id not in _MOVES:
            return self.observe()
        dx, dy = _MOVES[action.id]
        x, y = self.player[0] + dx, self.player[1] + dy
        inside = 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE
        if not inside or (x, y) in self.walls or (x, y) == self.target:
            return self.observe()
        self.player = (x, y)
        if (x, y) in self.lava:
            self.phase = Phase.GAME_OVER
            self.game_overs += 1
            return self.observe()
        if abs(x - self.target[0]) + abs(y - self.target[1]) == 1:
            self.levels += 1
            self.level_up_steps.append(self.steps)
            if self.levels >= self.win_levels:
                self.phase = Phase.WIN
            else:
                self._load()
        return self.observe()


class FakeClickGame:
    """The click-every-block game; see the module docstring."""

    def __init__(self, objects: int, *, win_levels: int = 2, shift: Cell = (3, 20)) -> None:
        self._starts: list[Cell] = [(4 + 10 * i, 6) for i in range(objects)]
        self._shift = shift
        self.win_levels = win_levels
        self.levels = 0
        self.level_up_steps: list[int] = []
        self.clicks: list[int] = []  # index of the block each click hit, -1 for a miss
        self.steps = 0
        self.phase = Phase.NOT_FINISHED
        self._load()

    def _load(self) -> None:
        by = (self._shift[0] * self.levels, self._shift[1] * self.levels)
        self.blocks = _shift(self._starts, by)
        self.colors = [OFF] * len(self.blocks)

    def _hit(self, x: int, y: int) -> int:
        for i, (bx, by) in enumerate(self.blocks):
            if bx <= x < bx + 2 and by <= y < by + 2:
                return i
        return -1

    def observe(self) -> Observation:
        g = _blank()
        for (bx, by), color in zip(self.blocks, self.colors, strict=True):
            g[by : by + 2, bx : bx + 2] = color
        return Observation(g, self.phase, self.levels, self.win_levels, (0, 6))

    def step(self, action: Action) -> Observation:
        self.steps += 1
        if action.id == RESET_ID:
            self.phase = Phase.NOT_FINISHED
            self._load()
            return self.observe()
        if self.phase != Phase.NOT_FINISHED or action.id != 6:
            return self.observe()
        i = self._hit(action.x, action.y)
        self.clicks.append(i)
        if i < 0:
            return self.observe()
        self.colors[i] = ON if self.colors[i] == OFF else OFF
        if all(c == ON for c in self.colors):
            self.levels += 1
            self.level_up_steps.append(self.steps)
            if self.levels >= self.win_levels:
                self.phase = Phase.WIN
            else:
                self._load()
        return self.observe()


def play(
    game: Game, agent: Agent, max_steps: int, until: Callable[[], bool] | None = None
) -> list[Action]:
    """Step ``agent`` on ``game``; stop on WIN, after ``max_steps``, or once ``until()``."""
    obs = game.observe()
    actions: list[Action] = []
    while len(actions) < max_steps and not obs.won and not (until is not None and until()):
        action = agent.choose(obs)
        actions.append(action)
        obs = game.step(action)
    return actions
