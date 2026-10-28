"""Novelty-driven exploration over the observed state graph.

Intent: the fast, model-free layer of the agent. It spends actions to find
out what changes the frame, remembers every (state, action) outcome, and
steers toward states that still have untried actions. It has no notion of
the goal beyond the engine's own level counter.

Contract:
- ``ExplorerPolicy.choose`` is called once per executed action with the
  observation that resulted from the previous action (the play loop's
  contract), so the policy learns the outcome of its last action from the
  next observation.
- State identity is the grid bytes plus the level counter and the game-over flag,
  minus clock lines: a
  row or column within ``CLOCK_BAND`` cells of the edge that holds a move counter
  or budget bar. A line is one once it has changed alone ``CLOCK_TICKS`` times (a
  move whose whole change is at most ``TICK_CELLS`` cells, all in the band), or on
  at least ``CLOCK_RATE`` of the moves once ``CLOCK_WARMUP`` moves are seen. A grid
  that doesn't change outside them is a no-op for that (state, action) pair. When
  the clock lines change, the graph starts over.
- A level change clears the graph (layouts change) but keeps the action
  priors (mechanics usually carry across levels).
- RESET is chosen only when the game is terminal or offers nothing else.

A move that ended the game is never an edge, so no frontier path runs through a game
over; one that ended it ``DEAD_AFTER`` times from a state is dead there and never tried
from that state again (a single game over may just be the move budget running out). A
terminal state gets no candidates, so it is never a frontier (before, the game-over
screen counted as unexplored and paths led into it: dc22 died every 11 moves).

Untried moves of an idle kind (``idle``: at least ``IDLE_TRIES`` tries, under
``IDLE_RATE`` of them changed the screen) wait: before trying one, the explorer walks
to the nearest known state with an untried move of a live kind, and tries idle moves
only when no such state is reachable (dc22: 2,075 of 3,000 moves were no-ops; m0r0
level 1: 2,974 of 3,600, dead clicks tried in every new state while arrow moves waited
elsewhere).

Heuristics, in order:
1. Untried actions in the current state, ranked by how often that action
   kind changed the frame before (Laplace-smoothed; clicks are keyed by the
   color under the cursor as well as by the action id).
2. Otherwise the shortest known path to a state with untried actions; the
   path is abandoned as soon as a step lands somewhere unexpected.
3. Otherwise, when the graph holds anything, a new round: the graph is cleared, the
   clock lines count in the state key again, and exploration starts over from the
   current state (the screen doesn't show all of the game's state, so what looked
   exhausted may not be; with the clock in the key, history tells states apart).
4. Otherwise the action kind most likely to change the frame, at random.

Bounded: ``max_states`` caps the graph; beyond it the policy keeps playing
from priors only.
"""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from arcagent.core.types import COORD_ACTION_ID, RESET_ID, Action, Grid, Observation
from arcagent.mind.types import ActionKey
from arcagent.synth.trace import components

StateKey = bytes
MAX_TARGETS = 64
MAX_STATES = 20_000
UCB_C = 1.0
IDLE_TRIES = 20  # tries before a kind of move can count as idle
IDLE_RATE = 0.05  # an idle kind changed the screen on fewer than this share of tries
DEAD_AFTER = 2  # game overs a move must cause from one state before it counts as dead
CLOCK_WARMUP = 16  # moves seen before clock lines are judged
CLOCK_RATE = 0.6  # share of moves on which a clock line changes (bars seen: 0.64)
CLOCK_BAND = 3  # rows and columns this close to the edge may be clock lines
CLOCK_TICKS = 3  # lone small changes that make an edge line a clock line
TICK_CELLS = 2  # most cells a lone clock tick changes


def state_key(obs: Observation) -> StateKey:
    """Grid bytes, the level counter, and whether the game is over (a game-over frame can
    look exactly like a normal one: tu93)."""
    return obs.grid.tobytes() + bytes([obs.levels_completed & 0xFF, obs.terminal])


def click_candidates(grid: Grid, limit: int = MAX_TARGETS) -> list[tuple[int, int]]:
    """Click targets covering components of every size.

    One cell (nearest the centroid) per non-background component. When there
    are more than ``limit`` components, take evenly spaced ranks of the
    size-sorted list, so both large regions and small buttons are kept.
    Falls back to a 4x4 lattice when the grid has no components.
    """
    comps = components(grid)
    if not comps:
        step = grid.shape[0] // 4
        return [
            (x, y)
            for y in range(step // 2, grid.shape[0], step)
            for x in range(step // 2, grid.shape[1], step)
        ]
    comps.sort(key=len, reverse=True)
    if len(comps) > limit:
        idx = [round(i * (len(comps) - 1) / (limit - 1)) for i in range(limit)]
        comps = [comps[i] for i in idx]
    out: list[tuple[int, int]] = []
    for cells in comps:
        cy = sum(c[0] for c in cells) / len(cells)
        cx = sum(c[1] for c in cells) / len(cells)
        y, x = min(cells, key=lambda c: ((c[0] - cy) ** 2 + (c[1] - cx) ** 2, c))
        out.append((x, y))
    return out


@dataclass(slots=True)
class Graph:
    edges: dict[StateKey, dict[ActionKey, StateKey]] = field(default_factory=dict)
    noop: dict[StateKey, set[ActionKey]] = field(default_factory=dict)
    cands: dict[StateKey, list[Action]] = field(default_factory=dict)
    dead: dict[StateKey, set[ActionKey]] = field(default_factory=dict)
    deaths: Counter[tuple[StateKey, ActionKey]] = field(default_factory=Counter)

    def untried(self, key: StateKey) -> list[Action]:
        tried = set(self.edges.get(key, {})) | self.noop.get(key, set()) | self.dead.get(key, set())
        return [a for a in self.cands.get(key, []) if a.key() not in tried]

    def path_to_frontier(
        self, start: StateKey, wanted: Callable[[StateKey, Action], bool] | None = None
    ) -> list[Action]:
        """Shortest action path from ``start`` to another state with an untried action
        (one ``wanted`` accepts, when given); [] if none."""
        prev: dict[StateKey, tuple[StateKey, Action] | None] = {start: None}
        queue = deque([start])
        while queue:
            key = queue.popleft()
            untried = self.untried(key)
            if wanted is not None:
                untried = [a for a in untried if wanted(key, a)]
            if key != start and untried:
                path: list[Action] = []
                while prev[key] is not None:
                    pk, act = prev[key]  # type: ignore[misc]
                    path.append(act)
                    key = pk
                path.reverse()
                return path
            for akey, nxt in self.edges.get(key, {}).items():
                if nxt not in prev:
                    prev[nxt] = (key, Action(*akey))
                    queue.append(nxt)
        return []


class ExplorerPolicy:
    """See the module docstring. Deterministic for a seed and observation sequence."""

    def __init__(
        self, seed: int = 0, *, max_targets: int = MAX_TARGETS, max_states: int = MAX_STATES
    ) -> None:
        self.rng = random.Random(seed)
        self.max_targets = max_targets
        self.max_states = max_states
        self.graph = Graph()
        self.changes: Counter[tuple[int, int]] = Counter()
        self.tries: Counter[tuple[int, int]] = Counter()
        self.level = -1
        self.last: tuple[StateKey, Action, int] | None = None
        self.plan: list[Action] = []
        self.expected: StateKey | None = None
        self._seen = False
        self._prev_grid: Grid | None = None
        self._moves = 0
        self._line_changes: Counter[tuple[str, int]] = Counter()
        self._ticks: Counter[tuple[str, int]] = Counter()
        self.clock: frozenset[tuple[str, int]] = frozenset()
        self.rounds = 0  # times the graph was exhausted and started over on this level
        self._shape: tuple[int, int] | None = None

    def key(self, obs: Observation) -> StateKey:
        """``state_key`` with the clock lines blanked (kept once a round has run out)."""
        if not self.clock or self.rounds:
            return state_key(obs)
        g = obs.grid.copy()
        for axis, i in self.clock:
            if axis == "r":
                g[i, :] = 0
            else:
                g[:, i] = 0
        return g.tobytes() + bytes([obs.levels_completed & 0xFF, obs.terminal])

    def _watch_clock(self, obs: Observation) -> bool:
        """Count edge lines changed since the last move; True when the clock lines change."""
        prev, self._prev_grid = self._prev_grid, obs.grid
        if self.last is None or prev is None or prev.shape != obs.grid.shape:
            return False
        self._moves += 1
        diff = prev != obs.grid
        h, w = diff.shape
        band_r = [*range(min(CLOCK_BAND, h)), *range(max(h - CLOCK_BAND, 0), h)]
        band_c = [*range(min(CLOCK_BAND, w)), *range(max(w - CLOCK_BAND, 0), w)]
        for i in set(band_r):
            self._line_changes["r", i] += bool(diff[i, :].any())
        for j in set(band_c):
            self._line_changes["c", j] += bool(diff[:, j].any())
        ys, xs = np.nonzero(diff)
        in_band = [
            (y, x)
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
            if y in band_r or x in band_c
        ]
        if 0 < len(ys) <= TICK_CELLS and len(in_band) == len(ys):
            for y, x in in_band:
                self._ticks["r" if y in band_r else "c", y if y in band_r else x] += 1
        clock = frozenset(k for k, n in self._ticks.items() if n >= CLOCK_TICKS)
        if self._moves >= CLOCK_WARMUP:
            rate = CLOCK_RATE * self._moves
            clock |= {k for k, n in self._line_changes.items() if n >= rate}
        if clock == self.clock:
            return False
        self.clock = clock
        return True

    # -- priors -------------------------------------------------------------
    @staticmethod
    def _kind(action: Action, grid: Grid) -> tuple[int, int]:
        if action.id == COORD_ACTION_ID:
            return (COORD_ACTION_ID, int(grid[action.y, action.x]))
        return (action.id, -1)

    def idle(self, kind: tuple[int, int]) -> bool:
        """A kind that almost never changes the screen: ``IDLE_TRIES`` tries or more, under
        ``IDLE_RATE`` of them changed it. Its untried moves wait until nothing else is left."""
        n = self.tries[kind]
        return n >= IDLE_TRIES and self.changes[kind] < IDLE_RATE * n

    def change_rate(self, kind: tuple[int, int]) -> float:
        """Laplace-smoothed fraction of tries of this kind that changed the frame."""
        return (self.changes[kind] + 1.0) / (self.tries[kind] + 2.0)

    # -- bookkeeping ----------------------------------------------------------
    def _record(self, key: StateKey, died: bool = False) -> None:
        if self.last is None:
            return
        prev_key, action, kind_color = self.last
        kind = (action.id, kind_color)
        self.tries[kind] += 1
        if died:
            # Never a path through a game over; a move that ended the game twice from
            # here is never tried from here again (once may be the move budget).
            self.graph.deaths[prev_key, action.key()] += 1
            if self.graph.deaths[prev_key, action.key()] >= DEAD_AFTER:
                self.graph.dead.setdefault(prev_key, set()).add(action.key())
            return
        if key != prev_key:
            self.changes[kind] += 1
            if len(self.graph.edges) < self.max_states:
                self.graph.edges.setdefault(prev_key, {})[action.key()] = key
        else:
            self.graph.noop.setdefault(prev_key, set()).add(action.key())

    def _candidates(self, obs: Observation) -> list[Action]:
        out: list[Action] = []
        for a in obs.available_actions:
            if a == RESET_ID:
                continue
            if a == COORD_ACTION_ID:
                out.extend(Action(a, x, y) for x, y in click_candidates(obs.grid, self.max_targets))
            else:
                out.append(Action(a))
        return out

    # -- choice -----------------------------------------------------------------
    def observe(self, obs: Observation) -> None:
        """Learn the outcome of the last action from ``obs`` without choosing.

        Idempotent until an action is chosen or noted, so a caller may observe
        first and then call ``choose`` with the same ``obs`` without double
        counting.
        A level change clears the graph; a surprise clears the frontier path.
        """
        if self._seen:
            return
        self._seen = True
        self._shape = (int(obs.grid.shape[0]), int(obs.grid.shape[1]))
        if self._watch_clock(obs):
            self.graph = Graph()
            self.plan = []
            self.last = None
        key = self.key(obs)
        self._record(key, died=obs.terminal and not obs.won)
        self.last = None
        if obs.levels_completed != self.level:
            self.level = obs.levels_completed
            self.graph = Graph()
            self.plan = []
            self.rounds = 0
        if self.plan and key != self.expected:
            self.plan = []
        if key not in self.graph.cands and len(self.graph.cands) < self.max_states:
            self.graph.cands[key] = [] if obs.terminal else self._candidates(obs)

    def note(self, obs: Observation, action: Action) -> None:
        """Record that ``action`` was taken from ``obs`` (possibly by another branch).

        Call after ``observe(obs)``; a prior ``choose`` of the same action
        makes it a no-op. A RESET leaves nothing to learn from. An action
        other than the explorer's own pending frontier step invalidates that
        path.
        """
        self._seen = False
        if self.last is not None and self.last[1] == action:
            return
        self.plan = []
        if action.id == RESET_ID or obs.terminal:
            self.last = None
            return
        self.last = (self.key(obs), action, self._kind(action, obs.grid)[1])

    def choose(self, obs: Observation, exclude: frozenset[ActionKey] = frozenset()) -> Action:
        """Pick the next action, skipping ``exclude`` unless nothing else is left."""
        self.observe(obs)
        key = self.key(obs)
        if obs.terminal or all(a == RESET_ID for a in obs.available_actions):
            self.last = None
            self.plan = []
            self._seen = False
            return Action(RESET_ID)
        action = self._pick(key, obs, exclude)
        self.last = (key, action, self._kind(action, obs.grid)[1])
        self._seen = False
        return action

    def _live_path(self, key: StateKey) -> list[Action]:
        """Path to the nearest other state with an untried move of a live (not idle) kind."""
        if self._shape is None:
            return []
        h, w = self._shape

        def live(k: StateKey, a: Action) -> bool:
            if a.id != COORD_ACTION_ID:
                return not self.idle((a.id, -1))
            if len(k) != h * w + 2:
                return True
            color = int(np.frombuffer(k, dtype=np.int8, count=h * w)[a.y * w + a.x])
            return not self.idle((a.id, color))

        return self.graph.path_to_frontier(key, live)

    def ucb(self, kind: tuple[int, int]) -> float:
        """Change rate plus an exploration bonus that shrinks with the kind's own tries.

        Without the bonus the policy locks onto the first kind that always
        changes the frame (a moving sprite, a counter) and never tries the rest.
        """
        total = sum(self.tries.values()) + 1
        return self.change_rate(kind) + UCB_C * math.sqrt(math.log(total) / (self.tries[kind] + 1))

    def _pick(
        self, key: StateKey, obs: Observation, exclude: frozenset[ActionKey] = frozenset()
    ) -> Action:
        untried = [a for a in self.graph.untried(key) if a.key() not in exclude]
        live = [a for a in untried if not self.idle(self._kind(a, obs.grid))]
        if live or (untried and not self.plan and not self._live_path(key)):
            self.plan = []
            return max(
                live or untried,
                key=lambda a: (self.ucb(self._kind(a, obs.grid)), self.rng.random()),
            )
        if not self.plan:
            self.plan = self._live_path(key) or self.graph.path_to_frontier(key)
        if self.plan and self.plan[0].key() in exclude:
            self.plan = []
        if self.plan:
            action = self.plan.pop(0)
            self.expected = self.graph.edges.get(key, {}).get(action.key())
            return action
        if self.graph.edges and not exclude:
            # Nothing reachable is untried, yet the level isn't won: the screen doesn't
            # show all the game's state. Start a new round on a fresh graph that keys on
            # the clock lines too, so the same screen later in a life counts as new (g50t:
            # 43 screens, then random moves that kept undoing progress).
            self.graph = Graph()
            self.graph.cands[key] = self._candidates(obs)
            self.rounds += 1
            return self._pick(key, obs, exclude)
        cands = self.graph.cands.get(key) or self._candidates(obs)
        if not cands:
            return Action(RESET_ID)
        cands = [a for a in cands if a.key() not in exclude] or cands
        weights = [self.change_rate(self._kind(a, obs.grid)) for a in cands]
        return self.rng.choices(cands, weights=weights, k=1)[0]
