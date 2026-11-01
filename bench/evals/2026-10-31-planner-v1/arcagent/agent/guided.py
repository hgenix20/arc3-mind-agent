"""The explorer, guided on later levels by what changed when an earlier level was won.

Intent: ``ExplorerPolicy`` finishes many first levels by coverage and almost no later
ones, because a later level is bigger and blind coverage runs out of moves. A level-up
says something about the goal: some measurement of the screen was at its most extreme
on the frame the winning move was made from. This policy credits those measurements
and, on later levels, sends the explorer to the known frontier state that moves them
the same way first. No game is named and nothing is memorized: the measurements are
color gaps and color counts on whatever screen the game draws.

Contract (on top of ``ExplorerPolicy``'s):
- Features of a state: for each pair of colors present (each covering at most
  ``MAX_SHARE`` of the board, background and clock lines excluded) the smallest
  Chebyshev gap between their cells, ``("gap", a, b)`` with ``a < b``; and each
  color's cell count, ``("count", c, -1)``. Computed once per new state.
- On a level-up, with ``before`` the state the winning move was made from: a gap
  whose value at ``before`` is the smallest seen on the level, a count at its smallest
  or largest, gets credit when the feature took at least two values on the level.
  Credit is ``direction`` (+1: smaller wins, -1: larger wins) times a weight that
  grows with how far ``before`` stands from the level's median. Credits add up over
  levels.
- Reach features: the mover's colors are those whose cell count stays the same while
  their cells change place, on at least ``MOVER_SHARE`` of the moves that changed the
  screen (at least ``MOVER_MIN_MOVES`` of them, counted over the whole game). A cell's
  usual color is its most common color over the distinct states seen on the level. The
  mover is the mover-colored cells that don't show their usual color (a floor keeps its
  count as an avatar crosses it, so it votes as a mover too, but it is a floor cell's
  usual color), keeping those connected to the rarest mover color. ``("reach", c, -1)``
  is the Chebyshev gap from the mover to the nearest cell of color ``c`` outside it.
  Why: an avatar often carries the color of its goal (tu93's exit tile is the avatar's
  color), so a plain color gap is zero on every frame and says nothing. Reach features are read
  from each state's own grid when needed (the key holds it), with the mover colors and
  usual colors as they stand then, and credited at a win like the rest.
- Guidance: ``score(state)`` is the credited, direction-signed sum of features (lower
  is better). With no credit the policy is exactly ``ExplorerPolicy``. With credit,
  when the current state has no untried action, the frontier state with the lowest
  score is the target (path length breaks ties), not the nearest one.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from collections.abc import Callable

import numpy as np

from arcagent.agent.explore import ExplorerPolicy, Graph, StateKey
from arcagent.core.types import Action, Grid, Observation
from arcagent.mind.types import ActionKey

Feature = tuple[str, int, int]
MAX_SHARE = 0.25  # colors covering more of the board are background or walls
MAX_PAIR_COLORS = 10  # fewest-cell colors kept for gap pairs
MOVER_SHARE = 0.5
MOVER_MIN_MOVES = 5
MAX_CREDIT_STATES = 2000  # states sampled for a reach feature's spread at a win


def chessboard_distance(mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance of every cell to the nearest True cell of ``mask`` (by dilation)."""
    h, w = mask.shape
    far = h + w
    dist = np.where(mask, 0, far)
    reached = mask.copy()
    step = 0
    while not reached.all() and step < far:
        step += 1
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= grown[:, :-1].copy()
        grown[:, :-1] |= grown[:, 1:].copy()
        dist[grown & ~reached] = step
        reached = grown
    return dist


def features(grid: Grid) -> dict[Feature, int]:
    """Color gaps and counts of ``grid`` (see the module docstring)."""
    vals, counts = np.unique(grid, return_counts=True)
    limit = MAX_SHARE * grid.size
    colors = [(int(n), int(v)) for v, n in zip(vals, counts, strict=True) if v >= 0 and n <= limit]
    out: dict[Feature, int] = {("count", c, -1): n for n, c in colors}
    keep = [c for _, c in sorted(colors)[:MAX_PAIR_COLORS]]
    dist: dict[int, np.ndarray] = {}
    for c in keep:
        dist[c] = chessboard_distance(grid == c)
    for i, a in enumerate(keep):
        for b in keep[i + 1 :]:
            out["gap", min(a, b), max(a, b)] = int(dist[b][grid == a].min())
    return out


def grown_within(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """The cells of ``allowed`` 8-connected to ``seed`` through ``allowed``."""
    reached = seed & allowed
    while True:
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= grown[:, :-1].copy()
        grown[:, :-1] |= grown[:, 1:].copy()
        grown &= allowed
        if np.array_equal(grown, reached):
            return reached
        reached = grown


def reach_features(grid: Grid, mover: np.ndarray) -> dict[Feature, int]:
    """``("reach", c, -1)``: the Chebyshev gap from the ``mover`` cells to color ``c``."""
    if not mover.any():
        return {}
    dist = chessboard_distance(mover)
    vals, counts = np.unique(grid[~mover], return_counts=True)
    limit = MAX_SHARE * grid.size
    out: dict[Feature, int] = {}
    for v, n in zip(vals, counts, strict=True):
        if v >= 0 and n <= limit:
            out["reach", int(v), -1] = int(dist[(grid == v) & ~mover].min())
    return out


class GuidedExplorer(ExplorerPolicy):
    """See the module docstring."""

    def __init__(self, seed: int = 0, **kwargs: int) -> None:
        super().__init__(seed, **kwargs)
        self.credit: dict[Feature, float] = {}
        self.feats: dict[StateKey, dict[Feature, int]] = {}
        self.mover_votes: Counter[int] = Counter()
        self.moved = 0  # moves that changed the screen, over the game
        self.tally: np.ndarray | None = None  # per cell, states showing each color
        self.usual: Grid | None = None
        self._before: Grid | None = None
        self._shape: tuple[int, int] | None = None
        self._reach: dict[StateKey, dict[Feature, int]] = {}

    def observe(self, obs: Observation) -> None:
        if self._seen:
            return
        before = self.last[0] if self.last is not None else None
        level = self.level
        super().observe(obs)
        grid = self._masked(obs.grid)
        if obs.levels_completed != level and level >= 0 and before in self.feats:
            self._credit_win(before)
            self.feats = {}
        if obs.levels_completed != level or self.usual is None or self.usual.shape != grid.shape:
            self.tally, self.usual, self._before = None, grid, None
            self._shape = (int(grid.shape[0]), int(grid.shape[1]))
            self._reach = {}
        elif not obs.terminal:
            self._watch_mover(grid)
        key = self.key(obs)
        if key not in self.feats and len(self.feats) < self.max_states:
            self.feats[key] = features(grid)
            self._tally(grid)

    def movers(self) -> set[int]:
        """The colors that move as one thing (see the module docstring)."""
        if self.moved < MOVER_MIN_MOVES:
            return set()
        return {c for c, n in self.mover_votes.items() if n >= MOVER_SHARE * self.moved}

    def _tally(self, grid: Grid) -> None:
        if self.tally is None or self.tally.shape[:2] != grid.shape:
            self.tally = np.zeros((*grid.shape, 16), np.int32)
        ys, xs = np.indices(grid.shape)
        self.tally[ys, xs, grid.astype(np.int64) & 15] += 1
        usual = self.tally.argmax(axis=2).astype(grid.dtype)
        if self.usual is None or not np.array_equal(usual, self.usual):
            self.usual, self._reach = usual, {}

    def _watch_mover(self, grid: Grid) -> None:
        prev, self._before = self._before, grid
        if prev is None or prev.shape != grid.shape:
            return
        changed = prev != grid
        if not changed.any():
            return
        self.moved += 1
        old, new = set(np.unique(prev[changed]).tolist()), set(np.unique(grid[changed]).tolist())
        for c in old & new:
            if c >= 0 and (prev == c).sum() == (grid == c).sum():
                self.mover_votes[int(c)] += 1

    def mover_mask(self, grid: Grid) -> np.ndarray:
        """The mover's cells in ``grid``: mover-colored cells off their usual color,
        connected to the rarest mover color (where the mover started also shows a color
        off its usual one, the floor's, until the tally learns that cell)."""
        colors = self.movers()
        if not colors or self.usual is None or self.usual.shape != grid.shape:
            return np.zeros(grid.shape, bool)
        off = np.isin(grid, list(colors)) & (grid != self.usual)
        present = [c for c in colors if (off & (grid == c)).any()]
        if not present:
            return off
        anchor = min(present, key=lambda c: (int((grid == c).sum()), c))
        return grown_within(off & (grid == anchor), off)

    def reach(self, key: StateKey) -> dict[Feature, int]:
        """Reach features of the state ``key`` names, read from its grid (cached)."""
        got = self._reach.get(key)
        if got is not None:
            return got
        if self._shape is None or len(key) != self._shape[0] * self._shape[1] + 2:
            return {}
        h, w = self._shape
        grid = np.frombuffer(key[: h * w], dtype=np.int8).reshape(h, w)
        got = reach_features(grid, self.mover_mask(grid))
        self._reach[key] = got
        return got

    def _masked(self, grid: Grid) -> Grid:
        if not self.clock:
            return grid
        g = grid.copy()
        bg = int(np.bincount((g.ravel() & 15).astype(np.int64)).argmax())
        for axis, i in self.clock:
            if axis == "r":
                g[i, :] = bg
            else:
                g[:, i] = bg
        return g

    def _credit_win(self, before: StateKey) -> None:
        keys = list(self.feats)
        if len(keys) > MAX_CREDIT_STATES:
            keys = [*self.rng.sample(keys, MAX_CREDIT_STATES), before]
        reach = {k: self.reach(k) for k in keys}
        self._credit(self.feats[before], list(self.feats.values()))
        self._credit(reach[before], list(reach.values()))

    def _credit(self, last: dict[Feature, int], states: list[dict[Feature, int]]) -> None:
        for f, v in last.items():
            seen = [fs[f] for fs in states if f in fs]
            if len(set(seen)) < 2:
                continue
            lo, hi, mid = min(seen), max(seen), statistics.median(seen)
            span = hi - lo
            if v == lo and (f[0] != "count" or v < mid):
                self.credit[f] = self.credit.get(f, 0.0) + (mid - v) / span
            elif f[0] == "count" and v == hi and v > mid:
                self.credit[f] = self.credit.get(f, 0.0) - (v - mid) / span

    def guiding(self) -> bool:
        """True when ``score`` means anything (some credit)."""
        return bool(self.credit)

    def score(self, key: StateKey) -> float:
        fs = self.feats.get(key)
        if fs is None:
            return 0.0
        reach = self.reach(key) if any(f[0] == "reach" for f in self.credit) else {}
        both = {**fs, **reach}
        return sum(w * both[f] for f, w in self.credit.items() if f in both)

    def _pick(
        self, key: StateKey, obs: Observation, exclude: frozenset[ActionKey] = frozenset()
    ) -> Action:
        stuck = not [a for a in self.graph.untried(key) if a.key() not in exclude]
        if self.guiding() and stuck and not self.plan:
            self.plan = best_frontier_path(self.graph, key, self.score)
        return super()._pick(key, obs, exclude)


def best_frontier_path(
    graph: Graph, start: StateKey, score: Callable[[StateKey], float]
) -> list[Action]:
    """Path from ``start`` to the frontier state with the lowest ``score`` (then nearest)."""
    prev: dict[StateKey, tuple[StateKey, Action] | None] = {start: None}
    depth = {start: 0}
    queue = deque([start])
    best: tuple[float, int, StateKey] | None = None
    while queue:
        k = queue.popleft()
        if k != start and graph.untried(k):
            cand = (score(k), depth[k], k)
            if best is None or cand < best:
                best = cand
        for akey, nxt in graph.edges.get(k, {}).items():
            if nxt not in prev:
                prev[nxt] = (k, Action(*akey))
                depth[nxt] = depth[k] + 1
                queue.append(nxt)
    if best is None:
        return []
    k = best[2]
    path: list[Action] = []
    while (p := prev[k]) is not None:
        k, act = p
        path.append(act)
    path.reverse()
    return path
