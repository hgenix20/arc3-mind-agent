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
- Guidance: ``score(state)`` is the credited, direction-signed sum of features (lower
  is better). With no credit the policy is exactly ``ExplorerPolicy``. With credit,
  when the current state has no untried action, the frontier state with the lowest
  score is the target (path length breaks ties), not the nearest one.
"""

from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Callable

import numpy as np

from arcagent.agent.explore import ExplorerPolicy, Graph, StateKey
from arcagent.core.types import Action, Grid, Observation
from arcagent.mind.types import ActionKey

Feature = tuple[str, int, int]
MAX_SHARE = 0.25  # colors covering more of the board are background or walls
MAX_PAIR_COLORS = 10  # fewest-cell colors kept for gap pairs


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


class GuidedExplorer(ExplorerPolicy):
    """See the module docstring."""

    def __init__(self, seed: int = 0, **kwargs: int) -> None:
        super().__init__(seed, **kwargs)
        self.credit: dict[Feature, float] = {}
        self.feats: dict[StateKey, dict[Feature, int]] = {}

    def observe(self, obs: Observation) -> None:
        if self._seen:
            return
        before = self.last[0] if self.last is not None else None
        level = self.level
        super().observe(obs)
        if obs.levels_completed != level and level >= 0 and before in self.feats:
            self._credit_win(before)
            self.feats = {}
        key = self.key(obs)
        if key not in self.feats and len(self.feats) < self.max_states:
            self.feats[key] = features(self._masked(obs.grid))

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
        last = self.feats[before]
        for f, v in last.items():
            seen = [fs[f] for fs in self.feats.values() if f in fs]
            if len(set(seen)) < 2:
                continue
            lo, hi, mid = min(seen), max(seen), statistics.median(seen)
            span = hi - lo
            if v == lo and (f[0] == "gap" or v < mid):
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
        return sum(w * fs[f] for f, w in self.credit.items() if f in fs)

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
