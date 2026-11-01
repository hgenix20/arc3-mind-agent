"""Progress or motion: does a move bring a goal closer, or just move things around?

``ProgressMonitor.observe`` takes each real frame's scene key and goal distances
(``Goal.distance``) and names the move:

- ``milestone``: a level-up, or a ``clear``/``count`` goal's distance fell below its best
  this life (things of the goal color gone, the kind of change that doesn't undo itself);
- ``progress``: some goal's distance fell below its best this life;
- ``regress``: the active goal's distance rose more than ``REGRESS_SLACK`` above its best
  this life;
- ``motion``: the scene key came back within the last ``MOTION_WINDOW`` frames and nothing
  got closer;
- ``none``: anything else (the first look at a goal only sets its bar).

``progressed()`` says whether this life has made progress or reached a milestone (the
agent refuses a level reset then). ``settled(goal)`` is true when the goal's distance is 0
or the last ``SETTLE_MOVES`` frames had the same scene key. ``reset_life`` (a RESET after a
game over) restarts the bests and the window; ``reset_level`` clears everything. Pure.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping
from typing import Literal

from arcagent.mind.goals import Goal

MOTION_WINDOW = 12  # frames a repeated scene key is looked for in
SETTLE_MOVES = 6  # frames in a row with one scene key that count as settled
REGRESS_SLACK = 2  # distance the active goal may lose before a move is a regress

Verdict = Literal["progress", "motion", "regress", "milestone", "none"]
_COUNTED = ("clear", "count")


class ProgressMonitor:
    """Per-life goal bests and a window of recent scene keys (module docstring)."""

    def __init__(self) -> None:
        self._best: dict[Goal, int] = {}
        self._first: dict[Goal, int] = {}
        self._recent: deque[Hashable] = deque(maxlen=MOTION_WINDOW)
        self._still = 0
        self._milestones = 0

    def reset_life(self) -> None:
        """A new attempt at the level: bests, window and milestones start over."""
        self._best.clear()
        self._first.clear()
        self._recent.clear()
        self._still = 0
        self._milestones = 0

    def reset_level(self) -> None:
        """A new level: everything starts over."""
        self.reset_life()

    def best(self, goal: Goal) -> int | None:
        """The goal's lowest distance this life, or None before it was measured."""
        return self._best.get(goal)

    def progressed(self) -> bool:
        """Whether this life got some goal closer than it started, or hit a milestone."""
        return self._milestones > 0 or any(self._best[g] < self._first[g] for g in self._best)

    def settled(self, goal: Goal) -> bool:
        """The goal's distance is 0, or the last ``SETTLE_MOVES`` frames had one scene key."""
        return self._best.get(goal) == 0 or self._still >= SETTLE_MOVES

    def observe(
        self,
        key: Hashable,
        distances: Mapping[Goal, int | None],
        *,
        active: Goal | None,
        level_up: bool = False,
    ) -> Verdict:
        """Name one move from its frame's scene key and goal distances."""
        repeated = key in self._recent
        self._still = self._still + 1 if self._recent and self._recent[-1] == key else 1
        self._recent.append(key)
        if level_up:
            self._milestones += 1
            return "milestone"
        closer: list[Goal] = []
        for g, d in distances.items():
            if d is None:
                continue
            if g not in self._best:
                self._best[g], self._first[g] = d, d
            elif d < self._best[g]:
                self._best[g] = d
                closer.append(g)
        if any(g.kind in _COUNTED for g in closer):
            self._milestones += 1
            return "milestone"
        if closer:
            return "progress"
        now = distances.get(active) if active is not None else None
        if active is not None and now is not None and now > self._best[active] + REGRESS_SLACK:
            return "regress"
        return "motion" if repeated else "none"
