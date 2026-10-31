"""Forks and the plan tree for one level instance, and the return route over known moves.

Intent (spec 4.7c, 3.6, and Kameron's answers in section 6): a fork is a plan or path that
could go two or more ways toward the same goal from the same state. The tree records each
fork with its state key, its paths in the order given (expected value order, best first),
each path's status, and the moves that led from the level start to the fork, kept because
Kameron asked for them. After a miss that matters the agent fails the branch it was on,
asks for the newest fork with an untried path, plans a return route to it over the
explorer's known transitions (the explorer's backtracking is the replan step), and takes
the next untried path. When no fork has an untried path, the agent replans from the root.

Contract:
- ``PlanTree`` holds at most ``MAX_FORKS`` forks for the current level instance; the
  oldest is dropped beyond that. One fork exists per (state key, goal id); opening the
  same pair again adds the paths that are new and changes nothing else.
- A fork holds at most ``MAX_PATHS`` paths. Empty paths and repeats are ignored.
- At most one path in the whole tree is active. Starting a path makes any other active
  path untried again. A failed or achieved status is final. An achieved path closes its
  fork: the goal is reached, so its sibling paths are no longer offered.
- No method raises on an unknown fork id, an index out of range, or a status change that
  the rules forbid: the request is ignored and ``misses`` grows by one, so the wiring
  agent can log it.
- ``reset_level`` drops every fork and the active path. Fork ids keep counting across
  resets so a game log never shows the same id for two forks. ``misses`` is a lifetime
  tally.
- ``return_route`` is a breadth-first search over the explorer ``Graph.edges`` mapping
  (``arcagent/agent/explore.py``), whose edges hold only key-changing, non-fatal
  transitions. It visits at most ``limit`` states and is deterministic: a state's moves
  are expanded in sorted key order.

This module logs nothing. The agent that wires it logs ``fork_open``, ``branch_failed``,
``return_route`` and ``replan_root`` (build plan section 4, WP5).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from arcagent.core.types import Action
from arcagent.mind.types import ActionKey

PathStatus = Literal["untried", "active", "failed", "achieved"]
FinalStatus = Literal["failed", "achieved"]

MAX_FORKS = 200  # forks kept per level instance; beyond it the oldest is dropped
MAX_PATHS = 16  # paths kept per fork; later ones are ignored
MAX_MOVES_HERE = 20_000  # the game's action cap; a longer moves_here is cut here
ROUTE_LIMIT = 20_000  # states return_route may visit unless the caller says otherwise
SUMMARY_LINES = 5  # briefing lines by default


@dataclass(slots=True)
class Fork:
    """One fork record (spec 4.7c rule 2 and 3).

    ``paths`` and ``status`` run in parallel: ``status[i]`` is the state of ``paths[i]``.
    ``moves_here`` is the move list from the level start to ``state_key`` as it was when
    the fork was first opened. ``created`` is the tree's running counter, so a higher value
    means a newer fork. Records come back from the tree by reference: read them, and change
    them only through the tree's methods, which keep the one-active-path rule.
    """

    id: str
    state_key: bytes
    goal_id: str
    paths: list[tuple[Action, ...]]
    status: list[PathStatus]
    moves_here: tuple[Action, ...]
    created: int


def _untried(fork: Fork) -> list[int]:
    """Indices of the paths still worth trying: none once the goal is achieved."""
    if "achieved" in fork.status:
        return []
    return [i for i, status in enumerate(fork.status) if status == "untried"]


def _is_open(fork: Fork) -> bool:
    """A fork still has something going or something left to try."""
    if "achieved" in fork.status:
        return False
    return any(status in ("untried", "active") for status in fork.status)


class PlanTree:
    """The forks of one level instance. See the module docstring for the rules.

    Attributes:
        misses: requests ignored because the fork id, the index or the status change was
            not valid. Never reset by the tree.
    """

    def __init__(self) -> None:
        self._forks: dict[str, Fork] = {}
        self._by_key: dict[tuple[bytes, str], str] = {}
        self._active: tuple[str, int] | None = None
        self._counter = 0
        self.misses = 0

    def open_fork(
        self,
        state_key: bytes,
        goal_id: str,
        paths: Sequence[Sequence[Action]],
        moves_here: Sequence[Action],
    ) -> str:
        """Record a fork at ``state_key`` toward ``goal_id`` and return its id.

        A fork that already exists for the pair gains the paths it does not hold yet and
        keeps its statuses and its first ``moves_here``. Paths are read in order until the
        fork holds ``MAX_PATHS``; empty paths and repeats are skipped. A new fork beyond
        ``MAX_FORKS`` drops the oldest fork, and the active path with it when it was there.
        """
        fork_id = self._by_key.get((state_key, goal_id))
        fork = self._forks.get(fork_id) if fork_id is not None else None
        if fork is None:
            self._counter += 1
            fork = Fork(
                id=f"f{self._counter}",
                state_key=state_key,
                goal_id=goal_id,
                paths=[],
                status=[],
                moves_here=tuple(moves_here)[:MAX_MOVES_HERE],
                created=self._counter,
            )
            self._forks[fork.id] = fork
            self._by_key[(state_key, goal_id)] = fork.id
            self._drop_oldest()
        for path in paths:
            if len(fork.paths) >= MAX_PATHS:
                break
            moves = tuple(path)
            if moves and moves not in fork.paths:
                fork.paths.append(moves)
                fork.status.append("untried")
        return fork.id

    def start_path(self, fork_id: str, index: int) -> None:
        """Mark one path active; any other active path in the tree goes back to untried.

        Starting the path that is already active changes nothing. A failed or achieved
        path cannot be started: that counts as a miss (spec 4.7c rule 7).
        """
        fork = self._lookup(fork_id, index)
        if fork is None:
            return
        if fork.status[index] == "active":
            return
        if fork.status[index] != "untried":
            self.misses += 1
            return
        if self._active is not None:
            other_id, other_index = self._active
            self._forks[other_id].status[other_index] = "untried"
        fork.status[index] = "active"
        self._active = (fork_id, index)

    def fail_branch(self, fork_id: str, index: int) -> None:
        """Mark one path failed for this level instance; it is never offered again."""
        self._finish(fork_id, index, "failed")

    def achieve(self, fork_id: str, index: int) -> None:
        """Mark one path achieved; the fork closes, so its siblings are not offered."""
        self._finish(fork_id, index, "achieved")

    def active(self) -> tuple[str, int] | None:
        """The (fork id, path index) being followed, or None."""
        return self._active

    def untried(self, fork_id: str) -> list[int]:
        """Indices of the fork's untried paths, in path order; [] for an unknown fork."""
        fork = self._forks.get(fork_id)
        if fork is None:
            self.misses += 1
            return []
        return _untried(fork)

    def newest_with_untried(self) -> Fork | None:
        """The fork with the highest ``created`` that still has an untried path (3.6)."""
        candidates = [fork for fork in self._forks.values() if _untried(fork)]
        return max(candidates, key=lambda fork: fork.created, default=None)

    def fork(self, fork_id: str) -> Fork | None:
        """The fork record by id, or None (and a miss) when unknown."""
        fork = self._forks.get(fork_id)
        if fork is None:
            self.misses += 1
        return fork

    def reset_level(self) -> None:
        """Drop every fork and the active path: a new level instance starts."""
        self._forks.clear()
        self._by_key.clear()
        self._active = None

    def summary(self, limit: int = SUMMARY_LINES) -> list[str]:
        """One briefing line per open fork, newest first, at most ``limit`` lines."""
        if limit <= 0:
            return []
        open_forks = sorted(
            (fork for fork in self._forks.values() if _is_open(fork)),
            key=lambda fork: fork.created,
            reverse=True,
        )
        return [
            f"fork {fork.id} at goal {fork.goal_id}: {len(_untried(fork))} untried of "
            f"{len(fork.paths)} paths, {len(fork.moves_here)} moves from start"
            for fork in open_forks[:limit]
        ]

    def counts(self) -> dict[str, int]:
        """Tallies over every fork: forks, untried, failed, achieved paths."""
        statuses = [status for fork in self._forks.values() for status in fork.status]
        return {
            "forks": len(self._forks),
            "untried": statuses.count("untried"),
            "failed": statuses.count("failed"),
            "achieved": statuses.count("achieved"),
        }

    def _lookup(self, fork_id: str, index: int) -> Fork | None:
        """The fork when the id is known and the index is in range; else a miss and None."""
        fork = self._forks.get(fork_id)
        if fork is None or not 0 <= index < len(fork.paths):
            self.misses += 1
            return None
        return fork

    def _finish(self, fork_id: str, index: int, status: FinalStatus) -> None:
        """Give a path a final status. Repeating it is a no-op; changing a final status
        is a miss. The active path is released when it is the one finished."""
        fork = self._lookup(fork_id, index)
        if fork is None:
            return
        current = fork.status[index]
        if current == status:
            return
        if current in ("failed", "achieved"):
            self.misses += 1
            return
        fork.status[index] = status
        if self._active == (fork_id, index):
            self._active = None

    def _drop_oldest(self) -> None:
        """Keep the tree within ``MAX_FORKS``; insertion order is creation order."""
        while len(self._forks) > MAX_FORKS:
            oldest_id = next(iter(self._forks))
            oldest = self._forks.pop(oldest_id)
            del self._by_key[(oldest.state_key, oldest.goal_id)]
            if self._active is not None and self._active[0] == oldest_id:
                self._active = None


def return_route(
    edges: Mapping[bytes, Mapping[ActionKey, bytes]],
    current: bytes,
    target: bytes,
    *,
    limit: int = ROUTE_LIMIT,
) -> list[Action] | None:
    """Shortest known action list from ``current`` to ``target`` (spec 4.7c rule 4).

    ``edges`` is the explorer graph's ``edges``: state key to {action key: next state key},
    where every action key is an ``Action.key()`` value. Returns [] when the two states are
    the same, None when ``target`` cannot be reached through at most ``limit`` distinct
    states (the start counts as one). Breadth-first, so the first route found is a
    shortest one; ties fall to the smaller action key because each state's moves are
    expanded in sorted order.
    """
    if current == target:
        return []
    prev: dict[bytes, tuple[bytes, ActionKey] | None] = {current: None}
    queue: deque[bytes] = deque([current])
    while queue:
        state = queue.popleft()
        moves = edges.get(state)
        if not moves:
            continue
        for akey in sorted(moves):
            nxt = moves[akey]
            if nxt in prev:
                continue
            if len(prev) >= limit:
                return None
            prev[nxt] = (state, akey)
            if nxt == target:
                return _walk_back(prev, nxt)
            queue.append(nxt)
    return None


def _walk_back(prev: dict[bytes, tuple[bytes, ActionKey] | None], end: bytes) -> list[Action]:
    """Follow the predecessor links from ``end`` to the start and rebuild the actions."""
    route: list[Action] = []
    state = end
    # Each state points to one discovered earlier, so the chain ends within len(prev) steps.
    for _ in range(len(prev)):
        link = prev[state]
        if link is None:
            break
        state, akey = link
        route.append(Action(*akey))
    route.reverse()
    return route
