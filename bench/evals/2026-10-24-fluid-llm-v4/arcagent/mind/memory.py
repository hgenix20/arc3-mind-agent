"""Working memory (the current level) and the per-game store (what sleep keeps).

Contract:
- Timers count actions, not seconds. ``tick`` is called once per action and
  decrements every timed item. An item put with ttl N is still readable after
  N-1 ticks and moves to ``store.cold`` on the Nth tick.
- ``recall`` is the repetition rule: a cold item comes back with twice its last
  ttl, so a fact that keeps being needed lives longer each time.
- ``clear_level`` drops live timed items without moving them to cold: they
  belong to the finished level and must not leak into the next one.
- Values are opaque objects to this module. Inhibition stores a set of action
  keys under ``veto:<frame>``.

Failure modes: ``put`` raises ``ValueError`` for ttl < 1. Nothing else raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arcagent.core.types import Grid
from arcagent.mind.types import Program, Step

DEFAULT_TTL = 200


@dataclass
class GameStore:
    """Everything sleep consolidates for one game. Plain mutable state."""

    episodes: list[list[Step]] = field(default_factory=lambda: [])
    # program, hits, trials
    procedural: dict[str, tuple[Program, int, int]] = field(default_factory=lambda: {})
    held: dict[str, Program] = field(default_factory=lambda: {})
    # (before, after) of each level-up
    goal_evidence: list[tuple[Grid, Grid]] = field(default_factory=lambda: [])
    # key -> (value, last ttl)
    cold: dict[str, tuple[object, int]] = field(default_factory=lambda: {})


@dataclass
class _Item:
    value: object
    ttl: int
    remaining: int


class WorkingMemory:
    """The current level's episode plus timed items."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._items: dict[str, _Item] = {}

    def put(self, key: str, value: object, ttl: int) -> None:
        """Store ``value`` for ``ttl`` actions, replacing any live item at ``key``."""
        if ttl < 1:
            raise ValueError(f"ttl must be >= 1, got {ttl}")
        self._items[key] = _Item(value, ttl, ttl)

    def get(self, key: str) -> object | None:
        """Return the live value at ``key``, or None (no ttl change)."""
        item = self._items.get(key)
        return None if item is None else item.value

    def tick(self, store: GameStore) -> list[str]:
        """Advance one action; move expired items to ``store.cold``, return their keys."""
        expired: list[str] = []
        for key, item in self._items.items():
            item.remaining -= 1
            if item.remaining <= 0:
                expired.append(key)
        for key in expired:
            item = self._items.pop(key)
            store.cold[key] = (item.value, item.ttl)
        return expired

    def recall(self, key: str, store: GameStore) -> object | None:
        """Return a live or cold item; a cold one is restored with double its ttl."""
        item = self._items.get(key)
        if item is not None:
            return item.value
        cold = store.cold.pop(key, None)
        if cold is None:
            return None
        value, last_ttl = cold
        self.put(key, value, 2 * last_ttl)
        return value

    def clear_level(self) -> list[Step]:
        """Return and empty the episode; drop live timed items (not to cold)."""
        out = self.steps
        self.steps = []
        self._items.clear()
        return out


def all_steps(store: GameStore, wm: WorkingMemory) -> list[Step]:
    """Every observed step: finished episodes in order, then the current level."""
    return [s for episode in store.episodes for s in episode] + list(wm.steps)
