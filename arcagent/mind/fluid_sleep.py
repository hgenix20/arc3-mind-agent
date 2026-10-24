"""Fluid sleep: fact bookkeeping at every level-up and at game end.

``fluid_sleep`` runs three passes over ``FactMemory``, in this order, and reports what
each pass changed. Callers run it before ``FactMemory.new_level`` (which clears
``missed``).

1. Winning-facts check. ``used`` holds the fact keys the executed predictions of the
   level that just ended relied on. A used fact that took a miss on this level
   (``key in mem.missed``) and is solid drops to thin: ``hits = min(hits, THIN_BELOW -
   1)``. Only facts whose ``hits`` actually fell are reported as ``demoted``; a thin
   fact can't drop further. Facts with no misses keep their counts.
2. Consolidation. Facts that share a label and context, have identical effects, and
   whose subjects are sibling sigs (same shape, any color) merge into the one with the
   most hits (ties to the lowest key). It keeps its own counts; the others are deleted
   and reported as ``merged``. A deleted subject then predicts by borrowing the kept
   fact through ``FactMemory.lookup``, so only facts whose lookup borrows are eligible:
   a simple action's label (clicks count borrowed predictions as open questions in
   ``tester``), context ``any`` (blocked, pushed and touch facts are read by exact key),
   not broken (broken facts are never lent), and no move, slide or teleport effect
   (``goals.mover_sigs`` reads subjects from facts, so deleting one would drop a mover).
3. Drop. A thin fact with zero hits, created on ``level`` and not in ``used``, is
   deleted and reported as ``dropped``. ``record`` seeds facts with one hit, so this
   only removes facts whose hits were cleared by other code.

Bounded by ``time_s`` of ``clock``, checked before every fact (and every merge group):
once it has passed, the remaining work is skipped and ``truncated`` is True. Pure apart
from mutating ``mem`` and reading the clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from arcagent.mind.facts import THIN_BELOW, Effect, FactKey, FactMemory

SLEEP_MAX_S = 0.5  # wall-clock cap per sleep call
_MOVE_KINDS = frozenset({"move", "slide", "teleport"})

__all__ = ["SLEEP_MAX_S", "FluidSleepReport", "fluid_sleep"]


@dataclass(frozen=True, slots=True)
class FluidSleepReport:
    """What one sleep changed, its wall-clock seconds and whether it ran out of time."""

    demoted: tuple[FactKey, ...]
    merged: tuple[FactKey, ...]
    dropped: tuple[FactKey, ...]
    seconds: float
    truncated: bool


class _OutOfTimeError(Exception):
    """Raised inside a pass once the time cap has passed."""


def _mergeable(key: FactKey, effects: tuple[Effect, ...]) -> str | None:
    """The subject's shape when the fact may merge with its siblings, else None."""
    label, subject, context = key
    if context != "any" or not label.startswith("A") or label.startswith("A6"):
        return None
    if any(e.kind in _MOVE_KINDS for e in effects):
        return None
    color, sep, shape = subject.partition(":")
    if not sep or not color.startswith("c"):
        return None
    return shape


def fluid_sleep(
    mem: FactMemory,
    used: Iterable[FactKey],
    level: int,
    *,
    time_s: float = SLEEP_MAX_S,
    clock: Callable[[], float] = time.monotonic,
) -> FluidSleepReport:
    """Demote, merge and drop facts after ``level`` ended; see the module docstring."""
    start = clock()
    used_set = set(used)
    demoted: list[FactKey] = []
    merged: list[FactKey] = []
    dropped: list[FactKey] = []

    def tick() -> None:
        if clock() - start >= time_s:
            raise _OutOfTimeError

    truncated = False
    try:
        for key in sorted(used_set & mem.missed):
            tick()
            fact = mem.get(key)
            if fact is not None and fact.hits >= THIN_BELOW:
                fact.hits = THIN_BELOW - 1
                demoted.append(key)
        groups: dict[tuple[str, str, str, tuple[Effect, ...]], list[FactKey]] = {}
        for key, fact in list(mem.facts.items()):
            tick()
            shape = _mergeable(key, fact.effects)
            if shape is None or fact.status == "broken":
                continue
            groups.setdefault((key[0], key[2], shape, fact.effects), []).append(key)
        for keys in groups.values():
            tick()
            if len(keys) < 2:
                continue
            keep = min(keys, key=lambda k: (-mem.facts[k].hits, k))
            for k in sorted(keys):
                if k != keep:
                    mem.delete(k)
                    merged.append(k)
        for key, fact in list(mem.facts.items()):
            tick()
            stale = fact.hits == 0 and fact.status == "thin" and fact.level == level
            if stale and key not in used_set:
                mem.delete(key)
                dropped.append(key)
    except _OutOfTimeError:
        truncated = True
    return FluidSleepReport(
        tuple(demoted), tuple(merged), tuple(dropped), clock() - start, truncated
    )
