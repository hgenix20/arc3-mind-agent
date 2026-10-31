"""Curiosity flags: what is worth a check, with a value and a cost (spec 4.11).

A flag is kept per kind. The agent calls ``FlagBook.note`` for each kind on screen after
each move; the flag goes to ``check`` when the kind is a new look or has unknown pairs
(frontier pairs with no rule; 4.11a, rule 1), or when the kind changed this move. The
value is the number of unknown pairs the check would resolve, plus ``NEW_LOOK_VALUE`` for a
new look and ``CHANGED_VALUE`` for a change (4.11b, rule 2, without the goal value, which
the caller may add to its own ranking). The caller sets the cost with ``set_cost`` (moves
to reach the thing plus one; 4.11b, rule 3). ``top`` ranks the flags to check by value per
cost and ``lines`` writes them for the briefing. The planner decides (4.11b, rule 5;
Kameron, 2026-10-30: "I will let the LLM decide on the checks on its own.").

Labels: the planner may name a kind (4.11a, rule 2; Kameron: "it should allow designer
labels that it can create"). ``label`` stores the name, cut to ``LABEL_CHARS``. A flag
keeps its source, its label and its outcome after a check (4.11a, rule 3) and across
levels: ``reset_level`` keeps them and resets state, value and cost.

States: ``new`` (nothing to check), ``check``, ``checked`` (the outcome is recorded) and
``ignored`` (the planner said to leave it; later notes refresh the value and keep the state
until ``reset_level``). Bounds: at most ``max_flags`` flags, and when the book is full a
new flag evicts the flag with the lowest value (ties by kind); an outcome keeps at most
``OUTCOME_MAX`` keys. Nothing here logs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

State = Literal["new", "check", "checked", "ignored"]
Source = Literal["novel_look", "unknown_pair", "changed", "label"]

MAX_FLAGS = 200  # flags a book holds by default
LABEL_CHARS = 30  # characters a label is cut to
OUTCOME_MAX = 20  # outcome keys kept per flag
NEW_LOOK_VALUE = 1.0  # value a new look adds to a check
CHANGED_VALUE = 0.5  # value a change this move adds to a check
MIN_COST = 1.0  # the cost floor in the value-per-cost ranking

__all__ = [
    "CHANGED_VALUE",
    "LABEL_CHARS",
    "MAX_FLAGS",
    "MIN_COST",
    "NEW_LOOK_VALUE",
    "OUTCOME_MAX",
    "Flag",
    "FlagBook",
    "Source",
    "State",
]


@dataclass(slots=True)
class Flag:
    """One kind's flag: its state, where it came from, its value and cost, and what the
    agent has learned about it: ``label`` (the planner's name for the kind, empty until
    given) and ``outcome`` (the rule keys a check changed). Mutable; the book owns it and
    ``FlagBook.get`` returns the live record."""

    kind: str
    state: State
    source: Source
    value: float
    cost: float
    label: str
    outcome: tuple[str, ...]


def _source(*, new_look: bool, unknown_pairs: int, changed: bool) -> Source:
    """The first of the three reasons that holds. With none, ``novel_look``: the kind is
    new to the book."""
    if new_look:
        return "novel_look"
    if unknown_pairs > 0:
        return "unknown_pair"
    if changed:
        return "changed"
    return "novel_look"


def _line(flag: Flag) -> str:
    label = f" '{flag.label}'" if flag.label else ""
    return f"check {flag.kind}{label}: value {flag.value:g}, cost {flag.cost:g}, {flag.source}"


class FlagBook:
    """The flags of one game, keyed by kind (module docstring).

    ``max_flags`` must be at least 1 (``ValueError`` otherwise). ``set_cost``, ``checked``
    and ``ignore`` raise ``KeyError`` for a kind with no flag; ``note`` and ``label``
    make the flag when it is missing.
    """

    def __init__(self, max_flags: int = MAX_FLAGS) -> None:
        if max_flags < 1:
            raise ValueError(f"max_flags must be at least 1, got {max_flags}")
        self._max = max_flags
        self._flags: dict[str, Flag] = {}

    def _add(self, flag: Flag) -> Flag:
        """Insert a new flag; when the book is full, the lowest-value flag goes first."""
        if len(self._flags) >= self._max:
            lowest = min(self._flags.values(), key=lambda f: (f.value, f.kind))
            del self._flags[lowest.kind]
        self._flags[flag.kind] = flag
        return flag

    def _flag(self, kind: str) -> Flag:
        flag = self._flags.get(kind)
        if flag is None:
            raise KeyError(f"no flag for kind {kind!r}")
        return flag

    def note(self, kind: str, *, new_look: bool, unknown_pairs: int, changed: bool = False) -> Flag:
        """Record what this move showed about ``kind`` and return its flag.

        The value becomes ``unknown_pairs`` plus ``NEW_LOOK_VALUE`` for a new look plus
        ``CHANGED_VALUE`` for a change. When any of the three holds, the state becomes
        ``check`` unless it is ``ignored``. A flag made here takes its source from the first
        reason that holds (novel look, unknown pair, changed); an existing flag keeps its
        source, label and outcome. A kind the book has never held with nothing to check is
        recorded ``new`` with source ``novel_look`` and value 0. ``unknown_pairs`` below 0
        raises ``ValueError``.
        """
        if unknown_pairs < 0:
            raise ValueError(f"unknown_pairs must be at least 0, got {unknown_pairs}")
        worth = new_look or unknown_pairs > 0 or changed
        flag = self._flags.get(kind)
        if flag is None:
            source = _source(new_look=new_look, unknown_pairs=unknown_pairs, changed=changed)
            flag = self._add(Flag(kind, "new", source, 0.0, 0.0, "", ()))
        flag.value = (
            float(unknown_pairs)
            + (NEW_LOOK_VALUE if new_look else 0.0)
            + (CHANGED_VALUE if changed else 0.0)
        )
        if worth and flag.state != "ignored":
            flag.state = "check"
        return flag

    def set_cost(self, kind: str, cost: float) -> None:
        """Set the check's cost: moves to reach the thing plus one. Below 0 raises
        ``ValueError``; a kind with no flag raises ``KeyError``."""
        if cost < 0:
            raise ValueError(f"cost must be at least 0, got {cost}")
        self._flag(kind).cost = cost

    def checked(self, kind: str, outcome: Sequence[str]) -> None:
        """A check of ``kind`` happened: keep the first ``OUTCOME_MAX`` rule keys it changed
        and set the state to ``checked``. A kind with no flag raises ``KeyError``; a single
        string in place of a sequence of keys raises ``TypeError``."""
        if isinstance(outcome, str):
            raise TypeError("outcome is a sequence of rule keys, not one string")
        flag = self._flag(kind)
        flag.state = "checked"
        flag.outcome = tuple(outcome[:OUTCOME_MAX])

    def ignore(self, kind: str) -> None:
        """The planner said to leave ``kind`` alone until the next level. ``KeyError`` for
        a kind with no flag."""
        self._flag(kind).state = "ignored"

    def label(self, kind: str, text: str) -> None:
        """Store the planner's name for ``kind``, stripped and cut to ``LABEL_CHARS``. The
        source stays; a kind with no flag gets one in state ``new`` with source ``label``."""
        flag = self._flags.get(kind)
        if flag is None:
            flag = self._add(Flag(kind, "new", "label", 0.0, 0.0, "", ()))
        flag.label = text.strip()[:LABEL_CHARS]

    def get(self, kind: str) -> Flag | None:
        """The live flag for ``kind``, or None."""
        return self._flags.get(kind)

    def top(self, n: int) -> list[Flag]:
        """At most ``n`` flags in state ``check``, best first: by value over
        ``max(cost, MIN_COST)`` descending, then by kind."""
        if n <= 0:
            return []
        due = [f for f in self._flags.values() if f.state == "check"]
        due.sort(key=lambda f: (-f.value / max(f.cost, MIN_COST), f.kind))
        return due[:n]

    def lines(self, n: int) -> list[str]:
        """Briefing lines for ``top(n)``: ``check <kind>[ '<label>']: value <v>, cost <c>,
        <source>``."""
        return [_line(f) for f in self.top(n)]

    def reset_level(self) -> None:
        """A new level: every flag goes back to ``new`` with value and cost 0; kinds,
        sources, labels and outcomes stay."""
        for flag in self._flags.values():
            flag.state = "new"
            flag.value = 0.0
            flag.cost = 0.0
