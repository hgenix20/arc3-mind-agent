"""The controller: when to change method, and why (spec 4.5c).

One owner of handoffs. ``Controller.decide`` reads the move's ``Signals`` and returns a
``Decision``: the method to use next, the rule that fired, and a reason that quotes the
signal values. The rules are checked in the controller's order and the first that fires
wins; the rest are skipped for that move (4.5c, rules 1 and 2). The agent that owns the
loop logs every decision with its rule name and reason (4.5c, rule 3); this module logs
nothing and keeps no state between moves.

Rules, named as in spec 4.5c (the signals that feed them are owned by other cards):

- ``H-a``: a gate tripped (4.10d, the per-move time gate or token gate): ``fallback``.
- ``H-b``: the segment broke at a miss that matters (3.6): ``return_to_fork``.
- ``H-c``: no progress for ``no_progress_moves`` moves on the active goal (4.13):
  ``replan``.
- ``H-d``: the budget is below the segment's length plus ``BUDGET_MARGIN`` (4.13):
  ``replan``. An unknown budget (None) never fires it.
- ``H-e``: ``stall_moves`` fallback moves in a row, the hybrid agent's stall rule
  (``arcagent.agent.hybrid.STALL_MOVES``): ``explore``.

The deadline sits outside the order: once it has passed, every decision is ``fallback``
with rule ``DEADLINE_RULE``, since only the fallback may run then. ``DEFAULT_ORDER`` is a
first guess; Kameron sets the order after seeing the workflow (4.5c rule 1, open item O2),
so the order is a constructor argument.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

Method = Literal["keep", "fallback", "return_to_fork", "explore", "replan"]

STALL_MOVES = 25  # fallback moves in a row that fire H-e; mirrors agent/hybrid.py
NO_PROGRESS_MOVES = 40  # moves without progress on the active goal that fire H-c
BUDGET_MARGIN = 2  # spare moves the budget must hold beyond the segment (H-d)
DEADLINE_RULE = "deadline"  # the rule name of a decision forced by the deadline
NO_RULE = "none"  # the rule name of a decision where nothing fired
DEFAULT_ORDER: tuple[str, ...] = ("H-a", "H-b", "H-e", "H-c", "H-d")

__all__ = [
    "BUDGET_MARGIN",
    "DEADLINE_RULE",
    "DEFAULT_ORDER",
    "NO_PROGRESS_MOVES",
    "NO_RULE",
    "RULES",
    "STALL_MOVES",
    "Controller",
    "Decision",
    "Method",
    "Signals",
]


@dataclass(frozen=True, slots=True)
class Signals:
    """The signal values the controller reads for one move.

    ``gate_tripped``: a per-move gate (4.10d) tripped. ``segment_broke``: the segment
    stopped at a miss that matters. ``fallbacks_in_row``: fallback moves in a row.
    ``no_progress_moves``: moves since the active goal last came closer. ``budget_left``:
    ``BudgetReader.remaining``, None while unknown. ``segment_left``: steps left in the
    current segment. ``deadline_passed``: the game or run deadline has passed.
    """

    gate_tripped: bool
    segment_broke: bool
    fallbacks_in_row: int
    no_progress_moves: int
    budget_left: int | None
    segment_left: int
    deadline_passed: bool


@dataclass(frozen=True, slots=True)
class Decision:
    """The controller's answer: the next method, the rule that fired, and why.

    ``rule`` is a name from ``RULES``, ``DEADLINE_RULE`` or ``NO_RULE``. ``reason`` quotes
    the signal values that fired, for the handoff log.
    """

    method: Method
    rule: str
    reason: str


Fire = tuple[Method, str]  # what a rule returns when it fires: the method and the reason
RuleFn = Callable[["Controller", Signals], Fire | None]


def _gate(_ctl: Controller, s: Signals) -> Fire | None:
    return ("fallback", "gate_tripped=True") if s.gate_tripped else None


def _segment_broke(_ctl: Controller, s: Signals) -> Fire | None:
    return ("return_to_fork", "segment_broke=True") if s.segment_broke else None


def _no_progress(ctl: Controller, s: Signals) -> Fire | None:
    if s.no_progress_moves < ctl.no_progress_moves:
        return None
    return ("replan", f"no_progress_moves={s.no_progress_moves} >= {ctl.no_progress_moves}")


def _budget_low(_ctl: Controller, s: Signals) -> Fire | None:
    if s.budget_left is None or s.budget_left >= s.segment_left + BUDGET_MARGIN:
        return None
    why = f"budget_left={s.budget_left} < segment_left={s.segment_left} + {BUDGET_MARGIN}"
    return ("replan", why)


def _stalled(ctl: Controller, s: Signals) -> Fire | None:
    if s.fallbacks_in_row < ctl.stall_moves:
        return None
    return ("explore", f"fallbacks_in_row={s.fallbacks_in_row} >= {ctl.stall_moves}")


# The one table of rules: its keys are the names an order may use.
_RULES: Mapping[str, RuleFn] = MappingProxyType(
    {
        "H-a": _gate,
        "H-b": _segment_broke,
        "H-c": _no_progress,
        "H-d": _budget_low,
        "H-e": _stalled,
    }
)
RULES: tuple[str, ...] = tuple(_RULES)


class Controller:
    """Ordered handoff rules; the first that fires wins (spec 4.5c).

    ``order`` names the rules to check, each from ``RULES`` and each at most once; a rule
    left out never fires. ``stall_moves`` and ``no_progress_moves`` are the limits of H-e
    and H-c and must be at least 1. Raises ``ValueError`` for an unknown or repeated rule
    name or a limit below 1. ``decide`` is pure.
    """

    def __init__(
        self,
        order: Sequence[str] = DEFAULT_ORDER,
        *,
        stall_moves: int = STALL_MOVES,
        no_progress_moves: int = NO_PROGRESS_MOVES,
    ) -> None:
        names = tuple(order)
        for i, name in enumerate(names):
            if name not in _RULES:
                raise ValueError(f"unknown handoff rule {name!r}; the rules are {RULES}")
            if name in names[:i]:
                raise ValueError(f"handoff rule {name!r} is named twice in the order")
        if stall_moves < 1:
            raise ValueError(f"stall_moves must be at least 1, got {stall_moves}")
        if no_progress_moves < 1:
            raise ValueError(f"no_progress_moves must be at least 1, got {no_progress_moves}")
        self._order = names
        self._stall_moves = stall_moves
        self._no_progress_moves = no_progress_moves

    @property
    def order(self) -> tuple[str, ...]:
        """The rule names in the order they are checked."""
        return self._order

    @property
    def stall_moves(self) -> int:
        """Fallback moves in a row at which H-e fires."""
        return self._stall_moves

    @property
    def no_progress_moves(self) -> int:
        """Moves without progress on the active goal at which H-c fires."""
        return self._no_progress_moves

    def decide(self, s: Signals) -> Decision:
        """The method for the next move, the rule that chose it, and the reason.

        A passed deadline wins before the order is read. Otherwise the rules are checked
        in ``order`` and the first that fires wins; when none fires the decision is
        ``keep`` with rule ``NO_RULE``.
        """
        if s.deadline_passed:
            return Decision("fallback", DEADLINE_RULE, "deadline_passed=True")
        for name in self._order:
            fired = _RULES[name](self, s)
            if fired is not None:
                method, reason = fired
                return Decision(method, name, reason)
        return Decision("keep", NO_RULE, "no rule fired")
