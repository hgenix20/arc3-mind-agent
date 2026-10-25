"""The goal rule-book: which goal kinds the agent proposes, and how much it trusts each.

A rule names a ``goals.GoalKind``, its prior (a weight for ranking guesses of that kind,
1.0 is neutral) and whether ``GoalBook.propose`` makes guesses of it. Only generic goal
forms belong here, never what a particular game needs: the rules describe kinds of goals
(reach a thing, clear a color, make two framed regions equal, put a thing on every target),
and the code recognizes them on any screen. Kept as code, not a data file, so it ships with
the package as is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoalRule:
    """One goal kind: its prior and whether it is proposed."""

    kind: str
    prior: float
    proposed: bool


GOAL_RULES: tuple[GoalRule, ...] = (
    GoalRule("reach", 1.0, proposed=True),  # a moving thing on or beside a target thing
    GoalRule("clear", 1.0, proposed=True),  # no thing of a color left
    GoalRule("match", 1.5, proposed=True),  # two framed regions of one size equal
    GoalRule("cover", 1.2, proposed=True),  # every target has its own mover on or beside it
    GoalRule("count", 0.5, proposed=False),  # n things of a color: evaluated, not proposed
)
_BY_KIND = {r.kind: r for r in GOAL_RULES}


def prior_of(kind: str) -> float:
    """The rule's prior for ``kind`` (1.0 for a kind with no rule)."""
    rule = _BY_KIND.get(kind)
    return rule.prior if rule is not None else 1.0


def proposed(kind: str) -> bool:
    """Whether ``GoalBook.propose`` makes guesses of ``kind``."""
    rule = _BY_KIND.get(kind)
    return rule is not None and rule.proposed
