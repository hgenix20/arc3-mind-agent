"""Hypotheses: candidate plans toward a goal, priced by expected value and ranked.

Spec M4.7a (hypothesis list), M4.7b (expected-value choice), section 3.4, and build plan
decision D-C.

Contract:
- ``Hypothesis`` is one candidate: the goal it serves, the plan, where the plan came from
  (``planner``, ``search`` or ``skill``), the rollout's numbers (path probability,
  expected moves, the worst harm probability on the path, the stop reason) and the
  expected value computed from them. ``id`` is ``hypothesis_id`` of the goal, source and
  plan, so the same plan toward the same goal from the same source has the same id in
  every replan.
- ``expected_value`` is decision D-C: ``EV = P_path * 100 * eff - P_harm * restart_moves``
  with ``eff = min(1, typical / expected_moves) ** 2``. The 100 is the level's reward,
  since the competition scores each level by ``min(human / agent, 1) ** 2 * 100``; the
  human count is unknown at runtime, so ``typical`` (the level's typical moves per
  completion, or the cap before the first completion; horizon rule, 4.7) stands in for
  it, and ``eff`` is that score formula applied to the path's expected moves. The adverse
  cost is the harm probability times the moves a restart costs (moves from the level
  start to the current state plus the segment length). Four decimals.
- ``from_rollout`` maps a ``Rollout`` to a ``Hypothesis``. The reward term counts only
  when the rollout reached a goal: spec 4.7b rule 2 gives reward for the goals a path
  reaches, so a path cut by the horizon or stopped early is priced at its harm cost alone,
  while its ``path_probability`` still reports how likely the recorded steps are.
- ``rank`` orders by expected value descending, then path probability to four decimals
  descending, then id ascending (4.7b rule 6).

Failure modes: ``expected_value`` raises ``ValueError`` for a ``typical`` at or below zero,
which has no meaning as a move count. Pure; nothing is logged here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from arcagent.core.types import Action
from arcagent.mind.rollout import Rollout

Source = Literal["planner", "search", "skill"]

LEVEL_REWARD = 100.0  # the value of completing the level, the scale of the per-level score
MIN_EXPECTED_MOVES = 1.0  # floor under expected moves in the efficiency factor
DECIMALS = 4  # expected values and probabilities are reported to four places
ID_CHARS = 10  # hex characters kept from the SHA-1 digest


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One candidate plan toward a goal with the numbers the choice is made on.

    ``plan`` is the move list as rolled out; ``path_probability``, ``expected_moves`` and
    ``stop`` are the rollout's; ``harm_probability`` is the largest harm probability on
    the path (0.0 when none was recorded); ``expected_value`` is D-C over those numbers.
    """

    id: str
    goal_id: str
    source: Source
    plan: tuple[Action, ...]
    path_probability: float
    expected_moves: float
    harm_probability: float
    expected_value: float
    stop: str


def expected_value(
    path_p: float, expected_moves: float, typical: float, harm_p: float, restart_moves: float
) -> float:
    """Decision D-C: ``path_p * 100 * eff - harm_p * restart_moves`` to four decimals.

    ``eff = min(1, typical / max(expected_moves, 1)) ** 2`` is the level score formula with
    ``typical`` standing in for the human move count: a path expected to take the typical
    moves or fewer earns the whole reward, one expected to take twice the typical moves
    earns a quarter. Raises ``ValueError`` when ``typical`` is at or below zero.
    """
    if typical <= 0:
        raise ValueError(f"typical moves must be above 0, got {typical}")
    eff = min(1.0, typical / max(expected_moves, MIN_EXPECTED_MOVES)) ** 2
    return round(path_p * LEVEL_REWARD * eff - harm_p * restart_moves, DECIMALS)


def hypothesis_id(goal_id: str, source: str, plan: Sequence[Action]) -> str:
    """Ten hex characters of SHA-1 over the goal id, the source and the plan; deterministic."""
    moves = " ".join(f"{a.id},{a.x},{a.y}" for a in plan)
    text = "\x1f".join((goal_id, source, moves))
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:ID_CHARS]


def from_rollout(
    goal_id: str,
    source: Source,
    plan: Sequence[Action],
    ro: Rollout,
    *,
    typical: float,
    restart_moves: float,
) -> Hypothesis:
    """A hypothesis from a rollout of ``plan`` toward ``goal_id``; see the module docstring."""
    moves = tuple(plan)
    harm_p = max((p for _, p in ro.harm), default=0.0)
    reward_p = ro.path_probability if ro.reached is not None else 0.0
    return Hypothesis(
        hypothesis_id(goal_id, source, moves),
        goal_id,
        source,
        moves,
        ro.path_probability,
        ro.expected_moves,
        harm_p,
        expected_value(reward_p, ro.expected_moves, typical, harm_p, restart_moves),
        ro.stop,
    )


def rank(hyps: Sequence[Hypothesis]) -> list[Hypothesis]:
    """Best first: expected value, then path probability to four decimals, then id."""
    return sorted(
        hyps, key=lambda h: (-h.expected_value, -round(h.path_probability, DECIMALS), h.id)
    )
