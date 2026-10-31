"""Rollout: a plan predicted step by step, with a probability per step, harm and a stop.

Spec M4.8a (rollout), M4.4c (path probability by the chain rule), build plan decision D-D
(implicit health 1 when there is no health bar).

Contract:
- ``rollout`` predicts ``actions`` from ``scene`` one step at a time with ``predict``
  (``assume_still`` = ``protect``), the way ``game_model.play_out`` does, including the
  optional rule function: the rule is asked every step; when the facts cannot predict the
  step its changes stand in, and when they can its changes are applied on top; a rule
  that raises adds nothing. ``play_out`` stays as it is for the DRIVE path.
- Step probability (M4.4c): the product over the distinct fact keys in the prediction's
  ``used`` set of ``mem.probability(key)``, to four decimals; the prior 0.5 when the set
  is empty (a step the rule alone predicted). ``used`` is already distinct, so a fact that
  covers two things counts once (rule 4), and a borrowed fact prices the step at the
  sibling's probability, since the sibling's key is what ``used`` holds.
- Path probability: the product of the step probabilities in path order, to four
  decimals. Each factor is a rule's probability conditioned on the simulated state it was
  evaluated in, so the product is the chain rule over those states; it claims no
  independence between the moves in the world. An empty path is the empty product, 1.0.
- Stops, the first that applies at step ``i``: ``unknown`` (no predicted scene and no
  rule changes; nothing is recorded for that step and ``unknown_at`` is ``i``);
  ``death`` (health reached zero: a predicted ``game_over`` and an adverse probability in
  the certain band each cost one health; the fatal scene is still appended and
  ``deaths_at`` is ``i``); ``goal`` (a goal holds in the new scene, ``reached``). After the
  loop: ``horizon`` when the horizon cut the actions short, else ``end``.
- Harm: ``adverse(scene_before, action)`` when given, clamped to [0, 1], recorded as
  ``(step, probability)`` when above zero. ``expected_moves`` is the sum over the recorded
  steps of ``1 / max(step probability, MIN_STEP_P)``, capped at
  ``EXPECTED_MOVES_HORIZONS * horizon`` and rounded to two decimals. ``assumed`` counts the
  assumed-still predictions over every step (M4.8d).

Failure modes: ``ValueError`` for a horizon below zero or a health below one. Pure apart
from the rule call, which is model code; its exceptions are swallowed as in ``play_out``.
Nothing is logged here: the agent that calls ``rollout`` writes the log (M4.8d).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from arcagent.core.types import Action
from arcagent.mind.facts import BAND_CERTAIN, PRIOR_A, PRIOR_B, FactKey, FactMemory
from arcagent.mind.game_model import RuleFn, apply_changes, rule_objects
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Scene
from arcagent.mind.predictor import predict

RolloutStop = Literal["goal", "unknown", "death", "horizon", "end"]
AdverseFn = Callable[[Scene, Action], float]  # harm probability of one action from one scene

PRIOR_STEP = round(PRIOR_A / (PRIOR_A + PRIOR_B), 4)  # a step no fact priced (M4.4c rule 5)
MIN_STEP_P = 0.05  # floor under a step's probability when counting its expected tries
EXPECTED_MOVES_HORIZONS = 3  # expected moves are capped at this many horizons
DECIMALS = 4  # probabilities are reported to four places (spec 4.4)


@dataclass(frozen=True, slots=True)
class Rollout:
    """What the model says a plan does, with the numbers the planner prices it by.

    ``scenes`` holds one predicted scene per recorded step and ``step_probabilities`` one
    probability per recorded step, so both have the same length. ``path_probability`` is
    their product (M4.4c). ``reached`` is the goal that held, ``unknown_at`` the index of
    the step no fact or rule could predict, ``harm`` the ``(step, probability)`` pairs the
    adverse function reported above zero, ``stop`` why the rollout ended, ``expected_moves``
    the expected tries to play the recorded steps, ``assumed`` the assumed-still
    predictions over them, and ``deaths_at`` the step index at which health reached zero.
    """

    scenes: tuple[Scene, ...]
    step_probabilities: tuple[float, ...]
    path_probability: float
    reached: Goal | None
    unknown_at: int | None
    harm: tuple[tuple[int, float], ...]
    stop: RolloutStop
    expected_moves: float
    assumed: int
    deaths_at: int | None


def _product(values: Iterable[float]) -> float:
    out = 1.0
    for v in values:
        out *= v
    return out


def step_probability(mem: FactMemory, used: Iterable[FactKey]) -> float:
    """Product of the distinct ``used`` facts' probabilities to four decimals (M4.4c).

    The prior 0.5 when ``used`` is empty: a step nothing priced is a step nobody has
    observed. Repeated keys count once; ``predict`` already dedupes ``Prediction.used``.
    """
    keys = dict.fromkeys(used)
    if not keys:
        return PRIOR_STEP
    return round(_product(mem.probability(key) for key in keys), DECIMALS)


def _expected_moves(step_probabilities: Sequence[float], horizon: int) -> float:
    tries = sum(1.0 / max(p, MIN_STEP_P) for p in step_probabilities)
    return round(min(tries, float(EXPECTED_MOVES_HORIZONS * horizon)), 2)


def _ask_rule(rule: RuleFn, scene: Scene, action: Action) -> list[dict[str, Any]] | None:
    """The rule's changes for one step; None when it raises, as ``play_out`` treats it."""
    try:
        return rule(rule_objects(scene), {"id": action.id, "x": action.x, "y": action.y})
    except Exception:  # a rule is model code: its failure adds nothing
        return None


def _clamped(p: float) -> float:
    return min(1.0, max(0.0, p))


def rollout(
    scene: Scene,
    actions: Sequence[Action],
    mem: FactMemory,
    goals: Sequence[Goal],
    protect: frozenset[str],
    *,
    horizon: int,
    health: int = 1,
    adverse: AdverseFn | None = None,
    rule: RuleFn | None = None,
) -> Rollout:
    """Predict ``actions`` up to ``horizon`` steps with probabilities; see the module docstring.

    ``protect`` is the set of sigs that must have a fact (the others are assumed still),
    ``health`` the lives the path may lose before it is a death (1 without a health bar,
    decision D-D), ``adverse`` the harm probability of an action from a scene, ``rule`` the
    optional model function ``play_out`` also takes. Raises ``ValueError`` when ``horizon``
    is below zero or ``health`` below one.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be at least 0, got {horizon}")
    if health < 1:
        raise ValueError(f"health must be at least 1, got {health}")
    scenes: list[Scene] = []
    probabilities: list[float] = []
    harm: list[tuple[int, float]] = []
    assumed = 0
    lives = health
    reached: Goal | None = None
    unknown_at: int | None = None
    deaths_at: int | None = None
    stop: RolloutStop = "horizon" if len(actions) > horizon else "end"
    cur = scene
    for i in range(min(len(actions), horizon)):
        action = actions[i]
        pred = predict(cur, action, mem, assume_still=protect)
        changes = None if rule is None else _ask_rule(rule, cur, action)
        nxt = pred.scene
        if nxt is None:
            if not changes:
                unknown_at, stop = i, "unknown"
                break
            nxt = apply_changes(cur, changes)
        elif changes:
            nxt = apply_changes(nxt, changes)
        probabilities.append(step_probability(mem, pred.used))
        assumed += len(pred.assumed)
        h = 0.0 if adverse is None else _clamped(adverse(cur, action))
        if h > 0.0:
            harm.append((i, h))
        scenes.append(nxt)
        cur = nxt
        if pred.game_over or h >= BAND_CERTAIN:
            lives -= 1
            if lives == 0:
                deaths_at, stop = i, "death"
                break
        hit = next((g for g in goals if g.holds(cur)), None)
        if hit is not None:
            reached, stop = hit, "goal"
            break
    return Rollout(
        tuple(scenes),
        tuple(probabilities),
        round(_product(probabilities), DECIMALS),
        reached,
        unknown_at,
        tuple(harm),
        stop,
        _expected_moves(probabilities, horizon),
        assumed,
        deaths_at,
    )
