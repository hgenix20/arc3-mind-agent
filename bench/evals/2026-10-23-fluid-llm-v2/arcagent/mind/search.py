"""Short breadth-first search over predicted scenes toward goal guesses.

``search`` expands the scene with every action the caller offers, using ``predict`` to
build each child. The first scene where any goal holds wins, so the plan has the fewest
predicted moves. The root is checked first: a goal that already holds there returns with
an empty plan.

Branch rules:

- Every prediction is made with ``assume_still``: the protected sigs are the mover sigs
  (``goals.mover_sigs``) plus each goal's own sigs (``goals.goal_sigs`` without reach
  targets, so a reach target is assumed still like any other side object). Everything
  else with no usable fact is predicted still, so a side object with unlearned or broken
  facts never ends a branch. Callers may pass their own protected set.
- A prediction with no scene (an unknown fact of a protected sig) ends that branch; its
  ``unknown`` pairs join the frontier in first-met order, with no duplicates.
- A prediction that ends the game is never expanded and never counts as reaching a goal.
- Predictions never include spawns (``spawns=False``): a spawned thing only adds states.
- Scenes are deduped on a key restricted to the protected things: ``(sig, x, y)`` of each
  thing whose sig is protected or whose color is a ``clear``/``count`` goal color. Side
  objects predicted to change therefore can't multiply the state space. With nothing
  protected, the full ``Scene.key()`` is used.
- A prediction whose key equals its parent's changes nothing that matters and is dropped.
- ``match`` goals need a grid, so slice 1 search skips them.

Bounded two ways: ``time_s`` of wall clock (read from ``clock`` before every expansion)
and ``max_nodes`` expansions (``predict`` calls). Pure apart from the clock.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from arcagent.core.types import Action
from arcagent.mind.facts import FactMemory
from arcagent.mind.goals import Goal, goal_sigs, mover_sigs
from arcagent.mind.objects import Scene
from arcagent.mind.predictor import Prediction, predict

SEARCH_TIME_S = 0.25
SEARCH_MAX_NODES = 20000

Stop = Literal["found", "exhausted", "time", "nodes"]


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One planned action and the prediction it is expected to produce."""

    action: Action
    expect: Prediction


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outcome of one search. ``goal`` is None unless ``stop == "found"``.

    ``frontier`` holds the unknown ``(label, subject)`` pairs met, by first meeting.
    ``depth`` is the plan length when found, else the deepest level expanded.
    ``protected`` is the number of protected sigs the search used.
    """

    goal: Goal | None
    plan: tuple[PlanStep, ...]
    frontier: tuple[tuple[str, str], ...]
    nodes: int
    depth: int
    stop: Stop
    protected: int = 0


def protected_sigs(mem: FactMemory, goals: Sequence[Goal], scene: Scene) -> frozenset[str]:
    """Search's protected sigs: the movers plus each goal's sigs, reach targets excluded."""
    out = set(mover_sigs(mem))
    for g in goals:
        out |= goal_sigs(g, scene, targets=False)
    return frozenset(out)


def _first_holding(goals: Sequence[Goal], scene: Scene) -> Goal | None:
    return next((g for g in goals if g.holds(scene)), None)


def search(
    scene: Scene,
    actions: Callable[[Scene], list[Action]],
    mem: FactMemory,
    goals: Sequence[Goal],
    *,
    time_s: float = SEARCH_TIME_S,
    max_nodes: int = SEARCH_MAX_NODES,
    clock: Callable[[], float] = time.monotonic,
    assume_still: frozenset[str] | None = None,
) -> SearchResult:
    """Breadth-first search from ``scene`` for the shortest plan that makes a goal hold.

    ``assume_still`` is the protected sig set passed to ``predict``; None computes it with
    ``protected_sigs``.
    """
    start = clock()
    live = [g for g in goals if g.kind != "match"]
    protect = protected_sigs(mem, live, scene) if assume_still is None else assume_still
    colors = frozenset(g.color for g in live if g.kind in ("clear", "count"))

    def key_of(s: Scene) -> tuple[tuple[str, int, int], ...]:
        if not protect and not colors:
            return s.key()
        return tuple(
            sorted((t.sig, t.x, t.y) for t in s.things if t.sig in protect or t.color in colors)
        )

    frontier: dict[tuple[str, str], None] = {}

    def done(
        stop: Stop,
        nodes: int,
        depth: int,
        goal: Goal | None = None,
        plan: tuple[PlanStep, ...] = (),
    ) -> SearchResult:
        return SearchResult(goal, plan, tuple(frontier), nodes, depth, stop, len(protect))

    hit = _first_holding(live, scene)
    if hit is not None:
        return done("found", 0, 0, hit)

    seen = {key_of(scene)}
    queue: deque[tuple[Scene, tuple[PlanStep, ...]]] = deque([(scene, ())])
    nodes = 0
    deepest = 0
    while queue:
        current, plan = queue.popleft()
        deepest = max(deepest, len(plan))
        parent_key = key_of(current)
        for action in actions(current):
            if clock() - start >= time_s:
                return done("time", nodes, deepest)
            if nodes >= max_nodes:
                return done("nodes", nodes, deepest)
            pred = predict(current, action, mem, assume_still=protect, spawns=False)
            nodes += 1
            if pred.scene is None:
                for pair in pred.unknown:
                    frontier.setdefault(pair, None)
                continue
            if pred.game_over:
                continue
            key = key_of(pred.scene)
            if key == parent_key or key in seen:
                continue
            seen.add(key)
            child = (*plan, PlanStep(action, pred))
            goal = _first_holding(live, pred.scene)
            if goal is not None:
                return done("found", nodes, len(child), goal, child)
            queue.append((pred.scene, child))
    return done("exhausted", nodes, deepest)
