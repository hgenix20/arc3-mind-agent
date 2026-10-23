"""Short breadth-first search over predicted scenes toward goal guesses.

``search`` expands the scene with every action the caller offers, using ``predict`` to
build each child. The first scene where any goal holds wins, so the plan has the fewest
predicted moves. The root is checked first: a goal that already holds there returns with
an empty plan.

Branch rules:

- A prediction with no scene (an unknown fact) ends that branch; its ``unknown`` pairs
  join the frontier in first-met order, with no duplicates.
- A prediction that ends the game is never expanded and never counts as reaching a goal.
- A prediction whose scene key equals its parent's changes nothing and is dropped.
- Scenes are deduped by ``Scene.key()``.
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
from arcagent.mind.goals import Goal
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
    """

    goal: Goal | None
    plan: tuple[PlanStep, ...]
    frontier: tuple[tuple[str, str], ...]
    nodes: int
    depth: int
    stop: Stop


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
) -> SearchResult:
    """Breadth-first search from ``scene`` for the shortest plan that makes a goal hold."""
    start = clock()
    live = [g for g in goals if g.kind != "match"]
    frontier: dict[tuple[str, str], None] = {}

    def done(
        stop: Stop,
        nodes: int,
        depth: int,
        goal: Goal | None = None,
        plan: tuple[PlanStep, ...] = (),
    ) -> SearchResult:
        return SearchResult(goal, plan, tuple(frontier), nodes, depth, stop)

    hit = _first_holding(live, scene)
    if hit is not None:
        return done("found", 0, 0, hit)

    seen = {scene.key()}
    queue: deque[tuple[Scene, tuple[PlanStep, ...]]] = deque([(scene, ())])
    nodes = 0
    deepest = 0
    while queue:
        current, plan = queue.popleft()
        deepest = max(deepest, len(plan))
        parent_key = current.key()
        for action in actions(current):
            if clock() - start >= time_s:
                return done("time", nodes, deepest)
            if nodes >= max_nodes:
                return done("nodes", nodes, deepest)
            pred = predict(current, action, mem)
            nodes += 1
            if pred.scene is None:
                for pair in pred.unknown:
                    frontier.setdefault(pair, None)
                continue
            if pred.game_over:
                continue
            key = pred.scene.key()
            if key == parent_key or key in seen:
                continue
            seen.add(key)
            child = (*plan, PlanStep(action, pred))
            goal = _first_holding(live, pred.scene)
            if goal is not None:
                return done("found", nodes, len(child), goal, child)
            queue.append((pred.scene, child))
    return done("exhausted", nodes, deepest)
