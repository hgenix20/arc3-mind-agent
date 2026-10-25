"""Short best-first search over predicted scenes toward goal guesses.

``search`` expands the scene with every action the caller offers, using ``predict`` to
build each child. Scenes are expanded in order of ``len(plan) + weight * h``, where ``h``
is the lowest ``Goal.distance`` over the goals in that scene (0 when none has one), ties
to the lower ``h`` and then in the order they were queued. With ``weight=0`` or no
distances, scenes go level by level as in breadth-first search, and the first scene where
a goal holds has the fewest predicted moves. With
distances, the search runs toward the nearest goal first and the plan is short but not
always the shortest. The root is checked first: a goal that already holds there returns
with an empty plan.

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
- ``match`` goals need a grid: when one is live, every scene is rendered
  (``regions.render``) and goals are checked and scored on that grid.
- ``noop`` holds ``(Scene.key(), Action.key())`` pairs seen to change nothing on a real
  frame. Such an action is not expanded from a scene with that full key, whatever the
  facts predict: the observation beats the prediction, and the pair costs no node.
- ``goal_test`` is an extra predicate goal (the fluid agent's novelty pseudo-goal): a
  scene where it holds wins like a goal does, with ``SearchResult.goal`` None. Goals in
  ``goals`` are checked first on each scene.

Anytime: every expanded scene is scored by ``Goal.distance`` against the root, per goal.
``SearchResult.best`` is the plan to the scene that lowered some goal's distance the most
(ties: the shorter plan, then the first found), with that goal and its distance there;
empty when no scene lowered any. At each node count in ``SEARCH_SLICES`` the search
notes the best plan's first action and distance; when that note is the same at
``STABLE_SLICES`` boundaries in a row, the search stops with ``"stable"``: searching on
isn't changing its answer. A search with no best never stops this way.

Quiescence: a scene where every action the caller offers is predicted to end the game is
a trap (``is_trap``). A plan whose end scene is a trap isn't returned at once: the search
keeps going for a plan whose end isn't a trap, at most ``QUIESCENCE_PLIES`` moves longer,
and returns the trap plan only when none turns up (or a bound stops the search first). The
best partial plan never ends in a trap. Plans come off the queue in priority order, not
length order, so a scene too long to beat a held trap plan is skipped, not a stop. A leaf
next to a death color isn't a trap by
itself: the move back is usually safe, and vetoes refuse the deadly moves. Slides can't be
caught mid-way: the predictor resolves each to where it stops.

``max_depth`` (the agent passes the moves left this life) caps plan length: a scene that
deep is not expanded.

Bounded two ways. ``max_nodes`` expansions (``predict`` calls) is the main bound: every
game runs as a thread of one process, so a wall-clock budget would buy less search the
more games share the GIL. ``time_s`` of wall clock (read from ``clock`` before every
expansion) is only a safety cap. Pure apart from the clock.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Literal

from arcagent.core.types import Action
from arcagent.mind.facts import FactMemory
from arcagent.mind.goals import Goal, goal_sigs, mover_sigs
from arcagent.mind.kinematics import Kinematics
from arcagent.mind.objects import Scene
from arcagent.mind.predictor import Prediction, predict
from arcagent.mind.regions import render

SEARCH_MAX_NODES = 3000  # the main bound: predict calls per search
SEARCH_TIME_S = 1.0  # wall-clock safety cap; shared with other game threads under the GIL
SEARCH_SLICES = (500, 1000, 2000)  # node counts where the best plan is noted
STABLE_SLICES = 2  # boundaries in a row with the same best plan that end the search
QUIESCENCE_PLIES = 2  # extra moves a safe plan may take over a plan ending in a trap
HEURISTIC_WEIGHT = 1.0  # weight of the goal distance in the best-first priority
_GRID_KINDS = frozenset({"match", "match_thing"})  # goal kinds scored on rendered scenes

Stop = Literal["found", "exhausted", "time", "nodes", "stable"]


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One planned action and the prediction it is expected to produce."""

    action: Action
    expect: Prediction


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outcome of one search. ``goal`` is None unless ``stop == "found"`` by a goal (a
    find by ``goal_test`` leaves it None).

    ``frontier`` holds the unknown ``(label, subject)`` pairs met, by first meeting.
    ``depth`` is the plan length when found, else the deepest level expanded.
    ``protected`` is the number of protected sigs the search used. ``best``,
    ``best_goal`` and ``best_distance`` are the anytime answer (module docstring).
    """

    goal: Goal | None
    plan: tuple[PlanStep, ...]
    frontier: tuple[tuple[str, str], ...]
    nodes: int
    depth: int
    stop: Stop
    protected: int = 0
    best: tuple[PlanStep, ...] = ()
    best_goal: Goal | None = None
    best_distance: int | None = None


def protected_sigs(mem: FactMemory, goals: Sequence[Goal], scene: Scene) -> frozenset[str]:
    """Search's protected sigs: the movers plus each goal's sigs, reach targets excluded."""
    out = set(mover_sigs(mem))
    for g in goals:
        out |= goal_sigs(g, scene, targets=False)
    return frozenset(out)


def _first_holding(goals: Sequence[Goal], scene: Scene) -> Goal | None:
    return next((g for g in goals if g.holds(scene)), None)


def _pins(live: list[Goal], scene: Scene, mem: FactMemory) -> frozenset[int]:
    """The match_thing references in ``scene`` that predictions hold still.

    A reference whose look is a known mover is left to its facts: it doesn't hold a
    picture in place, and pinning it hides every plan that moves it (in tn36 the
    reference is one of the pieces each click moves, and no plan for ``clear`` was found).
    """
    refs = {g.n for g in live if g.kind == "match_thing" and g.n >= 0}
    movers = mover_sigs(mem)
    return frozenset(
        t.id for t in scene.things if t.id in refs and t.sig not in movers and t.kind not in movers
    )


def is_trap(
    scene: Scene,
    actions: Callable[[Scene], list[Action]],
    mem: FactMemory,
    protect: frozenset[str],
    pinned: frozenset[int] = frozenset(),
) -> bool:
    """Every offered action from ``scene`` is predicted to end the game (and there is one).

    An action with an unknown outcome isn't a death. Stops at the first survivable action.
    """
    offered = actions(scene)
    return bool(offered) and all(
        predict(scene, a, mem, assume_still=protect, spawns=False, pinned=pinned).game_over
        for a in offered
    )


@dataclass(slots=True)
class _Best:
    """The anytime answer so far: goal, distance, plan, and the gain over the root.

    ``trap`` says whether a scene is a trap; a trap is never the best.
    """

    trap: Callable[[Scene], bool]
    kin: Kinematics | None = None
    grids: bool = False  # render scenes for match and match_thing goals
    gain: int = 0
    goal: Goal | None = None
    distance: int | None = None
    plan: tuple[PlanStep, ...] = ()

    def offer(
        self, root: dict[Goal, int], scene: Scene, plan: tuple[PlanStep, ...]
    ) -> tuple[Goal | None, int | None]:
        """Score ``scene`` per goal; return the first goal that holds there (distance 0)
        and the lowest distance of any goal."""
        hit: Goal | None = None
        low: int | None = None
        grid = render(scene) if self.grids else None
        for g, r in root.items():
            d = g.distance(scene, grid, self.kin)
            if d is None:
                continue
            low = d if low is None else min(low, d)
            if d == 0 and hit is None:
                hit = g
            gain = r - d
            better = gain > self.gain or (gain == self.gain > 0 and len(plan) < len(self.plan))
            if better and not self.trap(scene):
                self.gain, self.goal, self.distance, self.plan = gain, g, d, plan
        return hit, low

    def note(self) -> tuple[tuple[int, int, int], int | None] | None:
        return None if not self.plan else (self.plan[0].action.key(), self.distance)


_Note = tuple[tuple[int, int, int], int | None] | None


def _stable(notes: list[_Note], nodes: int, best: _Best) -> bool:
    """At a ``SEARCH_SLICES`` boundary not yet noted, note ``best``; True when the last
    ``STABLE_SLICES`` notes are the same non-empty one."""
    if nodes not in SEARCH_SLICES or len(notes) > SEARCH_SLICES.index(nodes):
        return False
    notes.append(best.note())
    last = notes[-STABLE_SLICES:]
    return len(last) == STABLE_SLICES and last[0] is not None and len(set(last)) == 1


def _distances(
    goals: Sequence[Goal], scene: Scene, kin: Kinematics | None = None
) -> dict[Goal, int]:
    """Each goal's distance in ``scene``, in goal order; goals with no estimate left out."""
    out: dict[Goal, int] = {}
    grid = render(scene) if any(g.kind in _GRID_KINDS for g in goals) else None
    for g in goals:
        d = g.distance(scene, grid, kin)
        if d is not None:
            out[g] = d
    return out


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
    goal_test: Callable[[Scene], bool] | None = None,
    noop: Collection[tuple[tuple[tuple[str, int, int], ...], tuple[int, int, int]]] = (),
    kin: Kinematics | None = None,
    weight: float = HEURISTIC_WEIGHT,
    max_depth: int | None = None,
) -> SearchResult:
    """Breadth-first search from ``scene`` for the shortest plan that makes a goal hold.

    ``kin`` measures reach distances in moves (``Goal.distance``); without it they are
    cell gaps. A caller that compares ``best_distance`` with its own distances must pass
    the same ``kin`` it measures with.

    ``assume_still`` is the protected sig set passed to ``predict``; None computes it with
    ``protected_sigs``. ``goal_test``, when given, is one more goal: the first scene where
    it holds (after no goal in ``goals`` did) ends the search as ``found``, goal None.
    """
    live = list(goals)
    protect = protected_sigs(mem, live, scene) if assume_still is None else assume_still
    run = _Run(actions, mem, live, protect, goal_test=goal_test, noop=frozenset(noop), kin=kin)
    run.weight = weight
    run.max_depth = max_depth
    return run.run(scene, clock, time_s, max_nodes)


Plan = tuple[PlanStep, ...]
Queue = list[tuple[float, int, int, Scene, Plan]]  # (priority, h, queued, scene, plan)


class _Run:
    """One search's state; ``run`` is the loop, ``_expand`` one predicted child."""

    def __init__(
        self,
        actions: Callable[[Scene], list[Action]],
        mem: FactMemory,
        live: list[Goal],
        protect: frozenset[str],
        *,
        goal_test: Callable[[Scene], bool] | None,
        noop: frozenset[tuple[tuple[tuple[str, int, int], ...], tuple[int, int, int]]],
        kin: Kinematics | None = None,
    ) -> None:
        self.actions, self.mem, self.live, self.protect = actions, mem, live, protect
        self.pinned: frozenset[int] = frozenset()  # set per run (``_pins``)
        self.goal_test, self.noop = goal_test, noop
        self.colors = frozenset(g.color for g in live if g.kind in ("clear", "count"))
        self.frontier: dict[tuple[str, str], None] = {}
        self.kin = kin
        self.best = _Best(self.trap, kin, any(g.kind in _GRID_KINDS for g in live))
        self.held: list[tuple[Goal | None, Plan]] = []  # a found plan ending in a trap
        self.root: dict[Goal, int] = {}
        self.unscored: list[Goal] = []
        self.seen: set[tuple[tuple[str, int, int], ...]] = set()
        self.nodes = 0
        self.deepest = 0
        self.weight = HEURISTIC_WEIGHT
        self.max_depth: int | None = None
        self.queued = 0  # tiebreak: the order scenes were queued in

    def trap(self, s: Scene) -> bool:
        return is_trap(s, self.actions, self.mem, self.protect, self.pinned)

    def key_of(self, s: Scene) -> tuple[tuple[str, int, int], ...]:
        if not self.protect and not self.colors:
            return s.key()
        return tuple(
            sorted(
                (t.sig, t.x, t.y)
                for t in s.things
                if t.sig in self.protect or t.kind in self.protect or t.color in self.colors
            )
        )

    def done(self, stop: Stop, goal: Goal | None = None, plan: Plan = ()) -> SearchResult:
        depth = len(plan) if stop == "found" else self.deepest
        if stop != "found" and self.held:
            (goal, plan), stop = self.held[0], "found"
            depth = len(plan)
        best = self.best
        return SearchResult(
            goal,
            plan,
            tuple(self.frontier),
            self.nodes,
            depth,
            stop,
            len(self.protect),
            best.plan,
            best.goal,
            best.distance,
        )

    def holds(self, s: Scene, goal: Goal | None) -> bool:
        return goal is not None or (self.goal_test is not None and self.goal_test(s))

    def run(
        self, scene: Scene, clock: Callable[[], float], time_s: float, max_nodes: int
    ) -> SearchResult:
        start = clock()
        self.pinned = _pins(self.live, scene, self.mem)
        self.root = _distances(self.live, scene, self.kin)
        self.unscored = [g for g in self.live if g not in self.root]
        hit = _first_holding(self.live, scene)
        if self.holds(scene, hit):
            return self.done("found", hit)
        self.seen.add(self.key_of(scene))
        queue: Queue = []
        self.push(queue, scene, (), min(self.root.values(), default=None))
        notes: list[_Note] = []
        while queue:
            *_, current, plan = heapq.heappop(queue)
            if self.held and len(plan) >= len(self.held[0][1]) + QUIESCENCE_PLIES:
                continue  # can't beat the held trap plan within the extra plies
            if self.max_depth is not None and len(plan) >= self.max_depth:
                continue
            self.deepest = max(self.deepest, len(plan))
            full_key = current.key() if self.noop else ()
            for action in self.actions(current):
                if self.noop and (full_key, action.key()) in self.noop:
                    continue
                if clock() - start >= time_s:
                    return self.done("time")
                if self.nodes >= max_nodes:
                    return self.done("nodes")
                if _stable(notes, self.nodes, self.best):
                    return self.done("stable")
                found = self._expand(current, plan, action, queue)
                if found is not None:
                    return found
        return self.done("exhausted")

    def push(self, queue: Queue, scene: Scene, plan: Plan, h: int | None) -> None:
        """Queue a scene at priority ``len(plan) + weight * h``, ties to the lower ``h``."""
        self.queued += 1
        low = h or 0
        heapq.heappush(queue, (len(plan) + self.weight * low, low, self.queued, scene, plan))

    def _expand(
        self, current: Scene, plan: Plan, action: Action, queue: Queue
    ) -> SearchResult | None:
        """Predict one child; queue it, hold it (a trap goal), or return it as found."""
        pred = predict(
            current, action, self.mem, assume_still=self.protect, spawns=False, pinned=self.pinned
        )
        self.nodes += 1
        if pred.scene is None:
            for pair in pred.unknown:
                self.frontier.setdefault(pair, None)
            return None
        key = self.key_of(pred.scene)
        if pred.game_over or key == self.key_of(current) or key in self.seen:
            return None
        self.seen.add(key)
        child = (*plan, PlanStep(action, pred))
        goal, low = self.best.offer(self.root, pred.scene, child)
        goal = goal or _first_holding(self.unscored, pred.scene)
        if not self.holds(pred.scene, goal):
            self.push(queue, pred.scene, child, low)
        elif not self.trap(pred.scene):
            return self.done("found", goal, child)
        elif not self.held:
            self.held.append((goal, child))
        return None
