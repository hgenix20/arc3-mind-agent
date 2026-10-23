"""Search over the LLM's own world model, so it can hand off a level it understands.

Why: in the 25-game evaluation (eval v2) about 54 of the 100 possible points per
game were lost on levels the agent never finished, against about 4 lost to
spending extra actions. Transcripts show the model deriving a Python world model
that already reproduces nearly every observed transition (sk48: 62 of 63) and then
running out of clock while it reasons out the action sequence turn by turn. A
breadth-first or best-first search over that model finds the sequence in seconds
of CPU and no tokens.

What: ``plan(step, start, goal, actions)`` searches from ``start`` with the
model's ``step(state, action) -> state`` until ``goal(state)`` is true, and
returns the shortest action list it found (fewest actions is what the score
rewards). With ``heuristic`` it runs A* (``g + h``); without, breadth-first.

Contract:
- ``step`` must be pure: it must not mutate its input. States are compared by a
  key: ``key(state)`` if given, else a structural freeze of nested lists, tuples,
  dicts, sets and objects with ``tobytes`` (numpy arrays) or ``__dict__``.
- ``step`` may return ``None`` for an illegal or losing action; that branch is
  pruned. Exceptions inside ``step``, ``goal``, ``heuristic`` or ``key`` prune
  that branch too and are counted in the result, never raised, because the
  world model is model-written code.
- Bounded: stops at ``max_nodes`` expanded states or ``time_limit_s`` seconds,
  whichever comes first, and says which in ``PlanResult.stopped``.

The file is copied verbatim into the Kaggle notebook by
``scripts/kaggle_duck_notebook.py``; standard library only, no ``from __future__``.
"""

import heapq
import itertools
import time
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_NODES = 200_000
TIME_LIMIT_S = 20.0


@dataclass
class PlanResult:
    """Outcome of one search. ``actions`` is ``None`` when no goal was reached."""

    actions: list[Any] | None
    expanded: int
    stopped: str  # "goal", "exhausted", "max_nodes", "time_limit", "goal_at_start"
    errors: int = 0
    first_error: str = ""
    seconds: float = 0.0
    best_partial: list[Any] = field(default_factory=list)

    def summary(self) -> str:
        found = f"{len(self.actions)} actions" if self.actions is not None else "no plan"
        err = f", {self.errors} model errors (first: {self.first_error})" if self.errors else ""
        return (
            f"{found}; stopped={self.stopped}; expanded={self.expanded}; {self.seconds:.1f}s{err}"
        )


def freeze(x: Any) -> Hashable:
    """A hashable, structure-preserving key for common state representations."""
    if isinstance(x, (str, bytes, int, float, bool, type(None))):
        return x
    tobytes = getattr(x, "tobytes", None)
    if callable(tobytes):
        return (
            type(x).__name__,
            getattr(x, "shape", None),
            str(getattr(x, "dtype", "")),
            tobytes(),
        )
    if isinstance(x, dict):
        items = sorted(((freeze(k), freeze(v)) for k, v in x.items()), key=repr)
        out: Hashable = ("d", tuple(items))
    elif isinstance(x, (set, frozenset)):
        out = ("s", tuple(sorted((freeze(v) for v in x), key=repr)))
    elif isinstance(x, (list, tuple)):
        out = ("l", tuple(freeze(v) for v in x))
    elif getattr(x, "__dict__", None) is not None:
        out = (type(x).__name__, freeze(vars(x)))
    else:
        out = repr(x)
    return out


class _Search:
    """One bounded search; model-code exceptions prune a branch and are counted."""

    def __init__(
        self,
        step: Callable[[Any, Any], Any],
        goal: Callable[[Any], bool],
        actions: Sequence[Any] | Callable[[Any], Sequence[Any]],
        heuristic: Callable[[Any], float] | None,
        key: Callable[[Any], Hashable] | None,
    ) -> None:
        self.step, self.goal, self.actions, self.heuristic = step, goal, actions, heuristic
        self.key = key or freeze
        self.errors = 0
        self.first_error = ""
        self.t0 = time.monotonic()
        # parent links instead of per-node paths keep memory linear in states seen
        self.parent: dict[Hashable, tuple[Hashable | None, Any]] = {}
        self.depth: dict[Hashable, int] = {}
        self.best_key: Hashable = None
        self.best_h = float("inf")

    def safe(self, fn: Callable[..., Any], *args: Any) -> tuple[bool, Any]:
        try:
            return True, fn(*args)
        except Exception as e:  # model-written code: prune the branch, keep searching
            self.errors += 1
            if not self.first_error:
                name = getattr(fn, "__name__", repr(fn))
                self.first_error = f"{name}: {type(e).__name__}: {e}"[:200]
            return False, None

    def path_to(self, k: Hashable) -> list[Any]:
        out: list[Any] = []
        while k in self.parent:
            prev, act = self.parent[k]
            if prev is None:
                break
            out.append(act)
            k = prev
        out.reverse()
        return out

    def result(self, actions: list[Any] | None, expanded: int, stopped: str) -> PlanResult:
        partial = self.path_to(self.best_key) if actions is None else list(actions)
        return PlanResult(
            actions,
            expanded,
            stopped,
            self.errors,
            self.first_error,
            time.monotonic() - self.t0,
            partial,
        )

    def children(self, s: Any) -> list[tuple[Any, Any]]:
        if callable(self.actions):
            ok, acts = self.safe(self.actions, s)
            if not ok:
                return []
        else:
            acts = self.actions
        out = []
        for a in acts:
            ok, ns = self.safe(self.step, s, a)
            if ok and ns is not None:
                out.append((a, ns))
        return out

    def score(self, s: Any) -> float:
        if self.heuristic is None:
            return 0.0
        ok, h = self.safe(self.heuristic, s)
        return float(h) if ok else float("inf")

    def run(
        self, start: Any, max_nodes: int, time_limit_s: float, max_depth: int | None
    ) -> PlanResult:
        ok, at_goal = self.safe(self.goal, start)
        if ok and at_goal:
            return self.result([], 0, "goal_at_start")
        ok, k0 = self.safe(self.key, start)
        if not ok:
            return self.result(None, 0, "exhausted")
        self.parent[k0], self.depth[k0], self.best_key = (None, None), 0, k0
        tie = itertools.count()
        bfs = self.heuristic is None
        frontier: Any = deque([(k0, start)]) if bfs else [(self.score(start), next(tie), k0, start)]
        expanded = 0
        while frontier:
            if expanded >= max_nodes:
                return self.result(None, expanded, "max_nodes")
            if time.monotonic() - self.t0 > time_limit_s:
                return self.result(None, expanded, "time_limit")
            k, s = frontier.popleft() if bfs else heapq.heappop(frontier)[2:]
            expanded += 1
            d = self.depth[k]
            if max_depth is not None and d >= max_depth:
                continue
            for a, ns in self.children(s):
                ok, nk = self.safe(self.key, ns)
                if not ok or nk in self.parent:
                    continue
                self.parent[nk], self.depth[nk] = (k, a), d + 1
                ok, g = self.safe(self.goal, ns)
                if ok and g:
                    return self.result(self.path_to(nk), expanded, "goal")
                if bfs:
                    frontier.append((nk, ns))
                    continue
                h = self.score(ns)
                if h < self.best_h:
                    self.best_key, self.best_h = nk, h
                heapq.heappush(frontier, (d + 1 + h, next(tie), nk, ns))
        return self.result(None, expanded, "exhausted")


def plan(
    step: Callable[[Any, Any], Any],
    start: Any,
    goal: Callable[[Any], bool],
    actions: Sequence[Any] | Callable[[Any], Sequence[Any]],
    *,
    heuristic: Callable[[Any], float] | None = None,
    key: Callable[[Any], Hashable] | None = None,
    max_nodes: int = MAX_NODES,
    time_limit_s: float = TIME_LIMIT_S,
    max_depth: int | None = None,
) -> PlanResult:
    """Shortest action sequence from ``start`` to a state where ``goal`` holds.

    ``actions`` is a fixed list or a function of the state (for example, only the
    clickable cells). Breadth-first without ``heuristic``; A* with it (optimal when
    the heuristic never overestimates the remaining actions). ``best_partial`` is
    the path to the lowest-heuristic state seen when no goal is reached.
    """
    return _Search(step, goal, actions, heuristic, key).run(
        start, max_nodes, time_limit_s, max_depth
    )
