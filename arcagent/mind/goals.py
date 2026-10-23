"""Goal guesses: predicates over objects that the agent proposes, tests and confirms.

A ``Goal`` is a pure predicate over a ``Scene`` (``match`` also reads the raw grid).
``GoalBook`` tracks each goal's status through a game: ``open`` goals are guesses still in
play; a goal that holds on a logged frame (``rule_out``) or on the current frame
(``note_reached``) while the level did not complete cannot be the win condition, so it is
``ruled_out`` or ``excluded``; a goal true in the frame before a level-up is ``confirmed``
and its form is carried to the next level as the first hypothesis. Only open goals change
status, and ``propose`` skips every goal already in ``status``, so a refuted guess is not
re-proposed within a game and a confirmed form is never excluded.

Slice 1 proposes only ``reach`` and ``clear`` (see ``GoalBook.propose``). ``match`` and
``count`` are evaluated but never proposed. Pure and deterministic; callers strip noise
things from the scene before calling ``propose``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from arcagent.core.types import GRID_SIZE, Grid
from arcagent.mind.facts import FactMemory
from arcagent.mind.objects import Scene, Thing

GoalKind = Literal["reach", "clear", "match", "count"]
GoalStatus = Literal["open", "ruled_out", "excluded", "confirmed"]
Region = tuple[int, int, int, int]  # (x0, y0, x1, y1), end exclusive

MAX_TARGET_AREA = GRID_SIZE * GRID_SIZE // 4  # bigger things are backgrounds or walls
MAX_REACH_TARGETS = 20  # per mover, fewest-cell sigs first
MAX_OPEN_GOALS = 200  # propose stops adding once this many goals are open
_MOVE_KINDS = frozenset({"move", "slide", "teleport"})
_CLICK_PREFIX = "A6:"


@dataclass(frozen=True, slots=True)
class Goal:
    """One guess. Unused fields keep their defaults so equal goals compare equal.

    ``reach``: ``a``, ``b`` are sigs. ``clear``: ``color``. ``count``: ``color`` and ``n``.
    ``match``: two grid regions ``(x0, y0, x1, y1)``, end exclusive.
    """

    kind: GoalKind
    a: str = ""
    b: str = ""
    color: int = -1
    n: int = -1
    region_a: Region = (0, 0, 0, 0)
    region_b: Region = (0, 0, 0, 0)

    def id(self) -> str:
        """Deterministic text id with full sigs, e.g. ``reach c1:ab12cdef01>c4:cd34ef5678``."""
        match self.kind:
            case "reach":
                return f"reach {self.a}>{self.b}"
            case "clear":
                return f"clear c{self.color}"
            case "count":
                return f"count c{self.color}={self.n}"
            case "match":
                return f"match {self.region_a}={self.region_b}"

    def holds(self, scene: Scene, grid: Grid | None = None) -> bool:
        """Whether the predicate is true in ``scene``. ``match`` is False without a grid."""
        match self.kind:
            case "reach":
                return _reaches(scene, self.a, self.b)
            case "clear":
                return not any(t.color == self.color for t in scene.things)
            case "count":
                return sum(1 for t in scene.things if t.color == self.color) == self.n
            case "match":
                return grid is not None and _regions_equal(grid, self.region_a, self.region_b)


def _bbox(t: Thing) -> tuple[int, int, int, int]:
    xs = [x for x, _ in t.cells]
    ys = [y for _, y in t.cells]
    return min(xs), min(ys), max(xs), max(ys)


def _reaches(scene: Scene, a: str, b: str) -> bool:
    """Some ``a`` cell overlaps a ``b`` cell or is 4-adjacent to one (a ring is not its hole)."""
    mine = [t for t in scene.things if t.sig == a]
    for target in (t for t in scene.things if t.sig == b):
        reach = set(target.cells)
        for x, y in target.cells:
            reach.update(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
        if any(t is not target and not t.cells.isdisjoint(reach) for t in mine):
            return True
    return False


def _regions_equal(grid: Grid, ra: Region, rb: Region) -> bool:
    for x0, y0, x1, y1 in (ra, rb):
        if not (0 <= x0 < x1 <= grid.shape[1] and 0 <= y0 < y1 <= grid.shape[0]):
            return False
    ax0, ay0, ax1, ay1 = ra
    bx0, by0, bx1, by1 = rb
    if (ax1 - ax0, ay1 - ay0) != (bx1 - bx0, by1 - by0):
        return False
    return bool((grid[ay0:ay1, ax0:ax1] == grid[by0:by1, bx0:bx1]).all())


def _sig_color(sig: str) -> int | None:
    """Color in ``c{color}:{shape}``; None for anything else."""
    head, sep, _ = sig.partition(":")
    digits = head[1:]
    if not sep or not head.startswith("c") or not digits.isascii() or not digits.isdigit():
        return None
    return int(head[1:])


def _subject_sig(subject: str, label: str) -> str | None:
    """The sig a fact is about: the subject itself, or the clicked sig for ``self``."""
    if subject == "self":
        sig = label[len(_CLICK_PREFIX) :] if label.startswith(_CLICK_PREFIX) else ""
        return sig if _sig_color(sig) is not None else None
    return subject if _sig_color(subject) is not None else None


def _mover_sigs(mem: FactMemory) -> list[str]:
    sigs: set[str] = set()
    for fact in mem.facts.values():
        if fact.status == "broken" or not any(e.kind in _MOVE_KINDS for e in fact.effects):
            continue
        sig = _subject_sig(fact.key[1], fact.key[0])
        if sig is not None:
            sigs.add(sig)
    return sorted(sigs)


def _clear_colors(mem: FactMemory) -> list[int]:
    colors: set[int] = set()
    for fact in mem.facts.values():
        sig = _subject_sig(fact.key[1], fact.key[0])
        old = _sig_color(sig) if sig is not None else None
        if old is None:
            continue
        for e in fact.effects:
            if e.kind == "recolor" and e.color >= 0:
                colors.update((old, e.color))
            elif e.kind == "disappear":
                colors.add(old)
    return sorted(colors)


def _is_target(t: Thing) -> bool:
    x0, y0, x1, y1 = _bbox(t)
    return (x1 - x0 + 1) * (y1 - y0 + 1) <= MAX_TARGET_AREA


@dataclass(slots=True)
class GoalBook:
    """Status of every goal guessed this game, plus the confirmed forms carried across levels."""

    status: dict[Goal, GoalStatus] = field(default_factory=dict[Goal, GoalStatus])
    confirmed_forms: list[Goal] = field(default_factory=list[Goal])
    sources: dict[Goal, str] = field(default_factory=dict[Goal, str])

    def propose(self, scene: Scene, mem: FactMemory, *, source: str) -> list[Goal]:
        """Add the heuristic guesses not yet in ``status``; return the new ones in order.

        ``reach(a, b)`` for every non-broken mover sig ``a`` present in the scene against
        every other sig ``b`` present, skipping targets whose bbox covers over a quarter of
        the board. ``clear(color)`` for the old and new color of every recolor fact and the
        color of every disappear fact. Reach goals come first, sorted by ``(a, b)``, then
        clear goals by color. Each mover gets at most ``MAX_REACH_TARGETS`` targets, fewest
        cells first (ties by sig), and at most ``MAX_OPEN_GOALS`` goals are open at once.
        ``source`` is kept in ``sources`` for logging.
        """
        present = {t.sig for t in scene.things}
        size: dict[str, int] = {}
        for t in scene.things:
            if _is_target(t):
                size[t.sig] = min(size.get(t.sig, len(t.cells)), len(t.cells))
        ranked = sorted(size, key=lambda g: (size[g], g))
        guesses = [
            Goal("reach", a=a, b=b)
            for a in _mover_sigs(mem)
            if a in present
            for b in [g for g in ranked if g != a][:MAX_REACH_TARGETS]
        ]
        guesses += [Goal("clear", color=c) for c in _clear_colors(mem)]
        added: list[Goal] = []
        room = MAX_OPEN_GOALS - sum(1 for _ in self._open_goals())
        for goal in guesses:
            if len(added) >= room:
                break  # dropped goals stay out of status so a later call can add them
            if goal not in self.status:
                self.status[goal] = "open"
                self.sources[goal] = source
                added.append(goal)
        return added

    def _open_goals(self) -> Iterator[Goal]:
        return (g for g, s in self.status.items() if s == "open")

    def rule_out(self, history: Sequence[Scene]) -> list[Goal]:
        """Open goals true on any logged frame: no level-up followed, so they are ruled out.

        Call ``level_up(before)`` first on a level-up, so the winning goal is confirmed
        before a history that includes ``before`` can rule it out.
        """
        out = [g for g in self._open_goals() if any(g.holds(s) for s in history)]
        for g in out:
            self.status[g] = "ruled_out"
        return out

    def note_reached(self, scene: Scene) -> list[Goal]:
        """Open goals true in the current frame without a level-up are excluded."""
        out = [g for g in self._open_goals() if g.holds(scene)]
        for g in out:
            self.status[g] = "excluded"
        return out

    def level_up(self, before: Scene) -> list[Goal]:
        """Open goals true in the frame before a level-up become confirmed forms."""
        out = [g for g in self._open_goals() if g.holds(before)]
        for g in out:
            self.status[g] = "confirmed"
            if g not in self.confirmed_forms:
                self.confirmed_forms.append(g)
        return out

    def open(self) -> list[Goal]:
        """Confirmed forms first, then open goals in insertion order."""
        return list(self.confirmed_forms) + list(self._open_goals())
