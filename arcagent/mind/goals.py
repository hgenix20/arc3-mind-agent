"""Goal guesses: predicates over objects that the agent proposes, tests and confirms.

A ``Goal`` is a pure predicate over a ``Scene`` (``match`` also reads the raw grid).
``GoalBook`` tracks each goal's status through a game: ``open`` goals are guesses still in
play; a goal that holds on a logged frame (``rule_out``) or on the current frame
(``note_reached``) while the level did not complete cannot be the win condition, so it is
``ruled_out`` or ``excluded``; a goal true in the frame before a level-up, or in the
predicted outcome of the winning move, is ``confirmed`` and its form is carried to the
next level as the first hypothesis. Only open goals change status, and ``propose`` skips
every goal already in ``status``, so a refuted guess is not re-proposed within a game and
a confirmed form is never excluded. A goal whose plans keep failing is ``stalled``
(``stall``, called by the agent after repeated failed attempts in one level) and ``open``
leaves it out; ``reopen_stalled`` puts it back in play on a level change. An open goal
the agent was pursuing that holds without a level-up can be set ``probing``
(``start_probe``) while the agent tries actions that might commit it; ``open`` leaves it
out, a level-up confirms it, and ``end_probe`` excludes it. A confirmed form
can stall too, for the rest of its level.

``GoalBook.propose`` makes the kinds the rule-book (``goal_rules``) marks proposed:
``reach``, ``clear``, ``match`` (from pairs of equal-size framed regions,
``regions.twin_regions``, when given the raw grid) and ``cover`` (every target of one kind
has its own mover or pushable thing on or beside it). ``count`` is evaluated but never
proposed. ``mover_sigs`` and ``goal_sigs`` name the sigs
a plan depends on; search and the fluid agent protect them from the stillness assumption.
Pure and deterministic; callers strip noise things from the scene before calling
``propose``.

Kinds: a reach goal's ``a`` and ``b`` match a thing by its sig or its kind
(``Thing.kind``), so a goal about a kind holds whatever look the thing shows. Mover and
controlled sets include kinds that have move facts. ``propose`` uses a thing's kind in
place of its sig when the two differ, so one sprite doesn't get a goal per look.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from arcagent.core.types import GRID_SIZE, Grid
from arcagent.mind.assign import ASSIGN_MAX, assign
from arcagent.mind.facts import Effect, FactMemory, clicked_sig
from arcagent.mind.goal_rules import proposed
from arcagent.mind.kinds import base_sig
from arcagent.mind.kinematics import Kinematics
from arcagent.mind.objects import Scene, Thing
from arcagent.mind.regions import twin_regions

GoalKind = Literal["reach", "clear", "match", "count", "cover"]
GoalStatus = Literal["open", "ruled_out", "excluded", "confirmed", "stalled", "probing"]
Region = tuple[int, int, int, int]  # (x0, y0, x1, y1), end exclusive

MAX_TARGET_AREA = GRID_SIZE * GRID_SIZE // 4  # bigger things are backgrounds or walls
MAX_REACH_TARGETS = 20  # per mover, fewest-cell sigs first
MAX_OPEN_GOALS = 200  # propose stops adding once this many goals are open
MAX_COVER_TARGETS = 5  # cover goals per mover, fewest-cell target sigs first
_MOVE_KINDS = frozenset({"move", "slide", "teleport"})


@dataclass(frozen=True, slots=True)
class Goal:
    """One guess. Unused fields keep their defaults so equal goals compare equal.

    ``reach`` and ``cover``: ``a``, ``b`` are sigs or kinds. ``clear``: ``color``.
    ``count``: ``color`` and ``n``.
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
            case "cover":
                return f"cover {self.a}>{self.b}"

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
            case "cover":
                return _cover_cost(scene, self.a, self.b) == 0

    def distance(
        self, scene: Scene, grid: Grid | None = None, kin: Kinematics | None = None
    ) -> int | None:
        """How far ``scene`` is from the goal; 0 exactly when ``holds``, None for no estimate.

        ``reach``: 0 when it holds; else, with ``kin``, the moves from ``a``'s top-left in
        ``kin.field`` (built on the first scene it saw for this pair: a search's root),
        when that is a positive number; else the Manhattan gap in cells between the bboxes
        of the nearest ``a`` and ``b`` less one (the last cell is the contact), at least 1.
        None when either is missing. ``clear``: things of the color left. ``count``:
        ``abs(n - count)``. ``match``: mismatched cells; None without a grid or with
        regions that are off the board or of different sizes. ``cover``: the least total,
        over a one-to-one matching of targets to movers, of each pair's cell gap less one
        (0 when they touch), plus ``GRID_SIZE`` per target left without a mover; 0 when no
        target is visible (all covered); None when targets show and no mover does. Cell gaps
        ignore walls and step sizes: a ranking of scenes, not a move count.
        """
        match self.kind:
            case "reach":
                return _reach_moves(scene, self.a, self.b, kin)
            case "clear":
                return sum(1 for t in scene.things if t.color == self.color)
            case "count":
                return abs(self.n - sum(1 for t in scene.things if t.color == self.color))
            case "match":
                return None if grid is None else _mismatched(grid, self.region_a, self.region_b)
            case "cover":
                return _cover_cost(scene, self.a, self.b)


def _reach_moves(scene: Scene, a: str, b: str, kin: Kinematics | None) -> int | None:
    """``reach``'s distance: moves from ``kin``'s field when positive, else the cell gap."""
    gap = _reach_distance(scene, a, b)
    if not gap or kin is None:
        return gap
    body = kin.body(scene, a)
    moves = kin.field(scene, a, b).get((body.x, body.y)) if body else None
    return moves if moves else gap


def _reach_distance(scene: Scene, a: str, b: str) -> int | None:
    mine = [t for t in scene.things if _is(t, a)]
    targets = [t for t in scene.things if _is(t, b)]
    if not mine or not targets:
        return None
    if _reaches(scene, a, b):
        return 0
    boxes = [(t, _bbox(t)) for t in mine]
    best: int | None = None
    for target in targets:
        bx0, by0, bx1, by1 = _bbox(target)
        for t, (ax0, ay0, ax1, ay1) in boxes:
            if t is target:
                continue
            gap = max(0, bx0 - ax1, ax0 - bx1) + max(0, by0 - ay1, ay0 - by1)
            best = gap if best is None else min(best, gap)
    return None if best is None else max(1, best - 1)


def _gap(a: Thing, b: Thing) -> int:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return max(0, bx0 - ax1, ax0 - bx1) + max(0, by0 - ay1, ay0 - by1)


def _cover_cost(scene: Scene, a: str, b: str) -> int | None:
    """``cover``'s distance (see ``Goal.distance``); None past ``assign.ASSIGN_MAX``."""
    targets = [t for t in scene.things if _is(t, b)]
    if not targets:
        return 0
    movers = [t for t in scene.things if _is(t, a)]
    if not movers or len(targets) > ASSIGN_MAX or len(movers) > ASSIGN_MAX:
        return None
    cost = [[float(max(0, _gap(m, t) - 1)) for m in movers] for t in targets]
    pairs = assign(cost, float("inf"))
    return int(sum(cost[i][j] for i, j in pairs)) + GRID_SIZE * (len(targets) - len(pairs))


def _mismatched(grid: Grid, ra: Region, rb: Region) -> int | None:
    for x0, y0, x1, y1 in (ra, rb):
        if not (0 <= x0 < x1 <= grid.shape[1] and 0 <= y0 < y1 <= grid.shape[0]):
            return None
    ax0, ay0, ax1, ay1 = ra
    bx0, by0, bx1, by1 = rb
    if (ax1 - ax0, ay1 - ay0) != (bx1 - bx0, by1 - by0):
        return None
    return int((grid[ay0:ay1, ax0:ax1] != grid[by0:by1, bx0:bx1]).sum())


def _bbox(t: Thing) -> tuple[int, int, int, int]:
    xs = [x for x, _ in t.cells]
    ys = [y for _, y in t.cells]
    return min(xs), min(ys), max(xs), max(ys)


def _is(t: Thing, name: str) -> bool:
    """Whether ``t`` answers to ``name``: its sig or its kind."""
    return name in (t.sig, t.kind)


def _reaches(scene: Scene, a: str, b: str) -> bool:
    """Some ``a`` cell overlaps a ``b`` cell or is 4-adjacent to one (a ring is not its hole)."""
    mine = [t for t in scene.things if _is(t, a)]
    for target in (t for t in scene.things if _is(t, b)):
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
    """Color in ``c{color}:{shape}`` (or a kind, ``k`` plus one); None for anything else."""
    head, sep, _ = base_sig(sig).partition(":")
    digits = head[1:]
    if not sep or not head.startswith("c") or not digits.isascii() or not digits.isdigit():
        return None
    return int(head[1:])


def _subject_sig(subject: str, label: str) -> str | None:
    """The sig a fact is about: the subject itself, or the clicked sig for ``self``."""
    if subject == "self":
        sig = clicked_sig(label) or ""
        return sig if _sig_color(sig) is not None else None
    return subject if _sig_color(subject) is not None else None


def mover_sigs(mem: FactMemory) -> frozenset[str]:
    """Every sig with a non-broken move, slide or teleport fact (``self`` facts included)."""
    return frozenset(_mover_sigs(mem))


def controlled_sigs(mem: FactMemory) -> frozenset[str]:
    """Sigs the player controls: their non-broken facts give different effects under at
    least two action labels (moves under A1 and none under A2, or up versus left).

    A thing that moves the same way whatever the action (a patrolling enemy) is not
    controlled. ``self`` facts count for the clicked sig; ``event`` and the action-wide
    ``any`` label don't.
    """
    seen: dict[str, dict[str, set[tuple[Effect, ...]]]] = {}
    for fact in mem.facts.values():
        label = fact.key[0]
        if fact.status == "broken" or label in ("event", "any"):
            continue
        sig = _subject_sig(fact.key[1], label)
        if sig is not None:
            seen.setdefault(sig, {}).setdefault(label, set()).add(fact.effects)
    out: set[str] = set()
    for sig, by_label in seen.items():
        if len(by_label) < 2:
            continue
        distinct = {effects for options in by_label.values() for effects in options}
        if len(distinct) >= 2:
            out.add(sig)
    return frozenset(out)


def goal_sigs(goal: Goal, scene: Scene, *, targets: bool = True) -> frozenset[str]:
    """The sigs a goal is about.

    ``reach`` and ``cover``: ``a``, plus ``b`` when ``targets``. ``clear`` and ``count``:
    the sigs of the things in ``scene`` with the goal's color. ``match``: none.
    """
    match goal.kind:
        case "reach" | "cover":
            return frozenset({goal.a, goal.b} if targets else {goal.a})
        case "clear" | "count":
            return frozenset(t.sig for t in scene.things if t.color == goal.color)
        case "match":
            return frozenset()


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


def _pushed_sigs(mem: FactMemory) -> list[str]:
    """Sigs with a non-broken ``pushed`` fact that moves them."""
    out: set[str] = set()
    for fact in mem.facts.values():
        if fact.key[2] != "pushed" or fact.status == "broken":
            continue
        if any(e.kind in _MOVE_KINDS for e in fact.effects) and _sig_color(fact.key[1]) is not None:
            out.add(fact.key[1])
    return sorted(out)


def _cover_guesses(
    scene: Scene,
    mem: FactMemory,
    shadowed: set[str],
    looks: dict[str, str],
    ranked: list[str],
) -> list[Goal]:
    """``cover(a, b)`` for each mover or pushable ``a`` on screen and each target sig ``b``
    with two or more things, at most ``MAX_COVER_TARGETS`` per mover, fewest cells first."""
    present = {t.sig for t in scene.things} | {t.kind for t in scene.things}
    many = {s for s in ranked if sum(1 for t in scene.things if t.sig == s) >= 2}
    out: list[Goal] = []
    for a in dict.fromkeys([*_mover_sigs(mem), *_pushed_sigs(mem)]):
        if a not in present or a in shadowed:
            continue
        targets = [b for b in ranked if b in many and b != a and looks.get(b) != a]
        out += [Goal("cover", a=a, b=b) for b in targets[:MAX_COVER_TARGETS]]
    return out


def _is_target(t: Thing) -> bool:
    x0, y0, x1, y1 = _bbox(t)
    return (x1 - x0 + 1) * (y1 - y0 + 1) <= MAX_TARGET_AREA


@dataclass(slots=True)
class GoalBook:
    """Status of every goal guessed this game, plus the confirmed forms carried across levels."""

    status: dict[Goal, GoalStatus] = field(default_factory=dict[Goal, GoalStatus])
    confirmed_forms: list[Goal] = field(default_factory=list[Goal])
    sources: dict[Goal, str] = field(default_factory=dict[Goal, str])
    stalled_forms: set[Goal] = field(default_factory=set[Goal])

    def propose(
        self, scene: Scene, mem: FactMemory, *, source: str, grid: Grid | None = None
    ) -> list[Goal]:
        """Add the heuristic guesses not yet in ``status``; return the new ones in order.

        ``reach(a, b)`` for every non-broken mover sig ``a`` present in the scene against
        every other sig ``b`` present, skipping targets whose bbox covers over a quarter of
        the board. ``clear(color)`` for the old and new color of every recolor fact and the
        color of every disappear fact. Reach goals come first, sorted by ``(a, b)``, then
        clear goals by color. Each mover gets at most ``MAX_REACH_TARGETS`` targets, fewest
        cells first (ties by sig), and at most ``MAX_OPEN_GOALS`` goals are open at once.
        ``source`` is kept in ``sources`` for logging.
        """
        present = {t.sig for t in scene.things} | {t.kind for t in scene.things}
        shadowed = {t.sig for t in scene.things if t.kind != t.sig}  # their kind stands in
        looks = {t.sig: t.kind for t in scene.things}  # never a target of its own kind
        size: dict[str, int] = {}
        for t in scene.things:
            if _is_target(t):
                size[t.sig] = min(size.get(t.sig, len(t.cells)), len(t.cells))
        ranked = sorted(size, key=lambda g: (size[g], g))
        guesses = [
            Goal("reach", a=a, b=b)
            for a in _mover_sigs(mem)
            if a in present and a not in shadowed
            for b in [g for g in ranked if g != a and looks.get(g) != a][:MAX_REACH_TARGETS]
        ]
        guesses += [Goal("clear", color=c) for c in _clear_colors(mem)]
        guesses += _cover_guesses(scene, mem, shadowed, looks, ranked)
        if grid is not None:
            guesses += [
                Goal("match", region_a=ra, region_b=rb) for ra, rb in twin_regions(scene, grid)
            ]
        guesses = [g for g in guesses if proposed(g.kind)]
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

    def rule_out(
        self,
        history: Sequence[Scene],
        grids: Sequence[Grid | None] = (),
        *,
        only: Collection[Goal] | None = None,
    ) -> list[Goal]:
        """Open goals true on any logged frame: no level-up followed, so they are ruled out.

        ``grids`` runs parallel to ``history`` (the raw frame of each scene, for ``match``);
        a missing entry is None. ``only`` limits the check to those goals (the ones just
        added): goals already open were checked against each frame as it arrived by
        ``note_reached``, so checking them again would only cost time. Call
        ``level_up(before)`` first on a level-up, so the winning goal is confirmed before a
        history that includes ``before`` can rule it out.
        """
        frames = [(s, grids[i] if i < len(grids) else None) for i, s in enumerate(history)]
        if only is None:
            check = list(self._open_goals())
        else:
            check = [g for g in only if self.status.get(g) == "open"]
        out = [g for g in check if any(g.holds(s, grid) for s, grid in frames)]
        for g in out:
            self.status[g] = "ruled_out"
        return out

    def note_reached(self, scene: Scene, grid: Grid | None = None) -> list[Goal]:
        """Open goals true in the current frame (``grid`` is its raw frame) are excluded."""
        out = [g for g in self._open_goals() if g.holds(scene, grid)]
        for g in out:
            self.status[g] = "excluded"
        return out

    def level_up(
        self, before: Scene, grid: Grid | None = None, *, after: Scene | None = None
    ) -> list[Goal]:
        """Open or probing goals true in the frame before a level-up (raw ``grid``) become
        confirmed.

        The frame a winning move returns already shows the next level, so the winning
        state is never seen. ``after``, the predicted outcome of that move, stands in for
        it: open goals true there are confirmed too.
        """
        live = [g for g, s in self.status.items() if s in ("open", "probing")]
        out = [g for g in live if g.holds(before, grid) or (after is not None and g.holds(after))]
        for g in out:
            self.status[g] = "confirmed"
            if g not in self.confirmed_forms:
                self.confirmed_forms.append(g)
        return out

    def start_probe(self, goal: Goal) -> bool:
        """Set an open goal ``probing``; True when it was open."""
        if self.status.get(goal) != "open":
            return False
        self.status[goal] = "probing"
        return True

    def end_probe(self) -> list[Goal]:
        """Exclude every probing goal (no commit worked); return them in order."""
        out = [g for g, s in self.status.items() if s == "probing"]
        for g in out:
            self.status[g] = "excluded"
        return out

    def stall(self, goal: Goal) -> bool:
        """Set an open goal (or a confirmed form) aside; True when it was in play."""
        if self.status.get(goal) == "open":
            self.status[goal] = "stalled"
            return True
        if goal in self.confirmed_forms and goal not in self.stalled_forms:
            self.stalled_forms.add(goal)
            return True
        return False

    def reopen_stalled(self) -> list[Goal]:
        """Put every stalled goal back in play (a level change); return them in order."""
        out = [g for g, s in self.status.items() if s == "stalled"]
        for g in out:
            self.status[g] = "open"
        out += [g for g in self.confirmed_forms if g in self.stalled_forms and g not in out]
        self.stalled_forms.clear()
        return out

    def open(self) -> list[Goal]:
        """Confirmed forms first, then open goals in insertion order; stalled ones left out."""
        forms = [g for g in self.confirmed_forms if g not in self.stalled_forms]
        return forms + list(self._open_goals())
