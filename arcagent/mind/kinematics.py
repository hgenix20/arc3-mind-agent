"""Kinematics of controlled things: how far each action carries them, and where they can go.

``Kinematics`` reads a thing's gaits from its facts: for each simple action label (clicks,
``any`` and ``event`` excluded), the non-broken context-``any`` fact for the subject that
moves it (``move``: a fixed step) or slides it (``slide``: one cell at a time until the next
cell would be blocked). A subject is a sig or a kind (``Thing.kind``).

Walks treat the rest of the scene as fixed: a body fits at a top-left when every cell of
its shape is on the board, not padding, and of a color (``objects.color_at``: a thing's,
else the terrain's) outside the blockers learned for its sig and kind. ``reachable`` walks
forward from the thing's place, slides included. ``field`` walks backward from every
top-left where the body touches a target (overlaps it or is 4-adjacent to it) over the
fixed-step gaits only (the start of a slide can't be recovered from its end), giving the
moves to the target from each top-left; it is cached per (subject, target) for the life of
the object, so a search builds it once from its root scene. Every walk visits at most
``KIN_LIMIT`` top-lefts, and one object builds at most ``KIN_FIELDS`` fields (later pairs
get an empty field, and goal distances fall back to cell gaps): a search may score up to
200 goals, and the first ones (confirmed forms, the executive's) matter most. ``mover``
names the one controlled thing when the agent knows it; ``Goal.distance`` then routes it
through what a ``clear`` goal has left (``route.route``). ``tour`` walks it through a list
of stops for real: the visiting order from ``route.route`` over walk distances, then the
gaits of each leg, a leg ending where the body overlaps its stop. Pure.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from arcagent.core.types import GRID_SIZE
from arcagent.mind.facts import FactMemory, cell_owner
from arcagent.mind.objects import Cell, Scene, Thing, color_at
from arcagent.mind.route import route

KIN_LIMIT = 4096  # top-lefts visited per walk
KIN_FIELDS = 8  # distance fields one Kinematics builds (one per move in the agent)
TOUR_STOPS = 12  # most stops ``tour`` walks through (one walk per stop and leg)
_NOT_GAITS = ("any", "event")
_TOUCH = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


@dataclass(frozen=True, slots=True)
class Gait:
    """One action's displacement for a subject; ``slides`` steps (dx, dy) until blocked."""

    action: str
    dx: int
    dy: int
    slides: bool


class Kinematics:
    """Gaits, forward walks and cached distance fields over one ``FactMemory``."""

    def __init__(self, mem: FactMemory, mover: str | None = None) -> None:
        self.mem = mem
        self.mover = mover  # the one controlled thing's name, when the agent knows it
        self._fields: dict[tuple[str, str], dict[Cell, int]] = {}

    def gaits(self, name: str) -> tuple[Gait, ...]:
        """The subject's gaits, by action label."""
        out: list[Gait] = []
        for (label, subject, context), fact in sorted(self.mem.facts.items()):
            if subject != name or context != "any" or fact.status == "broken":
                continue
            if label in _NOT_GAITS or label.startswith("A6"):
                continue
            for e in fact.effects:
                if e.kind == "move" and (e.dx or e.dy):
                    out.append(Gait(label, e.dx, e.dy, slides=False))
                    break
                if e.kind == "slide" and (e.dx or e.dy):
                    out.append(Gait(label, _sign(e.dx), _sign(e.dy), slides=True))
                    break
        return tuple(out)

    def body(self, scene: Scene, name: str) -> Thing | None:
        """The subject's thing in ``scene`` (lowest id when several)."""
        mine = [t for t in scene.things if name in (t.sig, t.kind)]
        return min(mine, key=lambda t: t.id) if mine else None

    def _fits(self, scene: Scene, body: Thing, name: str) -> Callable[[Cell], bool]:
        """Whether ``body``'s shape fits with its top-left at a cell (memoized)."""
        offs = [(x - body.x, y - body.y) for x, y in body.cells]
        owner = {c: v for c, v in cell_owner(scene).items() if v[0] != body.id}
        stop: set[int] = set()
        for n in {name, body.sig, body.kind}:
            stop |= self.mem.blockers.get(n, set())
        memo: dict[Cell, bool] = {}

        def fits(tl: Cell) -> bool:
            hit = memo.get(tl)
            if hit is None:
                hit = all(_open(scene, owner, stop, (tl[0] + ox, tl[1] + oy)) for ox, oy in offs)
                memo[tl] = hit
            return hit

        return fits

    def reachable(self, scene: Scene, name: str, *, limit: int = KIN_LIMIT) -> dict[Cell, int]:
        """Top-left to fewest moves, walking forward from the subject's thing."""
        body = self.body(scene, name)
        if body is None:
            return {}
        start = (body.x, body.y)
        dist = {start: 0}
        gaits = self.gaits(name)
        fits = self._fits(scene, body, name)
        queue = deque([start])
        while queue and len(dist) < limit:
            here = queue.popleft()
            for g in gaits:
                there = _step(here, g, fits)
                if there is None or there in dist:
                    continue
                dist[there] = dist[here] + 1
                if len(dist) >= limit:
                    break
                queue.append(there)
        return dist

    def tree(
        self, scene: Scene, name: str, start: Cell | None = None, *, limit: int = KIN_LIMIT
    ) -> dict[Cell, tuple[Cell, Gait] | None]:
        """Forward walk from ``start`` (the subject's place by default): each top-left
        reached, with the top-left and gait it was first reached from (None at the start).
        Breadth first, so following the links back gives a fewest-moves path. Empty when
        the subject isn't on screen."""
        body = self.body(scene, name)
        if body is None:
            return {}
        return self._tree(
            self._fits(scene, body, name), self.gaits(name), start or (body.x, body.y), limit
        )

    def _tree(
        self, fits: Callable[[Cell], bool], gaits: Sequence[Gait], start: Cell, limit: int
    ) -> dict[Cell, tuple[Cell, Gait] | None]:
        links: dict[Cell, tuple[Cell, Gait] | None] = {start: None}
        queue = deque([start])
        while queue and len(links) < limit:
            here = queue.popleft()
            for g in gaits:
                there = _step(here, g, fits)
                if there is None or there in links:
                    continue
                links[there] = (here, g)
                if len(links) >= limit:
                    break
                queue.append(there)
        return links

    def tour(
        self, scene: Scene, name: str, stops: Sequence[frozenset[Cell]]
    ) -> tuple[list[Gait], list[int]] | None:
        """The subject's gaits through every stop and the order it visits them, shortest
        order by walk distance (``route.route``, exact up to its ``ROUTE_EXACT`` stops). A
        stop is reached when the body overlaps one of its cells. None when the subject is
        missing, a stop can't be reached, or there are more than ``TOUR_STOPS``."""
        body = self.body(scene, name)
        if body is None or not stops or len(stops) > TOUR_STOPS:
            return None
        fits = self._fits(scene, body, name)
        gaits = self.gaits(name)
        offs = [(x - body.x, y - body.y) for x, y in body.cells]
        stands = [{(cx - ox, cy - oy) for cx, cy in stop for ox, oy in offs} for stop in stops]
        stands = [{tl for tl in st if fits(tl)} for st in stands]
        if any(not st for st in stands):
            return None
        start = (body.x, body.y)
        trees: dict[Cell, dict[Cell, tuple[Cell, Gait] | None]] = {}

        def walk(src: Cell) -> dict[Cell, tuple[Cell, Gait] | None]:
            if src not in trees:
                trees[src] = self._tree(fits, gaits, src, KIN_LIMIT)
            return trees[src]

        def nearest(src: Cell, i: int) -> tuple[Cell, int] | None:
            links = walk(src)
            best: tuple[int, Cell] | None = None
            for tl in sorted(stands[i]):
                if tl in links:
                    d = len(_path(links, tl))
                    if best is None or d < best[0]:
                        best = (d, tl)
            return None if best is None else (best[1], best[0])

        firsts = [nearest(start, i) for i in range(len(stops))]
        if any(f is None for f in firsts):
            return None
        reps = [f[0] for f in firsts if f is not None]
        index = {tl: i for i, tl in enumerate(reps)}

        def dist(a: Cell, b: Cell) -> int | None:
            found = nearest(a, index[b])
            return None if found is None else found[1]

        found = route(start, reps, dist)
        if found is None:
            return None
        order = found[0]
        steps: list[Gait] = []
        here = start
        for i in order:
            leg = nearest(here, i)
            if leg is None:
                return None
            steps += _path(walk(here), leg[0])
            here = leg[0]
        return steps, order

    def steps_to(self, scene: Scene, name: str, target: frozenset[Cell]) -> int | None:
        """Fewest moves until the subject's thing touches ``target``; None when unreachable."""
        body = self.body(scene, name)
        if body is None:
            return None
        near = _touching(target)
        offs = [(x - body.x, y - body.y) for x, y in body.cells]
        found = [
            d
            for (tx, ty), d in self.reachable(scene, name).items()
            if any((tx + ox, ty + oy) in near for ox, oy in offs)
        ]
        return min(found) if found else None

    def field(self, scene: Scene, name: str, target: str) -> dict[Cell, int]:
        """Top-left to moves until the subject touches a thing named ``target`` (cached)."""
        key = (name, target)
        cached = self._fields.get(key)
        if cached is not None:
            return cached
        dist: dict[Cell, int] = {}
        body = self.body(scene, name)
        if body is not None and len(self._fields) < KIN_FIELDS:
            dist = self._walk_back(scene, body, name, target)
        self._fields[key] = dist
        return dist

    def _walk_back(self, scene: Scene, body: Thing, name: str, target: str) -> dict[Cell, int]:
        fits = self._fits(scene, body, name)
        cells = frozenset(
            c
            for t in scene.things
            if target in (t.sig, t.kind) and t.id != body.id
            for c in t.cells
        )
        offs = [(x - body.x, y - body.y) for x, y in body.cells]
        dist: dict[Cell, int] = {}
        queue: deque[Cell] = deque()
        for cx, cy in _touching(cells):
            for ox, oy in offs:
                tl = (cx - ox, cy - oy)
                if tl not in dist and fits(tl):
                    dist[tl] = 0
                    queue.append(tl)
        steps = [g for g in self.gaits(name) if not g.slides]
        while queue and len(dist) < KIN_LIMIT:
            here = queue.popleft()
            for g in steps:
                prev = (here[0] - g.dx, here[1] - g.dy)
                if prev not in dist and fits(prev):
                    dist[prev] = dist[here] + 1
                    queue.append(prev)
        return dist


def _open(scene: Scene, owner: dict[Cell, tuple[int, int]], stop: set[int], c: Cell) -> bool:
    """On the board, not padding, and not a blocker color."""
    x, y = c
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or c in scene.blocked:
        return False
    return color_at(scene, owner, c) not in stop


def _path(links: dict[Cell, tuple[Cell, Gait] | None], end: Cell) -> list[Gait]:
    """The gaits from a tree's start to ``end`` (which must be in it)."""
    out: list[Gait] = []
    link = links[end]
    while link is not None:
        prev, g = link
        out.append(g)
        link = links[prev]
    return out[::-1]


def _touching(cells: frozenset[Cell]) -> set[Cell]:
    return {(x + dx, y + dy) for x, y in cells for dx, dy in _TOUCH}


def _step(here: Cell, g: Gait, fits: Callable[[Cell], bool]) -> Cell | None:
    """Where ``g`` takes a body at ``here``; None when it can't move."""
    if not g.slides:
        there = (here[0] + g.dx, here[1] + g.dy)
        return there if fits(there) else None
    pos = here
    for _ in range(GRID_SIZE):
        nxt = (pos[0] + g.dx, pos[1] + g.dy)
        if not fits(nxt):
            break
        pos = nxt
    return pos if pos != here else None
