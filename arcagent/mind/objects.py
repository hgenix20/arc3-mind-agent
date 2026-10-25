"""Objects, scenes, tracking and diffs: the perception layer of the fluid agent.

Pure and deterministic. A ``Thing`` is a 4-connected single-color component with a
position-invariant signature and fewer than ``TERRAIN_MIN_CELLS`` cells, plus the
small decorations of other colors it carries (``_join_parts``: a piece). A ``Tracker``
gives things stable ids across frames and reports what each one did. Padding cells
(value -1) are never things: they land in ``Scene.blocked``. Callers must call
``Tracker.reset`` on a level change or RESET so a whole-screen replacement is not
reported as every object moving.

Terrain: a component of ``TERRAIN_MIN_CELLS`` or more cells (a floor, a wide wall) is
never a thing, because a large region a player crosses would otherwise "reshape" every
move. ``Scene.terrain`` holds the color of every cell no thing covers (background,
terrain components, padding -1, and background where noise is masked). A cell covered by
a thing gets the color last seen there with no thing on it (the tracker keeps one memory
grid per level, cleared by ``reset``), else the most common non-thing color among its 4
neighbors, else the background. The same memory keeps a split floor whole: a component
with a cell remembered as terrain of its color is terrain, whatever its size. Terrain is
not keyed or diffed: ``Scene.key()`` and ``SceneDiff`` ignore it, and it is excluded from
``Scene`` equality and hashing.
``color_at`` reads a cell's color: the covering thing's, else the terrain's.

Matching (``Tracker.update``): a thing with the same sig at the same top-left keeps its id
first. The rest go to one global assignment (``assign.assign``) over the pairs that pass
one of these tests, in this order of preference for ``Tracker.how``: ``prior`` (the caller's
predicted scene puts this id at a top-left within ``MATCH_RADIUS`` with this sig), ``sig``
(same sig within ``MATCH_RADIUS``), ``recolor`` (same shape, same top-left), ``reshape``
(same color, overlapping cells), ``near`` (same color, sizes within ``NEAR_SIZE``, within
``MATCH_RADIUS``: a sprite that turned while it moved). A pair of different colors with
no overlap and no prediction is never matched, so a thing vanishing as another appears
isn't a reshape. Cost: top-left distance from the anchor (the predicted thing, else the
old one), plus ``KIND_COST`` of the test that passed (so the order above wins over
distance), ``W_OVERLAP`` times (1 - cell IoU with the anchor), ``W_SIZE`` times relative size
change, ``W_COLOR`` for a color change and ``W_SHAPE`` for a shape change against the
anchor. With more than ``ASSIGN_MAX`` leftovers on a side, the greedy passes run instead
(same sig nearest first, then recolor, then reshape).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from arcagent.core.types import GRID_SIZE, Grid
from arcagent.mind.assign import ASSIGN_MAX, assign
from arcagent.synth.trace import background_color, components

MATCH_RADIUS = 16  # Manhattan distance between top-left corners for a same-sig match
NEAR_SIZE = 0.25  # a "near" match changes the cell count by at most this fraction
W_OVERLAP = 4.0  # match cost weights; the distance term weighs 1 per cell
W_SIZE = 2.0
W_COLOR = 3.0
W_SHAPE = 1.0
# Added per match kind, 20 apart so the kind outranks any distance: a thing that keeps its
# look and moves beats a different thing "recoloring" in place (a player on lava).
KIND_COST: dict[str, float] = {
    "prior": 0.0,
    "sig": 0.0,
    "recolor": 20.0,
    "reshape": 40.0,
    "near": 60.0,
}
TERRAIN_MIN_CELLS = 128  # a component this large is terrain, never a Thing
PART_SHARE = 0.25  # a decoration has at most this share of its host's cells
PIECE_FILL = 0.75  # host and parts fill at least this share of the host's bbox
PIECE_REPEATS = 2  # a piece look shows this many times in a frame before it counts
_NO_MEMORY = -2  # terrain memory value for a cell never seen uncovered (-1 is padding)

Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Thing:
    """A single-color component, with any decorations it carries. ``x, y`` is the bbox
    top-left; ``cells`` are absolute (x, y).

    ``kind`` is the thing's identity class across looks (``kinds.KindBook``): the first sig
    its class was seen with. Left empty it is the thing's own sig. ``paint`` lists
    ``(dx, dy, color)`` for the cells whose color isn't ``color``, offsets from the bbox
    top-left (empty for a plain thing); read colors through ``colored``.
    """

    id: int
    color: int
    shape: str
    sig: str
    x: int
    y: int
    cells: frozenset[Cell]
    kind: str = ""
    paint: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            object.__setattr__(self, "kind", self.sig)

    def colored(self) -> Iterator[tuple[Cell, int]]:
        """Each cell with its color."""
        if not self.paint:
            for c in self.cells:
                yield c, self.color
            return
        other = {(self.x + dx, self.y + dy): col for dx, dy, col in self.paint}
        for c in self.cells:
            yield c, other.get(c, self.color)


_UNSET: Grid = np.zeros((0, 0), dtype=np.int8)


@dataclass(frozen=True, slots=True)
class Scene:
    """Things sorted by id, the background color, and padding cells (the grid edge is implicit).

    ``terrain`` is a (64, 64) int8 grid of the color under the things (see the module
    docstring); read-only by convention and left out of equality and hashing. A scene
    built without one gets the background everywhere and -1 on ``blocked`` cells.

    ``selected`` and ``active`` are the hidden mode (``modes.ModeTracker``): the ids of
    selected things and of the active controller. Both are part of ``key()``.
    """

    things: tuple[Thing, ...]
    background: int
    blocked: frozenset[Cell]
    terrain: Grid = field(default_factory=lambda: _UNSET, compare=False)
    selected: frozenset[int] = frozenset()
    active: int | None = None

    def __post_init__(self) -> None:
        if self.terrain.size == 0:
            plain = np.full((GRID_SIZE, GRID_SIZE), self.background, dtype=np.int8)
            for x, y in self.blocked:
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    plain[y, x] = -1
            object.__setattr__(self, "terrain", plain)

    def key(self) -> tuple[tuple[str, int, int], ...]:
        """Sorted (sig, x, y), plus ("sel:" + sig, x, y) for each selected thing and
        ("active:" + sig, x, y) for the active one; ids are ignored so equal layouts dedupe."""
        out = [(t.sig, t.x, t.y) for t in self.things]
        if self.selected or self.active is not None:
            for t in self.things:
                if t.id in self.selected:
                    out.append(("sel:" + t.sig, t.x, t.y))
                if t.id == self.active:
                    out.append(("active:" + t.sig, t.x, t.y))
        return tuple(sorted(out))

    def by_id(self) -> dict[int, Thing]:
        return {t.id: t for t in self.things}

    def occupancy(self) -> dict[Cell, int]:
        """Cell to the id of the thing covering it."""
        return {c: t.id for t in self.things for c in t.cells}


ChangeKind = Literal["moved", "recolored", "reshaped", "disappeared", "appeared", "same"]


@dataclass(frozen=True, slots=True)
class Change:
    """What happened to one id: top-left shift ``dx, dy``; ``color``/``sig`` are the new state.

    ``size`` is the new cell count for ``moved`` and ``reshaped`` (-1 otherwise or unknown).
    """

    id: int
    kind: ChangeKind
    dx: int = 0
    dy: int = 0
    color: int = -1
    sig: str = ""
    size: int = -1


@dataclass(frozen=True, slots=True)
class SceneDiff:
    """One change per id present before or after, sorted by id."""

    changes: tuple[Change, ...]
    stray_cells: int  # changed cells outside any thing

    @property
    def empty(self) -> bool:
        return self.stray_cells == 0 and all(c.kind == "same" for c in self.changes)


def shape_of(cells: Iterable[Cell], paint: Iterable[tuple[int, int, int]] = ()) -> str:
    """First 10 hex of SHA-1 over the sorted cell offsets from the bbox top-left, and the
    sorted ``paint`` when there is any (a plain thing's hash doesn't change)."""
    pts = list(cells)
    if not pts:
        return ""
    x0 = min(x for x, _ in pts)
    y0 = min(y for _, y in pts)
    offs = sorted((x - x0, y - y0) for x, y in pts)
    marks = sorted(paint)
    key = repr((offs, marks)) if marks else repr(offs)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def _visible(grid: Grid, mask: frozenset[Cell]) -> tuple[Grid, int]:
    """Grid with masked cells read as background, and the background color."""
    bg = background_color(grid)
    if not mask or bg < 0:
        return grid, bg
    g = grid.copy()
    for x, y in mask:
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and g[y, x] >= 0:
            g[y, x] = bg
    return g, bg


def _things(
    grid: Grid, memory: Grid | None = None, pieces: frozenset[str] = frozenset()
) -> list[Thing]:
    """Components under ``TERRAIN_MIN_CELLS`` cells; larger ones are terrain.

    With a terrain ``memory`` (the tracker's), a component with any cell remembered as
    terrain of its own color is terrain too, so a floor cut in two by a thing standing
    across it stays terrain.
    """
    out: list[Thing] = []
    if background_color(grid) < 0:
        return out
    for yx in components(grid):
        if len(yx) >= TERRAIN_MIN_CELLS:
            continue
        color = int(grid[yx[0]])
        if memory is not None and any(int(memory[y, x]) == color for y, x in yx):
            continue
        cells = frozenset((x, y) for y, x in yx)
        shape = shape_of(cells)
        out.append(
            Thing(
                id=-1,
                color=color,
                shape=shape,
                sig=f"c{color}:{shape}",
                x=min(x for x, _ in cells),
                y=min(y for _, y in cells),
                cells=cells,
            )
        )
    return _join_parts(out, pieces)


def _bbox(t: Thing) -> tuple[int, int, int, int]:
    xs = [x for x, _ in t.cells]
    ys = [y for _, y in t.cells]
    return min(xs), min(ys), max(xs), max(ys)


def _host(i: int, things: list[Thing], at: dict[Cell, int]) -> int | None:
    """The largest touching thing of another color whose bbox holds all of ``things[i]``,
    with some cell of it on the bbox's rim, and has at least ``1 / PART_SHARE`` times its
    cells."""
    t = things[i]
    best: int | None = None
    for x, y in t.cells:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = at.get((x + dx, y + dy))
            if j is None or j == i or things[j].color == t.color:
                continue
            h = things[j]
            if len(t.cells) > PART_SHARE * len(h.cells):
                continue
            x0, y0, x1, y1 = _bbox(h)
            if not all(x0 <= a <= x1 and y0 <= b <= y1 for a, b in t.cells):
                continue
            if not any(a in (x0, x1) or b in (y0, y1) for a, b in t.cells):
                continue  # inside, off the rim: a selection ring doesn't own what it rings
            if best is None or len(h.cells) > len(things[best].cells):
                best = j
    return best


def _join_parts(things: list[Thing], pieces: frozenset[str] = frozenset()) -> list[Thing]:
    """Merge each small decoration into the sprite that carries it (a piece).

    A thing joins its ``_host``; a host takes its parts only when together they fill at
    least ``PIECE_FILL`` of its bbox, so a hollow frame never swallows what stands inside
    it. A piece look must show at least twice in the frame, or be in ``pieces`` (looks the
    tracker saw twice this level): a decoration repeated on copies of a sprite is part of
    its look, while one thing touching one sprite may well be its own (r11l's mover). The
    piece keeps the host's color and lists the parts' cells in ``paint``. Order: hosts
    where they stood, parts dropped.
    """
    at = {c: i for i, t in enumerate(things) for c in t.cells}
    host_of = {i: j for i in range(len(things)) if (j := _host(i, things, at)) is not None}
    parts: dict[int, list[int]] = {}
    for i, j in host_of.items():
        if j not in host_of:  # a host that is itself a part keeps to itself
            parts.setdefault(j, []).append(i)
    joined: dict[int, Thing] = {}
    members: dict[int, list[int]] = {}
    for j, mine in parts.items():
        h = things[j]
        x0, y0, x1, y1 = _bbox(h)
        cells = h.cells.union(*(things[i].cells for i in mine))
        if len(cells) < PIECE_FILL * (x1 - x0 + 1) * (y1 - y0 + 1):
            continue
        paint = tuple(
            sorted((x - x0, y - y0, things[i].color) for i in mine for x, y in things[i].cells)
        )
        shape = shape_of(cells, paint)
        joined[j] = Thing(-1, h.color, shape, f"c{h.color}:{shape}", x0, y0, cells, paint=paint)
        members[j] = mine
    seen = Counter(t.sig for t in joined.values())
    keep = {j for j, t in joined.items() if seen[t.sig] >= PIECE_REPEATS or t.sig in pieces}
    dropped = {i for j in keep for i in members[j]}
    return [joined[i] if i in keep else t for i, t in enumerate(things) if i not in dropped]


def extract(grid: Grid, mask: frozenset[Cell] = frozenset()) -> list[Thing]:
    """Things in components order with id -1. Masked cells read as background."""
    return _things(_visible(grid, mask)[0])


def color_at(scene: Scene, owner: Mapping[Cell, tuple[int, int]], cell: Cell) -> int:
    """The covering thing's color (``owner`` maps cell to (id, color)), else the terrain's.

    -1 for off-board and padding cells.
    """
    hit = owner.get(cell)
    if hit is not None:
        return hit[1]
    x, y = cell
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        return -1
    return int(scene.terrain[y, x])


def _terrain(visible: Grid, things: Iterable[Thing], memory: Grid, bg: int) -> Grid:
    """Terrain colors for one frame; updates ``memory`` in place from uncovered cells."""
    covered = np.zeros(visible.shape, dtype=bool)
    for t in things:
        for x, y in t.cells:
            covered[y, x] = True
    terrain = visible.copy()
    memory[~covered] = visible[~covered]
    known = covered & (memory != _NO_MEMORY)
    terrain[known] = memory[known]
    h, w = visible.shape
    ys, xs = np.nonzero(covered & ~known)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        votes: Counter[int] = Counter()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and not covered[ny, nx] and visible[ny, nx] >= 0:
                votes[int(visible[ny, nx])] += 1
        terrain[y, x] = min(votes, key=lambda c: (-votes[c], c)) if votes else bg
    terrain.flags.writeable = False
    return terrain


def _with_id(t: Thing, tid: int) -> Thing:
    return Thing(tid, t.color, t.shape, t.sig, t.x, t.y, t.cells, paint=t.paint)


Pairs = dict[int, int]
MatchHow = Literal["still", "sig", "recolor", "reshape", "near", "prior"]


def _match_same_sig(before: list[Thing], after: list[Thing], pairs: Pairs, used: set[int]) -> None:
    """Same sig: stationary pairs first, then nearest top-left within the radius.

    Stationary pairs (same sig, same top-left) are unique, so a dict finds them in
    linear time. Only the leftovers pay for the greedy nearest search.
    """
    here = {(a.sig, a.x, a.y): j for j, a in enumerate(after)}
    for i, b in enumerate(before):
        j = here.get((b.sig, b.x, b.y))
        if j is not None:
            pairs[i] = j
            used.add(j)
    by_sig: dict[str, list[int]] = defaultdict(list)
    for j, a in enumerate(after):
        if j not in used:
            by_sig[a.sig].append(j)
    cands: list[tuple[int, int, int]] = []
    for i, b in enumerate(before):
        if i in pairs:
            continue
        for j in by_sig.get(b.sig, ()):
            d = abs(after[j].x - b.x) + abs(after[j].y - b.y)
            if d <= MATCH_RADIUS:
                cands.append((d, i, j))
    _take(sorted(cands), pairs, used)


def _match_recolor(before: list[Thing], after: list[Thing], pairs: Pairs, used: set[int]) -> None:
    """Same shape at the same top-left."""
    at: dict[tuple[str, int, int], int] = {}
    for j, a in enumerate(after):
        if j not in used:
            at.setdefault((a.shape, a.x, a.y), j)
    cands = [
        (0, i, at[(b.shape, b.x, b.y)])
        for i, b in enumerate(before)
        if i not in pairs and (b.shape, b.x, b.y) in at
    ]
    _take(cands, pairs, used)


def _match_reshape(before: list[Thing], after: list[Thing], pairs: Pairs, used: set[int]) -> None:
    """Same color with overlapping cells, largest overlap first."""
    cands: list[tuple[int, int, int]] = []
    for i, b in enumerate(before):
        if i in pairs:
            continue
        for j, a in enumerate(after):
            ov = len(b.cells & a.cells) if j not in used and a.color == b.color else 0
            if ov:
                cands.append((-ov, i, j))
    _take(sorted(cands), pairs, used)


def _take(cands: list[tuple[int, int, int]], pairs: Pairs, used: set[int]) -> None:
    for _, i, j in cands:
        if i not in pairs and j not in used:
            pairs[i] = j
            used.add(j)


def _match(
    before: list[Thing], after: list[Thing], prior: Mapping[int, Thing] | None = None
) -> tuple[Pairs, dict[int, MatchHow]]:
    """Map index in ``before`` to index in ``after``, and how each ``before`` id matched."""
    pairs: Pairs = {}
    used: set[int] = set()
    how: dict[int, MatchHow] = {}
    here = {(a.sig, a.x, a.y): j for j, a in enumerate(after)}
    for i, b in enumerate(before):
        j = here.get((b.sig, b.x, b.y))
        if j is not None and j not in used:
            pairs[i] = j
            used.add(j)
            how[b.id] = "still"
    rows = [i for i in range(len(before)) if i not in pairs]
    cols = [j for j in range(len(after)) if j not in used]
    if not rows or not cols:
        return pairs, how
    if len(rows) > ASSIGN_MAX or len(cols) > ASSIGN_MAX:
        _greedy(before, after, pairs, used, how)
        return pairs, how
    prior = prior or {}
    scored = [
        [_pair_cost(before[i], after[j], prior.get(before[i].id)) for j in cols] for i in rows
    ]
    cost = [[c[0] for c in row] for row in scored]
    for r, c in assign(cost, math.inf):
        i, j = rows[r], cols[c]
        pairs[i] = j
        how[before[i].id] = scored[r][c][1]
    return pairs, how


def _overlap(a: frozenset[Cell], b: frozenset[Cell]) -> float:
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


def _pair_cost(b: Thing, a: Thing, p: Thing | None) -> tuple[float, MatchHow]:
    """Cost and kind of matching ``b`` (before) to ``a`` (after); ``p`` is ``b`` predicted.

    Infinite cost when no test passes (the module docstring lists them).
    """
    dist_b = abs(a.x - b.x) + abs(a.y - b.y)
    small, large = sorted((len(a.cells), len(b.cells)))
    kind: MatchHow | None = None
    if p is not None and a.sig == p.sig and abs(a.x - p.x) + abs(a.y - p.y) <= MATCH_RADIUS:
        kind = "prior"
    elif a.sig == b.sig and dist_b <= MATCH_RADIUS:
        kind = "sig"
    elif a.shape == b.shape and dist_b == 0:
        kind = "recolor"
    elif a.color == b.color and not a.cells.isdisjoint(b.cells):
        kind = "reshape"
    elif a.color == b.color and dist_b <= MATCH_RADIUS and large - small <= NEAR_SIZE * large:
        kind = "near"
    if kind is None:
        return math.inf, "near"
    anchor = p if p is not None else b
    cost = (
        KIND_COST[kind]
        + abs(a.x - anchor.x)
        + abs(a.y - anchor.y)
        + W_OVERLAP * (1.0 - _overlap(a.cells, anchor.cells))
        + W_SIZE * (large - small) / large
        + W_COLOR * (a.color != anchor.color)
        + W_SHAPE * (a.shape != anchor.shape)
    )
    return cost, kind


def _greedy(
    before: list[Thing], after: list[Thing], pairs: Pairs, used: set[int], how: dict[int, MatchHow]
) -> None:
    """The fallback for big leftovers: same sig, then recolor, then reshape, greedily."""
    passes: tuple[tuple[Callable[[list[Thing], list[Thing], Pairs, set[int]], None], MatchHow], ...]
    passes = (
        (_match_same_sig, "sig"),
        (_match_recolor, "recolor"),
        (_match_reshape, "reshape"),
    )
    for step, kind in passes:
        known = set(pairs)
        step(before, after, pairs, used)
        for i in set(pairs) - known:
            how[before[i].id] = kind


def _classify(b: Thing, a: Thing) -> Change:
    dx, dy = a.x - b.x, a.y - b.y
    if a.sig == b.sig:
        if dx or dy:
            return Change(b.id, "moved", dx, dy, a.color, a.sig, len(a.cells))
        return Change(b.id, "same", dx, dy, a.color, a.sig)
    if a.shape == b.shape:
        return Change(b.id, "recolored", dx, dy, a.color, a.sig)
    return Change(b.id, "reshaped", dx, dy, a.color, a.sig, len(a.cells))


class Tracker:
    """Stable ids across frames. The first frame after ``reset`` yields no diff.

    Keeps a per-level terrain memory: the color last seen at each cell with no thing on it.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._pieces: set[str] = set()  # piece looks seen twice this level
        self._scene: Scene | None = None
        self._grid: Grid | None = None
        self._memory: Grid | None = None
        self.how: dict[int, MatchHow] = {}  # how each id kept its track on the last update

    def reset(self) -> None:
        """Forget the previous frame, the terrain memory and the piece looks; restart ids
        at 1."""
        self._next_id = 1
        self._pieces = set()
        self._scene = None
        self._grid = None
        self._memory = None
        self.how = {}

    def _terrain_of(self, visible: Grid, things: Iterable[Thing], bg: int) -> Grid:
        if self._memory is None or self._memory.shape != visible.shape:
            self._memory = np.full(visible.shape, _NO_MEMORY, dtype=np.int8)
        return _terrain(visible, things, self._memory, bg)

    def update(
        self, grid: Grid, mask: frozenset[Cell], prior: Scene | None = None
    ) -> tuple[Scene, SceneDiff | None]:
        """The new scene and its diff from the last one (None right after ``reset``).

        ``prior`` is the caller's predicted scene for this frame, ids as in the last
        scene; it lets a thing keep its id through a predicted change of look or a jump.
        """
        visible, bg = _visible(grid, mask)
        memory = (
            self._memory if self._memory is not None and self._memory.shape == grid.shape else None
        )
        fresh = _things(visible, memory, frozenset(self._pieces))
        self._pieces.update(t.sig for t in fresh if t.paint)
        ys, xs = np.nonzero(grid < 0)
        blocked = frozenset(zip(xs.tolist(), ys.tolist(), strict=True))
        prev = self._scene
        if prev is None:
            first = [_with_id(t, self._take_id()) for t in fresh]
            scene = Scene(tuple(first), bg, blocked, self._terrain_of(visible, first, bg))
            self._scene, self._grid = scene, grid.copy()
            return scene, None

        expected = {t.id: t for t in prior.things} if prior is not None else None
        pairs, self.how = _match(list(prev.things), fresh, expected)
        matched_after = set(pairs.values())
        things: list[Thing] = []
        changes: list[Change] = []
        for i, b in enumerate(prev.things):
            j = pairs.get(i)
            if j is None:
                changes.append(Change(b.id, "disappeared"))
                continue
            things.append(_with_id(fresh[j], b.id))
            changes.append(_classify(b, fresh[j]))
        for j, a in enumerate(fresh):
            if j not in matched_after:
                t = _with_id(a, self._take_id())
                things.append(t)
                changes.append(Change(t.id, "appeared", color=t.color, sig=t.sig))
        things.sort(key=lambda t: t.id)
        changes.sort(key=lambda c: c.id)
        scene = Scene(tuple(things), bg, blocked, self._terrain_of(visible, things, bg))
        stray = self._stray(prev, scene, grid, mask)
        self._scene, self._grid = scene, grid.copy()
        return scene, SceneDiff(tuple(changes), stray)

    def _take_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _stray(self, prev: Scene, now: Scene, grid: Grid, mask: frozenset[Cell]) -> int:
        """Changed cells that no thing covered before or after, ignoring masked cells."""
        old = self._grid
        if old is None or old.shape != grid.shape:
            return 0
        ys, xs = np.nonzero(old != grid)
        covered: set[Cell] = set(mask)
        for t in prev.things:
            covered |= t.cells
        for t in now.things:
            covered |= t.cells
        return sum(1 for c in zip(xs.tolist(), ys.tolist(), strict=True) if c not in covered)


def click_cell(t: Thing) -> Cell:
    """The click cell of a thing: its cell nearest the centroid (ties by smallest cell)."""
    cells = sorted(t.cells)
    cx = sum(x for x, _ in cells) / len(cells)
    cy = sum(y for _, y in cells) / len(cells)
    return min(cells, key=lambda c: ((c[0] - cx) ** 2 + (c[1] - cy) ** 2, c))
