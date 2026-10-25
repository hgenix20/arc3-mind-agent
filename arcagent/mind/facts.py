"""Fact memory: short facts about what actions do to objects, with a table of contents.

A fact is keyed by ``(action label, subject, context)`` and holds a tuple of structured
effects the predictor can apply. ``FactMemory.toc`` maps behavior tags (root mechanic
words plus phrases like "moves until blocked") to fact keys, so an object that behaves
like a known one is predicted from a same-shape sibling's fact at once.

``learn`` turns one observed transition into one ``record`` call per fact key. Context
rules pick the key before recording, which is how a wall turns a predicted move into a
``blocked`` fact without charging the move fact a miss. A miss never rewrites a fact's
effects; a fact breaks after ``FactMemory.broken_after`` misses in a row (default
``BROKEN_MISSES``; the fluid agent's metacognition module sets it) and the next observation
re-seeds it. ``slide`` and ``teleport`` rewrites happen only through the explicit rules
in ``learn``. Things that move together as one rigid sprite (same shift, 4-adjacent
before the move) are a co-moving group: none of them is pushed by, blocked by, or a
blocker of another. Pure and deterministic; callers strip noise things before ``learn``.

Blocking reads cell colors, terrain included (``objects.color_at``): a wall can be the
background or a terrain region, not only a thing. ``FactMemory.walkable`` holds, per sig,
the colors its things have moved onto; a walkable color is never a blocker for that sig.
Touch and near contexts stay thing colors only: terrain contacts are constant and would
charge the safe sets every move.

Kinds (``kinds.KindBook``, ``Thing.kind``): facts stay keyed by a thing's look (its sig).
For a thing whose kind differs from its sig, ``learn`` also records each observation under
the kind, with a move or a change of look normalized to "and its look becomes X" (so the
same action gives the same kind fact from any look), records death facts under the kind
too, and copies the look's blockers and walkable colors to the kind. ``lookup`` takes the
kind: a usable look fact wins, then a usable kind fact, then a broken look fact.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, Action
from arcagent.mind.objects import Cell, Change, Scene, SceneDiff, Thing, color_at

EffectKind = Literal[
    "move",
    "slide",
    "appear",
    "disappear",
    "recolor",
    "reshape",
    "toggle_group",
    "teleport",
    "none",
    "game_over",
    "level_up",
]
FactKey = tuple[str, str, str]  # (action label, subject, context)
SPAWN = "spawn"  # subject of the per-action spawn fact (label, SPAWN, "any")
Status = Literal["thin", "solid", "broken"]
RecordResult = Literal["new", "hit", "miss"]
Owner = Mapping[Cell, tuple[int, int]]  # cell -> (thing id, color)

THIN_BELOW = 5  # a fact is solid once this many observations agreed with it
BROKEN_MISSES = 2  # a fact is broken after this many misses in a row
EVIDENCE_KEEP = 20  # a fact keeps the move ids of its last this-many observations
KIN_KEEP = 8  # sigs kept per tracked thing (FactMemory.kin)

_DIRS: dict[tuple[int, int], str] = {
    (-1, 0): "left",
    (1, 0): "right",
    (0, -1): "up",
    (0, 1): "down",
}


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def _move_text(dx: int, dy: int) -> str:
    name = _DIRS.get((_sign(dx), _sign(dy)))
    if name is None or (dx and dy):
        return f"moves by ({dx}, {dy})"
    n = abs(dx) + abs(dy)
    return f"moves {n} cell{'' if n == 1 else 's'} {name}"


@dataclass(frozen=True, slots=True)
class Effect:
    """One structured effect. Unused fields keep their defaults so equal effects compare equal."""

    kind: EffectKind
    dx: int = 0
    dy: int = 0
    color: int = -1
    sig: str = ""
    at: tuple[int, int] = (-1, -1)

    def text(self) -> str:
        """Short English for the effect, e.g. "moves 1 cell left" or "no change"."""
        match self.kind:
            case "move":
                out = _move_text(self.dx, self.dy)
            case "slide":
                out = f"slides {_DIRS.get((self.dx, self.dy), (self.dx, self.dy))} until blocked"
            case "appear":
                out = f"spawns {self.sig} at {self.at}"
            case "disappear":
                out = "disappears"
            case "recolor":
                out = f"turns color {self.color}"
            case "reshape":
                out = f"reshapes to {self.sig}"
            case "toggle_group":
                out = f"every {self.sig} turns color {self.color}"
            case "teleport":
                out = f"teleports to {self.at}"
            case "none":
                out = "no change"
            case "game_over":
                out = "ends the game"
            case "level_up":
                out = "completes the level"
        return out


_NONE = (Effect("none"),)


def _abbrev(subject: str) -> str:
    """``c1:ab12cdef01`` -> ``c1:ab12..``; ``self``/``any`` and short names stay whole."""
    color, sep, shape = subject.partition(":")
    if not sep or len(shape) <= 4:
        return subject
    return f"{color}:{shape[:4]}.."


@dataclass(slots=True)
class Fact:
    """A short fact.

    ``hits`` counts agreeing observations since the fact was seeded (the seed included).
    ``misses`` is the lifetime miss total, shown in ``line``; ``streak`` is the misses in a
    row, which a hit resets and which decides ``broken``. ``level`` is the level the fact
    was created on (``FactMemory.level`` at its first ``record``), so rule reuse across
    levels can be measured. ``broken_after`` is the misses-in-a-row threshold, copied from
    ``FactMemory.broken_after`` when the fact is made and kept in step by its setter.
    """

    key: FactKey
    effects: tuple[Effect, ...]
    evidence: list[int]  # move ids of the last EVIDENCE_KEEP observations against this key
    hits: int = 0
    misses: int = 0
    tags: frozenset[str] = frozenset()
    streak: int = 0
    level: int = 0
    broken_after: int = BROKEN_MISSES

    @property
    def status(self) -> Status:
        if self.streak >= self.broken_after:
            return "broken"
        return "solid" if self.hits >= THIN_BELOW else "thin"

    def line(self) -> str:
        """One line, e.g. ``A3 c1:ab12.. any: moves 1 cell left [solid 6/0]``."""
        label, subject, context = self.key
        what = "; ".join(e.text() for e in self.effects)
        score = f"[{self.status} {self.hits}/{self.misses}]"
        return f"{label} {_abbrev(subject)} {context}: {what} {score}"

    def note(self, move: int) -> None:
        """Add a move id to the evidence, keeping only the last ``EVIDENCE_KEEP``."""
        self.evidence.append(move)
        del self.evidence[:-EVIDENCE_KEEP]


_KIND_TAGS: dict[str, frozenset[str]] = {
    "move": frozenset({"move"}),
    "slide": frozenset({"move", "moves until blocked"}),
    "none": frozenset({"no_effect"}),
    "disappear": frozenset({"collect"}),
    "recolor": frozenset({"toggle"}),
    "toggle_group": frozenset({"toggle"}),
    "appear": frozenset({"spawn"}),
    "game_over": frozenset({"die"}),
    "level_up": frozenset({"win"}),
    "teleport": frozenset({"teleport"}),
    "reshape": frozenset({"reshape"}),
}
_CONTEXT_TAGS: dict[str, str] = {"blocked": "block", "pushed": "push"}


def _tags(key: FactKey, effects: Iterable[Effect]) -> set[str]:
    """Behavior tags for a fact. Every fact gets at least one, so every fact is in the toc."""
    tags: set[str] = set()
    for e in effects:
        tags |= _KIND_TAGS[e.kind]
    context_tag = _CONTEXT_TAGS.get(key[2])
    if context_tag is not None:
        tags.add(context_tag)
    return tags


def _shape_of_subject(subject: str) -> str | None:
    """The shape part of a sig subject; None for ``self``, ``any`` and ``SPAWN``."""
    _, sep, shape = subject.partition(":")
    return shape if sep else None


class FactMemory:
    """Facts by key, a table of contents by behavior tag, and colors that blocked each sig.

    ``landed`` remembers the top-left where the last ``any``-context move observation of a
    key ended (the teleport rule), and ``last_move`` that observation's effect and whether
    it ended blocked (the slide rule); ``new_level`` clears both. After each transition that
    didn't end the game, ``safe_touch`` gains, per sig, the colors its things overlapped
    and ``safe_near`` the colors they sat 4-adjacent to without overlapping (the two modes
    of ``touch_contexts``). Spawn facts live under subject ``SPAWN`` and are never returned
    for another subject. ``blockers`` maps a sig to the cell colors (thing or terrain) that
    stopped it; ``walkable`` maps a sig to the colors its things moved onto, and a color in
    ``walkable`` is never in ``blockers`` for the same sig.

    ``level`` is the current level, set by the agent; a new fact is stamped with it.
    ``missed`` holds the keys that took a miss since the last ``new_level`` (the fluid
    sleep's winning-facts check reads it before the agent calls ``new_level``).
    ``broken_after`` is the misses-in-a-row threshold for ``Fact.status``; setting it
    updates every stored fact.
    """

    def __init__(self) -> None:
        self._broken_after = BROKEN_MISSES
        self.level = 0
        self.missed: set[FactKey] = set()
        self.facts: dict[FactKey, Fact] = {}
        self.toc: dict[str, set[FactKey]] = {}
        self.blockers: dict[str, set[int]] = {}
        self.walkable: dict[str, set[int]] = {}
        self.landed: dict[FactKey, Cell] = {}
        self.last_move: dict[FactKey, tuple[Effect, bool]] = {}
        self.safe_touch: dict[str, set[int]] = {}
        self.safe_near: dict[str, set[int]] = {}
        # (label, context, shape) -> keys; built alongside the toc for borrowing
        self._siblings: dict[tuple[str, str, str], set[FactKey]] = {}
        # track id -> the sigs that thing has carried since the tracker last reset, oldest
        # first; the agent keeps it (note_kin, kin_reset). Not a fact: forget leaves it.
        self.kin: dict[int, list[str]] = {}
        self._kin_last: tuple[Thing, ...] = ()

    @property
    def broken_after(self) -> int:
        return self._broken_after

    @broken_after.setter
    def broken_after(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"broken_after must be at least 1, got {n}")
        self._broken_after = n
        for fact in self.facts.values():
            fact.broken_after = n

    def new_level(self) -> None:
        """Forget absolute positions and this level's misses; called on a level change.

        Facts stay.
        """
        self.landed.clear()
        self.last_move.clear()
        self.missed.clear()

    def forget(self) -> None:
        """Clear every fact and everything learned with them (toc, blockers, walkable,
        safe sets, positions, misses). ``level`` and ``broken_after`` stay.
        """
        for table in (
            self.facts,
            self.toc,
            self.blockers,
            self.walkable,
            self.landed,
            self.last_move,
            self.safe_touch,
            self.safe_near,
            self._siblings,
        ):
            table.clear()
        self.missed.clear()

    def delete(self, key: FactKey) -> None:
        """Remove one fact with its toc, sibling and position entries; no-op when absent."""
        fact = self.facts.pop(key, None)
        if fact is None:
            return
        for tag in fact.tags:
            node = self.toc.get(tag)
            if node is not None:
                node.discard(key)
                if not node:
                    del self.toc[tag]
        shape = _shape_of_subject(key[1])
        if shape is not None:
            sib = self._siblings.get((key[0], key[2], shape))
            if sib is not None:
                sib.discard(key)
                if not sib:
                    del self._siblings[(key[0], key[2], shape)]
        self.landed.pop(key, None)
        self.last_move.pop(key, None)
        self.missed.discard(key)

    def get(self, key: FactKey) -> Fact | None:
        return self.facts.get(key)

    def note_kin(self, things: Iterable[Thing]) -> None:
        """Append each thing's sig to its track's history when it differs from the last.

        A split or merge gives a piece a new track id or ends one, so a thing also takes
        the sigs of same-color things from the last call whose cells it covers, when it is
        new or they're gone (largest overlap first, ahead of its own history).
        """
        now = tuple(things)
        alive = {t.id for t in now}
        owner = {c: p for p in self._kin_last for c in p.cells}
        before = {p.id: tuple(self.kin.get(p.id, ())) for p in self._kin_last}
        for t in now:
            new = t.id not in self.kin
            sigs = self.kin.setdefault(t.id, [])
            overlap: dict[int, int] = {}
            for c in t.cells:
                p = owner.get(c)
                if p is None or p.id == t.id or p.color != t.color:
                    continue
                if new or p.id not in alive:
                    overlap[p.id] = overlap.get(p.id, 0) + 1
            donated = [
                s
                for pid in sorted(overlap, key=lambda i: (-overlap[i], i))
                for s in before.get(pid, ())
            ]
            if donated:
                sigs[:0] = [s for s in dict.fromkeys(donated) if s not in sigs]
            if not sigs or sigs[-1] != t.sig:
                sigs.append(t.sig)
            del sigs[:-KIN_KEEP]
        self._kin_last = now

    def kin_reset(self) -> None:
        """Track ids restart with the tracker; their histories go with them."""
        self.kin.clear()
        self._kin_last = ()

    def earlier_sigs(self, t: Thing) -> tuple[str, ...]:
        """Sigs the same tracked thing carried before, newest first, without its own."""
        return tuple(s for s in reversed(self.kin.get(t.id, ())) if s != t.sig)

    def lookup(
        self, label: str, subject: str, context: str, kind: str | None = None
    ) -> tuple[Fact, bool] | None:
        """Most specific fact for a subject and whether it was borrowed from a sibling.

        Order: exact key, the subject's ``any`` fact (with a ``kind`` other than the
        subject, either only when not broken), the kind's exact then ``any`` fact when not
        broken, a broken fact of the subject, a non-broken fact of a same-shape subject
        (same context before ``any``; most hits, then lowest key), then the action-wide
        ``(label, "any", "any")`` fact unless it is a spawn fact.
        """
        own = [
            f
            for k in ((label, subject, context), (label, subject, "any"))
            if (f := self.facts.get(k)) is not None
        ]
        if kind is not None and kind != subject:
            usable = next((f for f in own if f.status != "broken"), None)
            if usable is not None:
                return usable, False
            for key in ((label, kind, context), (label, kind, "any")):
                fact = self.facts.get(key)
                if fact is not None and fact.status != "broken":
                    return fact, False
        if own:
            return own[0], False
        borrowed = self._borrow(label, subject, context)
        if borrowed is not None:
            return borrowed, True
        wide = self.facts.get((label, "any", "any"))
        if wide is None or any(e.kind == "appear" for e in wide.effects):
            return None
        return wide, False

    def _borrow(self, label: str, subject: str, context: str) -> Fact | None:
        shape = _shape_of_subject(subject)
        if shape is None:
            return None
        for ctx in (context, "any") if context != "any" else ("any",):
            cands = [
                self.facts[k]
                for k in self._siblings.get((label, ctx, shape), ())
                if k[1] != subject and self.facts[k].status != "broken"
            ]
            if cands:
                return min(cands, key=lambda f: (-f.hits, f.key))
        return None

    def record(
        self,
        key: FactKey,
        effects: tuple[Effect, ...],
        move: int,
        *,
        tags: frozenset[str] = frozenset(),
    ) -> RecordResult:
        """Add one observation. Equal effects are a hit, different ones a miss.

        Callers apply context rules first; a miss here is never absorbed and never
        replaces the fact's effects. The first observation after a fact went broken
        re-seeds it (effects replaced, hits 1, streak 0) and returns ``"new"``. ``tags``
        adds behavior tags the effects alone can't show (``rotate`` needs cell counts).
        """
        fact = self.facts.get(key)
        if fact is None:
            fact = Fact(
                key, effects, [move], hits=1, level=self.level, broken_after=self._broken_after
            )
            self.facts[key] = fact
            self._index(fact, _tags(key, effects) | tags)
            shape = _shape_of_subject(key[1])
            if shape is not None:
                self._siblings.setdefault((key[0], key[2], shape), set()).add(key)
            return "new"
        fact.note(move)
        if fact.status == "broken":
            fact.effects, fact.hits, fact.streak = effects, 1, 0
            self._index(fact, _tags(key, effects) | tags)
            return "new"
        if fact.effects == effects:
            fact.hits += 1
            fact.streak = 0
            if not tags <= fact.tags:
                self._index(fact, set(fact.tags | tags))
            return "hit"
        fact.misses += 1
        fact.streak += 1
        self.missed.add(key)
        return "miss"

    def revise(self, key: FactKey, effects: tuple[Effect, ...], move: int) -> RecordResult:
        """Rewrite an existing fact's effects (slide, teleport) and count the observation a hit."""
        fact = self.facts[key]
        fact.effects = effects
        fact.hits += 1
        fact.streak = 0
        fact.note(move)
        self._index(fact, _tags(key, effects))
        return "hit"

    def _index(self, fact: Fact, tags: set[str]) -> None:
        for tag in fact.tags - tags:
            node = self.toc.get(tag)
            if node is not None:
                node.discard(fact.key)
                if not node:
                    del self.toc[tag]
        for tag in tags:
            self.toc.setdefault(tag, set()).add(fact.key)
        fact.tags = frozenset(tags)

    def by_tag(self, tag: str) -> list[Fact]:
        """Facts under a behavior tag, sorted by key."""
        return [self.facts[k] for k in sorted(self.toc.get(tag, ()))]


def _clicked(scene: Scene, action: Action) -> Thing | None:
    if action.id != COORD_ACTION_ID:
        return None
    point = (action.x, action.y)
    return next((t for t in scene.things if point in t.cells), None)


def click_ring(scene: Scene, thing: Thing, owner: Owner | None = None) -> int:
    """Most common color on the one-cell ring just outside ``thing``'s bbox.

    Each ring cell reads ``color_at`` (a covering thing's color, else the terrain's);
    off-board and padding cells don't vote. Ties go to the lower color; -1 when no cell
    votes. Pass ``owner=cell_owner(scene)`` when one is at hand.
    """
    xs = [x for x, _ in thing.cells]
    ys = [y for _, y in thing.cells]
    x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    ring = [(x, y) for x in range(x0, x1 + 1) for y in (y0, y1)]
    ring += [(x, y) for y in range(y0 + 1, y1) for x in (x0, x1)]
    table = cell_owner(scene) if owner is None else owner
    votes: Counter[int] = Counter()
    for c in ring:
        color = color_at(scene, table, c)
        if color >= 0:
            votes[color] += 1
    return min(votes, key=lambda c: (-votes[c], c)) if votes else -1


def action_label(action: Action, before: Scene, owner: Owner | None = None) -> str:
    """``"A3"`` for a simple action; ``"A6:<clicked sig>@<ring>"`` or ``"A6:none"``.

    ``ring`` is ``click_ring`` of the clicked thing: identical things in different places
    (inside a bordered panel or out on the background) are different click labels, since
    a click's effect often depends on where the thing sits. ``owner`` is an optional
    ``cell_owner(before)`` to reuse.
    """
    if action.id != COORD_ACTION_ID:
        return f"A{action.id}"
    thing = _clicked(before, action)
    if thing is None:
        return "A6:none"
    return f"A6:{thing.sig}@{click_ring(before, thing, owner)}"


def clicked_sig(label: str) -> str | None:
    """The clicked sig in a click label (``"A6:<sig>@<ring>"``); None otherwise or for none."""
    if not label.startswith("A6:"):
        return None
    sig = label[3:].partition("@")[0]
    return None if sig == "none" else sig


def cell_owner(scene: Scene) -> dict[Cell, tuple[int, int]]:
    """Cell to ``(thing id, color)`` for every thing cell; build once per scene."""
    return {c: (t.id, t.color) for t in scene.things for c in t.cells}


def _neighbors(cells: frozenset[Cell]) -> set[Cell]:
    return {(x + dx, y + dy) for x, y in cells for dx, dy in _DIRS}


def _contexts(
    cells: frozenset[Cell], owner: Owner, skip: frozenset[int]
) -> tuple[set[int], set[int]]:
    """Colors overlapping ``cells`` and colors 4-adjacent but not overlapping; ``skip`` ids."""
    over: set[int] = set()
    over_ids: set[int] = set()
    for c in cells:
        hit = owner.get(c)
        if hit is not None and hit[0] not in skip:
            over.add(hit[1])
            over_ids.add(hit[0])
    near: set[int] = set()
    for c in _neighbors(cells) - cells:
        hit = owner.get(c)
        if hit is not None and hit[0] not in skip and hit[0] not in over_ids:
            near.add(hit[1])
    return over, near - over


def touch_contexts(
    thing_cells: frozenset[Cell], scene: Scene, own_id: int, *, owner: Owner | None = None
) -> list[str]:
    """Touch contexts of a thing whose cells are ``thing_cells`` in ``scene``.

    ``touch:<c>`` for each color of another thing overlapping the cells, then ``near:<c>``
    for each color of another thing 4-adjacent without overlapping, minus colors already
    under ``touch``. Each list is sorted by color. ``own_id`` is skipped (-1 skips nothing).
    Pass ``owner=cell_owner(scene)`` when calling many times on one scene; the cost is
    then linear in the thing's cells.
    """
    table = cell_owner(scene) if owner is None else owner
    over, near = _contexts(thing_cells, table, frozenset({own_id}))
    return [f"touch:{c}" for c in sorted(over)] + [f"near:{c}" for c in sorted(near)]


@dataclass(slots=True)
class _Obs:
    """One subject's observation in one transition, before grouping by key.

    ``mates`` are the ids of its co-moving group (itself included); ``group_cells`` and
    ``group_dest`` are the group's cells before and after.
    """

    thing: Thing
    subject: str
    effects: tuple[Effect, ...]
    shift: tuple[int, int] | None  # displacement when the thing moved (or moved and reshaped)
    dest: frozenset[Cell]  # cells after the change; empty when it disappeared
    context: str = "any"
    tags: frozenset[str] = field(default_factory=frozenset[str])
    mates: frozenset[int] = field(default_factory=frozenset[int])
    group_cells: frozenset[Cell] = field(default_factory=frozenset[Cell])
    group_dest: frozenset[Cell] = field(default_factory=frozenset[Cell])

    def __post_init__(self) -> None:
        self.mates = frozenset({self.thing.id})
        self.group_cells = self.thing.cells
        self.group_dest = self.dest


def _shifted(cells: Iterable[Cell], dx: int, dy: int) -> frozenset[Cell]:
    return frozenset((x + dx, y + dy) for x, y in cells)


def _observe(change: Change, thing: Thing, subject: str, after: dict[int, Thing]) -> _Obs:
    now = after.get(thing.id)
    dx, dy = change.dx, change.dy
    dest = now.cells if now is not None else _shifted(thing.cells, dx, dy)
    match change.kind:
        case "moved":
            return _Obs(thing, subject, (Effect("move", dx=dx, dy=dy),), (dx, dy), dest)
        case "disappeared":
            return _Obs(thing, subject, (Effect("disappear"),), None, frozenset())
        case "recolored":
            return _Obs(thing, subject, (Effect("recolor", color=change.color),), None, dest)
        case "reshaped":
            reshape = Effect("reshape", sig=change.sig)
            same_count = now is not None and len(now.cells) == len(thing.cells)
            tags = frozenset({"rotate"}) if same_count else frozenset[str]()
            if dx or dy:
                effects = (Effect("move", dx=dx, dy=dy), reshape)
                return _Obs(thing, subject, effects, (dx, dy), dest, tags=tags)
            return _Obs(thing, subject, (reshape,), None, dest, tags=tags)
        case _:  # "same"; "appeared" never reaches here
            return _Obs(thing, subject, _NONE, None, thing.cells)


def _observations(
    before: Scene, action: Action, diff: SceneDiff, after: Scene | None
) -> list[_Obs]:
    old = before.by_id()
    new = after.by_id() if after is not None else {}
    clicked = _clicked(before, action)
    out: list[_Obs] = []
    for change in diff.changes:
        thing = old.get(change.id)
        if change.kind == "appeared" or thing is None:
            continue
        subject = "self" if clicked is not None and clicked.id == thing.id else thing.sig
        out.append(_observe(change, thing, subject, new))
    return out


def _group(items: list[_Obs], key: Callable[[_Obs], Hashable | None]) -> None:
    """Join items with equal non-None ``key`` whose old cells are 4-adjacent; set group fields.

    Linear in total cells: one cell map, then union-find over neighbor hits.
    """
    keyed = [(o, k) for o in items if (k := key(o)) is not None]
    parent = list(range(len(keyed)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    at: dict[Cell, int] = {c: i for i, (o, _) in enumerate(keyed) for c in o.thing.cells}
    for i, (o, k) in enumerate(keyed):
        for c in _neighbors(o.thing.cells):
            j = at.get(c)
            if j is not None and j != i and keyed[j][1] == k:
                parent[find(i)] = find(j)
    members: dict[int, list[_Obs]] = {}
    for i, (o, _) in enumerate(keyed):
        members.setdefault(find(i), []).append(o)
    for group in members.values():
        if len(group) < 2:
            continue
        mates = frozenset(o.thing.id for o in group)
        cells = frozenset[Cell]().union(*(o.thing.cells for o in group))
        dest = frozenset[Cell]().union(*(o.dest for o in group))
        for o in group:
            o.mates, o.group_cells, o.group_dest = mates, cells, dest


@dataclass(frozen=True, slots=True)
class _Room:
    """What can stop a step in one scene: padding, each covered cell's id and color, terrain."""

    scene: Scene
    owner: dict[Cell, tuple[int, int]]

    @staticmethod
    def of(scene: Scene) -> _Room:
        return _Room(scene, cell_owner(scene))

    def stopped(
        self,
        cells: frozenset[Cell],
        step: tuple[int, int],
        skip: frozenset[int],
        stop: Collection[int] = (),
    ) -> tuple[bool, set[int]]:
        """Whether one step from ``cells`` hits the edge, padding, a thing not in ``skip``,
        or a terrain cell whose color is in ``stop``; also the colors of the things hit.
        """
        hit = False
        colors: set[int] = set()
        blocked = self.scene.blocked
        for x, y in _shifted(cells, *step) - cells:
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or (x, y) in blocked:
                hit = True
                continue
            other = self.owner.get((x, y))
            if other is not None:
                if other[0] not in skip:
                    hit = True
                    colors.add(other[1])
            elif stop and int(self.scene.terrain[y, x]) in stop:
                hit = True
        return hit, colors

    def wall(
        self,
        cells: frozenset[Cell],
        step: tuple[int, int],
        skip: frozenset[int],
        exempt: Collection[int],
    ) -> tuple[bool, set[int]]:
        """The wall rule's view of one step: (shut, blocker colors).

        Shut when the step hits the edge, padding, a thing not in ``skip``, or a terrain
        cell whose color is not in ``exempt``. The colors are every destination color
        (``color_at``: things and terrain) not in ``exempt``; padding and the edge add none.
        """
        shut = False
        colors: set[int] = set()
        for c in _shifted(cells, *step) - cells:
            other = self.owner.get(c)
            if other is not None and other[0] in skip:
                continue
            color = color_at(self.scene, self.owner, c)
            if color < 0 or c in self.scene.blocked:
                shut = True
                continue
            if other is not None:
                shut = True
            if color not in exempt:
                shut = True
                colors.add(color)
        return shut, colors


def _unit(e: Effect) -> tuple[int, int]:
    return _sign(e.dx), _sign(e.dy)


def _fact_step(mem: FactMemory, key: FactKey) -> tuple[int, int] | None:
    """The one-step displacement a stored move or slide fact predicts, if any."""
    fact = mem.facts.get(key)
    for e in fact.effects if fact is not None else ():
        if e.kind == "move":
            return e.dx, e.dy
        if e.kind == "slide":
            return _unit(e)
    return None


def _stays_put(mem: FactMemory, label: str, o: _Obs) -> bool:
    """Whether the subject's own non-broken ``any`` fact says it stays put under ``label``."""
    fact = mem.facts.get((label, o.subject, "any"))
    return fact is not None and fact.effects == _NONE and fact.status != "broken"


def _split_off_pushed(mem: FactMemory, label: str, obs: list[_Obs]) -> None:
    """Group co-movers; a member known to stay put under ``label`` is pushed, not sprite.

    The first shove of a never-seen box is ambiguous and stays a rigid group; once the
    box has been seen standing still under this action, a shove is a push. A push needs
    a pusher: the split happens only in a group where at least one member has no
    stay-put fact for ``label``. When every member stays put by its facts (a sprite
    that was pressed against a wall before it ever moved under this action), they all
    record under ``any``, which charges the stale ``none`` facts until they re-seed.
    """
    _group(obs, lambda o: o.shift)
    put = {o.thing.id for o in obs if len(o.mates) > 1 and _stays_put(mem, label, o)}
    split = [
        o
        for o in obs
        if o.thing.id in put and any(m not in put for m in o.mates if m != o.thing.id)
    ]
    if not split:
        return
    for o in obs:
        o.mates, o.group_cells, o.group_dest = frozenset({o.thing.id}), o.thing.cells, o.dest
    for o in split:
        o.context = "pushed"
    pushed = {o.thing.id for o in split}
    _group(obs, lambda o: None if o.thing.id in pushed else o.shift)


def _mark_pushed(mem: FactMemory, label: str, obs: list[_Obs]) -> None:
    """A mover whose old cells a non-mate mover with the same shift entered was pushed.

    Not when its own ``any`` fact already moves it by that shift. Linear: one map from
    destination cells to movers.
    """
    movers = [o for o in obs if o.shift is not None]
    entered: dict[Cell, list[_Obs]] = {}
    for p in movers:
        for c in p.dest:
            entered.setdefault(c, []).append(p)
    for o in movers:
        if _fact_step(mem, (label, o.subject, "any")) == o.shift:
            continue
        pushers = (p for c in o.thing.cells for p in entered.get(c, ()))
        if any(p.thing.id not in o.mates and p.shift == o.shift for p in pushers):
            o.context = "pushed"


def _note_walkable(mem: FactMemory, room: _Room, obs: list[_Obs]) -> None:
    """Each mover's sig gains the colors it moved onto; those leave its blockers.

    A destination cell counts when it is new to the thing and no thing that moved in this
    transition (a mate, a pushed box) covered it in ``before``: a pushed box's color is
    not floor. A co-moving group shares its floor: every member's sig gains the colors
    any member moved onto, since the wall rule reads the whole group's destination.
    """
    moved = {o.thing.id for o in obs if o.shift is not None}
    own: dict[int, set[int]] = {}
    for o in obs:
        if o.shift is None:
            continue
        colors: set[int] = set()
        for c in o.dest - o.thing.cells:
            other = room.owner.get(c)
            if other is not None and other[0] in moved:
                continue
            color = color_at(room.scene, room.owner, c)
            if color >= 0:
                colors.add(color)
        own[o.thing.id] = colors
    for o in obs:
        if o.thing.id not in own:
            continue
        colors = set[int]().union(*(own.get(m, set()) for m in o.mates))
        if not colors:
            continue
        sig = o.thing.sig
        mem.walkable.setdefault(sig, set()).update(colors)
        stop = mem.blockers.get(sig)
        if stop:
            stop -= colors


def _mark_blocked(mem: FactMemory, label: str, room: _Room, obs: list[_Obs]) -> None:
    """Wall rule: a ``none`` where a move was expected and the way was shut is ``blocked``.

    The expected step is the subject's move fact, else its free peers' majority shift.
    Still things expecting the same step and 4-adjacent form one group, so a sprite's
    parts never block each other. The way is shut by the edge, padding, a thing outside
    the group, or a terrain color that is not walkable for the sig or a mate's sig. Every
    destination color that is not the subject's own, a group-mate's or walkable for the
    group joins the blockers.
    """
    peer_shift: dict[str, Counter[tuple[int, int]]] = {}
    for o in obs:
        if o.shift is not None and o.context == "any":
            peer_shift.setdefault(o.subject, Counter())[o.shift] += 1
    expected: dict[int, tuple[int, int]] = {}
    for o in obs:
        if o.effects != _NONE:
            continue
        step = _fact_step(mem, (label, o.subject, "any"))
        if step is None and o.subject in peer_shift:
            step = peer_shift[o.subject].most_common(1)[0][0]
        if step is not None:
            expected[o.thing.id] = step
    still = [o for o in obs if o.thing.id in expected]
    _group(still, lambda o: expected[o.thing.id])
    color_of = {o.thing.id: o.thing.color for o in obs}
    sig_of = {o.thing.id: o.thing.sig for o in obs}
    for o in still:
        sig = o.thing.sig
        exempt = {color_of[i] for i in o.mates if i in color_of}
        for i in o.mates:  # the group shares its floor: any member's walkable exempts
            exempt |= mem.walkable.get(sig_of.get(i, sig), set())
        hit, colors = room.wall(o.group_cells, expected[o.thing.id], o.mates, exempt)
        if hit:
            o.context = "blocked"
            mem.blockers.setdefault(sig, set()).update(colors)


def _majority(group: list[_Obs]) -> tuple[tuple[Effect, ...], _Obs]:
    counts = Counter(o.effects for o in group)
    effects = counts.most_common(1)[0][0]
    return effects, next(o for o in group if o.effects == effects)


def _same_direction(a: Effect, b: Effect) -> bool:
    return _unit(a) == _unit(b) and _unit(a) != (0, 0)


def _record_move(
    mem: FactMemory,
    key: FactKey,
    seen: Effect,
    rep: _Obs,
    room: _Room,
    *,
    move: int,
) -> None:
    """Record an ``any``-context move, applying the slide and teleport rewrites.

    Slide: a stored ``move`` and an observed move in the same direction with a different
    magnitude, where the observed move ended blocked and so did the longer of the two.
    Teleport: a different displacement landing where the previous observation landed.
    """
    landing = (rep.thing.x + seen.dx, rep.thing.y + seen.dy)
    stop = mem.blockers.get(rep.thing.sig, set())
    ends_blocked = room.stopped(rep.group_dest, _unit(seen), rep.mates, stop)[0]
    fact = mem.facts.get(key)
    old = fact.effects[0] if fact is not None and len(fact.effects) == 1 else None
    last_at = mem.landed.get(key)
    last = mem.last_move.get(key)
    mem.landed[key] = landing
    mem.last_move[key] = (seen, ends_blocked)
    if old is None:
        mem.record(key, (seen,), move)
        return
    slide = (Effect("slide", dx=_unit(seen)[0], dy=_unit(seen)[1]),)
    longer_old = abs(old.dx) + abs(old.dy) > abs(seen.dx) + abs(seen.dy)
    longer_blocked = not longer_old or last == (old, True)
    if old == slide[0] and ends_blocked:
        mem.record(key, slide, move)
    elif (
        old.kind == "move"
        and old != seen
        and _same_direction(old, seen)
        and ends_blocked
        and longer_blocked
    ):
        mem.revise(key, slide, move)
    elif old.kind == "teleport" and old.at == landing:
        mem.record(key, (old,), move)
    elif old.kind == "move" and old != seen and last_at == landing:
        mem.revise(key, (Effect("teleport", at=landing),), move)
    else:
        mem.record(key, (seen,), move)


def _record_groups(
    mem: FactMemory, label: str, room: _Room, obs: list[_Obs], move: int
) -> list[FactKey]:
    groups: dict[FactKey, list[_Obs]] = {}
    for o in obs:
        groups.setdefault((label, o.subject, o.context), []).append(o)
    for key, group in groups.items():
        effects, rep = _majority(group)
        if key[2] == "any" and len(effects) == 1 and effects[0].kind == "move":
            _record_move(mem, key, effects[0], rep, room, move=move)
            continue
        tags = frozenset[str]().union(*(o.tags for o in group if o.effects == effects))
        mem.record(key, effects, move, tags=tags)
    return list(groups)


def _record_appear(
    mem: FactMemory, label: str, diff: SceneDiff, after: Scene | None, move: int
) -> list[FactKey]:
    now = after.by_id() if after is not None else {}
    spawned: list[Effect] = []
    for c in diff.changes:
        if c.kind == "appeared":
            t = now.get(c.id)
            spawned.append(Effect("appear", sig=c.sig, at=(t.x, t.y) if t else (-1, -1)))
    if not spawned:
        return []
    key = (label, SPAWN, "any")
    mem.record(key, tuple(sorted(spawned, key=lambda e: (e.sig, e.at))), move)
    return [key]


def _kind_observations(obs: list[_Obs], after: Scene | None) -> list[_Obs]:
    """Each observation of a thing whose kind isn't its sig, again under the kind.

    A move or a change of look becomes "moves (if it did) and its look becomes X", whatever
    the look before, so the kind fact is the same from every look. Clicked things (subject
    ``self``) are left out: their facts key on the click label.
    """
    now = after.by_id() if after is not None else {}
    out: list[_Obs] = []
    for o in obs:
        t = o.thing
        if o.subject == "self" or t.kind == t.sig:
            continue
        effects = o.effects
        changed = any(e.kind in ("move", "reshape") for e in effects)
        new = now.get(t.id)
        if changed and new is not None:
            kept = tuple(e for e in effects if e.kind != "reshape")
            effects = (*kept, Effect("reshape", sig=new.sig))
        k = replace(o, subject=t.kind, effects=effects)
        k.mates, k.group_cells, k.group_dest = o.mates, o.group_cells, o.group_dest
        out.append(k)
    return out


def _share_with_kinds(mem: FactMemory, obs: list[_Obs]) -> None:
    """Copy each look's blockers and walkable colors to its kind (walkable wins)."""
    for o in obs:
        sig, kind = o.thing.sig, o.thing.kind
        if kind == sig:
            continue
        if sig in mem.walkable:
            mem.walkable.setdefault(kind, set()).update(mem.walkable[sig])
        if sig in mem.blockers:
            mem.blockers.setdefault(kind, set()).update(mem.blockers[sig])
        if kind in mem.blockers:
            mem.blockers[kind] -= mem.walkable.get(kind, set())


def _learn_touch(mem: FactMemory, before: Scene, obs: list[_Obs], move: int) -> list[FactKey]:
    """Game over: each mover gets a death fact per likely deadly color at its destination.

    Candidates are ``touch:`` colors (overlap in ``before``), else ``near:`` colors;
    groupmates never count. Colors already safe in that mode for the sig are dropped
    unless every candidate is safe.
    """
    owner = cell_owner(before)
    keys: list[FactKey] = []
    for o in obs:
        if o.shift is None:
            continue
        sig = o.thing.sig
        over, near = _contexts(o.dest, owner, o.mates)
        mode, cands, safe = (
            ("touch", over, mem.safe_touch) if over else ("near", near, mem.safe_near)
        )
        fresh = cands - safe.get(sig, set())
        subjects = (sig,) if o.thing.kind == sig else (sig, o.thing.kind)
        for color in sorted(fresh or cands):
            for subject in subjects:
                key = ("any", subject, f"{mode}:{color}")
                if key not in keys:
                    keys.append(key)
    for key in keys:
        mem.record(key, (Effect("game_over"),), move)
    return keys


def _note_safe(
    mem: FactMemory, before: Owner, after: Owner, obs: list[_Obs], move: int
) -> list[FactKey]:
    """After a safe transition: grow the safe sets and charge a miss to matching death facts.

    Overlap is read against ``before`` (two things never share a cell within one frame,
    so overlap only shows against where things were); adjacency against ``after``.
    """
    refuted: list[FactKey] = []
    for o in obs:
        if not o.dest:
            continue
        sig = o.thing.sig
        over = _contexts(o.dest, before, o.mates)[0]
        near = _contexts(o.dest, after, o.mates)[1]
        mem.safe_touch.setdefault(sig, set()).update(over)
        mem.safe_near.setdefault(sig, set()).update(near)
        contexts = [f"touch:{c}" for c in sorted(over)] + [f"near:{c}" for c in sorted(near)]
        for ctx in contexts:
            key = ("any", sig, ctx)
            if key in mem.facts and key not in refuted:
                refuted.append(key)
    for key in refuted:
        mem.record(key, _NONE, move)
    return refuted


def learn(
    mem: FactMemory,
    before: Scene,
    action: Action,
    diff: SceneDiff,
    *,
    move: int,
    level_up: bool,
    game_over: bool,
    after: Scene | None = None,
) -> list[FactKey]:
    """Turn one transition into facts; return the keys touched, each recorded once.

    ``after`` gives appeared things their position for ``appear(sig, at)``, the exact
    destination cells of movers, and the cell counts for the ``rotate`` tag. Without it
    appear positions are ``(-1, -1)`` and destinations are the shifted old cells.

    Movers with the same shift that were 4-adjacent before form a co-moving group, except
    a member whose own non-broken ``(label, sig, "any")`` fact says ``none``: it is pushed.

    ``level_up`` records only ``("event", "level_up", "any")``. ``game_over`` records only
    death facts for subjects that moved: ``("any", sig, "touch:<c>")`` for colors of things in
    ``before`` overlapping the destination, else ``("any", sig, "near:<c>")`` for colors
    4-adjacent to it, minus colors in the same mode's safe set (``mem.safe_touch`` or
    ``mem.safe_near``) unless that removes them all. Otherwise every subject records one
    observation: unchanged subjects record ``none``; a mover whose old cells a non-mate
    mover with the same shift entered, and whose own ``any`` fact doesn't predict that
    shift, goes to ``pushed``; each mover's destination colors join ``mem.walkable`` (and
    leave its blockers); a ``none`` where a move was expected but the way was shut (edge,
    padding, a thing outside its group, or a terrain color not walkable for the sig) goes
    to ``blocked`` and adds every destination color that isn't its own, a mate's or
    walkable to ``mem.blockers``. Spawns go to ``(label, SPAWN, "any")``. Last, each
    subject's new cells feed the safe sets: colors they overlap in ``before`` join
    ``safe_touch`` and charge a miss to ``touch:<c>`` only; colors 4-adjacent in ``after``
    join ``safe_near`` and charge a miss to ``near:<c>`` only.
    """
    if level_up:
        key = ("event", "level_up", "any")
        mem.record(key, (Effect("level_up"),), move)
        return [key]
    obs = _observations(before, action, diff, after)
    label = action_label(action, before)
    _split_off_pushed(mem, label, obs)
    if game_over:
        return _learn_touch(mem, before, obs, move)
    room_before = _Room.of(before)
    room_after = _Room.of(after) if after is not None else room_before
    _mark_pushed(mem, label, obs)
    _note_walkable(mem, room_before, obs)
    _mark_blocked(mem, label, room_before, obs)
    keys = _record_groups(mem, label, room_after, obs, move)
    keys += _record_groups(mem, label, room_after, _kind_observations(obs, after), move)
    _share_with_kinds(mem, obs)
    keys += _record_appear(mem, label, diff, after, move)
    return keys + _note_safe(mem, room_before.owner, room_after.owner, obs, move)
