"""Fact memory: short facts about what actions do to objects, with a table of contents.

A fact is keyed by ``(action label, subject, context)`` and holds a tuple of structured
effects the predictor can apply. ``FactMemory.toc`` maps behavior tags (root mechanic
words plus phrases like "moves until blocked") to fact keys, so an object that behaves
like a known one is predicted from a sibling's fact at once.

``learn`` turns one observed transition into one ``record`` call per fact key. Context
rules pick the key before recording, which is how a wall turns a predicted move into a
``blocked`` fact without charging the move fact a miss. A miss never rewrites a fact's
effects; ``slide`` and ``teleport`` rewrites happen only through the explicit rules in
``learn``. Pure and deterministic; callers strip noise things before calling ``learn``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, Action
from arcagent.mind.objects import Cell, Change, Scene, SceneDiff, Thing

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
Status = Literal["thin", "solid", "broken"]
RecordResult = Literal["new", "hit", "miss"]

THIN_BELOW = 5  # a fact is solid once this many observations agreed with it
BROKEN_MISSES = 2  # a fact is broken after this many misses no context absorbed

_DIRS: dict[tuple[int, int], str] = {
    (-1, 0): "left",
    (1, 0): "right",
    (0, -1): "up",
    (0, 1): "down",
}
_MOVING = frozenset({"move", "slide"})


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
    """A short fact. ``hits`` counts agreeing observations, the first one included."""

    key: FactKey
    effects: tuple[Effect, ...]
    evidence: list[int]  # move ids of every observation recorded against this key
    hits: int = 0
    misses: int = 0
    tags: frozenset[str] = frozenset()

    @property
    def status(self) -> Status:
        if self.misses >= BROKEN_MISSES:
            return "broken"
        return "solid" if self.hits >= THIN_BELOW else "thin"

    def line(self) -> str:
        """One line, e.g. ``A3 c1:ab12.. any: moves 1 cell left [solid 6/0]``."""
        label, subject, context = self.key
        what = "; ".join(e.text() for e in self.effects)
        score = f"[{self.status} {self.hits}/{self.misses}]"
        return f"{label} {_abbrev(subject)} {context}: {what} {score}"


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


def _sibling_handles(subject: str) -> tuple[str, str] | None:
    """``("s:<shape>", "c:<color>")`` for a sig subject; None for ``self``/``any``."""
    color, sep, shape = subject.partition(":")
    if not sep:
        return None
    return f"s:{shape}", f"c:{color}"


class FactMemory:
    """Facts by key, a table of contents by behavior tag, and colors that blocked each sig.

    ``landed`` remembers the top-left where the last ``any``-context move observation of a
    key ended, which is what the teleport rule compares against.
    """

    def __init__(self) -> None:
        self.facts: dict[FactKey, Fact] = {}
        self.toc: dict[str, set[FactKey]] = {}
        self.blockers: dict[str, set[int]] = {}
        self.landed: dict[FactKey, Cell] = {}
        # (label, context, sibling handle) -> keys; built alongside the toc for borrowing
        self._siblings: dict[tuple[str, str, str], set[FactKey]] = {}

    def get(self, key: FactKey) -> Fact | None:
        return self.facts.get(key)

    def lookup(self, label: str, subject: str, context: str) -> tuple[Fact, bool] | None:
        """Most specific fact for a subject and whether it was borrowed from a sibling.

        Order: exact key, the subject's ``any`` fact, a non-broken fact of a sibling subject
        (same shape before same color; same context before ``any``; most hits, then lowest
        key), then the action-wide ``(label, "any", "any")`` fact.
        """
        for key in ((label, subject, context), (label, subject, "any")):
            fact = self.facts.get(key)
            if fact is not None:
                return fact, False
        borrowed = self._borrow(label, subject, context)
        if borrowed is not None:
            return borrowed, True
        wide = self.facts.get((label, "any", "any"))
        return None if wide is None else (wide, False)

    def _borrow(self, label: str, subject: str, context: str) -> Fact | None:
        handles = _sibling_handles(subject)
        if handles is None:
            return None
        contexts = (context, "any") if context != "any" else ("any",)
        for handle in handles:
            for ctx in contexts:
                cands = [
                    self.facts[k]
                    for k in self._siblings.get((label, ctx, handle), ())
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
        replaces the fact's effects. ``tags`` adds behavior tags the effects alone can't
        show (``rotate`` needs cell counts).
        """
        fact = self.facts.get(key)
        if fact is None:
            fact = Fact(key, effects, [move], hits=1)
            self.facts[key] = fact
            self._index(fact, _tags(key, effects) | tags)
            handles = _sibling_handles(key[1])
            for handle in handles or ():
                self._siblings.setdefault((key[0], key[2], handle), set()).add(key)
            return "new"
        fact.evidence.append(move)
        if fact.effects == effects:
            fact.hits += 1
            if not tags <= fact.tags:
                self._index(fact, set(fact.tags | tags))
            return "hit"
        fact.misses += 1
        return "miss"

    def revise(self, key: FactKey, effects: tuple[Effect, ...], move: int) -> RecordResult:
        """Rewrite an existing fact's effects (slide, teleport) and count the observation a hit."""
        fact = self.facts[key]
        fact.effects = effects
        fact.hits += 1
        fact.evidence.append(move)
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


def action_label(action: Action, before: Scene) -> str:
    """``"A3"`` for a simple action; ``"A6:<clicked sig>"`` or ``"A6:none"`` for a click."""
    if action.id != COORD_ACTION_ID:
        return f"A{action.id}"
    thing = _clicked(before, action)
    return "A6:none" if thing is None else f"A6:{thing.sig}"


@dataclass(slots=True)
class _Obs:
    """One subject's observation in one transition, before grouping by key."""

    thing: Thing
    subject: str
    effects: tuple[Effect, ...]
    shift: tuple[int, int] | None  # displacement when the thing moved (or moved and reshaped)
    dest: frozenset[Cell]  # cells after the change; empty when it disappeared
    context: str = "any"
    tags: frozenset[str] = field(default_factory=frozenset[str])


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


def _neighbors(cells: frozenset[Cell]) -> set[Cell]:
    return {(x + dx, y + dy) for x, y in cells for dx, dy in _DIRS}


@dataclass(frozen=True, slots=True)
class _Room:
    """What can stop a step in one scene: padding plus each covered cell's id and color."""

    blocked: frozenset[Cell]
    owner: dict[Cell, tuple[int, int]]  # cell -> (thing id, color)

    @staticmethod
    def of(scene: Scene) -> _Room:
        owner = {c: (t.id, t.color) for t in scene.things for c in t.cells}
        return _Room(scene.blocked, owner)

    def stopped(
        self, cells: frozenset[Cell], step: tuple[int, int], self_id: int
    ) -> tuple[bool, set[int]]:
        """Whether one step from ``cells`` hits the edge, padding or a thing; its colors."""
        hit = False
        colors: set[int] = set()
        for x, y in _shifted(cells, *step) - cells:
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or (x, y) in self.blocked:
                hit = True
                continue
            other = self.owner.get((x, y))
            if other is not None and other[0] != self_id:
                hit = True
                colors.add(other[1])
        return hit, colors


def _unit(e: Effect) -> tuple[int, int]:
    return _sign(e.dx), _sign(e.dy)


def _expected_step(
    mem: FactMemory, label: str, o: _Obs, peers: list[_Obs]
) -> tuple[int, int] | None:
    """The step a ``none`` subject should have taken: its move fact, else its free peers' move."""
    fact = mem.facts.get((label, o.subject, "any"))
    if fact is not None:
        for e in fact.effects:
            if e.kind == "move":
                return e.dx, e.dy
            if e.kind == "slide":
                return _unit(e)
    shifts = Counter(p.shift for p in peers if p.shift is not None and p.context == "any")
    return shifts.most_common(1)[0][0] if shifts else None


def _mark_pushed(obs: list[_Obs]) -> None:
    """A mover whose old cells another mover entered was pushed."""
    movers = [o for o in obs if o.shift is not None]
    for o in movers:
        if any(p is not o and o.thing.cells & p.dest for p in movers):
            o.context = "pushed"


def _mark_blocked(mem: FactMemory, label: str, before: Scene, obs: list[_Obs]) -> None:
    """Wall rule: a ``none`` where a move was expected and the way was shut is ``blocked``."""
    room = _Room.of(before)
    by_subject: dict[str, list[_Obs]] = {}
    for o in obs:
        by_subject.setdefault(o.subject, []).append(o)
    for o in obs:
        if o.effects != _NONE:
            continue
        step = _expected_step(mem, label, o, by_subject[o.subject])
        if step is None:
            continue
        hit, colors = room.stopped(o.thing.cells, step, o.thing.id)
        if hit:
            o.context = "blocked"
            mem.blockers.setdefault(o.thing.sig, set()).update(colors)


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
    """Record an ``any``-context move, applying the slide and teleport rewrites."""
    landing = (rep.thing.x + seen.dx, rep.thing.y + seen.dy)
    fact = mem.facts.get(key)
    old = fact.effects[0] if fact is not None and len(fact.effects) == 1 else None
    last = mem.landed.get(key)
    mem.landed[key] = landing
    if old is None:
        mem.record(key, (seen,), move)
        return
    ends_blocked = room.stopped(rep.dest, _unit(seen), rep.thing.id)[0]
    slide = (Effect("slide", dx=_unit(seen)[0], dy=_unit(seen)[1]),)
    if old.kind == "slide" and old == slide[0] and ends_blocked:
        mem.record(key, slide, move)
    elif old.kind == "move" and old != seen and _same_direction(old, seen) and ends_blocked:
        mem.revise(key, slide, move)
    elif old.kind == "teleport" and old.at == landing:
        mem.record(key, (old,), move)
    elif old.kind == "move" and old != seen and last == landing:
        mem.revise(key, (Effect("teleport", at=landing),), move)
    else:
        mem.record(key, (seen,), move)


def _record_groups(
    mem: FactMemory, label: str, scene: Scene, obs: list[_Obs], move: int
) -> list[FactKey]:
    room = _Room.of(scene)
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
    key = (label, "any", "any")
    mem.record(key, tuple(sorted(spawned, key=lambda e: (e.sig, e.at))), move)
    return [key]


def _learn_touch(mem: FactMemory, before: Scene, obs: list[_Obs], move: int) -> list[FactKey]:
    """Game over: every mover gets a touch fact for each color at or beside its destination."""
    keys: list[FactKey] = []
    for o in obs:
        if o.shift is None:
            continue
        zone = o.dest | _neighbors(o.dest)
        colors = sorted({t.color for t in before.things if t.id != o.thing.id and t.cells & zone})
        for color in colors:
            key = ("any", o.thing.sig, f"touch:{color}")
            if key not in keys:
                keys.append(key)
    for key in keys:
        mem.record(key, (Effect("game_over"),), move)
    return keys


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

    ``level_up`` records only ``("event", "level_up", "any")``. ``game_over`` records only
    ``("any", sig, "touch:<color>")`` facts for subjects that moved, one per color of another
    thing at or 4-adjacent to the destination in ``before``. Otherwise every subject records
    one observation: unchanged subjects record ``none``, pushed movers go to ``pushed``,
    and a ``none`` where a move was expected but the way was shut (edge, padding or another
    thing) goes to ``blocked`` and adds the blocking colors to ``mem.blockers``.
    """
    if level_up:
        key = ("event", "level_up", "any")
        mem.record(key, (Effect("level_up"),), move)
        return [key]
    obs = _observations(before, action, diff, after)
    if game_over:
        return _learn_touch(mem, before, obs, move)
    label = action_label(action, before)
    _mark_pushed(obs)
    _mark_blocked(mem, label, before, obs)
    scene = after if after is not None else before
    keys = _record_groups(mem, label, scene, obs, move)
    return keys + _record_appear(mem, label, diff, after, move)
