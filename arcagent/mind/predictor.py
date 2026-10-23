"""Predictor: apply stored facts to a scene and an action, then check the guess.

``predict`` looks up each thing's fact for the action and builds the next scene, or
reports the ``(label, subject)`` pairs it has no usable fact for (a broken fact counts
as none). ``check`` compares a prediction with the diff that actually happened and
names the facts whose prediction missed.

Movement rules, matching what ``facts.learn`` records:

- Movers are things whose fact moves them (``move``, ``slide``, ``teleport``). Movers
  with the same move or slide step whose cells are 4-adjacent form one rigid body: none
  blocks another, and the body is blocked when any member's destination hits the grid
  edge, padding, or a thing whose color is in ``mem.blockers`` for that member's sig.
  A blocked body stays whole; each member then takes its ``blocked`` fact when stored
  (else ``none``).
- ``slide`` steps one cell at a time until the next step would be blocked. A slide
  never pushes.
- A move whose destination overlaps a non-mover with a ``pushed`` fact pushes it (one
  level only). When the pushed thing would itself be blocked (by the edge, padding, a
  blocker color, or another pushable thing), or its pushed fact doesn't move it, the
  pusher's body is blocked.
- ``toggle_group(sig)`` recolors every thing with that sig. The spawn fact
  ``(label, SPAWN, "any")`` adds one predicted thing per ``appear`` effect.
- ``game_over`` is set when a fact predicts it outright, or when a thing that moves
  lands on (``touch:c``) or next to (``near:c``) a color with a non-broken death fact
  ``("any", sig, context)`` in the scene before the move.

``Prediction.effects`` holds resolved effects: a slide appears as the ``move`` it ends
in and a blocked mover as its blocked effects. Pure Python, no numpy; callers strip
noise things before calling.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, Action
from arcagent.mind.facts import (
    SPAWN,
    Effect,
    Fact,
    FactKey,
    FactMemory,
    action_label,
    cell_owner,
    touch_contexts,
)
from arcagent.mind.objects import Cell, Change, Scene, SceneDiff, Thing

_NONE = (Effect("none"),)
_MOVING = frozenset({"move", "slide", "teleport"})

Pair = tuple[str, str]
Owner = dict[Cell, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class Prediction:
    """What ``predict`` expects one action to do.

    ``scene`` is None when ``unknown`` is not empty. ``effects`` maps each thing id with
    a usable fact (and each predicted spawn, under a new id) to its resolved effects.
    ``sources`` maps the same ids to the key of the fact the prediction came from, which
    is what ``check`` reports on a miss; for a blocked mover with no stored ``blocked``
    fact it names that ``blocked`` key, since the blocker guess is what failed.
    """

    scene: Scene | None
    effects: dict[int, tuple[Effect, ...]]
    used: tuple[FactKey, ...]
    unknown: tuple[Pair, ...]
    game_over: bool
    borrowed: bool
    sources: dict[int, FactKey] = field(default_factory=dict[int, FactKey])


def _shifted(cells: Iterable[Cell], dx: int, dy: int) -> frozenset[Cell]:
    return frozenset((x + dx, y + dy) for x, y in cells)


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


@dataclass(slots=True)
class _Mover:
    thing: Thing
    kind: str  # "move", "slide" or "teleport"
    step: tuple[int, int]  # full shift for move and teleport, unit step for slide


class _Run:
    """State of one ``predict`` call."""

    def __init__(self, scene: Scene, action: Action, mem: FactMemory) -> None:
        self.scene = scene
        self.mem = mem
        self.label = action_label(action, scene)
        clicked = -1
        if action.id == COORD_ACTION_ID:
            point = (action.x, action.y)
            clicked = next((t.id for t in scene.things if point in t.cells), -1)
        self.subject = {t.id: "self" if t.id == clicked else t.sig for t in scene.things}
        self.effects: dict[int, tuple[Effect, ...]] = {}
        self.sources: dict[int, FactKey] = {}
        self.used: dict[FactKey, None] = {}
        self.pending: dict[int, Pair] = {}  # thing id -> pair, unless resolved later
        self.unknown: dict[Pair, None] = {}
        self.borrowed = False
        self.shift: dict[int, tuple[int, int]] = {}
        self._owner: Owner | None = None
        self._by_id: dict[int, Thing] | None = None
        # thing id -> other ids its game-over touch test ignores (mates, pushed things)
        self.skip: dict[int, frozenset[int]] = {}

    @property
    def owner(self) -> Owner:
        if self._owner is None:
            self._owner = cell_owner(self.scene)
        return self._owner

    @property
    def by_id(self) -> dict[int, Thing]:
        if self._by_id is None:
            self._by_id = self.scene.by_id()
        return self._by_id

    def use(self, fact: Fact, borrowed: bool = False) -> None:
        self.used[fact.key] = None
        self.borrowed = self.borrowed or borrowed

    def assign(self, tid: int, fact: Fact, effects: tuple[Effect, ...] | None = None) -> None:
        self.effects[tid] = fact.effects if effects is None else effects
        self.sources[tid] = fact.key
        self.pending.pop(tid, None)

    # --- blocking -------------------------------------------------------------------

    def hits(self, t: Thing, dest: frozenset[Cell], skip: frozenset[int]) -> bool:
        """Whether ``t`` landing on ``dest`` hits the edge, padding or a blocker color."""
        stop = self.mem.blockers.get(t.sig, ())
        owner = self.owner
        padding = self.scene.blocked
        for c in dest:
            x, y = c
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or c in padding:
                return True
            other = owner.get(c)
            if other is not None and other[0] not in skip and other[1] in stop:
                return True
        return False

    def body_hits(self, body: list[_Mover], dx: int, dy: int, skip: frozenset[int]) -> bool:
        return any(self.hits(m.thing, _shifted(m.thing.cells, dx, dy), skip) for m in body)

    def pushed_fact(self, tid: int) -> Fact | None:
        """The thing's pushed fact, own or borrowed; a broken one marks it unknown."""
        subject = self.subject[tid]
        found = self.mem.lookup(self.label, subject, "pushed")
        if found is None or found[0].key[2] != "pushed":
            return None
        if found[0].status == "broken":
            self.unknown[(self.label, subject)] = None
            return None
        return found[0]


def _displacement(t: Thing, effects: tuple[Effect, ...]) -> tuple[int, int] | None:
    for e in effects:
        if e.kind == "move":
            return e.dx, e.dy
        if e.kind == "teleport":
            return e.at[0] - t.x, e.at[1] - t.y
    return None


def _base(run: _Run) -> dict[int, _Mover]:
    """Look up every thing's ``any`` fact; return the movers."""
    movers: dict[int, _Mover] = {}
    for t in run.scene.things:
        subject = run.subject[t.id]
        found = run.mem.lookup(run.label, subject, "any")
        if found is None or found[0].status == "broken":
            run.pending[t.id] = (run.label, subject)
            continue
        fact, borrowed = found
        run.use(fact, borrowed)
        run.assign(t.id, fact)
        for e in fact.effects:
            if e.kind == "move" and (e.dx or e.dy):
                movers[t.id] = _Mover(t, "move", (e.dx, e.dy))
            elif e.kind == "slide" and (e.dx or e.dy):
                movers[t.id] = _Mover(t, "slide", (_sign(e.dx), _sign(e.dy)))
            elif e.kind == "teleport":
                movers[t.id] = _Mover(t, "teleport", (e.at[0] - t.x, e.at[1] - t.y))
            else:
                continue
            break
    return movers


def _bodies(movers: dict[int, _Mover]) -> list[list[_Mover]]:
    """Group movers with the same move or slide step whose cells are 4-adjacent."""
    ids = list(movers)
    parent = {i: i for i in ids}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    at: dict[Cell, int] = {c: i for i in ids for c in movers[i].thing.cells}
    for i in ids:
        m = movers[i]
        if m.kind == "teleport":
            continue
        for x, y in m.thing.cells:
            for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = at.get(c)
                if j is not None and j != i:
                    n = movers[j]
                    if n.kind == m.kind and n.step == m.step:
                        parent[find(i)] = find(j)
    groups: dict[int, list[_Mover]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(movers[i])
    return list(groups.values())


def _push_targets(
    run: _Run, body: list[_Mover], step: tuple[int, int], mates: frozenset[int], movers: set[int]
) -> dict[int, Fact] | None:
    """Things the body pushes, or None when a pushed thing can't move (body blocked)."""
    dx, dy = step
    by_id = run.by_id
    targets: dict[int, Fact] = {}
    for m in body:
        for c in _shifted(m.thing.cells, dx, dy):
            other = run.owner.get(c)
            if other is None or other[0] in mates or other[0] in movers:
                continue
            tid = other[0]
            if tid in targets or tid in run.shift:
                continue
            fact = run.pushed_fact(tid)
            if fact is not None:
                targets[tid] = fact
    for tid, fact in targets.items():
        t = by_id[tid]
        d = _displacement(t, fact.effects)
        if d is None:
            if any(e.kind == "disappear" for e in fact.effects):
                continue
            return None
        dest = _shifted(t.cells, *d)
        skip = mates | {tid}
        if run.hits(t, dest, skip):
            return None
        for c in dest:
            other = run.owner.get(c)
            if other is not None and other[0] not in skip and run.pushed_fact(other[0]):
                return None
    return targets


def _block(run: _Run, body: list[_Mover]) -> None:
    for m in body:
        tid = m.thing.id
        key = (run.label, run.subject[tid], "blocked")
        fact = run.mem.get(key)
        if fact is None:
            run.effects[tid] = _NONE
            run.sources[tid] = key
        elif fact.status == "broken":
            run.effects.pop(tid, None)
            run.unknown[(run.label, run.subject[tid])] = None
        else:
            run.use(fact)
            run.assign(tid, fact)


def _resolve_body(run: _Run, body: list[_Mover], movers: set[int]) -> None:
    mates = frozenset(m.thing.id for m in body)
    head = body[0]
    if head.kind == "slide":
        ux, uy = head.step
        n = 0
        while n < GRID_SIZE and not run.body_hits(body, ux * (n + 1), uy * (n + 1), mates):
            n += 1
        if n == 0:
            _block(run, body)
            return
        for m in body:
            tid = m.thing.id
            run.shift[tid] = (ux * n, uy * n)
            run.skip[tid] = mates
            rest = tuple(e for e in run.effects[tid] if e.kind != "slide")
            run.effects[tid] = (Effect("move", dx=ux * n, dy=uy * n), *rest)
        return
    dx, dy = head.step
    if run.body_hits(body, dx, dy, mates):
        _block(run, body)
        return
    targets: dict[int, Fact] = {}
    if head.kind == "move":
        found = _push_targets(run, body, (dx, dy), mates, movers)
        if found is None:
            _block(run, body)
            return
        targets = found
    pushed = frozenset(targets)
    for m in body:
        run.shift[m.thing.id] = (dx, dy)
        run.skip[m.thing.id] = mates | pushed
    by_id = run.by_id
    for tid, fact in targets.items():
        run.use(fact)
        run.assign(tid, fact)
        d = _displacement(by_id[tid], fact.effects)
        if d is not None:
            run.shift[tid] = d
            run.skip[tid] = mates


def _toggle(run: _Run) -> None:
    for tid, effects in list(run.effects.items()):
        for e in effects:
            if e.kind != "toggle_group":
                continue
            recolor = Effect("recolor", color=e.color)
            for t in run.scene.things:
                if t.sig != e.sig:
                    continue
                keep = tuple(
                    x for x in run.effects.get(t.id, ()) if x.kind not in ("none", "recolor")
                )
                run.effects[t.id] = (*keep, recolor)
                run.sources.setdefault(t.id, run.sources[tid])
                run.pending.pop(t.id, None)


def _spawn(run: _Run) -> list[Thing]:
    key = (run.label, SPAWN, "any")
    fact = run.mem.get(key)
    if fact is None:
        return []
    if fact.status == "broken":
        run.unknown[(run.label, SPAWN)] = None
        return []
    run.use(fact)
    next_id = max((t.id for t in run.scene.things), default=0) + 1
    out: list[Thing] = []
    for e in fact.effects:
        if e.kind != "appear":
            continue
        template = next((t for t in run.scene.things if t.sig == e.sig), None)
        x, y = e.at
        if template is None:  # shape unknown here: one cell stands in for it
            cells = frozenset({(x, y)})
        else:
            cells = _shifted(template.cells, x - template.x, y - template.y)
        color_text, _, shape = e.sig.partition(":")
        color = int(color_text[1:]) if color_text[1:].isdigit() else -1
        out.append(Thing(next_id, color, shape, e.sig, x, y, cells))
        run.effects[next_id] = (e,)
        run.sources[next_id] = key
        next_id += 1
    return out


def _deadly(run: _Run) -> bool:
    dead = any(e.kind == "game_over" for effs in run.effects.values() for e in effs)
    if not run.shift:
        return dead
    for tid, (dx, dy) in run.shift.items():
        if not (dx or dy):
            continue
        t = run.by_id[tid]
        owner = run.owner
        skip = run.skip.get(tid, frozenset[int]()) - {tid}
        if skip:  # rare: a sprite or a push; drop the ignored things from the cell map
            owner = {c: v for c, v in owner.items() if v[0] not in skip}
        for ctx in touch_contexts(_shifted(t.cells, dx, dy), run.scene, tid, owner=owner):
            fact = run.mem.get(("any", t.sig, ctx))
            if (
                fact is not None
                and fact.status != "broken"
                and any(e.kind == "game_over" for e in fact.effects)
            ):
                run.use(fact)
                dead = True
    return dead


def _apply(t: Thing, effects: tuple[Effect, ...], shift: tuple[int, int]) -> Thing | None:
    color, shape, sig = t.color, t.shape, t.sig
    for e in effects:
        if e.kind == "disappear":
            return None
        if e.kind == "recolor":
            color, sig = e.color, f"c{e.color}:{shape}"
        elif e.kind == "reshape":  # new cells are unknown; the old cells stand in
            color_text, _, shape = e.sig.partition(":")
            color = int(color_text[1:]) if color_text[1:].isdigit() else color
            sig = e.sig
    dx, dy = shift
    if not (dx or dy) and sig == t.sig:
        return t
    cells = _shifted(t.cells, dx, dy) if dx or dy else t.cells
    return Thing(t.id, color, shape, sig, t.x + dx, t.y + dy, cells)


def predict(scene: Scene, action: Action, mem: FactMemory) -> Prediction:
    """Predict the scene after ``action``, or the (label, subject) pairs with no fact."""
    run = _Run(scene, action, mem)
    movers = _base(run)
    if movers:
        ids = set(movers)
        for body in _bodies(movers):
            _resolve_body(run, body, ids)
    _toggle(run)
    spawned = _spawn(run)
    game_over = _deadly(run)
    for pair in run.pending.values():
        run.unknown[pair] = None
    unknown = tuple(run.unknown)
    new_scene: Scene | None = None
    if not unknown:
        things: list[Thing] = []
        for t in scene.things:
            moved = _apply(t, run.effects[t.id], run.shift.get(t.id, (0, 0)))
            if moved is not None:
                things.append(moved)
        new_scene = Scene((*things, *spawned), scene.background, scene.blocked)
    return Prediction(
        new_scene,
        run.effects,
        tuple(run.used),
        unknown,
        game_over,
        run.borrowed,
        run.sources,
    )


def _matches(t: Thing, effects: tuple[Effect, ...], change: Change) -> bool:
    """Whether a thing's predicted effects agree with what happened to it."""
    if any(e.kind == "disappear" for e in effects):
        return change.kind == "disappeared"
    if change.kind in ("disappeared", "appeared"):
        return False
    shift = _displacement(t, effects) or (0, 0)
    expected = _apply(t, effects, (0, 0))
    sig = t.sig if expected is None else expected.sig
    return (change.dx, change.dy) == shift and change.sig == sig


def check(pred: Prediction, before: Scene, diff: SceneDiff) -> tuple[bool, list[FactKey]]:
    """(hit, keys whose effect missed). A hit means every predicted effect matched its Change.

    Appeared things match predicted spawns by sig (as a multiset); a spawn mismatch is a
    miss of the spawn fact. An appear nothing predicted makes the check fail with no key
    to blame. Things with no prediction never count as a miss.
    """
    by_id = before.by_id()
    missed: dict[FactKey, None] = {}
    appeared: Counter[str] = Counter()
    for change in diff.changes:
        if change.kind == "appeared":
            appeared[change.sig] += 1
            continue
        t = by_id.get(change.id)
        effects = pred.effects.get(change.id)
        if t is None or effects is None:
            continue
        if not _matches(t, effects, change):
            key = pred.sources.get(change.id)
            if key is not None:
                missed[key] = None
    spawned: Counter[str] = Counter()
    spawn_key: FactKey | None = None
    for tid, effects in pred.effects.items():
        if tid in by_id:
            continue
        for e in effects:
            if e.kind == "appear":
                spawned[e.sig] += 1
                spawn_key = pred.sources.get(tid, spawn_key)
    surprise = False
    if spawned != appeared:
        if spawn_key is not None:
            missed[spawn_key] = None
        else:
            surprise = True
    keys = list(missed)
    return not keys and not surprise, keys
