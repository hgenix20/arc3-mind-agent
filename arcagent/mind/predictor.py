"""Predictor: apply stored facts to a scene and an action, then check the guess.

``predict`` looks up each thing's fact for the action and builds the next scene, or
reports the ``(label, subject)`` pairs it has no usable fact for (a broken fact counts
as none). With ``assume_still`` (the protected sigs), a thing whose sig is outside that
set and has no usable fact is predicted to stay as it is: its pair goes to
``Prediction.assumed``, not ``unknown``, and it gets no entry in ``effects`` so ``check``
never scores it. The clicked thing of a click is always protected. Search uses this so a
side object with unlearned or broken facts doesn't end every branch; tests and vetoes
call ``predict`` without it so they still target the unknowns. ``check`` compares a
prediction with the diff that actually happened and names the facts whose prediction
missed.

Movement rules, matching what ``facts.learn`` records:

- Movers are things whose fact moves them (``move``, ``slide``, ``teleport``). Movers
  with the same move or slide step whose cells are 4-adjacent form one rigid body: none
  blocks another, and the body is blocked when any member's destination hits the grid
  edge, padding, or a cell whose color (``objects.color_at``: the covering thing's, else
  the terrain's) is in ``mem.blockers`` for that member's sig. Terrain never moves in a
  prediction: the new scene shares the old scene's ``terrain``.
  A blocked body stays whole; each member then takes its ``blocked`` fact when stored
  (else ``none``).
- ``slide`` steps one cell at a time until the next step would be blocked. A slide
  never pushes.
- A move whose destination overlaps a non-mover with a ``pushed`` fact pushes it (one
  level only). When the pushed thing would itself be blocked (by the edge, padding, a
  blocker color, or another pushable thing), or its pushed fact doesn't move it, the
  pusher's body is blocked. A pushed thing is also stuck on a terrain cell whose color
  is in the blockers of its own sig or of the pusher's.
- ``toggle_group(sig)`` recolors every thing with that sig. The spawn fact
  ``(label, SPAWN, "any")`` adds one predicted thing per ``appear`` effect once it has
  ``SPAWN_MIN_HITS`` hits (a single appear is often a tracking artifact, such as a thing
  uncovered when the player steps off it). Spawn is never an unknown pair: with no spawn
  fact, a broken one or too few hits, no spawn is predicted (an unpredicted appear still
  fails ``check`` with no key to blame). ``spawns=False`` skips spawn facts entirely
  (search uses it: a predicted spawn only adds states).
- ``game_over`` is set when a fact predicts it outright, or when a thing that moves
  lands on (``touch:c``) or next to (``near:c``) a color with a non-broken death fact
  ``("any", sig, context)`` in the scene before the move.

``Prediction.effects`` holds resolved effects: a slide appears as the ``move`` it ends
in and a blocked mover as its blocked effects. Pure Python, no numpy; callers strip
noise things before calling.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable
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

SPAWN_MIN_HITS = 2  # a spawn fact predicts appears only after this many agreeing observations

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
    fact it names that ``blocked`` key, since the blocker guess is what failed. ``lent``
    holds the ids whose prediction came from a fact borrowed from a same-shape sibling;
    ``check`` never blames that fact for their misses. ``assumed`` holds the pairs predicted
    still under ``assume_still`` instead of being unknown.
    """

    scene: Scene | None
    effects: dict[int, tuple[Effect, ...]]
    used: tuple[FactKey, ...]
    unknown: tuple[Pair, ...]
    game_over: bool
    borrowed: bool
    sources: dict[int, FactKey] = field(default_factory=dict[int, FactKey])
    lent: frozenset[int] = frozenset()
    assumed: tuple[Pair, ...] = ()


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

    def __init__(
        self,
        scene: Scene,
        action: Action,
        mem: FactMemory,
        protect: frozenset[str] | None,
        spawns: bool = True,
    ) -> None:
        self.scene = scene
        self.mem = mem
        self.spawns = spawns
        self._owner: Owner | None = None
        click = action.id == COORD_ACTION_ID
        self.label = (
            action_label(action, scene, self.owner) if click else action_label(action, scene)
        )
        clicked = -1
        if action.id == COORD_ACTION_ID:
            point = (action.x, action.y)
            clicked = next((t.id for t in scene.things if point in t.cells), -1)
        self.clicked = clicked
        self.protect = protect
        self.assumed: dict[Pair, None] = {}
        self.subject = {t.id: "self" if t.id == clicked else t.sig for t in scene.things}
        self.effects: dict[int, tuple[Effect, ...]] = {}
        self.sources: dict[int, FactKey] = {}
        self.used: dict[FactKey, None] = {}
        self.pending: dict[int, Pair] = {}  # thing id -> pair, unless resolved later
        self.unknown: dict[Pair, None] = {}
        self.borrowed = False
        self.lent: set[int] = set()  # ids predicted from a borrowed fact
        self.push_borrowed: set[int] = set()  # ids whose pushed fact was borrowed
        self.movers: set[int] = set()
        self._pushable: dict[int, Fact | None] = {}
        self.shift: dict[int, tuple[int, int]] = {}
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

    def kind_of(self, tid: int) -> str | None:
        """The thing's kind for a fact lookup; None for the clicked thing (label facts)."""
        return None if self.subject[tid] == "self" else self.by_id[tid].kind

    def still(self, tid: int) -> bool:
        """Whether a thing with no usable fact is assumed still (outside the protected sigs)."""
        if self.protect is None or tid == self.clicked:
            return False
        t = self.by_id[tid]
        return t.sig not in self.protect and t.kind not in self.protect

    def open_pair(self, tid: int, pair: Pair) -> None:
        """Record a missing fact: ``assumed`` when the thing may be assumed still, else unknown."""
        if self.still(tid):
            self.assumed[pair] = None
        else:
            self.unknown[pair] = None

    def use(self, fact: Fact, borrowed: bool = False) -> None:
        self.used[fact.key] = None
        self.borrowed = self.borrowed or borrowed

    def assign(self, tid: int, fact: Fact, borrowed: bool = False) -> None:
        self.effects[tid] = fact.effects
        self.sources[tid] = fact.key
        self.pending.pop(tid, None)
        if borrowed:
            self.lent.add(tid)
        else:
            self.lent.discard(tid)

    # --- blocking -------------------------------------------------------------------

    def hits(self, t: Thing, dest: frozenset[Cell], skip: frozenset[int], *, pushing: bool) -> bool:
        """Whether ``t`` landing on ``dest`` hits the edge, padding or a blocker color.

        With ``pushing`` (a move body), a non-mover with a usable pushed fact never blocks
        here: ``_push_targets`` decides, since a box once blocked into a wall puts its
        color in the pusher's blockers.
        """
        stop = _blockers(self.mem, t)
        owner = self.owner
        padding = self.scene.blocked
        terrain = self.scene.terrain
        for c in dest:
            x, y = c
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or c in padding:
                return True
            other = owner.get(c)
            if other is None:
                if stop and int(terrain[y, x]) in stop:
                    return True
                continue
            if other[0] in skip or other[1] not in stop:
                continue
            if pushing and other[0] not in self.movers and self.pushed_fact(other[0]):
                continue
            return True
        return False

    def body_hits(
        self, body: list[_Mover], step: tuple[int, int], skip: frozenset[int], *, pushing: bool
    ) -> bool:
        dx, dy = step
        return any(
            self.hits(m.thing, _shifted(m.thing.cells, dx, dy), skip, pushing=pushing) for m in body
        )

    def stuck(
        self, dest: frozenset[Cell], skip: frozenset[int], stop: Collection[int] = ()
    ) -> bool:
        """Whether a pushed thing landing on ``dest`` hits the edge, padding, any thing,
        or a terrain cell whose color is in ``stop``."""
        owner = self.owner
        padding = self.scene.blocked
        terrain = self.scene.terrain
        for c in dest:
            x, y = c
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or c in padding:
                return True
            other = owner.get(c)
            if other is None:
                if stop and int(terrain[y, x]) in stop:
                    return True
            elif other[0] not in skip:
                return True
        return False

    def pushed_fact(self, tid: int) -> Fact | None:
        """The thing's pushed fact, own or borrowed, cached; a broken one marks it unknown."""
        if tid in self._pushable:
            return self._pushable[tid]
        fact = self._pushed_fact(tid)
        self._pushable[tid] = fact
        return fact

    def _pushed_fact(self, tid: int) -> Fact | None:
        subject = self.subject[tid]
        found = self.mem.lookup(self.label, subject, "pushed", self.kind_of(tid))
        if found is None or found[0].key[2] != "pushed":
            return None
        if found[0].status == "broken":
            self.open_pair(tid, (self.label, subject))
            return None
        if found[1]:
            self.push_borrowed.add(tid)
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
        found = run.mem.lookup(run.label, subject, "any", run.kind_of(t.id))
        if found is None or found[0].status == "broken":
            run.pending[t.id] = (run.label, subject)
            continue
        fact, borrowed = found
        run.use(fact, borrowed)
        run.assign(t.id, fact, borrowed)
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
    pusher_stop = set[int]().union(*(_blockers(run.mem, m.thing) for m in body))
    for tid, fact in targets.items():
        t = by_id[tid]
        d = _displacement(t, fact.effects)
        if d is None:
            if any(e.kind == "disappear" for e in fact.effects):
                continue
            return None
        stop = pusher_stop | _blockers(run.mem, t)
        if run.stuck(_shifted(t.cells, *d), mates | {tid}, stop):
            return None
    return targets


def _block(run: _Run, body: list[_Mover]) -> None:
    for m in body:
        tid = m.thing.id
        key = (run.label, run.subject[tid], "blocked")
        fact = run.mem.get(key)
        kind = run.kind_of(tid)
        if (fact is None or fact.status == "broken") and kind is not None:
            fallback = run.mem.get((run.label, kind, "blocked"))
            if fallback is not None and fallback.status != "broken":
                key, fact = fallback.key, fallback
        if fact is None:
            run.effects[tid] = _NONE
            run.sources[tid] = key
            run.lent.discard(tid)
        elif fact.status == "broken":
            run.effects.pop(tid, None)
            run.open_pair(tid, (run.label, run.subject[tid]))
        else:
            run.use(fact)
            run.assign(tid, fact)


def _resolve_body(run: _Run, body: list[_Mover], movers: set[int]) -> None:
    mates = frozenset(m.thing.id for m in body)
    head = body[0]
    if head.kind == "slide":
        ux, uy = head.step
        n = 0
        while n < GRID_SIZE and not run.body_hits(
            body, (ux * (n + 1), uy * (n + 1)), mates, pushing=False
        ):
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
    if run.body_hits(body, (dx, dy), mates, pushing=head.kind == "move"):
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
        lent = tid in run.push_borrowed
        run.use(fact, lent)
        run.assign(tid, fact, lent)
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
                if t.id not in run.sources:
                    run.sources[t.id] = run.sources[tid]
                    if tid in run.lent:
                        run.lent.add(t.id)
                run.pending.pop(t.id, None)


def _spawn(run: _Run) -> list[Thing]:
    if not run.spawns:
        return []
    key = (run.label, SPAWN, "any")
    fact = run.mem.get(key)
    if fact is None or fact.status == "broken" or fact.hits < SPAWN_MIN_HITS:
        return []  # never an open question: predict no spawn
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
            fact = run.mem.get(("any", t.sig, ctx)) or run.mem.get(("any", t.kind, ctx))
            if (
                fact is not None
                and fact.status != "broken"
                and any(e.kind == "game_over" for e in fact.effects)
            ):
                run.use(fact)
                dead = True
    return dead


def _blockers(mem: FactMemory, t: Thing) -> set[int]:
    """Colors that stopped this thing's look or its kind."""
    own = mem.blockers.get(t.sig, set())
    return own if t.kind == t.sig else own | mem.blockers.get(t.kind, set())


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
    return Thing(t.id, color, shape, sig, t.x + dx, t.y + dy, cells, t.kind)


def predict(
    scene: Scene,
    action: Action,
    mem: FactMemory,
    *,
    assume_still: frozenset[str] | None = None,
    spawns: bool = True,
) -> Prediction:
    """Predict the scene after ``action``, or the (label, subject) pairs with no fact.

    ``assume_still`` is the set of protected sigs: when given, a thing outside it with no
    usable fact stays as it is and its pair is listed in ``assumed``. ``spawns=False``
    predicts no spawn at all.
    """
    run = _Run(scene, action, mem, assume_still, spawns)
    movers = _base(run)
    if movers:
        run.movers = set(movers)
        for body in _bodies(movers):
            _resolve_body(run, body, run.movers)
    _toggle(run)
    spawned = _spawn(run)
    game_over = _deadly(run)
    for tid, pair in run.pending.items():
        run.open_pair(tid, pair)
    unknown = tuple(run.unknown)
    new_scene: Scene | None = None
    if not unknown:
        things: list[Thing] = []
        for t in scene.things:
            moved = _apply(t, run.effects.get(t.id, _NONE), run.shift.get(t.id, (0, 0)))
            if moved is not None:
                things.append(moved)
        new_scene = Scene((*things, *spawned), scene.background, scene.blocked, scene.terrain)
    return Prediction(
        new_scene,
        run.effects,
        tuple(run.used),
        unknown,
        game_over,
        run.borrowed,
        run.sources,
        frozenset(run.lent),
        tuple(p for p in run.assumed if p not in run.unknown),
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


def _spawns(pred: Prediction, before: dict[int, Thing]) -> tuple[Counter[str], FactKey | None]:
    """Predicted spawn sigs as a multiset, and the spawn fact they came from."""
    spawned: Counter[str] = Counter()
    spawn_key: FactKey | None = None
    for tid, effects in pred.effects.items():
        if tid in before:
            continue
        for e in effects:
            if e.kind == "appear":
                spawned[e.sig] += 1
                spawn_key = pred.sources.get(tid, spawn_key)
    return spawned, spawn_key


def _ignored_appears(
    diff: SceneDiff, before: dict[int, Thing], after: Scene | None, ignore: frozenset[int]
) -> set[int]:
    """Ids of appeared things that overlap an ignored thing's cells in ``before``."""
    if not ignore or after is None:
        return set()
    noise: set[Cell] = set()
    for tid in ignore:
        t = before.get(tid)
        if t is not None:
            noise |= t.cells
    now = after.by_id()
    out: set[int] = set()
    for c in diff.changes:
        t = now.get(c.id)
        if c.kind == "appeared" and t is not None and not t.cells.isdisjoint(noise):
            out.add(c.id)
    return out


def check(
    pred: Prediction,
    before: Scene,
    diff: SceneDiff,
    *,
    ignore: frozenset[int] = frozenset(),
    after: Scene | None = None,
) -> tuple[bool, list[FactKey]]:
    """(hit, keys whose effect missed). A hit means every predicted effect matched its Change.

    Changes whose id is in ``ignore`` (noise ids) are skipped, and so are appeared things
    overlapping an ignored thing's cells in ``before``; that overlap needs ``after`` (the
    scene the diff came from) to place appeared things, and without it no appear is skipped.

    Appeared things match predicted spawns by sig (as a multiset); a spawn mismatch is a
    miss of the spawn fact. An appear nothing predicted, or a miss of a fact borrowed from
    a sibling (``pred.lent``), makes the check fail with no key to blame: charging the
    sibling would break a fact that is right for its own sig. Things with no prediction
    never count as a miss.
    """
    by_id = before.by_id()
    skip = _ignored_appears(diff, by_id, after, ignore)
    missed: dict[FactKey, None] = {}
    surprise = False
    appeared: Counter[str] = Counter()
    for change in diff.changes:
        if change.id in ignore or change.id in skip:
            continue
        if change.kind == "appeared":
            appeared[change.sig] += 1
            continue
        t = by_id.get(change.id)
        effects = pred.effects.get(change.id)
        if t is None or effects is None:
            continue
        if not _matches(t, effects, change):
            key = pred.sources.get(change.id)
            if key is None or change.id in pred.lent:
                surprise = True  # a borrowed fact missed: not this subject's fact to break
            else:
                missed[key] = None
    spawned, spawn_key = _spawns(pred, by_id)
    if spawned != appeared:
        if spawn_key is not None:
            missed[spawn_key] = None
        else:
            surprise = True
    keys = list(missed)
    return not keys and not surprise, keys
