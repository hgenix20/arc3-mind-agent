"""Strategy skills: deterministic approaches the planner calls by name (spec 4.9d).

A skill turns a named approach into a short segment of moves over the current scene,
built from what the agent already holds: the mover's gaits and blockers through
``Kinematics``, the role the agent gave each kind (harm, collectible, ...), the open
goals and the curiosity flags. Kameron's examples (spec 4.9d): "enemy blocks path, then
kill it", "pitfall blocks path, then replan route around it", and collecting like
pacman's fruit before completing the level.

``run_skill(name, args, ctx)`` dispatches on the name and always returns a
``SkillResult``: ``applies`` False with the reason in ``why`` when the skill has nothing
it can do on this scene (no mover, no target, no path), never an exception. Every result
holds at most ``ctx.max_steps`` actions. The skills:

- ``S-AROUND``: the moves to a target that keep every cell of the mover's body off the
  cells of harm-role things and off ``args["avoid_cells"]``, by breadth-first search
  over the mover's top-lefts (spec 4.6a; ``Kinematics.reachable_avoiding``). The target
  is the thing ``args["target"]`` names by id when it is on screen, else the nearest
  target of the first reach or cover goal in ``ctx.goals`` whose mover is ``ctx.mover``
  and whose target is on screen.
- ``S-COLLECT``: the collectible-role things ordered with ``route.route`` (spec 4.6b)
  over kinematic leg lengths, then the moves to the first one, harm cells avoided. At
  most ``MAX_COLLECT`` collectibles are routed, the nearest by cell gap; the ones the
  mover cannot reach from where it stands are left out before routing.
- ``S-EXPLORE``: the first flag in ``ctx.flags`` with a thing on screen (spec 4.11b):
  one click on that thing when clicking is available, else the moves to touch it.
- ``S-ATTACK`` and ``S-WAIT``: named so the contract's skill list stays valid; both
  need the health model and do not apply in this version.

Contact: a body touches a target when a body cell overlaps a target cell or is 4-adjacent
to one, the rule ``Kinematics.field`` uses. ``S-COLLECT`` prefers a top-left whose body
overlaps the collectible, since that is how one is taken, and falls back to touching
when no overlapping top-left is reachable. A leg between two collectibles is measured by
a walk from a scene in which the mover stands at the first collectible's top-left and
that collectible is gone: ``Kinematics.steps_to`` measures only from the mover's own
place and ``Kinematics.field`` names its target by sig, so neither tells two same-sig
collectibles apart. A leg with no walk falls back to the cell gap. Pure; nothing here
logs, the agent that wires the skills does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeGuard

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, Action
from arcagent.mind.facts import FactMemory
from arcagent.mind.goals import MAX_OPEN_GOALS, Goal
from arcagent.mind.kinematics import Kinematics
from arcagent.mind.objects import Cell, Scene, Thing, answers
from arcagent.mind.route import route

Role = Literal["harm", "collectible", "blocker", "mover", "controlled", "unknown"]
Flag = tuple[str, float, float]  # a curiosity flag: kind, value, cost
_Reach = dict[Cell, tuple[int, Cell | None, Action | None]]  # Kinematics.reachable_avoiding

SKILL_NAMES: tuple[str, ...] = ("S-AROUND", "S-COLLECT", "S-EXPLORE", "S-ATTACK", "S-WAIT")
MAX_COLLECT = 8  # collectibles routed per S-COLLECT call, nearest by cell gap first
MAX_FLAGS = 100  # flags read per S-EXPLORE call
MAX_AVOID_CELLS = GRID_SIZE * GRID_SIZE  # cells read from args["avoid_cells"]
UNKNOWN_SKILL = "unknown skill"
NOT_BUILT = "not built in this version"
_TOUCH = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class SkillContext:
    """What a skill reads; the agent builds one per call.

    ``goals`` are the open goals in the agent's order. ``roles`` maps a thing's kind (its
    sig when no kind class is known) to its role; a thing with no entry is ``unknown``.
    ``flags`` are curiosity flags as (kind, value, cost), best first. ``can_click`` says
    whether action 6 is available. ``mover`` is the controlled thing's name as
    ``Kinematics`` knows it (sig or kind), None when the agent does not know its
    character. ``click_cell`` gives the cell to click on a thing (``objects.click_cell``
    in the agent). ``max_steps`` caps every result's actions.
    """

    scene: Scene
    mem: FactMemory
    kin: Kinematics
    goals: Sequence[Goal]
    roles: Mapping[str, Role]
    flags: Sequence[Flag]
    can_click: bool
    mover: str | None
    click_cell: Callable[[Thing], tuple[int, int]]
    max_steps: int


@dataclass(frozen=True)
class SkillResult:
    """What a skill returns: ``applies`` False carries the reason in ``why`` and no actions."""

    name: str
    actions: tuple[Action, ...]
    why: str
    applies: bool


def run_skill(name: str, args: Mapping[str, object], ctx: SkillContext) -> SkillResult:
    """Run the skill ``name`` names with the planner's ``args`` over ``ctx``.

    ``name`` is matched after stripping, upper-casing and reading ``_`` as ``-``; a name
    outside ``SKILL_NAMES`` gives ``applies`` False with ``why`` ``UNKNOWN_SKILL``.
    ``ctx.max_steps`` below 1 gives ``applies`` False, since no segment could be returned.
    ``args`` entries are read by key and type; a malformed entry counts as absent. Never
    raises on a scene a skill cannot handle: the reason comes back in ``why``.
    """
    key = name.strip().upper().replace("_", "-")
    if key not in SKILL_NAMES:
        return _no(name, UNKNOWN_SKILL)
    if ctx.max_steps < 1:
        return _no(key, "max_steps below 1")
    if key == "S-AROUND":
        return _around(args, ctx)
    if key == "S-COLLECT":
        return _collect(ctx)
    if key == "S-EXPLORE":
        return _explore(ctx)
    return _no(key, NOT_BUILT)


@dataclass(frozen=True)
class _Mover:
    """The controlled thing on screen with its gaits known: name, body, and the body's
    cell offsets from its top-left."""

    name: str
    body: Thing
    offs: tuple[Cell, ...]

    @property
    def start(self) -> Cell:
        return (self.body.x, self.body.y)


def _mover(ctx: SkillContext) -> tuple[_Mover | None, str]:
    """The mover, or None and the reason it cannot walk."""
    if ctx.mover is None:
        return None, "no mover"
    body = ctx.kin.body(ctx.scene, ctx.mover)
    if body is None:
        return None, f"mover {ctx.mover} is not on screen"
    if not ctx.kin.gaits(ctx.mover):
        return None, f"mover {ctx.mover} has no gaits"
    offs = tuple(sorted((x - body.x, y - body.y) for x, y in body.cells))
    return _Mover(ctx.mover, body, offs), ""


def _around(args: Mapping[str, object], ctx: SkillContext) -> SkillResult:
    mover, why = _mover(ctx)
    if mover is None:
        return _no("S-AROUND", why)
    target, how = _around_target(args, ctx, mover.body)
    if target is None:
        return _no("S-AROUND", f"no target: {how}")
    avoid = _harm_cells(ctx) | _arg_cells(args, "avoid_cells")
    reach = ctx.kin.reachable_avoiding(ctx.scene, mover.name, avoid)
    tag = f"#{target.id} {target.kind} ({how}) around {len(avoid)} avoided cells"
    return _finish(
        "S-AROUND",
        reach,
        mover,
        target.cells,
        tag=tag,
        max_steps=ctx.max_steps,
        overlap_first=False,
    )


def _around_target(
    args: Mapping[str, object], ctx: SkillContext, body: Thing
) -> tuple[Thing | None, str]:
    """The thing S-AROUND walks to and how it was chosen, or None and why there is none.

    ``args["target"]`` (a thing id) wins when that thing is on screen. Otherwise the
    nearest thing (cell gap, ties by id) answering to ``b`` of the first reach or cover
    goal whose ``a`` the mover answers to and that has a target on screen.
    """
    tid = _arg_int(args, "target")
    if tid is not None:
        t = ctx.scene.by_id().get(tid)
        if t is not None and t.id != body.id:
            return t, "from args"
    start = (body.x, body.y)
    for goal in list(ctx.goals)[:MAX_OPEN_GOALS]:
        if goal.kind not in ("reach", "cover") or not answers(body, goal.a):
            continue
        targets = [t for t in ctx.scene.things if t.id != body.id and answers(t, goal.b)]
        if targets:
            best = min(targets, key=lambda t: (_cells_apart((t.x, t.y), start), t.id))
            return best, f"goal {goal.id()}"
    if tid is not None:
        return None, f"#{tid} is not on screen and no reach or cover goal has a target on screen"
    return None, "no reach or cover goal for the mover has a target on screen"


def _collect(ctx: SkillContext) -> SkillResult:
    mover, why = _mover(ctx)
    if mover is None:
        return _no("S-COLLECT", why)
    found = [
        t
        for t in ctx.scene.things
        if t.id != mover.body.id and _role_of(t, ctx.roles) == "collectible"
    ]
    if not found:
        return _no("S-COLLECT", "no collectibles on screen")
    found.sort(key=lambda t: (_cells_apart((t.x, t.y), mover.start), t.id))
    harm = _harm_cells(ctx)
    reach = ctx.kin.reachable_avoiding(ctx.scene, mover.name, harm)
    targets = [
        t
        for t in found[:MAX_COLLECT]
        if _contact(reach, mover.offs, t.cells, overlap_first=True) is not None
    ]
    if not targets:
        return _no(
            "S-COLLECT",
            f"no path to any of {len(found)} collectibles around {len(harm)} harm cells",
        )
    first = targets[_route_order(ctx, mover, targets, reach, harm)[0]]
    tag = f"collectible #{first.id} {first.kind}, first of {len(targets)} routed"
    return _finish(
        "S-COLLECT", reach, mover, first.cells, tag=tag, max_steps=ctx.max_steps, overlap_first=True
    )


def _route_order(
    ctx: SkillContext, mover: _Mover, targets: Sequence[Thing], reach: _Reach, harm: frozenset[Cell]
) -> list[int]:
    """Indices into ``targets`` in visiting order (``route.route``).

    One walk per distinct collectible top-left stands in for "the mover after taking
    it" (``_standing_at``); a leg's length is the fewest moves from that walk to a
    top-left in contact with the next collectible, else the cell gap.
    """
    at: dict[Cell, list[Thing]] = {}
    for t in targets:
        at.setdefault((t.x, t.y), []).append(t)
    walks: dict[Cell, _Reach] = {mover.start: reach}
    for tl, group in at.items():
        if tl not in walks:
            scene = _standing_at(ctx.scene, mover.body, group[0])
            walks[tl] = ctx.kin.reachable_avoiding(scene, mover.name, harm)

    def dist(a: Cell, b: Cell) -> int | None:
        walk = walks.get(a)
        if walk is None:
            return _cells_apart(a, b)
        legs = [
            walk[end][0]
            for t in at.get(b, ())
            if (end := _contact(walk, mover.offs, t.cells, overlap_first=True)) is not None
        ]
        return min(legs) if legs else _cells_apart(a, b)

    found = route(mover.start, [(t.x, t.y) for t in targets], dist)
    return found[0] if found is not None else list(range(len(targets)))


def _standing_at(scene: Scene, body: Thing, taken: Thing) -> Scene:
    """``scene`` after the mover takes ``taken``: the mover's top-left moved to the
    collectible's, the collectible gone, everything else as it was."""
    dx, dy = taken.x - body.x, taken.y - body.y
    cells = frozenset((x + dx, y + dy) for x, y in body.cells)
    moved = Thing(body.id, body.color, body.shape, body.sig, taken.x, taken.y, cells, body.kind)
    things = tuple(moved if t.id == body.id else t for t in scene.things if t.id != taken.id)
    return replace(scene, things=things)


def _explore(ctx: SkillContext) -> SkillResult:
    mover, why = _mover(ctx)
    picked = _pick_flag(ctx, mover)
    if picked is None:
        return _no("S-EXPLORE", f"no flag with a thing on screen among {len(ctx.flags)} flags")
    (kind, value, cost), thing = picked
    tag = f"flag {kind} (#{thing.id}, value {value:.2f}, cost {cost:.2f})"
    if ctx.can_click:
        x, y = ctx.click_cell(thing)
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return _no("S-EXPLORE", f"click cell ({x}, {y}) for {tag} is off the board")
        return SkillResult("S-EXPLORE", (Action(COORD_ACTION_ID, x, y),), f"click {tag}", True)
    if mover is None:
        return _no("S-EXPLORE", f"no click and {why}")
    harm = _harm_cells(ctx)
    reach = ctx.kin.reachable_avoiding(ctx.scene, mover.name, harm)
    tag = f"{tag} around {len(harm)} harm cells"
    return _finish(
        "S-EXPLORE",
        reach,
        mover,
        thing.cells,
        tag=tag,
        max_steps=ctx.max_steps,
        overlap_first=False,
    )


def _pick_flag(ctx: SkillContext, mover: _Mover | None) -> tuple[Flag, Thing] | None:
    """The first flag (best first) with a thing on screen that is not the mover, and that
    thing: the nearest to the mover by cell gap (ties by id), the lowest id with no mover."""
    me = mover.body.id if mover is not None else None
    anchor = mover.start if mover is not None else (0, 0)
    for flag in list(ctx.flags)[:MAX_FLAGS]:
        things = [t for t in ctx.scene.things if t.id != me and answers(t, flag[0])]
        if things:
            return flag, min(things, key=lambda t: (_cells_apart((t.x, t.y), anchor), t.id))
    return None


def _finish(
    name: str,
    reach: _Reach,
    mover: _Mover,
    cells: frozenset[Cell],
    *,
    tag: str,
    max_steps: int,
    overlap_first: bool,
) -> SkillResult:
    """The result of walking ``reach`` to contact with ``cells``: no path, already in
    contact, or the moves cut to ``max_steps``."""
    end = _contact(reach, mover.offs, cells, overlap_first=overlap_first)
    if end is None:
        return _no(name, f"no path to {tag}")
    actions = _path(reach, end)
    if not actions:
        return _no(name, f"already touching {tag}")
    return _ok(name, actions, max_steps, f"{len(actions)} moves to {tag}")


def _contact(
    reach: _Reach, offs: Sequence[Cell], cells: frozenset[Cell], *, overlap_first: bool
) -> Cell | None:
    """The reachable top-left with the fewest moves (ties by cell) whose body touches
    ``cells``; with ``overlap_first``, one whose body overlaps them when any is reachable."""
    near = frozenset((x + dx, y + dy) for x, y in cells for dx, dy in _TOUCH)
    over: tuple[int, Cell] | None = None
    touch: tuple[int, Cell] | None = None
    for tl, (moves, _, _) in reach.items():
        body = [(tl[0] + ox, tl[1] + oy) for ox, oy in offs]
        if any(c in cells for c in body):
            if over is None or (moves, tl) < over:
                over = (moves, tl)
        elif any(c in near for c in body) and (touch is None or (moves, tl) < touch):
            touch = (moves, tl)
    if overlap_first and over is not None:
        return over[1]
    both = [x for x in (over, touch) if x is not None]
    return min(both)[1] if both else None


def _path(reach: _Reach, end: Cell) -> tuple[Action, ...]:
    """The actions from the start of ``reach`` to ``end``, read back through the previous
    top-lefts; empty when ``end`` is the start."""
    out: list[Action] = []
    here = end
    for _ in range(len(reach)):
        _, prev, action = reach[here]
        if prev is None or action is None:
            break
        out.append(action)
        here = prev
    out.reverse()
    return tuple(out)


def _ok(name: str, actions: tuple[Action, ...], max_steps: int, why: str) -> SkillResult:
    cut = actions[:max_steps]
    if len(cut) < len(actions):
        why = f"{why}, cut to {len(cut)}"
    return SkillResult(name, cut, why, True)


def _no(name: str, why: str) -> SkillResult:
    return SkillResult(name, (), why, False)


def _role_of(t: Thing, roles: Mapping[str, Role]) -> Role:
    """The role of a thing by its kind, then its sig; ``unknown`` when neither is mapped."""
    return roles.get(t.kind) or roles.get(t.sig) or "unknown"


def _harm_cells(ctx: SkillContext) -> frozenset[Cell]:
    """Every cell of every harm-role thing on screen."""
    return frozenset(
        c for t in ctx.scene.things if _role_of(t, ctx.roles) == "harm" for c in t.cells
    )


def _cells_apart(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _arg_int(args: Mapping[str, object], key: str) -> int | None:
    """``args[key]`` when it is an int (bools excluded), else None."""
    v = args.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _arg_cells(args: Mapping[str, object], key: str) -> frozenset[Cell]:
    """Cells in ``args[key]`` given as a list of ``[x, y]`` pairs of ints; anything else in
    the list is skipped, and a value that is not a list reads as no cells. At most
    ``MAX_AVOID_CELLS`` entries are read."""
    raw = args.get(key)
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    out: set[Cell] = set()
    for item in list(raw)[:MAX_AVOID_CELLS]:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        x, y = item
        if _is_int(x) and _is_int(y):
            out.add((x, y))
    return frozenset(out)


def _is_int(v: object) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool)
