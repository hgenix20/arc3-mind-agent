"""The reply contract: parse the model's JSON reply into goals, a segment, a skill and more.

Contract (``parse_reply``, never raises):
- Takes the first JSON object found in the text; code fences and prose around it are fine.
  No object, or a non-object, gives an empty ``Reply``.
- ``goals``: ``reach`` (``a``, ``b`` object ids, mapped to sigs), ``clear`` (``color``),
  ``count`` (``color``, ``n``), ``match`` (``region_a``, ``region_b`` as
  ``[x0, y0, x1, y1]``, end exclusive, inside the 64x64 grid). Each goal is validated;
  invalid ones are dropped and a reason goes into ``Reply.dropped``.
- ``segment`` (the v2 planner key, plan section 3) or ``plan`` (the DRIVE key): the steps
  ``{"action", "expect"}`` kept in ``Reply.plan``. A null key counts as absent. When both
  are given ``segment`` wins and ``plan`` is ignored with the reason
  ``plan ignored: segment given``. Up to ``max_steps`` steps are kept (default
  ``MAX_PLAN_STEPS``; a value below 1 reads as 1); the steps beyond that are dropped with
  one reason naming the key and the cap.
  ``"A1"``..``"A7"`` map to ``Action(n)`` when ``n`` is in ``available`` and is neither 0
  nor 6; ``"click #<id>"`` maps to ``Action(6, x, y)`` at ``click_cell`` of that thing when
  6 is in ``available`` and the id is on screen. Invalid steps are dropped with a reason.
- ``test``: one step plus a question, kept only when the plan came back empty (a reply
  with a valid plan keeps the plan and the test is dropped). Ids are valid for one turn.
- ``skill``: ``{"name", "args"}`` with ``name`` one of ``SKILL_NAMES`` (case and ``_``
  for ``-`` forgiven) and ``args`` an object (missing or null reads as ``{}``, at most
  ``MAX_SKILL_ARGS`` entries, values unchecked: the skill validates them). It becomes
  ``Reply.skill``. A missing name, an unknown name, a non-object skill or non-object args
  drop it with a reason. When the reply also carries a valid non-empty segment, the segment
  is kept and the skill is dropped with the reason ``skill dropped: segment given``.
- ``question``: one of ``QUESTION_IDS`` (same forgiveness as a skill name) kept in
  ``Reply.question``; an unknown id or a non-string drops it with a reason; missing, null
  or empty leaves ``""`` with no reason.
- ``labels``: ``[{"object": <id>, "label": "<text>"}]``, first ``MAX_LABELS`` items. An
  entry with an integer id on screen and a non-empty string label becomes
  ``(id, label[:MAX_LABEL_CHARS])`` (whitespace collapsed) in ``Reply.labels``; duplicates
  are kept once; every other entry is dropped with a reason.
- ``facts`` (the LLM's model of the game): ``{"action", "object", "effect", ...}`` with
  ``object`` an id on screen or ``"any"`` and ``effect`` one of ``move`` (``dx``, ``dy``
  integers), ``none``, ``disappear``, ``recolor`` (``color``), ``game_over``,
  ``level_up``. Each valid one is a ``FactClaim`` keyed like an observed fact: the action
  label (``facts.action_label``) and the thing's sig. A ``move`` needs a real object
  (with ``"any"`` it would move every unmodeled thing). Invalid ones are dropped with a
  reason.
- ``model`` (free text the LLM keeps across turns) is cut to ``MAX_MODEL_CHARS``; ``rule``
  (a Python function, run only in the tools worker) to ``MAX_RULE_CHARS``.
- Lenient where the model is sloppy but clear: a lone object where ``goals``, ``plan`` or
  ``facts`` wants a list counts as a list of one, and an integer may be written as a
  string (``"3"``, ``"#3"`` for an id, ``"c11"`` for a color as the screen text writes
  it), ``cover`` is ``reach``, the effects ``no change`` and ``blocked`` are ``none``, and
  ``disappears`` is ``disappear``.
- Unknown keys are ignored.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, Action
from arcagent.mind.facts import Effect, action_label
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Scene, Thing, click_cell

MAX_PLAN_STEPS = 8
MAX_FACTS = 24  # fact claims kept per reply
MAX_MODEL_CHARS = 1500
MAX_RULE_CHARS = 4000
MAX_LABELS = 24  # label entries read per reply
MAX_LABEL_CHARS = 30
MAX_SKILL_ARGS = 16
SKILL_NAMES = ("S-AROUND", "S-COLLECT", "S-EXPLORE", "S-ATTACK", "S-WAIT")
QUESTION_IDS = ("Q-BLOCK", "Q-REWARD", "Q-HARM", "Q-FORK", "Q-CHECK", "Q-SAFE", "Q-UNKNOWN")
_MAX_TEXT = 300
_ACTION_RE = re.compile(r"^A([1-7])$")
_CLICK_RE = re.compile(r"^click\s*#\s*(\d{1,6})$", re.IGNORECASE)
_INT_RE = re.compile(r"^#?\s*(-?\d{1,6})$")
_COLOR_RE = re.compile(r"^c?(\d{1,2})$", re.IGNORECASE)
_NONE_WORDS = frozenset({"no change", "no_change", "nothing", "no effect", "blocked"})
_GONE_WORDS = frozenset({"disappears", "disappeared", "vanish", "vanishes", "removed"})


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One step: the action and what the model expects (or, for a test, its question)."""

    action: Action
    expect: str


@dataclass(frozen=True, slots=True)
class FactClaim:
    """The LLM's claim about what an action does to a subject (a sig or ``"any"``)."""

    label: str
    subject: str
    effects: tuple[Effect, ...]


@dataclass(frozen=True, slots=True)
class SkillCall:
    """A strategy skill the planner asked for (spec 4.9d).

    ``name`` is one of ``SKILL_NAMES``; ``args`` are the reply's arguments with string keys
    and unchecked values, so the skill that runs them validates them. ``args`` stays out
    of the hash because a dict cannot be hashed; equality still compares it.
    """

    name: str
    args: dict[str, object] = field(hash=False)


@dataclass(frozen=True, slots=True)
class Reply:
    """What was valid in a reply; ``dropped`` holds one reason per dropped part.

    ``plan`` holds the segment (the v2 planner's ``segment`` key or the DRIVE ``plan`` key).
    ``skill``, ``question`` and ``labels`` are the v2 planner parts (plan section 3).
    """

    goals: tuple[Goal, ...] = ()
    plan: tuple[PlanItem, ...] = ()
    test: PlanItem | None = None
    notes: str = ""
    dropped: tuple[str, ...] = ()
    claims: tuple[FactClaim, ...] = ()
    model: str = ""
    rule: str = ""
    skill: SkillCall | None = None
    question: str = ""
    labels: tuple[tuple[int, str], ...] = ()


def _first_object(text: str) -> dict[str, Any] | None:
    dec = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = dec.raw_decode(text, pos)
        except (ValueError, RecursionError):
            obj = None
        if isinstance(obj, dict):
            return dict(obj)  # pyright: ignore[reportUnknownArgumentType]
        pos = text.find("{", pos + 1)
    return None


def _int(v: object) -> int | None:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, str) and (m := _INT_RE.match(v.strip())):
        return int(m.group(1))
    return None


def _color(v: object) -> int | None:
    """A color 0..15 as an integer or as the screen text writes it (``c11``)."""
    if isinstance(v, str) and (m := _COLOR_RE.match(v.strip())):
        v = int(m.group(1))
    c = _int(v)
    return c if c is not None and 0 <= c <= 15 else None


def _as_list(raw: object) -> object:
    """A lone object where a list belongs counts as a list of one."""
    return [raw] if isinstance(raw, dict) else raw


def _text(v: object) -> str:
    return v.strip()[:_MAX_TEXT] if isinstance(v, str) else ""


def _region(v: object) -> tuple[int, int, int, int] | None:
    if not isinstance(v, list | tuple) or len(v) != 4:  # pyright: ignore[reportUnknownArgumentType]
        return None
    vals = [_int(x) for x in v]  # pyright: ignore[reportUnknownVariableType]
    if any(x is None for x in vals):
        return None
    x0, y0, x1, y1 = (x for x in vals if x is not None)
    if not (0 <= x0 < x1 <= GRID_SIZE and 0 <= y0 < y1 <= GRID_SIZE):
        return None
    return x0, y0, x1, y1


def _reach(g: dict[str, Any], by_id: dict[int, Thing]) -> Goal | str:
    a, b = _int(g.get("a")), _int(g.get("b"))
    if a is None or b is None:
        return "reach needs integer ids a and b"
    if a not in by_id or b not in by_id:
        return f"reach id not on screen: #{a} or #{b}"
    if a == b:
        return "reach needs two different objects"
    return Goal("reach", a=by_id[a].sig, b=by_id[b].sig)


def _color_goal(kind: str, g: dict[str, Any]) -> Goal | str:
    color = _color(g.get("color"))
    if color is None:
        return f"{kind} needs a color 0..15"
    if kind == "clear":
        return Goal("clear", color=color)
    n = _int(g.get("n"))
    if n is None or n < 0:
        return "count needs n >= 0"
    return Goal("count", color=color, n=n)


def _match(g: dict[str, Any]) -> Goal | str:
    ra, rb = _region(g.get("region_a")), _region(g.get("region_b"))
    if ra is None or rb is None:
        return "match needs two regions [x0,y0,x1,y1] inside the grid"
    return Goal("match", region_a=ra, region_b=rb)


def _goal(raw: object, by_id: dict[int, Thing]) -> Goal | str:
    """A ``Goal``, or the reason it was dropped."""
    if not isinstance(raw, dict):
        return "goal is not an object"
    g: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    kind = g.get("kind")
    if kind in ("reach", "cover"):  # cover: a ends on b, which reach already means
        return _reach(g, by_id)
    if kind in ("clear", "count"):
        return _color_goal(kind, g)
    if kind == "match":
        return _match(g)
    return f"unknown goal kind: {str(kind)[:30]}"


def _action(name: str, by_id: dict[int, Thing], available: frozenset[int]) -> Action | str:
    """The ``Action`` named by ``"A1"``..``"A7"`` or ``"click #<id>"``, or a drop reason."""
    simple = _ACTION_RE.match(name.upper())
    if simple:
        n = int(simple.group(1))
        if n == COORD_ACTION_ID:
            return "A6 is a click: write click #<id>"
        return Action(n) if n in available else f"action not available: A{n}"
    click = _CLICK_RE.match(name)
    if click is None:
        return f"unknown action: {name[:30]}"
    if COORD_ACTION_ID not in available:
        return "clicks are not available"
    t = by_id.get(int(click.group(1)))
    if t is None:
        return f"click id not on screen: #{click.group(1)}"
    x, y = click_cell(t)
    return Action(COORD_ACTION_ID, x, y)


def _step(
    raw: object, by_id: dict[int, Thing], available: frozenset[int]
) -> tuple[PlanItem | None, str]:
    """A ``PlanItem`` built from ``{"action", "expect"|"question"}``, or a drop reason."""
    if not isinstance(raw, dict):
        return None, "step is not an object"
    s: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    name = s.get("action")
    if not isinstance(name, str):
        return None, "step has no action name"
    act = _action(name.strip(), by_id, available)
    if isinstance(act, str):
        return None, act
    return PlanItem(act, _text(s.get("expect")) or _text(s.get("question"))), ""


def parse_reply(
    text: str, scene: Scene, available: Sequence[int], *, max_steps: int = MAX_PLAN_STEPS
) -> Reply:
    """Parse ``text`` into the valid part of a ``Reply``; see the module docstring.

    ``max_steps`` is the segment cap (the planner's horizon); a value below 1 reads as 1.
    """
    try:
        return _parse(text, scene, frozenset(available), max(1, max_steps))
    except Exception as exc:  # the contract: a bad reply is data, never a crash
        return Reply(dropped=(f"parser failure: {type(exc).__name__}",))


def _goals(raw: object, by_id: dict[int, Thing], dropped: list[str]) -> list[Goal]:
    out: list[Goal] = []
    raw = _as_list(raw)
    if not isinstance(raw, list):
        if raw:
            dropped.append("goals is not a list")
        return out
    for item in raw:  # pyright: ignore[reportUnknownVariableType]
        g = _goal(item, by_id)
        if isinstance(g, Goal):
            if g not in out:
                out.append(g)
        else:
            dropped.append(g)
    return out


def _segment_source(obj: dict[str, Any], dropped: list[str]) -> tuple[str, object]:
    """The key the steps come from (``segment`` wins over ``plan``) and its raw value."""
    segment, plan = obj.get("segment"), obj.get("plan")
    if segment is None:
        return "plan", [] if plan is None else plan
    if plan is not None:
        dropped.append("plan ignored: segment given")
    return "segment", segment


def _plan(
    raw: object,
    by_id: dict[int, Thing],
    available: frozenset[int],
    dropped: list[str],
    *,
    key: str,
    max_steps: int,
) -> list[PlanItem]:
    out: list[PlanItem] = []
    raw = _as_list(raw)
    if not isinstance(raw, list):
        if raw:
            dropped.append(f"{key} is not a list")
        return out
    for item in raw:  # pyright: ignore[reportUnknownVariableType]
        step, why = _step(item, by_id, available)
        if step is None:
            dropped.append(why)
        elif len(out) < max_steps:
            out.append(step)
        else:
            dropped.append(f"{key} capped at {max_steps} steps")
            break
    return out


def _test(
    raw: object,
    by_id: dict[int, Thing],
    available: frozenset[int],
    has_plan: bool,
    dropped: list[str],
) -> PlanItem | None:
    """The test step, kept only when no plan step survived."""
    if raw is None:
        return None
    item, why = _step(raw, by_id, available)
    if item is None:
        dropped.append(f"test: {why}")
        return None
    if has_plan:
        dropped.append("test dropped: the plan wins")
        return None
    return item


def _canon_id(name: str) -> str:
    """A skill or question id as the lists spell it: upper case, ``-`` for ``_``."""
    return name.strip().upper().replace("_", "-")


def _skill(raw: object, dropped: list[str]) -> SkillCall | None:
    """The ``SkillCall`` a ``skill`` object names, or ``None`` with the reason recorded."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        dropped.append("skill is not an object")
        return None
    s: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    name = s.get("name")
    if not isinstance(name, str) or _canon_id(name) not in SKILL_NAMES:
        dropped.append(f"unknown skill: {str(name)[:30]}")
        return None
    args = s.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        dropped.append("skill args is not an object")
        return None
    items: list[tuple[object, object]] = list(args.items())
    if len(items) > MAX_SKILL_ARGS:
        dropped.append(f"skill args capped at {MAX_SKILL_ARGS} entries")
    return SkillCall(_canon_id(name), {str(k): v for k, v in items[:MAX_SKILL_ARGS]})


def _question(raw: object, dropped: list[str]) -> str:
    """The question id from ``QUESTION_IDS``, or ``""`` (with a reason unless it was empty)."""
    if raw is None or raw == "":
        return ""
    if not isinstance(raw, str):
        dropped.append("question is not a string")
        return ""
    q = _canon_id(raw)
    if q in QUESTION_IDS:
        return q
    dropped.append(f"unknown question: {raw.strip()[:30]}")
    return ""


def _label(raw: object, by_id: dict[int, Thing]) -> tuple[int, str] | str:
    """``(id, text)`` for one label entry, or the reason it was dropped."""
    if not isinstance(raw, dict):
        return "label is not an object"
    d: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    oid = _int(d.get("object"))
    if oid is None:
        return "label needs an integer object id"
    if oid not in by_id:
        return f"label object not on screen: #{oid}"
    text = d.get("label")
    if not isinstance(text, str) or not text.strip():
        return f"label for #{oid} needs text"
    return oid, " ".join(text.split())[:MAX_LABEL_CHARS]


def _labels(raw: object, by_id: dict[int, Thing], dropped: list[str]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    raw = _as_list(raw)
    if not isinstance(raw, list):
        if raw:
            dropped.append("labels is not a list")
        return out
    if len(raw) > MAX_LABELS:
        dropped.append(f"labels capped at {MAX_LABELS}")
    for item in raw[:MAX_LABELS]:  # pyright: ignore[reportUnknownVariableType]
        lab = _label(item, by_id)
        if isinstance(lab, str):
            dropped.append(lab)
        elif lab not in out:
            out.append(lab)
    return out


def _effect(f: dict[str, Any]) -> Effect | str:
    kind = f.get("effect")
    if isinstance(kind, str) and kind.strip().lower() in _NONE_WORDS:
        kind = "none"
    elif isinstance(kind, str) and kind.strip().lower() in _GONE_WORDS:
        kind = "disappear"
    if kind == "move":
        dx, dy = _int(f.get("dx", 0)), _int(f.get("dy", 0))
        if dx is None or dy is None or (dx, dy) == (0, 0):
            return "move needs integer dx, dy, not both 0"
        return Effect("move", dx, dy)
    if kind == "recolor":
        color = _color(f.get("color"))
        if color is None:
            return "recolor needs a color 0..15"
        return Effect("recolor", color=color)
    if kind in ("none", "disappear", "game_over", "level_up"):
        return Effect(kind)
    return f"unknown effect: {str(kind)[:30]}"


def _claim(  # noqa: PLR0911  (one return per rejection reason)
    raw: object, scene: Scene, by_id: dict[int, Thing], available: frozenset[int]
) -> FactClaim | str:
    if not isinstance(raw, dict):
        return "fact is not an object"
    f: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    name = f.get("action")
    if not isinstance(name, str):
        return "fact has no action name"
    act = _action(name.strip(), by_id, available)
    if isinstance(act, str):
        return f"fact: {act}"
    obj = f.get("object", "any")
    if obj == "any":
        subject = "any"
    else:
        oid = _int(obj)
        if oid is None or oid not in by_id:
            return f"fact object not on screen: #{str(obj)[:10]}"
        subject = by_id[oid].sig
    effect = _effect(f)
    if isinstance(effect, str):
        return f"fact: {effect}"
    if subject == "any" and effect.kind == "move":
        return "fact: a move names its object"  # a wide move moves walls in play-outs
    return FactClaim(action_label(act, scene), subject, (effect,))


def _claims(
    raw: object,
    scene: Scene,
    by_id: dict[int, Thing],
    available: frozenset[int],
    dropped: list[str],
) -> list[FactClaim]:
    out: list[FactClaim] = []
    raw = _as_list(raw)
    if not isinstance(raw, list):
        if raw:
            dropped.append("facts is not a list")
        return out
    for item in raw[:MAX_FACTS]:  # pyright: ignore[reportUnknownVariableType]
        c = _claim(item, scene, by_id, available)
        if isinstance(c, FactClaim):
            if c not in out:
                out.append(c)
        else:
            dropped.append(c)
    return out


def _long_text(v: object, cap: int) -> str:
    return v.strip()[:cap] if isinstance(v, str) else ""


def _parse(text: str, scene: Scene, available: frozenset[int], max_steps: int) -> Reply:
    obj = _first_object(text)
    if obj is None:
        return Reply(dropped=("no JSON object in reply",))
    by_id = {t.id: t for t in scene.things}
    dropped: list[str] = []
    goals = _goals(obj.get("goals", []), by_id, dropped)
    key, raw_plan = _segment_source(obj, dropped)
    plan = _plan(raw_plan, by_id, available, dropped, key=key, max_steps=max_steps)
    test = _test(obj.get("test"), by_id, available, bool(plan), dropped)
    claims = _claims(obj.get("facts", []), scene, by_id, available, dropped)
    skill = _skill(obj.get("skill"), dropped)
    if skill is not None and plan:  # segment and skill are alternatives; the segment wins
        dropped.append("skill dropped: segment given")
        skill = None
    question = _question(obj.get("question"), dropped)
    labels = _labels(obj.get("labels", []), by_id, dropped)
    return Reply(  # every part is parsed above so ``dropped`` is complete here
        goals=tuple(goals),
        plan=tuple(plan),
        test=test,
        notes=_text(obj.get("notes")),
        dropped=tuple(dropped),
        claims=tuple(claims),
        model=_long_text(obj.get("model"), MAX_MODEL_CHARS),
        rule=_long_text(obj.get("rule"), MAX_RULE_CHARS),
        skill=skill,
        question=question,
        labels=tuple(labels),
    )
