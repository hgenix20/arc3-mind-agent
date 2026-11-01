"""The reply contract: parse the model's JSON reply into goals, a plan and a test.

Contract (``parse_reply``, never raises):
- Takes the first JSON object found in the text; code fences and prose around it are fine.
  No object, or a non-object, gives an empty ``Reply``.
- ``goals``: ``reach`` (``a``, ``b`` object ids, mapped to sigs), ``clear`` (``color``),
  ``count`` (``color``, ``n``), ``match`` (``region_a``, ``region_b`` as
  ``[x0, y0, x1, y1]``, end exclusive, inside the 64x64 grid). Each goal is validated;
  invalid ones are dropped and a reason goes into ``Reply.dropped``.
- ``plan``: up to ``MAX_PLAN_STEPS`` steps ``{"action", "expect"}``. ``"A1"``..``"A7"``
  map to ``Action(n)`` when ``n`` is in ``available`` and is neither 0 nor 6;
  ``"click #<id>"`` maps to ``Action(6, x, y)`` at ``click_cell`` of that thing when 6 is
  in ``available`` and the id is on screen. Invalid steps are dropped with a reason.
- ``test``: one step plus a question, kept only when the plan came back empty (a reply
  with a valid plan keeps the plan and the test is dropped). Ids are valid for one turn.
- Unknown keys are ignored.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, Action
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Scene, Thing, click_cell

MAX_PLAN_STEPS = 8
_MAX_TEXT = 300
_ACTION_RE = re.compile(r"^A([1-7])$")
_CLICK_RE = re.compile(r"^click\s*#\s*(\d{1,6})$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One step: the action and what the model expects (or, for a test, its question)."""

    action: Action
    expect: str


@dataclass(frozen=True, slots=True)
class Reply:
    """What was valid in a reply; ``dropped`` holds one reason per dropped part."""

    goals: tuple[Goal, ...] = ()
    plan: tuple[PlanItem, ...] = ()
    test: PlanItem | None = None
    notes: str = ""
    dropped: tuple[str, ...] = ()


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
    return v if isinstance(v, int) and not isinstance(v, bool) else None


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
    color = _int(g.get("color"))
    if color is None or not 0 <= color <= 15:
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
    if kind == "reach":
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


def parse_reply(text: str, scene: Scene, available: Sequence[int]) -> Reply:
    """Parse ``text`` into the valid part of a ``Reply``; see the module docstring."""
    try:
        return _parse(text, scene, frozenset(available))
    except Exception as exc:  # the contract: a bad reply is data, never a crash
        return Reply(dropped=(f"parser failure: {type(exc).__name__}",))


def _goals(raw: object, by_id: dict[int, Thing], dropped: list[str]) -> list[Goal]:
    out: list[Goal] = []
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


def _plan(
    raw: object, by_id: dict[int, Thing], available: frozenset[int], dropped: list[str]
) -> list[PlanItem]:
    out: list[PlanItem] = []
    if not isinstance(raw, list):
        if raw:
            dropped.append("plan is not a list")
        return out
    for item in raw:  # pyright: ignore[reportUnknownVariableType]
        step, why = _step(item, by_id, available)
        if step is None:
            dropped.append(why)
        elif len(out) < MAX_PLAN_STEPS:
            out.append(step)
        else:
            dropped.append(f"plan capped at {MAX_PLAN_STEPS} steps")
            break
    return out


def _parse(text: str, scene: Scene, available: frozenset[int]) -> Reply:
    obj = _first_object(text)
    if obj is None:
        return Reply(dropped=("no JSON object in reply",))
    by_id = {t.id: t for t in scene.things}
    dropped: list[str] = []
    goals = _goals(obj.get("goals", []), by_id, dropped)
    plan = _plan(obj.get("plan", []), by_id, available, dropped)
    test: PlanItem | None = None
    if obj.get("test") is not None:
        item, why = _step(obj["test"], by_id, available)
        if item is None:
            dropped.append(f"test: {why}")
        elif plan:
            dropped.append("test dropped: the plan wins")
        else:
            test = item
    return Reply(tuple(goals), tuple(plan), test, _text(obj.get("notes")), tuple(dropped))
