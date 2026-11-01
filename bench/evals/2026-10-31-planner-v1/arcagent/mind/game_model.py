"""The LLM's model of the game in the agent's terms: its fact claims and a play-out over them.

Contract:
- ``seed_claims`` writes each ``FactClaim`` into ``FactMemory`` under
  ``(label, subject, "any")``, but only when that key holds no fact yet: a fact learned
  from a real move always wins over a claim, and a wrong claim is corrected by the
  ordinary miss rules once real moves contradict it. Claims are recorded with move id
  ``CLAIM_MOVE`` so their evidence is never mistaken for a real move. Returns the keys
  added, in claim order.
- ``play_out`` is the simulator: it predicts the LLM's plan step by step with the
  predictor (``assume_still`` = ``protect``) and stops at the first scene where a goal
  holds (``reached``) or the first step the model can't predict (``unknown_at``, the
  step's index). ``steps`` is the number of steps predicted, the reaching one included.
  A ``rule`` (the LLM's optional function, run by the caller wherever it runs) is asked
  for every step: when the facts can't predict the step, the rule's answer over the
  scene before the step stands in; when they can, the rule's changes are applied on top.
  A rule that returns None (failed, timed out) adds nothing.
- ``rule_objects`` and ``apply_changes`` convert between scenes and the plain dicts a
  rule sees: ``{"id", "sig", "color", "x", "y", "cells"}`` with ``cells`` a list of
  ``[x, y]``. A change names an ``id`` on the scene and any of ``color``, ``x``/``y``
  (shift the cells), ``cells``, or ``"gone": true``; anything else is ignored.
Pure apart from the rule call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from arcagent.core.types import GRID_SIZE, Action
from arcagent.mind.contract import FactClaim
from arcagent.mind.facts import CLAIM_MOVE, FactKey, FactMemory
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Cell, Scene, Thing, shape_of
from arcagent.mind.predictor import predict

RuleFn = Callable[[list[dict[str, Any]], dict[str, int]], list[dict[str, Any]] | None]


def seed_claims(mem: FactMemory, claims: Sequence[FactClaim]) -> list[FactKey]:
    """Record each claim whose key has no fact yet; return the keys added."""
    added: list[FactKey] = []
    for c in claims:
        key: FactKey = (c.label, c.subject, "any")
        if mem.get(key) is None:
            mem.record(key, c.effects, CLAIM_MOVE)
            added.append(key)
    return added


@dataclass(frozen=True, slots=True)
class PlayOut:
    """What the model says the plan does: the goal reached, steps predicted, first unknown."""

    reached: Goal | None
    steps: int
    unknown_at: int | None
    scenes: tuple[Scene, ...]


def rule_objects(scene: Scene) -> list[dict[str, Any]]:
    return [
        {
            "id": t.id,
            "sig": t.sig,
            "color": t.color,
            "x": t.x,
            "y": t.y,
            "cells": sorted([x, y] for x, y in t.cells),
        }
        for t in scene.things
    ]


def _cells(raw: object) -> frozenset[Cell] | None:
    if not isinstance(raw, list):
        return None
    out: set[Cell] = set()
    for c in raw:  # pyright: ignore[reportUnknownVariableType]
        if not (isinstance(c, list | tuple) and len(c) == 2):  # pyright: ignore[reportUnknownArgumentType]
            return None
        x, y = c  # pyright: ignore[reportUnknownVariableType]
        if not (isinstance(x, int) and isinstance(y, int)):
            return None
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return None
        out.add((x, y))
    return frozenset(out) if out else None


def _changed(t: Thing, ch: dict[str, Any]) -> Thing | None:
    """``t`` after one change dict; None when the change removes it."""
    if ch.get("gone") is True:
        return None
    color = ch.get("color", t.color)
    if not isinstance(color, int) or not 0 <= color <= 15:
        color = t.color
    cells = _cells(ch.get("cells"))
    if cells is None:
        x, y = ch.get("x", t.x), ch.get("y", t.y)
        dx = x - t.x if isinstance(x, int) else 0
        dy = y - t.y if isinstance(y, int) else 0
        cells = frozenset((cx + dx, cy + dy) for cx, cy in t.cells)
    if cells == t.cells and color == t.color:
        return t
    shape = shape_of(cells)
    sig = f"c{color}:{shape}"
    x0 = min(x for x, _ in cells)
    y0 = min(y for _, y in cells)
    return Thing(t.id, color, shape, sig, x0, y0, cells, t.kind)


def apply_changes(scene: Scene, changes: Sequence[object]) -> Scene:
    by_id: dict[int, dict[str, Any]] = {}
    for ch in changes:
        if isinstance(ch, dict):
            d: dict[str, Any] = ch  # pyright: ignore[reportUnknownVariableType]
            i = d.get("id")
            if isinstance(i, int):
                by_id[i] = d
    if not by_id:
        return scene
    things: list[Thing] = []
    for t in scene.things:
        ch = by_id.get(t.id)
        new = t if ch is None else _changed(t, ch)
        if new is not None:
            things.append(new)
    return replace(scene, things=tuple(things))


def _ask_rule(rule: RuleFn, scene: Scene, action: Action) -> list[dict[str, Any]] | None:
    try:
        return rule(rule_objects(scene), {"id": action.id, "x": action.x, "y": action.y})
    except Exception:  # a rule is model code: its failure adds nothing
        return None


def play_out(
    scene: Scene,
    actions: Sequence[Action],
    mem: FactMemory,
    goals: Sequence[Goal],
    protect: frozenset[str],
    *,
    rule: RuleFn | None = None,
) -> PlayOut:
    """Predict ``actions`` in the model; see the module docstring."""
    scenes: list[Scene] = []
    cur = scene
    for i, action in enumerate(actions):
        pred = predict(cur, action, mem, assume_still=protect)
        changes = None if rule is None else _ask_rule(rule, cur, action)
        nxt = pred.scene
        if nxt is None:
            if not changes:
                return PlayOut(None, i, i, tuple(scenes))
            nxt = apply_changes(cur, changes)
        elif changes:
            nxt = apply_changes(nxt, changes)
        scenes.append(nxt)
        cur = nxt
        hit = next((g for g in goals if g.holds(cur)), None)
        if hit is not None:
            return PlayOut(hit, i + 1, None, tuple(scenes))
    return PlayOut(None, len(scenes), None, tuple(scenes))
