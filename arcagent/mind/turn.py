"""The executive turn: build the prompt (text plus screen and diff images) and call the chat.

Contract:
- ``build_messages`` returns ``[system, user]``. The user content is one text part (wake
  reason, level, available actions, object list, facts, goals with statuses, open
  questions, last moves), then the screen image, then the diff image when ``prev_grid``
  exists. The text never exceeds ``MAX_PROMPT_CHARS``: the object list is capped at
  ``MAX_OBJECTS`` and every other section has its own line and width cap.
- Object order for the cap: sigs in ``movers`` first, then ids in ``changed``, then
  ascending cell count, ties by id. The caller passes ``scene`` already stripped of noise.
  The objects left out are summarized as ``and N more: c3 x12, c9 x4`` (count per color).
- ``take_turn`` enforces no time limit; the caller runs it on a pool with a deadline.
  ``ImagesRejected`` propagates. Any other exception from the chat or the parser gives an
  empty ``Reply`` whose ``dropped`` holds the reason. ``seed`` goes to the chat unchanged.
- Ids in the text are valid for this turn only; the parser maps them back to sigs.
- Facts and goals reach the prompt through ``render_fact`` and ``name_sigs`` (the caller
  renders them on the game thread): every sig becomes the ``#id`` list of the things on
  screen with that sig (at most ``MAX_IDS`` ids, then ``+N``), or ``(off screen)``. A
  click label ``A6:<sig>@<ring>`` reads ``click on #ids``, listing only the things of that
  sig whose ``click_ring`` is ``ring``. The model never sees a raw sig.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from arcagent.core.types import Grid
from arcagent.mind.contract import Reply, parse_reply
from arcagent.mind.facts import Fact, Owner, cell_owner, click_ring
from arcagent.mind.objects import Scene, Thing
from arcagent.mind.screen import data_url, diff_png, png
from arcagent.mind.vchat import ImagesRejected, VisionChat, VMessage

MAX_OBJECTS = 40
MAX_PROMPT_CHARS = 6000
MAX_PROMPT_CHARS_DRIVE = 9000  # drive turns add the model and show more facts and moves
MOVE_LOG_LINES = 8
MAX_IDS = 4  # ids listed per sig in a fact or goal line before ``+N``

_MAX_FACTS = 12
_MAX_GOALS = 10
_MAX_QUESTIONS = 4
_FACT_WIDTH = 140
_GOAL_WIDTH = 100
_QUESTION_WIDTH = 140
_MOVE_WIDTH = 60

SYSTEM_PROMPT = """\
You are the planner for an agent learning an unknown game from scratch.
You get two images and a text description of the current state.
The first image is the screen. The second is what changed, with changed cells bright.
The object list gives exact positions: #<id> c<color> <w>x<h> at (x,y).
Infer the goal from what you see and what changed.
Propose goal guesses as predicates: reach, clear, count, match.
Propose either a short plan, each step with its expected effect,
or one test move that answers a question.
Name a simple action as "A1" to "A7", and a click as "click #<id>".
Ids are valid for this turn only.
Never assume the game works like a game you know.
Use only these root mechanics as hypotheses:
move, block, push, collect, toggle, die, win, spawn, rotate, select, no_effect.
Reply with JSON only, in this exact shape, one object:
{"goals": [{"kind": "reach", "a": 12, "b": 7, "why": "..."},
           {"kind": "clear", "color": 9},
           {"kind": "count", "color": 8, "n": 0},
           {"kind": "match", "region_a": [x0,y0,x1,y1], "region_b": [x0,y0,x1,y1]}],
 "plan": [{"action": "A3", "expect": "player moves left"}],
 "test": {"action": "click #4", "question": "does clicking the red tile toggle it?"},
 "notes": "one line"}
"a" and "b" are object ids. Give either "plan" (up to 8 steps) or "test", never both.
Omit any key you have nothing for."""


DRIVE_SYSTEM_PROMPT = """\
You drive an agent playing an unknown grid game. You decide; code assists you.
You get the screen, an image of what changed (changed cells bright), and text:
objects with exact positions (#<id> c<color> <w>x<h> at (x,y)), facts learned
from real moves, goal guesses, the last moves, and your own model from earlier turns.
Your job: keep a model of the game and use it to finish levels in few moves.
1. "model": rewrite your model in plain text (under 1500 characters): what each action
   does, what each kind of object is, what ends or wins a level, what you are unsure of.
   It is shown to you next turn; nothing else of this turn is.
2. "facts": a list of the effects your model predicts, one per action and object, so
   code can simulate moves: [{"action": "A1" or "click #12", "object": <id> or "any",
   "effect": "move" (with "dx", "dy" in cells), "none", "disappear",
   "recolor" (with "color"), "game_over" or "level_up"}, ...].
3. "goals": a list of predicates that are true when the level is won:
   {"kind": "reach", "a": <id>, "b": <id>} (a ends on or touching b),
   {"kind": "clear", "color": c}, {"kind": "count", "color": c, "n": n},
   {"kind": "match", "region_a": [x0,y0,x1,y1], "region_b": [x0,y0,x1,y1]}.
4. Either "plan": up to 8 steps [{"action": ..., "expect": "<object> <effect>"}],
   or "test": {"action": ..., "question": "..."} when you need to learn something first.
   Code plays your plan out in your model before any real move and stops at the first
   step it can't predict, which then runs as an experiment.
5. Optional "rule": a Python function rule(objects, action) for an effect the facts
   can't express. objects is a list of dicts {"id","sig","color","x","y","cells"}
   (cells a list of [x,y]); action is {"id": n, "x": x, "y": y} where n is the action
   number 1..7 (6 is a click at x, y), never an object id. Return the list of
   changed objects as dicts. Keep it general: no fixed coordinates or move lists.
Actions are "A1" to "A7" and "click #<id>" with a real id, like "click #12".
Write ids in "object", "a" and "b" as bare integers (12, not "#12").
Ids are valid for this turn only.
Prefer the cheapest move that both makes progress and tests your least certain fact.
Reply with JSON only, one object with keys "model", "facts", "goals",
"plan" or "test", and optionally "rule"."""


@dataclass(frozen=True, slots=True)
class TurnInput:
    """Everything one turn needs. ``movers`` are sigs that move; ``changed`` are ids changed
    this level; ``goals`` are (goal id, status) pairs."""

    grid: Grid
    prev_grid: Grid | None
    scene: Scene
    available: tuple[int, ...]
    facts: tuple[str, ...]
    goals: tuple[tuple[str, str], ...]
    questions: tuple[str, ...]
    moves: tuple[str, ...]
    level: int
    wake: str
    movers: frozenset[str] = frozenset()
    changed: frozenset[int] = frozenset()
    model: str = ""  # the LLM's own model from its last turn (drive turns only)
    drive: bool = False


def _clip(s: str, width: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= width else s[: width - 3] + "..."


_SIG_RE = re.compile(r"c\d+:[0-9a-f]{10}")
OFF_SCREEN = "(off screen)"


def _id_list(things: Iterable[Thing]) -> str:
    ids = sorted(t.id for t in things)
    if not ids:
        return OFF_SCREEN
    shown = " ".join(f"#{i}" for i in ids[:MAX_IDS])
    return shown if len(ids) <= MAX_IDS else f"{shown} +{len(ids) - MAX_IDS}"


def name_sigs(text: str, scene: Scene) -> str:
    """``text`` with every full sig replaced by the ids of the things on screen with it."""
    return _SIG_RE.sub(lambda m: _id_list(t for t in scene.things if t.sig == m.group(0)), text)


def _click_text(label: str, scene: Scene, owner: Owner) -> str:
    """``A6:<sig>@<ring>`` as ``click on #ids`` (things of that sig with that ring)."""
    body = label[3:]
    sig, _, ring_text = body.partition("@")
    if sig == "none":
        return "click on nothing"
    ring = int(ring_text) if ring_text.lstrip("-").isdigit() else None
    hits = [
        t
        for t in scene.things
        if t.sig == sig and (ring is None or click_ring(scene, t, owner) == ring)
    ]
    return f"click on {_id_list(hits)}"


def render_fact(fact: Fact, scene: Scene, owner: Owner | None = None) -> str:
    """One fact line for the prompt, e.g. ``A3 #2 any: moves 1 cell left [solid 6/0]``.

    Same fields as ``Fact.line`` with sigs named by on-screen ids (see the module
    docstring). ``owner`` is ``cell_owner(scene)`` when the caller has one.
    """
    label, subject, context = fact.key
    if label.startswith("A6:"):
        label = _click_text(label, scene, cell_owner(scene) if owner is None else owner)
    what = "; ".join(e.text() for e in fact.effects)
    score = f"[{fact.status} {fact.hits}/{fact.misses}]"
    return name_sigs(f"{label} {subject} {context}: {what} {score}", scene)


def _object_line(t: Thing) -> str:
    xs = [c[0] for c in t.cells]
    ys = [c[1] for c in t.cells]
    w = max(xs) - min(xs) + 1
    h = max(ys) - min(ys) + 1
    return f"#{t.id} c{t.color} {w}x{h} at ({t.x},{t.y})"


def _salience(inp: TurnInput) -> list[Thing]:
    def key(t: Thing) -> tuple[int, int, int]:
        rank = 0 if t.sig in inp.movers else 1 if t.id in inp.changed else 2
        return (rank, len(t.cells), t.id)

    return sorted(inp.scene.things, key=key)


def _object_section(inp: TurnInput) -> str:
    ordered = _salience(inp)
    shown, rest = ordered[:MAX_OBJECTS], ordered[MAX_OBJECTS:]
    lines = [_object_line(t) for t in shown]
    if rest:
        counts = Counter(t.color for t in rest)
        parts = [f"c{c} x{n}" for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        lines.append(f"and {len(rest)} more: {', '.join(parts)}")
    return "\n".join(lines) if lines else "(none)"


def _available(ids: tuple[int, ...]) -> str:
    """Simple actions as ``A<n>``; ACTION6 as the click form the contract accepts.

    RESET (0) is never offered, and ``A6`` is never shown: the parser rejects it as a
    simple action, so listing it would invite a step that gets dropped.
    """
    names = [f"A{a}" for a in ids if a not in (0, 6)]
    if 6 in ids:
        names.append("click #<id>")
    return ", ".join(names) or "none"


def _block(title: str, lines: list[str]) -> str:
    return f"{title}:\n" + ("\n".join(lines) if lines else "(none)")


def _text(inp: TurnInput) -> str:
    n_facts = 2 * _MAX_FACTS if inp.drive else _MAX_FACTS
    n_moves = 2 * MOVE_LOG_LINES if inp.drive else MOVE_LOG_LINES
    cap = MAX_PROMPT_CHARS_DRIVE if inp.drive else MAX_PROMPT_CHARS
    facts = [_clip(f, _FACT_WIDTH) for f in inp.facts[-n_facts:]]
    goals = [_clip(f"{g}: {s}", _GOAL_WIDTH) for g, s in inp.goals[:_MAX_GOALS]]
    questions = [_clip(q, _QUESTION_WIDTH) for q in inp.questions[:_MAX_QUESTIONS]]
    moves = [_clip(m, _MOVE_WIDTH) for m in inp.moves[-n_moves:]]
    head = [f"Your model so far:\n{inp.model or '(none yet)'}"] if inp.drive else []
    parts = [
        *head,
        f"Wake: {_clip(inp.wake, 80)}",
        f"Level: {inp.level}",
        "Available actions: " + _available(inp.available),
        _block("Objects", _object_section(inp).split("\n")),
        _block("Facts", facts),
        _block("Goals", goals),
        _block("Open questions", questions),
        _block("Last moves", moves),
    ]
    text = "\n".join(parts)
    return text if len(text) <= cap else text[:cap]


def build_messages(inp: TurnInput) -> list[VMessage]:
    """``[system, user]`` messages; the user content is text, screen image, optional diff image."""
    content: list[dict[str, object]] = [
        {"type": "text", "text": _text(inp)},
        {"type": "image_url", "image_url": {"url": data_url(png(inp.grid))}},
    ]
    if inp.prev_grid is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url(diff_png(inp.prev_grid, inp.grid))},
            }
        )
    return [
        {"role": "system", "content": DRIVE_SYSTEM_PROMPT if inp.drive else SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def take_turn(chat: VisionChat, inp: TurnInput, *, seed: int) -> Reply:
    """Ask the model and parse the reply. See the module docstring for the failure rules."""
    try:
        completion = chat(build_messages(inp), seed)
        return parse_reply(completion.text, inp.scene, inp.available)
    except ImagesRejected:
        raise
    except Exception as exc:  # any chat or parser failure costs only the turn
        return Reply(dropped=(f"turn failed: {type(exc).__name__}: {exc}",))
