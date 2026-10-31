"""The planner's briefing: fixed lists rendered to text, plus the planner's messages.

Spec 4.9a (briefing lists), 4.9c (question library), 4.9f (planner call) and plan
section 3 (the v2 reply contract). The agent builds the briefing from its stores and the
simulator, never from the planner's own past reply, so the planner cannot read its own
guess back as a fact. The one exception is ``model``, the planner's notes from its last
reply, which render last under a title that says they are unverified.

Contract:
- ``Briefing`` holds the lists of spec 4.9a as tuples of pre-rendered lines (one line per
  item), the pre-answered questions as ``(id, answer)`` pairs, the model notes, and the
  wake reason, level and horizon. The builder orders every list with the lines to keep
  first (most important first; for ``recent``, newest first), because ``render`` keeps the
  head of each list and drops the tail.
- ``render`` writes a header (Wake, Level, Horizon), then one section per list in
  ``SECTION_ORDER`` under its title from ``SECTION_TITLES`` (questions as ``<id>: <answer>``),
  then the model notes under ``MODEL_TITLE``. Each list is cut at ``SECTION_CAPS[name]``
  lines with a trailing ``(+N more)`` line when cut; each line is clipped to ``LINE_WIDTH``
  characters with whitespace collapsed; an empty list reads ``(none)``; the model notes are
  cut at ``MAX_MODEL_CHARS``. The whole text is at most ``cap_chars`` characters (default
  ``MAX_BRIEFING_CHARS``): a longer text is cut on the last line boundary that fits and
  ends with the line ``CUT_LINE``.
- ``planner_messages`` returns ``[system, user]``: the system prompt with the horizon
  substituted for ``HORIZON_TOKEN``, and the user content as the rendered text, the screen
  image and, when ``prev_grid`` is given, the diff image (the parts ``turn.build_messages``
  sends).
- Pure functions, no logging: the agent that wires the call logs the briefing and reply.
"""

from __future__ import annotations

from dataclasses import dataclass

from arcagent.core.types import Grid
from arcagent.mind.contract import MAX_MODEL_CHARS
from arcagent.mind.screen import data_url, diff_png, png
from arcagent.mind.vchat import VMessage

LINE_WIDTH = 160
MAX_BRIEFING_CHARS = 12000
CUT_LINE = "(briefing cut)"
HORIZON_TOKEN = "<horizon>"
MODEL_TITLE = "Your notes from the last turn (unverified)"

SECTION_ORDER: tuple[str, ...] = (
    "state",
    "actions",
    "goals",
    "tasks",
    "level_rules",
    "game_rules",
    "hypotheses",
    "sequence",
    "forks",
    "flags",
    "self_focus",
    "recent",
    "lessons",
    "questions",
)

SECTION_TITLES: dict[str, str] = {
    "state": "Current state",
    "actions": "Actions",
    "goals": "Goals",
    "tasks": "Tasks",
    "level_rules": "Level rules",
    "game_rules": "Game rules",
    "hypotheses": "Hypotheses",
    "sequence": "Sequence",
    "forks": "Forks",
    "flags": "Flags",
    "self_focus": "Self and focus",
    "recent": "Recent moves",
    "lessons": "Lessons",
    "questions": "Questions",
}

SECTION_CAPS: dict[str, int] = {
    "state": 40,
    "actions": 1,
    "goals": 6,
    "tasks": 8,
    "level_rules": 16,
    "game_rules": 16,
    "hypotheses": 6,
    "sequence": 4,
    "forks": 5,
    "flags": 8,
    "self_focus": 2,
    "recent": 12,
    "lessons": 8,
    "questions": 7,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Briefing:
    """One planner call's input (spec 4.9a lists in order, then the question library).

    Every list field is a tuple of pre-rendered lines, ordered by the builder with the
    lines to keep first. ``questions`` pairs a question id from ``contract.QUESTION_IDS``
    with its computed answer. ``model`` is the planner's own notes from its last reply.
    ``horizon`` is the segment cap for this call, in moves.
    """

    state: tuple[str, ...]  # one line per thing: id, kind, role, known rules
    actions: tuple[str, ...]  # the valid actions and whether click is present
    goals: tuple[str, ...]  # level goal and game goal, each with its status
    tasks: tuple[str, ...]  # ordered tasks of the chosen hypothesis
    level_rules: tuple[str, ...]
    game_rules: tuple[str, ...]
    hypotheses: tuple[str, ...]  # top hypotheses with path probability and expected value
    sequence: tuple[str, ...]  # the current segment and its status
    forks: tuple[str, ...]  # open forks and their untried paths
    flags: tuple[str, ...]  # items marked check, with value and cost
    self_focus: tuple[str, ...]  # the self record line and the focus record line
    recent: tuple[str, ...]  # last moves with predicted and observed, newest first
    lessons: tuple[str, ...]  # long-term rules in words
    questions: tuple[tuple[str, str], ...]  # (question id, computed answer)
    model: str
    wake: str
    level: int
    horizon: int


def _clip(s: str, width: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= width else s[: width - 3] + "..."


def _block(title: str, lines: list[str]) -> str:
    return f"{title}:\n" + ("\n".join(lines) if lines else "(none)")


def _section(name: str, lines: tuple[str, ...]) -> str:
    """The titled block for one list: the head up to its cap, then ``(+N more)`` if cut."""
    cap = SECTION_CAPS[name]
    shown = [_clip(line, LINE_WIDTH) for line in lines[:cap]]
    if len(lines) > cap:
        shown.append(f"(+{len(lines) - cap} more)")
    return _block(SECTION_TITLES[name], shown)


def _lists(b: Briefing) -> list[tuple[str, tuple[str, ...]]]:
    """The list sections in ``SECTION_ORDER`` with the questions written as lines."""
    questions = tuple(f"{qid}: {answer}" for qid, answer in b.questions)
    return [
        ("state", b.state),
        ("actions", b.actions),
        ("goals", b.goals),
        ("tasks", b.tasks),
        ("level_rules", b.level_rules),
        ("game_rules", b.game_rules),
        ("hypotheses", b.hypotheses),
        ("sequence", b.sequence),
        ("forks", b.forks),
        ("flags", b.flags),
        ("self_focus", b.self_focus),
        ("recent", b.recent),
        ("lessons", b.lessons),
        ("questions", questions),
    ]


def _cut(text: str, cap_chars: int) -> str:
    """``text`` unchanged when it fits, else cut on a line boundary and ended with ``CUT_LINE``."""
    if len(text) <= cap_chars:
        return text
    tail = f"\n{CUT_LINE}"
    keep = max(0, cap_chars - len(tail))
    boundary = text.rfind("\n", 0, keep + 1)
    head = text[:boundary] if boundary > 0 else text[:keep]
    return head + tail


def render(b: Briefing, *, cap_chars: int = MAX_BRIEFING_CHARS) -> str:
    """The briefing as the planner reads it; see the module docstring for the layout.

    ``cap_chars`` below the length of the cut line plus one is raised to that length, so
    the result always ends on a whole ``CUT_LINE`` when it is cut.
    """
    cap_chars = max(cap_chars, len(CUT_LINE) + 1)
    model = b.model.strip()[:MAX_MODEL_CHARS]
    parts = [
        f"Wake: {_clip(b.wake, LINE_WIDTH)}",
        f"Level: {b.level}",
        f"Horizon: {b.horizon} moves",
        *(_section(name, lines) for name, lines in _lists(b)),
        _block(MODEL_TITLE, [model] if model else []),
    ]
    return _cut("\n".join(parts), cap_chars)


PLANNER_SYSTEM_PROMPT = """\
You plan moves for an agent playing an unknown grid game. You decide.
Your job: finish the level in few moves. The level score is min(human moves / your moves, 1)
squared, so every extra move costs, and twice the human's moves earns a quarter.
You get the screen, an image of what changed (changed cells bright), and a briefing in
fixed sections: Wake, Level, Horizon, Current state (each thing: #id, kind, role, known
rules), Actions, Goals, Tasks, Level rules, Game rules, Hypotheses (path probability and
expected value), Sequence, Forks, Flags, Self and focus, Recent moves (predicted and
observed), Lessons, Questions (answered from the stores), and your own notes from the last
turn, which are unverified: trust the lists over them.
Reply with JSON only, one object:
{"model": "<under 1500 chars: what each action does, what each kind is, what wins>",
 "facts": [{"action": "A1" or "click #12", "object": <id> or "any",
   "effect": "move" (with "dx", "dy" in cells; needs a real object id), "none", "disappear",
   "recolor" (with "color"), "game_over" or "level_up"}],
 "goals": [{"kind": "reach", "a": <id>, "b": <id>},
   {"kind": "clear", "color": c}, {"kind": "count", "color": c, "n": n},
   {"kind": "match", "region_a": [x0,y0,x1,y1], "region_b": [x0,y0,x1,y1]}],
 "segment": [{"action": "A3", "expect": "<object> <effect>"}],
 "skill": {"name": "S-AROUND", "args": {}},
 "question": "Q-BLOCK",
 "labels": [{"object": 12, "label": "door"}],
 "notes": "one line"}
"segment": up to <horizon> steps; actions are "A1" to "A7" or "click #<id>".
"skill": one of S-AROUND, S-COLLECT, S-EXPLORE, S-ATTACK, S-WAIT with its args. Give
"segment" or "skill", not both; when both appear the segment is kept.
"question": one of Q-BLOCK, Q-REWARD, Q-HARM, Q-FORK, Q-CHECK, Q-SAFE, Q-UNKNOWN; the
answer comes in the next briefing.
"labels": your names for kinds; "object" is an id on screen.
Ids are bare integers (12, not "#12") and valid this turn only.
The simulator checks the segment before any real move. A rejected segment comes back to
you with the failing step named; fix that step.
Omit any key you have nothing for. Never assume the game is one you know."""


def system_prompt(horizon: int) -> str:
    """``PLANNER_SYSTEM_PROMPT`` with the segment cap filled in (a horizon below 1 reads as 1)."""
    return PLANNER_SYSTEM_PROMPT.replace(HORIZON_TOKEN, str(max(1, horizon)))


def planner_messages(b: Briefing, grid: Grid, prev_grid: Grid | None) -> list[VMessage]:
    """``[system, user]``; the user content is the briefing text, the screen, and the diff
    image when ``prev_grid`` is given. The system prompt carries ``b.horizon``."""
    content: list[dict[str, object]] = [
        {"type": "text", "text": render(b)},
        {"type": "image_url", "image_url": {"url": data_url(png(grid))}},
    ]
    if prev_grid is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": data_url(diff_png(prev_grid, grid))}}
        )
    return [
        {"role": "system", "content": system_prompt(b.horizon)},
        {"role": "user", "content": content},
    ]
