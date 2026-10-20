"""Chat messages for synthesizing and repairing a next-frame rule module.

Contract:
- Only training transitions are shown (``trace.train[:max_transitions]``).
  Held-out transitions never appear in any prompt, including repair turns.
- The first transition shows its full starting grid; the rest are diffs.
  A transition that changes more than ``DIFF_CAP`` cells (a level change or
  a reset) is shown as a full next grid instead, at most
  ``MAX_EXTRA_FULL`` times, because a truncated diff would hide the new
  layout. Later large transitions fall back to the truncated diff.
- ``PROMPT_CHAR_CAP`` bounds the total characters of a first-round prompt
  for ``max_transitions=40`` on the public games; tests enforce it.
"""

from __future__ import annotations

import re

from arcagent.synth.render import (
    DIFF_CAP,
    action_name,
    render_diff,
    render_grid,
    render_transition,
    summary_line,
)
from arcagent.synth.trace import Trace
from arcagent.synth.verify import MAX_MISMATCHES, TransitionResult

Message = dict[str, str]

MAX_EXTRA_FULL = 2
PROMPT_CHAR_CAP = 40_000
MISMATCH_DIFF_CAP = 30
_FENCE = re.compile(r"```([A-Za-z]*)[^\n]*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = """You are inferring the rules of a turn-based grid game from observations.

Interface (ARC-AGI-3):
- The screen is a 64x64 grid of ints. Colors are 0..15; -1 marks padding outside the play area.
  A cell is grid[y][x]: x is the column, y the row, both 0..63.
- Actions: 0 = RESET (restart the current level), 1..5 and 7 = simple actions (often
  up/down/left/right/interact/undo, but you must infer what each does), 6 = ACTION6, a click
  at (x, y). Each step applies one action and yields the next grid.
- A game has several levels. Completing one raises the level counter and usually draws a new
  layout. Phases: NOT_FINISHED (playing), WIN (all levels done), GAME_OVER (lost; the next
  action is RESET).

Rendering used below:
- Colors are hex digits 0-f, padding is '.'.
- A full grid starts with 'crop x=.. y=.. w=.. h=..' then lines '<row>[-<row>]: tokens';
  a token 'c' is one cell of color c, 'c*n' is n cells of color c; a row range means those
  rows are identical. Row numbers are absolute.
- A diff lists changed cells as '(x,y) old->new'.

Task: write a Python module that defines

    def step(grid: list[list[int]], action: int, x: int = -1, y: int = -1) -> list[list[int]]:

returning the predicted next 64x64 grid after applying `action` (x, y are the click
coordinates for action 6, else -1). Requirements:
- Pure and deterministic: no randomness, no global state carried between calls, do not
  mutate the input grid.
- Standard library only. No file, network or process access.
- It must reproduce every transition shown, and it should generalize: model the game's
  objects and rules (movement, collisions, counters, what clicks do), not a lookup table.
- Optionally also define `def goal(grid: list[list[int]]) -> bool` returning True when the
  grid shows a completed level.

Answer with a single ```python fenced block containing the whole module and nothing else."""

_ANSWER_INSTRUCTION = (
    "Write the module now. Reply with exactly one ```python fenced block containing the "
    "complete module, with step() and optionally goal()."
)


def render_observations(trace: Trace, max_transitions: int) -> str:
    """The observation block of the first user message (see module contract)."""
    shown = trace.train[:max_transitions]
    if not shown:
        raise ValueError("trace has no training transitions to show")
    parts = [
        f"Game observations: {len(shown)} consecutive transitions "
        f"(the game has {trace.win_levels} levels).",
        "",
        "Starting grid (before transition #0):",
        render_grid(shown[0].grid_before),
        "",
    ]
    extra_full = 0
    for i, t in enumerate(shown):
        changed = int((t.grid_before != t.grid_after).sum())
        mode = "diff"
        if changed > DIFF_CAP and extra_full < MAX_EXTRA_FULL:
            mode = "full"
            extra_full += 1
        parts.append(render_transition(t, mode=mode, index=i))
    return "\n".join(parts)


def _hex(v: int) -> str:
    """Colors as the hex digits used everywhere else; padding and junk values as-is."""
    if v == -1:
        return "."
    return format(v, "x") if 0 <= v <= 15 else str(v)


def _render_mismatch(trace: Trace, m: TransitionResult) -> str:
    t = trace.train[m.index]
    a = t.action
    head = f"train #{m.index} {action_name(a.id, a.x, a.y)}: "
    if m.error is not None:
        head += f"crashed or invalid output: {m.error}"
    else:
        head += f"{m.cells_wrong} cells wrong"
    lines = [
        head,
        f"  observed: {summary_line(t, m.index)}",
        f"  true change: {render_diff(t.grid_before, t.grid_after, cap=MISMATCH_DIFF_CAP)}",
    ]
    if m.error is None:
        lines.append(f"  your change: {_render_cells(m.predicted, m.predicted_changes)}")
    if m.wrong:
        cells = "; ".join(f"({x},{y}) expected {_hex(e)} got {_hex(g)}" for x, y, e, g in m.wrong)
        more = m.cells_wrong - len(m.wrong)
        lines.append(f"  wrong cells: {cells}" + (f"; ... and {more} more" if more > 0 else ""))
    return "\n".join(lines)


def _render_cells(cells: tuple[tuple[int, int, int, int], ...], total: int) -> str:
    """``(x,y) old->new`` list in the diff format; ``(nothing)`` when the candidate changed
    no cell, which is the most common wrong answer and must be visible as such."""
    if total == 0:
        return "(nothing)"
    body = "; ".join(f"({x},{y}) {_hex(a)}->{_hex(b)}" for x, y, a, b in cells)
    more = total - len(cells)
    return body + (f"; ... and {more} more" if more > 0 else "")


def build_messages(
    trace: Trace,
    max_transitions: int,
    prior_code: str | None = None,
    mismatches: list[TransitionResult] | None = None,
    exact_train: tuple[int, int] | None = None,
) -> list[Message]:
    """System + user messages; with ``prior_code``, adds the assistant turn and a repair turn.

    ``mismatches`` must index ``trace.train`` (as ``verify`` produces them);
    at most ``MAX_MISMATCHES`` are shown. ``exact_train`` is ``(exact, total)``
    over the training split and opens the repair turn with a score line, so the
    model can tell a partial model from a broken one.
    """
    msgs: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_observations(trace, max_transitions) + "\n\n" + _ANSWER_INSTRUCTION,
        },
    ]
    if prior_code is None:
        return msgs
    msgs.append({"role": "assistant", "content": f"```python\n{prior_code.strip()}\n```"})
    shown = (mismatches or [])[:MAX_MISMATCHES]
    score = ""
    if exact_train is not None:
        k, n = exact_train
        score = f" It reproduced {k} of {n} training transitions exactly."
    if shown:
        body = "\n\n".join(_render_mismatch(trace, m) for m in shown)
        intro = (
            f"Your module was run on the observed transitions.{score} {len(shown)} of the "
            "transitions it got wrong are listed below (indices are training transitions; "
            "some may be later than the ones shown above). For each, 'true change' is what "
            "the game did, 'your change' is what your module did to the same grid."
        )
    else:
        body = "(no training mismatches; it still fails to generalize to later transitions)"
        intro = f"Your module was run on the observed transitions.{score}"
    repair = (
        f"{intro}\n\n{body}\n\n"
        "Fix the underlying rules so that these and similar transitions come out right. "
        "Do not special-case these examples by index, position or exact grid; a rule that "
        "only matches the listed cases will fail on unseen ones. "
        "First write at most five plain lines: which rule in your module produced the wrong "
        "cells, and what you are changing. Then reply with exactly one ```python fenced "
        "block containing the complete corrected module. Returning the module with no "
        "change to its logic wastes this round."
    )
    msgs.append({"role": "user", "content": repair})
    return msgs


def extract_code(reply: str) -> str | None:
    """Body of the last ```python fence in ``reply``, else the last bare fence; None if absent."""
    blocks = _FENCE.findall(reply)
    if not blocks:
        return None
    tagged = [body for lang, body in blocks if lang.lower() in ("python", "py")]
    body = tagged[-1] if tagged else blocks[-1][1]
    return body.strip() + "\n"
