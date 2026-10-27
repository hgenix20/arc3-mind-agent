"""The guided explorer with the LLM choosing what to head for.

Intent: ``GuidedExplorer`` heads for screens whose color gaps and counts move the way
they moved at an earlier win, but its own credit is often empty or noisy. A vision
LLM can usually say what a level wants ("the white block reaches the red circle").
This policy asks it, in the background, which measurements win the level, and puts
the answer into the explorer's credit. The explorer never waits for the model: it
keeps playing, and a reply that arrives changes where it heads next.

Contract:
- A question is sent at the start of each level (the first frame of a new level) and
  again every ``REASK_MOVES`` moves on a level without a win, at most one at a time.
- The prompt: the screen image; the colors on screen with cell counts and bounding
  boxes; the numbered objects with their boxes: single-color components and
  connected regions of any non-background colors (a picture on a tile is one region),
  each of at least ``MIN_OBJECT`` cells, at most ``MAX_OBJECTS``, largest first; how often each
  action kind changed the screen; and the agent's earlier answers.
- The reply is a JSON object ``{"goal": str, "measures": [...]}``. A measure is
  ``{"gap": [a, b]}`` (colors a and b should touch: smaller is better),
  ``{"count": c, "want": "zero" | "less" | "more"}``, or ``{"match": [i, j]}`` (the box
  of object i should come to look exactly like the box of object j, which must be the
  same size; the score is the number of cells that differ, read from each state's own
  grid). Unknown colors or ids, malformed entries and colors not on screen are dropped.
  A reply with no valid measure changes nothing.
- A valid reply replaces the LLM part of the credit (weight ``LLM_WEIGHT`` per measure);
  the statistical credit from earlier wins stays. Without a chat (or after a failed
  call) the policy is exactly ``GuidedExplorer``.
- Every question and answer is kept in ``history`` for the run log.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from arcagent.agent.guided import Feature, GuidedExplorer
from arcagent.core.types import Action, Grid, Observation
from arcagent.mind.screen import data_url, png
from arcagent.mind.vchat import VisionChat, VMessage
from arcagent.synth.trace import background_color, components

REASK_MOVES = 1500  # moves on a level without a win before asking again
LLM_WEIGHT = 2.0  # credit per measure the model names
MAX_MEASURES = 4
MIN_OBJECT = 4  # cells; smaller components aren't listed
MAX_OBJECTS = 60
Box = tuple[int, int, int, int]  # x0, y0, x1, y1 inclusive

SYSTEM_PROMPT = """You are helping an agent play an unknown grid game. Each level is won
by reaching some configuration of the screen. Colors are numbered 0-15 and written cN.

Look at the screen and the notes, then say what this level most likely wants, as
measurements the agent can track:
- {"gap": [a, b]}: something of color a should touch or reach something of color b.
- {"count": c, "want": "zero" | "less" | "more"}: the number of cells of color c
  should drop to zero, go down, or go up.
- {"match": [i, j]}: the box of object #i should come to look exactly like the box of
  object #j (both boxes the same size), for example a slot that must show the same
  picture as a key or legend entry.

Reply with one JSON object and nothing else:
{"goal": "<one sentence>", "measures": [<at most 4 measurements, most important first>]}
"""


def screen_notes(grid: Grid) -> str:
    """Colors on screen with cell counts and bounding boxes, largest first."""
    vals, counts = np.unique(grid, return_counts=True)
    lines = []
    for v, n in sorted(zip(vals.tolist(), counts.tolist(), strict=True), key=lambda t: -t[1]):
        if v < 0:
            continue
        ys, xs = np.nonzero(grid == v)
        lines.append(f"c{v}: {n} cells, x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}")
    return "\n".join(lines)


def regions(grid: Grid) -> list[list[tuple[int, int]]]:
    """4-connected regions of any non-background colors, as ``(y, x)`` cells."""
    bg = background_color(grid)
    fg = (grid >= 0) & (grid != bg)
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for y0, x0 in zip(*np.nonzero(fg), strict=True):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        cells: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append(cells)
    return out


def objects(grid: Grid) -> list[Box]:
    """Boxes of the multi-color regions and the single-color components of at least
    ``MIN_OBJECT`` cells (one box once), largest first, at most ``MAX_OBJECTS``; the
    list index is the object's number."""
    sized: dict[Box, int] = {}
    for cells in [*regions(grid), *components(grid)]:
        if len(cells) < MIN_OBJECT:
            continue
        ys = [y for y, _ in cells]
        xs = [x for _, x in cells]
        box = (min(xs), min(ys), max(xs), max(ys))
        sized[box] = max(sized.get(box, 0), len(cells))
    return sorted(sized, key=lambda b: (-sized[b], b))[:MAX_OBJECTS]


def object_notes(grid: Grid, boxes: list[Box]) -> str:
    """One line per object: number, size, box corner and the colors inside its box."""
    lines = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        colors = sorted({int(v) for v in np.unique(grid[y0 : y1 + 1, x0 : x1 + 1])})
        lines.append(f"#{i}: {x1 - x0 + 1}x{y1 - y0 + 1} at ({x0},{y0}), colors {colors}")
    return "\n".join(lines)


def mismatch(grid: Grid, a: Box, b: Box) -> int:
    """Cells that differ between the two same-size boxes of ``grid``."""
    ra = grid[a[1] : a[3] + 1, a[0] : a[2] + 1]
    rb = grid[b[1] : b[3] + 1, b[0] : b[2] + 1]
    return int((ra != rb).sum())


def _reply_doc(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m is None:
        return None
    try:
        doc = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None  # pyright: ignore[reportUnknownVariableType]


def parse_matches(text: str, boxes: list[Box]) -> list[tuple[Box, Box]]:
    """The reply's valid ``match`` measures as box pairs (same size, different boxes)."""
    doc = _reply_doc(text)
    raw: object = doc.get("measures", []) if doc else []
    out: list[tuple[Box, Box]] = []
    for item in raw if isinstance(raw, list) else []:  # pyright: ignore[reportUnknownVariableType]
        pair: object = item.get("match") if isinstance(item, dict) else None  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(pair, list) or len(pair) != 2:  # pyright: ignore[reportUnknownArgumentType]
            continue
        i, j = (_object_id(v) for v in pair)  # pyright: ignore[reportUnknownVariableType]
        if i is None or j is None or i == j or max(i, j) >= len(boxes):
            continue
        a, b = boxes[i], boxes[j]
        same = (a[2] - a[0], a[3] - a[1]) == (b[2] - b[0], b[3] - b[1])
        if same and len(out) < MAX_MEASURES:
            out.append((a, b))
    return out


def _object_id(v: object) -> int | None:
    if isinstance(v, str):
        m = re.fullmatch(r"\s*#?(\d{1,3})\s*", v)
        v = int(m.group(1)) if m else None
    return v if isinstance(v, int) and v >= 0 else None


def parse_measures(text: str, present: set[int]) -> tuple[str, dict[Feature, float]]:
    """The goal sentence and the credit the reply's valid gap and count measures give."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m is None:
        return "", {}
    try:
        doc = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "", {}
    if not isinstance(doc, dict):
        return "", {}
    goal = str(doc.get("goal", ""))[:200]  # pyright: ignore[reportUnknownArgumentType]
    raw = doc.get("measures", [])  # pyright: ignore[reportUnknownVariableType]
    credit: dict[Feature, float] = {}
    for item in raw if isinstance(raw, list) else []:  # pyright: ignore[reportUnknownVariableType]
        if len(credit) >= MAX_MEASURES or not isinstance(item, dict):
            continue
        gap = item.get("gap")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(gap, list) and len(gap) == 2:  # pyright: ignore[reportUnknownArgumentType]
            a, b = (_color(c) for c in gap)  # pyright: ignore[reportUnknownVariableType]
            if a is not None and b is not None and a != b and {a, b} <= present:
                credit["gap", min(a, b), max(a, b)] = LLM_WEIGHT
            continue
        c = _color(item.get("count"))  # pyright: ignore[reportUnknownMemberType]
        want = item.get("want")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if c is not None and c in present and want in ("zero", "less", "more"):
            credit["count", c, -1] = LLM_WEIGHT if want != "more" else -LLM_WEIGHT
    return goal, credit


def _color(v: object) -> int | None:
    if isinstance(v, str):
        m = re.fullmatch(r"\s*c?(\d{1,2})\s*", v, re.IGNORECASE)
        v = int(m.group(1)) if m else None
    return v if isinstance(v, int) and 0 <= v <= 15 else None


class AdvisedExplorer(GuidedExplorer):
    """See the module docstring."""

    def __init__(self, seed: int = 0, *, chat: VisionChat | None = None, **kwargs: int) -> None:
        super().__init__(seed, **kwargs)
        self._chat = chat
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="advice")
        self._pending: Future[str] | None = None
        self._asked_level = -1
        self._asked_at = 0
        self._moves = 0
        self._present: set[int] = set()
        self.llm_credit: dict[Feature, float] = {}
        self.matches: list[tuple[Box, Box]] = []
        self._boxes: list[Box] = []
        self._shape: tuple[int, int] | None = None
        self.history: list[dict[str, Any]] = []

    def choose(
        self, obs: Observation, exclude: frozenset[tuple[int, int, int]] = frozenset()
    ) -> Action:
        self._moves += 1
        self._collect()
        if self._chat is not None and self._pending is None and self._due(obs):
            self._ask(obs)
        return super().choose(obs, exclude)

    def close(self) -> None:
        """Stop the background thread without waiting for a question in flight."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _due(self, obs: Observation) -> bool:
        if obs.levels_completed != self._asked_level:
            return True
        return self._moves - self._asked_at >= REASK_MOVES

    def _ask(self, obs: Observation) -> None:
        assert self._chat is not None
        self._asked_level = obs.levels_completed
        self._asked_at = self._moves
        self._present = {int(v) for v in np.unique(obs.grid) if v >= 0}
        self._boxes = objects(obs.grid)
        self._shape = (int(obs.grid.shape[0]), int(obs.grid.shape[1]))
        messages = self._messages(obs)
        chat, seed = self._chat, len(self.history)
        self._pending = self._pool.submit(lambda: chat(messages, seed).text)
        self.history.append({"level": obs.levels_completed, "move": self._moves})

    def _collect(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        fut, self._pending = self._pending, None
        try:
            text = fut.result()
        except Exception as exc:  # a failed call costs only this question
            self.history[-1]["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return
        goal, credit = parse_measures(text, self._present)
        matches = parse_matches(text, self._boxes)
        self.history[-1].update(
            goal=goal, measures=[list(f) for f in credit], matches=[list(m) for m in matches]
        )
        if matches:
            self.matches = matches
            self.plan = []
        if credit:
            for f in self.llm_credit:
                self.credit.pop(f, None)
            self.llm_credit = credit
            self.credit.update(credit)
            self.plan = []

    def guiding(self) -> bool:
        return bool(self.credit or self.matches)

    def score(self, key: bytes) -> float:
        """The guided score plus ``LLM_WEIGHT`` times the cells each match pair still
        differs by, read from the state's own grid (the key holds it, clock lines blank)."""
        base = super().score(key)
        if not self.matches or self._shape is None:
            return base
        h, w = self._shape
        if len(key) != h * w + 2:
            return base
        grid = np.frombuffer(key[: h * w], dtype=np.int8).reshape(h, w)
        return base + LLM_WEIGHT * sum(mismatch(grid, a, b) for a, b in self.matches)

    def _messages(self, obs: Observation) -> list[VMessage]:
        rates = sorted(
            ((k, self.change_rate(k), self.tries[k]) for k in self.tries),
            key=lambda t: -t[2],
        )[:12]
        acts = "\n".join(
            f"action {k[0]}" + (f" on c{k[1]}" if k[1] >= 0 else "") + f": changed the screen "
            f"{r:.0%} of {n} tries"
            for k, r, n in rates
        )
        past = "\n".join(
            f"level {h['level']}: you said {h.get('goal', '?')!r} "
            f"(measures {h.get('measures', [])})"
            for h in self.history
            if "goal" in h
        )
        text = (
            f"Level {obs.levels_completed + 1}. Levels won so far: {obs.levels_completed}.\n"
            f"Colors on screen:\n{screen_notes(obs.grid)}\n"
            f"Objects:\n{object_notes(obs.grid, self._boxes) or '(none)'}\n"
            f"What actions did so far:\n{acts or '(nothing tried yet)'}\n"
            f"Your earlier answers:\n{past or '(none)'}"
        )
        content: list[dict[str, object]] = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url(png(obs.grid))}},
        ]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
