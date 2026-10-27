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
  boxes; how often each action kind changed the screen; and for each earlier level,
  the measurements the reply named and whether that level was then won.
- The reply is a JSON object ``{"goal": str, "measures": [...]}``. A measure is
  ``{"gap": [a, b]}`` (colors a and b should touch: smaller is better) or
  ``{"count": c, "want": "zero" | "less" | "more"}``. Unknown colors, malformed entries
  and colors not on screen are dropped. A reply with no valid measure changes nothing.
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

REASK_MOVES = 1500  # moves on a level without a win before asking again
LLM_WEIGHT = 2.0  # credit per measure the model names
MAX_MEASURES = 4

SYSTEM_PROMPT = """You are helping an agent play an unknown grid game. Each level is won
by reaching some configuration of the screen. Colors are numbered 0-15 and written cN.

Look at the screen and the notes, then say what this level most likely wants, as
measurements the agent can track:
- {"gap": [a, b]}: something of color a should touch or reach something of color b.
- {"count": c, "want": "zero" | "less" | "more"}: the number of cells of color c
  should drop to zero, go down, or go up.

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


def parse_measures(text: str, present: set[int]) -> tuple[str, dict[Feature, float]]:
    """The goal sentence and the credit the reply's valid measures give."""
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
        self.history[-1].update(goal=goal, measures=[list(f) for f in credit])
        if credit:
            for f in self.llm_credit:
                self.credit.pop(f, None)
            self.llm_credit = credit
            self.credit.update(credit)
            self.plan = []

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
