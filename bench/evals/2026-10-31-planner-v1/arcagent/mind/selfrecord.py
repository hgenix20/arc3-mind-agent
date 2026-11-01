"""Self and focus records: who "I" is, and what the agent is doing right now.

Spec 4.5a (self record) and 4.5b (focus record). The agent that owns the loop writes both
once per move and reads them in two places: the planner briefing (4.9a, through ``line()``)
and the move log (4.13a, through ``as_dict()``). Kameron, 2026-10-30: the labels "should
reach both since it might be talking to itself".

``SelfRecord``: "I" is always the agent, and the controlled thing, when one is known, is
an extension of it (``my_character``). ``click`` means exactly that action 6 is available.

``FocusRecord``: the level, the active goal and its status, the target thing, the mode
(the branch that chose the last move), the budget left and the signal values of the last
move. It is the first thing the planner briefing reads, so ``line()`` is one line under
``LINE_CHARS`` characters whatever the field values are.

Both records are frozen values with no side effects. Neither logs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from arcagent.core.types import COORD_ACTION_ID

AGENT = "agent"  # the one referent of "I" (spec 4.5a, rule 1)
LINE_CHARS = 200  # a focus line is shorter than this
GOAL_CHARS = 36  # a goal id in the focus line; a reach id with two full sigs is up to 35
TEXT_CHARS = 16  # a kind or sig in the focus line; a kind name is at most 15
WORD_CHARS = 10  # a status, mode or verdict in the focus line
NONE = "none"  # printed for a missing goal, target or character
UNKNOWN = "unknown"  # printed for a budget the reader has not measured yet

__all__ = [
    "AGENT",
    "GOAL_CHARS",
    "LINE_CHARS",
    "TEXT_CHARS",
    "WORD_CHARS",
    "FocusRecord",
    "SelfRecord",
]


def _cut(text: str, chars: int) -> str:
    """``text`` cut to ``chars`` characters; a cut ends in ``..`` as the fact lines do."""
    return text if len(text) <= chars else text[: chars - 2] + ".."


@dataclass(frozen=True, slots=True, kw_only=True)
class SelfRecord:
    """Who "I" is at this move (spec 4.5a).

    ``i`` is always ``AGENT``. ``click`` is True exactly when action 6 is among the
    available actions. ``my_character`` is the controlled thing's kind when one is
    identified, else None.
    """

    i: str = AGENT
    click: bool
    my_character: str | None

    @classmethod
    def from_state(cls, available: Sequence[int], lone_mover: str | None) -> SelfRecord:
        """Build the record from the available action ids and the one controlled kind."""
        return cls(click=COORD_ACTION_ID in available, my_character=lone_mover)

    def line(self) -> str:
        """One sentence for the briefing: ``I am the agent. Click: yes. My character: k..``."""
        click = "yes" if self.click else "no"
        return f"I am the {self.i}. Click: {click}. My character: {self.my_character or NONE}."


@dataclass(frozen=True, slots=True)
class FocusRecord:
    """What the agent is doing right now (spec 4.5b), rewritten after each move.

    The fields are the agent's signals as they stand: ``goal_id`` and ``goal_status`` from
    the goal book (None when no goal is active), ``target`` the goal's target kind or sig,
    ``mode`` the branch that chose the last move, ``budget_left`` from
    ``BudgetReader.remaining`` (None while unknown), ``verdict`` the progress verdict of
    the last move, ``fallbacks_in_row`` the fallback moves in a row, ``segment_left`` the
    steps left in the current segment, and the tokens and seconds the last move spent.
    """

    level: int
    goal_id: str | None
    goal_status: str | None
    target: str | None
    mode: str
    budget_left: int | None
    verdict: str
    fallbacks_in_row: int
    segment_left: int
    tokens_this_move: int
    seconds_this_move: float

    def line(self) -> str:
        """One line for the briefing, under ``LINE_CHARS`` characters.

        Text fields are cut to ``GOAL_CHARS``, ``TEXT_CHARS`` and ``WORD_CHARS``, and the
        whole line is cut as a last resort, so the bound holds for any values.
        """
        goal = NONE if self.goal_id is None else _cut(self.goal_id, GOAL_CHARS)
        if self.goal_status is not None:
            goal = f"{goal} ({_cut(self.goal_status, WORD_CHARS)})"
        target = NONE if self.target is None else _cut(self.target, TEXT_CHARS)
        budget = UNKNOWN if self.budget_left is None else str(self.budget_left)
        text = (
            f"L{self.level}; goal {goal}; target {target}; mode {_cut(self.mode, WORD_CHARS)}; "
            f"budget {budget}; verdict {_cut(self.verdict, WORD_CHARS)}; "
            f"fallbacks {self.fallbacks_in_row}; segment {self.segment_left}; "
            f"tokens {self.tokens_this_move}; {self.seconds_this_move:.1f}s"
        )
        return _cut(text, LINE_CHARS - 1)

    def as_dict(self) -> dict[str, object]:
        """Every field by name, for the move log."""
        return asdict(self)
