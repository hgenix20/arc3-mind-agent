"""The four swappable module Protocols and their do-nothing versions.

Each module has a null so the ablation can switch it off without touching the
agent. Modules advise and report; the agent applies the result and logs it.

Deviation from the original plan: ``Inhibition.record`` takes the ``GameStore``
as well, so it can merge a veto set that has already expired to cold instead of
overwriting it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from arcagent.core.types import Action, Observation
from arcagent.mind.memory import GameStore, WorkingMemory
from arcagent.mind.predictive import Belief, PredictionRecord
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.types import ActionKey, Step


class Metacognition(Protocol):
    def review(self, belief: Belief, step_index: int) -> list[tuple[str, str]]:
        """Return (pid, reason) for programs to drop. Never mutates ``belief``."""
        ...


class Inhibition(Protocol):
    def record(
        self,
        step: Step,
        records: Sequence[PredictionRecord],
        active_pid: str | None,
        wm: WorkingMemory,
        store: GameStore,
    ) -> list[tuple[str, ActionKey, str]]:
        """Veto (frame hash, action, reason) triples recorded for this step."""
        ...

    def vetoes(self, frame: str, wm: WorkingMemory, store: GameStore) -> frozenset[ActionKey]:
        """Actions vetoed in ``frame``, recalling a cold set if needed."""
        ...


class Voi(Protocol):
    def pick(
        self,
        belief: Belief,
        sandbox: ProgramSandbox,
        obs: Observation,
        candidates: Sequence[Action],
        visited: frozenset[str],
    ) -> tuple[Action | None, dict[ActionKey, float]]:
        """The most informative candidate (None for no preference) and every score."""
        ...


class Transfer(Protocol):
    def carry(self, store: GameStore, belief: Belief) -> list[str]:
        """Load carried programs into ``belief``; return their pids."""
        ...

    def check(self, belief: Belief, steps_on_level: int) -> bool:
        """True means the carried programs failed on this level."""
        ...


class NullMetacognition:
    def review(self, belief: Belief, step_index: int) -> list[tuple[str, str]]:
        return []


class NullInhibition:
    def record(
        self,
        step: Step,
        records: Sequence[PredictionRecord],
        active_pid: str | None,
        wm: WorkingMemory,
        store: GameStore,
    ) -> list[tuple[str, ActionKey, str]]:
        return []

    def vetoes(self, frame: str, wm: WorkingMemory, store: GameStore) -> frozenset[ActionKey]:
        return frozenset()


class NullVoi:
    def pick(
        self,
        belief: Belief,
        sandbox: ProgramSandbox,
        obs: Observation,
        candidates: Sequence[Action],
        visited: frozenset[str],
    ) -> tuple[Action | None, dict[ActionKey, float]]:
        return None, {}


class NullTransfer:
    def carry(self, store: GameStore, belief: Belief) -> list[str]:
        return []

    def check(self, belief: Belief, steps_on_level: int) -> bool:
        return False
