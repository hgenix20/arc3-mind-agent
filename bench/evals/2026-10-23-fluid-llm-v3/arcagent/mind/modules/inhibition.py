"""Disconfirm inhibition: remember (frame, action) pairs that went badly.

A step is vetoed for the first applicable reason in this order: ``game_over``,
``noop`` (frame unchanged and no level change), ``disconfirmed`` (the active
program's prediction missed). RESET (action id 0) is never vetoed. Veto sets
live in working memory under ``veto:<frame hash>`` with a ttl, so they expire
to cold and return, with doubled ttl, when the frame is seen again.
"""

from __future__ import annotations

from collections.abc import Sequence

from arcagent.core.types import Phase
from arcagent.mind.memory import DEFAULT_TTL, GameStore, WorkingMemory
from arcagent.mind.perception import frame_hash
from arcagent.mind.predictive import PredictionRecord
from arcagent.mind.types import ActionKey, Step

RESET_ACTION_ID = 0


def _key(frame: str) -> str:
    return f"veto:{frame}"


def _as_set(value: object | None) -> frozenset[ActionKey]:
    if not isinstance(value, (set, frozenset)):
        return frozenset()
    return frozenset(v for v in value if isinstance(v, tuple))  # pyright: ignore[reportUnknownVariableType]  # set contents are only ever ActionKey tuples written here


class DisconfirmInhibition:
    def __init__(self, ttl: int = DEFAULT_TTL) -> None:
        self.ttl = ttl

    def record(
        self,
        step: Step,
        records: Sequence[PredictionRecord],
        active_pid: str | None,
        wm: WorkingMemory,
        store: GameStore,
    ) -> list[tuple[str, ActionKey, str]]:
        if step.action.id == RESET_ACTION_ID:
            return []
        before = frame_hash(step.before)
        reason = self._reason(step, before, records, active_pid)
        if reason is None:
            return []
        action = step.action.key()
        key = _key(before)
        # recall, not get: a set that expired to cold must be merged, not overwritten.
        existing = _as_set(wm.recall(key, store))
        wm.put(key, set(existing | {action}), self.ttl)
        return [(before, action, reason)]

    @staticmethod
    def _reason(
        step: Step, before: str, records: Sequence[PredictionRecord], active_pid: str | None
    ) -> str | None:
        if step.phase_after == Phase.GAME_OVER:
            return "game_over"
        if frame_hash(step.after) == before and not step.level_up:
            return "noop"
        if active_pid is not None and any(r.pid == active_pid and not r.matched for r in records):
            return "disconfirmed"
        return None

    def vetoes(self, frame: str, wm: WorkingMemory, store: GameStore) -> frozenset[ActionKey]:
        return _as_set(wm.recall(_key(frame), store))
