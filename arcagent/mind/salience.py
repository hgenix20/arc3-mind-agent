"""Salience: fixed rules that score transitions and decide when to wake the executive.

Contract: pure functions, no I/O, no model. The wake decision is a function of
the events, counters and drive state only, so it can be replayed exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arcagent.core.types import Phase
from arcagent.mind.drive import DriveState, fatigue
from arcagent.mind.predictive import PredictionRecord
from arcagent.mind.types import Event, Step


@dataclass(frozen=True, slots=True)
class SalienceConfig:
    """``explore_steps``: steps seen before the first call. ``min_new_steps``:
    steps required between calls. The prediction-error threshold is
    ``base_threshold + fatigue_raise * fatigue``."""

    explore_steps: int = 20
    min_new_steps: int = 5
    base_threshold: float = 0.5
    fatigue_raise: float = 0.4


def score(
    step: Step,
    records: Sequence[PredictionRecord],
    belief_empty: bool,
    transfer_failed: bool,
) -> list[Event]:
    """Events for one transition, ordered level_up, game_over, prediction_error,
    belief_empty, transfer_failed.

    ``prediction_error`` magnitude is the share of live programs that missed,
    and is emitted only when there are records and at least one miss.
    """
    events: list[Event] = []
    if step.level_up:
        events.append(Event("level_up", step.index, 1.0, {}))
    if step.phase_after == Phase.GAME_OVER:
        events.append(Event("game_over", step.index, 1.0, {}))
    if records:
        misses = sum(1 for r in records if not r.matched)
        if misses:
            events.append(
                Event(
                    "prediction_error",
                    step.index,
                    misses / len(records),
                    {"misses": misses, "live": len(records)},
                )
            )
    if belief_empty:
        events.append(Event("belief_empty", step.index, 1.0, {}))
    if transfer_failed:
        events.append(Event("transfer_failed", step.index, 1.0, {}))
    return events


def should_wake(
    events: Sequence[Event],
    *,
    steps_seen: int,
    steps_since_call: int,
    calls_made: int,
    drive: DriveState,
    cfg: SalienceConfig,
) -> str | None:
    """The reason to call the executive now, or None.

    Order: no calls left -> None. First call (``calls_made == 0``): "new_game"
    once ``steps_seen >= explore_steps``, else None; events are ignored then,
    so belief_empty and prediction_error cannot trigger the first call. Later
    calls need ``steps_since_call >= min_new_steps``, then the first match of
    belief_empty, transfer_failed, prediction_error (magnitude at or above the
    fatigue-raised threshold).
    """
    if drive.synth_calls_left <= 0:
        return None
    if calls_made == 0:
        return "new_game" if steps_seen >= cfg.explore_steps else None
    if steps_since_call < cfg.min_new_steps:
        return None
    return _event_reason(events, drive, cfg)


def _event_reason(events: Sequence[Event], drive: DriveState, cfg: SalienceConfig) -> str | None:
    kinds = {e.kind for e in events}
    if "belief_empty" in kinds:
        return "belief_empty"
    if "transfer_failed" in kinds:
        return "transfer_failed"
    threshold = cfg.base_threshold + cfg.fatigue_raise * fatigue(drive)
    if any(e.kind == "prediction_error" and e.magnitude >= threshold for e in events):
        return "prediction_error"
    return None
