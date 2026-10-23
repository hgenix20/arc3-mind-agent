"""Salience scoring and the wake decision; both are fixed rules with no model."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import GRID_SIZE, Action, Phase
from arcagent.mind.drive import DriveState
from arcagent.mind.predictive import PredictionRecord
from arcagent.mind.salience import SalienceConfig, score, should_wake
from arcagent.mind.types import Event, EventKind, Step

CFG = SalienceConfig()


def _step(levels_after: int = 0, phase: Phase = Phase.NOT_FINISHED) -> Step:
    g = np.zeros((GRID_SIZE, GRID_SIZE), np.int8)
    return Step(7, g, Action(1), g, 0, levels_after, phase)


def _rec(pid: str, matched: bool) -> PredictionRecord:
    return PredictionRecord(pid, None, matched, 0.5)


def _drive(now: float = 0.0, calls_left: int = 3) -> DriveState:
    return DriveState(0.0, 100.0, 1000.0, now, 0, calls_left, 0)


def _ev(kind: EventKind, magnitude: float = 1.0) -> Event:
    return Event(kind, 7, magnitude, {})


def _wake(
    events: list[Event],
    *,
    seen: int = 50,
    since: int = 10,
    made: int = 1,
    drive: DriveState | None = None,
) -> str | None:
    return should_wake(
        events,
        steps_seen=seen,
        steps_since_call=since,
        calls_made=made,
        drive=drive or _drive(),
        cfg=CFG,
    )


def test_score_events_per_kind() -> None:
    assert score(_step(), [], False, False) == []
    assert score(_step(), [_rec("a", True)], False, False) == []

    up = score(_step(levels_after=1), [], False, False)
    assert [(e.kind, e.magnitude, e.step) for e in up] == [("level_up", 1.0, 7)]

    over = score(_step(phase=Phase.GAME_OVER), [], False, False)
    assert [(e.kind, e.magnitude) for e in over] == [("game_over", 1.0)]

    recs = [_rec("a", False), _rec("b", True), _rec("c", False), _rec("d", False)]
    err = score(_step(), recs, False, False)
    assert [e.kind for e in err] == ["prediction_error"]
    assert err[0].magnitude == 0.75
    assert err[0].detail == {"misses": 3, "live": 4}

    both = score(_step(levels_after=1, phase=Phase.GAME_OVER), recs, True, True)
    assert [e.kind for e in both] == [
        "level_up",
        "game_over",
        "prediction_error",
        "belief_empty",
        "transfer_failed",
    ]
    assert [e.magnitude for e in both if e.kind in ("belief_empty", "transfer_failed")] == [
        1.0,
        1.0,
    ]


def test_first_wake_after_explore_steps() -> None:
    assert _wake([], seen=19, since=0, made=0) is None
    assert _wake([], seen=20, since=0, made=0) == "new_game"
    # The first call waits for the explore steps whatever the events say.
    assert _wake([_ev("belief_empty")], seen=5, since=0, made=0) is None


def test_no_wake_without_calls_left_or_new_steps() -> None:
    ev = [_ev("belief_empty")]
    empty = _drive(calls_left=0)
    assert _wake(ev, drive=empty) is None
    assert _wake([], made=0, drive=empty) is None
    assert _wake(ev, since=4) is None
    assert _wake(ev, since=5) == "belief_empty"
    assert _wake([_ev("transfer_failed")], since=5) == "transfer_failed"
    assert _wake([_ev("transfer_failed"), _ev("belief_empty")], since=5) == "belief_empty"


def test_fatigue_raises_prediction_error_threshold() -> None:
    ev = [_ev("prediction_error", 0.7)]
    assert _wake(ev, drive=_drive(now=0.0)) == "prediction_error"
    assert _wake(ev, drive=_drive(now=100.0)) is None
    assert _wake([_ev("prediction_error", 0.3)], drive=_drive(now=0.0)) is None
