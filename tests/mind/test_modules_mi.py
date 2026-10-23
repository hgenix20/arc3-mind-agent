"""Nulls, metacognition and inhibition, against real memory and a scripted sandbox."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from arcagent.core.types import GRID_SIZE, Action, Grid, Observation, Phase
from arcagent.mind.memory import GameStore, WorkingMemory
from arcagent.mind.modules.base import (
    NullInhibition,
    NullMetacognition,
    NullTransfer,
    NullVoi,
)
from arcagent.mind.modules.inhibition import DisconfirmInhibition
from arcagent.mind.modules.metacognition import ConfidenceMetacognition
from arcagent.mind.perception import frame_hash
from arcagent.mind.predictive import Belief, PredictionRecord
from arcagent.mind.types import Program, Step


def grid(v: int = 0) -> Grid:
    g = np.zeros((GRID_SIZE, GRID_SIZE), np.int8)
    g[0, 0] = v
    return g


class FakeSandbox:
    def __init__(self) -> None:
        self.dropped: dict[str, str] = {}

    def load(self, pid: str, code: str) -> str | None:
        return None

    def unload(self, pid: str) -> None:
        return None

    def predict(self, pids: Sequence[str], grid: Grid, action: Action) -> dict[str, str | None]:
        return dict.fromkeys(pids)


def belief_with(**states: tuple[int, int, int]) -> Belief:
    """Each value is (hits, trials, misses_in_row)."""
    b = Belief(FakeSandbox())
    for pid, (hits, trials, misses) in states.items():
        b.add(Program(pid, "", False, "llm"), hits, trials)
        b.programs[pid].misses_in_row = misses
    return b


def mk_step(
    before: Grid,
    after: Grid,
    action: Action | None = None,
    phase: Phase = Phase.NOT_FINISHED,
    levels: tuple[int, int] = (0, 0),
) -> Step:
    action = action if action is not None else Action(1)
    return Step(0, before, action, after, levels[0], levels[1], phase)


def rec(pid: str, matched: bool) -> PredictionRecord:
    return PredictionRecord(pid, None, matched, 0.0)


def test_nulls_do_nothing() -> None:
    wm, store = WorkingMemory(), GameStore()
    belief = belief_with(a=(0, 10, 10))
    assert NullMetacognition().review(belief, 5) == []
    st = mk_step(grid(), grid())
    null_inh = NullInhibition()
    assert null_inh.record(st, [rec("a", False)], "a", wm, store) == []
    assert null_inh.vetoes(frame_hash(grid()), wm, store) == frozenset()
    assert wm.get(f"veto:{frame_hash(grid())}") is None
    obs = Observation(grid(), Phase.NOT_FINISHED, 0, 1, (1,))
    assert NullVoi().pick(belief, object(), obs, [Action(1)], frozenset()) == (None, {})  # pyright: ignore[reportArgumentType]  # sandbox unused by the null
    assert NullTransfer().carry(store, belief) == []
    assert NullTransfer().check(belief, 10_000) is False


def test_metacognition_drops_low_fit_after_min_trials() -> None:
    belief = belief_with(young=(0, 2, 2 - 2), old_bad=(1, 5, 0), old_good=(5, 5, 0))
    meta = ConfidenceMetacognition(min_trials=3, drop_fit=0.6, max_misses_in_row=3)
    assert meta.review(belief, 7) == [("old_bad", "low_fit")]


def test_metacognition_drops_after_misses_in_row() -> None:
    belief = belief_with(streak=(9, 10, 3), both=(0, 5, 4), fine=(9, 10, 2))
    meta = ConfidenceMetacognition(min_trials=3, drop_fit=0.6, max_misses_in_row=3)
    got = sorted(meta.review(belief, 7))
    assert got == [("both", "misses_in_row"), ("streak", "misses_in_row")]
    # misses_in_row applies even before min_trials
    early = belief_with(x=(0, 1, 3))
    assert meta.review(early, 1) == [("x", "misses_in_row")]


def test_inhibition_vetoes_noop_gameover_disconfirmed() -> None:
    wm, store = WorkingMemory(), GameStore()
    inh = DisconfirmInhibition()
    f0 = grid(0)
    fh = frame_hash(f0)

    noop = inh.record(mk_step(f0, grid(0), Action(1)), [], None, wm, store)
    assert noop == [(fh, (1, -1, -1), "noop")]

    over = inh.record(mk_step(f0, grid(2), Action(2), Phase.GAME_OVER), [], None, wm, store)
    assert over == [(fh, (2, -1, -1), "game_over")]

    dis = inh.record(mk_step(f0, grid(3), Action(3)), [rec("a", False)], "a", wm, store)
    assert dis == [(fh, (3, -1, -1), "disconfirmed")]

    # matched, or a different active pid, or no active pid: no veto
    assert inh.record(mk_step(f0, grid(3), Action(4)), [rec("a", True)], "a", wm, store) == []
    assert inh.record(mk_step(f0, grid(3), Action(4)), [rec("b", False)], "a", wm, store) == []
    assert inh.record(mk_step(f0, grid(3), Action(4)), [rec("a", False)], None, wm, store) == []

    # precedence: game_over beats noop beats disconfirmed
    both = inh.record(
        mk_step(f0, grid(0), Action(5), Phase.GAME_OVER), [rec("a", False)], "a", wm, store
    )
    assert both[0][2] == "game_over"
    nd = inh.record(mk_step(f0, grid(0), Action(7)), [rec("a", False)], "a", wm, store)
    assert nd[0][2] == "noop"

    # level change is not a noop
    assert (
        inh.record(mk_step(f0, grid(0), Action(6, 1, 1), levels=(0, 1)), [], None, wm, store) == []
    )

    # RESET is never vetoed
    assert inh.record(mk_step(f0, grid(0), Action(0), Phase.GAME_OVER), [], None, wm, store) == []

    assert inh.vetoes(fh, wm, store) == frozenset(
        {(1, -1, -1), (2, -1, -1), (3, -1, -1), (5, -1, -1), (7, -1, -1)}
    )
    assert inh.vetoes("other", wm, store) == frozenset()


def test_veto_expires_then_recalls_on_reseen_frame() -> None:
    wm, store = WorkingMemory(), GameStore()
    inh = DisconfirmInhibition(ttl=2)
    f0 = grid(0)
    fh = frame_hash(f0)
    inh.record(mk_step(f0, grid(0), Action(1)), [], None, wm, store)
    wm.tick(store)
    assert wm.get(f"veto:{fh}") is not None
    wm.tick(store)
    assert wm.get(f"veto:{fh}") is None
    assert f"veto:{fh}" in store.cold

    # a new veto on the same frame merges the cold set instead of overwriting it
    inh.record(mk_step(f0, grid(0), Action(2)), [], None, wm, store)
    assert inh.vetoes(fh, wm, store) == frozenset({(1, -1, -1), (2, -1, -1)})

    # expire again, then reseeing the frame restores it with doubled ttl
    for _ in range(4):
        wm.tick(store)
    assert wm.get(f"veto:{fh}") is None
    assert inh.vetoes(fh, wm, store) == frozenset({(1, -1, -1), (2, -1, -1)})
    assert f"veto:{fh}" not in store.cold
