"""Belief arithmetic runs against a small fake sandbox; one test drives a real worker."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from arcagent.core.types import GRID_SIZE, Action, Grid, Phase
from arcagent.mind.perception import frame_hash
from arcagent.mind.predictive import Belief, ProgramState
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.types import Program, Step


def zeros() -> Grid:
    return np.zeros((GRID_SIZE, GRID_SIZE), np.int8)


def marked(v: int) -> Grid:
    g = zeros()
    g[0, 0] = v
    return g


def step(index: int, after: Grid) -> Step:
    return Step(index, zeros(), Action(1), after, 0, 0, Phase.NOT_FINISHED)


def prog(pid: str, code: str = "") -> Program:
    return Program(pid, code, False, "llm")


class FakeSandbox:
    """Programs are scripted: ``outputs[pid]`` is the hash to predict, None for a miss."""

    def __init__(self) -> None:
        self.outputs: dict[str, str | None] = {}
        self.load_errors: dict[str, str] = {}
        self.loaded: set[str] = set()
        self.dropped: dict[str, str] = {}

    def load(self, pid: str, code: str) -> str | None:
        if pid in self.load_errors:
            return self.load_errors[pid]
        self.loaded.add(pid)
        return None

    def unload(self, pid: str) -> None:
        self.loaded.discard(pid)

    def predict(self, pids: Sequence[str], grid: Grid, action: Action) -> dict[str, str | None]:
        return {p: self.outputs.get(p) for p in pids}


def make(*pids: str) -> tuple[Belief, FakeSandbox]:
    sb = FakeSandbox()
    b = Belief(sb)
    for p in pids:
        b.add(prog(p))
    return b, sb


def test_observe_updates_fit_and_records() -> None:
    b, sb = make("right", "wrong")
    sb.outputs = {"right": frame_hash(marked(1)), "wrong": frame_hash(marked(2))}
    recs = {r.pid: r for r in b.observe(step(0, marked(1)))}
    assert recs["right"].matched and recs["right"].fit == 1.0
    assert not recs["wrong"].matched and recs["wrong"].fit == 0.0
    assert recs["wrong"].pred == frame_hash(marked(2))
    assert b.programs["right"].hits == 1 and b.programs["right"].trials == 1
    assert b.programs["wrong"].hits == 0 and b.programs["wrong"].trials == 1
    assert b.programs["wrong"].fit == 0.0
    assert ProgramState(prog("x"), 0, 0, 0, None, True).fit == 0.0


def test_none_prediction_is_a_miss() -> None:
    b, sb = make("p")
    sb.outputs = {"p": None}
    (r,) = b.observe(step(0, marked(1)))
    assert r.pred is None and not r.matched
    assert b.programs["p"].trials == 1 and b.programs["p"].hits == 0


def test_first_miss_and_misses_in_row_tracked() -> None:
    b, sb = make("p")
    st = b.programs["p"]
    assert st.last_ok and st.first_miss_step is None
    sb.outputs = {"p": frame_hash(marked(1))}
    b.observe(step(3, marked(1)))
    assert st.first_miss_step is None and st.last_ok
    b.observe(step(4, marked(9)))
    b.observe(step(5, marked(9)))
    assert st.first_miss_step == 4 and st.misses_in_row == 2 and not st.last_ok
    b.observe(step(6, marked(1)))
    assert st.misses_in_row == 0 and st.last_ok and st.first_miss_step == 4
    b.observe(step(7, marked(9)))
    assert st.first_miss_step == 4 and st.misses_in_row == 1


def test_best_respects_thresholds_and_ties() -> None:
    b, _ = make()
    for pid, hits, trials in (("b", 3, 4), ("a", 3, 4), ("c", 6, 8), ("d", 1, 1), ("e", 0, 5)):
        b.add(prog(pid), hits=hits, trials=trials)
    best = b.best(0.5, 2)
    assert best is not None and best.program.id == "c"  # 0.75 ties a and b, more trials wins
    b.remove("c")
    best = b.best(0.5, 2)
    assert best is not None and best.program.id == "a"  # tie on fit and trials: lowest id
    best = b.best(0.9, 1)
    assert best is not None and best.program.id == "d"
    assert b.best(0.9, 2) is None
    assert b.best(0.0, 100) is None


def test_entropy_normalized() -> None:
    b, _ = make()
    assert b.entropy() == 0.0
    b.add(prog("a"), hits=1, trials=2)
    assert b.entropy() == 0.0
    b.add(prog("b"), hits=1, trials=2)
    assert math.isclose(b.entropy(), 1.0)
    b.add(prog("c"), hits=0, trials=0)  # no trials: excluded
    assert math.isclose(b.entropy(), 1.0)
    b.add(prog("d"), hits=1, trials=1)
    p = [0.5, 0.5, 1.0]
    expected = -sum(x / 2 * math.log(x / 2) for x in p) / math.log(3)
    assert math.isclose(b.entropy(), expected)


def test_entropy_zero_when_all_fits_zero() -> None:
    b, _ = make()
    b.add(prog("a"), hits=0, trials=3)
    b.add(prog("b"), hits=0, trials=3)
    assert b.entropy() == 0.0


def test_add_returns_load_error_and_skips_program() -> None:
    b, sb = make()
    sb.load_errors["bad"] = "SyntaxError"
    assert b.add(prog("bad")) == "SyntaxError"
    assert "bad" not in b.programs
    assert b.add(prog("ok")) is None and "ok" in b.programs


def test_duplicate_add_replaces_state_and_reloads() -> None:
    b, sb = make("p")
    sb.outputs = {"p": frame_hash(marked(9))}
    b.observe(step(0, marked(1)))
    assert b.programs["p"].first_miss_step == 0
    sb.load_errors["p"] = "boom"
    assert b.add(prog("p")) == "boom"
    assert "p" not in b.programs  # the failed reload removes the stale state
    del sb.load_errors["p"]
    b.add(prog("p"))
    assert b.programs["p"].trials == 0 and b.programs["p"].first_miss_step is None


def test_remove_unloads_and_ignores_unknown() -> None:
    b, sb = make("p")
    b.remove("p")
    b.remove("nope")
    assert "p" not in b.programs and "p" not in sb.loaded


def test_sandbox_drop_removes_program_and_is_drained_once() -> None:
    b, sb = make("keep", "gone")
    sb.outputs = {"keep": frame_hash(marked(1)), "gone": None}
    sb.dropped = {"gone": "timeout", "stranger": "crash"}
    recs = {r.pid: r for r in b.observe(step(0, marked(1)))}
    assert not recs["gone"].matched and "gone" not in b.programs and "keep" in b.programs
    assert b.drained_drops() == [("gone", "timeout")]
    assert b.drained_drops() == []
    b.observe(step(1, marked(1)))
    assert b.drained_drops() == []


def test_real_sandbox_hang_is_dropped_from_belief() -> None:
    good = "def step(grid, action, x=-1, y=-1):\n    g=[r[:] for r in grid]; g[0][0]=1; return g\n"
    hang = "def step(grid, action, x=-1, y=-1):\n    while True: pass\n"
    with ProgramSandbox(timeout_s=1.0) as sb:
        b = Belief(sb)
        assert b.add(prog("good", good)) is None
        assert b.add(prog("hang", hang)) is None
        recs = {r.pid: r for r in b.observe(step(0, marked(1)))}
        assert recs["good"].matched and not recs["hang"].matched
        assert set(b.programs) == {"good"}
        assert b.drained_drops() == [("hang", "timeout")]
        assert sb.predict(["good"], zeros(), Action(1))["good"] == frame_hash(marked(1))
