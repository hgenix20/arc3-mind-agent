"""Program sandbox: every test drives a real worker subprocess."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import numpy as np
import pytest

from arcagent.core.types import GRID_SIZE, Action, Grid, Phase
from arcagent.mind.perception import frame_hash
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.types import Step

GOOD = "def step(grid, action, x=-1, y=-1):\n    g=[r[:] for r in grid]; g[0][0]=action; return g\n"
HANG = "def step(grid, action, x=-1, y=-1):\n    while True: pass\n"
JUNK = "def step(grid, action, x=-1, y=-1):\n    return None\n"
EVIL = "import os\ndef step(grid, action, x=-1, y=-1):\n    return os.listdir('.')\n"
IDENTITY = "def step(grid, action, x=-1, y=-1):\n    return [r[:] for r in grid]\n"
SHORT = "def step(grid, action, x=-1, y=-1):\n    return [r[:] for r in grid[:63]]\n"
BIG_VALUE = (
    "def step(grid, action, x=-1, y=-1):\n    g=[r[:] for r in grid]; g[5][5]=99; return g\n"
)
OPENER = "def step(grid, action, x=-1, y=-1):\n    open('x.txt', 'w')\n    return grid\n"
ALLOWED = (
    "import math, itertools, collections, functools, heapq, copy, re\n"
    "def step(grid, action, x=-1, y=-1):\n    return copy.deepcopy(grid)\n"
)

# A 1-D walker: the cell set to 1 in row 0 is the position. 1 moves right, 2 left.
WALKER = """
def _pos(grid):
    return grid[0].index(1)

def step(grid, action, x=-1, y=-1):
    g = [r[:] for r in grid]
    p = _pos(g)
    q = min(p + 1, 63) if action == 1 else max(p - 1, 0)
    g[0][p] = 0
    g[0][q] = 1
    return g

def goal(grid):
    return _pos(grid) == 3
"""

# A 4-digit base-16 counter with an unreachable goal: 65536 states, so a search
# can only stop on its time limit within the test's budget.
COUNTER = """
def step(grid, action, x=-1, y=-1):
    g = [r[:] for r in grid]
    for i in range(4):
        if g[0][i] < 15:
            g[0][i] += 1
            break
        g[0][i] = 0
    return g

def goal(grid):
    return False
"""


def zeros() -> Grid:
    return np.zeros((GRID_SIZE, GRID_SIZE), np.int8)


def walker_start() -> Grid:
    g = zeros()
    g[0, 0] = 1
    return g


@pytest.fixture
def sb() -> Iterator[ProgramSandbox]:
    with ProgramSandbox(timeout_s=1.0, plan_timeout_s=10.0) as s:
        yield s


def test_predict_matches_frame_hash(sb: ProgramSandbox) -> None:
    assert sb.load("good", GOOD) is None
    expected = zeros()
    expected[0, 0] = 3
    assert sb.predict(["good"], zeros(), Action(3)) == {"good": frame_hash(expected)}
    got = sb.predict_grid("good", zeros(), Action(3))
    assert got is not None and np.array_equal(got, expected)


def test_worker_hash_equals_frame_hash_with_padding_and_max_color(sb: ProgramSandbox) -> None:
    g = zeros()
    g[0, :] = -1
    g[10, 10] = 15
    g[63, 63] = 7
    assert sb.load("id", IDENTITY) is None
    assert sb.predict(["id"], g, Action(1))["id"] == frame_hash(g)


def test_predict_many_keys_by_program_and_action(sb: ProgramSandbox) -> None:
    sb.load("good", GOOD)
    sb.load("id", IDENTITY)
    acts = [Action(1), Action(6, 4, 5)]
    out = sb.predict_many(["good", "id"], zeros(), acts)
    assert set(out) == {(p, a.key()) for p in ("good", "id") for a in acts}
    assert out[("id", (1, -1, -1))] == frame_hash(zeros())
    six = zeros()
    six[0, 0] = 6
    assert out[("good", (6, 4, 5))] == frame_hash(six)


def test_hang_times_out_restarts_and_keeps_others(sb: ProgramSandbox) -> None:
    sb.load("good", GOOD)
    sb.load("hang", HANG)
    out = sb.predict(["good", "hang"], zeros(), Action(1))
    assert out["hang"] is None and out["good"] is not None
    assert sb.restarts >= 1
    assert sb.dropped == {"hang": "timeout"}
    assert sb.predict(["good"], zeros(), Action(2))["good"] is not None  # reloaded after restart
    assert sb.predict(["hang"], zeros(), Action(2)) == {"hang": None}  # stays unloaded


def test_lone_hang_restarts_once(sb: ProgramSandbox) -> None:
    sb.load("good", GOOD)
    sb.load("hang", HANG)
    assert sb.predict(["hang"], zeros(), Action(1)) == {"hang": None}
    assert sb.restarts == 1 and sb.dropped == {"hang": "timeout"}
    assert sb.predict(["good"], zeros(), Action(1))["good"] is not None


def test_hanging_load_returns_timeout_and_keeps_others(sb: ProgramSandbox) -> None:
    sb.load("good", GOOD)
    assert sb.load("slow", "while True:\n    pass\n") == "timeout"
    assert sb.restarts == 1
    assert sb.predict(["good"], zeros(), Action(1))["good"] is not None


def test_junk_and_wrong_shape_are_misses(sb: ProgramSandbox) -> None:
    for pid, code in (("junk", JUNK), ("short", SHORT), ("big", BIG_VALUE)):
        assert sb.load(pid, code) is None
    out = sb.predict(["junk", "short", "big"], zeros(), Action(1))
    assert out == {"junk": None, "short": None, "big": None}
    assert sb.predict_grid("short", zeros(), Action(1)) is None
    assert sb.restarts == 0


def test_raising_and_non_int_cells_are_misses(sb: ProgramSandbox) -> None:
    sb.load("boom", "def step(grid, action, x=-1, y=-1):\n    raise SystemExit(3)\n")
    sb.load(
        "flt", "def step(grid, action, x=-1, y=-1):\n    return [[0.0]*64 for _ in range(64)]\n"
    )
    sb.load("bool", "def step(grid, action, x=-1, y=-1):\n    return [[False]*64]*64\n")
    assert sb.predict(["boom", "flt", "bool"], zeros(), Action(1)) == {
        "boom": None,
        "flt": None,
        "bool": None,
    }
    assert sb.restarts == 0


def test_blocked_imports_fail_to_load(sb: ProgramSandbox) -> None:
    assert sb.load("evil", EVIL) is not None
    assert sb.load("sub", "import subprocess\ndef step(g, a, x=-1, y=-1):\n    return g\n")
    assert sb.load("no_step", "x = 1\n") is not None
    assert sb.load("syntax", "def step(:\n") is not None
    assert sb.load("ok", ALLOWED) is None
    assert sb.predict(["ok"], zeros(), Action(1))["ok"] == frame_hash(zeros())


def test_file_builtins_are_blocked(sb: ProgramSandbox) -> None:
    sb.load("open", OPENER)
    assert sb.predict(["open"], zeros(), Action(1)) == {"open": None}


def test_print_does_not_break_the_channel(sb: ProgramSandbox) -> None:
    code = "print('hi')\ndef step(grid, action, x=-1, y=-1):\n    print(grid)\n    return grid\n"
    assert sb.load("chatty", code) is None
    assert sb.predict(["chatty"], zeros(), Action(1))["chatty"] == frame_hash(zeros())


def test_goal_reports_none_without_goal(sb: ProgramSandbox) -> None:
    sb.load("walker", WALKER)
    sb.load("good", GOOD)
    sb.load("bad_goal", GOOD + "def goal(grid):\n    raise ValueError('x')\n")
    at3 = zeros()
    at3[0, 3] = 1
    assert sb.goal(["walker", "good", "bad_goal"], at3) == {
        "walker": True,
        "good": None,
        "bad_goal": None,
    }
    assert sb.goal(["walker"], walker_start()) == {"walker": False}


def _step(i: int, before: Grid, action: Action, after: Grid) -> Step:
    return Step(i, before, action, after, 0, 0, Phase.NOT_FINISHED)


def test_score_sums_across_chunks(sb: ProgramSandbox) -> None:
    sb.load("good", GOOD)
    steps = []
    for i in range(450):
        after = zeros()
        after[0, 0] = 1 if i % 3 else 2  # GOOD predicts the action id, 1 here
        steps.append(_step(i, zeros(), Action(1), after))
    assert sb.score("good", steps) == (300, 450)


def test_score_counts_exact_hits(sb: ProgramSandbox) -> None:
    sb.load("good", GOOD)
    a1 = zeros()
    a1[0, 0] = 1
    a2 = zeros()
    a2[0, 0] = 2
    steps = [
        _step(0, zeros(), Action(1), a1),
        _step(1, zeros(), Action(2), a2),
        _step(2, zeros(), Action(3), a1),  # GOOD predicts 3 here, not 1
    ]
    assert sb.score("good", steps) == (2, 3)
    assert sb.score("good", []) == (0, 0)
    assert sb.score("missing", steps) == (0, 3)


def test_plan_reaches_goal_and_novel_targets(sb: ProgramSandbox) -> None:
    sb.load("walker", WALKER)
    acts = [Action(1), Action(2)]
    start = walker_start()
    res = sb.plan(
        "walker",
        start,
        acts,
        target="goal",
        visited=frozenset(),
        max_nodes=1000,
        time_limit_s=5.0,
    )
    assert res.stopped == "goal"
    assert res.actions == (Action(1), Action(1), Action(1))
    novel = sb.plan(
        "walker",
        start,
        acts,
        target="novel",
        visited=frozenset({frame_hash(start)}),
        max_nodes=1000,
        time_limit_s=5.0,
    )
    assert novel.stopped == "goal" and novel.actions is not None and len(novel.actions) == 1


def test_plan_without_goal_function(sb: ProgramSandbox) -> None:
    sb.load("good", GOOD)
    res = sb.plan(
        "good",
        zeros(),
        [Action(1)],
        target="goal",
        visited=frozenset(),
        max_nodes=10,
        time_limit_s=1.0,
    )
    assert res.actions is None and res.stopped == "no_goal"


def test_plan_respects_time_limit(sb: ProgramSandbox) -> None:
    sb.load("counter", COUNTER)
    res = sb.plan(
        "counter",
        zeros(),
        [Action(1)],
        target="goal",
        visited=frozenset(),
        max_nodes=1_000_000,
        time_limit_s=0.0,
    )
    assert res.actions is None and res.stopped == "time_limit"


def test_hanging_plan_is_dropped() -> None:
    with ProgramSandbox(timeout_s=1.0, plan_timeout_s=1.0) as short:
        short.load("hang", HANG + "def goal(grid):\n    return False\n")
        res = short.plan(
            "hang",
            zeros(),
            [Action(1)],
            target="goal",
            visited=frozenset(),
            max_nodes=10,
            time_limit_s=0.5,
        )
        assert res.actions is None and res.stopped == "timeout"
        assert short.dropped == {"hang": "timeout"} and short.restarts == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS exists on Linux only")
def test_memory_hog_is_a_miss_under_the_address_space_cap(sb: ProgramSandbox) -> None:
    hog = "def step(grid, action, x=-1, y=-1):\n    return [0] * (4 * 10**9)\n"
    sb.load("hog", hog)
    sb.load("good", GOOD)
    out = sb.predict(["hog", "good"], zeros(), Action(1))
    assert out["hog"] is None and out["good"] is not None
