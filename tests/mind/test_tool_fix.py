"""Tool fixes: adopted only when they solve the failing case and agree with past cases."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast

import pytest

from arcagent.mind.route import Dist
from arcagent.mind.tool_fix import ToolFailure, extract_function, try_fix
from arcagent.mind.toolbox import Toolbox, ToolWorker
from arcagent.mind.vchat import scripted

DIST = cast("Dist", math.dist)
TARGETS = [(6, 0), (3, 0)]

# Visits targets in the given order: total 6 from (0,0) via (6,0),(3,0) is 9; reversed 6.
FIXED = """```python
def route(start, targets, dist, *, exact_limit=ROUTE_EXACT):
    order = sorted(range(len(targets)), key=lambda i: dist(start, targets[i]))
    total, cur = 0, start
    for i in order:
        total += dist(cur, targets[i])
        cur = targets[i]
    return order, total
```"""
BAD = """```python
def route(start, targets, dist, *, exact_limit=ROUTE_EXACT):
    return list(range(len(targets))), 100
```"""


def failure() -> ToolFailure:
    def check(got: Any) -> bool:
        return got is not None and got[1] <= 6

    return ToolFailure("route", ((0, 0), TARGETS, DIST), {}, "a route of length 6", "9", check)


@pytest.fixture
def worker() -> Iterator[ToolWorker]:
    w = ToolWorker()
    yield w
    w.close()


def test_fix_adopted_when_it_solves_and_agrees(worker: ToolWorker) -> None:
    box = Toolbox()
    box.add_case("route", ((0, 0), [(2, 0)], DIST), {}, ([0], 2.0))
    res = try_fix(scripted([FIXED]), failure(), worker, game="g1", seed=0, box=box)
    assert res.adopted == 1 and res.reason == "adopted"
    v = box.current("route")
    assert v is not None and v.game == "g1" and "sorted" in v.source


def test_fix_rejected_on_regression(worker: ToolWorker) -> None:
    box = Toolbox()
    box.add_case("route", ((0, 0), [(2, 0), (1, 0)], DIST), {}, ([1, 0], 1.0))
    res = try_fix(scripted([FIXED]), failure(), worker, game="g1", seed=0, box=box)
    assert res.adopted is None and res.reason == "regression case 0 disagrees"
    assert box.current("route") is None


def test_fix_rejected_when_it_does_not_solve(worker: ToolWorker) -> None:
    res = try_fix(scripted([BAD]), failure(), worker, game="g1", seed=0, box=Toolbox())
    assert res == res.__class__(None, "does not solve the failing case")


def test_fix_rejected_without_code(worker: ToolWorker) -> None:
    res = try_fix(scripted(["no idea"]), failure(), worker, game="g1", seed=0, box=Toolbox())
    assert res.adopted is None and res.reason == "no function in reply"


def test_extract_function_needs_the_name() -> None:
    assert extract_function("```python\ndef other():\n    pass\n```", "route") is None
    assert extract_function(FIXED, "route") is not None
