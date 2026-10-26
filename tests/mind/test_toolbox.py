"""Toolbox: patched tools run in the worker, fall back on failure, versions stay ordered."""

from __future__ import annotations

import math
import pickle
import threading
from collections.abc import Iterator
from typing import cast

import numpy as np
import pytest

from arcagent.core.types import Action
from arcagent.mind.objects import Scene, Tracker
from arcagent.mind.route import Dist, route
from arcagent.mind.toolbox import (
    MAX_CASES,
    TOOLS,
    SceneActions,
    Toolbox,
    ToolWorker,
    call_tool,
)

REVERSED = """
def route(start, targets, dist, *, exact_limit=ROUTE_EXACT):
    order = list(range(len(targets)))[::-1]
    return order, -1
"""
HANGS = "def route(start, targets, dist, *, exact_limit=6):\n    while True:\n        pass\n"
CRASHES = "import os\ndef route(start, targets, dist, *, exact_limit=6):\n    os._exit(3)\n"

TARGETS = [(3, 0), (6, 0)]
DIST = cast("Dist", math.dist)  # picklable, so it crosses to the worker


@pytest.fixture
def worker() -> Iterator[ToolWorker]:
    w = ToolWorker()
    yield w
    w.close()


def test_patched_tool_runs_in_worker(worker: ToolWorker) -> None:
    box = Toolbox()
    assert call_tool("route", worker, (0, 0), TARGETS, DIST, box=box) == (
        route((0, 0), TARGETS, DIST),
        0,
        "",
    )
    no = box.adopt("route", REVERSED, "g1", "test")
    got, version, err = call_tool("route", worker, (0, 0), TARGETS, DIST, box=box)
    assert got == ([1, 0], -1) and version == no and err == ""


def test_hanging_patch_falls_back(worker: ToolWorker) -> None:
    box = Toolbox()
    box.adopt("route", HANGS, "g1", "test")
    got, version, err = call_tool("route", worker, (0, 0), TARGETS, DIST, box=box, timeout_s=0.5)
    assert got == route((0, 0), TARGETS, DIST) and version == 0 and err == "timeout"
    assert worker.restarts == 1


def test_crash_restarts_worker(worker: ToolWorker) -> None:
    box = Toolbox()
    box.adopt("route", CRASHES, "g1", "test")
    _, version, err = call_tool("route", worker, (0, 0), TARGETS, DIST, box=box)
    assert version == 0 and err == "crash" and worker.restarts == 1
    box.rollback("route", "crashed")
    box.adopt("route", REVERSED, "g1", "second try")
    got, _, err = call_tool("route", worker, (0, 0), TARGETS, DIST, box=box)
    assert got == ([1, 0], -1) and err == ""


def test_adopt_and_rollback_versions() -> None:
    box = Toolbox()
    assert box.current("route") is None
    a = box.adopt("route", REVERSED, "g1", "first")
    b = box.adopt("route", HANGS, "g2", "second")
    current = box.current("route")
    assert current is not None and current.no == b > a
    assert box.rollback("route", "worse") == a
    assert box.rollback("route", "worse") == 0 and box.current("route") is None
    assert [h[0] for h in box.history()] == ["adopt", "adopt", "rollback", "rollback"]


def test_cases_capped() -> None:
    box = Toolbox()
    for i in range(MAX_CASES + 3):
        box.add_case("route", ((0, 0), [(i, 0)]), {}, ([0], i))
    cases = box.cases("route")
    assert len(cases) == MAX_CASES and cases[0].result == ([0], 3)


def test_concurrent_adopts_ordered() -> None:
    box = Toolbox()
    nos: list[int] = []
    lock = threading.Lock()

    def adopt() -> None:
        for _ in range(50):
            n = box.adopt("route", REVERSED, "g", "r")
            with lock:
                nos.append(n)

    threads = [threading.Thread(target=adopt) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(nos) == list(range(1, 201))


def test_scene_actions_pickle_and_match_clicks() -> None:
    g = np.zeros((64, 64), dtype=np.int8)
    g[2, 2] = 1
    scene: Scene = Tracker().update(g, frozenset())[0]
    acts = SceneActions((Action(1), Action(2)), can_click=True)
    again = pickle.loads(pickle.dumps(acts))
    assert again(scene) == acts(scene) == [Action(1), Action(2), Action(6, 2, 2)]


def test_agrees_rules() -> None:
    assert TOOLS["route"].agrees(([0, 1], 5), ([1, 0], 4))
    assert not TOOLS["route"].agrees(([0, 1], 5), None)
    assert TOOLS["route"].agrees(None, None)
