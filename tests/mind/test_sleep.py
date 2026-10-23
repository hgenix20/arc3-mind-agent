"""Sleep: compaction, consolidation, replay re-gating and goal evidence, on a real sandbox."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from arcagent.core.types import GRID_SIZE, Action, Grid, Phase
from arcagent.mind.memory import GameStore, WorkingMemory, all_steps
from arcagent.mind.predictive import Belief
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.sleep import sleep
from arcagent.mind.types import Program, Step

# Writes the action id into cell (0, 0); exact on every step built by ``good_step``.
GOOD = "def step(grid, action, x=-1, y=-1):\n    g=[r[:] for r in grid]; g[0][0]=action; return g\n"
IDENTITY = "def step(grid, action, x=-1, y=-1):\n    return [r[:] for r in grid]\n"
GOAL_AT_3 = GOOD + "def goal(grid):\n    return grid[0][0] == 3\n"


def zeros() -> Grid:
    return np.zeros((GRID_SIZE, GRID_SIZE), np.int8)


def good_step(index: int, action_id: int = 1, *, level_up: bool = False) -> Step:
    after = zeros()
    after[0, 0] = action_id
    return Step(
        index=index,
        before=zeros(),
        action=Action(action_id),
        after=after,
        levels_before=0,
        levels_after=1 if level_up else 0,
        phase_after=Phase.NOT_FINISHED,
    )


def prog(pid: str, code: str, *, has_goal: bool = False) -> Program:
    return Program(id=pid, code=code, has_goal=has_goal, origin="llm")


@pytest.fixture
def sb() -> Iterator[ProgramSandbox]:
    with ProgramSandbox(timeout_s=2.0) as s:
        yield s


def test_sleep_compacts_and_stores_goal_evidence(sb: ProgramSandbox) -> None:
    wm, store, belief = WorkingMemory(), GameStore(), Belief(sb)
    wm.steps = [good_step(0), good_step(1, 3, level_up=True)]
    wm.put("k", "v", 5)
    report = sleep("level_up", wm, store, belief, sb)
    assert report.reason == "level_up" and report.steps_compacted == 2
    assert wm.steps == [] and len(store.episodes) == 1 and len(store.episodes[0]) == 2
    last = store.episodes[0][-1]
    assert len(store.goal_evidence) == 1
    assert np.array_equal(store.goal_evidence[0][0], last.before)
    assert np.array_equal(store.goal_evidence[0][1], last.after)
    assert len(all_steps(store, wm)) == 2

    # game_end: an empty episode is not appended and no evidence is stored.
    end = sleep("game_end", wm, store, belief, sb)
    assert end.steps_compacted == 0 and len(store.episodes) == 1 and len(store.goal_evidence) == 1
    wm.steps = [good_step(2)]
    sleep("game_end", wm, store, belief, sb)
    assert len(store.episodes) == 2 and len(store.goal_evidence) == 1


def test_consolidation_keeps_only_fit_programs(sb: ProgramSandbox) -> None:
    wm, store, belief = WorkingMemory(), GameStore(), Belief(sb)
    belief.add(prog("good", GOOD), hits=9, trials=10)
    belief.add(prog("weak", GOOD), hits=1, trials=10)
    wm.steps = [good_step(0), good_step(1)]
    report = sleep("level_up", wm, store, belief, sb)
    assert report.consolidated == ["good"]
    assert set(store.procedural) == {"good"} and store.procedural["good"][1:] == (2, 2)
    assert report.replay == {"good": (2, 2)}
    assert set(belief.programs) == {"good", "weak"}  # belief is transfer's business


def test_replay_rescores_and_promotes_held_program(sb: ProgramSandbox) -> None:
    wm, store, belief = WorkingMemory(), GameStore(), Belief(sb)
    held = prog("h", GOOD)
    assert sb.load("h", GOOD) is None
    store.held["h"] = held
    wm.steps = [good_step(0), good_step(1, 2)]
    report = sleep("level_up", wm, store, belief, sb)
    assert report.replay == {"h": (2, 2)}
    assert "h" in store.procedural and store.procedural["h"][0] == held
    assert "h" not in store.held and report.dropped == []


def test_replay_holds_a_middling_program(sb: ProgramSandbox) -> None:
    wm, store, belief = WorkingMemory(), GameStore(), Belief(sb)
    # Identity fits the two unchanged steps and misses the one that writes cell (0, 0).
    sb.load("mid", IDENTITY)
    store.procedural["mid"] = (prog("mid", IDENTITY), 10, 10)
    same = Step(0, zeros(), Action(1), zeros(), 0, 0, Phase.NOT_FINISHED)
    wm.steps = [same, same, good_step(2)]  # identity fits 2 of 3 = 0.67
    report = sleep("level_up", wm, store, belief, sb)
    assert report.replay == {"mid": (2, 3)}
    assert "mid" in store.held and "mid" not in store.procedural and report.dropped == []


def test_replay_drops_program_that_fails_old_episodes(sb: ProgramSandbox) -> None:
    wm, store, belief = WorkingMemory(), GameStore(), Belief(sb)
    sb.load("bad", IDENTITY)
    store.procedural["bad"] = (prog("bad", IDENTITY), 5, 5)
    store.episodes.append([good_step(0), good_step(1, 2), good_step(2, 3)])
    wm.steps = [good_step(3)]
    report = sleep("level_up", wm, store, belief, sb)
    assert report.dropped == ["bad"] and report.replay == {"bad": (0, 4)}
    assert "bad" not in store.procedural and "bad" not in store.held
    # unloaded: scoring an unknown pid is a total miss
    assert sb.score("bad", [good_step(0)]) == (0, 1)


def test_goal_confirmed_needs_true_after_and_false_before(sb: ProgramSandbox) -> None:
    def run(code: str, has_goal: bool = True) -> list[str]:
        wm, store, belief = WorkingMemory(), GameStore(), Belief(sb)
        belief.add(prog("g", code, has_goal=has_goal), hits=10, trials=10)
        wm.steps = [good_step(0, 3, level_up=True)]  # after[0][0] == 3, before == 0
        report = sleep("level_up", wm, store, belief, sb)
        sb.unload("g")
        return report.goal_confirmed

    assert run(GOAL_AT_3) == ["g"]
    assert run(GOAL_AT_3, has_goal=False) == []
    assert run(GOOD) == []  # no goal function: None is not confirmed
    always = GOOD + "def goal(grid):\n    return True\n"
    assert run(always) == []  # true before as well

    wm, store, belief = WorkingMemory(), GameStore(), Belief(sb)
    belief.add(prog("g", GOAL_AT_3, has_goal=True), hits=10, trials=10)
    wm.steps = [good_step(0, 3)]
    assert sleep("game_end", wm, store, belief, sb).goal_confirmed == []
