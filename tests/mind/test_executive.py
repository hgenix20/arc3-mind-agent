"""Executive branch order, gating, and the Baseline wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from arcagent.core.types import GRID_SIZE, RESET_ID, Action, Observation, Phase
from arcagent.mind.baseline import Baseline
from arcagent.mind.executive import Executive, ExecutiveInputs
from arcagent.mind.predictive import ProgramState
from arcagent.mind.types import ActionKey, Program

A1, A2, A3, A4, A5 = (Action(i) for i in (1, 2, 3, 4, 5))


def obs_of(phase: Phase = Phase.NOT_FINISHED, avail: tuple[int, ...] = (1, 2, 3)) -> Observation:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    return Observation(grid, phase, 0, 3, avail)


def state(has_goal: bool, fit_hits: int = 9, last_ok: bool = True) -> ProgramState:
    return ProgramState(
        Program("p", "code", has_goal, "llm"), hits=fit_hits, trials=10, last_ok=last_ok
    )


def make(
    goal: list[Action] | None = None,
    novel: list[Action] | None = None,
    voi: Action | None = None,
) -> Executive:
    return Executive(
        plan_to_goal=lambda _p: goal,
        plan_to_novel=lambda _p: novel,
        voi_pick=lambda: voi,
        b0_pick=lambda _v: A5,
    )


def inputs(**kw: object) -> ExecutiveInputs:
    base: dict[str, object] = {
        "obs": obs_of(),
        "plan": [],
        "active": state(True),
        "best": state(True),
        "n_programs": 1,
        "vetoes": frozenset[ActionKey](),
        "plan_threshold": 0.8,
    }
    base.update(kw)
    return ExecutiveInputs(**base)  # type: ignore[arg-type]


CASES = [
    ("reset", make(), {"obs": obs_of(Phase.GAME_OVER), "plan": [A1]}, RESET_ID, []),
    ("follow", make(), {"plan": [A1, A2]}, 1, [A2]),
    ("plan", make(goal=[A3, A4]), {}, 3, [A4]),
    ("learn", make(voi=A4), {"n_programs": 2, "best": state(False)}, 4, []),
    ("learn", make(novel=[A2, A3]), {"best": state(False)}, 2, [A3]),
    ("fallback", make(), {"n_programs": 0, "best": None, "active": None}, 5, []),
]


@pytest.mark.parametrize(("branch", "ex", "kw", "action_id", "rest"), CASES)
def test_executive_order(
    branch: str, ex: Executive, kw: dict[str, object], action_id: int, rest: list[Action]
) -> None:
    decision, remaining = ex.decide(inputs(**kw))
    assert decision.branch == branch
    assert decision.action.id == action_id
    assert remaining == rest
    assert decision.why


def test_plan_dropped_when_active_program_missed() -> None:
    ex = make(goal=[A3])
    decision, rest = ex.decide(inputs(plan=[A1, A2], active=state(True, last_ok=False)))
    assert decision.branch == "plan"
    assert decision.action == A3
    assert rest == []
    decision, _ = ex.decide(inputs(plan=[A1], active=None, best=None, n_programs=0))
    assert decision.branch == "fallback"


def test_vetoed_plan_action_falls_through() -> None:
    veto = frozenset({A1.key()})
    decision, rest = make(goal=[A3]).decide(inputs(plan=[A1, A2], vetoes=veto))
    assert (decision.branch, decision.action, rest) == ("plan", A3, [])
    # A vetoed goal plan falls to the next branch (learn, one program).
    decision, _ = make(goal=[A1], novel=[A2]).decide(inputs(vetoes=veto))
    assert (decision.branch, decision.action) == ("learn", A2)
    # A vetoed VOI pick falls to B0; empty and None results never raise.
    decision, _ = make(voi=A1).decide(inputs(n_programs=2, best=state(False), vetoes=veto))
    assert decision.branch == "fallback"
    decision, _ = make(goal=[], novel=None).decide(inputs())
    assert decision.branch == "fallback"


def test_terminal_obs_resets_and_clears_plan() -> None:
    for o in (obs_of(Phase.WIN), obs_of(avail=(RESET_ID,))):
        decision, rest = make().decide(inputs(obs=o, plan=[A1, A2]))
        assert decision.branch == "reset"
        assert decision.action == Action(RESET_ID)
        assert rest == []


def test_baseline_choose_exclude_observe_and_reset() -> None:
    b = Baseline(0)
    o = obs_of(avail=(1, 2))
    b.observe(o)
    b.note(o, Action(1))
    assert b.choose(o, frozenset({(1, -1, -1)})) == Action(2)
    b.reset_priors()
    assert b.choose(o, frozenset({(1, -1, -1)})) == Action(2)
