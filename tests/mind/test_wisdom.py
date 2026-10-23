"""The wisdom function: deterministic gates on programs, actions and plan steps."""

from __future__ import annotations

from arcagent.core.types import Action
from arcagent.mind.predictive import ProgramState
from arcagent.mind.types import Program
from arcagent.mind.wisdom import admit_program, gate_action, gate_plan_step


def test_admit_pass_hold_drop_bands() -> None:
    assert admit_program(8, 10, None).verdict == "pass"
    assert admit_program(6, 10, None).verdict == "hold"
    assert admit_program(5, 10, None).verdict == "hold"
    assert admit_program(4, 10, None).verdict == "drop"
    assert admit_program(10, 10, "SyntaxError").verdict == "drop"
    assert admit_program(0, 0, None).verdict == "drop"
    for g in (admit_program(8, 10, None), admit_program(0, 0, None)):
        assert g.reason


def test_gate_action_and_plan_step() -> None:
    a = Action(6, 3, 4)
    assert gate_action(a, frozenset({a.key()})).verdict == "drop"
    assert gate_action(a, frozenset({(1, -1, -1)})).verdict == "pass"
    assert gate_action(a, frozenset()).verdict == "pass"

    prog = Program("p", "", False, "llm")
    assert gate_plan_step(None).verdict == "drop"
    assert gate_plan_step(ProgramState(prog, last_ok=False)).verdict == "drop"
    assert gate_plan_step(ProgramState(prog)).verdict == "pass"
