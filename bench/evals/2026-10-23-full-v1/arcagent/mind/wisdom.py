"""The wisdom function: deterministic gates on what may enter an executor.

Contract: pure functions of their arguments, no I/O, no model. Every verdict
carries a ``reason``; the caller logs each ``drop`` as a ``wisdom_drop`` event,
so nothing is discarded without a record.
"""

from __future__ import annotations

from arcagent.core.types import Action
from arcagent.mind.predictive import ProgramState
from arcagent.mind.types import ActionKey, Gate


def admit_program(
    hits: int,
    trials: int,
    load_error: str | None,
    *,
    admit_fit: float = 0.8,
    hold_fit: float = 0.5,
) -> Gate:
    """Gate a candidate program on its fit to the recorded transitions.

    Drops on a load error or zero trials (no evidence either way). Otherwise
    ``fit = hits / trials``: pass at ``fit >= admit_fit``, hold at
    ``fit >= hold_fit``, drop below. Equal to a threshold counts as reaching it.
    """
    if load_error is not None:
        return Gate("drop", f"load error: {load_error}")
    if trials <= 0:
        return Gate("drop", "no trials to judge the program on")
    fit = hits / trials
    detail = f"fit {hits}/{trials}"
    if fit >= admit_fit:
        return Gate("pass", f"{detail} reaches admit {admit_fit}")
    if fit >= hold_fit:
        return Gate("hold", f"{detail} reaches hold {hold_fit} but not admit {admit_fit}")
    return Gate("drop", f"{detail} below hold {hold_fit}")


def gate_action(action: Action, vetoes: frozenset[ActionKey]) -> Gate:
    """Drop an action whose key is vetoed (inhibition), pass any other."""
    if action.key() in vetoes:
        return Gate("drop", f"action {action.key()} is vetoed")
    return Gate("pass", "not vetoed")


def gate_plan_step(active: ProgramState | None) -> Gate:
    """Drop a plan step when no program is active or its last prediction missed."""
    if active is None:
        return Gate("drop", "no active program")
    if not active.last_ok:
        return Gate("drop", f"program {active.program.id} missed its last prediction")
    return Gate("pass", f"program {active.program.id} predicted its last step")
