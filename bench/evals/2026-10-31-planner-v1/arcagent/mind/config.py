"""Ablation configs and the one place that turns a config into module instances.

Contract:
- ``MindConfig`` is frozen; a run picks one by name from ``CONFIGS``.
- ``CONFIGS`` holds the spec's ten ablation configs plus ``EXPLORER``. ``B0`` has
  all four modules off but keeps the LLM and the planner, so ``B0+x`` and
  ``FULL-x`` isolate one module each. ``EXPLORER`` has no LLM at all: the
  model-free 0.36 baseline, kept for reference only. ``FLUID`` is the fluid
  agent (``arcagent.mind.fluid``): facts, goal guesses and short search, with
  no LLM and no program sandbox, all four of its own modules on
  (``arcagent.mind.fluid_modules``). ``FLUID-LLM`` is the same agent with the
  executive turn on (``llm=True``): a vision model proposes goal guesses, a plan
  or a test, and slice 1 logic checks and executes them. ``FLUID-LLM-<module>``
  is ``FLUID-LLM`` with that one module null, the fluid ablations. ``DRIVE`` is
  ``FLUID-LLM`` played by ``llm_drive.DriveAgent`` (``drive=True``): the LLM drives and
  the deterministic code is its toolkit. ``PLANNER`` is ``FLUID-LLM`` played by
  ``planner_agent.PlannerAgent`` (``planner=True``): the spec v2 planning loop
  with a long-term store that lives across games.
- ``build_modules`` is the only code that maps a flag to a MindAgent module class
  (Transfer selects ``CarryTransfer`` or its null); ``fluid_modules.build_fluid_modules``
  does the same for the fluid agent.
"""

from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass, replace

from arcagent.mind.modules.base import (
    Inhibition,
    Metacognition,
    NullInhibition,
    NullMetacognition,
    NullTransfer,
    NullVoi,
    Transfer,
    Voi,
)
from arcagent.mind.modules.inhibition import DisconfirmInhibition
from arcagent.mind.modules.metacognition import ConfidenceMetacognition
from arcagent.mind.modules.transfer import CarryTransfer
from arcagent.mind.modules.voi import DisagreementVoi

MODULE_NAMES = ("metacognition", "inhibition", "voi", "transfer")


@dataclass(frozen=True)
class MindConfig:
    """One agent configuration.

    ``k``: candidate programs per synthesis call. ``max_calls_per_game``: the
    synthesis budget. ``plan_threshold`` and ``plan_min_trials``: the fit and
    evidence a program needs before the executive plans with it.
    ``synth_cutoff_s``: no synthesis call starts with less than this left before
    the game (or run) deadline, so none is still running when the game ends.
    The agent caps it at ``SYNTH_CUTOFF_FRACTION`` (0.25) of the game's budget
    (agent start to deadline), so a game shorter than this value can still call
    the model during the first 75% of its time.
    ``fluid``: play with ``FluidAgent`` instead of ``MindAgent``.
    ``drive``: play a fluid config with ``DriveAgent``; ``planner``: with
    ``PlannerAgent``. No config sets both; ``submit.MindRun`` reads ``planner`` first.
    """

    name: str
    metacognition: bool
    inhibition: bool
    voi: bool
    transfer: bool
    llm: bool
    k: int = 4
    max_calls_per_game: int = 12
    plan_threshold: float = 0.9
    plan_min_trials: int = 5
    synth_cutoff_s: float = 300.0
    fluid: bool = False
    drive: bool = False
    planner: bool = False


@dataclass(frozen=True)
class Modules:
    """The four module instances an agent runs with."""

    metacognition: Metacognition
    inhibition: Inhibition
    voi: Voi
    transfer: Transfer


def _config(name: str, on: Container[str], *, llm: bool = True) -> MindConfig:
    return MindConfig(
        name,
        metacognition="metacognition" in on,
        inhibition="inhibition" in on,
        voi="voi" in on,
        transfer="transfer" in on,
        llm=llm,
    )


def _all_configs() -> dict[str, MindConfig]:
    every = set(MODULE_NAMES)
    out = {"FULL": _config("FULL", every), "B0": _config("B0", set())}
    for m in MODULE_NAMES:
        out[f"B0+{m}"] = _config(f"B0+{m}", {m})
    for m in MODULE_NAMES:
        out[f"FULL-{m}"] = _config(f"FULL-{m}", every - {m})
    out["EXPLORER"] = _config("EXPLORER", set(), llm=False)
    out["FLUID"] = MindConfig(
        "FLUID",
        metacognition=True,
        inhibition=True,
        voi=True,
        transfer=True,
        llm=False,
        fluid=True,
    )
    out["FLUID-LLM"] = replace(out["FLUID"], name="FLUID-LLM", llm=True)
    for m in MODULE_NAMES:
        name = f"FLUID-LLM-{m}"
        out[name] = replace(out["FLUID-LLM"], name=name, **{m: False})
    out["DRIVE"] = replace(out["FLUID-LLM"], name="DRIVE", drive=True)
    out["PLANNER"] = replace(out["FLUID-LLM"], name="PLANNER", planner=True)
    return out


CONFIGS: dict[str, MindConfig] = _all_configs()


def build_modules(cfg: MindConfig) -> Modules:
    """Fresh module instances for one game (modules hold no cross-game state)."""
    return Modules(
        metacognition=ConfidenceMetacognition() if cfg.metacognition else NullMetacognition(),
        inhibition=DisconfirmInhibition() if cfg.inhibition else NullInhibition(),
        voi=DisagreementVoi() if cfg.voi else NullVoi(),
        transfer=CarryTransfer() if cfg.transfer else NullTransfer(),
    )
