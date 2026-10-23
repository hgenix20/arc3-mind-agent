"""Ablation configs and the one place that turns a config into module instances.

Contract:
- ``MindConfig`` is frozen; a run picks one by name from ``CONFIGS``.
- ``CONFIGS`` holds the spec's ten ablation configs plus ``EXPLORER``. ``B0`` has
  all four modules off but keeps the LLM and the planner, so ``B0+x`` and
  ``FULL-x`` isolate one module each. ``EXPLORER`` has no LLM at all: the
  model-free 0.36 baseline, kept for reference only.
- ``build_modules`` is the only code that maps a flag to a module class. VOI and
  transfer map to their nulls until their real modules exist; Tasks 17 and 18
  change only this function.
"""

from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass

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

MODULE_NAMES = ("metacognition", "inhibition", "voi", "transfer")


@dataclass(frozen=True)
class MindConfig:
    """One agent configuration.

    ``k``: candidate programs per synthesis call. ``max_calls_per_game``: the
    synthesis budget. ``plan_threshold`` and ``plan_min_trials``: the fit and
    evidence a program needs before the executive plans with it.
    ``idle_act_s``: shared idle time after which a waiting game takes a B0
    action. ``synth_wait_poll_s``: the slice a waiting game sleeps between checks.
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
    idle_act_s: float = 600.0
    synth_wait_poll_s: float = 1.0


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
    return out


CONFIGS: dict[str, MindConfig] = _all_configs()


def build_modules(cfg: MindConfig) -> Modules:
    """Fresh module instances for one game (modules hold no cross-game state)."""
    return Modules(
        metacognition=ConfidenceMetacognition() if cfg.metacognition else NullMetacognition(),
        inhibition=DisconfirmInhibition() if cfg.inhibition else NullInhibition(),
        # Real VOI and transfer arrive in Tasks 17 and 18; until then both are nulls.
        voi=NullVoi(),
        transfer=NullTransfer(),
    )
