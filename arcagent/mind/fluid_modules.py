"""The paper's four modules for the fluid agent, as swappable Protocols with nulls.

Same pattern as ``arcagent.mind.modules.base``: modules advise, the agent applies the
result and logs it. ``build_fluid_modules`` is the only code that maps a config flag
to a class, so an ablation (``FLUID-LLM-<module>``) switches one module to its null
without touching the agent.

- Voi picks the next targeted test. Real: ``tester.choose_test`` with the agent's
  protected sigs and per-level exhausted pairs, so only countable pairs make a test.
  Null: the first untried action in ``Action.key()`` order whose prediction has an
  unknown pair (no ranking by how much it answers, no countable-pair rule).
- Inhibition refuses an action. Real: ``tester.veto``. Null: never refuses.
- Metacognition sets the misses-in-a-row threshold at which a fact is broken
  (``FactMemory.broken_after``, read once when the agent is built). Real:
  ``BROKEN_MISSES``. Null: ``NEVER_BROKEN``, so facts never break and broken facts
  never rank a test first.
- Transfer runs at every level-up, after sleep and ``FactMemory.new_level``. Real:
  keeps everything. Null: ``FactMemory.forget`` (facts, toc, blockers, walkable, safe
  sets, positions); ``GoalBook`` keeps its confirmed forms, which are goal knowledge.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

from arcagent.core.types import Action
from arcagent.mind.config import MindConfig
from arcagent.mind.facts import BROKEN_MISSES, FactMemory
from arcagent.mind.goals import GoalBook
from arcagent.mind.objects import Scene
from arcagent.mind.predictor import predict
from arcagent.mind.tester import Test, Tried, choose_test, question, veto

NEVER_BROKEN = 10**9  # the null metacognition's threshold: no fact ever breaks

__all__ = [
    "NEVER_BROKEN",
    "FluidInhibition",
    "FluidMetacognition",
    "FluidModules",
    "FluidTransfer",
    "FluidVoi",
    "KeepTransfer",
    "MissMetacognition",
    "NullFluidInhibition",
    "NullFluidMetacognition",
    "NullFluidTransfer",
    "NullFluidVoi",
    "TesterInhibition",
    "TesterVoi",
    "build_fluid_modules",
]


class FluidVoi(Protocol):
    def choose(  # noqa: PLR0917  # mirrors tester.choose_test
        self,
        scene: Scene,
        actions: list[Action],
        mem: FactMemory,
        frontier: Sequence[tuple[str, str]],
        tried: Tried,
        toward: Action | None,
        *,
        protected: frozenset[str] | None = None,
        exhausted: Collection[tuple[str, str]] = (),
    ) -> Test | None:
        """The next targeted test, or None when nothing is open.

        ``protected`` and ``exhausted`` restrict which open pairs count (see
        ``tester.choose_test``); a null may ignore them.
        """
        ...


class FluidInhibition(Protocol):
    def veto(self, scene: Scene, action: Action, mem: FactMemory, tried: Tried) -> str | None:
        """Why to refuse ``action`` here, or None."""
        ...


class FluidMetacognition(Protocol):
    def broken_after(self) -> int:
        """Misses in a row after which a fact is broken."""
        ...


class FluidTransfer(Protocol):
    def carry(self, mem: FactMemory, goals: GoalBook) -> None:
        """What crosses a level change; may clear ``mem``."""
        ...


class TesterVoi:
    def choose(  # noqa: PLR0917  # mirrors tester.choose_test
        self,
        scene: Scene,
        actions: list[Action],
        mem: FactMemory,
        frontier: Sequence[tuple[str, str]],
        tried: Tried,
        toward: Action | None,
        *,
        protected: frozenset[str] | None = None,
        exhausted: Collection[tuple[str, str]] = (),
    ) -> Test | None:
        return choose_test(
            scene, actions, mem, frontier, tried, toward, protected=protected, exhausted=exhausted
        )


class NullFluidVoi:
    def choose(  # noqa: PLR0917  # mirrors tester.choose_test
        self,
        scene: Scene,
        actions: list[Action],
        mem: FactMemory,
        frontier: Sequence[tuple[str, str]],
        tried: Tried,
        toward: Action | None,
        *,
        protected: frozenset[str] | None = None,
        exhausted: Collection[tuple[str, str]] = (),
    ) -> Test | None:
        skey = scene.key()
        for action in sorted(actions, key=Action.key):
            if (skey, action.key()) in tried:
                continue
            pred = predict(scene, action, mem)
            if pred.unknown:
                label, subject = pred.unknown[0]
                return Test(action, question(label, subject), len(pred.unknown))
        return None


class TesterInhibition:
    def veto(self, scene: Scene, action: Action, mem: FactMemory, tried: Tried) -> str | None:
        return veto(scene, action, mem, tried)


class NullFluidInhibition:
    def veto(self, scene: Scene, action: Action, mem: FactMemory, tried: Tried) -> str | None:
        return None


class MissMetacognition:
    def broken_after(self) -> int:
        return BROKEN_MISSES


class NullFluidMetacognition:
    def broken_after(self) -> int:
        return NEVER_BROKEN


class KeepTransfer:
    def carry(self, mem: FactMemory, goals: GoalBook) -> None:
        return None


class NullFluidTransfer:
    def carry(self, mem: FactMemory, goals: GoalBook) -> None:
        mem.forget()


@dataclass(frozen=True, slots=True)
class FluidModules:
    """The four module instances one fluid agent runs with."""

    voi: FluidVoi
    inhibition: FluidInhibition
    metacognition: FluidMetacognition
    transfer: FluidTransfer


def build_fluid_modules(cfg: MindConfig) -> FluidModules:
    """Fresh modules for one game: each flag picks the real module or its null."""
    return FluidModules(
        voi=TesterVoi() if cfg.voi else NullFluidVoi(),
        inhibition=TesterInhibition() if cfg.inhibition else NullFluidInhibition(),
        metacognition=MissMetacognition() if cfg.metacognition else NullFluidMetacognition(),
        transfer=KeepTransfer() if cfg.transfer else NullFluidTransfer(),
    )
