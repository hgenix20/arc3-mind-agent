"""The executive: one action per step in a fixed preference order.

Order: reset, follow the plan, plan to the goal, VOI pick (several programs)
or plan to explore novelty (one program), then B0. A branch whose action is
vetoed (``gate_action``) or whose plan is empty is dropped and the next is
tried. All side effects are injected callables, so the order is testable
with fakes. ``decide`` never raises for an empty plan or a None return.

Contract with the caller:
- ``plan`` is the plan left over from the previous step.
- ``best`` is the highest-fit program. When ``n_programs == 1`` it must be
  that single live program: it is the argument to ``plan_to_novel``. If
  ``best`` is None the ``learn`` branch is skipped.
- The returned plan is what remains after this step (empty unless a plan
  branch acted).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from arcagent.core.types import RESET_ID, Action, Observation
from arcagent.mind.predictive import ProgramState
from arcagent.mind.types import ActionKey, Branch, Decision
from arcagent.mind.wisdom import gate_action, gate_plan_step


@dataclass(frozen=True, slots=True)
class ExecutiveInputs:
    obs: Observation
    plan: list[Action]
    active: ProgramState | None
    best: ProgramState | None
    n_programs: int
    vetoes: frozenset[ActionKey]
    plan_threshold: float


class Executive:
    def __init__(
        self,
        *,
        plan_to_goal: Callable[[ProgramState], list[Action] | None],
        plan_to_novel: Callable[[ProgramState], list[Action] | None],
        voi_pick: Callable[[], Action | None],
        b0_pick: Callable[[frozenset[ActionKey]], Action],
    ) -> None:
        self._plan_to_goal = plan_to_goal
        self._plan_to_novel = plan_to_novel
        self._voi_pick = voi_pick
        self._b0_pick = b0_pick

    def decide(self, inp: ExecutiveInputs) -> tuple[Decision, list[Action]]:
        """Return the decision and the plan that remains after it."""
        obs = inp.obs
        if obs.terminal or all(a == RESET_ID for a in obs.available_actions):
            return Decision(Action(RESET_ID), "reset", "terminal or reset-only"), []
        vetoes = inp.vetoes

        if inp.plan and gate_plan_step(inp.active).verdict == "pass":
            first, rest = inp.plan[0], inp.plan[1:]
            if gate_action(first, vetoes).verdict == "pass":
                why = f"plan step, {len(rest)} left"
                return Decision(first, "follow", why), rest

        best = inp.best
        if best is not None and best.fit >= inp.plan_threshold and best.program.has_goal:
            got = self._take_plan(self._plan_to_goal(best), vetoes, "plan to goal", "plan")
            if got is not None:
                return got

        if inp.n_programs >= 2:
            pick = self._voi_pick()
            if pick is not None and gate_action(pick, vetoes).verdict == "pass":
                return Decision(pick, "learn", "voi pick"), []
        elif inp.n_programs == 1 and best is not None:
            got = self._take_plan(self._plan_to_novel(best), vetoes, "plan to novel", "learn")
            if got is not None:
                return got

        return Decision(self._b0_pick(vetoes), "fallback", "b0"), []

    @staticmethod
    def _take_plan(
        plan: list[Action] | None,
        vetoes: frozenset[ActionKey],
        why: str,
        branch: Branch,
    ) -> tuple[Decision, list[Action]] | None:
        if not plan or gate_action(plan[0], vetoes).verdict != "pass":
            return None
        decision = Decision(plan[0], branch, why)
        return decision, plan[1:]
