"""B0: the model-free explorer baseline.

Intent: the floor the mind must beat and the fallback when no program is
trusted. It wraps ``ExplorerPolicy``, whose UCB over action kinds' change
rates is B0's greedy use of learned action effects; nothing is added on top.

Contract: the agent calls ``observe`` once per step with the new observation
(even when another branch will act), then ``note`` with the action taken if
B0 did not choose it. When B0 chooses, ``choose`` already records it. This
keeps B0's state graph complete when plans or VOI picks act most of the time.
"""

from __future__ import annotations

from arcagent.agent.explore import ExplorerPolicy
from arcagent.core.types import Action, Observation
from arcagent.mind.types import ActionKey


class Baseline:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._policy = ExplorerPolicy(seed)

    def observe(self, obs: Observation) -> None:
        """Learn the previous action's outcome without choosing."""
        self._policy.observe(obs)

    def note(self, obs: Observation, action: Action) -> None:
        """Record an action taken by another branch from ``obs``."""
        self._policy.note(obs, action)

    def choose(self, obs: Observation, exclude: frozenset[ActionKey]) -> Action:
        """Explorer pick avoiding ``exclude``; ignores it only when nothing else is left."""
        return self._policy.choose(obs, exclude)

    def reset_priors(self) -> None:
        """Start from a fresh explorer (null transfer at level-up)."""
        self._policy = ExplorerPolicy(self.seed)
