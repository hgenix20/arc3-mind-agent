"""The fluid mind first on each level, the graph explorer once the mind has stalled.

Why: the fluid mind (``arcagent.mind.fluid.FluidAgent``, no LLM) models what actions
do and plans toward goal guesses, so the levels it wins take close to the human number
of moves (ls20 level 0 in 33 moves against the explorer's 300 to 600). It wins few
levels, though: when its model or its goal guesses are wrong it stalls. The explorer
wins about twice as many levels by covering the screens it can reach, slowly. Playing
the mind first and the explorer after a stall keeps the mind's efficient wins and
adds the explorer's coverage.

Contract:
- ``choose(obs)``: the mind plays until it has made ``mind_moves`` moves on the current
  level; then the explorer plays the rest of that level. A level change (any change of
  ``obs.levels_completed``) gives the mind the next level, with a fresh start
  (``FluidAgent.resume``) when the explorer played last, so the mind doesn't learn from
  a move it didn't make.
- The explorer watches every move the mind makes (``observe`` and ``note``), so when it
  takes over it already holds the level's graph.
- ``close`` closes the mind (its sleep and log); the explorer holds nothing to close.
"""

from __future__ import annotations

from typing import Protocol

from arcagent.agent.explore import ExplorerPolicy
from arcagent.core.types import Action, Observation


class Mind(Protocol):
    """What the hybrid needs of the mind: ``FluidAgent``'s play surface."""

    def choose(self, obs: Observation) -> Action: ...
    def resume(self) -> None: ...
    def close(self) -> None: ...


class HybridPolicy:
    """See the module docstring."""

    def __init__(self, mind: Mind, explorer: ExplorerPolicy, mind_moves: int) -> None:
        if mind_moves < 1:
            raise ValueError(f"mind_moves must be at least 1, got {mind_moves}")
        self.mind = mind
        self.explorer = explorer
        self.mind_moves = mind_moves
        self.level = -1
        self.level_moves = 0
        self.exploring = False

    def choose(self, obs: Observation) -> Action:
        if obs.levels_completed != self.level:
            self.level, self.level_moves = obs.levels_completed, 0
            if self.exploring:
                self.exploring = False
                self.mind.resume()
        if not self.exploring and self.level_moves >= self.mind_moves:
            self.exploring = True
        self.level_moves += 1
        if self.exploring:
            return self.explorer.choose(obs)
        action = self.mind.choose(obs)
        self.explorer.observe(obs)
        self.explorer.note(obs, action)
        return action

    def close(self) -> None:
        self.mind.close()
