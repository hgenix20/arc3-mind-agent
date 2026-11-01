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
  level, or until ``STALL_MOVES`` of its moves in a row came from its own fallback
  (branch ``fallback``: no plan and nothing to test); then the explorer plays the rest
  of that level. Why the stall rule: on vc33 362 of the mind's 400 moves were its
  fallback, a slower way to run the same explorer. A level change (any change of
  ``obs.levels_completed``) gives the mind the next level, with a fresh start
  (``FluidAgent.resume``) when the explorer played last, so the mind doesn't learn from
  a move it didn't make.
- The explorer is the mind's own fallback explorer (``FluidAgent.explorer``), which
  already watches every move the mind makes, so the takeover inherits the level's graph,
  its dead moves and its no-ops. Pass a separate explorer only when the mind has none
  (tests); then the hybrid feeds it the mind's moves itself.
- ``close`` closes the mind (its sleep and log); the explorer holds nothing to close.
"""

from __future__ import annotations

from typing import Protocol

from arcagent.agent.explore import ExplorerPolicy
from arcagent.core.types import Action, Observation

STALL_MOVES = 25  # fallback moves in a row that end the mind's turn on a level


class Mind(Protocol):
    """What the hybrid needs of the mind: ``FluidAgent``'s play surface."""

    last_branch: str

    def choose(self, obs: Observation) -> Action: ...
    def resume(self) -> None: ...
    def close(self) -> None: ...


class HybridPolicy:
    """See the module docstring."""

    def __init__(
        self, mind: Mind, explorer: ExplorerPolicy, mind_moves: int, *, shared: bool = False
    ) -> None:
        if mind_moves < 1:
            raise ValueError(f"mind_moves must be at least 1, got {mind_moves}")
        self.mind = mind
        self.explorer = explorer
        self.shared = shared  # the explorer already sees the mind's moves
        self.mind_moves = mind_moves
        self.level = -1
        self.level_moves = 0
        self.fallbacks = 0  # the mind's fallback moves in a row on this level
        self.exploring = False

    def choose(self, obs: Observation) -> Action:
        if obs.levels_completed != self.level:
            self.level, self.level_moves, self.fallbacks = obs.levels_completed, 0, 0
            if self.exploring:
                self.exploring = False
                self.mind.resume()
        stalled = self.fallbacks >= STALL_MOVES
        if not self.exploring and (self.level_moves >= self.mind_moves or stalled):
            self.exploring = True
        self.level_moves += 1
        if self.exploring:
            return self.explorer.choose(obs)
        action = self.mind.choose(obs)
        self.fallbacks = self.fallbacks + 1 if self.mind.last_branch == "fallback" else 0
        if not self.shared:
            self.explorer.observe(obs)
            self.explorer.note(obs, action)
        return action

    def close(self) -> None:
        self.mind.close()
