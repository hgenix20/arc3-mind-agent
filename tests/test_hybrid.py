"""The hybrid policy: the mind first on each level, the explorer after a stall."""

from __future__ import annotations

from arcagent.agent.explore import ExplorerPolicy
from arcagent.agent.hybrid import HybridPolicy
from arcagent.core.types import Action, Observation
from tests.test_guided import ReachEnv


class StuckMind:
    """Always presses action 2 (away from the post); counts resumes."""

    def __init__(self, branch: str = "plan") -> None:
        self.last_branch = ""
        self.branch = branch
        self.moves = 0
        self.resumes = 0
        self.closed = False

    def choose(self, obs: Observation) -> Action:
        del obs
        self.moves += 1
        self.last_branch = self.branch
        return Action(2)

    def resume(self) -> None:
        self.resumes += 1

    def close(self) -> None:
        self.closed = True


def test_the_explorer_takes_over_a_stalled_level_and_hands_the_next_back() -> None:
    env, mind = ReachEnv(), StuckMind()
    p = HybridPolicy(mind, ExplorerPolicy(0), mind_moves=5)
    o = env.observe()
    for _ in range(300):
        if env.level:
            break
        o = env.step(p.choose(o))
    assert env.level == 1 and mind.moves == 5 and p.exploring
    p.choose(o)
    assert not p.exploring and mind.resumes == 1 and mind.moves == 6
    p.close()
    assert mind.closed


def test_the_explorer_learns_from_the_minds_moves() -> None:
    env, mind = ReachEnv(), StuckMind()
    p = HybridPolicy(mind, ExplorerPolicy(0), mind_moves=3)
    o = env.observe()
    for _ in range(3):
        o = env.step(p.choose(o))
    p.explorer.observe(o)
    assert len(p.explorer.graph.edges) == 2  # x 2 to 1, 1 to 0; the third hits the wall
    assert p.explorer.graph.noop


def test_a_mind_stuck_in_its_fallback_hands_the_level_over_early() -> None:
    from arcagent.agent.hybrid import STALL_MOVES  # noqa: PLC0415

    env, mind = ReachEnv(), StuckMind(branch="fallback")
    p = HybridPolicy(mind, ExplorerPolicy(0), mind_moves=400)
    o = env.observe()
    for _ in range(STALL_MOVES + 1):
        o = env.step(p.choose(o))
    assert mind.moves == STALL_MOVES and p.exploring
