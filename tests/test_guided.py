"""The guided explorer: features, win credit and frontier choice."""

from __future__ import annotations

import numpy as np

from arcagent.agent.explore import Graph
from arcagent.agent.guided import GuidedExplorer, best_frontier_path, features
from arcagent.core.types import GRID_SIZE, Action, Observation, Phase


def test_features_measure_color_gaps_and_counts() -> None:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    g[5:8, 5:8] = 3
    g[5:8, 10:12] = 4
    f = features(g)
    assert f["count", 3, -1] == 9 and f["count", 4, -1] == 6
    assert f["gap", 3, 4] == 3  # x 7 to x 10; colors in order
    assert not any(k[1] == 0 for k in f)  # the background is never measured


def test_best_frontier_path_takes_the_lowest_score_over_the_nearest() -> None:
    g = Graph()
    a, b, c, d = (bytes([i]) for i in range(4))
    g.edges = {a: {(1, -1, -1): b, (2, -1, -1): c}, c: {(1, -1, -1): d}}
    g.cands = {a: [Action(1), Action(2)], b: [Action(3)], c: [Action(1)], d: [Action(3)]}
    score = {b: 5.0, d: 1.0}
    assert best_frontier_path(g, a, lambda k: score.get(k, 9.0)) == [Action(2), Action(1)]


class ReachEnv:
    """A block moves right with action 1 or left with action 2; touching the post wins.

    Each level puts the post further away, so blind coverage costs more moves on later
    levels than heading for the post does.
    """

    def __init__(self) -> None:
        self.level = 0
        self.x = 2

    def observe(self) -> Observation:
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        g[20, self.x] = 3
        g[20, self.post] = 9
        return Observation(g, Phase.NOT_FINISHED, self.level, 5, (1, 2))

    @property
    def post(self) -> int:
        return 10 + 10 * self.level

    def step(self, a: Action) -> Observation:
        self.x = min(max(self.x + (1 if a.id == 1 else -1), 0), GRID_SIZE - 1)
        if self.x == self.post - 1:
            self.level += 1
            self.x = 2
        return self.observe()


def test_a_win_credits_the_gap_that_closed() -> None:
    env, p = ReachEnv(), GuidedExplorer(0)
    o = env.observe()
    for _ in range(200):
        if env.level:
            break
        o = env.step(p.choose(o))
    p.observe(o)
    assert env.level == 1
    assert p.credit.get(("gap", 3, 9), 0) > 0


class AvatarEnv:
    """An avatar of colors 8 and 4 walks a floor of color 2 to an exit of color 8.

    The avatar carries the exit's color, so the plain color gap between 4 and 8 is zero
    on every frame; only the avatar's distance to the static 8 says how close it is.
    """

    def __init__(self) -> None:
        self.level = 0
        self.x = 2

    @property
    def exit(self) -> int:
        return 12 + 8 * self.level

    def observe(self) -> Observation:
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        g[19:22, 0:40] = 2
        g[19:22, self.exit] = 8
        g[19:22, self.x] = 8
        g[20, self.x] = 4
        return Observation(g, Phase.NOT_FINISHED, self.level, 5, (1, 2))

    def step(self, a: Action) -> Observation:
        self.x = min(max(self.x + (1 if a.id == 1 else -1), 0), 39)
        if self.x == self.exit - 1:
            self.level += 1
            self.x = 2
        return self.observe()


def test_a_win_credits_the_mover_reaching_a_color_it_also_wears() -> None:
    env, p = AvatarEnv(), GuidedExplorer(0)
    o = env.observe()
    for _ in range(200):
        if env.level:
            break
        o = env.step(p.choose(o))
    p.observe(o)
    assert env.level == 1
    assert {4, 8} <= p.movers()
    assert p.credit.get(("reach", 8, -1), 0) > 0
    assert p.credit.get(("gap", 4, 8), 0) == 0


def test_reach_reads_the_avatar_distance_from_a_state_key() -> None:
    env, p = AvatarEnv(), GuidedExplorer(0)
    o = env.observe()
    for _ in range(400):
        if env.level == 2:
            break
        o = env.step(p.choose(o))
    p.observe(o)
    assert env.level == 2
    env.x = 5
    far = p.key(env.observe())
    env.x = 20
    near = p.key(env.observe())
    assert p.reach(near)["reach", 8, -1] == 8  # x 20 to the exit at x 28
    assert p.reach(far)["reach", 8, -1] == 23
