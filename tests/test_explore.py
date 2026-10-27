"""The explorer policy: candidates, bookkeeping, frontier routing, level reset."""

from __future__ import annotations

import numpy as np
import pytest

from arcagent.agent import explore
from arcagent.agent.explore import ExplorerPolicy, click_candidates, state_key
from arcagent.agent.submit import make_policy
from arcagent.core.types import GRID_SIZE, Action, Grid, Observation, Phase


def grid_with(*blocks: tuple[int, int, int, int, int]) -> Grid:
    """Blank grid plus (x, y, w, h, color) blocks."""
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for x, y, w, h, c in blocks:
        g[y : y + h, x : x + w] = c
    return g


def obs(
    grid: Grid,
    level: int = 0,
    phase: Phase = Phase.NOT_FINISHED,
    avail: tuple[int, ...] = (1, 2, 6),
) -> Observation:
    return Observation(grid, phase, level, 3, tuple(avail))


class GraphEnv:
    """A tiny deterministic game: states are ints, ``table[state][action key]`` gives the next.

    Missing entries are no-ops. Reaching ``goal`` completes the level and
    moves to a fresh layout (state 100).
    """

    def __init__(self, table: dict[int, dict[tuple[int, int, int], int]], goal: int) -> None:
        self.table = table
        self.goal = goal
        self.state = 0
        self.level = 0

    def observe(self) -> Observation:
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        g[0, 0] = self.state % 16
        g[1, 0] = self.state // 16
        g[5:8, 5:8] = 7  # one clickable component
        return obs(g, self.level)

    def step(self, a: Action) -> Observation:
        nxt = self.table.get(self.state, {}).get(a.key())
        if nxt is not None:
            self.state = nxt
            if nxt == self.goal:
                self.level += 1
                self.state = 100
        return self.observe()


def test_click_candidates_cover_small_and_large_components() -> None:
    blocks = [(0, 0, 20, 20, 3), (40, 40, 2, 2, 5)] + [(x, 30, 1, 1, 9) for x in range(0, 64, 2)]
    g = grid_with(*blocks)
    cands = click_candidates(g, limit=8)
    assert len(cands) == 8
    assert (10, 10) in cands or (9, 9) in cands  # the large block
    assert any(g[y, x] == 9 for x, y in cands)  # a single-cell component survives
    assert all(g[y, x] != 0 for x, y in cands)  # never the background
    assert len(click_candidates(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8))) == 16


def test_state_key_includes_level() -> None:
    g = grid_with()
    assert state_key(obs(g, 0)) != state_key(obs(g, 1))
    assert state_key(obs(g, 1)) == state_key(obs(g.copy(), 1))


def test_explorer_records_noops_and_changes_and_routes_to_frontier() -> None:
    # 0 -(1)-> 1 -(2)-> 2(goal). Action 2 at 0 is a no-op; clicks are no-ops.
    env = GraphEnv({0: {(1, -1, -1): 1}, 1: {(2, -1, -1): 2}}, goal=2)
    pol = ExplorerPolicy(seed=1)
    o = env.observe()
    seen: list[tuple[int, tuple[int, int, int]]] = []
    for _ in range(40):
        a = pol.choose(o)
        seen.append((env.state, a.key()))
        o = env.step(a)
        if env.level == 1:
            break
    assert env.level == 1, seen
    # Every (state, action) was tried at most once before the level changed.
    assert len(seen) == len(set(seen))
    pol.choose(o)  # the policy learns of the new level from the next observation
    assert pol.level == 1 and len(pol.graph.cands) == 1  # fresh graph for the new layout


def test_explorer_prefers_kinds_that_changed_the_frame() -> None:
    pol = ExplorerPolicy(seed=0)
    pol.tries[(1, -1)] = 10
    pol.changes[(1, -1)] = 10
    pol.tries[(2, -1)] = 10
    assert pol.change_rate((1, -1)) > pol.change_rate((2, -1))
    assert pol.ucb((3, -1)) > pol.ucb((1, -1))  # an untried kind outranks a known one


def test_explorer_resets_on_terminal_and_drops_a_stale_plan() -> None:
    pol = ExplorerPolicy()
    g = grid_with((5, 5, 3, 3, 7))
    assert pol.choose(obs(g, phase=Phase.GAME_OVER)) == Action(0)
    assert pol.choose(obs(g, avail=(0,))) == Action(0)
    pol.plan = [Action(1)]
    pol.expected = b"elsewhere"
    pol.choose(obs(g))
    assert pol.plan == []


def test_path_to_frontier_is_shortest() -> None:
    gph = explore.Graph()
    a, b, c, d = b"a", b"b", b"c", b"d"
    gph.edges = {a: {(1, -1, -1): b, (2, -1, -1): c}, b: {(1, -1, -1): d}, c: {(1, -1, -1): d}}
    gph.cands = {
        a: [Action(1), Action(2)],
        b: [Action(1)],
        c: [Action(1)],
        d: [Action(1), Action(3)],
    }
    assert gph.untried(d) == [Action(1), Action(3)]
    assert len(gph.path_to_frontier(a)) == 2
    gph.cands[d] = []
    assert gph.path_to_frontier(a) == []


@pytest.mark.parametrize("policy", ["probe", "explore"])
def test_make_policy_names(policy: str) -> None:
    assert make_policy(policy, 0).choose(obs(grid_with(), phase=Phase.WIN)) == Action(0)


def test_make_policy_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        make_policy("llm", 0)


def test_explorer_exclude_skips_and_falls_back_when_all_excluded() -> None:
    pol = ExplorerPolicy(seed=0)
    o = obs(grid_with((5, 5, 2, 2, 3)), avail=(1, 2))
    banned = frozenset({(1, -1, -1)})
    assert pol.choose(o, banned) == Action(2)
    pol = ExplorerPolicy(seed=0)
    assert pol.choose(o, frozenset({(1, -1, -1), (2, -1, -1)})).id in (1, 2)
    # Default behavior is unchanged: no exclusion, same pick as a fresh policy.
    assert ExplorerPolicy(seed=0).choose(o) == ExplorerPolicy(seed=0).choose(o, frozenset())


def test_explorer_observe_and_note_learn_actions_taken_elsewhere() -> None:
    pol = ExplorerPolicy(seed=0)
    a = obs(grid_with((5, 5, 2, 2, 3)), avail=(1, 2))
    b = obs(grid_with((6, 5, 2, 2, 3)), avail=(1, 2))
    pol.observe(a)
    pol.note(a, Action(1))
    pol.observe(b)
    pol.observe(b)  # idempotent until an action is noted
    assert pol.graph.edges[state_key(a)][(1, -1, -1)] == state_key(b)
    assert pol.tries[(1, -1)] == 1


class CounterEnv:
    """A move counter fills the bottom row on every action; only action 1 moves the block."""

    def __init__(self) -> None:
        self.x = 10
        self.moves = 0

    def observe(self) -> Observation:
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        g[GRID_SIZE - 1, : self.moves % GRID_SIZE] = 9
        g[20:22, self.x : self.x + 2] = 7
        return obs(g, avail=(1, 2))

    def step(self, a: Action) -> Observation:
        self.moves += 1
        if a.id == 1 and self.x < 50:  # room enough that the graph isn't exhausted
            self.x += 2
        return self.observe()


def test_a_move_counter_on_the_edge_is_left_out_of_the_state() -> None:
    env, p = CounterEnv(), ExplorerPolicy(0)
    o = env.observe()
    for _ in range(explore.CLOCK_WARMUP + 4):
        o = env.step(p.choose(o))
    a, b = env.observe(), env.step(Action(2))
    assert not np.array_equal(a.grid, b.grid)
    assert p.key(a) == p.key(b)  # only the counter changed
    assert any((2, -1, -1) in v for v in p.graph.noop.values())  # 2 is seen as a no-op


class HiddenEnv:
    """The screen shows only the block; the level is won after 12 moves in a row with the
    block at x=4 (pressing 1 there changes nothing visible), a count the screen never
    shows. The move counter on the bottom row is the only clue that time passed."""

    def __init__(self) -> None:
        self.x, self.moves, self.since, self.level = 0, 0, -1, 0

    def observe(self) -> Observation:
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        g[GRID_SIZE - 1, : self.moves % GRID_SIZE] = 9
        g[20, 10 + self.x] = 7
        return Observation(g, Phase.NOT_FINISHED, self.level, 3, (1, 2))

    def step(self, a: Action) -> Observation:
        self.moves += 1
        self.x = min(self.x + 1, 4) if a.id == 1 else max(self.x - 1, 0)
        if self.x != 4:
            self.since = -1
        elif self.since < 0:
            self.since = self.moves
        if self.since >= 0 and self.moves - self.since >= 12:
            self.level += 1
        return self.observe()


def test_an_exhausted_graph_starts_a_round_that_keys_on_the_clock() -> None:
    env, p = HiddenEnv(), ExplorerPolicy(0)
    o = env.observe()
    for _ in range(2000):
        if env.level:
            break
        o = env.step(p.choose(o))
    assert env.level == 1


def test_a_move_that_ended_the_game_is_never_retried_or_routed_through() -> None:
    p = ExplorerPolicy(0)
    g = grid_with((5, 5, 3, 3, 9), (20, 20, 3, 3, 4))
    over = obs(grid_with((0, 0, 4, 4, 2)), phase=Phase.GAME_OVER, avail=(6,))
    first = p.choose(obs(g, avail=(6,)))
    assert p.choose(over).id == 0  # reset
    p.graph.cands[p.key(obs(g, avail=(6,)))] = [first]  # the same move ends it again
    assert p.choose(obs(g, avail=(6,))) == first
    assert p.choose(over).id == 0
    p.graph.cands[p.key(obs(g, avail=(6,)))] = p._candidates(obs(g, avail=(6,)))  # pyright: ignore[reportPrivateUsage]
    assert p.graph.cands.get(p.key(over), []) == []
    assert first.key() in p.graph.dead[p.key(obs(g, avail=(6,)))]
    assert p.key(over) not in p.graph.edges.get(p.key(obs(g, avail=(6,))), {}).values()
    nxt = p.choose(obs(g, avail=(6,)))
    assert nxt.key() != first.key()


def test_untried_moves_of_an_idle_kind_wait() -> None:
    p = ExplorerPolicy(0)
    g = grid_with((5, 5, 3, 3, 9), (20, 20, 3, 3, 4))
    p.tries[(6, 9)], p.changes[(6, 9)] = 40, 1
    assert p.idle((6, 9)) and not p.idle((6, 4))
    a = p.choose(obs(g, avail=(6,)))
    assert g[a.y, a.x] == 4
