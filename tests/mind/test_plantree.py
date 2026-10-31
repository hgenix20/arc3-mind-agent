"""Tests for the fork tree and the return route (spec 4.7c and 3.6)."""

from __future__ import annotations

from arcagent.core.types import Action
from arcagent.mind.plantree import (
    MAX_FORKS,
    MAX_PATHS,
    Fork,
    PlanTree,
    return_route,
)

Edges = dict[bytes, dict[tuple[int, int, int], bytes]]


def move(i: int) -> Action:
    return Action(i)


def click(x: int, y: int) -> Action:
    return Action(6, x, y)


UP, DOWN, LEFT, RIGHT = move(1), move(2), move(3), move(4)


def two_forks() -> tuple[PlanTree, str, str]:
    tree = PlanTree()
    first = tree.open_fork(b"s1", "g1", [(UP, UP), (LEFT, DOWN)], [RIGHT])
    second = tree.open_fork(b"s2", "g1", [(DOWN,), (LEFT, LEFT, LEFT)], [RIGHT, UP, UP])
    return tree, first, second


# --- forks -----------------------------------------------------------------------------


def test_open_two_forks_and_the_newest_with_untried_is_the_second() -> None:
    tree, first, second = two_forks()
    assert first != second
    newest = tree.newest_with_untried()
    assert newest is not None
    assert newest.id == second
    assert newest.state_key == b"s2"
    assert newest.goal_id == "g1"
    assert newest.paths == [(DOWN,), (LEFT, LEFT, LEFT)]
    assert newest.status == ["untried", "untried"]
    assert newest.moves_here == (RIGHT, UP, UP)
    older = tree.fork(first)
    assert older is not None
    assert older.created < newest.created
    assert tree.untried(second) == [0, 1]
    assert tree.active() is None


def test_fork_records_keep_the_moves_that_got_there() -> None:
    tree = PlanTree()
    fork_id = tree.open_fork(b"s", "g", [(UP,), (DOWN,)], [click(3, 4), RIGHT, RIGHT])
    fork = tree.fork(fork_id)
    assert isinstance(fork, Fork)
    assert fork.moves_here == (click(3, 4), RIGHT, RIGHT)
    # A later arrival at the same fork keeps the first record of the moves.
    tree.open_fork(b"s", "g", [(LEFT,)], [UP])
    again = tree.fork(fork_id)
    assert again is not None
    assert again.moves_here == (click(3, 4), RIGHT, RIGHT)


def test_starting_a_path_deactivates_the_other_active_path() -> None:
    tree, first, second = two_forks()
    tree.start_path(first, 1)
    assert tree.active() == (first, 1)
    tree.start_path(second, 0)
    assert tree.active() == (second, 0)
    older = tree.fork(first)
    newer = tree.fork(second)
    assert older is not None and newer is not None
    assert older.status == ["untried", "untried"]
    assert newer.status == ["active", "untried"]
    # An active path no longer counts as untried.
    assert tree.untried(second) == [1]
    assert tree.untried(first) == [0, 1]


def test_restarting_the_active_path_keeps_it_active() -> None:
    tree, first, _ = two_forks()
    tree.start_path(first, 0)
    tree.start_path(first, 0)
    assert tree.active() == (first, 0)
    assert tree.misses == 0


def test_failing_all_paths_empties_untried_and_newest_skips_the_fork() -> None:
    tree, first, second = two_forks()
    tree.start_path(second, 0)
    tree.fail_branch(second, 0)
    assert tree.active() is None
    assert tree.untried(second) == [1]
    tree.fail_branch(second, 1)
    assert tree.untried(second) == []
    newest = tree.newest_with_untried()
    assert newest is not None
    assert newest.id == first
    tree.fail_branch(first, 0)
    tree.fail_branch(first, 1)
    assert tree.newest_with_untried() is None
    assert tree.counts() == {"forks": 2, "untried": 0, "failed": 4, "achieved": 0}


def test_a_failed_path_cannot_be_started_again() -> None:
    tree, first, _ = two_forks()
    tree.fail_branch(first, 0)
    tree.start_path(first, 0)
    assert tree.active() is None
    fork = tree.fork(first)
    assert fork is not None
    assert fork.status == ["failed", "untried"]
    assert tree.misses == 1


def test_achieving_a_path_closes_the_fork() -> None:
    tree, first, second = two_forks()
    tree.start_path(second, 0)
    tree.achieve(second, 0)
    assert tree.active() is None
    fork = tree.fork(second)
    assert fork is not None
    assert fork.status == ["achieved", "untried"]
    # The goal is reached, so the sibling path is no longer worth a return trip.
    assert tree.untried(second) == []
    newest = tree.newest_with_untried()
    assert newest is not None
    assert newest.id == first
    assert tree.counts() == {"forks": 2, "untried": 3, "failed": 0, "achieved": 1}
    assert tree.summary() == ["fork f1 at goal g1: 2 untried of 2 paths, 1 moves from start"]


def test_same_state_and_goal_reuse_one_fork_and_add_new_paths_only() -> None:
    tree = PlanTree()
    first = tree.open_fork(b"s", "g", [(UP,), (DOWN,)], [RIGHT])
    tree.fail_branch(first, 0)
    again = tree.open_fork(b"s", "g", [(DOWN,), (LEFT,), (UP,)], [RIGHT, RIGHT])
    assert again == first
    fork = tree.fork(first)
    assert fork is not None
    assert fork.paths == [(UP,), (DOWN,), (LEFT,)]
    assert fork.status == ["failed", "untried", "untried"]
    assert tree.counts()["forks"] == 1
    # The same state toward another goal is a different fork.
    other = tree.open_fork(b"s", "g2", [(UP,)], [RIGHT])
    assert other != first
    assert tree.counts()["forks"] == 2


def test_empty_paths_and_duplicates_are_not_recorded() -> None:
    tree = PlanTree()
    fork_id = tree.open_fork(b"s", "g", [(), (UP,), (UP,), ()], [])
    fork = tree.fork(fork_id)
    assert fork is not None
    assert fork.paths == [(UP,)]
    assert fork.moves_here == ()


def test_paths_per_fork_are_bounded() -> None:
    tree = PlanTree()
    paths = [(click(x, 0),) for x in range(MAX_PATHS + 5)]
    fork_id = tree.open_fork(b"s", "g", paths, [])
    fork = tree.fork(fork_id)
    assert fork is not None
    assert len(fork.paths) == MAX_PATHS
    assert fork.paths == paths[:MAX_PATHS]
    tree.open_fork(b"s", "g", [(click(63, 63),)], [])
    assert len(fork.paths) == MAX_PATHS
    assert tree.untried(fork_id) == list(range(MAX_PATHS))


def test_the_tree_holds_at_most_max_forks_and_drops_the_oldest() -> None:
    tree = PlanTree()
    ids = [tree.open_fork(bytes([i % 256, i // 256]), "g", [(UP,)], []) for i in range(MAX_FORKS)]
    tree.start_path(ids[0], 0)
    assert tree.counts()["forks"] == MAX_FORKS
    extra = tree.open_fork(b"extra", "g", [(UP,)], [])
    assert tree.counts()["forks"] == MAX_FORKS
    assert tree.fork(ids[0]) is None
    assert tree.fork(ids[1]) is not None
    assert tree.fork(extra) is not None
    # The dropped fork held the active path, so nothing is active now.
    assert tree.active() is None
    assert tree.misses == 1


def test_unknown_fork_ids_and_bad_indexes_never_raise_and_count_as_misses() -> None:
    tree, first, _ = two_forks()
    assert tree.fork("nope") is None
    assert tree.untried("nope") == []
    tree.start_path("nope", 0)
    tree.fail_branch("nope", 0)
    tree.achieve("nope", 0)
    tree.start_path(first, 2)
    tree.fail_branch(first, -1)
    tree.achieve(first, 99)
    assert tree.misses == 8
    assert tree.active() is None
    fork = tree.fork(first)
    assert fork is not None
    assert fork.status == ["untried", "untried"]


def test_reset_level_empties_everything() -> None:
    tree, first, second = two_forks()
    tree.start_path(second, 1)
    tree.fail_branch(first, 0)
    tree.reset_level()
    assert tree.counts() == {"forks": 0, "untried": 0, "failed": 0, "achieved": 0}
    assert tree.active() is None
    assert tree.newest_with_untried() is None
    assert tree.summary() == []
    assert tree.fork(first) is None
    assert tree.fork(second) is None
    assert tree.untried(second) == []
    # A fork opened after the reset gets a fresh id and a fresh record.
    fresh = tree.open_fork(b"s1", "g1", [(UP,)], [])
    assert fresh not in {first, second}
    assert tree.counts()["forks"] == 1


def test_summary_lines_are_bounded_and_carry_the_counts() -> None:
    tree = PlanTree()
    ids = [
        tree.open_fork(bytes([i]), f"g{i}", [(UP,), (DOWN,), (LEFT,)], [RIGHT] * i)
        for i in range(8)
    ]
    tree.fail_branch(ids[7], 0)
    tree.start_path(ids[7], 1)
    lines = tree.summary()
    assert len(lines) == 5
    assert lines[0] == f"fork {ids[7]} at goal g7: 1 untried of 3 paths, 7 moves from start"
    assert lines[1] == f"fork {ids[6]} at goal g6: 3 untried of 3 paths, 6 moves from start"
    assert tree.summary(limit=2) == lines[:2]
    assert tree.summary(limit=0) == []
    assert tree.summary(limit=-3) == []
    assert len(tree.summary(limit=100)) == 8
    # Closed forks (every path failed, or the goal achieved) are left out.
    tree.achieve(ids[6], 2)
    for i in range(3):
        tree.fail_branch(ids[5], i)
    lines = tree.summary(limit=100)
    assert len(lines) == 6
    assert not any(" g6:" in line or " g5:" in line for line in lines)


def test_counts_tally_every_path_status() -> None:
    tree, first, second = two_forks()
    tree.start_path(first, 0)
    tree.fail_branch(second, 1)
    tree.achieve(second, 0)
    assert tree.counts() == {"forks": 2, "untried": 1, "failed": 1, "achieved": 1}


# --- return route ----------------------------------------------------------------------


def chain(n: int) -> Edges:
    """s0 -UP-> s1 -UP-> ... -UP-> s(n-1)."""
    return {f"s{i}".encode(): {UP.key(): f"s{i + 1}".encode()} for i in range(n - 1)}


def test_return_route_finds_the_shortest_action_list() -> None:
    edges: Edges = {
        b"s0": {LEFT.key(): b"s2", UP.key(): b"s1"},
        b"s1": {DOWN.key(): b"s3"},
        b"s2": {UP.key(): b"s4"},
        b"s4": {RIGHT.key(): b"s3"},
        b"s3": {click(5, 6).key(): b"s5"},
    }
    assert return_route(edges, b"s0", b"s3") == [UP, DOWN]
    assert return_route(edges, b"s0", b"s5") == [UP, DOWN, click(5, 6)]
    assert return_route(edges, b"s2", b"s3") == [UP, RIGHT]
    assert return_route(edges, b"s0", b"s4") == [LEFT, UP]


def test_return_route_is_empty_for_the_same_state() -> None:
    assert return_route(chain(3), b"s1", b"s1") == []
    assert return_route({}, b"nowhere", b"nowhere") == []
    assert return_route(chain(3), b"s0", b"s0", limit=0) == []


def test_return_route_is_none_when_unreachable() -> None:
    edges = chain(3)
    assert return_route(edges, b"s2", b"s0") is None
    assert return_route(edges, b"s0", b"missing") is None
    assert return_route({}, b"a", b"b") is None


def test_return_route_respects_the_visit_limit() -> None:
    edges = chain(6)
    found = return_route(edges, b"s0", b"s5", limit=6)
    assert found == [UP] * 5
    assert return_route(edges, b"s0", b"s5", limit=5) is None
    assert return_route(edges, b"s0", b"s5", limit=1) is None
    assert return_route(edges, b"s0", b"s1", limit=1) is None
    assert return_route(edges, b"s0", b"s1", limit=2) == [UP]


def test_return_route_breaks_ties_in_sorted_key_order() -> None:
    # Both routes take two moves; insertion order favors LEFT, sorted order favors UP.
    edges: Edges = {
        b"s0": {LEFT.key(): b"a", UP.key(): b"b"},
        b"a": {UP.key(): b"goal"},
        b"b": {UP.key(): b"goal"},
    }
    assert return_route(edges, b"s0", b"goal") == [UP, UP]
    # Clicks sort after the simple moves and among themselves by x then y.
    edges2: Edges = {
        b"s0": {click(9, 0).key(): b"a", click(2, 7).key(): b"b", move(7).key(): b"c"},
        b"a": {UP.key(): b"goal"},
        b"b": {UP.key(): b"goal"},
        b"c": {UP.key(): b"goal"},
    }
    assert return_route(edges2, b"s0", b"goal") == [click(2, 7), UP]


def test_return_route_rebuilds_coordinate_actions() -> None:
    edges: Edges = {b"s0": {(6, 12, 40): b"s1"}, b"s1": {(3, -1, -1): b"s2"}}
    found = return_route(edges, b"s0", b"s2")
    assert found == [Action(6, 12, 40), Action(3)]
    assert found is not None
    assert found[0].x == 12 and found[0].y == 40
