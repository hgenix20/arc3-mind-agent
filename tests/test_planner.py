"""Search over a model-written world model (arcagent.plan.planner)."""

from typing import Any

from arcagent.plan import planner

MOVES = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
WALLS = {(1, 0), (1, 1), (1, 2), (3, 2), (3, 3), (3, 4)}


def grid_step(s: dict[str, Any], a: str) -> dict[str, Any] | None:
    x, y = s["pos"]
    dx, dy = MOVES[a]
    nx, ny = x + dx, y + dy
    if not (0 <= nx < 5 and 0 <= ny < 5) or (nx, ny) in WALLS:
        return {"pos": [x, y]}  # bumping a wall wastes the action, like ARC games
    return {"pos": [nx, ny]}


def at_goal(s: dict[str, Any]) -> bool:
    return s["pos"] == [4, 0]


def test_bfs_finds_shortest_path_through_maze() -> None:
    r = planner.plan(grid_step, {"pos": [0, 0]}, at_goal, list("UDLR"))
    assert r.stopped == "goal"
    assert r.actions is not None and len(r.actions) == 10
    s: dict[str, Any] = {"pos": [0, 0]}
    for a in r.actions:
        s = grid_step(s, a)  # type: ignore[assignment]
    assert at_goal(s)


def test_astar_matches_bfs_length_with_admissible_heuristic() -> None:
    def h(s: dict[str, Any]) -> float:
        return abs(s["pos"][0] - 4) + abs(s["pos"][1])

    r = planner.plan(grid_step, {"pos": [0, 0]}, at_goal, list("UDLR"), heuristic=h)
    assert r.actions is not None and len(r.actions) == 10


def test_goal_at_start_returns_empty_plan() -> None:
    r = planner.plan(grid_step, {"pos": [4, 0]}, at_goal, list("UDLR"))
    assert r.actions == [] and r.stopped == "goal_at_start"


def test_unreachable_goal_exhausts_with_no_plan() -> None:
    r = planner.plan(grid_step, {"pos": [0, 0]}, lambda s: s["pos"] == [9, 9], list("UDLR"))
    assert r.actions is None and r.stopped == "exhausted"


def test_model_errors_prune_branch_and_are_reported() -> None:
    def buggy(s: dict[str, Any], a: str) -> dict[str, Any] | None:
        if a == "L":
            raise KeyError("left")
        return grid_step(s, a)

    r = planner.plan(buggy, {"pos": [0, 0]}, at_goal, list("UDLR"))
    assert r.actions is not None and "L" not in r.actions
    assert r.errors > 0 and "KeyError" in r.first_error


def test_none_from_step_prunes_branch() -> None:
    r = planner.plan(lambda s, a: None, {"pos": [0, 0]}, at_goal, list("UDLR"))
    assert r.actions is None and r.stopped == "exhausted"


def test_node_budget_stops_search_and_reports_partial() -> None:
    def step(s: int, a: int) -> int:
        return s * 2 + a  # infinite binary tree, goal never reached

    r = planner.plan(step, 1, lambda s: False, [0, 1], max_nodes=50)
    assert r.stopped == "max_nodes" and r.expanded == 50 and r.actions is None


def test_time_budget_stops_search() -> None:
    r = planner.plan(lambda s, a: s * 2 + a, 1, lambda s: False, [0, 1], time_limit_s=0.0)
    assert r.stopped == "time_limit"


def test_state_dependent_action_list() -> None:
    def acts(s: dict[str, Any]) -> list[str]:
        return ["R", "U"] if s["pos"][0] < 4 else ["U"]

    r = planner.plan(grid_step, {"pos": [0, 3]}, lambda s: s["pos"] == [2, 0], acts)
    assert r.actions is not None and len(r.actions) == 5
    assert set(r.actions) <= {"R", "U"}


def test_freeze_distinguishes_and_equates_structures() -> None:
    f = planner.freeze
    assert f([[1, 2], [3]]) == f([[1, 2], [3]])
    assert f([[1, 2], [3]]) != f([[1], [2, 3]])
    assert f({"a": 1, "b": [2]}) == f({"b": [2], "a": 1})
    assert f({1, 2}) == f({2, 1})


def test_freeze_numpy_like_object() -> None:
    class Arr:
        shape = (2, 2)
        dtype = "int8"

        def __init__(self, b: bytes) -> None:
            self.b = b

        def tobytes(self) -> bytes:
            return self.b

    assert planner.freeze(Arr(b"\x00\x01")) == planner.freeze(Arr(b"\x00\x01"))
    assert planner.freeze(Arr(b"\x00\x01")) != planner.freeze(Arr(b"\x01\x01"))


def test_summary_mentions_outcome() -> None:
    r = planner.plan(grid_step, {"pos": [0, 0]}, at_goal, list("UDLR"))
    assert "10 actions" in r.summary() and "stopped=goal" in r.summary()
