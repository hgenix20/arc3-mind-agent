"""Tests for the shortest route over targets."""

from __future__ import annotations

import itertools
import random

from arcagent.mind.route import ROUTE_EXACT, route

Cell = tuple[int, int]


def manhattan(a: Cell, b: Cell) -> int | None:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def brute(start: Cell, targets: list[Cell]) -> int:
    best = None
    for order in itertools.permutations(range(len(targets))):
        here, total = start, 0
        for i in order:
            d = manhattan(here, targets[i])
            assert d is not None
            total += d
            here = targets[i]
        best = total if best is None else min(best, total)
    return best or 0


def test_exact_route_beats_nearest_neighbor() -> None:
    # Greedy goes to the near target at 1 first, then must cross back past the start.
    start, targets = (5, 0), [(4, 0), (8, 0), (9, 0), (10, 0)]
    found = route(start, targets, manhattan)
    assert found is not None and found[1] == brute(start, targets) == 7
    greedy = route(start, targets, manhattan, exact_limit=0)
    assert greedy is not None and greedy[1] >= found[1]


def test_route_matches_brute_force_on_six_targets() -> None:
    rng = random.Random(3)
    for _ in range(20):
        start = (rng.randint(0, 20), rng.randint(0, 20))
        targets = [(rng.randint(0, 20), rng.randint(0, 20)) for _ in range(6)]
        found = route(start, targets, manhattan)
        assert found is not None and found[1] == brute(start, targets)
        assert sorted(found[0]) == list(range(6))
    assert ROUTE_EXACT == 6


def test_large_route_uses_the_heuristic_within_bounds() -> None:
    rng = random.Random(5)
    start = (0, 0)
    targets = [(rng.randint(0, 60), rng.randint(0, 60)) for _ in range(15)]
    found = route(start, targets, manhattan)
    assert found is not None and sorted(found[0]) == list(range(15))
    lower = max(manhattan(start, t) or 0 for t in targets)
    assert lower <= found[1] <= 15 * 120


def test_unreachable_target_is_none_and_no_targets_is_free() -> None:
    def walled(a: Cell, b: Cell) -> int | None:
        return None if b == (9, 9) or a == (9, 9) else manhattan(a, b)

    assert route((0, 0), [(1, 1), (9, 9)], walled) is None
    assert route((0, 0), [], manhattan) == ([], 0)
