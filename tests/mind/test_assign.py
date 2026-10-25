"""Tests for the Hungarian assignment used by the tracker."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from arcagent.mind.assign import ASSIGN_MAX, assign

INF = math.inf


def brute(cost: list[list[float]], gate: float) -> tuple[int, float]:
    """(pairs, total) of the best assignment: most pairs within the gate, then least cost."""
    rows, cols = len(cost), len(cost[0]) if cost else 0
    best = (0, 0.0)
    for k in range(min(rows, cols), 0, -1):
        totals = [
            sum(cost[i][j] for i, j in zip(rs, cs, strict=True))
            for rs in itertools.combinations(range(rows), k)
            for cs in itertools.permutations(range(cols), k)
            if all(cost[i][j] <= gate for i, j in zip(rs, cs, strict=True))
        ]
        if totals:
            return k, min(totals)
    return best


def score(cost: list[list[float]], pairs: list[tuple[int, int]]) -> tuple[int, float]:
    return len(pairs), sum(cost[i][j] for i, j in pairs)


def test_square_matrix_optimum() -> None:
    pairs = assign([[1, 2, 9], [2, 9, 9], [9, 9, 1]], INF)
    assert set(pairs) == {(0, 1), (1, 0), (2, 2)}  # greedy would take (0, 0) and pay 11


def test_rectangular_leaves_extra_unmatched() -> None:
    wide = assign([[5, 1, 7], [2, 8, 3]], INF)
    assert set(wide) == {(0, 1), (1, 0)}
    tall = assign([[5, 2], [1, 8], [7, 3]], INF)
    assert set(tall) == {(0, 1), (1, 0)}  # 2 + 1 beats 1 + 3


def test_gate_drops_expensive_pairs() -> None:
    assert assign([[1, 50], [50, 50]], 10) == [(0, 0)]
    assert assign([[50]], 10) == []


def test_gate_prefers_more_pairs_over_a_cheaper_single() -> None:
    # (0, 0) alone costs 1; two pairs within the gate cost 12 but match more things.
    assert set(assign([[1, 6], [6, 30]], 10)) == {(0, 1), (1, 0)}


def test_empty_matrix() -> None:
    assert assign([], INF) == []
    assert assign([[], []], INF) == []


def test_ragged_matrix_raises() -> None:
    with pytest.raises(ValueError, match="rows"):
        assign([[1, 2], [3]], INF)


def test_size_cap_raises() -> None:
    big = [[0.0] * (ASSIGN_MAX + 1) for _ in range(2)]
    with pytest.raises(ValueError, match="ASSIGN_MAX"):
        assign(big, INF)


def test_pairs_come_back_sorted_by_row() -> None:
    pairs = assign([[3, 1, 2], [1, 3, 2], [2, 1, 3]], INF)
    assert pairs == sorted(pairs)


def test_matches_brute_force_on_random_matrices() -> None:
    rng = random.Random(7)
    for _ in range(100):
        rows, cols = rng.randint(1, 5), rng.randint(1, 5)
        cost = [[float(rng.randint(0, 20)) for _ in range(cols)] for _ in range(rows)]
        gate = rng.choice([INF, 10.0, 5.0])
        pairs = assign(cost, gate)
        assert len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})
        assert all(cost[i][j] <= gate for i, j in pairs)
        got = score(cost, pairs)
        assert got[0] == brute(cost, gate)[0]
        assert got[1] == pytest.approx(brute(cost, gate)[1])
