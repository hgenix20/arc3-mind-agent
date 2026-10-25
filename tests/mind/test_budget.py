"""Tests for reading the move budget from a counter bar."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.budget import BudgetReader, Counter

BAR = 7


def bar_frame(cells: int, color: int = BAR, row: int = 0, fill: int = 0) -> Grid:
    """A one-row bar of ``cells`` cells along the top edge, the rest of the row ``fill``."""
    g = np.zeros((64, 64), dtype=np.int8)
    g[row, :] = fill
    g[row, :cells] = color
    return g


def run(reader: BudgetReader, counts: list[int], **kw: int) -> list[int | None]:
    reader.start_life(bar_frame(counts[0], **kw))
    out: list[int | None] = []
    for c in counts[1:]:
        reader.observe(bar_frame(c, **kw))
        out.append(reader.remaining())
    return out


def drain(full: int, budget: int) -> list[int]:
    """The bar after each move of a life: ``full`` cells spread over ``budget`` moves."""
    return [round(full * (budget - m) / budget) for m in range(budget + 1)]


def test_shrinking_bar_gives_moves_left() -> None:
    reader = BudgetReader()
    reader.set_counter(Counter(BAR, "row", frozenset({0}), shrinks=True))
    counts = drain(64, 50)
    got = run(reader, counts)
    for move, est in enumerate(got, start=1):
        if move >= 5:  # six cells drained: trusted, within two moves, then within one
            assert est is not None and abs(est - (50 - move)) <= (2 if move < 8 else 1), (move, est)
        elif move < 4:
            assert est is None


def test_growing_bar_gives_moves_left() -> None:
    reader = BudgetReader()
    reader.set_counter(Counter(4, "row", frozenset({0}), shrinks=False))
    counts = [64 - c for c in drain(64, 50)]
    got = run(reader, counts, color=4, fill=7)
    assert got[20] is not None and abs(got[20] - (50 - 21)) <= 1


def test_bar_that_ticks_every_third_move() -> None:
    reader = BudgetReader()
    reader.set_counter(Counter(BAR, "row", frozenset({0}), shrinks=True))
    got = run(reader, drain(64, 200)[:121])
    assert got[10] is None  # three cells drained: not trusted yet
    assert all(e is not None and abs(e - (200 - m)) <= 2 for m, e in enumerate(got, 1) if m >= 40)


def test_no_counter_means_unknown() -> None:
    reader = BudgetReader()
    reader.start_life(bar_frame(64))
    reader.observe(bar_frame(63))
    assert reader.remaining() is None and reader.per_life() is None


def test_no_drain_yet_means_unknown() -> None:
    reader = BudgetReader()
    reader.set_counter(Counter(BAR, "row", frozenset({0}), shrinks=True))
    assert run(reader, [64, 64, 64, 63]) == [None, None, None]


def test_per_life_learned_at_budget_game_over_makes_the_rate_exact() -> None:
    reader = BudgetReader()
    reader.set_counter(Counter(BAR, "row", frozenset({0}), shrinks=True))
    run(reader, drain(64, 50))
    reader.game_over()
    assert reader.per_life() == 50
    got = run(reader, drain(64, 50)[:3])
    assert got == [49, 48]


def test_game_over_with_bar_left_is_not_a_budget_death() -> None:
    reader = BudgetReader()
    reader.set_counter(Counter(BAR, "row", frozenset({0}), shrinks=True))
    run(reader, drain(64, 50)[:20])
    reader.game_over()
    assert reader.per_life() is None
