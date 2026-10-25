"""The move budget: how many moves this life has left, read from a counter bar.

A counter (``noise.NoiseMask`` flags one: a thin edge strip whose size changes the same
way, move after move) is kept here as a ``Counter``: its color, whether it is a row or a
column band, the band's lines, and whether the bar shrinks or grows. ``BudgetReader``
counts the bar's cells across the whole band on every frame (bars seen so far span the
board, and a bar redrawn full on a new life sticks out of the cells the mask flagged).

Which moves count against the budget differs between games: some bars drain on every move
(a slow one skipping cells now and then), others only on moves that do something. The
reader keeps two counts of this life's moves, ``all`` (every move) and ``busy`` (leaving
out moves that changed nothing on screen and left the bar as it was), fits a least-squares
line of bar cells against each, and uses the count whose line fits better (lower mean
squared residual, once both have ``MIN_FIT`` points; ``all`` before that). The line's slope
is the rate, which averages out the whole-cell steps of a bar that drains a fraction of a
cell per move; moves left are the cells left (for a growing bar, the band's free cells)
over the rate, in the chosen count.

``start_life`` takes the first frame of an attempt; ``observe`` each later frame. A game
over with less than ``BUDGET_DEATH_MOVES`` moves' worth of bar left is a budget death: each
count's moves that life become its ``per_life``, and from then on that count's rate is the
bar's starting size over its ``per_life``. ``remaining`` is None until a counter is known
and ``MIN_DRAIN`` cells of it drained this life. ``new_level`` forgets the counter and the
per-life counts: a new level can have its own budget. Pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from arcagent.core.types import GRID_SIZE, Grid

Axis = Literal["row", "col"]
Count = Literal["all", "busy"]
BUDGET_DEATH_MOVES = 2  # a game over with less than this many moves of bar left was the budget
MIN_DRAIN = 6  # bar cells drained this life before the rate is trusted
MIN_FIT = 5  # points each count needs before their fits are compared


@dataclass(frozen=True, slots=True)
class Counter:
    """A counter bar: its color, its band (rows or columns) and its direction."""

    color: int
    axis: Axis
    lines: frozenset[int]
    shrinks: bool


class _Fit:
    """Running least-squares sums of bar cells against one count of moves."""

    def __init__(self) -> None:
        self.moves = 0
        self.n = 0
        self.sm = self.sc = self.smm = self.smc = self.scc = 0.0

    def add(self, cells: int) -> None:
        m = self.moves
        self.n += 1
        self.sm += m
        self.sc += cells
        self.smm += m * m
        self.smc += m * cells
        self.scc += cells * cells

    def slope(self) -> float | None:
        sxx = self.smm - self.sm * self.sm / self.n if self.n else 0.0
        if sxx <= 0:
            return None
        return (self.smc - self.sm * self.sc / self.n) / sxx

    def misfit(self) -> float:
        """Mean squared residual of the line (infinite with too few points)."""
        slope = self.slope()
        if slope is None or self.n < MIN_FIT:
            return float("inf")
        syy = self.scc - self.sc * self.sc / self.n
        sxy = self.smc - self.sm * self.sc / self.n
        return max(0.0, syy - slope * sxy) / self.n


class BudgetReader:
    """Moves left this life from a counter bar (module docstring)."""

    def __init__(self) -> None:
        self._counter: Counter | None = None
        self._first: Grid | None = None
        self._start: int | None = None
        self._count: int | None = None
        self._fits: dict[Count, _Fit] = {"all": _Fit(), "busy": _Fit()}
        self._per_life: dict[Count, int] = {}

    @property
    def counter(self) -> Counter | None:
        return self._counter

    def set_counter(self, counter: Counter) -> None:
        """Use ``counter``; the life's start count is read from its first frame."""
        if counter == self._counter:
            return
        self._counter = counter
        self._per_life = {}
        moves = {name: fit.moves for name, fit in self._fits.items()}
        self._restart(self._first)
        for name, fit in self._fits.items():  # moves made before the counter was known count
            fit.moves = moves[name]

    def new_level(self) -> None:
        """A new level: its counter and budget may differ, so forget both."""
        self.__init__()

    def _restart(self, grid: Grid | None) -> None:
        self._fits = {"all": _Fit(), "busy": _Fit()}
        self._start = self._cells(grid) if grid is not None and self._counter else None
        self._count = self._start
        if self._start is not None:
            for fit in self._fits.values():
                fit.add(self._start)

    def _cells(self, grid: Grid) -> int:
        c = self._counter
        if c is None:
            return 0
        lines = sorted(x for x in c.lines if 0 <= x < GRID_SIZE)
        band = grid[lines, :] if c.axis == "row" else grid[:, lines]
        return int((band == c.color).sum())

    def start_life(self, grid: Grid) -> None:
        """The first frame of an attempt."""
        self._first = grid.copy()
        self._restart(grid)

    def observe(self, grid: Grid, *, changed: bool = True) -> None:
        """A frame after one move of this life; ``changed`` says whether the move changed
        anything on screen besides counters."""
        count = self._cells(grid) if self._counter is not None else None
        busy = changed or count != self._count
        self._count = count
        for name, fit in self._fits.items():
            if name == "all" or busy:
                fit.moves += 1
                if count is not None and self._start is not None:
                    fit.add(count)

    def chosen(self) -> Count:
        """The count whose line fits the bar better (``all`` until both can be compared)."""
        a, b = self._fits["all"].misfit(), self._fits["busy"].misfit()
        return "busy" if b < a else "all"

    def moves(self, count: Count | None = None) -> int:
        """Moves made this life in ``count`` (the chosen one by default)."""
        return self._fits[count or self.chosen()].moves

    def _left_cells(self) -> int | None:
        c, count = self._counter, self._count
        if c is None or count is None:
            return None
        return count if c.shrinks else len(c.lines) * GRID_SIZE - count

    def _rate(self, name: Count) -> float | None:
        """Bar cells per move in count ``name``."""
        c, start, count = self._counter, self._start, self._count
        if c is None or start is None or count is None:
            return None
        full = start if c.shrinks else len(c.lines) * GRID_SIZE - start
        per = self._per_life.get(name)
        if per:
            return full / per
        drained = (start - count) if c.shrinks else (count - start)
        slope = self._fits[name].slope()
        if drained < MIN_DRAIN or slope is None:
            return None
        rate = -slope if c.shrinks else slope
        return rate if rate > 0 else None

    def remaining(self) -> int | None:
        """Moves left this life in the chosen count, or None when unknown."""
        left, rate = self._left_cells(), self._rate(self.chosen())
        if left is None or not rate:
            return None
        return max(0, round(left / rate))

    def per_life(self) -> int | None:
        """Moves a life lasts in the chosen count, once one ended by the budget."""
        return self._per_life.get(self.chosen())

    def game_over(self) -> None:
        """The attempt ended; if the bar was all but spent, the budget ended it."""
        left, rate = self._left_cells(), self._rate(self.chosen())
        if left is not None and rate and left < BUDGET_DEATH_MOVES * rate:
            self._per_life = {name: fit.moves for name, fit in self._fits.items()}
