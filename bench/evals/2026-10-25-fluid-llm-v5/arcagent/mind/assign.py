"""Hungarian assignment: pair rows with columns at least total cost, within a gate.

``assign(cost, gate)`` returns ``(row, col)`` pairs, sorted by row. A pair whose cost
exceeds ``gate`` (or isn't finite) is never returned. Among assignments of allowed pairs
it picks the one with the most pairs, then the least total cost: a tracker would rather
explain two things with two plausible matches than one cheap match and an unexplained
appearance. Rows or columns left over stay unmatched.

Kuhn-Munkres with potentials, O(n² m) for n ≤ m (the matrix is transposed when it has
more rows than columns). Pure Python: the agent depends on numpy only, and the tracker
calls this on the few things left after exact matches, so sizes stay small. Bounded by
``ASSIGN_MAX`` per side; a bigger matrix raises ``ValueError`` and the caller falls back.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

ASSIGN_MAX = 40  # most rows or columns per call


def assign(cost: Sequence[Sequence[float]], gate: float) -> list[tuple[int, int]]:
    """Most pairs with cost at most ``gate``, then least total cost; pairs sorted by row."""
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if any(len(r) != cols for r in cost):
        raise ValueError("every row of the cost matrix needs the same length (ragged rows)")
    if rows > ASSIGN_MAX or cols > ASSIGN_MAX:
        raise ValueError(f"cost matrix {rows}x{cols} is over ASSIGN_MAX={ASSIGN_MAX}")
    if rows == 0 or cols == 0:
        return []
    allowed = [[math.isfinite(c) and c <= gate for c in r] for r in cost]
    finite = [
        c for r, ok in zip(cost, allowed, strict=True) for c, a in zip(r, ok, strict=True) if a
    ]
    # Any one forbidden pair must cost more than every allowed assignment together.
    span = max((abs(c) for c in finite), default=0.0)
    forbid = (span + 1.0) * (2 * min(rows, cols) + 1)
    grid = [
        [c if a else forbid for c, a in zip(r, ok, strict=True)]
        for r, ok in zip(cost, allowed, strict=True)
    ]
    flip = rows > cols
    if flip:
        grid = [list(col) for col in zip(*grid, strict=True)]
    pairs = _hungarian(grid)
    if flip:
        pairs = [(j, i) for i, j in pairs]
    return sorted((i, j) for i, j in pairs if allowed[i][j])


def _hungarian(a: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment of every row of an n x m matrix with n <= m."""
    n, m = len(a), len(a[0])
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j]: the row (1-based) matched to column j, 0 for none
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
