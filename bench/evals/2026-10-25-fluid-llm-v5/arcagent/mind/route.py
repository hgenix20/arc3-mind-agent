"""The shortest route from a start through every target, in some order.

``route(start, targets, dist)`` returns the visiting order (indices into ``targets``) and
its total length, or None when some leg has no distance (``dist`` returns None: a target
that can't be reached). Up to ``exact_limit`` targets (default ``ROUTE_EXACT``) it is exact
(Held-Karp dynamic programming over subsets, O(n^2 2^n)); above that it starts from the
nearest-neighbor order and improves it with up to ``TWO_OPT_PASSES`` passes of 2-opt
(reversing a stretch of the order when that shortens it). The route ends at its last
target: nothing is gained by coming back. Pure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise

Cell = tuple[int, int]
Dist = Callable[[Cell, Cell], int | None]

ROUTE_EXACT = 6  # most targets routed exactly
TWO_OPT_PASSES = 4  # improvement passes for bigger target sets


def route(
    start: Cell, targets: Sequence[Cell], dist: Dist, *, exact_limit: int = ROUTE_EXACT
) -> tuple[list[int], int] | None:
    """Visiting order and total length, or None when a needed leg is unreachable."""
    n = len(targets)
    if n == 0:
        return [], 0
    first = [dist(start, t) for t in targets]
    legs = [[dist(a, b) if a != b else 0 for b in targets] for a in targets]
    if any(d is None for d in first) or any(d is None for row in legs for d in row):
        return None
    f = [d or 0 for d in first]
    g = [[d or 0 for d in row] for row in legs]
    order = _exact(f, g) if n <= exact_limit else _two_opt(_nearest(f, g), f, g)
    return order, _length(order, f, g)


def _length(order: list[int], f: list[int], g: list[list[int]]) -> int:
    total = f[order[0]]
    for a, b in pairwise(order):
        total += g[a][b]
    return total


def _exact(f: list[int], g: list[list[int]]) -> list[int]:
    """Held-Karp: best[mask][j] = shortest path from start over ``mask`` ending at j."""
    n = len(f)
    full = (1 << n) - 1
    inf = float("inf")
    best: list[list[float]] = [[inf] * n for _ in range(full + 1)]
    back: list[list[int]] = [[-1] * n for _ in range(full + 1)]
    for j in range(n):
        best[1 << j][j] = f[j]
    for mask in range(1, full + 1):
        for j in range(n):
            here = best[mask][j]
            if here == inf or not mask & (1 << j):
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nxt = mask | (1 << k)
                cand = here + g[j][k]
                if cand < best[nxt][k]:
                    best[nxt][k] = cand
                    back[nxt][k] = j
    end = min(range(n), key=lambda j: best[full][j])
    order: list[int] = []
    mask, j = full, end
    while j != -1:
        order.append(j)
        prev = back[mask][j]
        mask ^= 1 << j
        j = prev
    return order[::-1]


def _nearest(f: list[int], g: list[list[int]]) -> list[int]:
    left = set(range(len(f)))
    here = min(left, key=lambda j: (f[j], j))
    order = [here]
    left.discard(here)
    while left:
        here = min(left, key=lambda j: (g[here][j], j))
        order.append(here)
        left.discard(here)
    return order


def _two_opt(order: list[int], f: list[int], g: list[list[int]]) -> list[int]:
    best, length = order, _length(order, f, g)
    for _ in range(TWO_OPT_PASSES):
        improved = False
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cand_len = _length(cand, f, g)
                if cand_len < length:
                    best, length, improved = cand, cand_len, True
        if not improved:
            break
    return best
