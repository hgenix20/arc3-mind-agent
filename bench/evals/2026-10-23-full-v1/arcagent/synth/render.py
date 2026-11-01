"""Compact text renderings of grids and transitions for the synthesis prompt.

A raw 64x64 grid is 4096 symbols. These renderings keep the information
exact (a full rendering decodes back to the same grid) at a fraction of the
size.

Conventions (shared with the prompt's system message):
- Coordinates are ``x`` = column, ``y`` = row, so a cell is ``grid[y][x]``.
- Colors print as one hex digit ``0``-``f``; padding (-1) prints as ``.``.
- A full grid renders as a header ``crop x=<x0> y=<y0> w=<w> h=<h>`` and one
  line per run of identical rows: ``<y0>[-<y1>]: <tokens>``, where each token
  is ``<c>`` (one cell) or ``<c>*<n>`` (n cells of color c). Row numbers are
  absolute grid rows. Cells outside the crop are padding.
"""

from __future__ import annotations

import numpy as np

from arcagent.synth.trace import Transition

DIFF_CAP = 60
_ACTION_NAMES = {0: "RESET"}


def _sym(v: int) -> str:
    return "." if v < 0 else format(v, "x")


def crop(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return ``(sub, x0, y0)``: the bounding box of non -1 cells.

    An all-padding grid returns an empty ``(0, 0)`` array at ``(0, 0)``.
    """
    ys, xs = np.nonzero(grid >= 0)
    if ys.size == 0:
        return grid[:0, :0], 0, 0
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return grid[y0 : y1 + 1, x0 : x1 + 1], x0, y0


def rle_row(row: np.ndarray) -> str:
    """Run-length encode one row as space-separated ``c`` / ``c*n`` tokens."""
    tokens: list[str] = []
    vals = [int(v) for v in row]
    i = 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        n = j - i
        tokens.append(_sym(vals[i]) if n == 1 else f"{_sym(vals[i])}*{n}")
        i = j
    return " ".join(tokens)


def render_grid(grid: np.ndarray) -> str:
    """Exact, compact rendering of a grid (see module docstring for the format)."""
    sub, x0, y0 = crop(grid)
    h, w = sub.shape
    lines = [f"crop x={x0} y={y0} w={w} h={h}"]
    r = 0
    while r < h:
        s = r
        while s + 1 < h and np.array_equal(sub[s + 1], sub[r]):
            s += 1
        label = f"{y0 + r}" if s == r else f"{y0 + r}-{y0 + s}"
        lines.append(f"{label}: {rle_row(sub[r])}")
        r = s + 1
    return "\n".join(lines)


def parse_grid(text: str, size: int = 64) -> np.ndarray:
    """Inverse of ``render_grid`` (used by tests to prove the format is exact)."""
    lines = text.strip().splitlines()
    head = dict(kv.split("=") for kv in lines[0].split()[1:])
    x0, w = int(head["x"]), int(head["w"])  # row labels are absolute, so y is unused
    g = np.full((size, size), -1, dtype=np.int8)
    for line in lines[1:]:
        label, body = line.split(": ", 1)
        a, _, b = label.partition("-")
        rows = range(int(a), int(b or a) + 1)
        cells: list[int] = []
        for tok in body.split():
            c, _, n = tok.partition("*")
            cells.extend([-1 if c == "." else int(c, 16)] * int(n or 1))
        if len(cells) != w:
            raise ValueError(f"row {label} has {len(cells)} cells, expected {w}")
        for y in rows:
            g[y, x0 : x0 + w] = cells
    return g


def action_name(action_id: int, x: int = -1, y: int = -1) -> str:
    """``RESET``, ``ACTION3`` or ``ACTION6(x=.., y=..)``."""
    if action_id == 6:
        return f"ACTION6(x={x}, y={y})"
    return _ACTION_NAMES.get(action_id, f"ACTION{action_id}")


def render_diff(before: np.ndarray, after: np.ndarray, cap: int = DIFF_CAP) -> str:
    """Changed cells as ``(x,y) old->new``, row-major, at most ``cap`` then a count."""
    ys, xs = np.nonzero(before != after)
    if ys.size == 0:
        return "no cells changed"
    parts = [
        f"({int(x)},{int(y)}) {_sym(int(before[y, x]))}->{_sym(int(after[y, x]))}"
        for y, x in zip(ys[:cap], xs[:cap], strict=True)
    ]
    out = "; ".join(parts)
    if ys.size > cap:
        out += f"; ... and {int(ys.size) - cap} more"
    return out


def summary_line(t: Transition, index: int | None = None) -> str:
    """One line: index, action, cells changed, level change, resulting phase."""
    changed = int(np.count_nonzero(t.grid_before != t.grid_after))
    a = t.action
    idx = t.step if index is None else index
    return (
        f"#{idx} {action_name(a.id, a.x, a.y)} changed={changed} "
        f"level {t.levels_before}->{t.levels_after} phase={t.phase_after}"
    )


def render_transition(t: Transition, mode: str = "diff", index: int | None = None) -> str:
    """Summary line plus either the cell diff (``diff``) or the full next grid (``full``)."""
    head = summary_line(t, index)
    if mode == "diff":
        return f"{head}\n{render_diff(t.grid_before, t.grid_after)}"
    if mode == "full":
        return f"{head}\nnext grid:\n{render_grid(t.grid_after)}"
    raise ValueError(f"mode must be 'diff' or 'full', got {mode!r}")
