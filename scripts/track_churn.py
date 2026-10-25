"""Replay recorded frames through the tracker and count track churn.

Why: a tracker that loses a thing (a sprite turning, a recolor while moving) reports it as
one disappearance plus one appearance, and every fact about it starts over. This counts
those per move so two trackers can be compared on the same recording, with no agent in the
loop (so no prediction prior: it measures the matching rules alone).

Input: an ``.npz`` with ``grids`` (frames, 64, 64), ``lvl`` (level per frame) and
``phase`` (per frame), as written by the play-observation tools. The tracker resets on a
level change and after a game over (phase 2), as the agent does.

    python scripts/track_churn.py <recording.npz> [...]

Prints, per file: moves, appeared + disappeared per move, and moves with any churn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from arcagent.mind.objects import Tracker

GAME_OVER_PHASE = 2


def churn(path: Path) -> tuple[int, int, int]:
    """(moves, appeared + disappeared changes, moves with any)."""
    z = np.load(path)
    grids, lvl, phase = z["grids"], z["lvl"], z["phase"]
    t = Tracker()
    moves = total = busy = 0
    for i, g in enumerate(grids):
        if i > 0 and (lvl[i] != lvl[i - 1] or phase[i - 1] == GAME_OVER_PHASE):
            t.reset()
        _, diff = t.update(g.astype(np.int8), frozenset())
        if diff is None:
            continue
        n = sum(1 for c in diff.changes if c.kind in ("appeared", "disappeared"))
        moves += 1
        total += n
        busy += n > 0
    return moves, total, busy


def main(argv: list[str]) -> int:
    for arg in argv:
        moves, total, busy = churn(Path(arg))
        rate = total / moves if moves else 0.0
        print(
            f"{Path(arg).stem:6} moves {moves:4}  churn/move {rate:6.2f}  moves with churn {busy}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
