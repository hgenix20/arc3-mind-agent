"""Replay recorded games through the budget reader and compare its estimates with the truth.

Why: ``budget.BudgetReader`` turns a counter bar into "moves left this life"; this checks
that number against the moves each life really had left before its game over, on
recordings, with no agent in the loop. Counters come from ``noise.NoiseMask`` exactly as
in the agent (the first shrinking counter, else the first counter).

Input: ``.npz`` recordings with ``grids`` (frames, 64, 64), ``acts`` (move, 3: id, x, y;
id 0 is RESET), ``phase`` (per frame, ``"GAME_OVER"`` marks a death) and ``lvl``.

    python scripts/budget_check.py <recording.npz> [...]

Each estimate is in the count of moves the reader chose when it made it (every move, or
moves that did something), and its truth is measured in that same count. Only lives that
ended with the bar spent (the reader estimates at most one move left at
the death) are scored: other deaths came from the game, not the budget. Prints per file:
budget deaths and all deaths, how many moves had an estimate, the mean and largest
absolute error in moves, and the learned moves per life.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from arcagent.mind.budget import BudgetReader, Count
from arcagent.mind.noise import NoiseMask
from arcagent.mind.objects import Scene, Tracker

GAME_OVER = "GAME_OVER"
RESET = 0


def check(path: Path) -> tuple[int, int, int, float, int, int | None]:
    """(budget deaths, deaths, estimated moves, mean abs error, max error, per life)."""
    z = np.load(path)
    grids, acts, phase, lvl = z["grids"], z["acts"], z["phase"], z["lvl"]
    tracker, noise, reader = Tracker(), NoiseMask(), BudgetReader()
    scene: Scene = tracker.update(grids[0].astype(np.int8), noise.cells)[0]
    reader.start_life(grids[0])
    life: list[tuple[Count, int, int]] = []  # (count, moves in it, estimate) this life
    errors: list[int] = []
    deaths = spent = 0
    for m, act in enumerate(acts):
        nxt = grids[m + 1].astype(np.int8)
        fresh = int(act[0]) == RESET or lvl[m + 1] != lvl[m]
        if fresh:
            tracker.reset()
            if lvl[m + 1] != lvl[m]:
                noise.reset_tracks()
                reader.new_level()
            scene = tracker.update(nxt, noise.cells)[0]
            reader.start_life(nxt)
            life = []
            continue
        after, diff = tracker.update(nxt, noise.cells)
        changed = False
        if diff is not None:
            noise.observe(scene, int(act[0]), diff)
            noisy = noise.noisy_ids
            changed = any(c.kind != "same" and c.id not in noisy for c in diff.changes)
        counters = noise.counters
        if counters and reader.counter is None:
            reader.set_counter(next((c for c in counters if c.shrinks), counters[0]))
        scene = after
        reader.observe(nxt, changed=changed)
        est = reader.remaining()
        if est is not None:
            mode = reader.chosen()
            life.append((mode, reader.moves(mode), est))
        if phase[m + 1] == GAME_OVER:
            deaths += 1
            if est is not None and est <= 1:
                spent += 1
                errors += [abs(e - (reader.moves(mode) - k)) for mode, k, e in life]
            reader.game_over()
    mean = float(np.mean(errors)) if errors else 0.0
    return spent, deaths, len(errors), mean, max(errors, default=0), reader.per_life()


def main(argv: list[str]) -> int:
    for arg in argv:
        spent, deaths, n, mean, worst, per = check(Path(arg))
        print(
            f"{Path(arg).stem:6} budget deaths {spent:2}/{deaths:2}  estimates {n:4}  "
            f"mean err {mean:5.1f}  max err {worst:4}  per life {per}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
