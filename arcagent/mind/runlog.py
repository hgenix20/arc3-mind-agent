"""Run log: the paper's dataset, one JSONL file per game.

Thread contract: a ``RunLog`` is written from one game's thread only (one file
per game), so it takes no lock. Every call writes one JSON line and flushes, so
a killed run keeps everything logged so far.

``action(**fields)`` writes ``{"type": "action", "t", "game", "config",
"seed", ...fields}``; ``event(kind, **fields)`` writes ``{"type": kind, ...}``
with ``t`` as well. ``t`` is ``time.time()`` rounded to milliseconds. The log
accepts arbitrary keyword fields; the lists below document what later tasks
write and what the metrics script reads.

Action record fields: ``level, step, frame_hash, action [id, x, y], branch,
why, programs [{"id", "pred", "matched", "fit"}], vetoes [[id, x, y]],
voi {"id,x,y": score}, entropy, wake``, plus ``active`` (the active program id
or null).

Event kinds: ``llm_call, plan, level_up, game_over, program_admitted,
program_dropped, veto, wisdom_drop, sleep, transfer, game_end``. The
``program_dropped`` event carries ``pid, reason, step, first_miss_step``.

Conversion: numpy integer, floating and bool scalars and arrays become plain
Python values and tuples become lists. Anything else unserializable raises
``TypeError`` (the line is not written); values are never stringified.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO

import numpy as np


def _plain(value: object) -> object:
    """``json.dumps`` default hook: numpy to Python, else fail explicitly."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class RunLog:
    """Append-only JSONL writer for one game (single-threaded use)."""

    def __init__(self, path: Path, game: str, config: str, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = path.open("a", encoding="utf-8", newline="\n")
        self._game = game
        self._config = config
        self._seed = seed

    def _write(self, record: dict[str, object]) -> None:
        # Serialize first so a TypeError leaves no partial line behind.
        line = json.dumps(record, default=_plain)
        self._fh.write(line + "\n")
        self._fh.flush()

    def action(self, **fields: object) -> None:
        """Write one action record."""
        self._write(
            {
                "type": "action",
                "t": round(time.time(), 3),
                "game": self._game,
                "config": self._config,
                "seed": self._seed,
                **fields,
            }
        )

    def event(self, kind: str, **fields: object) -> None:
        """Write one event record of the given kind."""
        self._write({"type": kind, "t": round(time.time(), 3), **fields})

    def close(self) -> None:
        """Close the file; later writes raise ``ValueError``."""
        self._fh.close()
