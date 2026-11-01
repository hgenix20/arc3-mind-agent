"""Run log: the paper's dataset, one JSONL file per game.

Thread contract: a ``RunLog`` is written from one game's thread only (one file
per game), so it takes no lock. Every call writes one JSON line and flushes, so
a killed run keeps everything logged so far.

Every record starts with the same envelope ``type, t, game, config, seed``:
``action(**fields)`` writes ``{"type": "action", ...envelope, ...fields}`` and
``event(kind, **fields)`` writes ``{"type": kind, ...envelope, ...fields}``.
``t`` is ``time.time()`` rounded to milliseconds. A caller field named like an
envelope key raises ``ValueError`` and nothing is written. Other keyword fields
are accepted; the lists below document what later tasks
write and what the metrics script reads.

Action record fields: ``level, step, frame_hash, action [id, x, y], branch,
why, programs [{"id", "pred", "matched", "fit"}], vetoes [[id, x, y]],
voi {"id,x,y": score}, entropy, wake``, plus ``active`` (the active program id
or null).

Event kinds: ``llm_call, plan, plan_skipped, level_up, game_over, program_admitted,
program_dropped, veto, wisdom_drop, sleep, transfer, game_end``. The
``program_dropped`` event carries ``pid, reason, step, first_miss_step``.
The fluid agent (``arcagent.mind.fluid``) adds ``fact`` (``step, key, status,
line``), ``goal`` (``step, id, status, source``), ``test`` (``step, action,
question``) and ``noise`` (``step, ids, cells``); its ``veto`` carries ``step,
action, reason`` and its ``game_end`` counts ``facts`` and ``goals`` by status
plus ``tests`` and ``vetoes``. Its action records score facts in ``programs``
(id is the fact key joined by spaces) and log ``active`` null and ``voi`` empty.
With the executive turn (``FLUID-LLM``) it also writes ``wake`` (``step, reason``;
the action record's ``wake`` holds the reason on the move that started a turn),
``exec_turn`` (``step, call, seconds, goals, plan_len, test, dropped``) and
``exec_failed`` (``step, call, reason``), and its ``game_end`` adds ``turns``.

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

ENVELOPE = frozenset({"type", "t", "game", "config", "seed"})


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
    """Append-only JSONL writer for one game (single-threaded use).

    Creates ``path``; raises ``FileExistsError`` if it already exists, so every
    log holds exactly one run.
    """

    def __init__(self, path: Path, game: str, config: str, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # "x": a second run into the same path would mix two runs in one log.
        self._fh: IO[str] = path.open("x", encoding="utf-8", newline="\n")
        self._game = game
        self._config = config
        self._seed = seed

    def _write(self, record: dict[str, object]) -> None:
        # Serialize first so a TypeError leaves no partial line behind.
        line = json.dumps(record, default=_plain)
        self._fh.write(line + "\n")
        self._fh.flush()

    def _record(self, kind: str, fields: dict[str, object]) -> dict[str, object]:
        clash = sorted(ENVELOPE.intersection(fields))
        if clash:
            raise ValueError(f"reserved record fields: {', '.join(clash)}")
        return {
            "type": kind,
            "t": round(time.time(), 3),
            "game": self._game,
            "config": self._config,
            "seed": self._seed,
            **fields,
        }

    def action(self, **fields: object) -> None:
        """Write one action record. Raises ``ValueError`` on an envelope field."""
        self._write(self._record("action", fields))

    def event(self, kind: str, **fields: object) -> None:
        """Write one event record of the given kind. Raises ``ValueError`` on an envelope field."""
        self._write(self._record(kind, fields))

    def close(self) -> None:
        """Close the file; later writes raise ``ValueError``."""
        self._fh.close()
