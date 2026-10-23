"""Carry-over of surviving programs into the next level (HCQM 6.3).

Sleep leaves its survivors in ``store.procedural`` (consolidated) and
``store.held`` (passed but unconsolidated). ``CarryTransfer`` loads them into the
new level's belief with a clean record, so the first transitions of the level
re-verify them; a program that no longer fits is dropped by the normal paths.
"""

from __future__ import annotations

import dataclasses

from arcagent.mind.memory import GameStore
from arcagent.mind.predictive import Belief


class CarryTransfer:
    """Hand last level's programs to the next level and judge whether they fit.

    Contract:
    - ``carry`` loads every ``store.procedural`` and ``store.held`` program into
      ``belief`` with hits and trials reset to 0 and ``origin="transfer"`` (same
      id, so the sandbox and the store still key on it). It returns the pids that
      loaded, in sorted order, and remembers them for ``check``. Each call
      replaces the remembered set, so a pid never leaks across levels.
    - ``check`` is True ("failed") only when ``steps_on_level >= check_steps`` and
      no remembered pid still in ``belief`` has ``fit >= admit_fit``. With nothing
      carried it is always False: an empty belief is salience's ``belief_empty``
      case, not a transfer failure.

    Failure mode: a program whose code no longer loads is skipped and left out of
    the returned list, so ``carry`` never raises for a bad program.
    """

    def __init__(self, check_steps: int = 5, admit_fit: float = 0.8) -> None:
        self._check_steps = check_steps
        self._admit_fit = admit_fit
        self._carried: list[str] = []

    def carry(self, store: GameStore, belief: Belief) -> list[str]:
        programs = {pid: prog for pid, (prog, _, _) in store.procedural.items()}
        programs.update(store.held)  # a pid in both stores is one program
        self._carried = []
        for pid in sorted(programs):
            moved = dataclasses.replace(programs[pid], origin="transfer")
            if belief.add(moved) is None:
                self._carried.append(pid)
        return list(self._carried)

    def check(self, belief: Belief, steps_on_level: int) -> bool:
        if not self._carried or steps_on_level < self._check_steps:
            return False
        return not any(
            (state := belief.programs.get(pid)) is not None and state.fit >= self._admit_fit
            for pid in self._carried
        )
