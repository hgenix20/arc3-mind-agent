"""Predictive loop: belief over the live world-model programs.

Each observed step is predicted by every live program from the step's
``before`` grid and ``action``; the prediction is compared with the hash of the
real ``after`` frame and the program's fit is updated. This module only
observes and scores. It never chooses an action and returns no ``Action``.

Invariants:
- ``programs`` holds exactly the programs loaded in the sandbox that this
  belief knows about. A program the sandbox drops (timeout, crash) is removed
  here in the same ``observe`` call; the (pid, reason) pair waits in
  ``drained_drops`` for the caller to log.
- A ``None`` prediction (crash, timeout, invalid output) is a trial and a miss.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from arcagent.core.types import Action, Grid
from arcagent.mind.perception import frame_hash
from arcagent.mind.types import Program, Step


class PredictorSandbox(Protocol):
    """The slice of ``ProgramSandbox`` that the belief uses (satisfied structurally)."""

    dropped: dict[str, str]

    def load(self, pid: str, code: str) -> str | None: ...

    def unload(self, pid: str) -> None: ...

    def predict(self, pids: Sequence[str], grid: Grid, action: Action) -> dict[str, str | None]: ...


@dataclass(slots=True)
class ProgramState:
    """A program and its running record.

    ``first_miss_step`` is the ``Step.index`` of the first miss since the
    program was added and is never reset. ``misses_in_row`` resets on a hit.
    ``last_ok`` is the latest outcome, True before any trial.
    """

    program: Program
    hits: int = 0
    trials: int = 0
    misses_in_row: int = 0
    first_miss_step: int | None = None
    last_ok: bool = True

    @property
    def fit(self) -> float:
        return self.hits / self.trials if self.trials else 0.0


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One program's prediction for one step; ``fit`` is after the update."""

    pid: str
    pred: str | None
    matched: bool
    fit: float


class Belief:
    """Live programs with their prediction fit, backed by one sandbox."""

    def __init__(self, sandbox: PredictorSandbox) -> None:
        self._sandbox = sandbox
        self.programs: dict[str, ProgramState] = {}
        self._drops: list[tuple[str, str]] = []

    def add(self, program: Program, hits: int = 0, trials: int = 0) -> str | None:
        """Load ``program`` and track it. Returns the load error, in which case it is not tracked.

        A duplicate id replaces the old state (and reloads the code); if the
        reload fails the stale state is removed too.
        """
        self.programs.pop(program.id, None)
        error = self._sandbox.load(program.id, program.code)
        if error is not None:
            return error
        self.programs[program.id] = ProgramState(program, hits, trials)
        return None

    def remove(self, pid: str) -> None:
        """Stop tracking and unload ``pid``; unknown ids are ignored."""
        self.programs.pop(pid, None)
        self._sandbox.unload(pid)

    def observe(self, step: Step) -> list[PredictionRecord]:
        """Score every live program on ``step`` and drop those the sandbox dropped."""
        pids = list(self.programs)
        if not pids:
            return []
        preds = self._sandbox.predict(pids, step.before, step.action)
        actual = frame_hash(step.after)
        records: list[PredictionRecord] = []
        for pid in pids:
            st = self.programs[pid]
            pred = preds.get(pid)
            matched = pred is not None and pred == actual
            st.trials += 1
            if matched:
                st.hits += 1
                st.misses_in_row = 0
            else:
                st.misses_in_row += 1
                if st.first_miss_step is None:
                    st.first_miss_step = step.index
            st.last_ok = matched
            records.append(PredictionRecord(pid, pred, matched, st.fit))
        for pid in pids:
            reason = self._sandbox.dropped.get(pid)
            if reason is not None:
                del self.programs[pid]
                self._drops.append((pid, reason))
        return records

    def drained_drops(self) -> list[tuple[str, str]]:
        """Return and clear the (pid, reason) pairs removed because the sandbox dropped them."""
        out, self._drops = self._drops, []
        return out

    def best(self, min_fit: float, min_trials: int) -> ProgramState | None:
        """Highest fit, then most trials, then lowest id, among programs meeting both minimums."""
        eligible = [
            s for s in self.programs.values() if s.fit >= min_fit and s.trials >= min_trials
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda s: (-s.fit, -s.trials, s.program.id))

    def entropy(self) -> float:
        """Normalized Shannon entropy of fit-weighted belief over programs with trials.

        0.0 for fewer than two such programs or when every fit is 0; 1.0 when
        all fits are equal.
        """
        weights = [s.fit for s in self.programs.values() if s.trials > 0]
        n = len(weights)
        total = sum(weights)
        if n <= 1 or total == 0.0:
            return 0.0
        h = -sum((w / total) * math.log(w / total) for w in weights if w > 0)
        return h / math.log(n)
