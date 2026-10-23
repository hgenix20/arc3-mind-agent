"""Sleep: bookkeeping at every level-up and at game end.

Contract: ``sleep`` compacts the finished level into the game store, keeps
well-fitting live programs, re-verifies every kept program against every
recorded transition, and stores goal evidence. It never chooses what the next
level's belief holds (transfer does) and never touches ``belief``.

Order matters. Compaction runs first so that ``all_steps`` afterwards is every
episode including the one just finished (working memory is empty by then), and
replay scores over that full list.

Invariants:
- A program that sleep keeps (procedural or held) stays loaded in the sandbox.
  A program that sleep drops is unloaded and removed from procedural and held.
- Gating is ``admit_program`` with its default thresholds; no model decides.
- With no recorded steps there is no evidence, so replay leaves stores as they are.

Failure modes: a program that times out or crashes during replay scores as a
total miss and is dropped; nothing here raises on model code.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Literal, cast

from arcagent.mind.memory import GameStore, WorkingMemory, all_steps
from arcagent.mind.predictive import Belief
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.types import Program
from arcagent.mind.wisdom import admit_program

# Read from wisdom so consolidation and replay can never disagree on "pass".
ADMIT_FIT = cast(float, inspect.signature(admit_program).parameters["admit_fit"].default)


@dataclass(frozen=True, slots=True)
class SleepReport:
    """What one sleep did. ``replay`` maps every re-scored program to ``(hits, trials)``."""

    reason: str
    steps_compacted: int
    consolidated: list[str]
    replay: dict[str, tuple[int, int]]
    goal_confirmed: list[str]
    dropped: list[str]


def sleep(
    reason: Literal["level_up", "game_end"],
    wm: WorkingMemory,
    store: GameStore,
    belief: Belief,
    sandbox: ProgramSandbox,
) -> SleepReport:
    """Run one sleep cycle; see the module docstring for the contract."""
    # 1. Compaction. Evidence only on level_up: the last step's frames are the
    # transition into the solved state.
    episode = wm.clear_level()
    if episode:
        store.episodes.append(episode)
        if reason == "level_up":
            last = episode[-1]
            store.goal_evidence.append((last.before, last.after))

    # 2. Consolidation.
    consolidated: list[str] = []
    for pid, state in belief.programs.items():
        if state.fit >= ADMIT_FIT:
            store.procedural[pid] = (state.program, state.hits, state.trials)
            consolidated.append(pid)

    # 3. Verification and replay over every transition ever recorded.
    steps = all_steps(store, wm)
    replay: dict[str, tuple[int, int]] = {}
    dropped: list[str] = []
    if steps:
        for pid in [*store.procedural, *store.held]:
            program = store.procedural[pid][0] if pid in store.procedural else store.held[pid]
            hits, trials = sandbox.score(pid, steps)
            replay[pid] = (hits, trials)
            verdict = admit_program(hits, trials, None).verdict
            if verdict == "pass":
                store.held.pop(pid, None)
                store.procedural[pid] = (program, hits, trials)
            elif verdict == "hold":
                store.procedural.pop(pid, None)
                store.held[pid] = program
            else:
                store.procedural.pop(pid, None)
                store.held.pop(pid, None)
                sandbox.unload(pid)
                dropped.append(pid)

    # 4. Goal check on the newest evidence pair, only at a level-up.
    confirmed: list[str] = []
    if reason == "level_up" and store.goal_evidence:
        before, after = store.goal_evidence[-1]
        candidates: dict[str, Program] = {s.program.id: s.program for s in belief.programs.values()}
        candidates.update({pid: entry[0] for pid, entry in store.procedural.items()})
        candidates.update(store.held)
        pids = [pid for pid, p in candidates.items() if p.has_goal and pid not in dropped]
        if pids:
            at_after = sandbox.goal(pids, after)
            at_before = sandbox.goal(pids, before)
            confirmed = [p for p in pids if at_after[p] is True and at_before[p] is False]

    return SleepReport(reason, len(episode), consolidated, replay, confirmed, dropped)
