"""The slow layer: ask an open-weight LLM for K candidate rule programs and gate them.

Contract of ``Synthesizer.propose``:
- Sends K concurrent requests (one worker thread each) with seeds
  ``request_seed(seed, call_no, i)``. Each request is independent: any exception from
  ``chat`` (HTTP error, timeout, bad JSON) is caught, its text (cut to ``MAX_ERROR_CHARS``)
  goes to ``SynthResult.error``, and that request yields no program. A reply with no code
  fence is the same: no program, and an error note. Load, score and gate then run on the
  caller thread in order k0..k(K-1), so results are deterministic.
- Each extracted program is loaded in the sandbox as ``f"{game}-c{call_no}-k{i}"``, scored
  on every step given, and gated by ``admit_program``. ``pass`` and ``hold`` programs stay
  loaded (sleep re-scores held ones); ``drop`` programs are unloaded again, so the sandbox
  holds only admitted and held programs. Dropped candidates are not in ``programs``;
  each one's ``(pid, reason)`` is in ``dropped`` so the caller can log every wisdom drop.
- ``REQUEST_META`` is set to ``{"game", "call", "k"}`` around each chat call and reset in
  a ``finally`` in the same thread, so a thread pool of games never sees another's value.
- Only the ``max_transitions`` most recent steps go into the prompt (the prompt has a
  character cap); scoring uses all steps.
- Repair: ``repair=(program, missed_steps)`` adds the prior code and a mismatch list to
  the prompt. A missed step is shown only if it falls inside the prompt window (matched by
  ``Step.index``). The prior program must still be loaded in the sandbox: its prediction
  is recomputed with ``predict_grid`` to fill the wrong and changed cells, and a step on
  which it crashes shows as an error. A step it now predicts exactly is not listed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from arcagent.core.types import GRID_SIZE, Grid
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.types import Gate, Program, Step
from arcagent.mind.wisdom import admit_program
from arcagent.synth.prompt import Message, build_messages, extract_code
from arcagent.synth.render import render_grid
from arcagent.synth.run import Chat, Completion, request_seed
from arcagent.synth.trace import Trace, Transition
from arcagent.synth.verify import TransitionResult

from .chat_store import REQUEST_META

MAX_CELLS = 20  # per mismatch, the same cap the probe's verifier uses
MAX_ERROR_CHARS = 300  # per request, so one long traceback cannot flood the event log
MAX_EVIDENCE = 3  # level-up grid pairs shown in the goal hint
_GOAL_DEF = re.compile(r"^\s*def goal\(", re.MULTILINE)

GOAL_HINT = (
    "Also define goal(grid) -> bool: your best hypothesis of what a completed level looks "
    "like, judged from the layout alone (for example targets or marked cells, matching "
    "colors, an object standing on a marked cell). goal must be True on a finished level "
    "and False while the level is unfinished."
)


@dataclass(frozen=True, slots=True)
class SynthResult:
    """Outcome of one ``propose``: surviving ``(program, gate, hits, trials)`` rows.

    ``dropped`` holds ``(pid, gate reason)`` for every extracted program the wisdom
    function dropped (requests with no program are counted in ``error`` instead).
    """

    programs: list[tuple[Program, Gate, int, int]]
    calls: int
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None
    dropped: list[tuple[str, str]] = field(default_factory=lambda: [])


@dataclass(slots=True)
class _AllTrainTrace(Trace):
    """A trace whose ``train`` is every transition: the synthesizer has no held-out split."""

    @property
    def n_train(self) -> int:
        return len(self.transitions)


def _transition(s: Step) -> Transition:
    return Transition(
        step=s.index,
        grid_before=s.before,
        action=s.action,
        grid_after=s.after,
        phase_after=str(s.phase_after),
        levels_before=s.levels_before,
        levels_after=s.levels_after,
    )


def _cells(a: Grid, b: Grid) -> tuple[int, tuple[tuple[int, int, int, int], ...]]:
    """Count of cells where ``a != b`` and up to ``MAX_CELLS`` as ``(x, y, a, b)``."""
    ys, xs = np.nonzero(a != b)
    cells = tuple((int(x), int(y), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs, strict=True))
    return len(cells), cells[:MAX_CELLS]


def _goal_text(evidence: Sequence[tuple[Grid, Grid]]) -> str:
    if not evidence:
        return GOAL_HINT
    parts = [GOAL_HINT, "", "Grids just before and after earlier level-ups in this game:"]
    for i, (before, after) in enumerate(evidence[-MAX_EVIDENCE:]):
        parts += [f"Level-up {i}, before:", render_grid(before), f"Level-up {i}, after:"]
        parts.append(render_grid(after))
    return "\n".join(parts)


class Synthesizer:
    """K LLM proposals per call, scored in the sandbox and gated by the wisdom function."""

    def __init__(self, chat: Chat, *, k: int = 4, max_transitions: int = 60) -> None:
        self.chat = chat
        self.k = k
        self.max_transitions = max_transitions

    def _mismatches(
        self,
        window: Sequence[Step],
        repair: tuple[Program, Sequence[Step]],
        sandbox: ProgramSandbox,
    ) -> list[TransitionResult]:
        program, missed = repair
        position = {s.index: i for i, s in enumerate(window)}
        out: list[TransitionResult] = []
        for s in missed:
            i = position.get(s.index)
            if i is None:
                continue
            pred = sandbox.predict_grid(program.id, s.before, s.action)
            if pred is None:
                out.append(
                    TransitionResult("train", i, False, GRID_SIZE * GRID_SIZE, "no valid grid")
                )
                continue
            wrong_n, wrong = _cells(s.after, pred)
            if wrong_n == 0:
                continue
            changed_n, changed = _cells(s.before, pred)
            out.append(
                TransitionResult("train", i, False, wrong_n, None, wrong, changed_n, changed)
            )
        return sorted(out, key=lambda r: r.index)

    def _messages(
        self,
        trace: Trace,
        window: Sequence[Step],
        evidence: Sequence[tuple[Grid, Grid]],
        repair: tuple[Program, Sequence[Step]] | None,
        sandbox: ProgramSandbox,
    ) -> list[Message]:
        if repair is None:
            msgs = build_messages(trace, self.max_transitions)
        else:
            hits, trials = sandbox.score(repair[0].id, window)
            msgs = build_messages(
                trace,
                self.max_transitions,
                prior_code=repair[0].code,
                mismatches=self._mismatches(window, repair, sandbox),
                exact_train=(hits, trials),
            )
        msgs[1] = {"role": "user", "content": msgs[1]["content"] + "\n\n" + _goal_text(evidence)}
        return msgs

    def propose(  # noqa: PLR0917  # the eight-argument signature is the Task 12 contract
        self,
        game: str,
        call_no: int,
        steps: Sequence[Step],
        win_levels: int,
        goal_evidence: Sequence[tuple[Grid, Grid]],
        repair: tuple[Program, Sequence[Step]] | None,
        sandbox: ProgramSandbox,
        seed: int,
    ) -> SynthResult:
        """Run K requests and return the admitted and held programs. See the module docstring."""
        t0 = time.perf_counter()
        window = list(steps[-self.max_transitions :])
        trace = _AllTrainTrace(game, seed, win_levels, [_transition(s) for s in window])
        msgs = self._messages(trace, window, goal_evidence, repair, sandbox)
        programs: list[tuple[Program, Gate, int, int]] = []
        dropped: list[tuple[str, str]] = []
        errors: list[str] = []
        prompt_tokens = completion_tokens = 0

        def request(i: int) -> Completion:
            # Runs in a worker thread, which does not inherit the caller's context.
            token = REQUEST_META.set({"game": game, "call": call_no, "k": i})
            try:
                return self.chat(msgs, request_seed(seed, call_no, i))
            finally:
                REQUEST_META.reset(token)

        with ThreadPoolExecutor(max_workers=max(1, self.k)) as pool:
            futures = [pool.submit(request, i) for i in range(self.k)]
        for i, fut in enumerate(futures):
            try:
                comp = fut.result()
            except Exception as exc:  # any server failure must cost one request, not the game
                errors.append(f"request {i}: {type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS])
                continue
            prompt_tokens += comp.prompt_tokens
            completion_tokens += comp.completion_tokens
            code = extract_code(comp.text)
            if code is None:
                errors.append(f"request {i}: no code fence in the reply")
                continue
            pid = f"{game}-c{call_no}-k{i}"
            load_error = sandbox.load(pid, code)
            hits, trials = (0, 0) if load_error is not None else sandbox.score(pid, steps)
            gate = admit_program(hits, trials, load_error)
            if gate.verdict == "drop":
                sandbox.unload(pid)
                dropped.append((pid, gate.reason))
                continue
            program = Program(pid, code, _GOAL_DEF.search(code) is not None, "llm")
            programs.append((program, gate, hits, trials))
        return SynthResult(
            programs=programs,
            calls=self.k,
            seconds=time.perf_counter() - t0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error="; ".join(errors) if errors else None,
            dropped=dropped,
        )
