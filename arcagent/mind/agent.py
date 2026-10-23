"""The mind agent: one game's loop, wiring every mind part into a ``Policy``.

Per ``choose(obs)``, in order:
1. Build the ``Step`` from the previous observation and action (skipped after a
   RESET: the start frame is not a consequence of any rule, so scoring programs
   on it would count every program as a miss). Then ``belief.observe``, log the
   sandbox's own drops, ``inhibition.record``, append to working memory and
   ``wm.tick``.
2. Level-up: ``sleep("level_up")``, hand the belief to transfer (null transfer
   resets B0's priors instead), clear the plan and the visited set. GAME_OVER:
   clear the plan only. Working memory and its vetoes persist across RESET,
   because the level restarts with the same frames; only sleep empties it.
3. ``metacognition.review``; remove and log what it names.
4. Collect a finished synthesis call: ``pass`` programs enter the belief,
   ``hold`` programs go to ``store.held``, and every drop is logged. A call
   submitted on an earlier level is stale: its fits are the old level's
   evidence, so all its programs go to ``store.held`` (a ``stale_result`` event
   each) for sleep and transfer to judge, never straight into the belief.
5. Salience scores the step; on a wake with no call in flight, submit
   ``synth.propose`` to the pool. The game never waits for a call: while one is
   in flight it keeps acting (B0 when the belief is empty, branch ``fallback``,
   why ``b0 while synthesis runs``) and picks the result up on a later step.
   So no game leaves the scorecard idle while the model works.
6. ``executive.decide``, one action record, return.

Past the game (or run) deadline the agent skips waking and planning and acts
with B0 at once, so ``play_game`` can stop.

Failure guard: any ``Exception`` from the mind side of a step (sandbox,
planner, collection, modules) is logged as ``agent_error`` and that step acts
with B0 (branch ``fallback``, why ``agent_error``). B0's own bookkeeping and
the log write run outside the guard.

Invariants:
- ``call_no`` is strictly increasing for the agent's lifetime (program ids
  embed it, and a repeat would overwrite programs in the sandbox).
- Every drop is one logged event: ``program_dropped`` for a program leaving the
  belief or the store (metacognition, sandbox, sleep replay, level end),
  ``wisdom_drop`` for a candidate refused at admission.
- The action record's ``wake`` is the reason only when it started a call.
- ``active`` is ``Decision.program``: the program whose plan produced the
  action, set by the executive. A goal plan needs ``plan_min_trials`` and
  ``plan_threshold``; a lone program below them may only plan to novelty.
- At most ``max_calls_per_game`` calls; planner bounds come from the drive state.
- No call starts within the effective cutoff of the game (or run) deadline:
  ``min(cfg.synth_cutoff_s, SYNTH_CUTOFF_FRACTION * budget)``, where ``budget``
  is ``min(game_deadline, run_deadline)`` minus the agent's start time. The
  fraction keeps games shorter than the absolute cutoff (the scheduler floor is
  120 s) able to call the model at all; long games keep the absolute cutoff.

Threading: the agent runs on its game's thread. Its synthesis call runs on a
pool thread and touches only the sandbox (which is locked) and its own
arguments, never the belief or the stores.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future
from dataclasses import replace
from typing import Literal

from arcagent.core.types import RESET_ID, Action, Observation, Phase
from arcagent.mind.baseline import Baseline
from arcagent.mind.config import MindConfig, build_modules
from arcagent.mind.drive import DriveState, planner_bounds
from arcagent.mind.executive import Executive, ExecutiveInputs
from arcagent.mind.memory import GameStore, WorkingMemory
from arcagent.mind.modules.base import NullTransfer
from arcagent.mind.perception import candidate_actions, frame_hash
from arcagent.mind.predictive import Belief, PredictionRecord, ProgramState
from arcagent.mind.runlog import RunLog
from arcagent.mind.salience import SalienceConfig, score, should_wake
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.sleep import SleepReport
from arcagent.mind.sleep import sleep as sleep_cycle
from arcagent.mind.synthesizer import Synthesizer, SynthResult
from arcagent.mind.types import ActionKey, Decision, Program, Step

MISS_MEMORY = 20  # recent missed step indices kept per program, for the repair prompt
SYNTH_CUTOFF_FRACTION = 0.25  # share of a game's budget in which no call may start
MAX_ERROR_CHARS = 300  # per logged error, so one traceback cannot flood the log


class MindAgent:
    """One game's mind. Implements ``arcagent.agent.play.Policy``.

    Raises ``ValueError`` when the config uses the LLM but no synthesizer is
    given. ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        game: str,
        cfg: MindConfig,
        *,
        seed: int,
        synth: Synthesizer | None,
        pool: Executor,
        sandbox: ProgramSandbox,
        log: RunLog,
        game_deadline: float,
        run_deadline: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cfg.llm and synth is None:
            raise ValueError(f"config {cfg.name} uses the LLM but no synthesizer was given")
        self.game = game
        self.cfg = cfg
        self.seed = seed
        self._synth = synth if cfg.llm else None
        self._pool = pool
        self._sandbox = sandbox
        self._log = log
        self._game_deadline = game_deadline
        self._run_deadline = run_deadline
        self._clock = clock
        self._started = clock()
        self._salience = SalienceConfig()

        self._modules = build_modules(cfg)
        self._null_transfer = isinstance(self._modules.transfer, NullTransfer)
        self.wm = WorkingMemory()
        self.store = GameStore()
        self.belief = Belief(sandbox)
        self._baseline = Baseline(seed)
        self._executive = Executive(
            plan_to_goal=lambda st: self._plan_to(st, "goal"),
            plan_to_novel=lambda st: self._plan_to(st, "novel"),
            voi_pick=self._voi_pick,
            b0_pick=lambda vetoes: self._baseline.choose(self._current(), vetoes),
        )

        self._plan: list[Action] = []
        self._active: str | None = None
        self._prev: tuple[Observation, Action] | None = None
        self._action_no = 0
        self._steps_seen = 0
        self._steps_since_call = 0
        self._calls_made = 0
        self._future: Future[SynthResult] | None = None
        self._future_call = -1
        self._visited: set[str] = set()
        self._misses: dict[str, list[int]] = {}
        self._carried = False
        self._closed = False
        # Per-step scratch read by the executive's callables.
        self._obs: Observation | None = None
        self._drive = self._drive_state(self._started)
        self._voi_scores: dict[str, float] = {}
        self._level = 0
        self._future_level = 0
        self._records: list[PredictionRecord] = []
        self._vetoes: frozenset[ActionKey] = frozenset()
        self._wake: str | None = None

    # -- the step ----------------------------------------------------------

    def choose(self, obs: Observation) -> Action:
        """One action for ``obs``; see the module docstring for the order."""
        now = self._clock()
        frame = frame_hash(obs.grid)
        self._obs = obs
        self._level = obs.levels_completed
        self._records, self._vetoes, self._wake = [], frozenset(), None
        self._baseline.observe(obs)
        try:
            decision = self._mind_step(obs, frame, now)
        except Exception as exc:  # a mind-side failure costs this step's reasoning, not the game
            self._log.event(
                "agent_error",
                step=self._action_no,
                error=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
            )
            self._plan, self._active, self._wake = [], None, None
            decision = Decision(self._baseline.choose(obs, self._vetoes), "fallback", "agent_error")
        records, vetoes, wake = self._records, self._vetoes, self._wake
        if decision.branch != "fallback":
            self._baseline.note(obs, decision.action)
        a = decision.action
        self._log.action(
            level=obs.levels_completed,
            step=self._action_no,
            frame_hash=frame,
            action=[a.id, a.x, a.y],
            branch=decision.branch,
            why=decision.why,
            programs=[
                {"id": r.pid, "pred": r.pred, "matched": r.matched, "fit": r.fit} for r in records
            ],
            vetoes=[list(v) for v in sorted(vetoes)],
            voi=self._voi_scores,
            entropy=self.belief.entropy(),
            wake=wake,
            active=self._active,
        )
        self._voi_scores = {}
        self._prev = (obs, a)
        self._action_no += 1
        return a

    def _mind_step(self, obs: Observation, frame: str, now: float) -> Decision:
        """Everything but B0's bookkeeping and the log write; ``choose`` guards it.

        Partial results go to ``_records``, ``_vetoes`` and ``_wake`` as they are
        computed, so a failure later in the step still logs and honors them.
        """
        step, self._records = self._observe(obs)
        self._visited.add(frame)
        self._collect()
        self._vetoes = self._modules.inhibition.vetoes(frame, self.wm, self.store)
        if now >= min(self._game_deadline, self._run_deadline):
            self._plan, self._active = [], None
            return Decision(self._baseline.choose(obs, self._vetoes), "fallback", "deadline passed")
        self._wake = self._maybe_wake(obs, step, self._records, now)
        decision = self._decide(obs, self._vetoes)
        if self._future is not None and decision.why == "b0":
            decision = replace(decision, why="b0 while synthesis runs")
        return decision

    def _observe(self, obs: Observation) -> tuple[Step | None, list[PredictionRecord]]:
        """Steps 1 to 3 of the loop. Returns the step (None if there is none) and its records."""
        if self._prev is None or self._prev[1].id == RESET_ID:
            return None, []
        prev_obs, prev_action = self._prev
        step = Step(
            index=self._action_no - 1,
            before=prev_obs.grid,
            action=prev_action,
            after=obs.grid,
            levels_before=prev_obs.levels_completed,
            levels_after=obs.levels_completed,
            phase_after=obs.phase,
        )
        states = dict(self.belief.programs)  # first_miss_step of a program the sandbox drops
        records = self.belief.observe(step)
        for r in records:
            if not r.matched:
                misses = self._misses.setdefault(r.pid, [])
                misses.append(step.index)
                del misses[:-MISS_MEMORY]
        for pid, reason in self.belief.drained_drops():
            self._log_drop(pid, reason, step.index, states.get(pid))
        inhibition = self._modules.inhibition
        for frame, key, reason in inhibition.record(
            step, records, self._active, self.wm, self.store
        ):
            self._log.event("veto", step=step.index, frame=frame, action=list(key), reason=reason)
        self.wm.steps.append(step)
        self.wm.tick(self.store)
        self._steps_seen += 1
        self._steps_since_call += 1
        if step.level_up:
            self._level_up(step)
        elif obs.phase == Phase.GAME_OVER:
            self._plan, self._active = [], None
            self._log.event("game_over", step=step.index, level=obs.levels_completed)
        for pid, reason in self._modules.metacognition.review(self.belief, step.index):
            state = self.belief.programs.get(pid)
            self.belief.remove(pid)
            self._log_drop(pid, reason, step.index, state)
        return step, records

    def _level_up(self, step: Step) -> None:
        """Sleep, then hand the belief to transfer; the old level's plan and frames go."""
        self._log.event("level_up", step=step.index, level=step.levels_after)
        report = sleep_cycle("level_up", self.wm, self.store, self.belief, self._sandbox)
        self._log_sleep(report, step.index)
        left = self._hand_over()
        carried = self._modules.transfer.carry(self.store, self.belief)
        if self._null_transfer:
            self._baseline.reset_priors()
        self._log.event("transfer", step=step.index, carried=carried, null=self._null_transfer)
        # One event per program: a sleep-replay drop is not also a level_end drop.
        states = dict(left)
        for pid, state in left:
            if pid not in report.dropped:
                self._log_drop(pid, "level_end", step.index, state)
        for pid in report.dropped:
            self._log_drop(pid, "sleep_replay", step.index, states.get(pid))
        self._carried = bool(carried)
        self._plan, self._active = [], None
        self._visited = set()

    def _hand_over(self) -> list[tuple[str, ProgramState]]:
        """Empty the belief for transfer; return the programs that left it for good.

        A program sleep kept (procedural or held) stays loaded, because the store
        owns it now and replay or transfer will need it; any other is unloaded.
        """
        left: list[tuple[str, ProgramState]] = []
        for pid, state in list(self.belief.programs.items()):
            if pid in self.store.procedural or pid in self.store.held:
                del self.belief.programs[pid]
            else:
                self.belief.remove(pid)
                left.append((pid, state))
        self._misses.clear()
        return left

    # -- synthesis ---------------------------------------------------------

    def _maybe_wake(
        self, obs: Observation, step: Step | None, records: Sequence[PredictionRecord], now: float
    ) -> str | None:
        """Salience's wake reason if it started a call, else None."""
        if self._synth is None or self._future is not None or step is None:
            return None
        deadline = min(self._game_deadline, self._run_deadline)
        cutoff = min(self.cfg.synth_cutoff_s, SYNTH_CUTOFF_FRACTION * (deadline - self._started))
        if now >= deadline - cutoff:
            return None  # a call started now could not finish before the game ends
        if not self.wm.steps:
            return None  # a call with no transitions to score on can only be dropped
        transfer_failed = False
        if self._carried and self._modules.transfer.check(self.belief, len(self.wm.steps)):
            transfer_failed, self._carried = True, False
        events = score(step, records, not self.belief.programs, transfer_failed)
        wake = should_wake(
            events,
            steps_seen=self._steps_seen,
            steps_since_call=self._steps_since_call,
            calls_made=self._calls_made,
            drive=self._drive_state(now),
            cfg=self._salience,
        )
        if wake is None or not self._submit(obs):
            return None
        return wake

    def _submit(self, obs: Observation) -> bool:
        """Start one call; False if it could not be submitted (the call is still spent)."""
        synth = self._synth
        if synth is None:
            return False
        call_no = self._calls_made
        self._calls_made += 1
        self._steps_since_call = 0
        try:
            self._future = self._pool.submit(
                synth.propose,
                self.game,
                call_no,
                list(self.wm.steps),
                obs.win_levels,
                list(self.store.goal_evidence),
                self._repair(),
                self._sandbox,
                self.seed,
            )
        except RuntimeError as exc:  # a shut-down pool costs this call, not the game
            self._log_call(call_no, None, 0, 0, 0, f"submit failed: {exc}")
            return False
        self._future_call = call_no
        self._future_level = obs.levels_completed
        if self._future.done():
            self._collect()
        return True

    def _repair(self) -> tuple[Program, list[Step]] | None:
        """The best live program and its recent misses on this level, if it has any."""
        best = self.belief.best(0.0, 1)
        if best is None:
            return None
        idx = set(self._misses.get(best.program.id, []))
        missed = [s for s in self.wm.steps if s.index in idx]
        return (best.program, missed) if missed else None

    def _collect(self) -> None:
        """Gate a finished call's programs into the belief or the held store; log it."""
        fut = self._future
        if fut is None or not fut.done():
            return
        self._future = None
        call = self._future_call
        try:
            res = fut.result()
        except Exception as exc:  # propose itself failed: the call is lost, the game goes on
            self._log_call(call, None, 0, 0, 0, f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS])
            return
        if self._future_level != self._level:
            self._collect_stale(call, res)
            return
        refused = list(res.dropped)
        admits: list[tuple[Program, str, str, int, int]] = []
        for program, gate, hits, trials in res.programs:
            if gate.verdict == "hold":
                self.store.held[program.id] = program
                admits.append((program, "hold", gate.reason, hits, trials))
                continue
            error = self.belief.add(program, hits, trials)
            if error is not None:
                refused.append((program.id, f"load error: {error}"))
                continue
            admits.append((program, gate.verdict, gate.reason, hits, trials))
        held = sum(1 for a in admits if a[1] == "hold")
        self._log_call(call, res, len(admits) - held, held, len(refused), res.error)
        for program, verdict, reason, hits, trials in admits:
            self._log.event(
                "program_admitted",
                step=self._action_no,
                call=call,
                pid=program.id,
                verdict=verdict,
                reason=reason,
                hits=hits,
                trials=trials,
                into="held" if verdict == "hold" else "belief",
            )
        for pid, reason in refused:
            self._log.event("wisdom_drop", step=self._action_no, call=call, pid=pid, reason=reason)

    def _collect_stale(self, call: int, res: SynthResult) -> None:
        """A call from an earlier level: its fits are the old level's evidence.

        Its programs never enter the belief directly (that would skip both the
        new level's evidence and transfer). Passing and held ones go to
        ``store.held``, still loaded, for sleep and transfer to judge.
        """
        for program, _gate, _hits, _trials in res.programs:
            self.store.held[program.id] = program
        self._log_call(call, res, 0, len(res.programs), len(res.dropped), res.error)
        for program, gate, hits, trials in res.programs:
            self._log.event(
                "stale_result",
                step=self._action_no,
                call=call,
                pid=program.id,
                verdict=gate.verdict,
                hits=hits,
                trials=trials,
                submit_level=self._future_level,
                level=self._level,
            )
        for pid, reason in res.dropped:
            self._log.event("wisdom_drop", step=self._action_no, call=call, pid=pid, reason=reason)

    # -- the executive -----------------------------------------------------

    def _decide(self, obs: Observation, vetoes: frozenset[ActionKey]) -> Decision:
        self._drive = self._drive_state(self._clock())
        n = len(self.belief.programs)
        best = self.belief.best(self.cfg.plan_threshold, self.cfg.plan_min_trials)
        if best is None and n == 1:
            # The novel-state path needs the single live program even below threshold.
            only = next(iter(self.belief.programs.values()))
            best = only if only.trials >= 1 else None
        active = None if self._active is None else self.belief.programs.get(self._active)
        inp = ExecutiveInputs(obs, self._plan, active, best, n, vetoes, self.cfg.plan_threshold)
        decision, self._plan = self._executive.decide(inp)
        self._active = decision.program
        return decision

    def _plan_to(
        self, state: ProgramState, target: Literal["goal", "novel"]
    ) -> list[Action] | None:
        if self._synth is None:
            return None  # no LLM, no programs, no planner
        if target == "goal" and (
            state.trials < self.cfg.plan_min_trials or state.fit < self.cfg.plan_threshold
        ):
            return None  # a lone program passed for novelty has not earned a goal plan
        obs = self._current()
        pid = state.program.id
        max_nodes, time_limit_s = planner_bounds(self._drive)
        res = self._sandbox.plan(
            pid,
            obs.grid,
            candidate_actions(obs),
            target=target,
            visited=frozenset(self._visited),
            max_nodes=max_nodes,
            time_limit_s=time_limit_s,
        )
        self._log.event(
            "plan",
            step=self._action_no,
            pid=pid,
            target=target,
            stopped=res.stopped,
            expanded=res.expanded,
            seconds=res.seconds,
            errors=res.errors,
            length=None if res.actions is None else len(res.actions),
        )
        if res.actions is None:
            return None
        return list(res.actions)

    def _voi_pick(self) -> Action | None:
        obs = self._current()
        pick, scores = self._modules.voi.pick(
            self.belief, self._sandbox, obs, candidate_actions(obs), frozenset(self._visited)
        )
        self._voi_scores = {f"{k[0]},{k[1]},{k[2]}": float(v) for k, v in scores.items()}
        return pick

    def _current(self) -> Observation:
        if self._obs is None:
            raise RuntimeError("the executive ran before choose() set the observation")
        return self._obs

    def _drive_state(self, now: float) -> DriveState:
        return DriveState(
            started=self._started,
            game_deadline=self._game_deadline,
            run_deadline=self._run_deadline,
            now=now,
            level_actions=len(self.wm.steps),
            synth_calls_left=max(0, self.cfg.max_calls_per_game - self._calls_made),
            queue_depth=0 if self._future is None else 1,
        )

    # -- logging and shutdown ------------------------------------------------

    def _log_drop(self, pid: str, reason: str, step: int, state: ProgramState | None) -> None:
        self._misses.pop(pid, None)
        first_miss = None if state is None else state.first_miss_step
        self._log.event(
            "program_dropped", pid=pid, reason=reason, step=step, first_miss_step=first_miss
        )

    def _log_call(  # noqa: PLR0917  # one positional per llm_call field
        self,
        call: int,
        res: SynthResult | None,
        admitted: int,
        held: int,
        dropped: int,
        error: str | None,
    ) -> None:
        self._log.event(
            "llm_call",
            step=self._action_no,
            call=call,
            k=0 if self._synth is None else self._synth.k,
            seconds=0.0 if res is None else res.seconds,
            prompt_tokens=0 if res is None else res.prompt_tokens,
            completion_tokens=0 if res is None else res.completion_tokens,
            admitted=admitted,
            held=held,
            dropped=dropped,
            error=error,
        )

    def _log_sleep(self, report: SleepReport, step: int) -> None:
        self._log.event(
            "sleep",
            step=step,
            reason=report.reason,
            steps_compacted=report.steps_compacted,
            consolidated=report.consolidated,
            replay={pid: list(ht) for pid, ht in report.replay.items()},
            goal_confirmed=report.goal_confirmed,
            dropped=report.dropped,
        )

    def close(self) -> None:
        """Sleep on the game, log ``game_end`` and close the sandbox. Safe to call twice.

        A call still queued is cancelled. One already running cannot be stopped;
        it finishes on its pool thread, its sandbox requests fail once the
        sandbox is closed, and its result is never read.
        """
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
            self._future = None
        step = self._action_no
        report = sleep_cycle("game_end", self.wm, self.store, self.belief, self._sandbox)
        self._log_sleep(report, step)
        for pid in report.dropped:
            self._log_drop(pid, "sleep_replay", step, None)
        level = 0 if self._prev is None else self._prev[0].levels_completed
        self._log.event(
            "game_end",
            step=step,
            level=level,
            calls=self._calls_made,
            sandbox_restarts=self._sandbox.restarts,
        )
        self._sandbox.close()
