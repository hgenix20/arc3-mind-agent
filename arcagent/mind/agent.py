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
   ``hold`` programs go to ``store.held``, and every drop is logged.
5. Salience scores the step; on a wake with no call in flight, submit
   ``synth.propose`` to the pool. While a call is in flight and the belief is
   empty, wait in ``synth_wait_poll_s`` slices until the call finishes, the
   shared idle clock reaches ``idle_act_s`` (one B0 action, branch ``wait``),
   or the game deadline passes (one B0 action). With programs in the belief the
   game acts and picks the result up on a later step.
6. ``executive.decide``, ``activity.touch()``, one action record, return.

Past the game deadline the agent skips waking, waiting and planning and acts
with B0 at once, so ``play_game`` can stop.

Invariants:
- ``call_no`` is strictly increasing for the agent's lifetime (program ids
  embed it, and a repeat would overwrite programs in the sandbox).
- Every drop is one logged event: ``program_dropped`` for a program leaving the
  belief or the store (metacognition, sandbox, sleep replay, level end),
  ``wisdom_drop`` for a candidate refused at admission.
- The action record's ``wake`` is the reason only when it started a call.
- At most ``max_calls_per_game`` calls; planner bounds come from the drive state.

Threading: the agent runs on its game's thread. Its synthesis call runs on a
pool thread and touches only the sandbox (which is locked) and its own
arguments, never the belief or the stores.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future
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


class ActivityClock:
    """When any game last sent an action. Shared by every game's thread."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._last = clock()

    def touch(self) -> None:
        """Record an action sent now (by any game)."""
        with self._lock:
            self._last = self._clock()

    def idle_s(self, now: float) -> float:
        """Seconds since the last action, never negative."""
        with self._lock:
            return max(0.0, now - self._last)


class MindAgent:
    """One game's mind. Implements ``arcagent.agent.play.Policy``.

    Raises ``ValueError`` when the config uses the LLM but no synthesizer is
    given. ``sleep`` is the wait function (injectable with ``clock`` for tests).
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
        activity: ActivityClock,
        game_deadline: float,
        run_deadline: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
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
        self._activity = activity
        self._game_deadline = game_deadline
        self._run_deadline = run_deadline
        self._clock = clock
        self._sleep = sleep
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
        self._planned_pid: str | None = None

    # -- the step ----------------------------------------------------------

    def choose(self, obs: Observation) -> Action:
        """One action for ``obs``; see the module docstring for the order."""
        now = self._clock()
        step, records = self._observe(obs)
        frame = frame_hash(obs.grid)
        self._visited.add(frame)
        self._obs = obs
        self._baseline.observe(obs)
        self._collect()
        vetoes = self._modules.inhibition.vetoes(frame, self.wm, self.store)
        wake: str | None = None
        if now >= min(self._game_deadline, self._run_deadline):
            decision = Decision(self._baseline.choose(obs, vetoes), "fallback", "deadline passed")
            self._plan, self._active = [], None
        else:
            wake = self._maybe_wake(obs, step, records, now)
            decision = self._wait(obs, vetoes) or self._decide(obs, vetoes)
        if decision.branch not in ("fallback", "wait"):
            self._baseline.note(obs, decision.action)
        self._activity.touch()
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
        for pid, state in left:
            self._log_drop(pid, "level_end", step.index, state)
        for pid in report.dropped:
            self._log_drop(pid, "sleep_replay", step.index, None)
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
        if wake is None:
            return None
        self._submit(obs)
        return wake

    def _submit(self, obs: Observation) -> None:
        synth = self._synth
        if synth is None:
            return
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
            return
        self._future_call = call_no
        if self._future.done():
            self._collect()

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
            self._log_call(call, None, 0, 0, 0, f"{type(exc).__name__}: {exc}"[:300])
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

    def _wait(self, obs: Observation, vetoes: frozenset[ActionKey]) -> Decision | None:
        """Block while a call runs and nothing in the belief can act; None means decide normally."""
        fut = self._future
        if fut is None or self.belief.programs or obs.terminal:
            return None
        poll = self.cfg.synth_wait_poll_s
        start = self._clock()
        # Bounded by count as well as by time, in case the clock stalls.
        max_polls = math.ceil(max(0.0, self._game_deadline - start) / poll) + 1
        for _ in range(max_polls):
            if fut.done():
                self._collect()
                return None
            now = self._clock()
            if now >= self._game_deadline:
                break
            idle = self._activity.idle_s(now)
            if idle >= self.cfg.idle_act_s:
                why = f"idle {idle:.0f}s while synthesis runs"
                return Decision(self._baseline.choose(obs, vetoes), "wait", why)
            self._sleep(poll)
        return Decision(self._baseline.choose(obs, vetoes), "fallback", "game deadline passed")

    # -- the executive -----------------------------------------------------

    def _decide(self, obs: Observation, vetoes: frozenset[ActionKey]) -> Decision:
        self._drive = self._drive_state(self._clock())
        self._planned_pid = None
        n = len(self.belief.programs)
        best = self.belief.best(self.cfg.plan_threshold, self.cfg.plan_min_trials)
        if best is None and n == 1:
            # The novel-state path needs the single live program even below threshold.
            only = next(iter(self.belief.programs.values()))
            best = only if only.trials >= 1 else None
        active = None if self._active is None else self.belief.programs.get(self._active)
        inp = ExecutiveInputs(obs, self._plan, active, best, n, vetoes, self.cfg.plan_threshold)
        decision, self._plan = self._executive.decide(inp)
        # The executive's plan branches: "plan" (to the goal) and "learn" via "plan to novel".
        if decision.branch == "plan" or decision.why == "plan to novel":
            self._active = self._planned_pid
        elif decision.branch != "follow":
            self._active = None
        return decision

    def _plan_to(
        self, state: ProgramState, target: Literal["goal", "novel"]
    ) -> list[Action] | None:
        if self._synth is None:
            return None  # no LLM, no programs, no planner
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
        self._planned_pid = pid
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
