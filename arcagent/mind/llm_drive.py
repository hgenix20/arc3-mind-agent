"""The LLM-driven agent: the LLM's model of the game plays first, the tools fix themselves.

``DriveAgent`` is a ``FluidAgent`` that changes only who decides (spec
``docs/superpowers/specs/2026-10-26-llm-driven-agent-design.md``). Without a chat it
plays exactly like ``FluidAgent``.

Modes, one per real move (counted and logged in ``drive_end``):
- ``sim``: a move the model played out (a search plan step, a followed plan step, an LLM
  plan step the play-out predicted).
- ``experiment``: an LLM test, or the first LLM plan step the model can't predict.
- ``fallback``: a slice 1 move nobody modeled (a test, novelty, a forced or commit
  move, B0) taken because no turn could run (none due, timed out, turns off, late).

Turns (changes from the fluid executive):
- No per-level or per-game caps and no spacing. A turn may start until ``DRIVE_CUTOFF``
  of the game's budget is left; it waits up to ``DRIVE_TURN_S`` for the reply.
- A slice 1 choice that isn't modeled wakes the LLM (``no modeled move``) whatever the
  other rules say, so the LLM picks experiments, not the tester.
- The prompt carries the LLM's own model text from its last reply.

Applying a reply: its fact claims go into memory first (``game_model.seed_claims``:
observed facts win), its model text and rule are kept, then the plan is played out in
the model (``game_model.play_out``; the rule runs in this game's tools worker). The plan
is cut after the first step the model can't predict; that step is the experiment and a
wake follows it. When the play-out reaches a goal, the search is asked for the same goal:
no plan, or a longer one, is a ``search`` failure, and a fix turn (``tool_fix.try_fix``)
runs on the pool while the LLM's plan is played. At most one fix per tool per level.

Tools: every search goes through ``toolbox.call_tool`` with ``SceneActions``. A patched
search that errors is rolled back at once; one that finds no plan where the version
before it finds one is rolled back too. A search plan whose moves all played out as
predicted becomes a regression case (its arguments are snapshotted when it is found).

Events: ``tool_fail`` (step, tool, expected, got), ``tool_fix`` (step, tool, adopted,
reason), ``tool_rollback`` (step, tool, reason, now), ``drive_end`` (modes, fixes,
rollbacks, claims).
"""

from __future__ import annotations

import pickle
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import replace
from typing import Any

from arcagent.core.types import Action, Observation
from arcagent.mind.contract import Reply
from arcagent.mind.facts import FactMemory
from arcagent.mind.fluid import NOVEL_WHY, FluidAgent, _Choice, wake_reason  # pyright: ignore[reportPrivateUsage]
from arcagent.mind.game_model import PlayOut, RuleFn, play_out, seed_claims
from arcagent.mind.goals import Goal, goal_sigs, mover_sigs
from arcagent.mind.objects import Scene
from arcagent.mind.search import SearchResult, search
from arcagent.mind.tool_fix import FixResult, ToolFailure, try_fix
from arcagent.mind.toolbox import TOOLBOX, SceneActions, ToolWorker, call_tool
from arcagent.mind.turn import TurnInput

DRIVE_TURN_S = 90.0  # the longest a move waits for its turn
DRIVE_MAX_TOKENS = 2048  # reply budget per drive turn (the model, facts and a plan)
DRIVE_CUTOFF = 0.1  # no turn starts with less than this share of the game's budget left
RULE_S = 2.0  # deadline per rule call in the tools worker
NO_MODELED_MOVE = "no modeled move"
EXPERIMENT_RESULT = "experiment result"
# Wakes that run even while a modeled plan holds: the LLM looks before a level's first
# move, after a game over, after its model mispredicted, and after each experiment.
FORCED_WAKES = ("level start", "game over", "miss that matters", EXPERIMENT_RESULT)
_UNMODELED_BRANCHES = frozenset({"learn", "fallback", "forced", "commit"})
_UNMODELED_WHYS = frozenset({NOVEL_WHY, "sequence try"})


def modeled(choice: _Choice) -> bool:
    """Whether the model played this slice 1 choice out (a plan or follow step)."""
    if choice.branch in _UNMODELED_BRANCHES or choice.why in _UNMODELED_WHYS:
        return False
    return choice.branch in ("plan", "follow")


class DriveAgent(FluidAgent):
    """See the module docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_text = ""
        self._rule_src = ""
        self._worker: ToolWorker | None = ToolWorker() if self._chat is not None else None
        self._search_fn = self._toolbox_search
        self._exp_in: int | None = None  # LLM plan steps left before the experiment step
        self._mode_counts: Counter[str] = Counter()
        self._snap = b""  # pickled args of this move's found search, for a regression case
        self._last_search: SearchResult | None = None
        self._force: str | None = None  # a wake the next move owes the LLM
        self._claims = 0
        self._fixes: list[tuple[str, Future[FixResult]]] = []
        self._fixed_level: set[tuple[str, int]] = set()
        self._fix_count = 0
        self._rollbacks = 0
        self._case: tuple[bytes, SearchResult] | None = None  # the followed plan's search

    # -- tools ----------------------------------------------------------------

    def _scene_actions(self) -> SceneActions:
        offsets = self._clicks._offset  # pyright: ignore[reportPrivateUsage]
        return SceneActions(tuple(self._simple), self._can_click, offsets)

    def _toolbox_search(
        self,
        scene: Scene,
        _actions: Callable[[Scene], list[Action]],
        mem: FactMemory,
        goals: Sequence[Goal],
        **kwargs: Any,
    ) -> SearchResult:
        args = (scene, self._scene_actions(), mem, list(goals))
        res, version, err = call_tool("search", self._worker, *args, **kwargs)
        if err and not err.startswith("unpicklable"):
            self._rollback("search", f"patched search failed: {err[:200]}")
        elif version and res.stop != "found":
            orig = search(*args, **kwargs)
            if orig.stop == "found":
                self._rollback("search", "original finds a plan the patch missed")
                res = orig
        if res.stop == "found" and res.plan and self._chat is not None:
            try:
                self._snap = pickle.dumps((args, kwargs))
            except Exception:  # an unpicklable argument only costs the regression case
                self._snap = b""
        return res

    def _rollback(self, tool: str, reason: str) -> None:
        now = TOOLBOX.rollback(tool, reason)
        self._rollbacks += 1
        self._log.event("tool_rollback", step=self._action_no, tool=tool, reason=reason, now=now)

    def _rule_fn(self) -> RuleFn | None:
        src, worker = self._rule_src, self._worker
        if not src or worker is None:
            return None

        def rule(
            objects: list[dict[str, Any]], action: dict[str, int]
        ) -> list[dict[str, Any]] | None:
            ok, got = worker.run_rule(src, objects, action, RULE_S)
            return got if ok and isinstance(got, list) else None

        return rule

    def _poll_fixes(self) -> None:
        keep: list[tuple[str, Future[FixResult]]] = []
        for tool, fut in self._fixes:
            if not fut.done():
                keep.append((tool, fut))
                continue
            try:
                res = fut.result()
            except Exception as exc:  # a fix job that blew up is a failed fix
                res = FixResult(None, f"{type(exc).__name__}: {exc}"[:200])
            if res.adopted is not None:
                self._fix_count += 1
            self._log.event(
                "tool_fix", step=self._action_no, tool=tool, adopted=res.adopted, reason=res.reason
            )
        self._fixes = keep

    # -- the move ---------------------------------------------------------------

    def choose(self, obs: Observation) -> Action:
        self._snap = b""
        self._poll_fixes()
        return super().choose(obs)

    def _choose(self, obs: Observation, scene: Scene) -> _Choice:
        choice = super()._choose(obs, scene)
        self._mode_counts[self._mode_of(choice)] += 1
        if choice.branch == "plan" and choice.why.startswith("goal") and self._snap:
            res = self._last_search
            if res is not None:
                self._case = (self._snap, res)
        return choice

    def _mode_of(self, choice: _Choice) -> str:
        if self._llm_step:
            if self._exp_in is not None:
                if self._exp_in == 0:
                    self._exp_in = None
                    self._force = EXPERIMENT_RESULT
                    return "experiment"
                self._exp_in -= 1
            return "sim"
        if choice.test is not None and choice.why.startswith("llm test"):
            self._force = EXPERIMENT_RESULT
            return "experiment"
        return "sim" if modeled(choice) else "fallback"

    def _search(
        self, scene: Scene, goals: list[Goal], goal_test: Any = None, **kwargs: Any
    ) -> SearchResult:
        res = super()._search(scene, goals, goal_test, **kwargs)
        self._last_search = res
        return res

    def _check_plan(self, pred: Any, before: Scene, after: Scene, keys: Any, grid: Any) -> None:
        had = bool(self._plan)
        super()._check_plan(pred, before, after, keys, grid)
        if self._case is None:
            return
        if "miss that matters" in self._notes:
            self._case = None
        elif had and not self._plan:
            snap, res = self._case
            self._case = None
            try:
                args, kwargs = pickle.loads(snap)
            except Exception:  # a bad snapshot costs the case only
                return
            TOOLBOX.add_case("search", args, kwargs, res)

    # -- turns -------------------------------------------------------------------

    def _wake_reason(self, choice: _Choice) -> str | None:
        if self._turns_off:
            return None
        if self._force is not None:
            self._pending.add(self._force)
            self._force = None
        forced = next((r for r in FORCED_WAKES if r in self._pending), None)
        if forced is not None:
            return forced
        if self._llm_step or modeled(choice):
            return None
        if self._chat is not None:
            self._pending.add(NO_MODELED_MOVE)
        return wake_reason(self._pending) or (
            NO_MODELED_MOVE if NO_MODELED_MOVE in self._pending else None
        )

    def _maybe_turn(
        self, obs: Observation, scene: Scene, choice: _Choice, llm_step: bool, late: bool
    ) -> tuple[_Choice, bool]:
        self._pending |= self._notes
        reason = self._wake_reason(choice)
        chat, pool = self._chat, self._pool
        if reason is None or chat is None or pool is None:
            return choice, llm_step
        now = self._clock()
        if now >= self._deadline - DRIVE_CUTOFF * (self._deadline - self._started):
            return choice, llm_step
        call = self._calls
        self._calls += 1
        self._turns_level += 1
        self._stretch_from = self._action_no
        inp = self._turn_input(obs, scene, reason)
        self._pending.clear()
        self._wake_text = ""
        self._log.event("wake", step=self._action_no, reason=reason)
        self._wake = reason
        got = self._await_turn(chat, pool, inp, call, min(DRIVE_TURN_S, self._deadline - now))
        self._turn_done = self._action_no
        if got is None:
            return choice, llm_step
        new, new_llm = self._apply(got[0], call, scene, got[1], late)
        return (choice, llm_step) if new is None else (new, new_llm)

    def _turn_input(self, obs: Observation, scene: Scene, reason: str) -> TurnInput:
        inp = super()._turn_input(obs, scene, reason)
        return replace(inp, model=self._model_text, drive=True)

    def _apply(
        self, reply: Reply, call: int, scene: Scene, seconds: float, late: bool
    ) -> tuple[_Choice | None, bool]:
        self._claims += len(seed_claims(self.mem, reply.claims))
        if reply.model:
            self._model_text = reply.model
        if reply.rule:
            self._rule_src = reply.rule
        self._exp_in = None
        if reply.plan:
            reply = self._played_out(reply, scene)
        return super()._apply(reply, call, scene, seconds, late)

    def _played_out(self, reply: Reply, scene: Scene) -> Reply:
        """The reply with its plan cut after the first step the model can't predict."""
        goals = list(reply.goals) or self.goals.open()
        sigs: set[str] = set(mover_sigs(self.mem))
        for g in goals:
            sigs |= goal_sigs(g, scene)
        protect = frozenset(sigs)
        actions = [p.action for p in reply.plan]
        po = play_out(scene, actions, self.mem, goals, protect, rule=self._rule_fn())
        if po.unknown_at is not None:
            self._exp_in = po.unknown_at
            return replace(reply, plan=reply.plan[: po.unknown_at + 1])
        if po.reached is not None:
            self._check_search(scene, po, protect)
            return replace(reply, plan=reply.plan[: po.steps])
        return reply

    def _check_search(self, scene: Scene, po: PlayOut, protect: frozenset[str]) -> None:
        """Ask search for the goal the LLM's plan reaches; a miss starts a fix turn."""
        goal = po.reached
        assert goal is not None
        key = ("search", self._level)
        if (
            self._chat is None
            or self._pool is None
            or self._worker is None
            or key in self._fixed_level
        ):
            return
        res = self._search(scene, [goal])
        if res.stop == "found" and len(res.plan) <= po.steps:
            return
        got = "no plan" if res.stop != "found" else f"a plan of {len(res.plan)} moves"
        expected = (
            f"a plan of at most {po.steps} moves reaching goal {goal.id()} "
            f"exists (the model plays one out); search stopped with {res.stop!r}"
        )
        self._fixed_level.add(key)
        self._log.event(
            "tool_fail", step=self._action_no, tool="search", expected=expected, got=got
        )
        mem = pickle.loads(pickle.dumps(self.mem))
        steps = po.steps

        def check(found: Any) -> bool:
            if getattr(found, "stop", None) != "found" or len(found.plan) > steps:
                return False
            acts = [s.action for s in found.plan]
            return play_out(scene, acts, mem, [goal], protect).reached is not None

        args = (scene, self._scene_actions(), mem, [goal])
        failure = ToolFailure("search", args, {"kin": self._kin}, expected, got, check)
        try:
            fut = self._pool.submit(
                try_fix,
                self._chat,
                failure,
                self._worker,
                game=self.game,
                seed=self.seed * 7919 + self._calls,
            )
        except RuntimeError:  # a shut-down pool costs the fix
            return
        self._fixes.append(("search", fut))

    def close(self) -> None:
        if self._closed:
            return
        self._poll_fixes()
        self._log.event(
            "drive_end",
            step=self._action_no,
            modes=dict(sorted(self._mode_counts.items())),
            fixes=self._fix_count,
            rollbacks=self._rollbacks,
            claims=self._claims,
            turns=self._calls,
            at=round(time.time(), 1),
        )
        super().close()
        if self._worker is not None:
            self._worker.close()
