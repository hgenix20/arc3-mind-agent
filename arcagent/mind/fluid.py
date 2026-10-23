"""The fluid agent: one game's per-move policy over facts, goal guesses and short search.

No LLM and no program sandbox. It wires the slice 1 modules (``objects``, ``noise``,
``facts``, ``predictor``, ``goals``, ``search``, ``tester``) into a ``Policy``.

Per ``choose(obs)``, in order:

1. ``GAME_OVER``: learn the death from the move that caused it (``game_over=True``,
   the tracker diff of the final frame), log ``game_over``, reset the tracker and the
   noise tracks, clear the plan and return RESET (branch ``reset``). ``NOT_PLAYED``
   returns RESET too and learns nothing. The frame after a RESET learns nothing: the
   start frame is not a consequence of any rule.
2. ``levels_completed`` rose: record the ``level_up`` event fact, ``goals.level_up``
   on the frame before, log ``level_up``, then reset the tracker, the noise tracks and
   ``FactMemory`` positions, and clear the plan, the level's scene history, ``tried``
   and the frontier. Facts and goal statuses stay. The new screen teaches nothing.
3. Otherwise update tracker and noise, strip noisy things from both scenes and the
   diff, ``learn`` from the transition and ``check`` the previous prediction (its
   misses feed only the log and the plan stop; ``learn`` charges facts itself). Then
   ``goals.note_reached`` on the new scene.
4. A plan stops only on a miss that matters (``plan_breaks``): a missed or newly broken
   fact about a mover sig or one of the plan goal's sigs (``self`` and ``any`` subjects
   count; spawn and other side objects don't), or a mover that ended somewhere other
   than predicted. A game over or level change always ends the plan (steps 1 and 2).
   Every other miss is only logged and learned. A plan with steps left gives the next
   step (branch ``follow``). A plan or follow step whose plan breaks while its goal
   doesn't hold is a failed attempt; after ``STALL_AFTER`` failed attempts in one level
   the goal is stalled (``GoalBook.stall``, goal event ``stalled``) and search stops
   offering it. A level change reopens stalled goals (goal event ``open``) and clears
   the counts. Plan steps are predicted with ``assume_still`` set to the
   mover sigs plus the goal's sigs, so the check covers what the plan relies on.
5. Else ``goals.propose`` and ``search`` over ``goals.open()``; a found plan starts
   (branch ``plan``, why ``goal <id>``). A goal that already holds at the root is
   treated as reached (``note_reached``) and the search runs once more without it.
6. Else ``choose_test`` with the search's frontier (branch ``learn``, why the test's
   question).
7. Else B0 with every vetoed candidate excluded (branch ``fallback``).
8. ``veto`` runs on each option before it is taken; a refused option is logged as a
   ``veto`` event and the next option in the same order is tried. B0 is last and is
   never vetoed. The chosen action's prediction is kept for step 3 of the next call.

Deviations the brief leaves open, decided here:

- A plan step is refused only for ``predicted game_over`` or ``known no effect
  here``. ``settled test`` guards tests: a planned move through a scene the agent
  already acted in (every replan after a RESET) is not a test and must not be refused.
- ``rule_out(history)`` runs after a ``propose`` that added goals. Goals that were
  already open were checked against every earlier frame by ``note_reached`` while
  they were open, so rechecking them each move would only cost time. History keeps
  one scene per distinct ``Scene.key()``.
- The fallback excludes every candidate ``veto`` refuses (in the action record's
  ``vetoes``), so B0 does not walk into a known death; no event is logged for those.

Past the game or run deadline: no search, no tests, no vetoes; follow the plan or B0.

Failure guard: any ``Exception`` in the mind side of a move is logged as
``agent_error`` (truncated to ``MAX_ERROR_CHARS``) and that move goes to B0 (RESET on
a GAME_OVER or NOT_PLAYED frame). If it happened before the new scene was built, the
tracker restarts so the next frame learns nothing. B0's bookkeeping and the log write
run outside the guard.

Log: action records keep the v1 fields (``programs`` scores the previous move's
prediction, one entry per used fact; ``entropy`` is the frontier size; ``active`` and
``wake`` are null, ``voi`` is empty). Events: ``fact`` (``step, key, status, line``)
when a fact is new or changes status, ``goal`` (``step, id, status, source``),
``test`` (``step, action, question``), ``noise`` (``step, ids, cells``), ``veto``
(``step, action, reason``), ``level_up`` and ``game_over`` (``step, level``) with
``step`` the index of the move that caused them, ``agent_error`` and ``game_end``.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from arcagent.core.types import COORD_ACTION_ID, RESET_ID, Action, Observation, Phase
from arcagent.mind.agent import MAX_ERROR_CHARS
from arcagent.mind.baseline import Baseline
from arcagent.mind.config import MindConfig
from arcagent.mind.facts import FactKey, FactMemory, Status, learn
from arcagent.mind.goals import Goal, GoalBook, goal_sigs, mover_sigs
from arcagent.mind.noise import NoiseMask
from arcagent.mind.objects import Cell, Scene, SceneDiff, Thing, Tracker
from arcagent.mind.perception import candidate_actions, frame_hash
from arcagent.mind.predictor import Prediction, check, predict
from arcagent.mind.runlog import RunLog
from arcagent.mind.search import PlanStep, search
from arcagent.mind.tester import GAME_OVER, NO_EFFECT, Tried, choose_test, veto
from arcagent.mind.types import ActionKey, Branch

STALL_AFTER = 2  # failed attempts at one goal in one level before it is set aside
GOAL_SOURCE = "heuristic"  # slice 1 goals come from the heuristics in GoalBook.propose
_PLAN_VETOES = frozenset({GAME_OVER, NO_EFFECT})  # the veto reasons a plan step honors
_EMPTY_DIFF = SceneDiff((), 0)

__all__ = ["MAX_ERROR_CHARS", "FluidAgent"]


@dataclass(frozen=True, slots=True)
class _Choice:
    action: Action
    branch: Branch
    why: str


def _strip(scene: Scene, ids: frozenset[int]) -> Scene:
    if not ids:
        return scene
    return replace(scene, things=tuple(t for t in scene.things if t.id not in ids))


def _strip_diff(diff: SceneDiff, ids: frozenset[int]) -> SceneDiff:
    if not ids:
        return diff
    return SceneDiff(tuple(c for c in diff.changes if c.id not in ids), diff.stray_cells)


def plan_breaks(  # noqa: PLR0917  # each input is a distinct part of the rule
    pred: Prediction,
    before: Scene,
    after: Scene,
    keys: Iterable[FactKey],
    movers: frozenset[str],
    protected: frozenset[str],
) -> bool:
    """Whether a plan must stop after the move ``pred`` predicted.

    True when a key in ``keys`` (missed or newly broken facts) is about a protected sig,
    a ``self`` or an ``any`` subject (``SPAWN`` never is), or when a thing whose sig is in
    ``movers`` is not where ``pred.scene`` put it in ``after`` (same id, bbox top-left
    compared, so a sprite that animates in place still matches; a thing present in only
    one of the two counts as elsewhere).
    """
    for _, subject, _ in keys:
        if subject in ("self", "any") or subject in protected:
            return True
    if pred.scene is None:
        return True
    expect = pred.scene.by_id()
    now = after.by_id()
    for t in before.things:
        if t.sig not in movers:
            continue
        e, a = expect.get(t.id), now.get(t.id)
        if (e is None) != (a is None) or (
            e is not None and a is not None and (e.x, e.y) != (a.x, a.y)
        ):
            return True
    return False


class _Clicks:
    """One click cell per thing: the cell nearest its centroid, cached by shape."""

    def __init__(self) -> None:
        self._offset: dict[str, Cell] = {}

    def cell(self, t: Thing) -> Cell:
        off = self._offset.get(t.shape)
        if off is None:
            cells = sorted(t.cells)
            cx = sum(x for x, _ in cells) / len(cells)
            cy = sum(y for _, y in cells) / len(cells)
            x, y = min(cells, key=lambda c: ((c[0] - cx) ** 2 + (c[1] - cy) ** 2, c))
            off = (x - t.x, y - t.y)
            self._offset[t.shape] = off
        return t.x + off[0], t.y + off[1]


class FluidAgent:
    """One game's fluid agent. Implements ``arcagent.agent.play.Policy`` and ``close``.

    ``clock`` is injectable for tests. ``mem`` and ``goals`` are public for inspection.
    """

    def __init__(
        self,
        game: str,
        cfg: MindConfig,
        *,
        seed: int,
        log: RunLog,
        game_deadline: float,
        run_deadline: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.game = game
        self.cfg = cfg
        self.seed = seed
        self._log = log
        self._deadline = min(game_deadline, run_deadline)
        self._clock = clock
        self.mem = FactMemory()
        self.goals = GoalBook()
        self._tracker = Tracker()
        self._noise = NoiseMask()
        self._baseline = Baseline(seed)
        self._clicks = _Clicks()

        self._action_no = 0
        self._level = 0
        self._fresh = True  # the next frame starts a level or follows a RESET
        self._last_action: Action | None = None
        self._last_full: Scene | None = None  # the tracker's scene, noise included
        self._last_scene: Scene | None = None  # the same scene with noise stripped
        self._pred: Prediction | None = None
        self._plan: list[PlanStep] = []
        self._plan_goal: Goal | None = None
        self._failed: dict[Goal, int] = {}  # failed plan attempts per goal, this level
        self._history: dict[tuple[tuple[str, int, int], ...], Scene] = {}
        self._tried: Tried = set()
        self._frontier: tuple[tuple[str, str], ...] = ()
        self._simple: list[Action] = []
        self._can_click = False
        self._tests = 0
        self._vetoes_total = 0
        self._closed = False
        # Per-move scratch, read by the action record even after a failure.
        self._programs: list[dict[str, object]] = []
        self._vetoes: set[ActionKey] = set()
        self._scene_ok = False

    # -- the move ------------------------------------------------------------

    def choose(self, obs: Observation) -> Action:
        """One action for ``obs``; see the module docstring for the order."""
        frame = frame_hash(obs.grid)
        self._programs, self._vetoes, self._scene_ok = [], set(), False
        self._baseline.observe(obs)
        try:
            choice = self._mind_step(obs)
        except Exception as exc:  # a mind-side failure costs this move's reasoning, not the game
            self._log.event(
                "agent_error",
                step=self._action_no,
                error=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
            )
            self._plan, self._pred = [], None
            if not self._scene_ok:
                self._restart()
            if obs.phase in (Phase.GAME_OVER, Phase.NOT_PLAYED):
                choice = _Choice(Action(RESET_ID), "reset", "agent_error")
            else:
                exclude = frozenset(self._vetoes)
                choice = _Choice(self._baseline.choose(obs, exclude), "fallback", "agent_error")
        a = choice.action
        if choice.branch != "fallback":
            self._baseline.note(obs, a)
        self._log.action(
            level=obs.levels_completed,
            step=self._action_no,
            frame_hash=frame,
            action=[a.id, a.x, a.y],
            branch=choice.branch,
            why=choice.why,
            programs=self._programs,
            vetoes=[list(v) for v in sorted(self._vetoes)],
            voi={},
            entropy=len(self._frontier),
            wake=None,
            active=None,
        )
        self._last_action = a
        if a.id == RESET_ID:
            self._fresh = True
        self._action_no += 1
        return a

    def _restart(self) -> None:
        """Forget the tracked frame and noise tracks: the next frame is a fresh start."""
        self._tracker.reset()
        self._noise.reset_tracks()
        self._fresh = True
        self._pred = None

    def _mind_step(self, obs: Observation) -> _Choice:
        move = self._action_no - 1  # index of the move that produced obs
        if obs.phase == Phase.GAME_OVER:
            self._game_over(obs, move)
            return _Choice(Action(RESET_ID), "reset", "game over")
        if obs.phase == Phase.NOT_PLAYED:
            self._restart()
            self._plan = []
            return _Choice(Action(RESET_ID), "reset", "not played")
        self._simple = [a for a in candidate_actions(obs) if a.id != COORD_ACTION_ID]
        self._can_click = COORD_ACTION_ID in obs.available_actions
        if self._fresh or self._last_scene is None or self._last_action is None:
            scene = self._start(obs)
        elif obs.levels_completed > self._level:
            self._level_up(obs, move, self._last_scene, self._last_action)
            scene = self._start(obs)
        else:
            scene = self._transition(obs, move, self._last_scene, self._last_action)
        self._level = obs.levels_completed
        return self._decide(obs, scene)

    def _start(self, obs: Observation) -> Scene:
        """First frame of a level or after a RESET: no diff, nothing learned."""
        self._tracker.reset()
        full, _ = self._tracker.update(obs.grid, self._noise.cells)
        scene = _strip(full, self._noise.noisy_ids)
        self._keep(full, scene)
        self._fresh, self._pred = False, None
        return scene

    def _keep(self, full: Scene, scene: Scene) -> None:
        self._last_full, self._last_scene = full, scene
        self._history.setdefault(scene.key(), scene)
        self._scene_ok = True

    def _game_over(self, obs: Observation, move: int) -> None:
        prev_full, action = self._last_full, self._last_action
        if not self._fresh and prev_full is not None and action is not None:
            full, diff = self._tracker.update(obs.grid, self._noise.cells)
            noisy = self._noise.noisy_ids
            if diff is not None:
                before = _strip(prev_full, noisy)
                snap = self._statuses()
                keys = learn(
                    self.mem,
                    before,
                    action,
                    _strip_diff(diff, noisy),
                    move=move,
                    level_up=False,
                    game_over=True,
                    after=_strip(full, noisy),
                )
                self._log_facts(keys, snap, move)
            self._log.event("game_over", step=move, level=obs.levels_completed)
        self._restart()
        self._plan = []
        self._scene_ok = True

    def _level_up(self, obs: Observation, move: int, before: Scene, action: Action) -> None:
        snap = self._statuses()
        keys = learn(
            self.mem, before, action, _EMPTY_DIFF, move=move, level_up=True, game_over=False
        )
        self._log_facts(keys, snap, move)
        self._log_goals(self.goals.level_up(before), "confirmed")
        self._log.event("level_up", step=move, level=obs.levels_completed)
        self.reopen_goals()
        self._restart()
        self.mem.new_level()
        self._plan, self._history, self._tried, self._frontier = [], {}, set(), ()

    def _transition(self, obs: Observation, move: int, before_s: Scene, action: Action) -> Scene:
        prev_full = self._last_full
        full, diff = self._tracker.update(obs.grid, self._noise.cells)
        if diff is None or prev_full is None:  # cannot happen after a non-fresh frame
            scene = _strip(full, self._noise.noisy_ids)
            self._keep(full, scene)
            return scene
        flagged = self._noise.observe(prev_full, action.id, diff)
        if flagged:
            self._log.event("noise", step=move, ids=flagged, cells=len(self._noise.cells))
        noisy = self._noise.noisy_ids
        before = _strip(prev_full, noisy)
        after = _strip(full, noisy)
        self._keep(full, after)
        snap = self._statuses()
        keys = learn(
            self.mem,
            before,
            action,
            _strip_diff(diff, noisy),
            move=move,
            level_up=False,
            game_over=False,
            after=after,
        )
        broke = self._log_facts(keys, snap, move)
        pred = self._pred
        if pred is not None:
            _, missed = check(pred, before_s, diff, ignore=noisy, after=after)
            self._score(pred, set(missed))
            goal = self._plan_goal
            if goal is not None:
                movers, protected = self._protected(before_s)
                if plan_breaks(pred, before_s, after, [*missed, *broke], movers, protected):
                    self._plan = []
                    if not goal.holds(after):
                        self.note_failed(goal)
        self._log_goals(self.goals.note_reached(after), "excluded")
        return after

    # -- deciding ------------------------------------------------------------

    def _decide(self, obs: Observation, scene: Scene) -> _Choice:
        late = self._clock() >= self._deadline
        choice: _Choice | None = None
        if self._plan:
            step = self._plan.pop(0)
            choice = self._offer(scene, step.action, "follow", "plan step", late, plan=True)
        if choice is None and not late:
            choice = self._plan_choice(scene)
        if choice is None and not late:
            choice = self._test_choice(obs, scene)
        if choice is None:
            if not late:
                self._exclude_vetoed(obs, scene)
            action = self._baseline.choose(obs, frozenset(self._vetoes))
            choice = _Choice(action, "fallback", "deadline passed" if late else "b0")
        self._tried.add((scene.key(), choice.action.key()))
        if choice.branch in ("plan", "follow"):
            protected = self._protected(scene)[1]
            self._pred = predict(scene, choice.action, self.mem, assume_still=protected)
        else:
            self._plan_goal = None
            self._pred = predict(scene, choice.action, self.mem)
        return choice

    def note_failed(self, goal: Goal) -> None:
        """Count a failed plan attempt; stall the goal at ``STALL_AFTER`` in this level."""
        n = self._failed.get(goal, 0) + 1
        self._failed[goal] = n
        if n >= STALL_AFTER and self.goals.stall(goal):
            self._log_goals([goal], "stalled")

    def reopen_goals(self) -> None:
        """Level change: stalled goals are back in play and failure counts restart."""
        self._failed.clear()
        self._log_goals(self.goals.reopen_stalled(), "open")

    def _protected(self, scene: Scene) -> tuple[frozenset[str], frozenset[str]]:
        """(mover sigs, mover sigs plus the plan goal's sigs) for the plan rules."""
        movers = mover_sigs(self.mem)
        goal = self._plan_goal
        return movers, movers | (goal_sigs(goal, scene) if goal is not None else frozenset())

    def _offer(
        self, scene: Scene, action: Action, branch: Branch, why: str, late: bool, *, plan: bool
    ) -> _Choice | None:
        """The option as a choice, or None (and a ``veto`` event) when it is refused."""
        if not late:
            reason = veto(scene, action, self.mem, self._tried)
            if reason is not None and (not plan or reason in _PLAN_VETOES):
                self._refuse(action, reason)
                if plan:
                    self._plan = []
                return None
        return _Choice(action, branch, why)

    def _refuse(self, action: Action, reason: str) -> None:
        self._vetoes.add(action.key())
        self._vetoes_total += 1
        self._log.event(
            "veto", step=self._action_no, action=[action.id, action.x, action.y], reason=reason
        )

    def _plan_choice(self, scene: Scene) -> _Choice | None:
        added = self.goals.propose(scene, self.mem, source=GOAL_SOURCE)
        if added:
            self._log_goals(added, "open")
            self._log_goals(self.goals.rule_out(list(self._history.values())), "ruled_out")
        goals = self.goals.open()
        if not goals:
            self._frontier = ()
            return None
        res = search(scene, self._actions, self.mem, goals)
        if res.stop == "found" and not res.plan and res.goal is not None:
            self._log_goals(self.goals.note_reached(scene), "excluded")
            rest = [g for g in self.goals.open() if g != res.goal]
            res = search(scene, self._actions, self.mem, rest)
        self._frontier = res.frontier
        if res.stop != "found" or not res.plan or res.goal is None:
            return None
        first, *rest_steps = res.plan
        self._plan = list(rest_steps)
        self._plan_goal = res.goal
        choice = self._offer(scene, first.action, "plan", f"goal {res.goal.id()}", False, plan=True)
        if choice is None:
            self._plan = []
        return choice

    def _test_choice(self, obs: Observation, scene: Scene) -> _Choice | None:
        test = choose_test(
            scene, candidate_actions(obs), self.mem, self._frontier, self._tried, None
        )
        if test is None:
            return None
        choice = self._offer(scene, test.action, "learn", test.question, False, plan=False)
        if choice is not None:
            self._tests += 1
            a = test.action
            self._log.event(
                "test", step=self._action_no, action=[a.id, a.x, a.y], question=test.question
            )
        return choice

    def _exclude_vetoed(self, obs: Observation, scene: Scene) -> None:
        """Add every candidate ``veto`` refuses to this move's vetoes (no events)."""
        for action in candidate_actions(obs):
            if veto(scene, action, self.mem, self._tried) is not None:
                self._vetoes.add(action.key())

    def _actions(self, scene: Scene) -> list[Action]:
        """Search's actions: the simple actions, then one click per thing when clicks exist."""
        out = list(self._simple)
        if self._can_click:
            out.extend(Action(COORD_ACTION_ID, *self._clicks.cell(t)) for t in scene.things)
        return out

    # -- logging -------------------------------------------------------------

    def _statuses(self) -> dict[FactKey, Status]:
        return {k: f.status for k, f in self.mem.facts.items()}

    def _log_facts(
        self, keys: Iterable[FactKey], before: dict[FactKey, Status], move: int
    ) -> list[FactKey]:
        """Log new facts and status changes; return the keys that turned broken."""
        broke: list[FactKey] = []
        for key in dict.fromkeys(keys):
            fact = self.mem.get(key)
            if fact is None or before.get(key) == fact.status:
                continue
            if fact.status == "broken":
                broke.append(key)
            self._log.event("fact", step=move, key=list(key), status=fact.status, line=fact.line())
        return broke

    def _log_goals(self, goals: Iterable[Goal], status: str) -> None:
        for g in goals:
            self._log.event(
                "goal",
                step=self._action_no,
                id=g.id(),
                status=status,
                source=self.goals.sources.get(g, GOAL_SOURCE),
            )

    def _score(self, pred: Prediction, missed: set[FactKey]) -> None:
        """The ``programs`` entries for the previous move's prediction."""
        out: list[dict[str, object]] = []
        for key in dict.fromkeys(pred.used):
            fact = self.mem.get(key)
            total = 0 if fact is None else fact.hits + fact.misses
            out.append(
                {
                    "id": " ".join(key),
                    "pred": " ".join(key) if fact is None else fact.line(),
                    "matched": key not in missed,
                    "fit": None if fact is None or not total else fact.hits / total,
                }
            )
        self._programs = out

    def close(self) -> None:
        """Log ``game_end`` with fact and goal counts by status. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        facts = Counter(f.status for f in self.mem.facts.values())
        goals = Counter(self.goals.status.values())
        self._log.event(
            "game_end",
            step=self._action_no,
            level=self._level,
            facts=dict(sorted(facts.items())),
            goals=dict(sorted(goals.items())),
            tests=self._tests,
            vetoes=self._vetoes_total,
        )
