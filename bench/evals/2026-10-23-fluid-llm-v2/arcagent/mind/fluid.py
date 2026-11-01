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
   the counts. A plan survives goal-list changes (proposals, rule-outs, other goals
   stalling) and ends only when its own goal leaves the open list or a step breaks it.
   Each search logs a ``search`` event (``step, stop, nodes, depth, goals, frontier,
   protected``). Plan steps are predicted with ``assume_still`` set to the
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

Executive turn (slice 2, only with a ``chat``; without one the agent is slice 1 exactly):

- Wake rules, noted while a move is processed and checked after the move is chosen, in
  ``WAKE_ORDER``: level start (the first frame of the game or of a new level); game
  over (the first frame after the RESET that follows one; the prompt's wake text adds
  ``last attempt ended in game over after N moves``); a miss that matters
  (``plan_breaks`` stopped a plan); plan finished or blocked (a vetoed step or a click
  whose thing is gone); a goal excluded, ruled out or stalled; a fact turning broken;
  no plan and no test (the move fell back to B0); long stretch (``LONG_STRETCH_MOVES``
  moves since the last turn started on this level, or since the level started). No
  wake while a plan holds (the chosen branch is ``plan`` or ``follow``), while a turn
  is in flight, or once turns are off for the game. Spacing: after a turn finishes
  (applied, failed, timed out or stale) the next wake needs ``MIN_MOVES_BETWEEN_TURNS``
  moves, except level start and game over; a wake suppressed this way logs nothing.
- Budgets: ``MAX_TURNS_PER_LEVEL`` and ``MAX_TURNS_PER_GAME`` turns, and no turn
  starts past ``1 - SYNTH_CUTOFF_FRACTION`` (75%) of the game's budget (agent start to
  the game or run deadline). The turn's seed is ``seed * 1000 + call``.
- A turn runs ``turn.take_turn`` on ``pool`` with the frame the move was chosen on;
  the game never waits. The future is polled at the start of each decision. A turn
  older than ``TURN_DEADLINE_S`` is abandoned (``exec_failed`` reason ``timeout``) and
  its result ignored. ``ImagesRejected`` logs ``exec_failed`` ``images rejected`` and
  turns stop for the game. Any other failure logs ``exec_failed`` with the reason and
  costs nothing else. A turn asked before the last level change or game over (the
  level epoch) is logged as ``exec_stale`` and never applied: dropped at the next poll,
  finished or not, so the new level or attempt can wake at once.
- Applying a reply: goals enter ``GoalBook`` with source ``"llm"`` (a goal already
  known keeps its status and source) and every open one the reply named moves ahead
  of the other open goals; new goals then go through ``rule_out`` and
  ``note_reached`` like heuristic ones. A plan replaces any current plan; each step
  is taken as a ``plan`` move (why ``llm plan``), predicted with movers plus the
  reply's goal sigs assumed protected; a step the facts can't predict is still taken
  (a test in all but name) and the plan stops only on a miss that matters about those
  sigs. Plan steps honor the same vetoes as slice 1 plan steps. A test becomes the
  next move unless vetoed (branch ``learn``, why ``llm test: <question>``, a ``test``
  event). Ids came from the asked-on scene: a click is re-mapped to the current
  thing with the clicked thing's sig nearest its old position, and dropped (a
  ``veto`` event, reason ``click target gone``) when no thing has that sig.

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
With a chat: ``wake`` (``step, reason``; the action record's ``wake`` holds the same
reason), ``exec_turn`` (``step, call, seconds, goals, plan_len, test, dropped``),
``exec_failed`` (``step, call, reason``) and ``exec_stale`` (``step, call,
asked_level``); ``game_end`` adds ``turns``.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable, Collection, Iterable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

from arcagent.core.types import COORD_ACTION_ID, RESET_ID, Action, Grid, Observation, Phase
from arcagent.mind.agent import MAX_ERROR_CHARS, SYNTH_CUTOFF_FRACTION
from arcagent.mind.baseline import Baseline
from arcagent.mind.config import MindConfig
from arcagent.mind.contract import PlanItem, Reply
from arcagent.mind.facts import FactKey, FactMemory, Status, learn
from arcagent.mind.goals import Goal, GoalBook, GoalStatus, goal_sigs, mover_sigs
from arcagent.mind.noise import NoiseMask
from arcagent.mind.objects import Cell, Scene, SceneDiff, Thing, Tracker, click_cell
from arcagent.mind.perception import candidate_actions, frame_hash
from arcagent.mind.predictor import Prediction, check, predict
from arcagent.mind.runlog import RunLog
from arcagent.mind.search import PlanStep, SearchResult, search
from arcagent.mind.tester import GAME_OVER, NO_EFFECT, Tried, choose_test, question, veto
from arcagent.mind.turn import MOVE_LOG_LINES, TurnInput, take_turn
from arcagent.mind.types import ActionKey, Branch
from arcagent.mind.vchat import ImagesRejected, VisionChat

STALL_AFTER = 2  # failed attempts at one goal in one level before it is set aside
GOAL_SOURCE = "heuristic"  # slice 1 goals come from the heuristics in GoalBook.propose
LLM_SOURCE = "llm"  # goals named by the executive turn
_PLAN_VETOES = frozenset({GAME_OVER, NO_EFFECT})  # the veto reasons a plan step honors
_EMPTY_DIFF = SceneDiff((), 0)

TURN_DEADLINE_S = 30.0  # a turn older than this is abandoned
TURN_MAX_TOKENS = 1024  # reply budget per turn (the HTTP chat's max_tokens)
MAX_TURNS_PER_LEVEL = 8
MAX_TURNS_PER_GAME = 40
MIN_MOVES_BETWEEN_TURNS = 6  # moves from a finished turn to the next wake (spacing)
LONG_STRETCH_MOVES = 40  # moves without a turn on one level before "long stretch" wakes
TURN_SEED_STRIDE = 1000  # a turn's seed is seed * TURN_SEED_STRIDE + call
WAKE_ORDER = (
    "level start",
    "game over",
    "miss that matters",
    "plan finished",
    "plan blocked",
    "goal excluded",
    "goal ruled_out",
    "goal stalled",
    "fact broken",
    "no plan and no test",
    "long stretch",
)
_UNSPACED = frozenset({"level start", "game over"})  # wakes exempt from the spacing rule
_GOAL_WAKES = frozenset({"excluded", "ruled_out", "stalled"})
_DIRS: dict[tuple[int, int], str] = {
    (-1, 0): "left",
    (1, 0): "right",
    (0, -1): "up",
    (0, 1): "down",
}

__all__ = [
    "LONG_STRETCH_MOVES",
    "MAX_ERROR_CHARS",
    "MAX_TURNS_PER_GAME",
    "MAX_TURNS_PER_LEVEL",
    "MIN_MOVES_BETWEEN_TURNS",
    "TURN_DEADLINE_S",
    "TURN_MAX_TOKENS",
    "WAKE_ORDER",
    "FluidAgent",
    "remap_click",
    "wake_reason",
]


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
    *,
    unknown_ok: bool = False,
) -> bool:
    """Whether a plan must stop after the move ``pred`` predicted.

    True when a key in ``keys`` (missed or newly broken facts) is about a protected sig,
    a ``self`` or an ``any`` subject (``SPAWN`` never is), or when a thing whose sig is in
    ``movers`` and that ``pred`` actually predicted (it has an ``effects`` entry; a thing
    assumed still has none, and its move fact may postdate the prediction) is not where
    ``pred.scene`` put it in ``after`` (same id, bbox top-left
    compared, so a sprite that animates in place still matches; a thing present in only
    one of the two counts as elsewhere). A prediction with unknowns (no scene) breaks the
    plan, unless ``unknown_ok`` (an LLM plan step taken as a test): then only the keys
    count.
    """
    for _, subject, _ in keys:
        if subject in ("self", "any") or subject in protected:
            return True
    if pred.scene is None:
        return not unknown_ok
    expect = pred.scene.by_id()
    now = after.by_id()
    for t in before.things:
        if t.sig not in movers or t.id not in pred.effects:
            continue
        e, a = expect.get(t.id), now.get(t.id)
        if (e is None) != (a is None) or (
            e is not None and a is not None and (e.x, e.y) != (a.x, a.y)
        ):
            return True
    return False


def wake_reason(notes: Collection[str]) -> str | None:
    """The first reason in ``WAKE_ORDER`` among ``notes``, or None."""
    return next((r for r in WAKE_ORDER if r in notes), None)


def remap_click(
    action: Action, asked: Scene, now: Scene, cell_of: Callable[[Thing], Cell]
) -> Action | None:
    """``action`` for the current scene; a click follows the clicked thing by sig.

    A simple action comes back unchanged. A click finds the thing it hit in ``asked``
    (the scene the reply's ids came from), then the thing of the same sig in ``now``
    nearest that thing's top-left (Manhattan, ties by id), and clicks ``cell_of`` it.
    None when the click hit nothing in ``asked`` or no thing in ``now`` has the sig.
    """
    if action.id != COORD_ACTION_ID:
        return action
    point = (action.x, action.y)
    src = next((t for t in asked.things if point in t.cells), None)
    if src is None:
        return None
    same = [t for t in now.things if t.sig == src.sig]
    if not same:
        return None
    best = min(same, key=lambda t: (abs(t.x - src.x) + abs(t.y - src.y), t.id))
    return Action(COORD_ACTION_ID, *cell_of(best))


def _action_name(action: Action, scene: Scene) -> str:
    """``A3``, or ``click #<id>`` naming the thing hit (``click (x,y)`` when none)."""
    if action.id != COORD_ACTION_ID:
        return f"A{action.id}"
    point = (action.x, action.y)
    hit = next((t for t in scene.things if point in t.cells), None)
    return f"click #{hit.id}" if hit is not None else f"click ({action.x},{action.y})"


def _outcome(diff: SceneDiff, before: Scene) -> str:
    """A few words on what a move changed, e.g. ``c1 moved left`` or ``no change``."""
    by_id = before.by_id()
    words: list[str] = []
    for c in diff.changes:
        old = by_id.get(c.id)
        color = old.color if old is not None else c.color
        if c.kind == "moved":
            words.append(f"c{color} moved {_DIRS.get((c.dx, c.dy), f'by ({c.dx},{c.dy})')}")
        elif c.kind == "recolored":
            words.append(f"c{color} became c{c.color}")
        elif c.kind in ("disappeared", "appeared", "reshaped"):
            words.append(f"c{color} {c.kind}")
    if not words:
        return "no change"
    extra = f" +{len(words) - 2} more" if len(words) > 2 else ""
    return ", ".join(words[:2]) + extra


@dataclass(frozen=True, slots=True)
class _Turn:
    """One turn in flight: its future, call number, level epoch and asked-on scene."""

    future: Future[tuple[Reply, float]]
    call: int
    epoch: int
    level: int
    started: float
    scene: Scene


def _run_turn(chat: VisionChat, inp: TurnInput, seed: int) -> tuple[Reply, float]:
    """The pool job: ``take_turn`` plus its wall-clock seconds."""
    t0 = time.perf_counter()
    reply = take_turn(chat, inp, seed=seed)
    return reply, time.perf_counter() - t0


class _Clicks:
    """One click cell per thing: the cell nearest its centroid, cached by shape."""

    def __init__(self) -> None:
        self._offset: dict[str, Cell] = {}

    def cell(self, t: Thing) -> Cell:
        off = self._offset.get(t.shape)
        if off is None:
            x, y = click_cell(t)
            off = (x - t.x, y - t.y)
            self._offset[t.shape] = off
        return t.x + off[0], t.y + off[1]


class FluidAgent:
    """One game's fluid agent. Implements ``arcagent.agent.play.Policy`` and ``close``.

    ``clock`` is injectable for tests. ``mem`` and ``goals`` are public for inspection.
    ``chat`` turns the executive on; its turns run on ``pool`` (a private one-thread pool,
    shut down by ``close``, when none is given).
    """

    def __init__(  # noqa: PLR0915  # one assignment per piece of state
        self,
        game: str,
        cfg: MindConfig,
        *,
        seed: int,
        log: RunLog,
        game_deadline: float,
        run_deadline: float,
        clock: Callable[[], float] = time.monotonic,
        chat: VisionChat | None = None,
        pool: Executor | None = None,
    ) -> None:
        self.game = game
        self.cfg = cfg
        self.seed = seed
        self._log = log
        self._deadline = min(game_deadline, run_deadline)
        self._clock = clock
        self._started = clock()
        self._chat = chat
        self._own_pool: ThreadPoolExecutor | None = None
        if chat is not None and pool is None:
            self._own_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn")
            pool = self._own_pool
        self._pool = pool
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
        # Executive turn state (unused without a chat).
        self._turn: _Turn | None = None
        self._calls = 0
        self._turns_level = 0
        self._turns_off = False  # the server rejected images: no more turns this game
        self._epoch = 0  # bumps on every level change and game over
        self._level_start = True  # the next fresh frame starts a level
        self._attempt_from = 0  # step of the current attempt's first frame
        self._died_after: int | None = None  # moves in the attempt a game over just ended
        self._wake_text = ""  # the prompt's wake line when it says more than the reason
        self._turn_done = -MIN_MOVES_BETWEEN_TURNS  # step the last turn finished
        self._stretch_from = 0  # step of the last turn start or level start on this level
        self._moves: deque[str] = deque(maxlen=MOVE_LOG_LINES)
        self._changed: set[int] = set()  # ids changed this level
        self._last_grid: Grid | None = None
        self._prev_grid: Grid | None = None  # the frame before this one, None after a start
        self._llm_plan: list[PlanItem] = []
        self._llm_test: PlanItem | None = None
        self._llm_scene: Scene | None = None  # the scene the reply's clicks refer to
        self._llm_protect: frozenset[str] = frozenset()
        self._llm_step = False  # the last move was an LLM plan step
        # Per-move scratch, read by the action record even after a failure.
        self._programs: list[dict[str, object]] = []
        self._vetoes: set[ActionKey] = set()
        self._scene_ok = False
        self._notes: set[str] = set()
        self._wake: str | None = None

    # -- the move ------------------------------------------------------------

    def choose(self, obs: Observation) -> Action:
        """One action for ``obs``; see the module docstring for the order."""
        frame = frame_hash(obs.grid)
        self._programs, self._vetoes, self._scene_ok = [], set(), False
        self._notes, self._wake = set(), None
        self._wake_text = ""
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
            self._llm_plan, self._llm_test, self._llm_step = [], None, False
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
            wake=self._wake,
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
        self._llm_plan, self._llm_test, self._llm_step = [], None, False
        self._changed.clear()
        self._last_grid = None

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
            self._prev_grid = self._last_grid
            scene = self._transition(obs, move, self._last_scene, self._last_action)
        self._last_grid = obs.grid
        self._level = obs.levels_completed
        choice = self._decide(obs, scene)
        if self._chat is not None:
            self._maybe_turn(obs, scene, choice)
        return choice

    def _start(self, obs: Observation) -> Scene:
        """First frame of a level or after a RESET: no diff, nothing learned."""
        self._tracker.reset()
        full, _ = self._tracker.update(obs.grid, self._noise.cells)
        scene = _strip(full, self._noise.noisy_ids)
        self._keep(full, scene)
        self._fresh, self._pred = False, None
        self._prev_grid = None
        self._attempt_from = self._action_no
        if self._level_start:
            self._level_start = False
            self._stretch_from = self._action_no
            self._notes.add("level start")
        elif self._died_after is not None:
            self._notes.add("game over")
            self._wake_text = (
                f"game over (last attempt ended in game over after {self._died_after} "
                f"move{'' if self._died_after == 1 else 's'})"
            )
        self._died_after = None
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
            self._died_after = move - self._attempt_from + 1
            if self._last_scene is not None:
                self._moves.append(f"{move} {_action_name(action, self._last_scene)} -> game over")
        self._epoch += 1
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
        self._moves.append(f"{move} {_action_name(action, before)} -> level up")
        self._epoch += 1
        self._turns_level = 0
        self._level_start = True
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
        if broke:
            self._notes.add("fact broken")
        self._changed.update(c.id for c in diff.changes if c.kind != "same")
        pred = self._pred
        hit = False
        if pred is not None:
            hit, missed = check(pred, before_s, diff, ignore=noisy, after=after)
            hit = hit and pred.scene is not None
            self._score(pred, set(missed))
            self._check_plan(pred, before_s, after, [*missed, *broke])
        if self._chat is not None:
            tag = "predicted" if hit else "unpredicted"
            what = _outcome(_strip_diff(diff, noisy), before_s)
            self._moves.append(f"{move} {_action_name(action, before_s)} -> {what} ({tag})")
        self._log_goals(self.goals.note_reached(after), "excluded")
        return after

    def _check_plan(
        self, pred: Prediction, before: Scene, after: Scene, keys: list[FactKey]
    ) -> None:
        """Stop a plan on a miss that matters; note the miss or a finished plan for waking."""
        goal = self._plan_goal
        if goal is not None:
            movers, protected = self._protected(before)
            if plan_breaks(pred, before, after, keys, movers, protected):
                self._plan = []
                self._notes.add("miss that matters")
                if not goal.holds(after):
                    self.note_failed(goal)
            elif not self._plan:
                self._notes.add("plan finished")
        elif self._llm_step:
            movers = mover_sigs(self.mem)
            protected = movers | self._llm_protect
            if plan_breaks(pred, before, after, keys, movers, protected, unknown_ok=True):
                self._llm_plan = []
                self._notes.add("miss that matters")
            elif not self._llm_plan:
                self._notes.add("plan finished")

    # -- deciding ------------------------------------------------------------

    def _decide(self, obs: Observation, scene: Scene) -> _Choice:
        late = self._clock() >= self._deadline
        if self._turn is not None:
            self._poll(scene)
        if self._plan and self._plan_goal not in self.goals.open():
            self._plan = []  # its goal was ruled out, excluded or stalled
        choice, llm_step = self._llm_choice(scene, late)
        if choice is None and self._plan:
            step = self._plan.pop(0)
            choice = self._offer(scene, step.action, "follow", "plan step", late, plan=True)
        if choice is None and not late:
            choice = self._plan_choice(scene)
        if choice is None and not late:
            choice = self._test_choice(obs, scene)
        if choice is None:
            if not late:
                self._exclude_vetoed(obs, scene)
                self._notes.add("no plan and no test")
            action = self._baseline.choose(obs, frozenset(self._vetoes))
            choice = _Choice(action, "fallback", "deadline passed" if late else "b0")
        self._tried.add((scene.key(), choice.action.key()))
        self._llm_step = llm_step
        if llm_step:
            self._plan_goal = None
            protected = mover_sigs(self.mem) | self._llm_protect
            self._pred = predict(scene, choice.action, self.mem, assume_still=protected)
        elif choice.branch in ("plan", "follow"):
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
                    self._notes.add("plan blocked")
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
        res = self._search(scene, goals)
        if res.stop == "found" and not res.plan and res.goal is not None:
            self._log_goals(self.goals.note_reached(scene), "excluded")
            rest = [g for g in self.goals.open() if g != res.goal]
            res = self._search(scene, rest)
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

    def _search(self, scene: Scene, goals: list[Goal]) -> SearchResult:
        res = search(scene, self._actions, self.mem, goals)
        self._log.event(
            "search",
            step=self._action_no,
            stop=res.stop,
            nodes=res.nodes,
            depth=res.depth,
            goals=len(goals),
            frontier=len(res.frontier),
            protected=res.protected,
        )
        return res

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

    def _remap(self, item: PlanItem, scene: Scene) -> Action | None:
        """The reply step's action on ``scene``; None (and a veto event) when its thing is gone."""
        asked = self._llm_scene if self._llm_scene is not None else scene
        action = remap_click(item.action, asked, scene, self._clicks.cell)
        if action is None:
            self._refuse(item.action, "click target gone")
        return action

    def _llm_choice(self, scene: Scene, late: bool) -> tuple[_Choice | None, bool]:
        """The reply's pending test, else its next plan step; and whether it is a plan step."""
        if self._llm_test is not None:
            item, self._llm_test = self._llm_test, None
            choice = None if late else self._llm_test_choice(scene, item)
            if choice is not None:
                return choice, False
        if self._llm_plan:
            choice = self._llm_follow(scene, late)
            return choice, choice is not None
        return None, False

    def _llm_follow(self, scene: Scene, late: bool) -> _Choice | None:
        """The next LLM plan step as a ``plan`` move, or None (the plan ends) when refused."""
        item = self._llm_plan.pop(0)
        action = self._remap(item, scene)
        choice = None
        if action is not None:
            choice = self._offer(scene, action, "plan", "llm plan", late, plan=True)
        if choice is None:
            self._llm_plan = []
            self._notes.add("plan blocked")
        return choice

    def _llm_test_choice(self, scene: Scene, item: PlanItem) -> _Choice | None:
        """The reply's test as the next move unless a veto refuses it."""
        action = self._remap(item, scene)
        if action is None:
            return None
        choice = self._offer(scene, action, "learn", f"llm test: {item.expect}", False, plan=False)
        if choice is not None:
            self._tests += 1
            self._log.event(
                "test",
                step=self._action_no,
                action=[action.id, action.x, action.y],
                question=item.expect,
            )
        return choice

    # -- the executive turn --------------------------------------------------

    def _maybe_turn(self, obs: Observation, scene: Scene, choice: _Choice) -> None:
        """Start a turn when a wake rule fired, no plan holds and the budgets allow it."""
        reason = self._wake_reason(choice)
        chat, pool = self._chat, self._pool
        if reason is None or chat is None or pool is None:
            return
        now = self._clock()
        if now >= self._deadline - SYNTH_CUTOFF_FRACTION * (self._deadline - self._started):
            return
        call = self._calls
        self._calls += 1
        self._turns_level += 1
        self._stretch_from = self._action_no
        inp = self._turn_input(obs, scene, reason)
        self._log.event("wake", step=self._action_no, reason=reason)
        self._wake = reason
        try:
            fut = pool.submit(_run_turn, chat, inp, self.seed * TURN_SEED_STRIDE + call)
        except RuntimeError as exc:  # a shut-down pool costs this turn, not the game
            self._log_failed(call, f"submit failed: {exc}")
            self._turn_done = self._action_no
            return
        self._turn = _Turn(fut, call, self._epoch, obs.levels_completed, now, scene)

    def _wake_reason(self, choice: _Choice) -> str | None:
        """This move's wake reason after the plan, in-flight, spacing and count rules."""
        if self._turns_off or self._turn is not None or choice.branch in ("plan", "follow"):
            return None
        if self._calls >= MAX_TURNS_PER_GAME or self._turns_level >= MAX_TURNS_PER_LEVEL:
            return None
        if self._action_no - self._stretch_from >= LONG_STRETCH_MOVES:
            self._notes.add("long stretch")
        reason = wake_reason(self._notes)
        spaced = self._action_no - self._turn_done >= MIN_MOVES_BETWEEN_TURNS
        return reason if reason in _UNSPACED or spaced else None

    def _turn_input(self, obs: Observation, scene: Scene, reason: str) -> TurnInput:
        sigs = {t.sig for t in scene.things}
        facts = [f for f in self.mem.facts.values() if f.key[1] in sigs or f.key[0] == "event"]
        facts.sort(key=lambda f: f.evidence[-1] if f.evidence else -1)
        open_goals = self.goals.open()
        listed = set(open_goals)
        status = self.goals.status
        goals = [(g.id(), status.get(g, "confirmed")) for g in open_goals]
        goals += [(g.id(), s) for g, s in status.items() if g not in listed]
        return TurnInput(
            grid=obs.grid,
            prev_grid=self._prev_grid,
            scene=scene,
            available=tuple(obs.available_actions),
            facts=tuple(f.line() for f in facts),
            goals=tuple(goals),
            questions=tuple(question(label, subject) for label, subject in self._frontier),
            moves=tuple(self._moves),
            level=obs.levels_completed,
            wake=self._wake_text if reason == "game over" else reason,
            movers=mover_sigs(self.mem),
            changed=frozenset(self._changed),
        )

    def _poll(self, scene: Scene) -> None:
        """Collect a finished turn, abandon one past its deadline, else leave it running."""
        turn = self._turn
        if turn is None:
            return
        if not turn.future.done():
            if turn.epoch != self._epoch:
                turn.future.cancel()
                self._turn, self._turn_done = None, self._action_no
                self._log_stale(turn)
            elif self._clock() - turn.started > TURN_DEADLINE_S:
                turn.future.cancel()
                self._turn, self._turn_done = None, self._action_no
                self._log_failed(turn.call, "timeout")
            return
        self._turn, self._turn_done = None, self._action_no
        try:
            reply, seconds = turn.future.result()
        except ImagesRejected:
            self._turns_off = True
            self._log_failed(turn.call, "images rejected")
            return
        except Exception as exc:  # the turn is lost; the move goes on with slice 1 logic
            self._log_failed(turn.call, f"{type(exc).__name__}: {exc}")
            return
        if turn.epoch != self._epoch:
            self._log_stale(turn)
            return
        failed = next((d for d in reply.dropped if d.startswith("turn failed:")), None)
        if failed is not None and not (reply.goals or reply.plan or reply.test):
            self._log_failed(turn.call, failed)
            return
        self._apply(reply, turn, scene, seconds)

    def _apply(self, reply: Reply, turn: _Turn, scene: Scene, seconds: float) -> None:
        """Goals first, then the plan or the test; see the module docstring."""
        self._log.event(
            "exec_turn",
            step=self._action_no,
            call=turn.call,
            seconds=round(seconds, 3),
            goals=[g.id() for g in reply.goals],
            plan_len=len(reply.plan),
            test=None if reply.test is None else reply.test.expect,
            dropped=list(reply.dropped),
        )
        if reply.goals:
            self._prefer_goals(reply.goals, scene)
        if not reply.plan and reply.test is None:
            return
        self._llm_scene = turn.scene
        self._plan, self._plan_goal = [], None
        if reply.plan:
            self._llm_plan, self._llm_test = list(reply.plan), None
            sigs: set[str] = set()
            for g in reply.goals:
                sigs |= goal_sigs(g, scene)
            self._llm_protect = frozenset(sigs)
        else:
            self._llm_plan, self._llm_test = [], reply.test

    def _prefer_goals(self, goals: Iterable[Goal], scene: Scene) -> None:
        """Add the reply's goals (source ``llm``) and move the open ones ahead of the rest."""
        book = self.goals
        new = [g for g in dict.fromkeys(goals) if g not in book.status]
        for g in new:
            book.status[g] = "open"
            book.sources[g] = LLM_SOURCE
        named = [g for g in dict.fromkeys(goals) if book.status.get(g) == "open"]
        front = set(named)
        order: dict[Goal, GoalStatus] = dict.fromkeys(named, "open")
        order.update((g, s) for g, s in book.status.items() if g not in front)
        book.status.clear()
        book.status.update(order)
        self._log_goals(new, "open")
        if new:
            self._log_goals(book.rule_out(list(self._history.values())), "ruled_out")
            self._log_goals(book.note_reached(scene), "excluded")

    def _log_stale(self, turn: _Turn) -> None:
        self._log.event("exec_stale", step=self._action_no, call=turn.call, asked_level=turn.level)

    def _log_failed(self, call: int, reason: str) -> None:
        self._log.event(
            "exec_failed", step=self._action_no, call=call, reason=reason[:MAX_ERROR_CHARS]
        )

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
            if status in _GOAL_WAKES:
                self._notes.add(f"goal {status}")
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
        """Log ``game_end`` with fact and goal counts by status. Safe to call twice.

        A turn still queued is cancelled; one already running finishes on its pool thread
        and its result is never read. A private pool is shut down without waiting.
        """
        if self._closed:
            return
        self._closed = True
        if self._turn is not None:
            self._turn.future.cancel()
            self._turn = None
        if self._own_pool is not None:
            self._own_pool.shutdown(wait=False, cancel_futures=True)
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
            **({} if self._chat is None else {"turns": self._calls}),
        )
