"""The fluid agent: one game's per-move policy over facts, goal guesses and short search.

No LLM and no program sandbox. It wires the slice 1 modules (``objects``, ``noise``,
``facts``, ``predictor``, ``goals``, ``search``, ``tester``) into a ``Policy``.

Per ``choose(obs)``, in order:

1. ``GAME_OVER``: learn the death from the move that caused it (``game_over=True``,
   the tracker diff of the final frame), log ``game_over``, reset the tracker and the
   noise tracks (keeping the cells of move counters, ``NoiseMask.new_attempt``),
   clear the plan and return RESET (branch ``reset``). ``NOT_PLAYED`` returns RESET
   too and learns nothing. The frame after a RESET learns nothing: the
   start frame is not a consequence of any rule.
2. ``levels_completed`` rose: record the ``level_up`` event fact, ``goals.level_up``
   on the frame before and the winning move's prediction, log ``level_up``, then reset
   the tracker and the noise tracks,
   run ``fluid_sleep`` over the fact keys the executed moves of the level relied on
   (each move's prediction ``used``, collected per level), reset ``FactMemory``
   positions, run the Transfer module's ``carry`` (the null clears every fact), and
   clear the plan, the level's scene history, ``tried`` and the frontier. Facts (with
   the real Transfer) and goal statuses stay. The new screen teaches nothing. ``close``
   sleeps once more over the current level's keys (game end). ``FactMemory.level``
   follows ``levels_completed`` so each new fact records the level it was made on.
3. Otherwise update tracker (the last prediction's scene is its prior), kinds
   (``kinds.KindBook``), the mode (``modes.ModeTracker``: selection and the active thing,
   stored on the scene) and noise,
   strip noisy things from both scenes and the diff, ``learn`` from the transition and
   ``check`` the previous prediction (its misses feed only the log and the plan stop;
   ``learn`` charges facts itself). Then
   ``goals.note_reached`` on the new scene. Just before, if the goal of the plan being
   followed (a found or a partial plan) is open and holds with no level-up, it is set
   ``probing`` instead of excluded, the plan ends, and a commit probe starts: while the
   scene stays the same, the next moves (branch ``commit``, event ``commit_probe``) try
   A5, then clicks on small non-mover things whose color is unique on screen, each once,
   at most ``COMMIT_PROBES``. They skip the ``known no effect`` veto, since a commit shows
   nothing until the winning state, but honor a predicted game over and a no-op seen on
   this scene. A level-up confirms the goal; a probe that ends any other way (the scene
   changed, no candidate left, game over) excludes it.
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
   protected, novel``). Plan steps are predicted with ``assume_still`` set to the
   mover sigs plus the goal's sigs, so the check covers what the plan relies on.
5. Else, when exactly one of search's actions survives the plan vetoes (``settled
   test`` doesn't count: a settled action can still be a plan step), it's taken with
   no search (branch ``forced``, why ``only move left``). Else ``goals.propose`` and
   ``search``: first over the open goals the executive named
   (source ``llm``) alone, then, only if that finds no plan, over all of
   ``goals.open()``. A found plan starts (branch ``plan``, why ``goal <id>``). A goal
   that already holds at the root is treated as reached (``note_reached``) and the
   search runs once more without it. A first step that a veto refuses counts as a
   failed attempt at that goal (see step 4). Search goals come from ``GoalBook.ranked``: the
   confirmed forms, then the best twelve open goals by prior, salience and distance,
   leaving out goals farther than the moves left this life, plus every goal the
   executive named.
6. Else, when the budget is tight (``budget_tight``: the best partial plan's goal is
   within the moves left with at most ``TEST_RESERVE`` to spare), that plan's step.
   Else the Voi module's test (``tester.choose_test`` when real) with the search's
   frontier (branch ``learn``, why the test's question), unless this level attempt
   already made ``MAX_TESTS_PER_ATTEMPT`` test moves (the count restarts on a level
   change and on game over); then Voi is skipped.
7. Else the goal search's best partial plan (``SearchResult.best``), when that search
   found no plan: its first step is taken and the rest held as a plan toward its goal
   (branch ``plan``, why ``toward <goal id>``), but only when it ends closer to the goal
   than any real frame of this life has been, so walking back to a spot already reached
   doesn't count. A refused step isn't a failed attempt.
   Else novelty: ``search`` with no goals and one pseudo-goal (``goal_test``), "some
   controlled mover thing sits at a top-left its sig hasn't held on this level" (every
   ``(sig, x, y)`` of every scene seen this level counts as held; controlled per
   ``goals.controlled_sigs``, so an enemy that moves whatever the action doesn't
   count), under half the goal search's caps (``NOVEL_NODES``, ``NOVEL_TIME_S``).
   Its first step is taken (branch ``plan``, why ``explore novel``) unless a plan veto
   refuses it; no plan is kept, so the next move searches again. It notes ``no plan
   and no test`` for waking, and unlike a held plan it doesn't block a wake. It is
   skipped when no controlled mover is on screen.
8. Else B0 with every vetoed candidate excluded (branch ``fallback``).
9. The Inhibition module's ``veto`` (``tester.veto`` when real) runs on each option
   before it is taken; a refused option is logged as a
   ``veto`` event and the next option in the same order is tried. B0 is last and is
   never vetoed. The chosen action's prediction is kept for step 3 of the next call.

Deviations the brief leaves open, decided here:

- A plan step is refused only for ``predicted game_over`` or ``known no effect
  here``. ``settled test`` guards tests: a planned move through a scene the agent
  already acted in (every replan after a RESET) is not a test and must not be refused.
- ``rule_out(history)`` runs after a ``propose`` (or a reply) that added goals and
  checks only the added goals. Goals already open are checked against each new frame
  as it arrives by ``note_reached``, so rechecking them would only cost time. History
  keeps one scene per distinct ``Scene.key()``, with its raw grid.
- Every goal check on a real frame (``note_reached``, ``rule_out``, ``level_up``, a
  plan's goal after a miss) passes that frame's raw grid, so ``match`` goals can hold.
  Search runs on predicted scenes with no grid and skips ``match`` goals.
- The fallback excludes every candidate ``veto`` refuses (in the action record's
  ``vetoes``), so B0 does not walk into a known death; no event is logged for those.

Past the game or run deadline: no search, no tests, no vetoes; follow the plan or B0.

Executive turn (slice 2, only with a ``chat``; without one the agent is slice 1 exactly):

- Wake rules, noted while a move is processed: level start (the first frame of the game
  or of a new level); game over (the first frame after the RESET that follows one; the
  prompt's wake text adds ``last attempt ended in game over after N moves``); a miss
  that matters (``plan_breaks`` stopped a plan); plan finished or blocked (a vetoed step
  or a click whose thing is gone); a goal excluded, ruled out or stalled; a fact turning
  broken; no plan and no test (the move fell back to novelty or B0); long stretch
  (``LONG_STRETCH_MOVES`` moves since the last turn started on this level, or since the
  level started). Noted reasons stay pending until a turn consumes them (all of them,
  whatever its outcome) or the level epoch changes (level change, game over,
  NOT_PLAYED); a new level or attempt then notes its own start. After the move's slice 1
  choice, the first pending reason in ``WAKE_ORDER`` wakes the executive, except while a
  plan holds (the choice's branch is ``plan`` or ``follow``, a novelty move excepted) or
  once turns are off for the game. Spacing: the next wake needs
  ``MIN_MOVES_BETWEEN_TURNS`` moves after the last turn, except level start and game
  over; a wake held back this way logs nothing.
- Budgets: ``MAX_TURNS_PER_LEVEL`` and ``MAX_TURNS_PER_GAME`` turns, and no turn
  starts past ``1 - SYNTH_CUTOFF_FRACTION`` (75%) of the game's budget (agent start to
  the game or run deadline). The turn's seed is ``seed * 1000 + call``.
- The turn is synchronous within its move: ``turn.take_turn`` runs on ``pool`` with the
  frame the move was chosen on, and this game's ``choose`` waits for it up to
  ``TURN_DEADLINE_S`` (less when the game deadline is nearer). Other games keep playing
  on their own threads. On a timeout the future is cancelled (a call already running
  finishes on its pool thread, unread), ``exec_failed`` reason ``timeout`` is logged and
  the move keeps its slice 1 choice (the best targeted test, else B0).
  ``ImagesRejected`` logs ``exec_failed`` ``images rejected`` and turns stop for the
  game. Any other failure logs ``exec_failed`` with the reason and the move keeps its
  slice 1 choice.
- Applying a reply, at once and to the scene it was asked about: goals enter
  ``GoalBook`` with source ``"llm"`` (a goal already known keeps its status and source)
  and every open one the reply named moves ahead of the other open goals; new goals
  then go through ``rule_out`` and ``note_reached`` like heuristic ones. A plan replaces
  any current plan and its first step replaces this move's choice; each step is taken
  as a ``plan`` move (why ``llm plan``), predicted with movers plus the reply's goal sigs
  assumed protected; a step the facts can't predict is still taken (a test in all but
  name) and the plan stops only on a miss that matters about those sigs. Plan steps
  honor the same vetoes as slice 1 plan steps. A test replaces this move's choice
  unless vetoed (branch ``learn``, why ``llm test: <question>``, a ``test`` event). A
  reply with goals and no plan or test re-runs step 5, which searches the named goals
  first; a found plan replaces this move's choice. A refused or empty reply step leaves
  the slice 1 choice in place. Later plan steps refer to the asked-on scene's ids: a
  click is re-mapped to the current thing with the clicked thing's sig nearest its old
  position, and dropped (a ``veto`` event, reason ``click target gone``) when no thing
  has that sig.

Failure guard: any ``Exception`` in the mind side of a move is logged as
``agent_error`` (truncated to ``MAX_ERROR_CHARS``) and that move goes to B0 (RESET on
a GAME_OVER or NOT_PLAYED frame). If it happened before the new scene was built, the
tracker restarts so the next frame learns nothing. B0's bookkeeping and the log write
run outside the guard.

Log: action records keep the v1 fields (``programs`` scores the previous move's
prediction, one entry per used fact, with ``level`` the fact's creation level or null
when the fact is gone; ``entropy`` is the frontier size; ``active`` and
``wake`` are null, ``voi`` is empty). Events: ``fact`` (``step, key, status, line``)
when a fact is new or changes status, ``goal`` (``step, id, status, source``),
``test`` (``step, action, question``), ``noise`` (``step, ids, cells``), ``veto``
(``step, action, reason``), ``level_up`` and ``game_over`` (``step, level``) with
``step`` the index of the move that caused them, ``sleep`` (``step, demoted, merged,
dropped, seconds, truncated``, counts) with one ``demoted`` and one ``merged``
(``step, key``) per fact key, ``agent_error`` and ``game_end``.
With a chat: ``wake`` (``step, reason``; the action record's ``wake`` holds the same
reason), ``exec_turn`` (``step, call, seconds, goals, plan_len, test, dropped``),
and ``exec_failed`` (``step, call, reason``); ``game_end`` adds ``turns``.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable, Collection, Iterable
from concurrent.futures import Executor, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace

from arcagent.core.types import (
    COORD_ACTION_ID,
    GRID_SIZE,
    RESET_ID,
    Action,
    Grid,
    Observation,
    Phase,
)
from arcagent.mind.agent import MAX_ERROR_CHARS, SYNTH_CUTOFF_FRACTION
from arcagent.mind.baseline import Baseline
from arcagent.mind.budget import BudgetReader
from arcagent.mind.config import MindConfig
from arcagent.mind.contract import PlanItem, Reply
from arcagent.mind.facts import (
    SPAWN,
    Effect,
    FactKey,
    FactMemory,
    Status,
    action_label,
    cell_owner,
    clicked_sig,
    learn,
)
from arcagent.mind.fluid_modules import NullFluidInhibition, build_fluid_modules
from arcagent.mind.fluid_sleep import FluidSleepReport, fluid_sleep
from arcagent.mind.goals import (
    Goal,
    GoalBook,
    GoalStatus,
    controlled_sigs,
    goal_sigs,
    mover_sigs,
)
from arcagent.mind.kinds import KindBook
from arcagent.mind.kinematics import Kinematics
from arcagent.mind.modes import ModeTracker
from arcagent.mind.noise import NoiseMask
from arcagent.mind.objects import Cell, Scene, SceneDiff, Thing, Tracker, click_cell
from arcagent.mind.perception import candidate_actions, frame_hash
from arcagent.mind.predictor import Prediction, check, predict
from arcagent.mind.progress import ProgressMonitor, Verdict
from arcagent.mind.runlog import RunLog
from arcagent.mind.search import SEARCH_MAX_NODES, SEARCH_TIME_S, PlanStep, SearchResult, search
from arcagent.mind.tester import GAME_OVER, NO_EFFECT, Noop, Tried, question, tried_key
from arcagent.mind.turn import MOVE_LOG_LINES, TurnInput, name_sigs, render_fact, take_turn
from arcagent.mind.types import ActionKey, Branch
from arcagent.mind.vchat import ImagesRejected, VisionChat

STALL_AFTER = 2  # failed attempts at one goal in one level before it is set aside
GOAL_SOURCE = "heuristic"  # slice 1 goals come from the heuristics in GoalBook.propose
LLM_SOURCE = "llm"  # goals named by the executive turn
SEEN_NO_EFFECT = "seen no effect here"  # this action changed nothing on this very scene
UNDOES_PROGRESS = "undoes progress"  # a known level reset while this life has progress
_PLAN_VETOES = frozenset(
    {GAME_OVER, NO_EFFECT, SEEN_NO_EFFECT, UNDOES_PROGRESS}
)  # reasons a plan step honors
MAX_TESTS_PER_ATTEMPT = 200  # test moves per level attempt before Voi is skipped
NOVEL_NODES = SEARCH_MAX_NODES // 2  # novelty search caps: half the goal search's
NOVEL_TIME_S = SEARCH_TIME_S / 2
NOVEL_WHY = "explore novel"  # the why of a novelty fallback move (branch ``plan``)
PARTIAL_WHY = "toward"  # why prefix of a best-partial-plan step: ``toward <goal id>``
MOTION_CUT = 3  # motion verdicts in a row after which novelty yields
PROGRESS_GOALS = 20  # goals the progress monitor measures per move (pursued and forms first)
LEVEL = "level"  # subject of a reset-level fact: (action label, LEVEL, "any")
TEST_RESERVE = 2  # moves to spare that still leave room for a test before a fitting plan
COMMIT_PROBES = 4  # most commit-probe moves on one reached goal
COMMIT_MAX_CELLS = 16  # a commit-probe click target has at most this many cells
FORCED_WHY = "only move left"  # the why of a forced move (branch ``forced``)
_EMPTY_DIFF = SceneDiff((), 0)

TURN_DEADLINE_S = 30.0  # the longest a move waits for its turn
TURN_MAX_TOKENS = 1536  # reply budget per turn (the HTTP chat's max_tokens)
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
    """A move. ``test`` is a test's question, logged only once the move is final."""

    action: Action
    branch: Branch
    why: str
    test: str | None = None


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
        if (t.sig not in movers and t.kind not in movers) or t.id not in pred.effects:
            continue
        e, a = expect.get(t.id), now.get(t.id)
        if (e is None) != (a is None) or (
            e is not None and a is not None and (e.x, e.y) != (a.x, a.y)
        ):
            return True
    return False


def budget_tight(*, remaining: int | None, distance: int) -> bool:
    """Whether a plan ``distance`` moves long fits the moves left with under
    ``TEST_RESERVE`` + 1 to spare: no room for a test first."""
    return remaining is not None and distance <= remaining <= distance + TEST_RESERVE


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


def _run_turn(chat: VisionChat, inp: TurnInput, seed: int) -> tuple[Reply, float]:
    """The pool job: ``take_turn`` plus its wall-clock seconds."""
    t0 = time.perf_counter()
    reply = take_turn(chat, inp, seed=seed)
    return reply, time.perf_counter() - t0


@dataclass(slots=True)
class _Probe:
    """Commit probing of a reached goal: the scene key it holds on, the actions tried."""

    goal: Goal
    key: tuple[tuple[str, int, int], ...]
    tried: set[tuple[int, int, int]]


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

    ``clock`` is injectable for tests. ``mem``, ``goals`` and ``modules`` (built from
    ``cfg`` by ``build_fluid_modules``) are public for inspection.
    ``chat`` turns the executive on; its turns run on ``pool`` (a private one-thread pool,
    shut down by ``close``, when none is given) while the move that asked waits.
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
        self.modules = build_fluid_modules(cfg)
        self.mem = FactMemory()
        self.mem.broken_after = self.modules.metacognition.broken_after()
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
        self._partial: SearchResult | None = None  # this move's best partial plan, if any
        self._probe: _Probe | None = None  # a goal reached with no level-up, being committed
        self._kin = Kinematics(self.mem)  # rebuilt each move: walks read that move's scene
        self._budget = BudgetReader()  # moves left this life, from a counter bar
        self._progress = ProgressMonitor()  # progress or motion, per life
        self._verdict: Verdict = "none"  # this move's progress verdict
        self._motion = 0  # motion verdicts in a row
        self._start_key: tuple[tuple[str, int, int], ...] = ()  # the attempt's first scene
        self._closest: dict[Goal, int] = {}  # lowest real distance per partial-plan goal, this life
        self._failed: dict[Goal, int] = {}  # failed plan attempts per goal, this level
        # one (scene, raw grid) per distinct scene key this level
        self._history: dict[tuple[tuple[str, int, int], ...], tuple[Scene, Grid]] = {}
        self._tried: Tried = set()
        self._noop: Noop = set()  # (scene key, action key) seen to change nothing, this level
        self._frontier: tuple[tuple[str, str], ...] = ()
        self._simple: list[Action] = []
        self._can_click = False
        self._tests = 0
        self._tests_attempt = 0  # test moves since the level or attempt started
        self._vetoes_total = 0
        self._used_level: set[FactKey] = set()  # fact keys the executed moves relied on
        self._seen_at: set[tuple[str, int, int]] = set()  # (kind, x, y) seen this level
        self._kinds = KindBook()  # identity classes of looks, for the whole game
        self._modes = ModeTracker()  # selection and the active thing, per attempt
        self._closed = False
        # Executive turn state (unused without a chat).
        self._calls = 0
        self._turns_level = 0
        self._turns_off = False  # the server rejected images: no more turns this game
        self._epoch = 0  # bumps on every level change, game over and NOT_PLAYED
        self._pending: set[str] = set()  # wake reasons no turn has consumed yet
        self._level_start = True  # the next fresh frame starts a level
        self._attempt_from = 0  # step of the current attempt's first frame
        self._died_after: int | None = None  # moves in the attempt a game over just ended
        self._wake_text = ""  # the game over wake line, pending with its reason
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
            progress=self._verdict,
            budget=self._budget.remaining(),
        )
        self._last_action = a
        if a.id == RESET_ID:
            self._fresh = True
        self._action_no += 1
        return a

    def _restart(self, *, new_attempt: bool = False) -> None:
        """Forget the tracked frame and noise tracks: the next frame is a fresh start.

        ``new_attempt`` (a game over) keeps the noise mask's counter cells: the bar sits
        in the same strip on every attempt of a level.
        """
        self._tracker.reset()
        self.mem.kin_reset()
        self._modes.reset()
        if new_attempt:
            self._noise.new_attempt()
        else:
            self._noise.reset_tracks()
        self._fresh = True
        self._pred = None
        self._llm_plan, self._llm_test, self._llm_step = [], None, False
        self._changed.clear()
        self._last_grid = None
        self._closest.clear()

    def _mind_step(self, obs: Observation) -> _Choice:
        move = self._action_no - 1  # index of the move that produced obs
        if obs.phase == Phase.GAME_OVER:
            self._game_over(obs, move)
            return _Choice(Action(RESET_ID), "reset", "game over")
        if obs.phase == Phase.NOT_PLAYED:
            self._new_epoch()
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
        self.mem.level = self._level
        return self._decide(obs, scene)

    def _start(self, obs: Observation) -> Scene:
        """First frame of a level or after a RESET: no diff, nothing learned."""
        self._tracker.reset()
        self.mem.kin_reset()
        full, _ = self._tracker.update(obs.grid, self._noise.cells)
        full = self._kinds.label(full)
        scene = _strip(full, self._noise.noisy_ids)
        self._keep(full, scene, obs.grid)
        self._start_key = scene.key()
        self._budget.start_life(obs.grid)
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

    def _keep(self, full: Scene, scene: Scene, grid: Grid) -> None:
        self._last_full, self._last_scene = full, scene
        self.mem.note_kin(scene.things)
        self._seen_at.update((t.kind, t.x, t.y) for t in scene.things)
        self._history.setdefault(scene.key(), (scene, grid))
        self._scene_ok = True

    def _new_epoch(self) -> None:
        """Level change, game over or NOT_PLAYED: pending wake reasons end here."""
        self._epoch += 1
        self._pending.clear()
        self._wake_text = ""

    def _rule_out(self, only: Collection[Goal]) -> list[Goal]:
        """``GoalBook.rule_out`` of ``only`` against this level's history."""
        frames = list(self._history.values())
        return self.goals.rule_out([f[0] for f in frames], [f[1] for f in frames], only=only)

    def _game_over(self, obs: Observation, move: int) -> None:
        self._tests_attempt = 0  # a new attempt gets its own test budget
        self._end_probe()
        self._budget.observe(obs.grid)  # the last move drained the bar too
        self._budget.game_over()
        self._progress.reset_life()
        self._motion = 0
        prev_full, action = self._last_full, self._last_action
        if not self._fresh and prev_full is not None and action is not None:
            full, diff = self._tracker.update(obs.grid, self._noise.cells)
            full = self._kinds.label(full)
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
        self._new_epoch()
        self._restart(new_attempt=True)
        self._plan = []
        self._scene_ok = True

    def _level_up(self, obs: Observation, move: int, before: Scene, action: Action) -> None:
        snap = self._statuses()
        keys = learn(
            self.mem, before, action, _EMPTY_DIFF, move=move, level_up=True, game_over=False
        )
        self._log_facts(keys, snap, move)
        after = self._pred.scene if self._pred is not None else None
        self._log_goals(self.goals.level_up(before, self._last_grid, after=after), "confirmed")
        self._end_probe()  # a probing goal true before the win was just confirmed
        self._progress.reset_level()
        self._budget.new_level()
        self._motion = 0
        self._log.event("level_up", step=move, level=obs.levels_completed)
        self._moves.append(f"{move} {_action_name(action, before)} -> level up")
        self._new_epoch()
        self._turns_level = 0
        self._level_start = True
        self.reopen_goals()
        self._restart()
        try:
            self._sleep(move)
        except Exception as exc:  # a sleep failure must not skip the level reset
            self._log.event(
                "agent_error",
                step=move,
                error=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
            )
        self._tests_attempt = 0
        self.mem.new_level()
        self.modules.transfer.carry(self.mem, self.goals)
        self._plan, self._history, self._tried, self._frontier = [], {}, set(), ()
        self._noop = set()
        self._seen_at = set()

    def _transition(self, obs: Observation, move: int, before_s: Scene, action: Action) -> Scene:
        prev_full = self._last_full
        prior = self._pred.scene if self._pred is not None else None
        full, diff = self._tracker.update(obs.grid, self._noise.cells, prior)
        if diff is not None and prev_full is not None:
            joined = self._kinds.observe(prev_full, full, diff, self._tracker.how)
            if joined:
                self._log.event("kind", step=move, kinds=joined)
                prev_full = self._kinds.label(prev_full)
        full = self._kinds.label(full)
        if diff is not None and prev_full is not None:
            full = self._note_mode(prev_full, action, full, diff)
        if diff is None or prev_full is None:  # cannot happen after a non-fresh frame
            scene = _strip(full, self._noise.noisy_ids)
            self._keep(full, scene, obs.grid)
            return scene
        flagged = self._noise.observe(prev_full, action.id, diff)
        if flagged:
            self._log.event("noise", step=move, ids=flagged, cells=len(self._noise.cells))
        noisy = self._noise.noisy_ids
        before = _strip(prev_full, noisy)
        after = _strip(full, noisy)
        if _strip_diff(diff, noisy).empty:
            self._noop.add((before_s.key(), action.key()))
        self._read_budget(obs.grid, changed=not _strip_diff(diff, noisy).empty)
        self._keep(full, after, obs.grid)
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
        self._note_reset(before, action, after, _strip_diff(diff, noisy), move)
        self._changed.update(c.id for c in diff.changes if c.kind != "same")
        pred = self._pred
        hit = False
        if pred is not None:
            hit, missed = check(pred, before_s, diff, ignore=noisy, after=after)
            hit = hit and pred.scene is not None
            self._score(pred, set(missed))
            self._check_plan(pred, before_s, after, [*missed, *broke], obs.grid)
        if self._chat is not None:
            tag = "predicted" if hit else "unpredicted"
            what = _outcome(_strip_diff(diff, noisy), before_s)
            self._moves.append(f"{move} {_action_name(action, before_s)} -> {what} ({tag})")
        self._start_probe(after, obs.grid)
        self._log_goals(self.goals.note_reached(after, obs.grid), "excluded")
        return after

    def _start_probe(self, after: Scene, grid: Grid) -> None:
        """The pursued goal holds with no level-up: set it probing (instead of excluded)."""
        goal = self._plan_goal
        if goal is None or self.goals.status.get(goal) != "open" or not goal.holds(after, grid):
            return
        self._end_probe()
        self.goals.start_probe(goal)
        self._probe = _Probe(goal, after.key(), set())
        self._plan = []
        self._log_goals([goal], "probing")

    def _end_probe(self) -> None:
        """Stop probing; a goal still probing is excluded."""
        self._probe = None
        self._log_goals(self.goals.end_probe(), "excluded")

    def _note_mode(self, before: Scene, action: Action, after: Scene, diff: SceneDiff) -> Scene:
        """Run the mode tracker; ``after`` with its selection and active thing. Things seen
        selected make their sig and kind selectable for predictions."""
        mode = self._modes.observe(
            before, action, after, diff, self._noise.cells, controlled=controlled_sigs(self.mem)
        )
        by_id = after.by_id()
        for sid in mode.selected:
            t = by_id.get(sid)
            if t is not None:
                self.mem.selectable.update((t.sig, t.kind))
        return replace(after, selected=mode.selected, active=mode.active)

    def _check_plan(
        self, pred: Prediction, before: Scene, after: Scene, keys: list[FactKey], grid: Grid
    ) -> None:
        """Stop a plan on a miss that matters; note the miss or a finished plan for waking."""
        goal = self._plan_goal
        if goal is not None:
            movers, protected = self._protected(before)
            if plan_breaks(pred, before, after, keys, movers, protected):
                self._plan = []
                self._notes.add("miss that matters")
                if not goal.holds(after, grid):
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
        self._partial = None
        self._kin = Kinematics(self.mem, self._lone_mover(scene))  # this move's walks
        self._note_closest(scene)
        self._note_progress(scene)
        probe = self._probe_choice(scene)
        return self._commit(scene, probe) if probe is not None else self._choose(obs, scene)

    def _choose(self, obs: Observation, scene: Scene) -> _Choice:
        """Every branch but a commit probe, in the order of the module docstring."""
        late = self._clock() >= self._deadline
        if self._plan and self._plan_goal not in self.goals.open():
            self._plan = []  # its goal was ruled out, excluded or stalled
        choice, llm_step = self._llm_choice(scene, late)
        if choice is None and self._plan:
            step = self._plan.pop(0)
            choice = self._offer(scene, step.action, "follow", "plan step", late, plan=True)
        if choice is None and not late:
            choice = self._forced(scene) or self._plan_choice(scene)
        if choice is None and not late:
            choice = self._tight_choice(scene) or (
                self._test_choice(obs, scene)
                if self._tests_attempt < MAX_TESTS_PER_ATTEMPT
                else None
            )
        if choice is None and not late:
            choice = self._partial_choice(scene) or (
                self._novel_choice(scene) if self._motion < MOTION_CUT else None
            )
        if choice is None:
            if not late:
                self._exclude_vetoed(obs, scene)
                self._notes.add("no plan and no test")
            action = self._baseline.choose(obs, frozenset(self._vetoes))
            choice = _Choice(action, "fallback", "deadline passed" if late else "b0")
        if self._chat is not None:
            choice, llm_step = self._maybe_turn(obs, scene, choice, llm_step, late)
        self._tried.add(tried_key(scene, choice.action))
        if choice.test is not None:
            self._tests += 1
            self._tests_attempt += 1
            a = choice.action
            self._log.event(
                "test", step=self._action_no, action=[a.id, a.x, a.y], question=choice.test
            )
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
        self._used_level.update(self._pred.used)
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
            reason = self._veto(scene, action)
            if reason is not None and (not plan or reason in _PLAN_VETOES):
                self._refuse(action, reason)
                if plan:
                    self._plan = []
                    self._notes.add("plan blocked")
                return None
        return _Choice(action, branch, why)

    def _veto(self, scene: Scene, action: Action) -> str | None:
        """``SEEN_NO_EFFECT`` when ``action`` already changed nothing on this scene, else
        the Inhibition module's veto. The null module (an ablation) refuses nothing."""
        if (scene.key(), action.key()) in self._known_noop():
            return SEEN_NO_EFFECT
        if self._resets(scene, action) and self._progress.progressed():
            return UNDOES_PROGRESS
        return self.modules.inhibition.veto(scene, action, self.mem, self._tried)

    def _resets(self, scene: Scene, action: Action) -> bool:
        """Whether a non-broken fact says ``action`` puts the level back as it started."""
        fact = self.mem.get((action_label(action, scene), LEVEL, "any"))
        return fact is not None and fact.status != "broken"

    def _known_noop(self) -> Noop:
        """The no-op pairs vetoes and search use: none under the null Inhibition module."""
        return set() if isinstance(self.modules.inhibition, NullFluidInhibition) else self._noop

    def _refuse(self, action: Action, reason: str) -> None:
        self._vetoes.add(action.key())
        self._vetoes_total += 1
        self._log.event(
            "veto", step=self._action_no, action=[action.id, action.x, action.y], reason=reason
        )

    def _forced(self, scene: Scene) -> _Choice | None:
        """The one action no plan veto refuses, as a choice; None when there are more or
        none. Stops at the second survivor, so a usual move costs two vetoes."""
        alive: Action | None = None
        for action in self._actions(scene):
            if self._veto(scene, action) in _PLAN_VETOES:
                continue
            if alive is not None:
                return None
            alive = action
        return None if alive is None else _Choice(alive, "forced", FORCED_WHY)

    def _plan_choice(self, scene: Scene) -> _Choice | None:
        added = self.goals.propose(scene, self.mem, source=GOAL_SOURCE, grid=self._last_grid)
        if added:
            self._log_goals(added, "open")
            self._log_goals(self._rule_out(added), "ruled_out")
        goals = self.goals.open()
        if not goals:
            self._frontier = ()
            return None
        sources = self.goals.sources
        n_llm = sum(1 for g in goals if sources.get(g) == LLM_SOURCE)
        res: SearchResult | None = None
        if 0 < n_llm < len(goals):  # the executive's goals alone first
            res = self._search_toward(scene, lambda g: sources.get(g) == LLM_SOURCE)
            if res.stop != "found" or not res.plan:
                res = None
        if res is None:
            res = self._search_toward(scene, lambda _g: True)
        self._frontier = res.frontier
        if res.stop != "found" or not res.plan or res.goal is None:
            self._partial = res if res.best and res.best_goal is not None else None
            return None
        first, *rest_steps = res.plan
        self._plan = list(rest_steps)
        self._plan_goal = res.goal
        choice = self._offer(scene, first.action, "plan", f"goal {res.goal.id()}", False, plan=True)
        if choice is None:
            self._plan = []
            self.note_failed(res.goal)  # a vetoed first step is a failed attempt
        return choice

    def _probe_choice(self, scene: Scene) -> _Choice | None:
        """The next commit-probe action, or None (ending any probe that is over).

        A probe is over when the scene moved off the goal state, ``COMMIT_PROBES`` moves
        were spent, or no candidate is left. Candidates skip the ``known no effect`` veto
        (a commit action shows nothing until the winning state) but not a predicted game
        over or a no-op seen on this very scene.
        """
        probe = self._probe
        if probe is None:
            return None
        if scene.key() == probe.key and len(probe.tried) < COMMIT_PROBES:
            for action in self._commit_candidates(scene):
                if action.key() in probe.tried:
                    continue
                if self._veto(scene, action) in (GAME_OVER, SEEN_NO_EFFECT):
                    continue
                probe.tried.add(action.key())
                return _Choice(action, "commit", f"commit probe {probe.goal.id()}")
        self._end_probe()
        return None

    def _commit(self, scene: Scene, choice: _Choice) -> _Choice:
        """Take a commit-probe move: logged, tried, predicted like a test."""
        a = choice.action
        self._log.event("commit_probe", step=self._action_no, action=[a.id, a.x, a.y])
        self._tried.add(tried_key(scene, a))
        self._plan_goal, self._llm_step = None, False
        self._pred = predict(scene, a, self.mem)
        self._used_level.update(self._pred.used)
        return choice

    def _commit_candidates(self, scene: Scene) -> list[Action]:
        """A5 when offered, then clicks on small non-mover things whose color no other thing
        on screen has, fewest cells first."""
        out = [a for a in self._simple if a.id == 5]
        if self._can_click:
            movers = mover_sigs(self.mem)
            colors = Counter(t.color for t in scene.things)
            small = [
                t
                for t in scene.things
                if len(t.cells) <= COMMIT_MAX_CELLS and t.sig not in movers and colors[t.color] == 1
            ]
            for t in sorted(small, key=lambda t: (len(t.cells), t.id)):
                x, y = self._clicks.cell(t)
                out.append(Action(COORD_ACTION_ID, x, y))
        return out

    def _read_budget(self, grid: Grid, *, changed: bool) -> None:
        """Feed the budget reader one frame, picking up a counter the noise mask flagged
        (the first shrinking one, else the first)."""
        counters = self._noise.counters
        if counters and self._budget.counter is None:
            self._budget.set_counter(next((c for c in counters if c.shrinks), counters[0]))
        self._budget.observe(grid, changed=changed)

    def _lone_mover(self, scene: Scene) -> str | None:
        """The kind of the one controlled thing on screen, or None (none, or several)."""
        controlled = controlled_sigs(self.mem)
        mine = [t for t in scene.things if t.sig in controlled or t.kind in controlled]
        return mine[0].kind if len(mine) == 1 else None

    def _note_progress(self, scene: Scene) -> None:
        """Name this move with the progress monitor, over the pursued goal, the confirmed
        forms and the first open goals (``PROGRESS_GOALS`` in all)."""
        pursued = [self._plan_goal] if self._plan_goal is not None else []
        goals = list(dict.fromkeys([*pursued, *self.goals.open()]))[:PROGRESS_GOALS]
        dists = {g: g.distance(scene, self._last_grid, self._kin) for g in goals}
        self._verdict = self._progress.observe(scene.key(), dists, active=self._plan_goal)
        self._motion = self._motion + 1 if self._verdict == "motion" else 0

    def _note_reset(
        self, before: Scene, action: Action, after: Scene, diff: SceneDiff, move: int
    ) -> None:
        """An action that changed two or more things and left the scene exactly as this
        attempt started puts the level back: record it (one thing moving back is a step)."""
        changed = sum(1 for c in diff.changes if c.kind != "same")
        if changed < 2 or not self._start_key or after.key() != self._start_key:
            return
        if before.key() == self._start_key:
            return
        key = (action_label(action, before), LEVEL, "any")
        snap = self._statuses()
        self.mem.record(key, (Effect("reset_level"),), move)
        self._log_facts([key], snap, move)

    def _note_closest(self, scene: Scene) -> None:
        """Lower each partial-plan goal's closest distance this life to the real frame's."""
        for g, best in self._closest.items():
            d = g.distance(scene, None, self._kin)
            if d is not None and d < best:
                self._closest[g] = d

    def _tight_choice(self, scene: Scene) -> _Choice | None:
        """The partial plan's step ahead of tests, when its goal just fits the moves left
        (``budget_tight``)."""
        res = self._partial
        if res is None or res.best_distance is None:
            return None
        if not budget_tight(remaining=self._budget.remaining(), distance=res.best_distance):
            return None
        return self._partial_choice(scene)

    def _partial_choice(self, scene: Scene) -> _Choice | None:
        """The first step of this move's best partial plan (``SearchResult.best``), held as a
        plan toward its goal. None without one, when it ends no closer to its goal than some
        real frame of this life already was (motion, not progress), or when a plan veto
        refuses the step; a refused step isn't a failed attempt at the goal."""
        res = self._partial
        goal, end = (None, None) if res is None else (res.best_goal, res.best_distance)
        if res is None or goal is None or end is None or not res.best:
            return None
        now = goal.distance(scene, None, self._kin)
        if now is not None:
            self._closest[goal] = min(self._closest.get(goal, now), now)
        if end >= self._closest.get(goal, end + 1):
            return None
        first, *rest = res.best
        self._plan, self._plan_goal = list(rest), goal
        choice = self._offer(
            scene, first.action, "plan", f"{PARTIAL_WHY} {goal.id()}", False, plan=True
        )
        if choice is None:
            self._plan, self._plan_goal = [], None
        return choice

    def _search_toward(self, scene: Scene, keep: Callable[[Goal], bool]) -> SearchResult:
        """Search the ranked goals ``keep`` accepts; a goal holding at the root is reached."""
        res = self._search(scene, [g for g in self._ranked(scene) if keep(g)])
        if res.stop == "found" and not res.plan and res.goal is not None:
            self._log_goals(self.goals.note_reached(scene, self._last_grid), "excluded")
            rest = [g for g in self._ranked(scene) if g != res.goal and keep(g)]
            res = self._search(scene, rest)
        return res

    def _ranked(self, scene: Scene) -> list[Goal]:
        """``GoalBook.ranked`` for this move (salient: controlled things and things that
        changed this level; the budget's moves left), plus every open goal the executive
        named."""
        changed = {s for t in scene.things if t.id in self._changed for s in (t.sig, t.kind)}
        salient = frozenset(controlled_sigs(self.mem) | changed)
        ranked = self.goals.ranked(
            scene,
            grid=self._last_grid,
            kin=self._kin,
            salient=salient,
            budget=self._budget.remaining(),
        )
        named = [
            g
            for g in self.goals.open()
            if self.goals.sources.get(g) == LLM_SOURCE and g not in ranked
        ]
        return ranked + named

    def _search(
        self,
        scene: Scene,
        goals: list[Goal],
        goal_test: Callable[[Scene], bool] | None = None,
        *,
        max_nodes: int = SEARCH_MAX_NODES,
        time_s: float = SEARCH_TIME_S,
    ) -> SearchResult:
        res = search(
            scene,
            self._actions,
            self.mem,
            goals,
            goal_test=goal_test,
            max_nodes=max_nodes,
            time_s=time_s,
            noop=self._known_noop(),
            kin=self._kin,
            max_depth=self._budget.remaining(),
        )
        self._log.event(
            "search",
            step=self._action_no,
            stop=res.stop,
            nodes=res.nodes,
            depth=res.depth,
            goals=len(goals),
            frontier=len(res.frontier),
            protected=res.protected,
            novel=goal_test is not None,
        )
        return res

    def _novel_choice(self, scene: Scene) -> _Choice | None:
        """The first step toward a mover at a top-left it hasn't held on this level.

        None when no mover is on screen, the search finds no such scene, or the first
        step is refused (plan vetoes only). Logs a ``search`` event with ``novel`` set.
        Only the first step is taken; the next move searches again with the new cell seen.
        """
        movers = mover_sigs(self.mem) & controlled_sigs(self.mem)
        if not any(t.sig in movers or t.kind in movers for t in scene.things):
            return None
        seen = self._seen_at

        def novel(s: Scene) -> bool:
            return any(
                (t.sig in movers or t.kind in movers) and (t.kind, t.x, t.y) not in seen
                for t in s.things
            )

        res = self._search(scene, [], goal_test=novel, max_nodes=NOVEL_NODES, time_s=NOVEL_TIME_S)
        if res.stop != "found" or not res.plan:
            return None
        self._notes.add("no plan and no test")
        self._plan, self._plan_goal = [], None
        action = res.plan[0].action
        reason = self._veto(scene, action)
        if reason is not None and reason in _PLAN_VETOES:
            self._refuse(action, reason)
            return None
        return _Choice(action, "plan", NOVEL_WHY)

    def _test_choice(self, obs: Observation, scene: Scene) -> _Choice | None:
        test = self.modules.voi.choose(
            scene, candidate_actions(obs), self.mem, self._frontier, self._tried, None
        )
        if test is None:
            return None
        choice = self._offer(scene, test.action, "learn", test.question, False, plan=False)
        return None if choice is None else replace(choice, test=test.question)

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
        return None if choice is None else replace(choice, test=item.expect)

    # -- the executive turn --------------------------------------------------

    def _maybe_turn(
        self, obs: Observation, scene: Scene, choice: _Choice, llm_step: bool, late: bool
    ) -> tuple[_Choice, bool]:
        """Run a turn when a pending reason wakes it; return the move's final choice."""
        self._pending |= self._notes
        reason = self._wake_reason(choice)
        chat, pool = self._chat, self._pool
        if reason is None or chat is None or pool is None:
            return choice, llm_step
        now = self._clock()
        if now >= self._deadline - SYNTH_CUTOFF_FRACTION * (self._deadline - self._started):
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
        got = self._await_turn(chat, pool, inp, call, min(TURN_DEADLINE_S, self._deadline - now))
        self._turn_done = self._action_no
        if got is None:
            return choice, llm_step
        new, new_llm = self._apply(got[0], call, scene, got[1], late)
        return (choice, llm_step) if new is None else (new, new_llm)

    def _await_turn(
        self, chat: VisionChat, pool: Executor, inp: TurnInput, call: int, wait_s: float
    ) -> tuple[Reply, float] | None:
        """Submit the turn and wait up to ``wait_s``; the reply and its seconds, or None."""
        try:
            fut = pool.submit(_run_turn, chat, inp, self.seed * TURN_SEED_STRIDE + call)
        except RuntimeError as exc:  # a shut-down pool costs this turn, not the game
            self._log_failed(call, f"submit failed: {exc}")
            return None
        done, _ = wait([fut], timeout=max(wait_s, 0.0))
        if not done:
            fut.cancel()  # a call already running finishes on its pool thread, unread
            self._log_failed(call, "timeout")
            return None
        try:
            reply, seconds = fut.result()
        except ImagesRejected:
            self._turns_off = True
            self._log_failed(call, "images rejected")
            return None
        except Exception as exc:  # the turn is lost; the move keeps its slice 1 choice
            self._log_failed(call, f"{type(exc).__name__}: {exc}")
            return None
        failed = next((d for d in reply.dropped if d.startswith("turn failed:")), None)
        if failed is not None and not (reply.goals or reply.plan or reply.test):
            self._log_failed(call, failed)
            return None
        return reply, seconds

    def _wake_reason(self, choice: _Choice) -> str | None:
        """The pending reason that wakes a turn now, after the plan, spacing and count rules."""
        held = choice.branch in ("plan", "follow") and choice.why != NOVEL_WHY
        if self._turns_off or held:
            return None
        if self._calls >= MAX_TURNS_PER_GAME or self._turns_level >= MAX_TURNS_PER_LEVEL:
            return None
        if self._action_no - self._stretch_from >= LONG_STRETCH_MOVES:
            self._pending.add("long stretch")
        reason = wake_reason(self._pending)
        spaced = self._action_no - self._turn_done >= MIN_MOVES_BETWEEN_TURNS
        return reason if reason in _UNSPACED or spaced else None

    def _turn_input(self, obs: Observation, scene: Scene, reason: str) -> TurnInput:
        sigs = {t.sig for t in scene.things}

        def shown(key: FactKey) -> bool:
            """On-screen sig facts, self/any/spawn facts, event facts; no off-screen clicks."""
            label, subject, _ = key
            if label == "event":
                return True
            clicked = clicked_sig(label)
            if clicked is not None and clicked not in sigs:
                return False
            return subject in sigs or subject in ("self", "any", SPAWN)

        facts = [f for f in self.mem.facts.values() if shown(f.key)]
        facts.sort(key=lambda f: f.evidence[-1] if f.evidence else -1)
        owner = cell_owner(scene)
        open_goals = self.goals.open()
        listed = set(open_goals)
        status = self.goals.status
        pairs = [(g, status.get(g, "confirmed")) for g in open_goals]
        pairs += [(g, s) for g, s in status.items() if g not in listed]
        return TurnInput(
            grid=obs.grid,
            prev_grid=self._prev_grid,
            scene=scene,
            available=tuple(obs.available_actions),
            facts=tuple(render_fact(f, scene, owner) for f in facts),
            goals=tuple((name_sigs(g.id(), scene), s) for g, s in pairs),
            questions=tuple(question(label, subject) for label, subject in self._frontier),
            moves=tuple(self._moves),
            level=obs.levels_completed,
            wake=self._wake_text if reason == "game over" and self._wake_text else reason,
            movers=mover_sigs(self.mem),
            changed=frozenset(self._changed),
        )

    def _apply(
        self, reply: Reply, call: int, scene: Scene, seconds: float, late: bool
    ) -> tuple[_Choice | None, bool]:
        """Apply the reply to ``scene``; the choice it gives this move (None keeps slice 1)."""
        self._log.event(
            "exec_turn",
            step=self._action_no,
            call=call,
            seconds=round(seconds, 3),
            goals=[g.id() for g in reply.goals],
            plan_len=len(reply.plan),
            test=None if reply.test is None else reply.test.expect,
            dropped=list(reply.dropped),
        )
        if reply.goals:
            self._prefer_goals(reply.goals, scene)
        if not reply.plan and reply.test is None:
            if reply.goals and not late:
                return self._plan_choice(scene), False
            return None, False
        self._llm_scene = scene
        self._plan, self._plan_goal = [], None
        if reply.plan:
            self._llm_plan, self._llm_test = list(reply.plan), None
            sigs: set[str] = set()
            for g in reply.goals:
                sigs |= goal_sigs(g, scene)
            self._llm_protect = frozenset(sigs)
        else:
            self._llm_plan, self._llm_test = [], reply.test
        return self._llm_choice(scene, late)

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
            self._log_goals(self._rule_out(new), "ruled_out")
            self._log_goals(book.note_reached(scene, self._last_grid), "excluded")

    def _log_failed(self, call: int, reason: str) -> None:
        self._log.event(
            "exec_failed", step=self._action_no, call=call, reason=reason[:MAX_ERROR_CHARS]
        )

    def _exclude_vetoed(self, obs: Observation, scene: Scene) -> None:
        """Add every candidate ``veto`` refuses to this move's vetoes (no events)."""
        for action in candidate_actions(obs):
            if self._veto(scene, action) is not None:
                self._vetoes.add(action.key())

    def _actions(self, scene: Scene) -> list[Action]:
        """Search's actions: the simple actions, then one click per thing when clicks exist.

        A predicted scene can carry a thing past the board's edge; it gets no click.
        """
        out = list(self._simple)
        if self._can_click:
            cells = (self._clicks.cell(t) for t in scene.things)
            out.extend(
                Action(COORD_ACTION_ID, x, y)
                for x, y in cells
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE
            )
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
                    "level": None if fact is None else fact.level,
                }
            )
        self._programs = out

    def _sleep(self, step: int) -> FluidSleepReport:
        """``fluid_sleep`` over this level's used keys; log ``sleep``, ``demoted``, ``merged``."""
        report = fluid_sleep(self.mem, self._used_level, self._level)
        self._used_level = set()
        self._log.event(
            "sleep",
            step=step,
            demoted=len(report.demoted),
            merged=len(report.merged),
            dropped=len(report.dropped),
            seconds=round(report.seconds, 4),
            truncated=report.truncated,
        )
        for key in report.demoted:
            self._log.event("demoted", step=step, key=list(key))
        for key in report.merged:
            self._log.event("merged", step=step, key=list(key))
        return report

    def close(self) -> None:
        """Sleep once more (game end), then log ``game_end`` with fact and goal counts.

        Safe to call twice. A failing sleep is logged as ``agent_error`` and the game still
        ends. A private pool is shut down without waiting; a turn call still running after
        its move timed out finishes on its pool thread and its result is never read.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._sleep(self._action_no - 1)
        except Exception as exc:  # a sleep failure costs the bookkeeping, not the game end
            self._log.event(
                "agent_error",
                step=self._action_no,
                error=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
            )
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
