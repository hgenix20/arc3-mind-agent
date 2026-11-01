"""The planner agent: the planning loop of spec v2 section 3 on top of ``FluidAgent``.

``PlannerAgent`` is a ``FluidAgent`` (spec sections 3.1 to 3.12, 4.5, 4.7 to 4.13; build
plan sections 2 and 3). Without a chat it plays exactly like ``FluidAgent``: every choice
is the base ladder's, and only the learning that never changes a choice runs (the adverse
book, the inhibition network, the flags). With a chat it is the model-first agent:

1. Observe, learn and check the last prediction (``FluidAgent`` as is), then record the
   transition in the adverse book and the inhibition network (4.10a, 4.10b), note the
   health bar (4.10a) and the curiosity flags (4.11a).
2. The controller (4.5c) reads the move's signals in its order and names the method:
   ``keep``, ``replan`` (H-c, H-d), ``return_to_fork`` (H-b), ``fallback`` (H-a, deadline)
   or ``explore`` (H-e). Every handoff is logged with its rule.
3. ``return_to_fork``: forks with an untried path are tried newest first (3.6, 4.7c
   rule 5): the first with a known return route over the explorer's transitions (4.7c
   rule 6) gets the route as its segment, and an empty route starts its next untried
   path at once. A fork with no known route, or past ``ROUTE_TRIES`` routes, has its
   untried paths failed. When no fork is left, ``replan_root`` is logged and the move
   replans. A ``replan`` decision drops any pending fork resume first, so the planning
   loop really runs.
4. ``replan``, or no segment left: the planning loop (3.2). The horizon is the level's
   typical moves from the long-term store, else ``HORIZON_CAP``. Hypotheses come from the
   code (a search toward each of the top ``TOP_GOALS`` open goals, rolled out with
   probabilities, 4.7a, 4.8a) and from the planner: a briefing (4.9a, 4.9c) and one
   planner call ask for a segment or a skill (4.9f, decision D-A); the proposal is rolled
   out and, when it dies or reaches an unknown step before any goal, the planner is asked
   once more with the failing step named (3.5). A hypothesis whose rollout died is
   dropped wherever a plan is chosen (3.9: a predicted game over is an adverse path).
   The winner by expected value (3.4, 4.7b, decision D-C) becomes the segment; two or
   more distinct plans that reach the same goal open a fork (4.7c). When nothing reaches
   a goal, the best hypothesis still plays when its path probability is at least
   ``BAND_LOW``; else the base ladder plays the move. The briefing's sequence list (4.9a
   item 8) shows the segment being followed or replaced, step by step, with whether each
   step was played.
5. The segment runs one move per ``choose`` through the base's reply executor
   (``_llm_plan``): each step is vetoed, predicted and checked, and a miss that matters
   stops it (4.9e). When the segment is a fork path, that path (and only that path) is
   then marked failed; a fork path replaced by a replan or a fallback goes back to
   untried, so the tree's active path is always the segment being followed.
6. Gates (4.10d): a move that passed ``T_MOVE`` seconds or spent more than ``N_TOKEN``
   planner tokens logs ``gate_trip``; the next move's controller sees it (H-a). When the
   planner call itself overran the gate, its proposal is dropped and the move is the
   explorer fallback (4.10e).
7. Level up: ``FluidAgent._level_up`` (sleep and transfer), then consolidation (4.12b):
   facts in state ``fact`` are promoted to the long-term store as rules under their
   colour-free canonical key, with the evidence gathered since their last promotion,
   the completion's move count is recorded (the horizon rule, 4.7) and the level's move
   sequence is written; then the tree, the flags and the level counters reset. Game over
   keeps the tree (the level restarts from the same frame) and fails the fork path the
   segment was following, when it was one. Either way the level instance ends: the
   fallback and no-progress counts restart with it, so every attempt gives the planner
   another chance, while within one instance the explorer keeps the level once H-e has
   fired after ``STALL_MOVES`` fallbacks in a row (the hybrid rule). ``close`` writes the
   network's weights to the store (4.12c).

Harm (4.10a, decision D-D): with a health bar, a move after which the bar is lower; a
counter the noise mask flagged is a health bar once its drops happened on fewer than
``HEALTH_BAR_MAX_DROP_RATE`` of at least ``HEALTH_BAR_MIN_MOVES`` moves of the level
instance (a move counter drops on most moves). Without a bar, health is 1 and only a game
over is harm. Each transition records ``(action label, touched kind, "touch")`` for every
kind the controlled things touch afterwards, or ``(label, "self", "any")`` when they touch
nothing, in the adverse book and the network. ``_harm_of`` prices a move by the kinds the
controlled body would overlap or touch after one gait step (the network's bias starts at
``NET_PRIOR_BIAS``, so an unseen situation reads as low-band harm: a 0.5 prior would put
every first move of a game in the medium band and spend the perseverance trials on
harmless moves). The veto adds ``ADVERSE_CERTAIN`` for the certain band and
``ADVERSE_MEDIUM`` when perseverance (4.10c) refuses a medium-band move; both stop a
segment step. A medium-band move that is taken counts as a trial of its key for the
level instance (``MAX_TRIALS`` per key) and logs ``perseverance``.

Log (``runlog`` docstring, "Planner agent (v2) fields"): action records gain ``self_rec``,
``focus``, ``hypothesis_id``, ``segment_id``, ``fork_id``, ``planner_call``, ``p_step``,
``p_path``, ``prev_matched`` and ``gate``; events ``segment``, ``handoff``, ``harm``,
``consolidate``, ``longterm``, ``fork_open``, ``branch_failed``, ``return_route``,
``replan_root``, ``gate_trip``, ``planner_call``, ``flag`` and ``perseverance``.

Failure modes: a planner call that fails or times out costs the call only (``exec_failed``)
and the move falls through to the base ladder; a malformed ``net_state`` in the store is
logged as ``agent_error`` and the network starts fresh; a store that cannot be written at
level up or at close logs ``agent_error`` and the game goes on. Every loop here is bounded:
at most two planner calls per replan, ``TOP_GOALS`` searches per hypothesis pass,
``ROUTE_TRIES`` return routes per fork, ``ROLLOUT_GOALS`` goals per rollout step.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, wait
from dataclasses import asdict, replace

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, RESET_ID, Action, Grid, Observation
from arcagent.mind.adverse import (
    WIDTH,
    AdverseBook,
    AdverseKey,
    Features,
    InhibitionNet,
    Perseverance,
    band_of,
    features,
    harm_probability,
)
from arcagent.mind.briefing import Briefing, planner_messages
from arcagent.mind.budget import Counter as BarCounter
from arcagent.mind.config import MindConfig
from arcagent.mind.contract import QUESTION_IDS, PlanItem, Reply, parse_reply
from arcagent.mind.controller import DEADLINE_RULE, NO_RULE, Controller, Decision, Signals
from arcagent.mind.facts import BAND_LOW, SPAWN, Fact, FactKey, action_label
from arcagent.mind.flags import FlagBook
from arcagent.mind.fluid import (
    _PLAN_VETOES,  # pyright: ignore[reportPrivateUsage]
    GOAL_SOURCE,
    LEVEL,
    MAX_ERROR_CHARS,
    TURN_SEED_STRIDE,
    FluidAgent,
    _action_name,  # pyright: ignore[reportPrivateUsage]
    _Choice,  # pyright: ignore[reportPrivateUsage]
    wake_reason,
)
from arcagent.mind.game_model import seed_claims
from arcagent.mind.goals import Goal, controlled_sigs, mover_sigs
from arcagent.mind.hypotheses import Hypothesis, Source, from_rollout, hypothesis_id, rank
from arcagent.mind.longterm import (
    NO_HARM,
    LongTermStore,
    RuleKind,
    RuleRecord,
    SequenceRecord,
    canonical_key,
    effect_text,
    rule_id,
    statement_for,
)
from arcagent.mind.objects import Cell, Scene, Thing
from arcagent.mind.plantree import Fork, PlanTree, return_route
from arcagent.mind.predictor import predict
from arcagent.mind.rollout import Rollout, rollout, step_probability
from arcagent.mind.runlog import RunLog
from arcagent.mind.search import protected_sigs
from arcagent.mind.selfrecord import FocusRecord, SelfRecord
from arcagent.mind.skills import Role, SkillContext, run_skill
from arcagent.mind.tester import question, tried_key
from arcagent.mind.turn import name_sigs
from arcagent.mind.types import Branch
from arcagent.mind.vchat import ImagesRejected, VisionChat, VMessage
from arcagent.synth.run import Completion

T_MOVE = 60.0  # seconds one move may take, observation to action (spec 4.10d placeholder)
N_TOKEN = 4096  # planner completion tokens one move may spend (spec 4.10d placeholder)
MIN_CALL_S = 5.0  # no planner call starts with less than this left in the move gate
SEGMENT_CAP = 12  # steps a segment may hold, whatever the horizon says
HORIZON_CAP = 200  # horizon before a level's first completion (spec 4.7 placeholder)
TOP_GOALS = 3  # open goals the code searches toward per hypothesis pass
HEALTH_BAR_MAX_DROP_RATE = 0.3  # a counter dropping on fewer moves than this share is health
HEALTH_BAR_MIN_MOVES = 10  # moves of a level instance watched before a counter is judged
NO_PROGRESS_MOVES = 40  # moves without progress on the active goal before H-c replans
ADVERSE_CERTAIN = "adverse certain"  # veto reason: harm probability in the certain band
ADVERSE_MEDIUM = "adverse medium"  # veto reason: medium band and perseverance refused
NET_PRIOR_BIAS = -2.2  # a fresh network's bias: an unseen situation reads about 0.1 harm
ROUTE_TRIES = 2  # return routes tried per fork before its untried paths are given up
CALL_GAP_MOVES = 3  # moves without a planner call after one that gave no segment or failed
ROLLOUT_GOALS = 50  # open goals a rollout checks per step
STATE_LINES = 40  # things listed in the briefing's state list
RULE_LINES = 32  # fact lines handed to each rule list (the briefing cuts them further)
FLAG_LINES = 8  # flag lines in the briefing
QUESTION_PAIRS = 3  # frontier pairs Q-UNKNOWN names
FORK_SOURCE = "fork"  # segment source: a fork's untried path
ROUTE_SOURCE = "route"  # segment source: a return route to a fork
NO_GOAL = "none"  # goal id of a hypothesis whose rollout reached no goal
_ADVERSE_VETOES = frozenset({ADVERSE_CERTAIN, ADVERSE_MEDIUM})
_SEGMENT_VETOES = _PLAN_VETOES | _ADVERSE_VETOES  # reasons a segment step honours
_PROGRESS = frozenset({"progress", "milestone"})
_ACTION_SUBJECTS = frozenset({"self", "any", SPAWN, LEVEL})
_NEIGHBOURS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

__all__ = [
    "ADVERSE_CERTAIN",
    "ADVERSE_MEDIUM",
    "CALL_GAP_MOVES",
    "HEALTH_BAR_MAX_DROP_RATE",
    "HEALTH_BAR_MIN_MOVES",
    "HORIZON_CAP",
    "MIN_CALL_S",
    "NET_PRIOR_BIAS",
    "NO_PROGRESS_MOVES",
    "N_TOKEN",
    "ROUTE_TRIES",
    "SEGMENT_CAP",
    "TOP_GOALS",
    "T_MOVE",
    "PlannerAgent",
    "bar_cells",
    "is_health_bar",
]


def _run_planner(chat: VisionChat, messages: list[VMessage], seed: int) -> tuple[Completion, float]:
    """The pool job: the chat call plus its wall-clock seconds."""
    t0 = time.perf_counter()
    completion = chat(messages, seed)
    return completion, time.perf_counter() - t0


def bar_cells(grid: Grid, counter: BarCounter) -> int:
    """Cells of ``counter``'s colour in its band of ``grid``: the bar's size on this frame."""
    lines = sorted(x for x in counter.lines if 0 <= x < GRID_SIZE)
    band = grid[lines, :] if counter.axis == "row" else grid[:, lines]
    return int((band == counter.color).sum())


def is_health_bar(*, drops: int, moves: int) -> bool:
    """Spec 4.10a as plan D-D reads it: after ``HEALTH_BAR_MIN_MOVES`` moves of a level
    instance, a counter whose drops happened on fewer than ``HEALTH_BAR_MAX_DROP_RATE`` of
    them is a health bar; a move counter drops on most moves and is not harm."""
    return moves >= HEALTH_BAR_MIN_MOVES and drops < HEALTH_BAR_MAX_DROP_RATE * moves


def _fresh_net(state: Mapping[str, object] | None) -> tuple[InhibitionNet, str | None]:
    """A network from the store's state, or a fresh one with the ``NET_PRIOR_BIAS`` bias;
    the second value says why a given state was rejected."""
    net = InhibitionNet()
    if state is not None:
        try:
            net.load(state)
        except ValueError as exc:
            net, _ = _fresh_net(None)
            return net, f"net_state rejected: {exc}"
        return net, None
    weights = [0.0] * (WIDTH - 1) + [NET_PRIOR_BIAS]
    net.load({"weights": weights, "n_updates": 0})
    return net, None


def _size(t: Thing) -> tuple[int, int]:
    xs = [x for x, _ in t.cells]
    ys = [y for _, y in t.cells]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _nearest_gap(body: Thing, cells: frozenset[Cell]) -> int:
    """Manhattan cells from the body's top-left to the nearest of ``cells``."""
    return min(abs(x - body.x) + abs(y - body.y) for x, y in cells)


def _corridor(start: Cell, end: Cell) -> set[Cell]:
    """The cells of the two L-shaped straight routes between two cells."""
    (x0, y0), (x1, y1) = start, end
    out: set[Cell] = set()
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    out.update((x, y0) for x in xs)
    out.update((x1, y) for y in ys)
    out.update((x0, y) for y in ys)
    out.update((x, y1) for x in xs)
    return out


class PlannerAgent(FluidAgent):
    """See the module docstring. ``store`` is the long-term store shared by the run's
    games (``None`` plays without long-term memory); its ``net_state`` seeds the
    inhibition network and its lessons enter the briefing."""

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
        store: LongTermStore | None = None,
    ) -> None:
        super().__init__(
            game,
            cfg,
            seed=seed,
            log=log,
            game_deadline=game_deadline,
            run_deadline=run_deadline,
            clock=clock,
            chat=chat,
            pool=pool,
        )
        self._store = store
        self._adverse = AdverseBook()
        self._net, net_error = _fresh_net(None if store is None else store.net_state)
        if net_error is not None:
            self._log.event("agent_error", step=0, error=net_error[:MAX_ERROR_CHARS])
        self._persevere = Perseverance()
        self._tree = PlanTree()
        self._flags = FlagBook()
        self._controller = Controller(no_progress_moves=NO_PROGRESS_MOVES)
        self._model_text = ""  # the planner's model from its last reply
        self._pending_q = ""  # a question the planner asked, answered in the next briefing
        self._segment_id = ""
        self._segment_source = ""
        self._hypothesis_id = ""
        self._fork_id = ""
        self._segment_path: tuple[str, int] | None = None  # the fork path the segment is
        self._segment_items: list[PlanItem] = []  # the segment's steps, for the briefing
        self._segment_played = 0  # steps of the segment played so far
        self._level_moves = 0  # moves in this level instance
        self._level_start_moves: list[Action] = []  # moves from the level start
        self._no_progress = 0  # moves since the active goal last came closer
        self._fallbacks = 0  # fallback moves in a row
        self._health: int | None = None  # None until a health bar is identified
        self._bar_drops = 0
        self._bar_moves = 0
        self._bar_drop_cells = 0
        self._bar_last: int | None = None  # the bar's cells on the previous frame
        self._bar_is_health = False  # the level's counter is a health bar, once judged so
        self._move_t0 = clock()
        self._move_tokens = 0
        self._gate_trip: str | None = None  # this move's trip
        self._tripped_last: str | None = None  # the previous move's trip, read by H-a
        self._p_step: float | None = None
        self._p_path: float | None = None
        self._prev_matched: bool | None = None
        self._planner_calls = 0
        self._tokens_total = 0
        # State beyond the brief's list.
        self._segment_broke = False  # the segment stopped at a miss that matters
        self._hyps: list[Hypothesis] = []  # the last replan's hypotheses, best first
        self._rollouts: dict[str, Rollout] = {}  # by hypothesis id
        self._expects: dict[str, list[str]] = {}  # per-step expectations by hypothesis id
        self._protects: dict[str, frozenset[str]] = {}  # the rollout's protected sigs by id
        self._goal_by_id: dict[str, Goal] = {}
        self._active_goal: Goal | None = None  # the goal of the active segment
        self._roles_now: dict[str, Role] = {}  # roles per kind and sig, this move
        self._controlled_now: frozenset[str] = frozenset()
        self._trials: dict[AdverseKey, int] = {}  # medium-band trials per key, this instance
        self._promoted: dict[FactKey, tuple[int, int]] = {}  # (hits, misses) at last promote
        self._kinds_seen: set[str] = set()
        self._fork_ids: dict[tuple[bytes, str], str] = {}  # (state key, goal id) -> fork id
        self._route_tries: dict[str, int] = {}
        self._pending_fork: tuple[str, int] | None = None  # the fork a return route heads to
        self._planner_call_id: int | None = None  # the call made on this move
        self._quiet_until = 0  # no planner call before this move after an empty one
        self._fields: dict[str, object] = self._blank_fields()

    # -- the move --------------------------------------------------------------

    def choose(self, obs: Observation) -> Action:
        self._move_t0 = self._clock()
        self._move_tokens = 0
        self._tripped_last, self._gate_trip = self._gate_trip, None
        self._planner_call_id = None
        self._fields = self._blank_fields()
        action = super().choose(obs)
        if action.id != RESET_ID:
            self._level_moves += 1
            self._level_start_moves.append(action)
        return action

    def resume(self) -> None:
        super().resume()
        self._clear_segment()

    def _action_fields(self) -> dict[str, object]:
        seconds = round(self._clock() - self._move_t0, 3)
        gate = {"seconds": seconds, "tokens": self._move_tokens, "trip": self._gate_trip}
        return {**self._fields, "gate": gate}

    @staticmethod
    def _blank_fields() -> dict[str, object]:
        return {
            "self_rec": None,
            "focus": None,
            "hypothesis_id": None,
            "segment_id": None,
            "fork_id": None,
            "planner_call": None,
            "p_step": None,
            "p_path": None,
            "prev_matched": None,
        }

    def _decide(self, obs: Observation, scene: Scene) -> _Choice:
        # Roles are read by the vetoes the base runs before ``_choose`` (the commit probe).
        self._controlled_now = controlled_sigs(self.mem)
        self._roles_now = self._roles(scene)
        # The base returns a commit probe without ``_choose``, so the per-move bookkeeping
        # runs here, after whichever branch chose.
        choice = super()._decide(obs, scene)
        self._after_choice(obs, scene, choice)
        return choice

    def _choose(self, obs: Observation, scene: Scene) -> _Choice:
        if self._prev_grid is None:  # a start frame: nothing to compare with
            self._prev_matched = None
            self._bar_last = None
        self._no_progress = 0 if self._verdict in _PROGRESS else self._no_progress + 1
        if self._chat is None:
            return super()._choose(obs, scene)
        return self._planner_choose(obs, scene)

    def _planner_choose(self, obs: Observation, scene: Scene) -> _Choice:
        """The model-first move (module docstring steps 2 to 6)."""
        late = self._clock() >= self._deadline
        signals = Signals(
            gate_tripped=self._tripped_last is not None,
            segment_broke=self._segment_broke,
            fallbacks_in_row=self._fallbacks,
            no_progress_moves=self._no_progress,
            budget_left=self._budget.remaining(),
            segment_left=len(self._llm_plan),
            deadline_passed=late,
        )
        decision = self._controller.decide(signals)
        self._segment_broke = False
        if decision.method != "keep":
            self._log.event(
                "handoff",
                step=self._action_no,
                rule=decision.rule,
                method=decision.method,
                reason=decision.reason,
            )
        if decision.method == "fallback":
            why = "deadline passed" if late else f"gate {self._tripped_last}"
            return self._fallback_choice(obs, scene, why, late)
        if decision.method == "explore":
            return self._fallback_choice(obs, scene, "explore", late)
        if decision.method == "return_to_fork":
            self._return_to_fork(obs, scene)
        elif decision.method == "replan":
            self._no_progress = 0
            self._pending_fork = None  # the planning loop runs, not the fork resume
            self._llm_plan, self._llm_test = [], None
        if not self._llm_plan and self._llm_test is None and self._pending_fork is not None:
            self._resume_fork(obs, scene)
        if not self._llm_plan and self._llm_test is None:
            self._replan(obs, scene, decision)
            if self._gate_trip is not None:
                return self._fallback_choice(obs, scene, f"gate {self._gate_trip}", late)
        choice = super()._choose(obs, scene)
        if choice.why == "llm plan" and self._segment_source:
            choice = replace(choice, why=f"segment {self._segment_source}")
        return choice

    def _fallback_choice(self, obs: Observation, scene: Scene, why: str, late: bool) -> _Choice:
        """The fallback of spec 4.10e: the explorer's move with the vetoed candidates
        excluded, predicted like any other move. Ends any segment."""
        self._llm_plan, self._llm_test = [], None
        self._clear_segment()
        if not late:
            self._exclude_vetoed(obs, scene)
            self._notes.add("no plan and no test")
        action = self._baseline.choose(obs, frozenset(self._vetoes))
        self._tried.add(tried_key(scene, action))
        self._llm_step = False
        self._plan_goal = None
        self._pred = predict(scene, action, self.mem)
        self._used_level.update(self._pred.used)
        return _Choice(action, "fallback", why)

    def _after_choice(self, obs: Observation, scene: Scene, choice: _Choice) -> None:
        """Bookkeeping after every choice: p_step, the fallback count, perseverance
        trials, the gates and the action record's fields."""
        pred = self._pred
        self._p_step = (
            None if pred is None or pred.scene is None else step_probability(self.mem, pred.used)
        )
        self._fallbacks = self._fallbacks + 1 if choice.branch == "fallback" else 0
        seconds = self._clock() - self._move_t0
        if self._chat is not None and choice.action.id != RESET_ID:
            self._note_trial(scene, choice.action)
            if self._gate_trip is None:
                if seconds > T_MOVE:
                    self._trip("time", seconds)
                elif self._move_tokens > N_TOKEN:
                    self._trip("tokens", seconds)
        self_rec = SelfRecord.from_state(obs.available_actions, self._lone_mover(scene))
        self._fields = {
            "self_rec": asdict(self_rec),
            "focus": self._focus(choice.branch, seconds).as_dict(),
            "hypothesis_id": self._hypothesis_id or None,
            "segment_id": self._segment_id or None,
            "fork_id": self._fork_id or None,
            "planner_call": self._planner_call_id,
            "p_step": self._p_step,
            "p_path": self._p_path if self._segment_id else None,
            "prev_matched": self._prev_matched,
        }

    def _trip(self, gate: str, seconds: float) -> None:
        self._gate_trip = gate
        self._log.event(
            "gate_trip",
            step=self._action_no,
            gate=gate,
            seconds=round(seconds, 3),
            tokens=self._move_tokens,
        )

    def _focus(self, mode: str, seconds: float) -> FocusRecord:
        goal = self._active_goal
        target = goal.b if goal is not None and goal.kind in ("reach", "cover") else None
        return FocusRecord(
            level=self._level,
            goal_id=None if goal is None else goal.id(),
            goal_status=None if goal is None else self.goals.status.get(goal),
            target=target,
            mode=mode,
            budget_left=self._budget.remaining(),
            verdict=self._verdict,
            fallbacks_in_row=self._fallbacks,
            segment_left=len(self._llm_plan),
            tokens_this_move=self._move_tokens,
            seconds_this_move=round(seconds, 3),
        )

    def _maybe_turn(
        self, obs: Observation, scene: Scene, choice: _Choice, llm_step: bool, late: bool
    ) -> tuple[_Choice, bool]:
        # The planner call replaces the base's executive turn; nothing wakes here.
        del obs, scene, late
        return choice, llm_step

    def _offer(
        self, scene: Scene, action: Action, branch: Branch, why: str, late: bool, *, plan: bool
    ) -> _Choice | None:
        # FluidAgent._offer with the adverse reasons added to the set a plan step honours;
        # the base keeps that set in a module constant with no hook.
        if not late:
            reason = self._veto(scene, action)
            if reason is not None and (not plan or reason in _SEGMENT_VETOES):
                self._refuse(action, reason)
                if plan:
                    self._plan = []
                    self._notes.add("plan blocked")
                return None
        return _Choice(action, branch, why)

    # -- the segment -------------------------------------------------------------

    def _clear_segment(self) -> None:
        """Forget the segment; a fork path it was following goes back to untried."""
        path = self._segment_path
        if path is not None and self._tree.active() == path:
            self._tree.release(*path)
        self._segment_path = None
        self._segment_id = ""
        self._segment_source = ""
        self._hypothesis_id = ""
        self._fork_id = ""
        self._p_path = None
        self._active_goal = None

    def _set_segment(
        self,
        scene: Scene,
        items: Sequence[PlanItem],
        source: str,
        hyp_id: str,
        goal_id: str,
        *,
        p_path: float | None,
        ev: float | None,
        protect: frozenset[str],
    ) -> None:
        """Make ``items`` the segment the base executor follows, and log ``segment``."""
        self._llm_plan, self._llm_test = list(items), None
        self._llm_scene = scene
        self._llm_protect = protect
        self._plan, self._plan_goal = [], None
        self._segment_path = None  # a fork path's caller sets it after this
        self._segment_items = list(items)
        self._segment_played = 0
        self._segment_id = hyp_id
        self._segment_source = source
        self._hypothesis_id = hyp_id
        self._p_path = p_path
        self._active_goal = self._goal_by_id.get(goal_id)
        self._log.event(
            "segment",
            step=self._action_no,
            id=hyp_id,
            source=source,
            length=len(items),
            p_path=p_path,
            ev=ev,
            goal=goal_id,
        )

    def _fail_active(self, reason: str) -> None:
        """Mark the fork path the segment follows failed for this level instance (4.7c
        rule 7). A segment that is not a fork path fails nothing in the tree."""
        path = self._segment_path
        if path is None or self._tree.active() != path:
            return
        fork_id, index = path
        self._segment_path = None
        self._tree.fail_branch(fork_id, index)
        self._log.event(
            "branch_failed", step=self._action_no, fork=fork_id, index=index, reason=reason
        )

    def _return_to_fork(self, obs: Observation, scene: Scene) -> None:
        """Spec 3.6 and 4.7c rule 5: forks with an untried path, newest first; the first
        with a known return route gets the route as the segment (an empty route starts
        its path at once). A fork with no route, or past ``ROUTE_TRIES``, has its untried
        paths failed. No fork left: a replan from the root."""
        current = self.explorer.key(obs)
        for fork in self._tree.forks_with_untried():
            tries = self._route_tries.get(fork.id, 0)
            if tries >= ROUTE_TRIES:
                self._give_up_fork(fork, f"return route failed {tries} times")
                continue
            route = return_route(self.explorer.graph.edges, current, fork.state_key)
            if route is None:
                self._give_up_fork(fork, "no known route to the fork")
                continue
            self._route_tries[fork.id] = tries + 1
            self._log.event("return_route", step=self._action_no, fork=fork.id, length=len(route))
            if not route:
                if self._start_fork_path(fork, scene):
                    return
                continue
            self._set_route_segment(fork, route, scene)
            return
        self._replan_root("no fork with an untried path and a known route")

    def _set_route_segment(self, fork: Fork, route: Sequence[Action], scene: Scene) -> None:
        """The return route as the segment; ``_pending_fork`` names the path to take on
        arrival."""
        self._pending_fork = (fork.id, self._tree.untried(fork.id)[0])
        self._fork_id = fork.id
        items = [PlanItem(a, f"return to fork {fork.id}") for a in route]
        self._set_segment(
            scene,
            items,
            ROUTE_SOURCE,
            hypothesis_id(fork.goal_id, ROUTE_SOURCE, route),
            fork.goal_id,
            p_path=None,
            ev=None,
            protect=frozenset(),
        )

    def _resume_fork(self, obs: Observation, scene: Scene) -> None:
        """The return route finished: take the fork's next untried path when the route
        did arrive, else replan from the root."""
        assert self._pending_fork is not None
        fork_id, _ = self._pending_fork
        self._pending_fork = None
        fork = self._tree.fork(fork_id)
        if fork is None:
            return
        if self.explorer.key(obs) != fork.state_key:
            self._replan_root(f"return route did not reach fork {fork_id}")
            return
        if not self._start_fork_path(fork, scene):
            self._replan_root(f"every path of fork {fork_id} is failed")

    def _start_fork_path(self, fork: Fork, scene: Scene) -> bool:
        """Start the fork's first untried path that still rolls out alive (at most one
        rollout per path); paths that now die are failed. False when none is left."""
        goal = self._goal_by_id.get(fork.goal_id)
        goals = [goal] if goal is not None else self._rollout_goals()
        protect = protected_sigs(self.mem, [] if goal is None else [goal], scene)
        for index in self._tree.untried(fork.id):
            path = list(fork.paths[index])
            ro = rollout(
                scene,
                path,
                self.mem,
                goals,
                protect,
                horizon=max(1, len(path)),
                health=self._health or 1,
                adverse=self._harm_of,
            )
            if ro.stop == "death":
                self._tree.fail_branch(fork.id, index)
                self._log.event(
                    "branch_failed",
                    step=self._action_no,
                    fork=fork.id,
                    index=index,
                    reason=f"rollout death at step {ro.deaths_at}",
                )
                continue
            self._tree.start_path(fork.id, index)
            self._fork_id = fork.id
            # The hypothesis carries the segment's id, so perseverance (``_no_other_way``,
            # ``_ev_gain``) and the questions find it under ``_hypothesis_id``.
            seg_id = hypothesis_id(fork.goal_id, FORK_SOURCE, path)
            hyp = from_rollout(
                fork.goal_id,
                "search",
                path,
                ro,
                typical=self._typical(),
                restart_moves=self._level_moves + len(path),
            )
            hyp = replace(hyp, id=seg_id)
            self._rollouts[seg_id] = ro
            same = (hyp.goal_id, hyp.plan)
            self._hyps = rank([hyp, *[h for h in self._hyps if (h.goal_id, h.plan) != same]])
            items = [PlanItem(a, f"fork {fork.id} path {index}") for a in path]
            self._set_segment(
                scene,
                items,
                FORK_SOURCE,
                seg_id,
                fork.goal_id,
                p_path=ro.path_probability,
                ev=hyp.expected_value,
                protect=protect,
            )
            self._segment_path = (fork.id, index)
            return True
        return False

    def _give_up_fork(self, fork: Fork, reason: str) -> None:
        """Fail the fork's untried paths for this level instance, each logged."""
        for index in self._tree.untried(fork.id):
            self._tree.fail_branch(fork.id, index)
            self._log.event(
                "branch_failed", step=self._action_no, fork=fork.id, index=index, reason=reason
            )

    def _replan_root(self, reason: str) -> None:
        self._pending_fork = None
        self._log.event("replan_root", step=self._action_no, reason=reason)

    # -- the planning loop ---------------------------------------------------------

    def _typical(self) -> float:
        """The level's typical moves per completion (horizon rule, 4.7): this game's
        completions, else every game's at this level index, else ``HORIZON_CAP``."""
        store = self._store
        typical: float | None = None
        if store is not None:
            typical = store.typical_moves(self.game, self._level)
            if typical is None:
                typical = store.typical_for_level(self._level)
        return max(1.0, float(HORIZON_CAP if typical is None else typical))

    def _replan(self, obs: Observation, scene: Scene, decision: Decision) -> None:
        """The planning loop (spec 3.2 to 3.5): hypotheses, the planner call, the
        expected-value choice, the segment and any fork."""
        self._llm_plan, self._llm_test = [], None
        self._clear_segment()
        self._pending_fork = None
        self._rollouts, self._expects, self._protects = {}, {}, {}
        typical = self._typical()
        horizon = max(1, math.ceil(typical))
        budget = self._budget.remaining()
        max_steps = min(horizon, SEGMENT_CAP)
        if budget is not None:
            max_steps = max(1, min(max_steps, budget))
        searched: set[str] = set()
        goals = self._propose(scene)
        hyps = self._code_hypotheses(scene, goals[:TOP_GOALS], typical, horizon, searched)
        self._hyps = rank(hyps)
        proposals: list[Hypothesis] = []
        if self._can_call():
            wake = self._wake_for(decision)
            self._wake = wake
            proposals, named, changed = self._consult(obs, scene, wake, horizon, max_steps, typical)
            if self._gate_trip is not None:
                return
            if changed:
                # The reply's claims and goals can give the search what it lacked: one
                # more pass over the named goals, then the guesses they make possible.
                candidates = {g.id(): g for g in [*named, *self._propose(scene)]}
                fresh = [g for gid, g in candidates.items() if gid not in searched][:TOP_GOALS]
                hyps += self._code_hypotheses(scene, fresh, typical, horizon, searched)
        ranked = rank(self._dedupe([*proposals, *hyps]))
        ranked = [h for h in ranked if not self._failed_here(obs, h) and not self._dies(h)]
        self._hyps = ranked
        if not ranked:
            self._quiet_until = self._action_no + CALL_GAP_MOVES
            return
        reaching = [h for h in ranked if h.stop == "goal"]
        winner = ranked[0]
        if not reaching and winner.path_probability < BAND_LOW:
            self._quiet_until = self._action_no + CALL_GAP_MOVES
            return
        protect = self._protects.get(winner.id, mover_sigs(self.mem))
        expects = self._expects.get(winner.id, [])
        items = [
            PlanItem(a, expects[i] if i < len(expects) else "") for i, a in enumerate(winner.plan)
        ]
        self._set_segment(
            scene,
            items,
            winner.source,
            winner.id,
            winner.goal_id,
            p_path=winner.path_probability,
            ev=winner.expected_value,
            protect=protect,
        )
        if reaching:
            self._maybe_fork(obs, winner, reaching)

    def _propose(self, scene: Scene) -> list[Goal]:
        """New goal guesses into the book (as ``_plan_choice`` does), then the ranked goals."""
        added = self.goals.propose(
            scene, self.mem, source=GOAL_SOURCE, grid=self._last_grid, static=self._static
        )
        if added:
            self._log_goals(added, "open")
            self._log_goals(self._rule_out(added), "ruled_out")
        return self._ranked(scene)

    def _rollout_goals(self) -> list[Goal]:
        """The goals a rollout may reach: this move's sequence steps, then the open goals."""
        return list(dict.fromkeys([*self._seq_now, *self.goals.open()]))[:ROLLOUT_GOALS]

    def _code_hypotheses(
        self,
        scene: Scene,
        goals: Sequence[Goal],
        typical: float,
        horizon: int,
        searched: set[str],
    ) -> list[Hypothesis]:
        """One search plan per goal, rolled out with probabilities (4.7a steps 1 to 3).
        Every goal tried joins ``searched`` so a later pass in the same replan skips it."""
        out: list[Hypothesis] = []
        for goal in goals:
            searched.add(goal.id())
            if goal.holds(scene, self._last_grid):
                continue
            res = self._search(scene, [goal])
            self._frontier = res.frontier
            if res.stop != "found" or not res.plan or res.goal is None:
                continue
            actions = [s.action for s in res.plan]
            protect = protected_sigs(self.mem, [goal], scene)
            ro = rollout(
                scene,
                actions,
                self.mem,
                [goal],
                protect,
                horizon=horizon,
                health=self._health or 1,
                adverse=self._harm_of,
            )
            hyp = from_rollout(
                goal.id(),
                "search",
                actions,
                ro,
                typical=typical,
                restart_moves=self._level_moves + len(actions),
            )
            self._rollouts[hyp.id] = ro
            self._protects[hyp.id] = protect
            self._expects[hyp.id] = [f"toward {goal.id()}"] * len(actions)
            self._goal_by_id[goal.id()] = goal
            out.append(hyp)
        return out

    @staticmethod
    def _dedupe(hyps: Sequence[Hypothesis]) -> list[Hypothesis]:
        """One hypothesis per (goal, plan): the first given wins, so the planner's own
        proposal keeps its source when the search found the same plan."""
        seen: dict[tuple[str, tuple[Action, ...]], Hypothesis] = {}
        for h in hyps:
            seen.setdefault((h.goal_id, h.plan), h)
        return list(seen.values())

    def _dies(self, hyp: Hypothesis) -> bool:
        """Whether ``hyp``'s rollout ended in a predicted game over (spec 3.9: an adverse
        path, never played)."""
        ro = self._rollouts.get(hyp.id)
        return hyp.stop == "death" or (ro is not None and ro.deaths_at is not None)

    def _failed_here(self, obs: Observation, hyp: Hypothesis) -> bool:
        """Whether ``hyp``'s plan is a failed path of this state's fork for its goal
        (4.7c rule 7: not retried in the same level instance)."""
        fork_id = self._fork_ids.get((self.explorer.key(obs), hyp.goal_id))
        fork = None if fork_id is None else self._tree.fork(fork_id)
        if fork is None or hyp.plan not in fork.paths:
            return False
        return fork.status[fork.paths.index(hyp.plan)] == "failed"

    def _maybe_fork(
        self, obs: Observation, winner: Hypothesis, reaching: Sequence[Hypothesis]
    ) -> None:
        """Two or more distinct plans to the winner's goal from this state open a fork
        (4.7c rules 1 to 3); the winner's path is started."""
        plans = list(dict.fromkeys(h.plan for h in reaching if h.goal_id == winner.goal_id))
        if len(plans) < 2:
            return
        key = self.explorer.key(obs)
        fork_id = self._tree.open_fork(key, winner.goal_id, plans, self._level_start_moves)
        self._fork_ids[(key, winner.goal_id)] = fork_id
        fork = self._tree.fork(fork_id)
        if fork is None:
            return
        if winner.plan in fork.paths:
            index = fork.paths.index(winner.plan)
            self._tree.start_path(fork_id, index)
            self._segment_path = (fork_id, index)
        self._fork_id = fork_id
        self._log.event(
            "fork_open",
            step=self._action_no,
            id=fork_id,
            goal=winner.goal_id,
            paths=len(fork.paths),
        )

    # -- the planner call -----------------------------------------------------------

    def _gate_left(self) -> float:
        return T_MOVE - (self._clock() - self._move_t0)

    def _can_call(self) -> bool:
        if self._chat is None or self._pool is None or self._turns_off:
            return False
        if self._gate_left() < MIN_CALL_S:
            return False
        fresh = self._notes & {"level start", "game over"}
        return bool(fresh) or self._action_no >= self._quiet_until

    def _wake_for(self, decision: Decision) -> str:
        if decision.rule not in (NO_RULE, DEADLINE_RULE):
            return f"{decision.rule} {decision.method}: {decision.reason}"
        reason = wake_reason(self._notes)
        if reason == "game over" and self._wake_text:
            return self._wake_text
        return reason or "segment needed"

    def _consult(  # noqa: PLR0917  # the call's inputs, one each
        self,
        obs: Observation,
        scene: Scene,
        wake: str,
        horizon: int,
        max_steps: int,
        typical: float,
    ) -> tuple[list[Hypothesis], list[Goal], bool]:
        """Up to two planner calls (3.5): the accepted proposal as a hypothesis, the
        goals the replies named, and whether a reply changed memory (claims or goals)."""
        proposals: list[Hypothesis] = []
        named: list[Goal] = []
        changed = False
        rejection: str | None = None
        for _ in range(2):
            b = self._briefing(obs, scene, wake if rejection is None else rejection, max_steps)
            messages = planner_messages(b, obs.grid, self._prev_grid)
            seed = self.seed * TURN_SEED_STRIDE + self._planner_calls
            got = self._planner_turn(
                messages,
                seed,
                max_steps,
                scene,
                tuple(obs.available_actions),
                self._deadline - self._clock(),
            )
            if got is None:
                self._quiet_until = self._action_no + CALL_GAP_MOVES
                break
            reply, seconds, tokens = got
            seeded, opened = self._absorb(reply, scene)
            named += opened
            changed = changed or bool(seeded or opened)
            proposal, why = self._proposal(reply, scene, max_steps, typical, horizon)
            self._log.event(
                "planner_call",
                step=self._action_no,
                call=self._planner_call_id,
                seconds=round(seconds, 3),
                tokens=tokens,
                segment_len=len(reply.plan),
                skill=None if reply.skill is None else reply.skill.name,
                question=reply.question or None,
                dropped=[*reply.dropped, *([] if why is None else [why])],
            )
            elapsed = self._clock() - self._move_t0
            if elapsed > T_MOVE:
                self._trip("time", elapsed)
                break
            if self._move_tokens > N_TOKEN:
                self._trip("tokens", elapsed)
                break
            if proposal is not None:
                proposals.append(proposal)
                break
            if (
                why is None
                or not why.startswith("segment rejected")
                or self._gate_left() < MIN_CALL_S
            ):
                break
            rejection = why
        return proposals, named, changed

    def _planner_turn(  # noqa: PLR0917  # the call's inputs, one each
        self,
        messages: list[VMessage],
        seed: int,
        max_steps: int,
        scene: Scene,
        available: tuple[int, ...],
        wait_s: float,
    ) -> tuple[Reply, float, int] | None:
        """Submit one planner call and wait at most ``wait_s``, bounded by the move gate;
        the parsed reply, its seconds and its completion tokens, or None after a failure
        (logged as ``exec_failed``, which costs the call only)."""
        chat, pool = self._chat, self._pool
        if chat is None or pool is None:
            return None
        call = self._planner_calls
        self._planner_calls += 1
        self._planner_call_id = call
        try:
            fut = pool.submit(_run_planner, chat, messages, seed)
        except RuntimeError as exc:  # a shut-down pool costs this call, not the game
            self._log_failed(call, f"submit failed: {exc}")
            return None
        done, _ = wait([fut], timeout=max(0.0, min(wait_s, self._gate_left())))
        if not done:
            fut.cancel()  # a call already running finishes on its pool thread, unread
            self._log_failed(call, "timeout")
            return None
        try:
            completion, seconds = fut.result()
        except ImagesRejected:
            self._turns_off = True
            self._log_failed(call, "images rejected")
            return None
        except Exception as exc:  # the call is lost; the move goes on without it
            self._log_failed(call, f"{type(exc).__name__}: {exc}")
            return None
        tokens = max(0, int(completion.completion_tokens))
        self._move_tokens += tokens
        self._tokens_total += tokens
        reply = parse_reply(completion.text, scene, available, max_steps=max_steps)
        return reply, seconds, tokens

    def _absorb(self, reply: Reply, scene: Scene) -> tuple[int, list[Goal]]:
        """Take the reply's claims, model, labels, question and goals: the number of
        claims seeded and the reply's goals that are open afterwards."""
        seeded = len(seed_claims(self.mem, reply.claims))
        if reply.model:
            self._model_text = reply.model
        by_id = scene.by_id()
        for oid, text in reply.labels:
            t = by_id.get(oid)
            if t is not None:
                self._flags.label(t.kind, text)
        self._pending_q = reply.question
        if not reply.goals:
            return seeded, []
        self._prefer_goals(reply.goals, scene)
        return seeded, [g for g in reply.goals if self.goals.status.get(g) == "open"]

    def _proposal(
        self, reply: Reply, scene: Scene, max_steps: int, typical: float, horizon: int
    ) -> tuple[Hypothesis | None, str | None]:
        """The reply's segment or skill as a rolled-out hypothesis, or the reason it was
        dropped (``segment rejected at step k: death|unknown`` asks the planner again)."""
        if reply.plan:
            actions = [item.action for item in reply.plan]
            expects = [item.expect for item in reply.plan]
            source: Source = "planner"
        elif reply.skill is not None:
            result = run_skill(
                reply.skill.name, reply.skill.args, self._skill_context(scene, max_steps)
            )
            if not result.applies:
                return None, f"skill {result.name}: {result.why}"
            actions = list(result.actions)
            expects = [result.why] * len(actions)
            source = "skill"
        else:
            return None, None
        # Only the planner's own goals protect things: a thing with no fact under any
        # other open goal would otherwise make the first step unknown.
        goals = self._rollout_goals()
        protect = protected_sigs(self.mem, reply.goals, scene)
        ro = rollout(
            scene,
            actions,
            self.mem,
            goals,
            protect,
            horizon=horizon,
            health=self._health or 1,
            adverse=self._harm_of,
        )
        if ro.reached is None and ro.stop in ("death", "unknown"):
            at = ro.deaths_at if ro.stop == "death" else ro.unknown_at
            return None, f"segment rejected at step {at}: {ro.stop}"
        goal_id = NO_GOAL if ro.reached is None else ro.reached.id()
        if ro.reached is not None:
            self._goal_by_id[goal_id] = ro.reached
        hyp = from_rollout(
            goal_id,
            source,
            actions,
            ro,
            typical=typical,
            restart_moves=self._level_moves + len(actions),
        )
        self._rollouts[hyp.id] = ro
        self._protects[hyp.id] = protect
        self._expects[hyp.id] = expects
        return hyp, None

    def _skill_context(self, scene: Scene, max_steps: int) -> SkillContext:
        mover = self._lone_mover(scene)
        if mover is None:
            controlled = self._controlled_now
            mover = next(
                (t.kind for t in scene.things if t.sig in controlled or t.kind in controlled), None
            )
        flags = [(f.kind, f.value, f.cost) for f in self._flags.top(FLAG_LINES)]
        return SkillContext(
            scene=scene,
            mem=self.mem,
            kin=self._kin,
            goals=self.goals.open(),
            roles=self._roles_now,
            flags=flags,
            can_click=self._can_click,
            mover=mover,
            click_cell=self._clicks.cell,
            max_steps=max_steps,
        )

    # -- the briefing (spec 4.9a, 4.9c) ------------------------------------------------

    def _briefing(self, obs: Observation, scene: Scene, wake: str, horizon: int) -> Briefing:
        best = self._hyps[0] if self._hyps else None
        self_rec = SelfRecord.from_state(obs.available_actions, self._lone_mover(scene))
        focus = self._focus(self.last_branch or "none", self._clock() - self._move_t0)
        level_rules, game_rules = self._rule_lines(scene)
        return Briefing(
            state=tuple(self._state_lines(scene)),
            actions=(self._actions_line(obs),),
            goals=tuple(self._goal_lines(obs, scene, best)),
            tasks=tuple(self._task_lines(scene, best)),
            level_rules=tuple(level_rules),
            game_rules=tuple(game_rules),
            hypotheses=tuple(self._hypothesis_line(h, scene) for h in self._hyps),
            sequence=tuple(self._sequence_lines(scene)),
            forks=tuple(self._tree.summary()),
            flags=tuple(self._flags.lines(FLAG_LINES)),
            self_focus=(self_rec.line(), focus.line()),
            recent=tuple(reversed(self._moves)),
            lessons=() if self._store is None else tuple(self._store.lessons()),
            questions=self._questions(scene, best),
            model=self._model_text,
            wake=wake,
            level=self._level,
            horizon=horizon,
        )

    def _sequence_lines(self, scene: Scene) -> list[str]:
        """Spec 4.9a item 8: the segment being followed or replaced, one line per step,
        with whether the step was played (the wake line says why it is being replaced)."""
        played = self._segment_played
        return [
            f"{i}. {_action_name(item.action, scene)} expect {item.expect}: "
            f"{'played' if i <= played else 'not played'}"
            for i, item in enumerate(self._segment_items, 1)
        ]

    def _state_lines(self, scene: Scene) -> list[str]:
        """One line per thing: id, colour, size, place, kind, role, rules and their mean
        probability; movers first, then things that changed this level."""
        movers = mover_sigs(self.mem)
        by_subject: dict[str, list[Fact]] = {}
        for fact in self.mem.facts.values():
            if fact.status != "broken":
                by_subject.setdefault(fact.key[1], []).append(fact)

        def order(t: Thing) -> tuple[int, int, int]:
            rank_ = 0 if t.sig in movers or t.kind in movers else 1 if t.id in self._changed else 2
            return (rank_, len(t.cells), t.id)

        out: list[str] = []
        for t in sorted(scene.things, key=order)[:STATE_LINES]:
            facts = by_subject.get(t.kind, [])
            if t.sig != t.kind:
                facts = facts + by_subject.get(t.sig, [])
            p = sum(f.probability for f in facts) / len(facts) if facts else 0.5
            w, h = _size(t)
            role = self._roles_now.get(t.kind, "unknown")
            out.append(
                f"#{t.id} c{t.color} {w}x{h} at ({t.x},{t.y}) kind={t.kind} role={role} "
                f"rules={len(facts)} p={p:.4f}"
            )
        return out

    def _actions_line(self, obs: Observation) -> str:
        names = [f"A{a}" for a in obs.available_actions if a not in (RESET_ID, COORD_ACTION_ID)]
        return f"{' '.join(names) or 'none'}; click: {'yes' if self._can_click else 'no'}"

    def _goal_lines(self, obs: Observation, scene: Scene, best: Hypothesis | None) -> list[str]:
        level = "level: win the level (open)"
        if best is not None and best.goal_id != NO_GOAL:
            goal = self._goal_by_id.get(best.goal_id)
            status = "open" if goal is None else self.goals.status.get(goal, "open")
            level = f"level: {name_sigs(best.goal_id, scene)} ({status})"
        game = f"game: win every level ({obs.levels_completed}/{obs.win_levels})"
        return [level, game]

    def _task_lines(self, scene: Scene, best: Hypothesis | None) -> list[str]:
        if best is None:
            return []
        expects = self._expects.get(best.id, [])
        return [
            f"{i}. {_action_name(a, scene)} expect {expects[i - 1] if i <= len(expects) else ''}"
            for i, a in enumerate(best.plan, 1)
        ]

    def _rule_lines(self, scene: Scene) -> tuple[list[str], list[str]]:
        """This level's facts by last evidence (newest first), and the other facts ranked
        by probability; sigs are named by on-screen ids."""
        facts = list(self.mem.facts.values())
        level = [f for f in facts if f.level == self._level]
        level.sort(key=lambda f: -(f.evidence[-1] if f.evidence else -1))
        others = {f.key: f for f in facts if f.level != self._level}
        ranked = [others[k] for k in self.mem.ranked(others)]

        def lines(facts_: Sequence[Fact]) -> list[str]:
            return [name_sigs(f.line(), scene) for f in facts_[:RULE_LINES]]

        return lines(level), lines(ranked)

    def _hypothesis_line(self, h: Hypothesis, scene: Scene) -> str:
        return (
            f"{h.id} {h.source} goal {name_sigs(h.goal_id, scene)} p={h.path_probability:.4f} "
            f"ev={h.expected_value:.2f} moves={h.expected_moves:.1f} stop={h.stop}"
        )

    def _questions(self, scene: Scene, best: Hypothesis | None) -> tuple[tuple[str, str], ...]:
        """The seven answers of the question library (4.9c); the one the planner asked
        comes first, marked."""
        answers = {
            "Q-BLOCK": self._q_block(scene, best),
            "Q-REWARD": self._q_reward(scene),
            "Q-HARM": self._q_harm(best),
            "Q-FORK": self._q_fork(),
            "Q-CHECK": self._q_check(),
            "Q-SAFE": self._q_safe(best),
            "Q-UNKNOWN": self._q_unknown(),
        }
        order = list(QUESTION_IDS)
        asked = self._pending_q
        if asked in order:
            order.remove(asked)
            order.insert(0, asked)
            answers[asked] = f"(asked) {answers[asked]}"
        return tuple((qid, answers[qid]) for qid in order)

    def _q_block(self, scene: Scene, best: Hypothesis | None) -> str:
        goal = None if best is None else self._goal_by_id.get(best.goal_id)
        if goal is None or goal.kind not in ("reach", "cover"):
            return "no reach goal chosen"
        body = self._kin.body(scene, goal.a)
        targets = [t for t in scene.things if goal.b in (t.sig, t.kind)]
        if body is None or not targets:
            return "mover or target off screen"
        target = min(targets, key=lambda t: _nearest_gap(body, t.cells))
        end = min(target.cells, key=lambda c: abs(c[0] - body.x) + abs(c[1] - body.y))
        corridor = _corridor((body.x, body.y), end)
        skip = {body.id, target.id}
        hits = {
            t.kind for t in scene.things if t.id not in skip and not t.cells.isdisjoint(corridor)
        }
        if not hits:
            return "nothing on the straight route"
        return "; ".join(f"{k} role={self._roles_now.get(k, 'unknown')}" for k in sorted(hits))

    def _q_reward(self, scene: Scene) -> str:
        parts: list[str] = []
        ev_by_goal = {h.goal_id: h.expected_value for h in reversed(self._hyps)}
        for goal in self.goals.open()[:FLAG_LINES]:
            if goal.kind in ("reach", "cover"):
                ev = ev_by_goal.get(goal.id())
                tail = "" if ev is None else f" ev={ev:.2f}"
                parts.append(f"{name_sigs(goal.b, scene)} goal {goal.kind}{tail}")
        kinds = sorted(k for k, r in self._roles_now.items() if r == "collectible")
        parts += [f"{k} collectible" for k in kinds[:FLAG_LINES]]
        return "; ".join(parts) or "no goal target or collectible in view"

    def _q_harm(self, best: Hypothesis | None) -> str:
        ro = None if best is None else self._rollouts.get(best.id)
        if ro is None:
            return "no hypothesis"
        harm = dict(ro.harm)
        first = harm.get(0, 0.0)
        if not harm:
            return f"first step p={first:.2f}; no harm on the path"
        step, p = max(harm.items(), key=lambda kv: (kv[1], -kv[0]))
        return f"first step p={first:.2f}; worst step {step} p={p:.2f}"

    def _q_fork(self) -> str:
        fork = self._tree.newest_with_untried()
        if fork is None:
            return "no open fork"
        untried = self._tree.untried(fork.id)
        return (
            f"fork {fork.id} at goal {fork.goal_id}: {len(untried)} untried of "
            f"{len(fork.paths)} paths, next path {untried[0]}"
        )

    def _q_check(self) -> str:
        lines = self._flags.lines(1)
        return lines[0] if lines else "nothing flagged"

    def _q_safe(self, best: Hypothesis | None) -> str:
        ro = None if best is None else self._rollouts.get(best.id)
        if ro is None:
            return "no hypothesis"
        if ro.stop == "goal" and ro.reached is not None:
            health = self._health or 1
            return f"yes: reaches {ro.reached.id()} in {len(ro.scenes)} steps, health {health}"
        at = ro.deaths_at if ro.deaths_at is not None else ro.unknown_at
        where = "" if at is None else f" at step {at}"
        return f"no: stops {ro.stop}{where}"

    def _q_unknown(self) -> str:
        pairs = self._frontier[:QUESTION_PAIRS]
        return "; ".join(question(label, subject) for label, subject in pairs) or "no open pair"

    # -- roles, harm and vetoes (spec 4.10) --------------------------------------------

    def _roles(self, scene: Scene) -> dict[str, Role]:
        """Roles per kind (and per sig): controlled, mover, harm, blocker, collectible or
        unknown, from the facts and the adverse book (4.8b rule 5: inferred, never named)."""
        controlled = self._controlled_now
        movers = mover_sigs(self.mem)
        harm_kinds, harm_colors = self._harm_marks()
        blockers: set[int] = set()
        for name in controlled:
            blockers |= self.mem.blockers.get(name, set())
        collectible = {
            f.key[1]
            for f in self.mem.by_tag("collect")
            if f.status != "broken" and f.key[0] not in ("any", "event")
        }
        out: dict[str, Role] = {}
        for t in scene.things:
            role: Role
            if t.sig in controlled or t.kind in controlled:
                role = "controlled"
            elif t.sig in movers or t.kind in movers:
                role = "mover"
            elif t.kind in harm_kinds or t.sig in harm_kinds or t.color in harm_colors:
                role = "harm"
            elif t.color in blockers:
                role = "blocker"
            elif t.sig in collectible or t.kind in collectible:
                role = "collectible"
            else:
                role = "unknown"
            out[t.kind] = role
            out.setdefault(t.sig, role)
        return out

    def _harm_marks(self) -> tuple[set[str], set[int]]:
        """Kinds the adverse book has seen harm from (touch keys at or above ``BAND_LOW``
        with at least one harm), and colours a death fact names."""
        known = self._adverse.keys()  # a method of the book, not a dict view
        kinds = {
            key[1]
            for key in known
            if key[2] == "touch"
            and self._adverse.harms(key) > 0
            and self._adverse.probability(key) >= BAND_LOW
        }
        colors: set[int] = set()
        for fact in self.mem.by_tag("die"):
            tag, _, color = fact.key[2].partition(":")
            if fact.status != "broken" and tag in ("touch", "near") and color.isdigit():
                colors.add(int(color))
        return kinds, colors

    def _features(self, label: str, kind: str, shape: str, context: str) -> Features:
        role = self._roles_now.get(kind, "unknown")
        return features(
            label, kind, shape, context, roles=frozenset({role}), health_bar=self._bar_is_health
        )

    def _gait_step(self, body: Thing, label: str) -> tuple[int, int] | None:
        """The unit displacement ``label`` gives ``body`` (a slide reads as one step)."""
        for name in (body.kind, body.sig):
            for gait in self._kin.gaits(name):
                if gait.action == label:
                    return gait.dx, gait.dy
        return None

    def _harm_detail(self, scene: Scene, action: Action) -> tuple[float, AdverseKey]:
        """The harm probability of ``action`` from ``scene`` and the key it came from: the
        largest over the kinds the controlled bodies would overlap or touch after one
        gait step; the ``self`` key when no body has a gait for it or nothing is touched."""
        label = action_label(action, scene)
        controlled = self._controlled_now
        bodies = [t for t in scene.things if t.sig in controlled or t.kind in controlled]
        mine = {t.id for t in bodies}
        candidates: list[tuple[AdverseKey, str, str]] = []
        seen: set[str] = set()
        for body in bodies:
            step = self._gait_step(body, label)
            if step is None:
                continue
            near = {
                (x + step[0] + ox, y + step[1] + oy)
                for x, y in body.cells
                for ox, oy in _NEIGHBOURS
            }
            for t in scene.things:
                if t.id in mine or t.kind in seen or t.cells.isdisjoint(near):
                    continue
                seen.add(t.kind)
                candidates.append(((label, t.kind, "touch"), t.kind, t.shape))
        if not candidates:
            candidates.append(((label, "self", "any"), "self", "self"))
        best_p, best_key = -1.0, candidates[0][0]
        for key, kind, shape in candidates:
            p, _ = harm_probability(
                self._adverse, self._net, key, self._features(label, kind, shape, key[2])
            )
            if p > best_p:
                best_p, best_key = p, key
        return best_p, best_key

    def _harm_of(self, scene: Scene, action: Action) -> float:
        """The rollout's adverse function: ``_harm_detail``'s probability."""
        return self._harm_detail(scene, action)[0]

    def _no_other_way(self) -> bool:
        """Perseverance reason A: no other hypothesis reaches the goal."""
        return not any(h.stop == "goal" and h.id != self._hypothesis_id for h in self._hyps)

    def _ev_gain(self) -> float:
        """Perseverance reason B: the active plan's expected value over the best other."""
        current = next((h for h in self._hyps if h.id == self._hypothesis_id), None)
        if current is None:
            return 0.0
        others = [h.expected_value for h in self._hyps if h.id != current.id]
        return current.expected_value - (max(others) if others else 0.0)

    def _veto(self, scene: Scene, action: Action) -> str | None:
        reason = super()._veto(scene, action)
        if reason is not None or self._chat is None or action.id == RESET_ID:
            return reason
        p, key = self._harm_detail(scene, action)
        band = band_of(p)
        if band == "low":
            return None
        if band == "certain":
            return ADVERSE_CERTAIN
        allowed, _ = self._persevere.allow(
            p,
            no_other_way=self._no_other_way(),
            ev_gain=self._ev_gain(),
            trials_used=self._trials.get(key, 0),
        )
        return None if allowed else ADVERSE_MEDIUM

    def _note_trial(self, scene: Scene, action: Action) -> None:
        """A medium-band move that was taken is a trial of its key (4.10c rule 7)."""
        p, key = self._harm_detail(scene, action)
        if band_of(p) != "medium":
            return
        used = self._trials.get(key, 0)
        _, reason = self._persevere.allow(
            p, no_other_way=self._no_other_way(), ev_gain=self._ev_gain(), trials_used=used
        )
        self._trials[key] = used + 1
        self._log.event(
            "perseverance",
            step=self._action_no,
            action=[action.id, action.x, action.y],
            p=p,
            reason=reason,
        )

    # -- learning from a transition (spec 3.7, 4.10a, 4.11a) -----------------------------

    def _transition(self, obs: Observation, move: int, before_s: Scene, action: Action) -> Scene:
        following = self._llm_step and bool(self._segment_id)
        if following:
            self._segment_played += 1
        after = super()._transition(obs, move, before_s, action)
        pred = self._pred
        self._prev_matched = (
            pred is not None and pred.scene is not None and pred.scene.key() == after.key()
        )
        self._segment_broke = following and "miss that matters" in self._notes
        if self._segment_broke:
            self._fail_active("miss that matters")
        harmed = self._note_bar(obs.grid)
        self._learn_harm(before_s, action, after, harmed, move)
        self._note_flags(after)
        return after

    def _note_bar(self, grid: Grid) -> bool:
        """Follow the counter bar; True when it is a health bar and this move lowered it."""
        counter = self._budget.counter
        if counter is None:
            self._bar_last = None
            return False
        cells = bar_cells(grid, counter)
        last, self._bar_last = self._bar_last, cells
        if last is None:
            return False
        self._bar_moves += 1
        drop = last - cells if counter.shrinks else cells - last
        if drop > 0:
            self._bar_drops += 1
            self._bar_drop_cells += drop
        if not self._bar_is_health and is_health_bar(drops=self._bar_drops, moves=self._bar_moves):
            self._bar_is_health = True
        if not self._bar_is_health:
            return False
        per_hit = max(1, round(self._bar_drop_cells / self._bar_drops)) if self._bar_drops else 1
        left = cells if counter.shrinks else len(counter.lines) * GRID_SIZE - cells
        self._health = max(1, math.ceil(left / per_hit))
        return drop > 0

    def _learn_harm(
        self, before: Scene, action: Action, after: Scene, harmed: bool, move: int
    ) -> None:
        """Record the move against the kinds the controlled things touch in ``after`` (the
        ``self`` key when none) in the adverse book and the network; log ``harm``."""
        label = action_label(action, before)
        touched = sorted(self._touched(after))
        shapes = {t.kind: t.shape for t in after.things}
        keys: list[AdverseKey] = [(label, k, "touch") for k in touched] or [(label, "self", "any")]
        for key in keys:
            kind = key[1]
            self._adverse.record(key, harmed)
            self._net.update(self._features(label, kind, shapes.get(kind, kind), key[2]), harmed)
        if harmed:
            first = keys[0]
            self._log.event(
                "harm", step=move, kind=first[1], label=label, p=self._adverse.probability(first)
            )

    def _note_flags(self, after: Scene) -> None:
        """Flags per kind on screen (4.11a rule 1): a new look or unknown pairs set
        ``check``; the cost is the cells to the controlled body plus one."""
        unknown: Counter[str] = Counter(subject for _, subject in self._frontier)
        controlled = self._controlled_now
        body = next((t for t in after.things if t.sig in controlled or t.kind in controlled), None)
        for t in after.things:
            new_look = t.kind not in self._kinds_seen
            self._kinds_seen.add(t.kind)
            pairs = unknown.get(t.kind, 0) + (unknown.get(t.sig, 0) if t.sig != t.kind else 0)
            before = self._flags.get(t.kind)
            was = None if before is None else before.state
            flag = self._flags.note(t.kind, new_look=new_look, unknown_pairs=pairs)
            cost = (
                1.0 if body is None or body.id == t.id else float(_nearest_gap(body, t.cells) + 1)
            )
            self._flags.set_cost(t.kind, cost)
            if flag.state == "check" and was != "check":
                self._log.event(
                    "flag",
                    step=self._action_no,
                    kind=t.kind,
                    state=flag.state,
                    source=flag.source,
                    value=flag.value,
                    cost=flag.cost,
                )

    # -- level up, game over, close (spec 4.12) ---------------------------------------------

    def _game_over(self, obs: Observation, move: int) -> None:
        before, action, pred, fresh = self._last_scene, self._last_action, self._pred, self._fresh
        super()._game_over(obs, move)
        if not fresh and before is not None and action is not None:
            after = pred.scene if pred is not None and pred.scene is not None else before
            self._learn_harm(before, action, after, True, move)
        self._fail_active("game over")
        self._end_instance()

    def _level_up(self, obs: Observation, move: int, before: Scene, action: Action) -> None:
        completed, moves = self._level, self._level_moves
        sequence = list(self._level_start_moves)
        super()._level_up(obs, move, before, action)
        self._consolidate(completed, moves, sequence)
        self._tree.reset_level()
        self._fork_ids = {}
        self._flags.reset_level()
        self._bar_is_health = False
        self._end_instance()

    def _end_instance(self) -> None:
        """A level instance ended (level up or game over): the per-instance state restarts,
        the handoff counters with it (module docstring step 7)."""
        self._fallbacks = 0
        self._no_progress = 0
        self._segment_items = []
        self._segment_played = 0
        self._level_moves = 0
        self._level_start_moves = []
        self._trials = {}
        self._bar_moves = self._bar_drops = self._bar_drop_cells = 0
        self._bar_last = None
        self._health = None
        self._llm_plan, self._llm_test = [], None
        self._clear_segment()
        self._pending_fork = None
        self._route_tries = {}
        self._segment_broke = False
        self._prev_matched = None
        self._hyps = []
        self._rollouts, self._expects, self._protects = {}, {}, {}

    def _consolidate(self, level: int, moves: int, sequence: Sequence[Action]) -> None:
        """Level consolidation (4.12b): promote the facts, record the completion and the
        sequence, save the store; log ``consolidate``."""
        promoted = sequences = 0
        typical: float | None = None
        store = self._store
        if store is not None:
            try:
                with store.lock:
                    for fact in list(self.mem.facts.values()):
                        if fact.state != "fact":
                            continue
                        record = self._rule_record(fact)
                        if record is None:
                            continue
                        try:
                            store.promote(record)
                        except ValueError:  # a key that still names a colour: not a long rule
                            continue
                        promoted += 1
                    store.record_completion(self.game, level, moves)
                    if sequence:
                        store.add_sequence(self._sequence_record(level, sequence))
                        sequences = 1
                    store.net_state = self._net_state()
                    store.save()
                typical = store.typical_moves(self.game, level)
            except OSError as exc:  # the store is a convenience; the game goes on
                self._log.event(
                    "agent_error",
                    step=self._action_no,
                    error=f"store: {type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
                )
        self._log.event(
            "consolidate", level=level, promoted=promoted, sequences=sequences, typical=typical
        )

    def _rule_record(self, fact: Fact) -> RuleRecord | None:
        """``fact`` as a rule record carrying the evidence since its last promotion, or
        None when nothing new was gathered (``promote`` adds counts)."""
        hits0, misses0 = self._promoted.get(fact.key, (0, 0))
        confirmations = max(0, fact.hits - hits0)
        contradictions = max(0, fact.misses - misses0)
        if confirmations == 0 and contradictions == 0:
            return None
        self._promoted[fact.key] = (fact.hits, fact.misses)
        key = canonical_key(fact.key, fact.effects)
        label, subject, context = fact.key
        kind: RuleKind = (
            "event" if label == "event" else "action" if subject in _ACTION_SUBJECTS else "object"
        )
        if any(e.kind == "game_over" for e in fact.effects):
            harm = "death"
        else:
            adverse_key: AdverseKey = (
                label,
                subject,
                "touch" if context.startswith("touch:") else "any",
            )
            harmful = (
                self._adverse.harms(adverse_key) > 0
                and self._adverse.probability(adverse_key) >= BAND_LOW
            )
            harm = "harm" if harmful else NO_HARM
        return RuleRecord(
            id=rule_id(key),
            kind=kind,
            statement=statement_for(fact.key, fact.effects),
            canonical_key=key,
            effect=effect_text(fact.effects),
            probability=fact.probability,
            confirmations=confirmations,
            contradictions=contradictions,
            state=fact.state,
            scope="working",
            games_seen=[self.game],
            created_game=self.game,
            created_level=fact.level,
            last_revised=0.0,
            harm=harm,
            source="observed",
        )

    def _sequence_record(self, level: int, moves: Sequence[Action]) -> SequenceRecord:
        triples = [[a.id, a.x, a.y] for a in moves]
        text = f"{self.game}\x1f{level}\x1f" + " ".join(f"{a.id},{a.x},{a.y}" for a in moves)
        digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        forms = self.goals.confirmed_forms
        goal_id = forms[-1].id() if forms else f"level {level}"
        return SequenceRecord(
            id=f"s{digest}",
            goal_id=goal_id,
            moves=triples,
            game=self.game,
            level=level,
            runs=1,
            successes=1,
            deterministic=False,
        )

    def _net_state(self) -> dict[str, object]:
        """The network's weights and update count in the store's opaque shape."""
        state: dict[str, object] = {**self._net.state()}
        return state

    def close(self) -> None:
        if self._closed:
            return
        store = self._store
        if store is not None:
            try:
                with store.lock:
                    store.net_state = self._net_state()
                    store.save()
            except OSError as exc:  # the game end still gets logged
                self._log.event(
                    "agent_error",
                    step=self._action_no,
                    error=f"store: {type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
                )
            self._log.event(
                "longterm",
                game_id=self.game,
                rules=len(store.rules),
                sequences=len(store.sequences),
            )
        self._calls = self._planner_calls  # ``game_end.turns`` counts planner calls
        super().close()
