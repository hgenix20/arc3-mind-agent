# Stack + Genix Mind + Offline LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Kaggle ARC-AGI-3 agent built from our own parts: the stack (explorer, information-gain probes, A*), Genix Mind's loops and gates, and Qwen3.6-27B writing game rules offline.

**Architecture:** A new package `arcagent/mind/` holds one file per mind part (perception, predictive loop, salience, wisdom, working memory and store, sleep, drive state, executive) plus the four HCQM modules as `Protocol`s with null versions. LLM-written rule programs run in a persistent sandbox subprocess per game, which also runs the A* planner (`arcagent/plan/planner.py`, stdlib only) so search never leaves the sandbox. Games run concurrently against one scorecard and one vLLM server; every action and event goes to a JSONL log that one metrics script reads.

**Tech Stack:** Python 3.12, numpy, the competition's `arc_agi`/`arcengine` wheels, vLLM 0.19 serving Qwen3.6-27B-FP8 (OpenAI-compatible HTTP), pytest, ruff, pyright strict.

**Spec:** `docs/superpowers/specs/2026-10-23-stack-mind-llm-design.md`

## Global Constraints

- Nothing from Franzen in the entry: no code, prompt, configuration, or the names "duck" or "franzen" in `arcagent/` or in the notebook builder. Task 1 adds a test that enforces it.
- Interpreter: `C:\arc-agent-venv\Scripts\python.exe`. Run every command from `C:\arc-agent`.
- Completion gate for every task: `ruff format`, `ruff check`, `pyright`, and `pytest` all pass on the whole repo.
- Commits end with exactly one trailer line, `Claude-Session: https://claude.ai/code/session_01ED1JLjSNaMyvkVda5AScAg`. No Co-Authored-By line.
- Prose in docs and commit messages: no em-dashes, contractions allowed, no stock AI words ("robust", "seamless", "surface" as a verb, "load-bearing").
- Kaggle: notebook only, internet off, under 9 hours, RTX PRO 6000, one submission per day, deadline 2026-11-02 23:59 UTC. Both final selections are this agent.
- Model: Qwen3.6-27B-FP8 under vLLM 0.19 from datasets `driessmit1/arc3-vllm-h100-wheelhouse-v3` and `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`, thinking off (`ENABLE_THINKING = False` in `arcagent/synth/run.py`).
- The scorecard closes after 15 minutes with no action. Nothing in the agent may let every game go idle that long.
- No generative model decides salience, forgetting or consolidation; the predictive loop never chooses an action; every drop by the wisdom function is a logged event.
- Model-written code runs only inside the sandbox subprocess (`python -I`, blocked file/process/network builtins, per-request timeout). It never runs in the agent's own process.
- Every loop and search is bounded by count and by time.
- Keep `docs/experiment-log.md` current: one row per GPU eval or submission.

## Review Focus

1. **Model code that hangs, loops forever, or allocates without limit.** The request times out, the worker restarts with its other programs reloaded, the program is dropped with reason `timeout`, and the game keeps acting. Test in Task 3.
2. **Every game waiting on the LLM at once.** No game may leave the scorecard idle past 600 s; the game that sees the idle clock past that takes a B0 action while it waits. Test in Task 12.
3. **The model server failing** (HTTP error, timeout, a reply with no code fence). The call yields zero programs, an `llm_call` event records the error, and the game continues on B0. Test in Task 10.
4. **`step()` returning junk** (None, wrong shape, non-int cells, values outside -1..15). The prediction counts as a miss, never a crash. Test in Task 3.
5. **GAME_OVER in the middle of a plan.** The agent sends RESET, clears the plan, and inhibition vetoes the action that caused the loss in the state it was taken from. Test in Task 12.

---

### Task 1: Remove the Franzen period from the entry

The planner was first written to plug into Franzen's notebook, which is built on Tufa Labs' "Duck" harness; the name comes from there. It becomes `arcagent/plan/planner.py`. Every Franzen-period file moves out of the shipped tree into `archive/franzen-period/`, where git history keeps it for the paper's account and nothing imports it.

**Files:**
- Move: `scripts/duck_planner.py` -> `arcagent/plan/planner.py`; create `arcagent/plan/__init__.py`
- Move: `tests/test_duck_planner.py` -> `tests/test_planner.py`
- Move to `archive/franzen-period/`: `scripts/kaggle_duck_notebook.py`, `scripts/duck_keepalive.py`, `scripts/duck_noise.py`, `tests/test_duck_keepalive.py`, `third_party/franzen-m2/`, `docs/duck-runs/`
- Create: `data/public_human_actions.json`
- Modify: `scripts/plan_ceiling.py`, `pyproject.toml` (pytest `testpaths`, ruff exclude `archive`)
- Create: `tests/test_no_franzen.py`

**Interfaces:**
- Produces: `arcagent.plan.planner.plan(step, start, goal, actions, *, heuristic=None, key=None, max_nodes=200_000, time_limit_s=20.0, max_depth=None) -> PlanResult`, `PlanResult`, `freeze` (unchanged behavior). The module stays standard-library only with no `from __future__`, because Task 3's sandbox worker loads it by file path under `python -I`.
- Produces: `data/public_human_actions.json`, `{"<game4>": [h1, h2, ...]}`: human action counts per level for the 25 public games.

- [ ] **Step 1:** `git mv` the planner and its test, then the archive moves. Rewrite the planner docstring: keep Contract and Bounded, drop the eval-v2 paragraph and the notebook sentence, and state that it's stdlib-only because the sandbox worker loads it by path. Update `tests/test_planner.py` to import `arcagent.plan.planner` normally.
- [ ] **Step 2:** Write `data/public_human_actions.json` from the `agent/human` pairs in `docs/duck-runs/eval-v2-baseline-levels.txt` (read before the move; human counts are the game's, not the run's). Point `scripts/plan_ceiling.py` at it and at `arcagent.plan.planner`.
- [ ] **Step 3: Write the guard test**

```python
# tests/test_no_franzen.py
SHIPPED = [ROOT / "arcagent", ROOT / "scripts" / "kaggle_mind_notebook.py", ROOT / "data"]

def test_shipped_tree_has_no_franzen_period_names():
    hits = [f"{p}:{n}" for p, n, line in iter_lines(SHIPPED)
            if re.search(r"duck|franzen", line, re.IGNORECASE)]
    assert hits == []

def test_human_actions_cover_the_public_games():
    data = json.loads((ROOT / "data" / "public_human_actions.json").read_text())
    assert len(data) == 25 and all(len(v) >= 1 and min(v) > 0 for v in data.values())
```

`iter_lines` walks `.py`, `.json`, `.md` files and skips paths that don't exist yet (the notebook builder arrives in Task 16).

- [ ] **Step 4:** Run `pytest tests/test_planner.py tests/test_no_franzen.py -v`. Expected: 14 pass. Then the full gate.
- [ ] **Step 5:** Commit `Move the planner into arcagent.plan and archive the Franzen-period files`.

---

### Task 2: Mind types and perception

**Files:**
- Create: `arcagent/mind/__init__.py`, `arcagent/mind/types.py`, `arcagent/mind/perception.py`
- Test: `tests/mind/test_perception.py`

**Interfaces:**
- Produces in `types.py`:
  - `Step(index: int, before: Grid, action: Action, after: Grid, levels_before: int, levels_after: int, phase_after: Phase)`, frozen; property `level_up -> bool`.
  - `Program(id: str, code: str, has_goal: bool, origin: Literal["llm", "transfer"])`, frozen.
  - `Event(kind: EventKind, step: int, magnitude: float, detail: Mapping[str, object])`, frozen. `EventKind = Literal["new_game", "prediction_error", "level_up", "game_over", "belief_empty", "transfer_failed"]`.
  - `Gate(verdict: Literal["pass", "caveat", "hold", "drop"], reason: str)`, frozen.
  - `Branch = Literal["follow", "plan", "learn", "fallback", "wait", "reset"]`; `Decision(action: Action, branch: Branch, why: str)`, frozen.
  - `ActionKey = tuple[int, int, int]`.
- Produces in `perception.py`:
  - `frame_hash(grid: Grid) -> str`: first 16 hex chars of `sha1(grid.astype(np.int8).tobytes())`.
  - `Obj(color: int, size: int, bbox: tuple[int, int, int, int], centroid: tuple[float, float])`, frozen; bbox is `(x0, y0, x1, y1)` inclusive.
  - `Percept(grid: Grid, hash: str, objects: tuple[Obj, ...])`; `perceive(grid: Grid) -> Percept` using `arcagent.synth.trace.components` and `background_color`.
  - `changed_cells(a: Grid, b: Grid) -> int`; `changed_bbox(a: Grid, b: Grid) -> tuple[int, int, int, int] | None`.
  - `candidate_actions(obs: Observation) -> list[Action]`: the simple actions in `available_actions` (not RESET, not 6) plus `Action(6, x, y)` for each `click_candidates(obs.grid)` target. This is the one action list the planner, VOI and B0 all use.

- [ ] **Step 1: Tests**

```python
def test_frame_hash_is_stable_and_sensitive():
    g = np.zeros((64, 64), np.int8); h = frame_hash(g)
    assert len(h) == 16 and h == frame_hash(g.copy())
    g2 = g.copy(); g2[5, 7] = 3
    assert frame_hash(g2) != h

def test_perceive_finds_objects_with_bbox():
    g = np.zeros((64, 64), np.int8); g[2:4, 10:13] = 5
    p = perceive(g)
    assert [(o.color, o.size, o.bbox) for o in p.objects] == [(5, 6, (10, 2, 12, 3))]

def test_changed_bbox_none_when_equal():
    g = np.zeros((64, 64), np.int8)
    assert changed_bbox(g, g) is None and changed_cells(g, g) == 0

def test_candidate_actions_excludes_reset_and_expands_clicks():
    obs = Observation(grid_with_two_objects(), Phase.NOT_FINISHED, 0, 3, (0, 1, 6))
    acts = candidate_actions(obs)
    assert Action(1) in acts and all(a.id != 0 for a in acts)
    assert sum(a.id == 6 for a in acts) == 2
```

- [ ] **Step 2:** Run them and watch them fail on import.
- [ ] **Step 3:** Implement the two modules.
- [ ] **Step 4:** Run them to pass, then the full gate.
- [ ] **Step 5:** Commit `Add mind types and perception`.

---

### Task 3: Program sandbox

One worker subprocess per game holds that game's loaded programs and answers requests over stdin/stdout JSON lines. It also runs the planner on a program, so search costs no round trips.

**Files:**
- Create: `arcagent/mind/sandbox.py` (parent side), `arcagent/mind/_worker.py` (child side, standard library only, no `from __future__`, no `arcagent` imports)
- Test: `tests/mind/test_sandbox.py`

**Interfaces:**
- Consumes: `arcagent/plan/planner.py` (the worker loads it by file path), `frame_hash` (Task 2: the worker computes the same hash from list grids, `sha1(bytes(v & 0xFF for row in grid for v in row))[:16]`).
- Produces in `sandbox.py`:
  - `class ProgramSandbox`: `__init__(self, *, timeout_s: float = 5.0, plan_timeout_s: float = 30.0)`, context manager.
  - `load(pid: str, code: str) -> str | None`: the load error, or None.
  - `unload(pid: str) -> None`.
  - `predict(pids: Sequence[str], grid: Grid, action: Action) -> dict[str, str | None]`: each program's predicted frame hash; None means crash, timeout, or an invalid grid.
  - `predict_grid(pid: str, grid: Grid, action: Action) -> Grid | None`.
  - `predict_many(pids: Sequence[str], grid: Grid, actions: Sequence[Action]) -> dict[tuple[str, ActionKey], str | None]`.
  - `goal(pids: Sequence[str], grid: Grid) -> dict[str, bool | None]`: None when the program has no `goal` or it raised.
  - `score(pid: str, steps: Sequence[Step]) -> tuple[int, int]`: `(exact_hits, trials)` over the steps.
  - `plan(pid: str, grid: Grid, actions: Sequence[Action], *, target: Literal["goal", "novel"], visited: frozenset[str], max_nodes: int, time_limit_s: float) -> SandboxPlan`.
  - `SandboxPlan(actions: tuple[Action, ...] | None, stopped: str, expanded: int, seconds: float, errors: int)`, frozen.
  - `restarts: int`: how many times the worker was killed and restarted.
- A valid predicted grid is 64 rows of 64 ints in -1..15. Anything else counts as None. With `target="novel"`, the goal is "frame hash not in `visited`".

- [ ] **Step 1: Tests** (each uses a real worker)

```python
GOOD = "def step(grid, action, x=-1, y=-1):\n    g=[r[:] for r in grid]; g[0][0]=action; return g\n"
HANG = "def step(grid, action, x=-1, y=-1):\n    while True: pass\n"
JUNK = "def step(grid, action, x=-1, y=-1):\n    return None\n"
EVIL = "import os\ndef step(grid, action, x=-1, y=-1):\n    return os.listdir('.')\n"

def test_predict_matches_frame_hash(): ...      # GOOD; predicted hash == frame_hash(expected grid)
def test_hang_times_out_restarts_and_keeps_others(sb):
    sb.load("good", GOOD); sb.load("hang", HANG)
    out = sb.predict(["good", "hang"], zeros(), Action(1))
    assert out["hang"] is None and out["good"] is not None and sb.restarts == 1
    assert sb.predict(["good"], zeros(), Action(2))["good"] is not None   # reloaded after restart
def test_junk_and_wrong_shape_are_misses(): ...  # JUNK, 63-row grid, value 99 -> None
def test_blocked_imports_fail_to_load(): assert sb.load("evil", EVIL) is not None
def test_score_counts_exact_hits(): ...          # 3 steps, GOOD exact on 2 -> (2, 3)
def test_plan_reaches_goal_and_novel_targets(): ... # a 1-D walker program with goal(); goal plan length 3; novel plan length 1
def test_plan_respects_time_limit(): ...          # time_limit_s=0.0 -> stopped "time_limit"
```

- [ ] **Step 2:** Run and watch them fail.
- [ ] **Step 3:** Implement. Block builtins the way `arcagent/synth/verify.py`'s `_RUNNER` does, and block `__import__` of everything except `math`, `itertools`, `collections`, `functools`, `heapq`, `copy`, `re`. The parent reads replies with a deadline (a reader thread plus a queue); on timeout it kills the process, starts a new worker, and reloads every program except the one that timed out, which gets unloaded with reason `timeout`. The worker caps its own memory with `resource.setrlimit(RLIMIT_AS, 2 GB)` where `resource` exists (Kaggle is Linux; on Windows the timeout is the only bound).
- [ ] **Step 4:** Run them to pass, then the full gate.
- [ ] **Step 5:** Commit `Add the program sandbox`.

---

### Task 4: Run log and drive state

**Files:**
- Create: `arcagent/mind/runlog.py`, `arcagent/mind/drive.py`
- Test: `tests/mind/test_runlog.py`, `tests/mind/test_drive.py`

**Interfaces:**
- Produces in `runlog.py`:
  - `class RunLog`: `__init__(self, path: Path, game: str, config: str, seed: int)`; `action(**fields: object) -> None` writes `{"type": "action", "t", "game", "config", "seed", ...fields}`; `event(kind: str, **fields: object) -> None` writes `{"type": kind, ...}`; one JSON line per call, flushed; numpy values converted to plain ints and floats; `close()`.
  - Action record fields (Task 12 fills them; Task 15 reads them): `level, step, frame_hash, action [id, x, y], branch, why, programs [{"id", "pred", "matched", "fit"}], vetoes [[id, x, y]], voi {"id,x,y": score}, entropy, wake`.
  - Event kinds: `llm_call, plan, level_up, game_over, program_admitted, program_dropped, veto, wisdom_drop, sleep, transfer, game_end`.
- Produces in `drive.py`:
  - `DriveState(started: float, game_deadline: float, run_deadline: float, now: float, level_actions: int, synth_calls_left: int, queue_depth: int)`, frozen.
  - `fatigue(d: DriveState) -> float`: `(now - started) / (min(game_deadline, run_deadline) - started)`, clamped to [0, 1].
  - `planner_bounds(d: DriveState, base_nodes: int = 50_000, base_seconds: float = 10.0) -> tuple[int, float]`: both scaled by `1 - 0.75 * fatigue`, floors of 2_000 nodes and 1.0 s.

- [ ] **Step 1: Tests:** `test_runlog_writes_one_json_line_per_call_with_plain_types` (an np.int64 field reads back as int); `test_runlog_lines_survive_without_close` (read the file before `close`); `test_fatigue_clamps_and_uses_earlier_deadline`; `test_planner_bounds_shrink_to_floor` (fatigue 1.0 gives (12_500, 2.5); a tiny base hits the floors).
- [ ] **Step 2:** Watch them fail. **Step 3:** Implement. **Step 4:** Pass, then the full gate.
- [ ] **Step 5:** Commit `Add the run log and drive state`.

---

### Task 5: Working memory and the game store

**Files:**
- Create: `arcagent/mind/memory.py`
- Test: `tests/mind/test_memory.py`

**Interfaces:**
- Produces:
  - `class WorkingMemory`: `steps: list[Step]` (the current level's episode); `put(key: str, value: object, ttl: int) -> None`; `get(key: str) -> object | None`; `tick(store: GameStore) -> list[str]` advances one action and moves expired items to `store.cold`, returning their keys; `recall(key: str, store: GameStore) -> object | None` restores a cold item with double its last ttl (repetition); `clear_level() -> list[Step]` returns and empties `steps` and the timed items (used by sleep).
  - `class GameStore`: `episodes: list[list[Step]]` (one list per finished level), `procedural: dict[str, tuple[Program, int, int]]` (program, hits, trials), `held: dict[str, Program]` (programs wisdom held for sleep), `goal_evidence: list[tuple[Grid, Grid]]` (before and after of each level-up), `cold: dict[str, tuple[object, int]]` (value, last ttl).
  - `all_steps(store: GameStore, wm: WorkingMemory) -> list[Step]`.
- Default ttl: `DEFAULT_TTL = 200` actions.

- [ ] **Step 1: Tests:** `test_item_expires_to_cold_after_ttl_ticks`; `test_recall_restores_with_double_ttl`; `test_clear_level_returns_steps_and_empties`; `test_all_steps_orders_episodes_then_current`.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add working memory and the game store`.

---

### Task 6: Predictive loop (belief)

**Files:**
- Create: `arcagent/mind/predictive.py`
- Test: `tests/mind/test_predictive.py`

**Interfaces:**
- Consumes: `ProgramSandbox.predict`, `.goal`, `.load`, `.unload` (Task 3); `Step`, `Program` (Task 2).
- Produces:
  - `ProgramState(program: Program, hits: int, trials: int, misses_in_row: int, first_miss_step: int | None, last_ok: bool)`; property `fit -> float` (`hits / trials`, 0.0 with no trials).
  - `PredictionRecord(pid: str, pred: str | None, matched: bool, fit: float)`, frozen.
  - `class Belief`: `__init__(self, sandbox: ProgramSandbox)`; `programs: dict[str, ProgramState]`; `add(program: Program, hits: int = 0, trials: int = 0) -> str | None` (loads into the sandbox; returns the load error); `remove(pid: str) -> None`; `observe(step: Step) -> list[PredictionRecord]` (predicts the step's `before`/`action` with every live program, compares to `frame_hash(step.after)`, updates counts); `best(min_fit: float, min_trials: int) -> ProgramState | None` (highest fit, then most trials, then lowest id); `entropy() -> float`.
  - `entropy`: weights `w_i = fit_i` over live programs with `trials > 0`; `H = -sum p_i log p_i` with `p_i = w_i / sum w`; normalized by `log(n)`; 0.0 when n <= 1 or `sum w == 0`.
- The predictive loop never picks actions: nothing in this module returns an `Action`.

- [ ] **Step 1: Tests:** `test_observe_updates_fit_and_records` (two fake programs, one right, one wrong); `test_first_miss_and_misses_in_row_tracked`; `test_best_respects_thresholds_and_ties`; `test_entropy_normalized` (two equal fits gives 1.0; one program gives 0.0); `test_add_returns_load_error_and_skips_program`.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add the predictive loop`.

---

### Task 7: Wisdom function and salience

**Files:**
- Create: `arcagent/mind/wisdom.py`, `arcagent/mind/salience.py`
- Test: `tests/mind/test_wisdom.py`, `tests/mind/test_salience.py`

**Interfaces:**
- Produces in `wisdom.py` (pure functions; the caller logs every `drop` as `wisdom_drop`):
  - `admit_program(hits: int, trials: int, load_error: str | None, *, admit_fit: float = 0.8, hold_fit: float = 0.5) -> Gate`: drop on a load error or zero trials; pass at `fit >= admit_fit`; hold at `fit >= hold_fit`; drop below that.
  - `gate_action(action: Action, vetoes: frozenset[ActionKey]) -> Gate`: drop when vetoed, else pass.
  - `gate_plan_step(active: ProgramState | None) -> Gate`: drop when there's no active program or `not active.last_ok`; pass otherwise.
- Produces in `salience.py`:
  - `SalienceConfig(explore_steps: int = 20, min_new_steps: int = 5, base_threshold: float = 0.5, fatigue_raise: float = 0.4)`.
  - `score(step: Step, records: Sequence[PredictionRecord], belief_empty: bool, transfer_failed: bool) -> list[Event]`: `level_up` (magnitude 1.0) on a level rise; `game_over` (1.0); `prediction_error` with magnitude = the share of live programs that missed (only when there are records); `belief_empty` (1.0) when `belief_empty`; `transfer_failed` (1.0) when flagged.
  - `should_wake(events: Sequence[Event], *, steps_seen: int, steps_since_call: int, calls_made: int, drive: DriveState, cfg: SalienceConfig) -> str | None`: returns the reason, or None. Rules: never when `drive.synth_calls_left == 0`, and never when `steps_since_call < cfg.min_new_steps` unless it's the first call; `"new_game"` when `calls_made == 0` and `steps_seen >= cfg.explore_steps`; `"belief_empty"` or `"transfer_failed"` when such an event is present; `"prediction_error"` when an event's magnitude is at or above `cfg.base_threshold + cfg.fatigue_raise * fatigue(drive)`.

- [ ] **Step 1: Tests:** `test_admit_pass_hold_drop_bands` (8/10 pass, 6/10 hold, 4/10 drop, load error drop, 0/0 drop); `test_gate_action_and_plan_step`; `test_score_events_per_kind`; `test_first_wake_after_explore_steps`; `test_no_wake_without_calls_left_or_new_steps`; `test_fatigue_raises_prediction_error_threshold` (magnitude 0.7 wakes at fatigue 0, not at fatigue 1).
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add the wisdom function and salience`.

---

### Task 8: B0 and the executive

**Files:**
- Modify: `arcagent/agent/explore.py` (`ExplorerPolicy.choose` gains `exclude`)
- Create: `arcagent/mind/baseline.py`, `arcagent/mind/executive.py`
- Test: `tests/test_explore.py` (one new test), `tests/mind/test_executive.py`

**Interfaces:**
- Modifies: `ExplorerPolicy.choose(self, obs: Observation, exclude: frozenset[ActionKey] = frozenset()) -> Action`: excluded keys are skipped among untried actions and in the random fallback; if every candidate is excluded, it ignores the exclusion (the agent must act). Default behavior is unchanged.
- Produces in `baseline.py`: `class Baseline`: `__init__(self, seed: int)`; `choose(obs: Observation, exclude: frozenset[ActionKey]) -> Action`; `reset_priors() -> None` (new `ExplorerPolicy`, for null transfer at level-up). The explorer's UCB over action kinds' change rates is B0's "greedy use of learned action effects"; nothing is added on top.
- Produces in `executive.py`:
  - `ExecutiveInputs(obs: Observation, plan: list[Action], active: ProgramState | None, best: ProgramState | None, n_programs: int, vetoes: frozenset[ActionKey], plan_threshold: float)`.
  - `class Executive`: `__init__(self, *, plan_to_goal: Callable[[ProgramState], list[Action] | None], plan_to_novel: Callable[[ProgramState], list[Action] | None], voi_pick: Callable[[], Action | None], b0_pick: Callable[[frozenset[ActionKey]], Action])`; `decide(inp: ExecutiveInputs) -> tuple[Decision, list[Action]]` returns the decision and the plan that remains.
  - Order, per the spec: terminal or RESET-only obs -> `reset` (Action 0, plan cleared). Plan non-empty and `gate_plan_step(active)` passes -> `follow`. `best` at or above the threshold with `has_goal` -> `plan_to_goal`; a non-empty result -> `plan`. Otherwise `n_programs >= 2` -> `voi_pick`; `n_programs == 1` -> `plan_to_novel` -> `learn`. Otherwise -> `fallback` via `b0_pick(vetoes)`. Any chosen action failing `gate_action` drops that branch and the next one is tried; `fallback` passes the vetoes to B0, which ignores them only when nothing else is left.

- [ ] **Step 1: Tests:** `test_explorer_exclude_skips_and_falls_back_when_all_excluded`; `test_executive_order` (parametrized over the six branches with fake callables); `test_plan_dropped_when_active_program_missed`; `test_vetoed_plan_action_falls_through`; `test_terminal_obs_resets_and_clears_plan`.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add B0 and the executive`.

---

### Task 9: Module protocols, nulls, metacognition and inhibition

**Files:**
- Create: `arcagent/mind/modules/__init__.py`, `base.py`, `metacognition.py`, `inhibition.py`
- Test: `tests/mind/test_modules_mi.py`

**Interfaces:**
- Produces in `base.py`, the `Protocol`s and their null versions:
  - `Metacognition.review(belief: Belief, step_index: int) -> list[tuple[str, str]]` (pid, reason). `NullMetacognition` returns [].
  - `Inhibition.record(step: Step, records: Sequence[PredictionRecord], active_pid: str | None, wm: WorkingMemory) -> list[tuple[str, ActionKey, str]]` (frame hash, action, reason) and `Inhibition.vetoes(frame: str, wm: WorkingMemory, store: GameStore) -> frozenset[ActionKey]`. `NullInhibition` records nothing and vetoes nothing.
  - `Voi.pick(belief: Belief, sandbox: ProgramSandbox, obs: Observation, candidates: Sequence[Action], visited: frozenset[str]) -> tuple[Action | None, dict[ActionKey, float]]`. `NullVoi` returns `(None, {})`.
  - `Transfer.carry(store: GameStore, belief: Belief) -> list[str]` (pids carried) and `Transfer.check(belief: Belief, steps_on_level: int) -> bool` (True means transfer failed). `NullTransfer` carries nothing and never fails.
- Produces in `metacognition.py`: `ConfidenceMetacognition(min_trials: int = 3, drop_fit: float = 0.6, max_misses_in_row: int = 3)`: drops with reason `low_fit` or `misses_in_row`.
- Produces in `inhibition.py`: `DisconfirmInhibition(ttl: int = DEFAULT_TTL)`. It vetoes `(before frame, action)` when the step changed nothing (`noop`), ended in GAME_OVER (`game_over`), or the active program predicted it and missed (`disconfirmed`). Vetoes live in working memory under key `veto:<frame>` as a set, so they expire to cold and come back through `recall` when the frame is seen again.

- [ ] **Step 1: Tests:** `test_nulls_do_nothing`; `test_metacognition_drops_low_fit_after_min_trials`; `test_metacognition_drops_after_misses_in_row`; `test_inhibition_vetoes_noop_gameover_disconfirmed`; `test_veto_expires_then_recalls_on_reseen_frame`.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add the module protocols with metacognition and inhibition`.

---

### Task 10: The LLM synthesizer

**Files:**
- Create: `arcagent/mind/synthesizer.py`, `arcagent/mind/chat_store.py`
- Test: `tests/mind/test_synthesizer.py`

**Interfaces:**
- Consumes: `arcagent.synth.prompt.build_messages`, `extract_code`; `arcagent.synth.run.Chat`, `Completion`, `request_seed`, `http_chat`; `arcagent.synth.trace.Trace`, `Transition`; `ProgramSandbox.score`, `.load` (Task 3); `admit_program` (Task 7).
- Produces in `synthesizer.py`:
  - `GOAL_HINT`: one extra user paragraph that asks for `goal(grid)` as the best hypothesis of a completed level from the layout (targets, matching colors, an object on a marked cell), and, when evidence exists, shows the stored before and after grids of earlier level-ups with `render_grid`.
  - `SynthResult(programs: list[tuple[Program, Gate, int, int]], calls: int, seconds: float, prompt_tokens: int, completion_tokens: int, error: str | None)`.
  - `class Synthesizer`: `__init__(self, chat: Chat, *, k: int = 4, max_transitions: int = 60)`; `propose(game: str, call_no: int, steps: Sequence[Step], win_levels: int, goal_evidence: Sequence[tuple[Grid, Grid]], repair: tuple[Program, Sequence[Step]] | None, sandbox: ProgramSandbox, seed: int) -> SynthResult`. It sends K requests with seeds `request_seed(seed, call_no, i)`, extracts code, scores each candidate on all `steps` with `sandbox.score`, and gates it with `admit_program`. Program ids are `f"{game}-c{call_no}-k{i}"`; `has_goal` means the code defines `def goal(`. Any exception from `chat` is caught per request and its text goes to `error`.
  - The `Trace` built from `steps` puts every step in train (`TRAIN_FRACTION` doesn't apply here): build it with a subclass or a wrapper whose `train` returns all transitions.
- Produces in `chat_store.py`:
  - `recording_chat(inner: Chat, directory: Path) -> Chat`: saves each reply as `<sha1 of messages json + seed>.json`, with `{"game", "call", "k"}` metadata taken from a context variable the synthesizer sets.
  - `replay_chat(directory: Path) -> Chat`: answers from the hash match, then from the same `(game, call, k)`, then with the identity program (`arcagent.synth.verify.IDENTITY_CODE`).

- [ ] **Step 1: Tests:** `test_propose_admits_good_holds_partial_drops_broken` (a fake chat answering three canned programs); `test_server_error_yields_no_programs_and_error_text` (the chat raises `requests.ConnectionError`); `test_no_fence_reply_is_dropped`; `test_goal_hint_includes_evidence_grids`; `test_record_then_replay_round_trip`; `test_replay_falls_back_to_game_call_k_then_identity`.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add the LLM synthesizer and reply store`.

---

### Task 11: Sleep

**Files:**
- Create: `arcagent/mind/sleep.py`
- Test: `tests/mind/test_sleep.py`

**Interfaces:**
- Consumes: `WorkingMemory`, `GameStore`, `all_steps` (Task 5); `Belief` (Task 6); `ProgramSandbox.score`, `.goal` (Task 3); `admit_program` (Task 7).
- Produces:
  - `SleepReport(reason: str, steps_compacted: int, consolidated: list[str], replay: dict[str, tuple[int, int]], goal_confirmed: list[str], dropped: list[str])`.
  - `sleep(reason: Literal["level_up", "game_end"], wm: WorkingMemory, store: GameStore, belief: Belief, sandbox: ProgramSandbox) -> SleepReport`. In order: (1) compaction, `wm.clear_level()` appended to `store.episodes`; on `level_up`, the last step's `(before, after)` goes to `store.goal_evidence`. (2) Consolidation: every live program with `fit >= 0.8` goes to `store.procedural`. (3) Verification and replay: every procedural and held program is re-scored with `sandbox.score` over `all_steps`, and re-gated with `admit_program`; a pass stays or moves to `procedural`, a hold stays in `held`, a drop is removed and named in `dropped`. (4) Goal check: a program is `goal_confirmed` when `goal(after)` is True and `goal(before)` is False for the newest evidence pair.
- Sleep only does the bookkeeping; it doesn't choose what the next level's belief holds. Transfer does that.

- [ ] **Step 1: Tests:** `test_sleep_compacts_and_stores_goal_evidence`; `test_replay_rescores_and_promotes_held_program`; `test_replay_drops_program_that_fails_old_episodes`; `test_goal_confirmed_needs_true_after_and_false_before`.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add sleep`.

---

### Task 12: The mind agent

**Files:**
- Create: `arcagent/mind/agent.py`, `arcagent/mind/config.py`
- Test: `tests/mind/test_agent.py`

**Interfaces:**
- Consumes: everything from Tasks 2 to 11.
- Produces in `config.py`:
  - `MindConfig(name: str, metacognition: bool, inhibition: bool, voi: bool, transfer: bool, llm: bool, k: int = 4, max_calls_per_game: int = 12, plan_threshold: float = 0.9, plan_min_trials: int = 5, idle_act_s: float = 600.0, synth_wait_poll_s: float = 1.0)`, frozen.
  - `CONFIGS`: `"FULL"`; `"B0"` (all four modules off, `llm` and planner on); `"B0+metacognition"`, `"B0+inhibition"`, `"B0+voi"`, `"B0+transfer"`; `"FULL-metacognition"`, `"FULL-inhibition"`, `"FULL-voi"`, `"FULL-transfer"`; `"EXPLORER"` (everything off, no LLM: our 0.36 baseline, for reference only). That's the spec's ten ablation configs plus EXPLORER.
- Produces in `agent.py`:
  - `class ActivityClock`: thread-safe; `touch() -> None` on every action sent by any game; `idle_s(now: float) -> float`.
  - `class MindAgent` (implements `arcagent.agent.play.Policy`): `__init__(self, game: str, cfg: MindConfig, *, seed: int, synth: Synthesizer | None, pool: Executor, sandbox: ProgramSandbox, log: RunLog, activity: ActivityClock, game_deadline: float, run_deadline: float, clock: Callable[[], float] = time.monotonic)`; `choose(obs: Observation) -> Action`; `close() -> None` (runs `sleep("game_end")`, logs `game_end`, closes the sandbox).
  - Per `choose`: build the `Step` from the previous observation and action; `belief.observe`; `inhibition.record`; `wm.tick`; on level-up, run `sleep("level_up")`, then `transfer.carry` (or `baseline.reset_priors()` under null transfer), and clear the plan; on GAME_OVER, clear the plan; `metacognition.review` and remove the programs it names; `salience.score` and `should_wake`. On a wake with no call in flight, submit `synth.propose` to `pool` (repairing from `belief.best(0.0, 1)`). If a call is in flight and the belief is empty, wait for it in `synth_wait_poll_s` slices, unless `activity.idle_s(now) >= idle_act_s` or the game deadline has passed, in which case B0 acts. When a call finishes, gate its programs into the belief (pass) or into `store.held` (hold), and log `llm_call` plus each admit or drop. Then `executive.decide`, `activity.touch()`, log the action record (Task 4's fields), and return.
  - `plan_to_goal` and `plan_to_novel` call `sandbox.plan` with `candidate_actions(obs)` and `planner_bounds(drive)`, and log a `plan` event. `visited` is the set of frame hashes seen on this level.

- [ ] **Step 1: Tests** (a fake game: a 1-D corridor `Observation` generator, a fake chat returning the corridor's true rules with `goal()`, a real sandbox)
  - `test_explorer_config_never_calls_llm`.
  - `test_full_config_wakes_after_explore_steps_then_plans_to_goal` (the action log shows branch `plan` and the level completes).
  - `test_waits_for_synthesis_without_acting_until_idle_limit` (a fake clock: no action while idle is under 600 s; a B0 action with branch `wait` once it passes).
  - `test_game_over_mid_plan_resets_clears_plan_and_vetoes` (Review Focus 5).
  - `test_server_failure_keeps_playing_on_b0` (Review Focus 3, end to end).
  - `test_every_action_record_has_required_fields`.
  - `test_level_up_runs_sleep_and_transfer` (the log has `sleep` then `transfer` events).
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add the mind agent and ablation configs`.

---

### Task 13: Concurrent scheduler and the mind policy in submit

**Files:**
- Create: `arcagent/agent/scheduler.py`
- Modify: `arcagent/agent/submit.py`
- Test: `tests/test_scheduler.py`, `tests/test_agent.py` (new cases)

**Interfaces:**
- Consumes: `play_game` (`arcagent/agent/play.py`), `MindAgent`, `ActivityClock`, `CONFIGS`, `Synthesizer`, `recording_chat`, `replay_chat`, `http_chat`.
- Produces in `scheduler.py`:
  - `game_budget(now: float, run_deadline: float, games_left: int, workers: int, reserve_s: float = 300.0, min_s: float = 120.0) -> float`: `max(min_s, (run_deadline - reserve_s - now) * workers / max(games_left, 1))`, never past the run deadline.
  - `play_concurrent(arcade: Any, games: Sequence[str], *, workers: int, make_agent: Callable[[str, float], MindAgent], total_seconds: float, max_actions: int = 20_000, clock: Callable[[], float] = time.monotonic, report: Callable[[str], None] = print) -> tuple[list[GameResult], list[dict[str, Any]], dict[str, Any] | None]`: one scorecard, a thread pool of `workers` games, each game's deadline from `game_budget` when it starts, `agent.close()` in a `finally`, and per-game errors isolated as in `play_all`.
- Modifies `submit.py`: `POLICIES` gains `"mind"`; new flags `--config` (default `FULL`), `--endpoint` (default `http://127.0.0.1:1234/v1`), `--model` (default `qwen`), `--workers` (default 8), `--k`, `--log-dir` (default `<out>/logs`), `--replies-dir` (default `<out>/replies`), `--replay` (use `replay_chat` instead of HTTP). New `wait_for_model(endpoint: str, timeout_s: float = 3600.0, ...) -> float` polls `<endpoint>/models` the same way `wait_for_gateway` polls. With `--policy mind`, the model check passes before `arcade.open_scorecard`, so the scorecard opens only once actions can flow. `TAGS` becomes `["agent", "arcagent-mind"]` for the mind policy.

- [ ] **Step 1: Tests:** `test_game_budget_spreads_time_and_respects_floor`; `test_play_concurrent_isolates_a_crashing_game` (fake arcade); `test_wait_for_model_times_out_with_last_error`; `test_mind_policy_waits_for_model_before_scorecard` (fake arcade records the call order); `test_one_scorecard_accepts_concurrent_games` (uses the `env_dir` fixture: two real public games in two threads on one OFFLINE scorecard, 30 actions each; both appear in the closed scorecard). This last test is the spec's concurrency check; if it fails, set `--workers 1` as the default and record the finding in the experiment log before going on.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Run games concurrently under one scorecard with the mind policy`.

---

### Task 14: Offline loop

**Files:**
- Create: `scripts/offline_loop.py`
- Test: `tests/test_offline_loop.py`

**Interfaces:**
- Consumes: `submit.main` with `--policy mind --mode offline --replay`.
- Produces: `python scripts/offline_loop.py [--games ...] [--config FULL] [--replies DIR] [--game-seconds 120] [--out bench/offline/<stamp>]`. It runs the public games on CPU with replayed replies (the default `DIR` is the newest `bench/evals/*/replies`; with none, it runs on the identity fallback) and prints one line per game plus the `mind_metrics` summary from Task 15 once that exists. The script is a thin wrapper and adds no agent logic.

- [ ] **Step 1: Test:** `test_offline_loop_runs_two_games_end_to_end` (uses `env_dir`; `--games ls20 ft09 --game-seconds 20`; asserts `submission.parquet` exists and both logs have action records).
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add the offline loop`.

---

### Task 15: Metrics script

**Files:**
- Create: `scripts/mind_metrics.py`
- Test: `tests/test_mind_metrics.py`

**Interfaces:**
- Consumes: the JSONL logs (Task 4 schema); `data/public_human_actions.json`.
- Produces: `python scripts/mind_metrics.py LOG_DIR [LOG_DIR ...] [--json out.json]`, plus importable `compute(records: Iterable[dict[str, Any]]) -> dict[str, Any]`. Per game and overall:
  - `levels_completed`; `actions_per_completed_level`; `score_estimate` (public games only, with the competition formula: per level `min(h/a, 1)^2`, weighted by level index, over all levels).
  - `prediction_accuracy`: matched over all program predictions in action records.
  - `abandonment_latency`: mean of `drop_step - first_miss_step` over `program_dropped` events (the event carries both).
  - `disconfirmed_repeat_rate`: share of actions taken in a `(frame_hash, action)` pair whose earlier outcome was `noop`, `game_over` or `disconfirmed` in the same level. It's computed from the raw records, independent of whether inhibition was on.
  - `entropy_change_per_action`: mean of `entropy[t] - entropy[t+1]` over consecutive action records within a level.
  - `transfer_actions`: mean actions per completed level for level 1 and for levels 2 and up, reported separately.
  - Definitions live in this file's docstring; this is the one place they're fixed (the spec's call).

- [ ] **Step 1: Tests:** one hand-built log per metric with the answer worked out by hand (`test_score_estimate_matches_competition_formula`: a 3-level game with human counts [10, 20, 30], agent [10, 40] and level 3 unfinished, gives `100 * (1*1 + 2*0.25) / 6 = 25.0`); `test_repeat_rate_counts_only_after_a_bad_outcome`; `test_latency_from_drop_events`; `test_entropy_change_within_levels_only`.
- [ ] **Step 2–4:** Fail, implement, pass, gate.
- [ ] **Step 5:** Commit `Add the metrics script`.

---

### Task 16: Kaggle notebook and the first public-game eval

**Files:**
- Create: `scripts/kaggle_mind_notebook.py`
- Test: `tests/test_kaggle_mind_notebook.py`
- Modify: `docs/experiment-log.md`

**Interfaces:**
- Consumes: the serve cell and dataset constants from `scripts/kaggle_synth_notebook.py` (import `SERVE_CELL`, `WHEELHOUSE`, `MODEL_SNAPSHOT`, `SERVED_NAME`; don't copy them); the install pattern and `DOCKER_IMAGE` from `scripts/kaggle_agent_notebook.py`.
- Produces: `python scripts/kaggle_mind_notebook.py [--slug arc3-mind-agent] [--games all|<ids>] [--config FULL] [--workers 8]` writes `build/kaggle-mind/notebook.ipynb` and `kernel-metadata.json` (RTX PRO 6000, internet off, competition source, both driessmit1 datasets, private). Cells: install the arc-agi wheels and the vLLM wheelhouse; serve the model; write every file under `arcagent/` plus `data/`; run `python -m arcagent.agent.submit --mode auto --policy mind ...`; then run `scripts/mind_metrics.py` on the logs. In an interactive commit it plays `--games`; in the competition rerun, all games.

- [ ] **Step 1: Tests:** `test_notebook_builds_valid_json_with_all_mind_sources`; `test_metadata_has_gpu_no_internet_and_datasets`; `test_notebook_text_has_no_franzen_period_names`.
- [ ] **Step 2–4:** Fail, implement, pass, gate. Run the offline loop on all 25 games (`--game-seconds 60`) and confirm no game errors.
- [ ] **Step 5:** Commit `Add the Kaggle notebook for the mind agent`.
- [ ] **Step 6:** Push the eval kernel with `--games all` (`kaggle kernels push -p build/kaggle-mind`, CLI at `C:/Users/Sagac/arc-scratch-venv/Scripts/kaggle.exe`). When it finishes, fetch the output into `bench/evals/<date>-<config>/` (logs, replies, results.json). The replies become the offline loop's replay set.
- [ ] **Step 7:** Add the eval row to `docs/experiment-log.md`: levels completed, score estimate, the metrics summary, GPU hours. Commit `Log the first mind-agent eval`.

---

### Task 17: VOI

**Files:**
- Create: `arcagent/mind/modules/voi.py`
- Modify: `arcagent/mind/config.py` (wire `voi=True` to `DisagreementVoi`)
- Test: `tests/mind/test_voi.py`

**Interfaces:**
- Consumes: `Voi` protocol (Task 9), `ProgramSandbox.predict_many` (Task 3).
- Produces: `DisagreementVoi(max_candidates: int = 48)`. For each candidate (first `max_candidates` in `candidate_actions` order), it takes every live program's predicted hash and computes the fit-weighted entropy of that distribution (None predictions get their own bucket). It returns the highest-scoring action and the score map. Ties go to the earlier candidate. If every score is 0 it returns `(None, scores)` so the executive falls through to B0.

- [ ] **Step 1: Tests:** `test_picks_action_where_programs_disagree`; `test_returns_none_when_all_agree`; `test_none_predictions_form_their_own_bucket`; `test_respects_max_candidates`.
- [ ] **Step 2–4:** Fail, implement, pass, gate. Run the offline loop on the replay set; confirm the action logs show `learn` branches and non-empty `voi` maps.
- [ ] **Step 5:** Commit `Add the VOI module`.

---

### Task 18: Transfer and goal evidence

**Files:**
- Create: `arcagent/mind/modules/transfer.py`
- Modify: `arcagent/mind/config.py`, `arcagent/mind/agent.py` (pass `store.goal_evidence` into `propose`)
- Test: `tests/mind/test_transfer.py`

**Interfaces:**
- Consumes: `Transfer` protocol (Task 9), `GameStore` (Task 5), `Belief.add` (Task 6).
- Produces: `CarryTransfer(check_steps: int = 5, admit_fit: float = 0.8)`. `carry` loads every `store.procedural` and `store.held` program into the new level's belief with fit reset to 0/0 and `origin="transfer"`, and returns their ids. `check` returns True (failed) when `steps_on_level >= check_steps` and no carried program has `fit >= admit_fit`, which salience turns into a `transfer_failed` wake. Under transfer, B0's priors carry too (the same `Baseline` instance); under `NullTransfer`, the agent calls `reset_priors()`.

- [ ] **Step 1: Tests:** `test_carry_loads_procedural_and_held_with_reset_fit`; `test_check_fails_only_after_check_steps_without_a_fit`; `test_agent_passes_goal_evidence_to_synthesizer` (a fake synthesizer records its arguments).
- [ ] **Step 2–4:** Fail, implement, pass, gate. Run the offline loop and confirm `transfer` events appear after level-ups.
- [ ] **Step 5:** Commit `Add the transfer module and goal evidence`.

---

### Task 19: Daily eval and submission routine

**Files:**
- Modify: `docs/experiment-log.md`, `docs/agent-submission.md`

- [ ] **Step 1:** Document the routine in `docs/agent-submission.md`: on every change, run the offline loop; at most once a day, run one GPU eval kernel; submit on a day when the eval completes more levels than the last submitted version. The first submission goes in once the agent completes levels that EXPLORER can't (compare `levels_completed` per game in the two evals). Only this agent uses the daily slots, and both final selections on 2026-11-02 are this agent.
- [ ] **Step 2:** After the Task 18 eval, compare it with the Task 16 eval and with EXPLORER (0.36 offline). If the rule above is met, submit the kernel version from the competition page (one per day) and record the submission id, date and score in the experiment log.
- [ ] **Step 3:** Commit `Document the daily eval and submission routine`.
