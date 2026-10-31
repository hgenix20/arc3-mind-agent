# As-Built Spec: The Current Agent (the Hybrid)

Status: describes the code as of 2026-10-30. Base commit 5c4b4e0 on branch `llm-drive`,
plus the uncommitted changes listed in section 2. This section records what exists. It
does not propose changes; the refactor parts (1 to 9) and the review do that. Its purpose
is to make reuse decisions easy to check: section 10 lists each component and whether the
refactor should keep, change, or replace it.

## 1. What "the current agent" means

The most recent agent is the **hybrid**: the fluid mind plays each level first, and the
graph explorer takes over after a limit on mind moves or a stall. The hybrid makes no
model call. It is the candidate for the prize-eligible entry in the 2026-10-29 plan. The
Host question on the eligibility of model-based entries is still open.

Two other entries exist and are not part of the hybrid:

- **FLUID-LLM** (`configs` in `arcagent/mind/config.py`). The fluid agent plus a vision LLM
  that proposes goals, plans, and tests. Submitted 2026-10-24 (see
  `docs/agent-submission.md`).
- **DRIVE** (`llm_drive.py`). The LLM drives the game through toolbox and tool-fix turns.
  Baseline submission 57746426 used DRIVE v11 with Qwen3.6-27B.

Config names come from `_all_configs()`: `FLUID` (model-free, the mind alone), `FLUID-LLM`,
`FLUID-LLM-<module>` (ablations with one of the four paper modules nulled), and `DRIVE`.
The hybrid is not a config. It is built by `MindRun.make_agent` when `hybrid_moves > 0`
and `cfg.fluid` is set (`arcagent/agent/submit.py`).

## 2. Uncommitted state

These changes are in the working tree and are not in any commit. The refactor starts from
the committed code plus these.

| File | Change | Benched? |
|---|---|---|
| `arcagent/agent/hybrid.py` | Stall rule: `STALL_MOVES = 25` fallback moves in a row hand the level to the explorer. The `shared` flag, so the explorer that is the mind's own (`FluidAgent.explorer`) is not fed the mind's moves twice. | No. The 2026-10-29 log says "not yet benched." |
| `arcagent/agent/submit.py` | `HybridPolicy(fluid, fluid.explorer, n, shared=True)` in `make_agent`. | No. |
| `arcagent/mind/fluid.py` | `resume()`, `explorer` property, `last_branch`. | No. |
| `arcagent/mind/baseline.py` | `policy` property. | No. |
| `tests/test_hybrid.py`, `tests/test_kaggle_mind_notebook.py` | Stall and shared tests; notebook option test. | Tests only. |

The committed hybrid (`3fbefd2`, then `2b48c31` for the veto fix) has no stall rule and no
shared flag. Its `HybridPolicy` feeds the explorer the mind's moves itself.

## 3. Runtime contract

- **Game interface.** Each move gets an `Observation` (`arcagent/core/types.py`): a 64x64
  `int8` grid with values 0 to 15 and padding -1 (`grid_from_frame`), a phase (not played,
  in play, won, game over), `levels_completed`, `win_levels`, and `available_actions`.
  Actions are integer ids. Action 0 is RESET. Action 6 is a click at (x, y). The others are
  simple actions whose effect is not named in the observation.
- **Scoring.** Per level, `min(human_actions / agent_actions, 1)^2` (`docs/agent-submission.md`).
  The per-game score is the mean over levels weighted by level index, and the total is the
  mean over games. The experiment log reports offline results as scorecard means. Public
  leaderboard figures appear in the log in a different unit, and the log does not say how
  the two relate (section 12).
- **Budget.** The entry plays 110 private games in a Kaggle rerun with 9 hours of runtime,
  no internet, and the games' own move counters where the game has one (`budget.py`).
- **Seeds.** A game's seed is the run seed plus its index in the game list (`MindRun`).

## 4. Components and what each one does

Module, then its job, then its entry point. Entry points are the functions the agent calls
each move.

**Top level**
- `arcagent/agent/submit.py`. Builds the arcade, selects games, builds one policy per game
  (`MindRun.make_agent`), writes the submission parquet. Entry: `play_mind`, `main`.
- `arcagent/agent/hybrid.py`. `HybridPolicy.choose(obs)`. Hands each move to the mind or
  the explorer by the mind-move limit and the stall rule.

**Mind (fluid path)**
- `arcagent/mind/fluid.py`. `FluidAgent.choose(obs)`: the seven-step decision order in its
  docstring. Holds the plan, the goal book, the fact memory, the tracker, and the search.
- `arcagent/mind/fluid_modules.py`. The four paper modules as protocols with nulls: Voi,
  Inhibition, Metacognition (fact break threshold), Transfer.
- `arcagent/mind/objects.py`, `screen.py`, `regions.py`. Things, scenes, diffs, rendering.
- `arcagent/mind/kinds.py`. Identity across looks (`KindBook`).
- `arcagent/mind/noise.py`. Masks for counters and noise cells.
- `arcagent/mind/modes.py`. Selection marks and the active controller (`ModeTracker`).
- `arcagent/mind/facts.py`. `FactMemory`: facts keyed by (label, subject, context), effects,
  touch and near contexts, blockers, walkable colors, toc by behavior tag, sibling index.
- `arcagent/mind/predictor.py`. `predict(scene, action, mem, ...)`, read-only, with
  `assume_still`. `check` compares a prediction with the observed diff.
- `arcagent/mind/kinematics.py`. Gaits from facts, reachable tops, distance fields.
- `arcagent/mind/goals.py`. `GoalBook`: guesses (reach, clear, count, cover, match,
  match_thing, sequence), `ranked`, `level_up`, `learn_win`, `generalize_forms`.
- `arcagent/mind/search.py`. Best-first search over predicted scenes toward goal distances,
  with node, time, and stable-slice bounds.
- `arcagent/mind/route.py`. Visiting order over targets (Held-Karp up to 6, then 2-opt).
- `arcagent/mind/tester.py`. Fact tests (`choose_test`) and vetoes (`veto`).
- `arcagent/mind/progress.py`. Milestone, progress, regress, motion, settled.
- `arcagent/mind/indicator.py`. Progress indicators (rows of boxes that light).
- `arcagent/mind/budget.py`. Moves left, read from the counter bar.
- `arcagent/mind/fluid_sleep.py`. Fact bookkeeping at level-up and game end.
- `arcagent/mind/baseline.py`. The B0 baseline explorer, shared with the hybrid.

**Explorer**
- `arcagent/agent/explore.py`. `ExplorerPolicy`: state graph keyed by `StateKey`, untried
  actions, idle kinds, dead moves, live path, rounds, frontier path. Constants: `MAX_STATES =
  20000`, `IDLE_TRIES = 20`, `IDLE_RATE = 0.05`, `DEAD_AFTER = 2`.
- `arcagent/agent/guided.py`. `GuidedExplorer`: the explorer with win credit on colour gaps
  and counts, and mover-based reach features.

**Not on the hybrid path**
- `arcagent/mind/agent.py`, `memory.py`, `sleep.py`, `wisdom.py`, `sandbox.py`,
  `synthesizer.py`, `executive.py`, `assign.py`, `contract.py`, `game_model.py`,
  `toolbox.py`, `tool_fix.py`, `vchat.py`, `turn.py`, `llm_drive.py`, `drive.py`: the
  program path and the LLM paths.

## 5. Per-move decision order (the fluid mind)

From the module docstring of `fluid.py`. Branch names are what the run log records.

1. **Game over or not played.** Learn the death from the move that caused it, reset the
   tracker and noise, clear the plan, return RESET (branch `reset`).
2. **Level up.** Record the level-up fact, call `goals.level_up` and `learn_win` (when
   nothing was confirmed), run `fluid_sleep`, reset positions in `FactMemory`, run the
   transfer step, clear the plan and the level's history. The new screen teaches nothing.
3. **Update.** Update tracker, kinds, mode, noise. Learn from the transition, check the last
   prediction. A goal whose plan was being followed and still open is set probing, and a
   commit probe starts (branch `commit`, event `commit_probe`, at most `COMMIT_PROBES = 4`).
4. **Plan.** A plan stops only on a miss that matters (`plan_breaks`). A plan with steps
   left gives the next step (branch `follow`).
5. **Forced and bump.** The only move no plan veto refuses (branch `forced`). A step whose
   goal already holds gets the learned bump move (branch `plan`).
6. **Budget, indicator, test.** A budget-tight partial plan step; an indicator step (branch
   `plan`, why `sequence step` or `sequence try`); the tester's test (branch `learn`), at
   most `MAX_TESTS_PER_ATTEMPT = 200` per level attempt.
7. **Partial plan, then novelty.** The goal search's best partial plan (branch `plan`, why
   `toward <goal id>`); else novelty search (why `explore novel`).

Goal search runs with `SEARCH_MAX_NODES = 3000` predictions and `SEARCH_TIME_S = 1.0` seconds.
A goal fails an attempt when its plan breaks without the goal holding. After `STALL_AFTER =
2` failed attempts in one level, the goal is set aside until the level changes.

## 6. The hybrid's switching rule

- Each level starts with the mind (`HybridPolicy.choose`).
- The mind plays until it has made `mind_moves` moves on the level (the committed hybrid),
  or, with the uncommitted stall rule, until `STALL_MOVES = 25` of its moves in a row were
  fallback moves (branch `fallback`).
- Then the explorer plays the rest of that level.
- A level change gives the mind the next level. If the explorer played last, the mind is
  resumed (`FluidAgent.resume`) so it does not learn from moves it did not make.
- `mind_moves` is 150 in the 2026-10-28 and 2026-10-29 benches and was 400 in the first
  hybrid run. `submit.py` defaults `--hybrid-moves` to 0, which plays the mind alone; the
  notebook passes the value it is given.

## 7. Learning in the fluid mind

- **Facts.** Each move's transition is learned (`facts.learn`). A fact is solid after
  `THIN_BELOW = 5` agreeing observations and broken after `BROKEN_MISSES = 2` misses in a
  row. A miss does not rewrite a fact's effects.
- **Goals.** `GoalBook.level_up` confirms goals at a level-up. Probe-excluded goals are
  credited (`probed`).
- **Learning from a win.** `learn_win` credits the gaps the winning move closed
  (`WIN_GAP = 2`, `WIN_LEARNED = 2`, `WIN_FRAMES = 80`) when no goal was confirmed.
- **Generalising forms.** `generalize_forms` extends reach and cover forms to colours
  (`cN:*`).
- **Sleep.** `fluid_sleep` runs at each level-up and at game end. It demotes facts a winning
  level relied on after misses, logs merge candidates, and drops thin zero-hit facts. Time
  cap `SLEEP_MAX_S = 0.5` per call.
- **Transfer.** `CarryTransfer` keeps facts across levels; the null forgets them.

## 8. Data structures

- `Thing`: tracked id, sig (colour and shape), kind, cells, bounding box.
- `Scene`: things, terrain, background. `SceneDiff`: changes between two scenes.
- `Fact`: key (label, subject, context), effects, hits, misses, status, level of creation.
- `Goal`: kind (`reach`, `clear`, `count`, `cover`, `match`, `match_thing`, `sequence`),
  arguments, status, distance.
- `PlanStep`, `SearchResult` (goal, plan, frontier, depth, best partial plan, stop reason).
- `Observation`, `Action`, `Phase`, `Grid` (`arcagent/core/types.py`).

## 9. Constants that set behaviour

Values as found in code on 2026-10-30. The refactor should keep these numbers as
starting values for the logged runs.

| Constant | Value | Where |
|---|---|---|
| `STALL_MOVES` | 25 (uncommitted) | `agent/hybrid.py` |
| `SEARCH_MAX_NODES` | 3000 | `mind/search.py` |
| `SEARCH_TIME_S` | 1.0 | `mind/search.py` |
| `STALL_AFTER` | 2 | `mind/fluid.py` |
| `MAX_TESTS_PER_ATTEMPT` | 200 | `mind/fluid.py` |
| `COMMIT_PROBES` | 4 | `mind/fluid.py` |
| `TEST_RESERVE` | 2 | `mind/fluid.py` |
| `SEQ_TRIES` | 4 | `mind/fluid.py` |
| `TURN_DEADLINE_S` | 30.0 | `mind/fluid.py` |
| `TURN_MAX_TOKENS` | 1536 | `mind/fluid.py` (LLM variants) |
| `THIN_BELOW` | 5 | `mind/facts.py` |
| `BROKEN_MISSES` | 2 | `mind/facts.py` |
| `KIN_LIMIT` | 4096 | `mind/kinematics.py` |
| `KIN_FIELDS` | 8 | `mind/kinematics.py` |
| `MAX_REACH_TARGETS` | 20 | `mind/goals.py` |
| `MAX_OPEN_GOALS` | 200 | `mind/goals.py` |
| `UNKNOWN_COST` | 20 | `mind/goals.py` |
| `SALIENCE` | 2.0 | `mind/goals.py` |
| `WIN_GAP` / `WIN_LEARNED` / `WIN_FRAMES` | 2 / 2 / 80 | `mind/goals.py` |
| `MOTION_WINDOW` | 12 | `mind/progress.py` |
| `REGRESS_SLACK` | 2 | `mind/progress.py` |
| `SLOT_MIN` / `MAX_ROWS` | 3 / 8 | `mind/indicator.py` |
| `BUDGET_DEATH_MOVES` / `MIN_DRAIN` / `MIN_FIT` | 2 / 6 / 5 | `mind/budget.py` |
| `MAX_STATES` | 20000 | `agent/explore.py` |
| `IDLE_TRIES` / `IDLE_RATE` / `DEAD_AFTER` | 20 / 0.05 / 2 | `agent/explore.py` |
| `SLEEP_MAX_S` | 0.5 | `mind/fluid_sleep.py` |

## 10. Reuse map for the refactor

Keep means the component is reused with no change in its contract. Change means its
contract or its use changes. Replace means a new component takes its place. The refactor
parts name the target for each.

| Component | Call | Why, and which part |
|---|---|---|
| `objects.py`, `screen.py`, `regions.py` | Keep | Perception works; no part asks to change it. |
| `kinds.py` | Keep | Identity across looks; used by Part 9's "seen before." |
| `noise.py`, `budget.py` | Keep | Masks and move budget; Part 7 reads the budget. |
| `modes.py` | Change | Active controller is the base for Part 3's self record. |
| `facts.py` | Change | Add scope, canonical key, probability (Parts 1, 2, 8). Lookup order changes (Part 2). |
| `predictor.py` | Keep | `predict` is read-only, as Part 8 asks. Add probability, not behaviour. |
| `kinematics.py` | Change | Shared BFS and a truncation flag (Part 4). `gaits` reads an index, not the table (Part 2). |
| `goals.py` | Change | `ranked` formula becomes a path probability (Part 8). Goal records (Part 1). |
| `search.py` | Change | Probability bound, plan tree, segments (Parts 5, 8). Heuristic admissibility is open (Part 4). |
| `route.py` | Keep | Visiting order; not in the source list (review, Q4 of Part 4). |
| `tester.py` | Change | Veto takes reasons and adverse rules (Parts 6, 7). |
| `progress.py`, `indicator.py` | Keep | Signals for Part 3's controller. |
| `fluid.py` | Change | Decision ladder is the owner of handoffs (Parts 3, 7, 9). Plan tree (Part 5). |
| `fluid_modules.py` | Change | The four paper modules stay, and become the ablation points for the refactor. |
| `fluid_sleep.py` | Change | Promotion to long-term memory (Part 1). |
| `hybrid.py` | Change | Switching rule becomes the controller (Part 3). Uncommitted stall rule must be benched first. |
| `explore.py`, `guided.py` | Keep | Fallback explorer; Part 5's backtracking question (Q4) is open. |
| `baseline.py` | Keep | B0 explorer used by the hybrid. |
| `agent.py`, `memory.py`, `sleep.py`, `wisdom.py`, `sandbox.py`, `synthesizer.py` | Keep, off path | Program path. Part 6 keeps it as a separate mode. |
| LLM paths (`turn.py`, `llm_drive.py`, `toolbox.py`, `tool_fix.py`, `vchat.py`) | Keep, off path | Separate mode; Part 6 Q4 decides whether it stays beside the model-free agent. |

## 11. Tests on record

- `tests/test_hybrid.py`: takeover and handback, learning from mind moves, the stall rule
  (uncommitted), the shared explorer (uncommitted).
- `tests/test_guided.py`: guided explorer on the avatar fixture.
- `tests/mind/test_goals.py`: learning from wins, generalised forms.
- `tests/test_kaggle_mind_notebook.py`: notebook option handling.
- Explorer tests: idle moves wait for a live move; a veto never forces a known no-op.

The last type check recorded in this session reported 0 errors. No test count is recorded
here, and this section does not claim the suite passes on the uncommitted tree.

## 12. Measured results and what they cover

From `docs/experiment-log.md`. Kaggle CPU notebooks, 4 seeds, 10k moves unless stated.

| Run | Mean levels | Scorecard mean | Note |
|---|---|---|---|
| explore, idle fix (e2a542d), 10k | 25.5 | 0.468 | |
| guided, e2a542d, 10k | 24.25 | 0.477 | |
| explore, 100k, seed 0 | 37 | 0.662 | single seed |
| hybrid, 400 mind moves (3fbefd2), 10k | 26.5 | 0.453 | before idle fix |
| hybrid, 150 mind moves, veto fix (2b48c31) | 26.75 | 0.473 | 5 of 30 wins learned from |
| + learn_win (eb350a1) | 27.25 | 0.474 | 25 of 30 wins learned from |
| + colour forms (4a64461) | 27.0 | 0.473 | no change in levels |
| hybrid, 400 mind moves, later | 26.75 | 0.556 | one level (tu93 L1) drives the mean; not adopted |
| DRIVE v11, offline 25 games | 11 levels | 0.577 | public score 0.39 |

Public scores on record: duck v2 at 27.22 (leaderboard, 2026-10-22), DRIVE v11 at 0.39
(submission 57746426). The log writes these in different units, and it does not state
whether the leaderboard and the offline scorecard use the same scale. That must be
reconciled before any comparison is quoted (section 13, item 1).

The stall rule and the shared explorer have no benchmark yet.

## 13. Known limits of the current agent

These match the refactor parts. Each is a fact about the code, not a proposal.

1. **Unit mismatch in results.** Public and offline numbers are in different units in the
   log (section 12). Not reconciled.
2. **Uncommitted and unbenched.** The working tree changes `hybrid.py`, `submit.py`,
   `fluid.py`, `baseline.py`, and two test files (section 2). None of it has a benchmark.
   The benched hybrids were built from commits, so their behaviour is not the working tree's.
3. **No long-term memory.** Modules are fresh per game (`config.build_modules`). Facts
   persist within a game only (Part 1).
4. **Retrieval scans.** `kinematics.gaits` and several `goals.py` helpers read the whole fact
   table per call (Part 2).
5. **Lookup order is fixed.** `FactMemory.lookup` returns the first match, not the most
   probable one (Part 2).
6. **Silent truncation.** `kinematics.reachable` stops at `KIN_LIMIT` with no flag (Part 4).
7. **No plan tree.** A broken plan is replaced by a fresh search from the current scene.
   Failed branches are not recorded (Part 5).
8. **Binary vetoes.** A veto holds or it does not. No probability and no override reason.
   The forced move never overrides a predicted game over (Parts 6, 7).
9. **No self record.** "I" and "my character" are not represented (Part 3).
10. **Controller is implicit.** Handoffs happen in `fluid.py` and `hybrid.py` by counts.
    Nothing records why the handoff happened beyond the branch label (Part 3).
11. **Hypothesis ranking is a formula.** `GoalBook.ranked` is not a probability (Part 8).
12. **Enemy motion.** An uncontrolled mover with no fact is predicted still, and the miss is
    not recorded as an assumption failure (Part 8).
13. **Token totals not tracked.** Per-reply caps exist; no per-task total (Part 7).
14. **Level 1 only in the survey.** The 2026-10-29 frame survey covered the first frame of
    each game. Later levels are unchecked.
15. **Gait model covers fixed steps and slides.** `kinematics.gaits` reads `move` and `slide`
    effects only. Teleports and other effects are in the fact model but not in the walk.
