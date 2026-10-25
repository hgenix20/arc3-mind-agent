# Progress, identity, mode, goal kinds and budget: task list

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Kameron's 2026-10-24 direction on the seven open problems into buildable tasks, after one clean A/B of the two fixes already on `counter-and-noop`.

**Architecture:** Everything stays in the fluid agent's deterministic loop (perception, facts, predictor, goals, search, tester), numpy and the standard library only, with no LLM in the per-move path. New pieces are small pure modules (`assign`, `kinds`, `modes`, `kinematics`, `progress`, `regions`, `budget`, `route`) wired into `fluid.py` one section at a time, each section gated by a paired offline A/B.

**Tech Stack:** Python 3.12, numpy, pytest, ruff, pyright strict. Offline games through `scripts/offline_loop.py`, scored by `scripts/mind_metrics.py`.

**Spec:** `docs/superpowers/specs/2026-10-23-fluid-agent-design.md` (approved 2026-10-23), amended by Kameron's direction of 2026-10-24 (pasted in chat; summarized at the top of each section below as "Direction"). The fact-model and goal-model changes in sections 3 and 5 are his decisions, given in that direction.

## Global Constraints

- No genre priors: no known game is named in code, prompts, comments or test names. Root mechanics only, each a hypothesis that must pass a test.
- No memorized solutions: nothing here may encode what a specific public game needs.
- Dependencies: numpy and the standard library. No scipy (it isn't in the venv), no torch, no trained models.
- Pure, deterministic modules; every loop and search is bounded by a named constant.
- No LLM call in the per-move path. The executive turn stays as it is today (woken by salience), and nothing new depends on it.
- Gates per task: `ruff format`, `ruff check`, `pyright` (strict), `pytest tests/mind -q`, all clean.
- Commits end with `Claude-Session: https://claude.ai/code/session_01ED1JLjSNaMyvkVda5AScAg`. No Co-Authored-By line (CLAUDE.md overrides the harness).
- Prose: no em-dashes, contractions, none of the flagged words.
- Merge to main only when a paired offline A/B (two runs per arm) gives more levels or a higher score. One run proves nothing: the same code varies 4 to 7 points between runs.
- Offline A/B command: `scripts/offline_loop.py --config FLUID --game-seconds 240 --workers 2 --out <dir>`. Two workers: four ran the laptop out of memory on 2026-10-24.
- Every A/B result gets a row in `docs/experiment-log.md` in the same commit as the decision.
- Kaggle submissions: one per day, only on Kameron's go.

## What the research changed

Each section below says which parts of the direction carry over as written, which carry over in adapted form, and why. Three findings shaped most of it.

1. **The scorecard charges every action, RESET included.** `arc_agi/scorecard.py` (`inc_reset_count` adds to `actions`), and a level's cost is all actions since the previous level-up, lost lives included. A RESET mid-life is a level reset in `arcengine/base_game.py` (`handle_reset`: a full reset only when no action was taken yet). So a life that can't finish should end with one RESET, not with its remaining moves. This is the real form of "early termination" here (section 6).
2. **The board gives exact objects already.** Frames are 64×64 with 16 colors; components are exact, with no detector noise. Learned embeddings (CNN, slot attention, GNN, LSTM, transformers) would need training data and a GPU per game, and the spec rules out both the GPU outside granted windows and memorized solutions. Their ideas carry over as explicit structure: a motion prior, bipartite matching, decoupled attributes and relational features. That's the same split SORT made: a motion model plus Hungarian assignment, nothing learned (Bewley et al., 2016, arXiv 1602.00763).
3. **A training-free graph explorer ranked 3rd in the preview.** Rudakov, Shock and Cowley (arXiv 2512.24156) masked probable status bars before hashing states, ranked click targets into five salience tiers, and moved toward the nearest untested state-action pair. Their re-run bug is ours too: actions that triggered a reset weren't marked tested, so the agent looped. Their stated limit: cost grows with the state space, so exhaustive search fails on later levels. That's why sections 2 and 4 add a progress heuristic instead of more exploration.

## Execution order

Sections are listed in the order Kameron gave them. Build them in this order instead, because later ones read what earlier ones produce:

0 (clean A/B) → 3 (identity) → 4 (mode state) → 2 (progress) → 6, Task 6.1 (read the budget) → 1 (stop and submit) → 5 (goal kinds) → 6, Tasks 6.2 to 6.4.

Nine days remain to the final selection on 2026-11-02. Not all of this fits. If time runs short, the order above is also the order of expected gain: identity (tu93, wa30 lose their player every turn), progress and best-first search (lf52's 8-click rebuild), the doomed-life RESET (every game with a budget bar), then the submit probe (tn36).

## Review Focus

1. A recolor or reshape of a thing that stays put and a different thing appearing on the same spot must not merge into one kind (Task 3.3, `test_new_thing_on_vacated_spot_is_not_kin`).
2. A RESET must never fire on a life that can still finish, and never as the first action of a game, where the engine does a full reset (Task 6.3, `test_no_reset_when_cheapest_plan_fits` and `test_no_reset_before_first_action`).
3. The selection detector must not mark a thing selected because a counter or noise strip changed after the click (Task 4.1, `test_masked_change_is_not_a_selection_marker`).
4. Best-first search must still return the shortest plan when no goal has a distance (Task 2.4, `test_without_distance_search_is_breadth_first`).
5. Budget reading must say "unknown" for games without a counter instead of guessing (Task 6.1, `test_no_counter_means_unknown`).

---

## Section 0: Clean A/B of the committed fixes

### Task 0.1: Keep counter cells across lives

The branch has `NoiseMask.new_attempt()` documented and tested, but `FluidAgent._game_over` still calls `_restart()`, which calls `reset_tracks()`, so counter cells are dropped on every game over.

**Files:**
- Modify: `arcagent/mind/fluid.py` (`_game_over`, `_restart`)
- Test: `tests/mind/test_fluid_agent.py`

**Interfaces:**
- Consumes: `NoiseMask.new_attempt() -> None`, `NoiseMask.cells -> frozenset[Cell]` (exists).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test** `test_counter_cells_survive_game_over`: drive a `FakeGame` whose bottom edge holds a 1-cell-thick bar that shrinks one cell per move until the agent's noise mask flags it (3 moves), then send a `GAME_OVER` observation and the RESET frame. Assert `agent._noise.cells` still contains the bar's bbox cells after the RESET, and that `_level_up` still clears them (a second assertion after a level-up observation).
- [ ] **Step 2: Run** `pytest tests/mind/test_fluid_agent.py::test_counter_cells_survive_game_over -v`. Expected: FAIL (cells empty after the RESET).
- [ ] **Step 3: Implement.** `_game_over` calls `self._noise.new_attempt()` in place of `reset_tracks()`; the level-up path keeps `reset_tracks()`. Update the `fluid.py` docstring step 1 to say so.
- [ ] **Step 4: Run the gates** (ruff, pyright, `pytest tests/mind -q`). Expected: all pass.
- [ ] **Step 5: Commit** "Keep counter cells across lives".

### Task 0.2: Paired offline A/B, main against the branch

**Files:**
- Modify: `docs/experiment-log.md`

- [ ] **Step 1: Check free memory** before each run: `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory`. Start a run only above 6 GB free; otherwise stop and tell Kameron.
- [ ] **Step 2: Make a worktree for main:** `git worktree add ../arc-agent-main c2a7001`.
- [ ] **Step 3: Run four runs one after another** (about 50 minutes each with two workers), outputs in the scratchpad: `ab2-main-1`, `ab2-branch-1`, `ab2-main-2`, `ab2-branch-2`. Alternate arms so a slow machine period doesn't land on one arm.
- [ ] **Step 4: Score each run** with `scripts/mind_metrics.py` and build a per-game table: levels and actions per level for all four runs.
- [ ] **Step 5: Decide.** Merge `counter-and-noop` to main only if the branch has more levels summed over its two runs, or a higher mean score. Otherwise keep the branch and record why.
- [ ] **Step 6: Commit** the experiment-log row (runs, per-game table summary, decision).

**Acceptance:** four complete runs (25 games each, every game with a `game_end`), a logged decision.

---

## Section 1: Knowing when to stop and submit

**Direction:** time management paired with search termination: a per-move time budget from the clock, iterative deepening that submits the best move from the last finished depth, an early stop when the best move stays the same and the score settles, an instant move when only one legal move exists, and quiescence: never stop in a chaotic position.

**What carries over:**
- **Time budget per move:** carries over, mapped to our two clocks. The wall clock is the per-game time from the scheduler; the move clock is the life's remaining budget (Task 6.1). Search caps scale with both.
- **Iterative deepening with "best from the last finished iteration":** carries over as an anytime best-first search. Our breadth-first search already finishes depth by depth; Task 2.4 makes it best-first, and this task makes it return its best partial plan when a cap stops it.
- **Early stop on a stable best move:** carries over: run the search in growing node slices and stop when the first action and the best distance are unchanged across two slices.
- **Forced move:** carries over directly.
- **Quiescence:** carries over as "don't accept a plan ending in an unstable predicted state", with unstable defined for this world (below).
- **Opening book:** has no analog. Nothing is memorized, by rule.
- **The "submit" part:** in most games the level ends by itself once the win state holds. tn36 showed the other case: the board was right at move 8 and the level needed one more action. Task 1.4 adds a bounded commit probe.

### Task 1.1: Take a forced move without searching

**Files:**
- Modify: `arcagent/mind/fluid.py` (`_decide`)
- Test: `tests/mind/test_fluid_agent.py`

**Interfaces:**
- Consumes: `FluidAgent._veto(scene, action) -> str | None`, `FluidAgent._actions(scene) -> list[Action]`.
- Produces: branch name `forced` in the action log.

- [ ] **Step 1: Failing test** `test_single_unvetoed_action_is_taken_without_search`: a scene where every action but one is vetoed `SEEN_NO_EFFECT`; patch `search` to raise if called. Assert the chosen action is the remaining one and `branch == "forced"`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** in `_decide`, after following a held plan and before `_plan_choice`: when exactly one action survives `_veto`, return it with branch `forced`. With no plan held and zero survivors, fall through to the existing fallback.
- [ ] **Step 4: Gates.** **Step 5: Commit** "Take a forced move without searching".

### Task 1.2: Anytime search with a stable-best stop

**Files:**
- Modify: `arcagent/mind/search.py`
- Test: `tests/mind/test_search.py`

**Interfaces:**
- Consumes: `Goal.distance` (Task 2.1), best-first frontier (Task 2.4).
- Produces: `SearchResult.best: tuple[PlanStep, ...]` (the plan to the expanded scene with the lowest goal distance, empty when none improves on the root) and `SearchResult.best_distance: int | None`; new `Stop` value `"stable"`; constants `SEARCH_SLICES = (500, 1000, 2000, 3000)`, `STABLE_SLICES = 2`.

- [ ] **Step 1: Failing tests.**
  - `test_cap_returns_best_partial_plan`: a reach goal 12 steps away with `max_nodes=40`. Assert `result.stop == "nodes"`, `result.best` is non-empty, and the mover's predicted distance after `result.best` is below the root distance.
  - `test_stable_best_stops_early`: an open board where the best first step can't change after the first slice. Assert `result.stop == "stable"` and `result.nodes < 3000`.
  - `test_no_progress_means_empty_best`: no expanded scene lowers the distance. Assert `result.best == ()`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Keep the best node by `(distance, len(plan))`. At each slice boundary in `SEARCH_SLICES`, record `(best[0].action.key(), best_distance)`; stop with `"stable"` when it's unchanged for `STABLE_SLICES` boundaries in a row and the plan isn't found yet. `max_nodes` still caps everything.
- [ ] **Step 4: Gates.** **Step 5: Commit** "Search returns its best partial plan and stops when the best is stable".

### Task 1.3: Quiescence: don't end a plan in a trap

**As built (changed from the first draft):** the draft defined "unstable" as (a) next to a death color, (b) mid-slide, or (c) an unknown pair one move away. (b) can't happen: the predictor resolves each slide to where it stops. (a) isn't dangerous by itself: the move back is usually safe and vetoes refuse deadly moves. (c) is uncertainty, not danger. What quiescence guards against here is a leaf where every next move is predicted to end the game: a trap.

**Files:**
- Modify: `arcagent/mind/search.py` (the loop moved into a `_Run` class to stay inside the lint limits; behavior otherwise unchanged)
- Test: `tests/mind/test_search.py`

**Interfaces:**
- Produces: `is_trap(scene: Scene, actions: Callable[[Scene], list[Action]], mem: FactMemory, protect: frozenset[str]) -> bool`; `QUIESCENCE_PLIES = 2`. A found plan ending in a trap is held while the search looks for a safe plan at most `QUIESCENCE_PLIES` moves longer; the held plan is returned when none turns up or a bound stops the search. The best partial plan never ends in a trap. Applies to `goal_test` finds (novelty) too.

- [x] Tests: `test_plan_ending_in_a_trap_yields_to_a_safe_plan_a_little_longer`, `test_plan_ending_in_a_trap_is_returned_when_no_safe_plan_exists`, `test_safe_plan_too_much_longer_loses_to_the_trap_plan`, `test_best_partial_plan_never_ends_in_a_trap`.

### Task 1.4: Commit probe when a goal holds and no level-up came

**As built:** the draft read `ProgressMonitor.settled()` (Task 2.3, not built yet). The trigger is the event itself: the goal of the plan being followed (found or partial) is open and holds on a real frame with no level-up. Only the pursued goal triggers it; a reach guess that becomes true by accident doesn't, or probes would cost moves all the time.

**Files:**
- Modify: `arcagent/mind/fluid.py` (`_start_probe`, `_probe_choice`, `_commit`, `_commit_candidates`, `_end_probe`; `_decide` split into `_decide` and `_choose`), `arcagent/mind/goals.py` (status `probing`, `start_probe`, `end_probe`; `level_up` confirms probing goals), `arcagent/mind/types.py` (branch `commit`)
- Test: `tests/mind/test_fluid_agent.py`, `tests/mind/test_goals.py`

**Interfaces:**
- Produces: `COMMIT_PROBES = 4`, `COMMIT_MAX_CELLS = 16`; branch `commit`, event `commit_probe`, goal event status `probing`. Candidates: A5 when offered, then clicks on small non-mover things whose color is unique on screen, fewest cells first, each once. Probes skip the `known no effect` veto (a commit shows nothing until the winning state) and honor `predicted game_over` and `seen no effect here`. A level-up confirms the goal; any other end (scene changed, candidates spent, game over) excludes it.

- [x] Tests: `test_goal_holding_without_level_up_triggers_commit_probe` (A5 known to do nothing, wins only next to the target), `test_commit_probe_is_bounded`, `test_probing_goal_leaves_open_and_is_confirmed_or_excluded`.

**Section gate:** paired offline A/B against the section-2 result. Watch tn36 (should win earlier) and the `forced`/`stable` counts in the logs.

---

## Section 2: Telling progress from motion

**Direction:** judge whether movement changes the functional state, completes a milestone or follows a path toward a target; recognize back-and-forth as motion; model the game's "pose and kinematics" the way rehab AI models joints; read trajectories over time; check completion against a rule set.

**What carries over:**
- **Functional state change versus repetitive motion:** carries over. A scene key that comes back within a window with no goal distance improved is motion (Task 2.3).
- **Pose estimation and kinematics:** carries over as an explicit kinematic model per controlled kind: its step vector per action, slides, blockers and walkable colors, all read from facts (Task 2.2). There's no skeleton to estimate: a game's "joints" are its controlled things and the rigid groups the facts already find.
- **Trajectory toward a destination:** carries over as a goal distance tracked over time (Tasks 2.1, 2.3). An LSTM or transformer would need training sequences; a distance series over the last moves answers the same question.
- **Completion criteria against a rule set:** carries over as `Goal.distance == 0` plus the rule-book of section 5.
- **CNN/YOLO object identity:** already exact on this board (see "What the research changed").

### Tasks 2.1 and 2.2: goal distances and the kinematic model

**As built:** `Goal.distance(scene, grid=None, kin=None)` (0 exactly when the goal holds; None for no estimate): reach is the bbox cell gap less one, or with `kin` the moves from `kin.field`; clear is things left; count is `abs(n - count)`; match is mismatched cells. `arcagent/mind/kinematics.py`: `Gait`, `Kinematics.gaits` (move and slide facts in context `any`, simple actions only), `reachable` (forward walk, slides included), `steps_to`, and `field` (backward walk from every top-left touching the target, fixed-step gaits only, cached per (subject, target)). Bounds: `KIN_LIMIT = 4096` top-lefts per walk, `KIN_FIELDS = 8` fields per object; the agent builds one `Kinematics` per move and passes it to its searches, so search distances and the partial-plan guard use the same metric.

- [x] Tests: `tests/mind/test_kinematics.py` (gaits, blockers, slides, around a wall, unreachable, the walk limit, the field, goal distance in moves around a wall) and the distance tests in `tests/mind/test_goals.py`.

### Task 2.3: Progress monitor

**As built:** `arcagent/mind/progress.py` `ProgressMonitor` (verdicts `milestone`, `progress`, `regress`, `motion`, `none`; `progressed()`, `settled(goal)`, `best(goal)`, `reset_life`, `reset_level`; `MOTION_WINDOW = 12`, `SETTLE_MOVES = 6`, `REGRESS_SLACK = 2`). Bests are per life: a new attempt has to make its own progress. The agent measures the pursued goal, the confirmed forms and the first open goals (`PROGRESS_GOALS = 20`) each move, logs the verdict as `progress` on the action record, and lets novelty run only while fewer than `MOTION_CUT = 3` motion verdicts came in a row. A transition that changed two or more things and left the scene exactly as the attempt started records `(label, "level", "any") -> reset_level` (one thing stepping back is a step, not a reset); `_veto` refuses such an action with `undoes progress` while `progressed()`, and plan steps honor it.

- [x] Tests: `tests/mind/test_progress.py` (6), `test_a_return_to_the_start_picture_is_learned_as_a_level_reset`, `test_one_thing_stepping_back_to_the_start_is_not_a_reset`, `test_a_known_reset_is_refused_only_while_this_life_has_progress`.

### Task 2.4: Best-first search on goal distance

**As built:** the queue is a heap ordered by `len(plan) + weight * h` (`h` the lowest goal distance in the scene, 0 when none), ties to the lower `h`, then queue order; `HEURISTIC_WEIGHT = 1.0`, `search(..., weight=)`. The tie to lower `h` matters: with an exact distance, about 1,400 scenes tie on priority for a far target, and queue order alone explores them breadth-first. With `weight=0` or no distances the search goes level by level as before. A scene too long to beat a held trap plan is skipped instead of ending the search.

- [x] Tests: `test_best_first_finds_a_far_target_that_breadth_first_runs_out_on` (75 moves found under the cap; breadth-first order runs out), `test_without_distances_search_is_breadth_first`; `test_search_cap_falls_back_to_the_best_partial_plan` now forces breadth-first order, since best-first finds that plan whole.

**Section gate:** paired offline A/B. Watch lf52 level 2 (`search` events should show `found` again after move 40), novelty share on wa30 and ls20, and `undoes progress` vetoes.

---

## Section 3: Same object, new look

**Direction:** move from purely spatial tracking to kinematics, temporal context and feature invariance: IoU anchoring, appearance-invariant identity with mutable state properties, Hungarian matching on a global cost matrix, Kalman prediction, and an object-centric model with decoupled attributes keyed by identity and context instead of color and shape; the action engine tells the fact model what it changed.

**What carries over:**
- **IoU anchoring:** carries over as an overlap term in the match cost.
- **Hungarian matching:** carries over. No scipy, so `assign.py` implements it (O(n³)) and runs only on the things left after exact stationary matches, which are few on most frames. Capped at `ASSIGN_MAX = 40` per side; above that the current greedy order runs.
- **Kalman prediction:** carries over in adapted form. Moves here are discrete and caused by our own action, so the best motion prior is the predictor's own expected scene for the action just taken. When no fact covers a thing, the prior is its last displacement under the same action id (the constant-velocity part of a Kalman filter, without the noise model, which a noise-free grid doesn't need).
- **Explicit action context:** already there in spirit: `predict` runs before every move. This section makes the tracker use it: match against the prediction, so "object 42 turned to face left" is expected, not discovered.
- **Decoupled attributes:** carries over. A thing gets a stable `kind` (identity class) alongside its `sig` (current appearance). Facts key on kind; appearance becomes an effect the predictor applies (`reshape` to a sig) instead of a change of identity.
- **Relational identity (GNN idea):** carries over as a cost term: the offset to the nearest unchanged anchor thing.
- **Slot attention, learned embeddings:** not built. They need training data and a GPU; the board already gives exact components.

### Task 3.1: Hungarian assignment

**Files:**
- Create: `arcagent/mind/assign.py`
- Test: `tests/mind/test_assign.py`

**Interfaces:**
- Produces: `assign(cost: Sequence[Sequence[float]], gate: float) -> list[tuple[int, int]]`, pairs sorted by row: among assignments that use only pairs with cost at most `gate` (and finite), the one with the most pairs, then the least total cost (as built: a tracker would rather explain two things with two plausible matches than one cheap match and an unexplained appearance). `ASSIGN_MAX = 40`; a matrix with a side above it, or ragged rows, raises `ValueError` (callers fall back). Measured: 1.4 ms for 40×40, 0.08 ms for 10×10.

- [ ] **Step 1: Failing tests.** `test_square_matrix_optimum` (a 3×3 case where greedy is wrong: `[[1,2,9],[2,9,9],[9,9,1]]` gives `{(0,1),(1,0),(2,2)}`), `test_rectangular_leaves_extra_unmatched`, `test_gate_drops_expensive_pairs`, `test_empty_matrix`, `test_size_cap_raises`, `test_matches_brute_force_on_random_5x5` (100 seeded matrices against `itertools.permutations`).
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement** (Kuhn-Munkres with potentials, pure Python). **Step 4: Gates.** **Step 5: Commit** "Hungarian assignment".

### Task 3.2: Tracker matches against the predicted scene

**Files:**
- Modify: `arcagent/mind/objects.py` (`Tracker.update`, `_match`)
- Modify: `arcagent/mind/fluid.py` (pass the last prediction's scene)
- Create: `scripts/track_churn.py` (replay check)
- Test: `tests/mind/test_objects.py`

**Interfaces:**
- Consumes: `assign`.
- Produces: `Tracker.update(grid, mask, prior: Scene | None = None)`. Order: exact stationary same-sig pairs; then `assign` over the leftovers with cost = `W_DIST * manhattan(after top-left, prior top-left or last displacement) + W_OVERLAP * (1 - IoU with prior cells) + W_SIZE * |Δcells|/max + W_COLOR * (color changed) + W_SHAPE * (shape changed) + W_REL * (anchor offset changed)`, gated at `MATCH_GATE`. Starting weights: 1.0, 4.0, 2.0, 3.0, 1.0, 1.0; gate 24. A record of which ids were matched through the prior: `Tracker.via_prior: frozenset[int]`.

- [ ] **Step 1: Failing tests.**
  - `test_facing_change_while_moving_keeps_the_id`: a sprite moves one cell and its shape flips (a facing marker on the other side). Assert one `reshaped` change with the old id, no `appeared`.
  - `test_recolor_and_move_keeps_the_id_with_prior`.
  - `test_predicted_teleport_keeps_the_id`: the prior places the thing far away; assert same id.
  - `test_unpredicted_far_jump_is_disappear_and_appear` (beyond the gate, no prior).
  - `test_two_same_sig_things_crossing_keep_their_ids` (the prior says they swapped sides).
  - Every existing `test_objects.py` case passes with `prior=None`.
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement.**
- [x] **Step 4: Replay check.** `scripts/track_churn.py <npz>` replays a recorded run and prints appeared+disappeared per move. Result on all 25 recordings, old tracker against new (no prior in a replay, so this is the matching rules alone): no game worse; total churn per move 20.5 to 17.6; tu93 0.52 to 0.09, lp85 0.38 to 0.01, dc22 0.92 to 0.47, cn04 0.26 to 0.14, wa30 0.38 to 0.18.

**As built:** the cost adds `KIND_COST` per match kind (prior and sig 0, recolor 20, reshape 40, near 60), so the kind outranks distance; without it a player stepping onto a same-shape lava dot lost its id to the lava "recoloring" in place (caught by `test_game_over_prefers_the_overlapped_color`). Cross-color pairs without overlap or a prediction are never candidates. The gate is the candidate tests themselves (radius 16), so `MATCH_GATE` isn't needed; the relational term was left out, since with stationary anchors it equals the distance term. `Tracker.how` records each id's match kind for Task 3.3.
- [ ] **Step 5: Gates.** **Step 6: Commit** "Tracker matches against the predicted scene".

### Task 3.3: Kinds: identity classes across appearances

**As built:** `arcagent/mind/kinds.py` (`KindBook`, `KIN_EVIDENCE = 2`) and `Thing.kind` (defaults to the sig). Changes from the draft, my calls:
- A kind is named by the first sig of its class (still a `c<color>:<shape>` string), so every parser of subject colors keeps working when facts key on kinds.
- Evidence: one track switching between two sigs, matched by `prior`, `recolor`, `near`, or a same-color overlap with size change within `NEAR_SIZE`. Growth and appear-plus-disappear never count.
- Dropped the "two sigs on screen together never join" rule: it would keep on/off tiles apart forever, and the tracking evidence already guards false joins.
- Unions last the whole game (no next-level confirmation): the same sigs recur on later levels, and the extra rule bought little.

`KindBook.observe(before, after, diff, how) -> list[str]` (kinds joined), `kind_of(sig) -> str`, `label(scene) -> Scene`.

- [x] Tests: `test_turning_sprite_becomes_one_kind`, `test_one_observation_is_not_enough`, `test_new_thing_on_vacated_spot_is_not_kin`, `test_a_growing_strip_is_not_kin`, `test_recolor_in_place_twice_is_kin`, `test_kinds_chain_and_label_fills_things`, `test_unlabeled_things_are_their_own_kind`.

### Task 3.4: Facts and goals read kinds (spec change)

**As built (changed from the draft):** keying facts on kind alone would break state-relative looks: a toggle tile gets "click turns it off" from one look and "click turns it on" from the other, under one key. So facts stay keyed by look, and each observation of a thing whose kind differs from its sig is also recorded under the kind, with a move or look change normalized to "and its look becomes X" (the same fact from every look). `FactMemory.lookup(..., kind)` tries a usable look fact, then a usable kind fact, then a broken look fact; the predictor passes the kind (not for the clicked thing) and falls back to kind `blocked` and death facts the same way, reads blockers through both names, and keeps `kind` on predicted things. A kind of two or more looks is named `k` + its first sig (`kinds.KIND_PREFIX`, `base_sig`), so its facts never share a key with that look's. Goals match things by sig or kind and propose a kind in place of its looks (never a kind's own look as its target); search protects and keys by either; novelty counts `(kind, x, y)`, so turning back on a visited cell isn't new. The agent runs `KindBook` on every transition, logs `kind` events and labels its scenes. The spec's component 2 records the decision.

- [x] Tests: `tests/mind/test_kind_facts.py` (learned under the kind with the new look; a kind fact predicts an unseen look; a look fact wins until it breaks; search chains moves across looks; a goal on a kind holds in either look; kinds are movers), `test_a_turning_player_becomes_one_kind_and_reaches_the_target`. On that game: 22 moves with kinds, 26 without (one plan held instead of six replans).

**Section gate:** paired offline A/B. Watch tu93 (search timeouts were 200 of 448; fallback share near 50%), wa30 oscillation on moves 52 to 67, and prediction accuracy (must not fall below the current 0.89).

---

## Section 4: Hidden mode state

**Direction:** treat selection and activity as relational tags on the objects already tracked, in the fact layer, with no LLM and no global mode slot: `is_selected` on the selected piece (or an edge from a hidden cursor entity to it), and `is_active` on the controller that's driving.

**What carries over:** all of it. My calls are how the tags get detected from the screen and how facts read them, since nothing announces a selection: the agent has to infer it from what a click changed.

### Tasks 4.1 to 4.3: selection, the active controller, and facts that read them

**As built:**
- 4.1 `arcagent/mind/modes.py` `ModeTracker`: a candidate mark is an outline appearing around a clicked thing (color and shape) or the clicked thing recoloring; it is a known selection marker once it moves to another clicked thing (the tracker may keep the outline's id as it jumps, so a moved mark counts) or a second click takes it away. Masked cells and the clicked thing's own move are never marks.
- 4.2 `Mode.active`: after a non-click action in which exactly one controlled thing (`goals.controlled_sigs`) moved, that thing is active, but only while two or more controlled things are on screen: with one, filing its moves under `active` would pull them out of the `any` facts that the slide, teleport and blocked rules read (a test caught it). Action numbers carry no meaning.
- 4.3 `Scene.selected` and `Scene.active` are part of the scene and its key. `modes.mode_contexts`: `selected:dx,dy`, `between:dx,dy`, `sel:dx,dy` (past `SEL_RANGE = 8`: `far`) for a click while something is selected; `active` for the active thing under other actions. `learn` gives those contexts to observations still in `any`; the predictor looks them up first (falling back to `any`) and predicts the next mode (`modes.predicted_mode`: a click on a sig or kind in `FactMemory.selectable` selects it alone, any other click clears the selection). The agent runs the tracker each transition and stores the mode on the scene; spec component 11 records the decision.

- [x] Tests: `tests/mind/test_modes.py` (11), `tests/mind/test_mode_facts.py` (6), `test_the_agent_learns_a_selection_marker_and_files_clicks_under_it`. On a five-slot peg game the agent wins in 22 moves with the mode and doesn't win in 300 without it.

### Task 4.4: Click tests key on local view and mode

**As built:** `tester.tried_key(scene, action)`: a simple action keys on `(scene.key(), action.key())` as before; a click keys on the clicked thing's sig, the colors within `MARK_RING` of its bbox (of the point on empty space), and each selected thing's sig and offset from the click. `Tried` is a set of those keys; the tester reads it as a `Container`. The no-op memory keeps the exact scene key (`tester.Noop`).

- [x] Tests: `test_moving_an_unrelated_thing_does_not_reopen_click_tests`, `test_selection_change_does_reopen_click_tests`, `test_identical_neighborhoods_share_a_click_key_but_different_ones_do_not`, `test_simple_actions_keep_the_whole_scene_key`, `test_a_tried_click_settles_its_twins`.

**Section gate:** paired offline A/B. Watch lf52 level 2 (`search` events should show `found` again after move 40), novelty share on wa30 and ls20, and `undoes progress` vetoes.

---

## Section 3: Same object, new look

**Direction:** move from purely spatial tracking to kinematics, temporal context and feature invariance: IoU anchoring, appearance-invariant identity with mutable state properties, Hungarian matching on a global cost matrix, Kalman prediction, and an object-centric model with decoupled attributes keyed by identity and context instead of color and shape; the action engine tells the fact model what it changed.

**What carries over:**
- **IoU anchoring:** carries over as an overlap term in the match cost.
- **Hungarian matching:** carries over. No scipy, so `assign.py` implements it (O(n³)) and runs only on the things left after exact stationary matches, which are few on most frames. Capped at `ASSIGN_MAX = 40` per side; above that the current greedy order runs.
- **Kalman prediction:** carries over in adapted form. Moves here are discrete and caused by our own action, so the best motion prior is the predictor's own expected scene for the action just taken. When no fact covers a thing, the prior is its last displacement under the same action id (the constant-velocity part of a Kalman filter, without the noise model, which a noise-free grid doesn't need).
- **Explicit action context:** already there in spirit: `predict` runs before every move. This section makes the tracker use it: match against the prediction, so "object 42 turned to face left" is expected, not discovered.
- **Decoupled attributes:** carries over. A thing gets a stable `kind` (identity class) alongside its `sig` (current appearance). Facts key on kind; appearance becomes an effect the predictor applies (`reshape` to a sig) instead of a change of identity.
- **Relational identity (GNN idea):** carries over as a cost term: the offset to the nearest unchanged anchor thing.
- **Slot attention, learned embeddings:** not built. They need training data and a GPU; the board already gives exact components.

### Task 3.1: Hungarian assignment

**Files:**
- Create: `arcagent/mind/assign.py`
- Test: `tests/mind/test_assign.py`

**Interfaces:**
- Produces: `assign(cost: Sequence[Sequence[float]], gate: float) -> list[tuple[int, int]]`, pairs sorted by row: among assignments that use only pairs with cost at most `gate` (and finite), the one with the most pairs, then the least total cost (as built: a tracker would rather explain two things with two plausible matches than one cheap match and an unexplained appearance). `ASSIGN_MAX = 40`; a matrix with a side above it, or ragged rows, raises `ValueError` (callers fall back). Measured: 1.4 ms for 40×40, 0.08 ms for 10×10.

- [ ] **Step 1: Failing tests.** `test_square_matrix_optimum` (a 3×3 case where greedy is wrong: `[[1,2,9],[2,9,9],[9,9,1]]` gives `{(0,1),(1,0),(2,2)}`), `test_rectangular_leaves_extra_unmatched`, `test_gate_drops_expensive_pairs`, `test_empty_matrix`, `test_size_cap_raises`, `test_matches_brute_force_on_random_5x5` (100 seeded matrices against `itertools.permutations`).
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement** (Kuhn-Munkres with potentials, pure Python). **Step 4: Gates.** **Step 5: Commit** "Hungarian assignment".

### Task 3.2: Tracker matches against the predicted scene

**Files:**
- Modify: `arcagent/mind/objects.py` (`Tracker.update`, `_match`)
- Modify: `arcagent/mind/fluid.py` (pass the last prediction's scene)
- Create: `scripts/track_churn.py` (replay check)
- Test: `tests/mind/test_objects.py`

**Interfaces:**
- Consumes: `assign`.
- Produces: `Tracker.update(grid, mask, prior: Scene | None = None)`. Order: exact stationary same-sig pairs; then `assign` over the leftovers with cost = `W_DIST * manhattan(after top-left, prior top-left or last displacement) + W_OVERLAP * (1 - IoU with prior cells) + W_SIZE * |Δcells|/max + W_COLOR * (color changed) + W_SHAPE * (shape changed) + W_REL * (anchor offset changed)`, gated at `MATCH_GATE`. Starting weights: 1.0, 4.0, 2.0, 3.0, 1.0, 1.0; gate 24. A record of which ids were matched through the prior: `Tracker.via_prior: frozenset[int]`.

- [ ] **Step 1: Failing tests.**
  - `test_facing_change_while_moving_keeps_the_id`: a sprite moves one cell and its shape flips (a facing marker on the other side). Assert one `reshaped` change with the old id, no `appeared`.
  - `test_recolor_and_move_keeps_the_id_with_prior`.
  - `test_predicted_teleport_keeps_the_id`: the prior places the thing far away; assert same id.
  - `test_unpredicted_far_jump_is_disappear_and_appear` (beyond the gate, no prior).
  - `test_two_same_sig_things_crossing_keep_their_ids` (the prior says they swapped sides).
  - Every existing `test_objects.py` case passes with `prior=None`.
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement.**
- [x] **Step 4: Replay check.** `scripts/track_churn.py <npz>` replays a recorded run and prints appeared+disappeared per move. Result on all 25 recordings, old tracker against new (no prior in a replay, so this is the matching rules alone): no game worse; total churn per move 20.5 to 17.6; tu93 0.52 to 0.09, lp85 0.38 to 0.01, dc22 0.92 to 0.47, cn04 0.26 to 0.14, wa30 0.38 to 0.18.

**As built:** the cost adds `KIND_COST` per match kind (prior and sig 0, recolor 20, reshape 40, near 60), so the kind outranks distance; without it a player stepping onto a same-shape lava dot lost its id to the lava "recoloring" in place (caught by `test_game_over_prefers_the_overlapped_color`). Cross-color pairs without overlap or a prediction are never candidates. The gate is the candidate tests themselves (radius 16), so `MATCH_GATE` isn't needed; the relational term was left out, since with stationary anchors it equals the distance term. `Tracker.how` records each id's match kind for Task 3.3.
- [ ] **Step 5: Gates.** **Step 6: Commit** "Tracker matches against the predicted scene".

### Task 3.3: Kinds: identity classes across appearances

**As built:** `arcagent/mind/kinds.py` (`KindBook`, `KIN_EVIDENCE = 2`) and `Thing.kind` (defaults to the sig). Changes from the draft, my calls:
- A kind is named by the first sig of its class (still a `c<color>:<shape>` string), so every parser of subject colors keeps working when facts key on kinds.
- Evidence: one track switching between two sigs, matched by `prior`, `recolor`, `near`, or a same-color overlap with size change within `NEAR_SIZE`. Growth and appear-plus-disappear never count.
- Dropped the "two sigs on screen together never join" rule: it would keep on/off tiles apart forever, and the tracking evidence already guards false joins.
- Unions last the whole game (no next-level confirmation): the same sigs recur on later levels, and the extra rule bought little.

`KindBook.observe(before, after, diff, how) -> list[str]` (kinds joined), `kind_of(sig) -> str`, `label(scene) -> Scene`.

- [x] Tests: `test_turning_sprite_becomes_one_kind`, `test_one_observation_is_not_enough`, `test_new_thing_on_vacated_spot_is_not_kin`, `test_a_growing_strip_is_not_kin`, `test_recolor_in_place_twice_is_kin`, `test_kinds_chain_and_label_fills_things`, `test_unlabeled_things_are_their_own_kind`.

### Task 3.4: Facts and goals read kinds (spec change)

**As built (changed from the draft):** keying facts on kind alone would break state-relative looks: a toggle tile gets "click turns it off" from one look and "click turns it on" from the other, under one key. So facts stay keyed by look, and each observation of a thing whose kind differs from its sig is also recorded under the kind, with a move or look change normalized to "and its look becomes X" (the same fact from every look). `FactMemory.lookup(..., kind)` tries a usable look fact, then a usable kind fact, then a broken look fact; the predictor passes the kind (not for the clicked thing) and falls back to kind `blocked` and death facts the same way, reads blockers through both names, and keeps `kind` on predicted things. A kind of two or more looks is named `k` + its first sig (`kinds.KIND_PREFIX`, `base_sig`), so its facts never share a key with that look's. Goals match things by sig or kind and propose a kind in place of its looks (never a kind's own look as its target); search protects and keys by either; novelty counts `(kind, x, y)`, so turning back on a visited cell isn't new. The agent runs `KindBook` on every transition, logs `kind` events and labels its scenes. The spec's component 2 records the decision.

- [x] Tests: `tests/mind/test_kind_facts.py` (learned under the kind with the new look; a kind fact predicts an unseen look; a look fact wins until it breaks; search chains moves across looks; a goal on a kind holds in either look; kinds are movers), `test_a_turning_player_becomes_one_kind_and_reaches_the_target`. On that game: 22 moves with kinds, 26 without (one plan held instead of six replans).

**Section gate:** paired offline A/B. Watch tu93 (search timeouts were 200 of 448; fallback share near 50%), wa30 oscillation on moves 52 to 67, and prediction accuracy (must not fall below the current 0.89).

---

## Section 4: Hidden mode state

**Direction:** treat selection and activity as relational tags on the objects already tracked, in the fact layer, with no LLM and no global mode slot: `is_selected` on the selected piece (or an edge from a hidden cursor entity to it), and `is_active` on the controller that's driving.

**What carries over:** all of it. My calls are how the tags get detected from the screen and how facts read them, since nothing announces a selection: the agent has to infer it from what a click changed.

### Tasks 4.1 to 4.3: selection, the active controller, and facts that read them

**As built:**
- 4.1 `arcagent/mind/modes.py` `ModeTracker`: a candidate mark is an outline appearing around a clicked thing (color and shape) or the clicked thing recoloring; it is a known selection marker once it moves to another clicked thing (the tracker may keep the outline's id as it jumps, so a moved mark counts) or a second click takes it away. Masked cells and the clicked thing's own move are never marks.
- 4.2 `Mode.active`: after a non-click action in which exactly one controlled thing (`goals.controlled_sigs`) moved, that thing is active, but only while two or more controlled things are on screen: with one, filing its moves under `active` would pull them out of the `any` facts that the slide, teleport and blocked rules read (a test caught it). Action numbers carry no meaning.
- 4.3 `Scene.selected` and `Scene.active` are part of the scene and its key. `modes.mode_contexts`: `selected:dx,dy`, `between:dx,dy`, `sel:dx,dy` (past `SEL_RANGE = 8`: `far`) for a click while something is selected; `active` for the active thing under other actions. `learn` gives those contexts to observations still in `any`; the predictor looks them up first (falling back to `any`) and predicts the next mode (`modes.predicted_mode`: a click on a sig or kind in `FactMemory.selectable` selects it alone, any other click clears the selection). The agent runs the tracker each transition and stores the mode on the scene; spec component 11 records the decision.

- [x] Tests: `tests/mind/test_modes.py` (11), `tests/mind/test_mode_facts.py` (6), `test_the_agent_learns_a_selection_marker_and_files_clicks_under_it`. On a five-slot peg game the agent wins in 22 moves with the mode and doesn't win in 300 without it.

### Task 4.4: Tests and no-op memory key on mode, not the whole scene

This is the tried-set scope fix that was held back: keying a test on its local context was unsafe while selection was invisible to the agent.

**Files:**
- Modify: `arcagent/mind/tester.py` (`Tried` key), `arcagent/mind/fluid.py`
- Test: `tests/mind/test_tester.py`, `tests/mind/test_fluid_agent.py`

**Interfaces:**
- Produces: `TestKey = tuple[str, str, Mode-signature]`: (click label or action label, the clicked kind's local neighborhood signature within `MARK_RING`, the selected kinds and their offsets). `Tried` becomes `set[TestKey]`. The `_noop` set keeps the full scene key (an observation on that exact scene is still the strongest evidence).

- [ ] **Step 1: Failing test** `test_moving_an_unrelated_thing_does_not_reopen_tests`: after 8 empty-slot clicks with no effect and no selection, one move of an unrelated carrier must not make those 8 clicks testable again. Plus `test_selection_change_does_reopen_tests`.
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement.** **Step 4: Gates.** **Step 5: Commit** "Tests key on local context and mode".

**Section gate:** paired offline A/B. Watch lf52 (the 8-click loop of moves 41 to 120 should be gone), and any game with selectable pieces.

---

## Section 5: Goal kinds beyond reach and clear

**Direction:** two tiers: a deterministic parser of screen primitives feeding a symbolic layer that runs the state machine and scoring. Keep the LLM out of the frame loop. Use an LLM only offline, to compile the goal set into a structured rule-book the code reads locally at run time.

**What carries over:**
- **Two tiers:** carries over. Tier 1 is `objects` plus a new `regions` module. Tier 2 is a rule-book file plus the goal code that evaluates it.
- **LLM out of the frame loop:** already true for goals; this keeps it so.
- **Offline LLM compile:** carries over with one guard. The rule-book must describe generic goal forms (match a reference, cover every target), never what a public game needs; anything else breaks the "no memorized solutions" rule. My recommendation: write the rule-book by hand from root mechanics; if an LLM drafts entries, it sees only the generic descriptions, and every entry is reviewed before commit.

### Tasks 5.1 and 5.2: region primitives, the rule-book, match and cover

**As built:** `arcagent/mind/regions.py` (`framed_regions`, `twin_regions` with `TWIN_MIN = 3` and `MAX_TWINS = 20`, `static_cells`, `render`). The rule-book is `arcagent/mind/goal_rules.py`, a typed Python data module instead of JSON, so it ships inside the package with no data-file step (`GoalRule(kind, prior, proposed)`, `prior_of`, `proposed`). Goal kinds: `match` (already evaluated, never proposed before) is proposed for each twin pair of framed regions when `propose` gets the raw grid, and search now plans toward it by rendering predicted scenes; `cover(a, b)` is new: every visible thing named `b` has its own thing named `a` on or beside it, one-to-one by `assign` over cell gaps, proposed for movers and pushables against target sigs with two or more things (`MAX_COVER_TARGETS = 5`). Not built: `match_thing` (region match covers multi-color patterns), `sequence` (needs a progress-indicator detector first), `hold_commit` (the Task 1.4 commit probe does it). Spec component 4 records the decision.

**Research for `sequence` (2026-10-25):** a sequence goal reads a progress indicator, so the first question was whether the public games show one. A scan of the 25 probe traces (`bench/synth/traces`, 120 moves each) looked for cell groups that change on few moves, then followed each candidate along its edge strip. None fills on a correct event and empties on a wrong one. What the edge strips turned out to be:

- a click budget that drains five cells per click (lp85's left column; the budget reader already handles this kind);
- a palette whose pieces get outlined when clicked and leave the row when placed (sb26's bottom row);
- slot rows that a click or an action edits: glyphs that cycle in place (tr87), cells that toggle on and off (sc25, tn36).

A random probe rarely makes progress, so 120 moves can't rule an indicator out. Still, nothing in the public traces would let a detector be checked. The toggle and glyph rows are evidence for a different kind: set a row of slots to a target pattern. That's close to `match` when a template is on screen, and close to `match_thing` otherwise.

### Task 5.5: match_thing

Kameron's go (2026-10-25). The draft's definition: a thing of kind A has the same shape and colors as a reference R (a pattern the player changes or carries against a displayed target); distance is the differing cells with top-lefts aligned.

**As built:** `Goal("match_thing", a=picture, b=family, n=ref id, region_b=ref)`.
- **Family:** `restyle_families(mem)` joins looks by recolor and reshape facts (a fact's subject and the look it produces), so the goal follows the thing through looks that kinds haven't joined yet.
- **Picture kept in the goal:** `a` is the reference's cells when proposed (`picture`). The reference shares a look with the family, so predictions apply the family's restyles and moves to it too. References are static by construction, so the stored picture is the right one.
- **Pinned reference:** `n` is the tracked id of the thing on the reference. It never counts as the family thing, and search passes it to `predict(..., pinned=)`, which leaves pinned things where they are whatever the facts say. Without that, search "matched" by moving the reference away, or lost the branch on a predicted restyle of the reference.
- **Proposal** (`_match_thing_guesses`, needs the agent's `static` mask: cells unchanged since the level's first frame): regions of a family thing's size that are a thing's bbox or a framed region's inside, all static; only once a family thing has moved or changed this level (until then its own spot looks like a reference too); skipping pictures a family thing elsewhere already shows; at most `MAX_MATCH_THINGS = 10` a call. Prior 1.3.
- **Two fixes the toy game needed outside goals:** the predictor treats a borrowed fact (a sibling's, an earlier look's) that restyles a thing into its current look as unknown, kind facts excepted (`_restyled_here`); the tester no longer counts earlier-look facts as settling a look the same action produced (`_made`). Both said "no change" for a cycling restyle, so the agent never pressed again.

In `RestyleGame` (a piece whose color A5 cycles, a reference of another color, two levels) the agent wins level 2 in 3 moves once the cycle is known; without `match_thing` it doesn't finish level 2 in the 270 moves left.

- [x] Tests: `test_match_thing_compares_pictures_wherever_the_thing_is`, `test_match_thing_is_proposed_for_a_restyled_thing_and_same_size_references`, `test_match_thing_follows_the_family_through_its_looks`, `test_search_plans_toward_a_match_thing_goal`, `test_the_restyle_that_made_a_look_does_not_settle_the_new_look`, `test_pinned_things_stay_put_whatever_the_facts_say`, `test_the_agent_restyles_a_piece_to_match_a_reference`.

**Gate:** full offline run against main (10 levels, 0.626).

- [x] Tests: `tests/mind/test_regions.py` (6), `tests/mind/test_goal_kinds.py` (5: rule-book kinds, cover holds and distance, cover with one side missing, cover and match proposed, search plans toward a match goal).

### Task 5.3: Rank goals instead of capping at 200

**As built:** `GoalBook.ranked(scene, grid=, kin=, salient=, budget=)`: confirmed forms, then the best `SEARCH_GOALS = 12` open goals by `prior * salience / (1 + cost)` (`SALIENCE = 2.0` when a goal's sigs touch `salient`, `UNKNOWN_COST = 20` with no distance), leaving out goals farther than `budget` for that ranking only. The agent's searches use it (`_ranked`: salient = controlled things and things changed this level, budget = the reader's moves left) plus every goal the executive named. `open()` keeps its insertion order for everything else.

**Changed after the bisect (2026-10-25):** the budget filter is gone. Distances estimate moves without bounding them, so a goal one sliding click away could be left out as "too far". The twelve-goal cut is gone too: search gets every open goal, best first. On the full offline run this took the branch from 9 levels and 0.390 to 10 levels and 0.626 (ls20 back to 60 actions, vc33's second level in 195).

- [x] Tests: `test_ranked_puts_cheap_salient_goals_first`, `test_goal_beyond_budget_is_left_out_this_life_only`, `test_confirmed_form_stays_first_and_search_gets_every_open_goal`.

**Section gate:** paired offline A/B. Watch ls20 (pattern against target), vc33 and wa30 (carry into the frame), and the goal count per game (was up to 202).

---

## Section 6: Budget awareness

**Direction:** with the counter readable, move from a blind mode to a shortest-path efficiency engine: drop goals that can't fit the remaining budget; stop the moment the goal is done or the budget can't start another sequence; prune by move efficiency, using the shortest route over remaining targets when it fits the budget.

**What carries over:**
- **Goal feasibility against the budget:** carries over (Task 5.3 uses it; Task 6.2 adds it to search depth).
- **Early termination:** carries over in a changed form. A level ends by itself when its goal holds, so there's nothing to hold once the goal is done. The waste is at the other end: a life that can't finish still costs every move left in it, because the scorecard counts lost lives. A RESET is one action. So when the cheapest feasible plan is longer than the moves left and no test is worth its move, the agent should RESET at once (Task 6.3).
- **Shortest route over targets:** carries over: Held-Karp for up to `ROUTE_EXACT = 10` targets, nearest neighbor plus 2-opt above that, over kinematic step distances (Task 6.4).

### Task 6.1: Read the budget

**As built:** `arcagent/mind/budget.py`. The recordings showed full-width one-row bars draining a fraction of a cell per move (vc33 64 cells for 50 moves, wa30 64 for 200), and bars that drain only on some moves (lp85, ft09). So: `NoiseMask.counters` keeps each flagged counter as a `budget.Counter` (color, row or column band, direction), through game overs; `BudgetReader` counts the bar's cells across the band, keeps two counts of the life's moves (`all`, and `busy`: leaving out moves that changed nothing and left the bar alone), fits a least-squares line against each and uses the better-fitting count (`MIN_FIT = 5`); moves left = cells left over the fitted rate, trusted after `MIN_DRAIN = 6` cells drained; a game over with under `BUDGET_DEATH_MOVES = 2` moves of bar left sets `per_life`, after which the rate is exact; `new_level` forgets it all. The agent feeds it every frame and logs `budget` on each action record.

**Replay check** (`scripts/budget_check.py`, all 25 recordings, scoring only budget deaths): mean error 0.1 to 2.3 moves on 16 of the 21 games with a counter (vc33 0.4, tu93 0.5, s5i5 0.2, sp80 0.1, ka59 0.8, re86 0.9, r11l 0.7), worse on g50t 3.5, sc25 3.5, ar25 4.0, m0r0 5.1, lp85 8.4. Maximum errors reach 14 to 85 moves early in a life when the chosen count flips, so decisions on the estimate need a margin (Task 6.3). sb26 and sk48 have no counter detected (the known miss); lf52 and ls20 had no deaths to score.

- [x] Tests: `tests/mind/test_budget.py` (7), `test_the_agent_reads_its_budget_and_knows_it_exactly_after_a_budget_death`.

### Task 6.2: Budget-bounded search and tests

**As built:** `search(..., max_depth=)` doesn't expand a scene whose plan is that long; the agent passes the reader's moves left. `budget_tight(remaining=, distance=)`: a plan fits with at most `TEST_RESERVE = 2` moves to spare; then the best partial plan's step goes ahead of tests (`_tight_choice`).

Also in this commit: `tester.test_key` is renamed `tried_key`. Pytest had been collecting the imported `test_key` in `tests/mind/test_tester.py` as a test since Task 4.4 (an ERROR in every run that a FAILED-only filter didn't show); the suite is clean again (518 passed, 1 skipped).

- [x] Tests: `test_search_never_plans_past_the_budget`, `test_a_tight_budget_puts_a_fitting_partial_plan_before_tests`.

### Task 6.3: RESET a life that can't finish

**As built:** `life_doomed(remaining=, distances=, exact=, acted=)`: only with an exact budget (a life already ended by it on this level, so the reader's early errors can't trigger it), after an action this game (the engine resets the whole game otherwise), with goals present, every distance known, and the nearest more than `DOOM_MARGIN = 2` moves past the moves left. Goals about things not on screen are skipped (they can't be reached now). It runs after tests and before partial plans and novelty: an open test still beats a RESET. The RESET (branch `reset`, why `doomed life`) restarts the attempt's state like a game over.

Two fixes the test game exposed, in the same commit: the noise mask now masks a counter's whole band (a bar redrawn full on a new life reached past the flagged box and showed up as things, which drew tests and goals), and the reader keeps the moves made before its counter was flagged and sees the final frame of a life (per-life counts came out 2 or 3 short). Budget replay after these: summed mean error about 30 moves over the 21 games with a counter (was about 43).

- [x] Tests: `test_life_doomed_needs_an_exact_budget_known_distances_and_no_way_through`, `test_a_doomed_second_life_is_reset_at_once`, `test_no_doomed_reset_when_the_target_fits`; the noise tests now expect the band.

**Removed after the bisect (2026-10-25):** the doomed-life RESET came out; the band mask and reader fixes stay. On the five-game bisect it cost ls20 and su15 their only level (su15 reset 1,678 times). The check needs a lower bound on the moves to a goal, and goal distances aren't one: in su15 a click slides a piece many cells, so the cell gap overstates the moves. Every life looked doomed and ended early, and with it the moves the agent needed to learn.

### Task 6.4: Shortest route over targets

**As built:** `arcagent/mind/route.py` `route(start, targets, dist, exact_limit=ROUTE_EXACT)`: Held-Karp for up to `ROUTE_EXACT = 6` targets (the draft's 10 would cost about 100k steps a call), nearest neighbor plus up to `TWO_OPT_PASSES = 4` passes of 2-opt above that; None when a leg is unreachable. Used by the `clear` distance when the agent knows its one controlled thing (`Kinematics.mover`, set from `_lone_mover`): things left plus the cells of a heuristic route through them (`MAX_ROUTE_TARGETS = 20`). Inside search it uses the heuristic order only (exact routing at every node would cost seconds a move). Not built: locking a whole route as a plan; best-first search on this distance does the ordering instead.

- [x] Tests: `tests/mind/test_route.py` (exact beats greedy, matches brute force on six targets, the heuristic on fifteen, unreachable and empty), `test_clear_distance_with_a_mover_adds_the_route_through_what_is_left`.

**Section gate:** paired offline A/B. Watch actions per won level on budget games and the count of `doomed life` resets.

---

## Sources

- Rudakov, Shock, Cowley. Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks. arXiv 2512.24156 (status-bar masking, five salience tiers, nearest untested pair, the reset-loop bug, the state-space limit).
- Bewley, Ge, Ott, Ramos, Upcroft. Simple Online and Realtime Tracking. ICIP 2016, arXiv 1602.00763 (Kalman motion plus Hungarian assignment).
- Local: `arc_agi/scorecard.py` (`inc_reset_count` counts a RESET as an action; per-level actions include lost lives) and `arcengine/base_game.py` (`handle_reset`: level reset after the first action).
