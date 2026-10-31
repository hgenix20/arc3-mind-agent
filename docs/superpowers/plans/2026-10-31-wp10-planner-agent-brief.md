# WP10: the planner agent (integration)

Spec: `C:/arc-agent/docs/superpowers/specs/2026-10-31-agent-spec-v2.md` sections 3.1 to
3.12 (the graphs), 4.5, 4.7 to 4.13. Plan: `C:/arc-agent/docs/superpowers/plans/2026-10-31-agent-v2-build.md`
sections 2 and 3 and the decisions D-A to D-E. Code map:
`C:/arc-agent/docs/superpowers/plans/2026-10-31-v2-code-map.md` (confirm against the files).

Wave-1 modules are merged on `agent-v2`. Their implementers' interface reports and
concerns are collected in
`C:/Users/Sagac/AppData/Local/Temp/claude/C--Program-Files-Git/76676058-e8f8-4289-9a20-0d0f4ca84f38/scratchpad/wave1-interfaces.md`
(read it first; then read each module's docstring for the final word). Known
differences from the names below: `adverse.features(label, kind, shape_class, context,
*, roles=..., health_bar=...)` takes `roles` and `health_bar` as keywords; `adverse`
imports the band constants from `facts`; the harm verdict (health bar drop, or death
when there is no bar) is yours to compute, the book only records `harmed: bool`. Public
interfaces (read each file's docstring for the exact names): `facts.Fact.probability/band/state`, `FactMemory.probability`,
`FactMemory.ranked`; `adverse.AdverseBook`, `adverse.features`, `adverse.InhibitionNet`,
`adverse.harm_probability`, `adverse.Perseverance`, `adverse.band_of`; `longterm.LongTermStore`,
`longterm.RuleRecord`, `longterm.SequenceRecord`, `longterm.canonical_key`,
`longterm.statement_for`; `rollout.rollout`, `rollout.Rollout`; `hypotheses.Hypothesis`,
`hypotheses.expected_value`, `hypotheses.from_rollout`, `hypotheses.rank`,
`hypotheses.hypothesis_id`; `plantree.PlanTree`, `plantree.return_route`;
`selfrecord.SelfRecord`, `selfrecord.FocusRecord`; `controller.Controller`,
`controller.Signals`, `controller.Decision`; `flags.FlagBook`; `briefing.Briefing`,
`briefing.render`, `briefing.planner_messages`, `briefing.PLANNER_SYSTEM_PROMPT`;
`contract.parse_reply(..., max_steps=)`, `contract.Reply.skill/question/labels`,
`contract.SkillCall`; `skills.run_skill`, `skills.SkillContext`, `skills.SkillResult`;
`runlog` docstring (v2 fields and events); `scripts/mind_metrics.v2_metrics`.

## Part A: `arcagent/mind/planner_agent.py` (new) and one additive hook in `fluid.py`

`PlannerAgent(FluidAgent)`. Without a chat it plays like `FluidAgent`. With a chat it is
the model-first agent of spec section 3. Constructor: `FluidAgent`'s parameters plus
`store: LongTermStore | None = None`.

### A1. State added in `__init__`

- `self._store`, `self._adverse = AdverseBook()`, `self._net = InhibitionNet()` (load
  `store.net_state` when present), `self._persevere = Perseverance()`.
- `self._tree = PlanTree()`, `self._flags = FlagBook()`, `self._controller = Controller()`.
- `self._model_text = ""` (the planner's model from its last reply), `self._pending_q = ""`
  (a question the planner asked, answered in the next briefing).
- `self._segment_id = ""`, `self._segment_source = ""`, `self._hypothesis_id = ""`,
  `self._fork_id = ""`, `self._need_segment = True`, `self._return_route: list[Action]`.
- `self._level_moves = 0` (moves in this level instance), `self._level_start_moves:
  list[Action]` (moves from the level start, for forks and the sequence record),
  `self._no_progress = 0`, `self._fallbacks = 0`.
- `self._health: int | None` (None until a health bar is identified), `self._bar_drops`,
  `self._bar_moves` (to tell a health bar from a move counter, see A6).
- `self._move_t0: float`, `self._move_tokens: int`, `self._gate_trip: str | None`.
- `self._p_step: float | None`, `self._p_path: float | None`, `self._prev_matched: bool | None`.
- `self._planner_calls = 0`, `self._tokens_total = 0`.

### A2. Hook in `fluid.py` (additive only)

Add `def _action_fields(self) -> dict[str, object]: return {}` to `FluidAgent` and merge
its result into the `self._log.action(...)` call in `choose` (`**self._action_fields()`),
after the fixed fields. Nothing else in `fluid.py` changes. `PlannerAgent._action_fields`
returns `self_rec`, `focus`, `hypothesis_id`, `segment_id`, `fork_id`, `planner_call`,
`p_step`, `p_path`, `prev_matched`, `gate` (per the `runlog` docstring; the field is
`self_rec` because `RunLog.action`'s first parameter is `self`; never use an envelope key
such as `game` as a field name, `RunLog` raises on it).

### A3. The move (override `_choose`)

1. Start the move clock: `self._move_t0 = self._clock()`, `self._move_tokens = 0`,
   `self._gate_trip = None`.
2. Build `Signals` (gate_tripped False at this point, segment_broke from the last
   `_check_plan`, fallbacks_in_row, no_progress_moves, budget_left from
   `self._budget.remaining()`, segment_left `len(self._llm_plan)`, deadline_passed `late`)
   and ask the controller. Log `handoff` when the method is not `keep`.
3. Method `return_to_fork`: if the tree has a newest fork with an untried path, compute the
   route with `plantree.return_route(graph.edges, current_key, fork.state_key)` where the
   graph is the explorer's (`self._baseline.policy`, read `explore.py` for the attribute
   that holds the `Graph`), and `current_key` is that policy's key of this observation
   (its `key(obs)` method). Set the route as the segment (`_llm_plan` of `PlanItem`s with
   expect "return to fork <id>"), log `return_route`; when the route is empty (already at
   the fork), start the untried path as the segment (`start_path`). When no fork has an
   untried path, log `replan_root` and fall through to `replan`.
4. Method `explore` or `fallback`: take the base class's fallback branch for this move
   (call `super()._choose` after clearing `_llm_plan`; the base's own ladder then runs).
5. Method `keep` with a segment step left: `super()._choose` (the base follows `_llm_plan`
   through `_llm_choice`); record `p_step` from a one-step `rollout` of that action.
6. Method `replan` or no segment left (`_need_segment`): run the planning loop (A4), which
   sets a new segment or leaves none; then `super()._choose`.
7. After the choice, if the move took longer than `T_MOVE` (module constant, 60.0 s) or
   spent more than `N_TOKEN` (4096) planner tokens, set `self._gate_trip` and log
   `gate_trip` (the move still returns the choice made; the next move's controller sees
   `gate_tripped`).

### A4. The planning loop (one method, `_replan(obs, scene, late)`)

1. Horizon: `typical = store.typical_moves(game, level)` else
   `store.typical_for_level(level)` else `HORIZON_CAP` (200); `horizon = ceil(typical)`,
   `max_steps = min(horizon, SEGMENT_CAP)` with `SEGMENT_CAP = 12`.
2. Hypotheses from the code: for up to `TOP_GOALS` (3) open goals (the goal book's
   `ranked` order), search toward each (`self._search_toward` as `_plan_choice` does) and
   roll the plan out with `rollout(... horizon, health=self._health or 1, adverse=self._harm_of)`;
   `from_rollout(...)` with `typical` and `restart_moves = self._level_moves + len(plan)`.
3. Briefing (A5), planner call (A7) unless `late`, the chat is None, or the move gate has
   less than `MIN_CALL_S` (5.0) left. The reply's `claims` go to `seed_claims`, `model` to
   `_model_text`, `labels` to `_flags.label`, `question` to `_pending_q`.
4. The planner's proposal: a `segment` (its actions remapped on the scene as `_remap`
   does) or a `skill` run through `run_skill` with a `SkillContext` built from the scene
   (roles from A6). Roll the proposal out. If it stops with `death` or `unknown` before
   reaching a goal, and the move gate allows, ask once more with the briefing's `wake`
   set to "segment rejected at step <k>: <reason>" and the same lists. If rejected again,
   drop it (log `planner_call` with `dropped`).
5. Choose by expected value over the planner's hypothesis (source `planner` or `skill`)
   and the code's hypotheses: `hypotheses.rank`. The winner becomes the segment:
   `_llm_plan` as `PlanItem`s, `_llm_protect` as the base sets it, `_plan_goal` None,
   `_segment_id = hypothesis_id`, log `segment` (step, id, source, length, p_path, ev, goal).
6. Forks: if two or more distinct plans reached the same goal, `open_fork(current_key,
   goal_id, plans, moves_here=self._level_start_moves)` and `start_path` on the winner;
   set `_fork_id`. Log `fork_open`.
7. If nothing reached a goal: take the best hypothesis by expected value anyway when its
   path probability is at least `BAND_LOW` (0.3), else leave no segment (the base ladder
   plays: tests, partial plans, novelty, fallback).

### A5. The briefing

Build a `Briefing` from the state (spec 4.9a). Lists:

- `state`: one line per thing: `#<id> <kind> role=<role> rules=<n> p=<mean probability>`;
  `actions`: the available ids and "click: yes|no"; `goals`: "level: <goal id or 'win the
  level'> (<status>)" and "game: win every level (<levels_completed>/<win_levels>)";
  `tasks`: the chosen hypothesis's plan as "<i>. <action> expect <...>"; `level_rules`:
  facts created on this level (`Fact.line()` with probability); `game_rules`: the other
  facts, ranked by probability; `hypotheses`: "<id> <source> goal <g> p=<path> ev=<ev>
  moves=<expected>"; `sequence`: the remaining segment steps; `forks`:
  `self._tree.summary()`; `flags`: `self._flags.lines(8)`; `self_focus`: the two record
  lines; `recent`: `self._moves` with the previous move's predicted and observed match;
  `lessons`: `store.lessons()` when a store is given; `questions`: the seven answers (A8);
  `model`: `_model_text`, which `briefing.render` must title "Your notes from the last
  turn (unverified)" so the planner never reads its own guess as a fact (spec 4.9a);
  `wake`: the reason; `level`; `horizon`.
- Messages from `briefing.planner_messages(b, obs.grid, self._prev_grid)`.

### A6. Roles, harm, and vetoes

- Roles per kind: `controlled` when in `controlled_sigs`; `mover` when in `mover_sigs` and
  not controlled; `harm` when `self._adverse.probability((label, kind, "touch")) >= BAND_LOW`
  for any label with trials, or when a fact says touching it ends the game; `blocker`
  when the kind's colour is in `mem.blockers` for the controlled sig; `collectible` when
  a fact records the kind disappearing on the controlled thing's touch; else `unknown`.
- Health bar: a counter the noise mask flagged whose drops happened on fewer than 30% of
  the moves of this level instance after at least 10 moves is a health bar; its drops are
  harm. A move counter drops on most moves and is not harm. Without a health bar, health
  is 1 and only game over is harm (spec 4.10a, plan D-D).
- On each transition (override `_transition` to call super then record): harmed =
  game over on this move, or a health bar drop. For each kind in `_touched(after)`
  record `(label, kind, "touch")` in the adverse book and update the net with
  `features(label, kind, shape_class, "touch", roles, health_bar)`; when nothing was
  touched record `(label, "self", "any")`. Log `harm` when harmed.
- `_harm_of(scene, action) -> float`: the max `harm_probability` over the kinds whose
  things the controlled body would overlap or touch after one gait step of `action`
  (`Kinematics.gaits`; when the mover has no gait for the action, use the `self` key).
- `_veto` override: the base reasons first; then `p = self._harm_of(scene, action)`: band
  certain returns reason `"adverse certain"`; band medium asks `Perseverance.allow(p,
  no_other_way=<no other hypothesis reaches the goal>, ev_gain=<this plan's ev minus the
  best alternative's>, trials_used=<this level instance's trials for the key>)` and
  returns `"adverse medium"` when refused, else None and logs `perseverance` (step,
  action, p, reason). Add `"adverse certain"` and `"adverse medium"` to the plan vetoes
  the segment honours.

### A7. The planner call

Copy the pool and deadline pattern of `_await_turn` into `_planner_turn(messages, seed,
max_steps, scene, available, wait_s)`: submit `chat(messages, seed)` to the pool, wait at
most `min(wait_s, time left in the move gate)`, parse with `parse_reply(text, scene,
available, max_steps=max_steps)`, count `completion.completion_tokens` into
`_move_tokens` and `_tokens_total`, log `planner_call` (step, call, seconds, tokens,
segment_len, skill, question, dropped). Failures cost the call only (log `exec_failed`).

### A8. The seven questions, computed

Q-BLOCK: the kinds whose cells lie on the shortest route to the goal target and their
roles. Q-REWARD: kinds with role collectible or named by an open goal, with the goal's
expected value. Q-HARM: the harm probability of the first step of the best hypothesis
and the step with the highest harm. Q-FORK: `newest_with_untried` summary. Q-CHECK: the
top flag by value per cost. Q-SAFE: whether the best hypothesis's rollout reaches its
goal with health left. Q-UNKNOWN: the first three frontier pairs as questions
(`tester.question` or the frontier text the base uses). Each answer is one line.

### A9. Level up, game over, close

- `_level_up` override: call super (it runs sleep and transfer), then consolidate: promote
  every fact with `state == "fact"` whose canonical key has no colour (every key from
  `canonical_key`) to the store as a `RuleRecord` (statement from `statement_for`,
  confirmations and contradictions from the fact's counts, harm from the adverse book),
  `store.record_completion(game, level, self._level_moves)`, `store.add_sequence(...)`
  with `self._level_start_moves`, `store.save()`, log `consolidate`; then reset the tree,
  the flags' level state, the level counters.
- `_game_over` override: call super; the tree keeps its forks (the level restarts from
  the same frame, spec 4.7c rule 7 marks the active branch failed); `fail_branch` on the
  active path; working memory stays.
- `close` override: `store.net_state = self._net.state()`, `store.save()`, log `longterm`
  (game_id, rules, sequences; `game` is an envelope key), then super.

### A10. Constants (module level, with comments)

`T_MOVE = 60.0`, `N_TOKEN = 4096`, `MIN_CALL_S = 5.0`, `SEGMENT_CAP = 12`,
`HORIZON_CAP = 200`, `TOP_GOALS = 3`, `HEALTH_BAR_MAX_DROP_RATE = 0.3`,
`HEALTH_BAR_MIN_MOVES = 10`, `NO_PROGRESS_MOVES = 40`.

## Part B: configuration, run, notebook, tests

- `config.py`: `MindConfig.planner: bool = False`; `out["PLANNER"] = replace(out["FLUID-LLM"],
  name="PLANNER", planner=True)`. `fallback_config("PLANNER")` is already `FLUID`.
- `submit.py`: `MindRun.__init__` gains `longterm: Path | None = None`; when `cfg.planner`
  it builds one `LongTermStore(longterm or log_dir / "longterm.json")` and loads it;
  `make_agent` builds `PlannerAgent(..., store=self.store)` when `cfg.planner` (else as
  today). `parse_args` gains `--longterm` (default None). `play_mind` passes it. The
  results record gains `"longterm"`.
- `scripts/kaggle_mind_notebook.py`: nothing but the `--config PLANNER` choice, which
  `CONFIGS` provides; confirm `tests/test_kaggle_mind_notebook.py` still passes and add a
  test that `--config PLANNER` builds with fallback `FLUID`.
- `docs/agent-submission.md`: a section "The planner agent (v2)" with the build and run
  commands.

## Tests: `tests/mind/test_planner_agent.py`

Use `tests/mind/fakegames.py` (`FakeGame`, `FakeClickGame`, `play`) and `vchat.scripted`
the way `tests/mind/test_llm_drive.py` does; read it first. Each test uses a `tmp_path`
`RunLog` and a store path under `tmp_path`.

1. A scripted reply with `facts` (the four moves as `move` claims on the player), a
   `reach` goal (player to target) and a `segment` of the right moves makes the agent win
   level 1 in at most the segment's length plus 2 moves; the log has `planner_call`,
   `segment` (source planner), and action records with `p_step`, `p_path`, `segment_id`.
2. A first reply whose segment walks into lava (the fake game's lava ends the game; give
   a claim that says so, or let a prior death have taught it) is rejected by the rollout
   with stop `death`; the second scripted reply is valid and is followed; the log shows
   two `planner_call` events on the same step and one `segment`.
3. A segment whose claimed move is wrong (a wall blocks it) breaks on the first real move;
   `handoff` with rule `H-b` follows on the next move; a fork with two paths (give two
   plans to the same goal) records `branch_failed` and a `return_route`.
4. A chat whose call sleeps past `T_MOVE` (inject a clock or a small `T_MOVE` via a
   module constant override) leaves a `gate_trip` event and the move is a fallback branch.
5. After a level win, the store file exists with one completion and at least one promoted
   rule; `consolidate` is logged with `typical`.
6. A second `PlannerAgent` on the same store gets `lessons` lines in the rendered
   briefing (capture the messages the scripted chat received).
7. The agent with `chat=None` plays `FakeGame` to a level win like `FluidAgent`
   (regression).

Gates: `ruff format` on changed files, `ruff check .`, `pyright` 0 errors, full pytest
green. Commit per part on `agent-v2` with the repo's message style.
