# Agent v2 build plan (from spec v2, 2026-10-31)

Spec: `docs/superpowers/specs/2026-10-31-agent-spec-v2.md` (Kameron's decisions in its
section 6). Code map: `docs/superpowers/plans/2026-10-31-v2-code-map.md`.
Deadline (Kaggle CLI, 2026-10-31 01:20 UTC): competition closes 2026-11-02 23:59 UTC.
One submission per day; slots left: 10-31, 11-01, 11-02.

## 1. Goal and scope cut

Goal: the planner agent of spec v2, runnable on Kaggle with the existing Qwen3.6-27B-FP8
server, finishing levels in few moves. The score is `min(human/agent, 1)^2 * 100` per
level, so finishing a level with 10x the human moves earns about 1%. Fewer moves per
finished level is the whole game.

Built in this pass (spec sections in brackets):

- Rule probability, bands, hypothesis/fact status, retirement [4.4, 4.2 status rules].
- Rollout with per-step probability, path probability by the chain rule, harm stops [4.8a, 4.4c].
- Hypothesis list ranked by path probability; expected-value choice among the planner's
  segment and the search plans [4.7a, 4.7b, 3.4].
- Horizon per level from the agent's own completions [4.7 horizon rule].
- Fork tree with return routes over known moves [4.7c, 3.6].
- Harm definition, adverse book with Beta estimates, inhibition network (online logistic
  model over canonical features), perseverance with logged reasons, bands 0.8/0.3 [4.10].
- Per-move time gate and per-move LLM token gate, logged [4.10d].
- Self record, focus record, controller with ordered handoff rules [4.5].
- Curiosity flags with value and cost, in the briefing [4.11].
- Briefing with the twelve lists and the question library pre-answered [4.9a, 4.9c].
- Planner reply contract: segment (up to horizon), optional skill call, one question [4.9f].
- Skills S-AROUND, S-COLLECT, S-EXPLORE [4.9d]; S-ATTACK and S-WAIT only if health
  modelling lands (stretch).
- Executor: segment checked step by step; miss that matters stops it [4.9e].
- Level and game consolidation to a long-term store that persists across games in a run;
  loaded at game start as hypotheses in words for the briefing [4.12b, 4.12c, 4.2].
- Move record fields, level and game summaries, calibration and moves-vs-typical reports [4.13].
- Explorer kept as the fallback method; hybrid stall rule becomes controller rule H-e.

Deferred (named in the gaps report after the Kaggle run): inverted index with real-time
cut [4.3] beyond probability-ranked lookup; learned retrieval scorer [4.3]; JPS [4.6d];
Q-learning value estimate [4.6f]; shared BFS refactor [4.6a] (truncation flag only);
first-contact evaluation [4.12d] as a report over the run order (cheap version included).

## 2. Architecture

New class `PlannerAgent(FluidAgent)` in `arcagent/mind/planner_agent.py`, config
`PLANNER` (`fluid=True, llm=True, planner=True`). Without a chat it plays like
`FluidAgent`. With a chat:

1. Observe, learn, check the last prediction (FluidAgent as is).
2. Controller reads the focus record and the signals, in its order, and names the method:
   `follow` (segment step ready and valid), `replan`, `return_to_fork`, `explore`
   (fallback), `reset`.
3. On `replan`: build hypotheses (search toward each open goal, bounded), roll each out
   with probabilities, compute expected value; build the briefing (lists, pre-answered
   questions, flags, forks, self and focus); ask the planner for one segment or a skill
   call; roll the segment out; if unsafe or unknown at step k, ask once more with the
   step noted; if unsafe again, take the best code hypothesis by expected value; if none,
   `explore`.
4. Execute the segment one move per `choose`, predicting each step; a miss that matters
   stops the segment and records the branch as failed in the fork tree.
5. Level up: consolidate (promote rules, write the sequence, record typical moves).
   Game end: write the long-term store.

Reuse (from the code map): `turn.build_messages`/`take_turn` pattern and `vchat`;
`contract.parse_reply` extended with `skill` and `question`; `game_model.play_out`
extended to `rollout` with probabilities; `DriveAgent._await_turn` pattern for the pool
and deadline; `fluid_sleep` for level consolidation; `ExplorerPolicy.Graph` for return
routes; `BudgetReader` counters for the health bar.

Decisions recorded here (Kameron may change them):

- D-A. One planner call per segment, with the question library answered inside the
  briefing, instead of a multi-call tool loop. Reason: a 27B model on Kaggle costs
  seconds per call and the time gate is per move. The reply may still ask one question,
  answered in the next briefing.
- D-B. Long-term rules are given to the planner in words (statement, probability,
  games seen) and are not auto-seeded into `FactMemory`. Reason: the canonical key
  across games is kind and role, and auto-seeding a wrong fact costs moves; the planner
  can weigh a loaded rule against the scene. Seeding is a later step.
- D-C. Expected value: `EV = P_path * 100 * eff - P_harm * restart_moves`, with
  `eff = min(1, typical / expected_moves)^2` where `typical` is the level's typical moves
  (horizon rule) or the cap before the first completion. Restart moves = moves from the
  level start to the current state (known) plus the segment length.
- D-D. Harm when no health bar: implicit health is 1, so any predicted death is certain
  band harm.
- D-E. The inhibition network is an online logistic regression over canonical features
  (action label role, subject shape class, context tag, kind role flags, counter present)
  with a few hundred weights, trained from every observed transition; used when the
  adverse rule has fewer than 5 confirmations.
- D-F. A fact's probability reads its lifetime miss count, so a fact re-seeded after a
  break starts below 0.5 and a twice re-seeded fact starts retired (1 hit, 4 misses, p
  0.2857). This is kept on purpose: the key has been wrong four times, and the rollout's
  chain rule prices any plan that leans on it low, so the planner is steered away from it
  without a hard rule. Promotion to "fact" uses a run counter of hits since the last miss
  (WP1 review, 2026-10-31), not a proxy on the miss streak.

## 3. Planner reply contract (v2)

JSON object. Keys, all optional unless stated:

```
{"model": "<under 1500 chars, the planner's own model of the game>",
 "facts": [ ...as DRIVE... ],
 "goals": [ ...as DRIVE... ],
 "segment": [{"action": "A3" | "click #12", "expect": "..."}],   // up to horizon steps
 "skill": {"name": "S-AROUND" | "S-COLLECT" | "S-EXPLORE", "args": {...}},  // or segment
 "question": "Q-BLOCK" | "Q-REWARD" | "Q-HARM" | "Q-FORK" | "Q-CHECK" | "Q-SAFE" | "Q-UNKNOWN",
 "labels": [{"object": 12, "label": "door"}],   // agent-made labels for kinds (4.11a)
 "notes": "one line"}
```

`segment` and `skill` are alternatives; both present keeps `segment`. `plan` is accepted
as an alias of `segment` for the DRIVE prompt shape. Invalid parts are dropped with a
reason in `Reply.dropped`, as today.

## 4. Work packages, file ownership, interfaces

Each package is one implementer agent in its own worktree branch off `agent-v2`.
Packages in the same wave touch disjoint files. Each package: tests first, ruff, pyright
strict, full suite green, one or more commits on its branch, report back the branch name
and the public interface it produced.

Wave 0 (sequential, small): `facts.py` probability.

- WP1 `facts.py` (+ `tests/mind/test_facts.py`). Add `Fact.probability` (Beta mean,
  `(hits + 1) / (hits + misses + 2)`), `Fact.band` in {"certain", "medium", "low"} with
  edges 0.8 and 0.3, `Fact.state` in {"hypothesis", "fact", "retired"} derived from
  probability and the existing counts: fact when hits since last refinement >= 5 with no
  miss since and band certain; retired when band low; else hypothesis. Keep `status`
  (solid, thin, broken) unchanged for every existing caller. Add `FactMemory.probability(key)`
  and `FactMemory.ranked(keys)` (sorted by probability desc, four decimals, then key).
  `lookup` keeps its order but, among sibling candidates, picks the highest probability
  (today: most hits) with ties by key. Constants: `PROMOTE_HITS = 5`, `BAND_CERTAIN = 0.8`,
  `BAND_LOW = 0.3`, `PRIOR_A = 1`, `PRIOR_B = 1`.

Wave 1 (parallel, disjoint files):

- WP2 `arcagent/mind/adverse.py` (+ `tests/mind/test_adverse.py`). `AdverseBook`:
  `record(label, subject_kind, context, harmed: bool)`, `probability(label, subject_kind,
  context) -> float` (Beta), `confirmations(...)`, `band(...)`. `InhibitionNet`: online
  logistic regression over a fixed feature vector built by `features(label, subject_kind,
  shape_class, context, roles: frozenset[str], health_bar: bool) -> np.ndarray`
  (one-hot per category with hashing to a fixed width, bias term), `update(x, harmed)`,
  `predict(x) -> float`; learning rate 0.1, L2 0.001, weights float64, width 256.
  `harm_probability(book, net, ...)`: the rule's Beta when confirmations >= 5 else the
  net's output. `Perseverance.allow(p: float, no_other_way: bool, ev_gain: float,
  trials_used: int) -> tuple[bool, str]` with the band rules of spec 4.10c (certain never,
  low always, medium needs a reason, trials cap 3). Pure, numpy only.
- WP3 `arcagent/mind/longterm.py` (+ `tests/mind/test_longterm.py`). `RuleRecord`
  dataclass (id, kind, statement, canonical_key, effect, probability, confirmations,
  contradictions, state, scope, games_seen, created_game, created_level, last_revised,
  harm, source). `SequenceRecord` (id, goal_id, moves as list of [id, x, y], level,
  runs, successes, deterministic). `LongTermStore(path)`: `load()`, `save()` (atomic
  write, file lock via `os.replace` and a `.lock` file with retry), `promote(record)`,
  `sequences_for(goal_kind)`, `rules_for(features) -> list[RuleRecord]` (match on
  canonical key fields, ranked by probability), `typical_moves(game, level)` and
  `record_completion(game, level, moves)`, `lessons(limit) -> list[str]` (rule statements
  in words for the briefing). Canonical key: `(action_role, shape_class, context_tag,
  effect_kind)`, never a color index. `canonical_key(fact_key, effects, scene) -> tuple`.
  Thread-safe for concurrent games in one run (a lock around load-modify-save).
- WP4 `arcagent/mind/rollout.py` (+ `tests/mind/test_rollout.py`). `rollout(scene,
  actions, mem, goals, protect, *, horizon, health, adverse_fn) -> Rollout` where
  `Rollout` has `scenes`, `step_probabilities: tuple[float, ...]`, `path_probability`,
  `reached: Goal | None`, `unknown_at: int | None`, `harm: tuple[tuple[int, float], ...]`
  (step, probability), `stop: Literal["goal", "unknown", "death", "horizon", "end"]`,
  `expected_moves: float`, `assumed: int`. Step probability = product of the distinct
  facts' probabilities used at that step (`Prediction.sources` and `used`, via
  `mem.probability`); a borrowed fact (`lent`) uses the sibling's probability; a step with
  no fact for a protected thing is `unknown`. Health: starts at `health`; a predicted
  `game_over` effect is death; an adverse probability from `adverse_fn(label, kind,
  context)` is recorded per step and, when band certain, is death. `expected_moves` =
  sum over steps of 1 / step probability (expected tries), capped at horizon times 3.
  Also `hypotheses.py`: `Hypothesis` (goal, plan, path_probability, expected_moves,
  harm_probability, expected_value, source in {"planner", "search", "skill"}),
  `expected_value(path_p, expected_moves, typical, harm_p, restart_moves) -> float` per
  D-C, `rank(hyps) -> list[Hypothesis]` (by expected value, ties by four-decimal path
  probability then id).
- WP5 `arcagent/mind/plantree.py` (+ `tests/mind/test_plantree.py`). `PlanTree` per
  level instance: `open_fork(state_key, paths: list[tuple[Action, ...]], goal_id,
  moves_here: tuple[Action, ...]) -> fork_id`, `fail_branch(fork_id, path_index)`,
  `untried(fork_id)`, `newest_with_untried() -> fork | None`, `return_route(graph,
  current_key, fork_key) -> list[Action] | None` using breadth-first search over the
  explorer `Graph`'s known transitions (code map names the API), `reset_level()`,
  `summary()` for the briefing. Log event names: `fork_open`, `branch_failed`,
  `return_route`, `replan_root`.
- WP6 `arcagent/mind/controller.py` + `arcagent/mind/selfrecord.py` (+ tests).
  `SelfRecord.from_state(available_actions, controlled_kinds, lone_mover) -> SelfRecord`
  with `i="agent"`, `click: bool` (6 available), `my_character: str | None`;
  `FocusRecord` (level, goal_id, goal_status, target, mode, budget_left, verdict,
  fallbacks_in_row, segment_left, tokens_this_move, seconds_this_move). `Controller`
  with ordered rules H-a (gate tripped) → H-b (segment broke) → H-e (fallback count
  >= STALL_MOVES) → H-c (no progress for N=40 moves on the active goal) → H-d (budget
  below segment length + 2): `decide(focus, signals) -> Decision(method, rule, reason)`
  with methods in {"follow", "replan", "return_to_fork", "explore", "keep"}. Every
  decision is returned with the rule name for logging. `arcagent/mind/flags.py`:
  `FlagBook` keyed by kind: `note_scene(scene, known_pairs)` sets `check` for new looks
  and unknown pairs, `checked(kind, outcome_keys)`, `label(kind, text)`, `value_cost(kind,
  distance, unknown_pairs) `, `top(n)` for the briefing.
- WP7 `arcagent/mind/briefing.py` + `contract.py` extension (+ tests). `Briefing`
  dataclass with the twelve lists of spec 4.9a as tuples of strings, plus `questions:
  tuple[tuple[str, str], ...]` (question id, computed answer) and `render(briefing,
  cap_chars) -> str` and `PLANNER_SYSTEM_PROMPT` asking for the v2 reply contract
  (section 3 above). `contract.parse_reply` gains `segment` (alias `plan`), `skill`
  (name in the skill list, args dict), `question` (one of the ids), `labels` (object id
  to text), kept in new `Reply` fields `skill: SkillCall | None`, `question: str`,
  `labels: tuple[tuple[int, str], ...]`; `plan` keeps its meaning (the segment). All
  existing tests keep passing.
- WP8 `arcagent/mind/skills.py` (+ tests). `Skill` protocol: `trigger(ctx) -> bool`,
  `steps(ctx) -> list[Action] | None`. `S_AROUND` (route to the goal avoiding harm-role
  kinds' cells using `Kinematics.reachable` with those cells blocked), `S_COLLECT`
  (`route.route` over collectible-role things, then the moves between them from
  `Kinematics.field`), `S_EXPLORE` (the top flag by value per cost, moves to touch or
  click it). `ctx` is a small dataclass (scene, mem, kin, goals, roles: dict[kind, role],
  flags, can_click). Each skill returns `None` when it does not apply and logs why.
- WP9 `runlog.py` fields + `scripts/mind_metrics.py` reports (+ tests). Action record
  gains `self`, `focus`, `hypothesis_id`, `segment_id`, `fork_id`, `planner_call`,
  `p_step`, `p_path`, `gate` ({seconds, tokens, trip}). New events: `segment`
  (step, id, source, length, p_path, ev), `handoff` (step, rule, method, reason),
  `harm` (step, kind, label), `consolidate` (level, promoted, sequences, typical),
  `longterm` (game, rules, sequences). `mind_metrics.py` adds per-game: calibration bins
  (p_step vs matched), moves per completed level against typical, gate trips, handoffs
  by rule, planner calls and tokens. Docstring lists stay the source of truth.

Wave 2 (sequential, one implementer): WP10 integration.

- WP10 `arcagent/mind/planner_agent.py`, `config.py` (`PLANNER`), `submit.py`
  (`make_agent` builds `PlannerAgent` when `cfg.planner`; `--longterm` path flag
  default `<log_dir>/longterm.json`), `scripts/kaggle_mind_notebook.py` (config
  `PLANNER`, fallback `FLUID` with `--hybrid-moves 150`), tests in
  `tests/mind/test_planner_agent.py` on `fakegames` with `vchat.scripted` replies: the
  agent follows a valid segment to a level win; a segment the rollout rejects is re-asked
  once with the step named; a miss that matters stops the segment and the fork tree
  records it; the gates log; consolidation writes the store; a second game in the same
  run reads the store's lessons into its briefing.

Wave 3: whole-branch review (adversarial), fixes, merge to `agent-v2`.

## 5. Gates

Every package: `ruff check .`, `ruff format --check .`, `pyright` (0 errors), `pytest`
(full suite green). Commit per package. No `Co-Authored-By` trailer; end commit messages
with the `Claude-Session` line used in this repo's recent commits.

## 6. Kaggle procedure

1. `python scripts/kaggle_mind_notebook.py --games all --config PLANNER --hybrid-moves 150`
2. `kaggle kernels push -p build/kaggle-mind`, then `kaggle kernels status hgenix20/arc3-mind-agent`.
3. On completion, `kaggle kernels output hgenix20/arc3-mind-agent -p bench/evals/2026-10-31-planner-v1`.
4. `python scripts/mind_metrics.py bench/evals/2026-10-31-planner-v1/logs`; read the logs
   for the gaps report (which levels, moves vs typical, where the planner's segments broke,
   calibration).
5. Submit with `kaggle competitions submit ...` on Kameron's standing instruction for this
   agent ("Once complete submit it"), after the interactive eval completes and its parquet
   exists. Record the submission id in `docs/experiment-log.md`.

## 7. Gaps report

`docs/superpowers/plans/2026-10-31-v2-gaps.md`: per game, levels, moves vs typical,
planner calls, segments broken and why, handoffs, harm events, calibration; then the gaps
in learning and intelligence the logs show, each tied to a spec section.
