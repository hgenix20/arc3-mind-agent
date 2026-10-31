# Wave-1 public interfaces (from the implementers' reports, 2026-10-31)

Read the module docstrings for the final word; these are the implementers' own summaries plus their concerns.


## wp2 (v2/wp2-adverse)

Module arcagent.mind.adverse (spec 4.10a, 4.10b, 4.10c, 4.4). Pure, no logging, no I/O; imports numpy and four constants plus the Band type from arcagent.mind.facts.

Types:
- AdverseKey = tuple[str, str, str]: (action label, subject kind, context); the kind is a Thing.kind string the caller passes.
- Features = npt.NDArray[np.float64]: shape (width,), the last entry is the bias.
- Band = Literal["certain", "medium", "low"]: re-exported from facts so facts and adverse rules share one band type.
- Source = Literal["rule", "net"]: where harm_probability's estimate came from.
- Reason = Literal["certain", "low", "trials exhausted", "no other way", "expected value", "no reason"]: the second element Perseverance.allow returns, for the log.

Constants: WIDTH = 256 (feature width), RULE_TRIALS = 5 (trials after which the rule's own probability is used), MAX_KEYS = 5000 (book bound), MAX_TRIALS = 3 (medium-band trials per rule per level instance), MAX_ROLES = 16 (roles one situation may carry), LEARNING_RATE = 0.1, L2 = 0.001. Re-exported from facts: PRIOR_A = 1, PRIOR_B = 1, BAND_CERTAIN = 0.8, BAND_LOW = 0.3.

Functions:
- band_of(p: float) -> Band: certain at p >= 0.8, low at p < 0.3, else medium; ValueError when p is outside [0, 1] (NaN included). Shared by AdverseBook.band and Perseverance.allow.
- features(label: str, kind: str, shape_class: str, context: str, *, roles: frozenset[str], health_bar: bool) -> Features: binary vector of shape (WIDTH,), float64; each token ("label=..", "kind=..", "shape=..", "ctx=..", one "role=.." per role, "hb=0|1") sets index zlib.crc32(token.encode()) % (WIDTH - 1) to 1.0; index WIDTH - 1 is the bias, always 1.0; deterministic across processes; ValueError for more than MAX_ROLES roles. NOTE: roles and health_bar are keyword-only (see concerns).
- harm_probability(book: AdverseBook, net: InhibitionNet, key: AdverseKey, x: Features) -> tuple[float, Source]: (book.probability(key), "rule") when book.evidenced(key), else (net.predict(x), "net").

Classes:
- AdverseBook(max_keys: int = MAX_KEYS): per-key tallies. record(key: AdverseKey, harmed: bool) -> None counts one trial and one harm when harmed; when a new key arrives at capacity it evicts the key with the fewest trials (ties by key order) and returns normally. trials(key) -> int and harms(key) -> int (0 for unknown). probability(key) -> float = round((harms + PRIOR_A) / (trials + PRIOR_A + PRIOR_B), 4), 0.5 for an unknown key. band(key) -> Band. evidenced(key) -> bool when trials(key) >= RULE_TRIALS. keys() -> list[AdverseKey] sorted. ValueError for max_keys < 1.
- InhibitionNet(width: int = WIDTH, lr: float = LEARNING_RATE, l2: float = L2): online logistic regression, weights float64 zeros (fresh net predicts 0.5). Attributes width, lr, l2, n_updates: int. predict(x: Features) -> float = sigmoid(w . x) rounded to 4 decimals, computed in an overflow-safe form. update(x: Features, harmed: bool) -> None does one SGD step on the logistic loss with L2 on every weight but the bias and increments n_updates. state() -> dict[str, list[float] | int] = {"weights": [...], "n_updates": n}. load(state: Mapping[str, object]) -> None validates (list of numbers of length width, finite, non-negative int count) before writing; a ValueError leaves the net unchanged. predict and update raise ValueError when x.shape != (width,). ValueError for width < 2, lr <= 0, l2 < 0.
- Perseverance: stateless. Perseverance.allow(p: float, *, no_other_way: bool, ev_gain: float, trials_used: int, max_trials: int = MAX_TRIALS) -> tuple[bool, Reason] (staticmethod, callable on the class or an instance): p >= 0.8 -> (False, "certain"); p < 0.3 -> (True, "low"); medium: trials_used >= max_trials -> (False, "trials exhausted"); no_other_way -> (True, "no other way"); ev_gain > 0 -> (True, "expected value"); else (False, "no reason"). ValueError for negative trials_used or max_trials, or p outside [0, 1].

The caller owns the harm verdict (spec 4.10a) and the learning loop: after each observed transition it calls book.record(key, harmed) and net.update(features(...), harmed); nothing in this module writes to either on its own.

### Implementer concerns
1. features() signature differs from the brief's literal form: roles and health_bar are keyword-only (features(label, kind, shape_class, context, *, roles=..., health_bar=...)). Reason: ruff's PLR0917 caps positional arguments at five and the repo otherwise suppresses it per site; keyword-only arguments avoid a suppression and keep the two non-string inputs from being mixed up. The integrator (WP10) must pass them as keywords.
2. The prior and band edges (PRIOR_A, PRIOR_B, BAND_CERTAIN, BAND_LOW) and the Band type are imported from arcagent.mind.facts rather than redefined, so there is one source of truth for the spec 4.4 numbers; adverse therefore depends on facts (and through it on modes, objects, core.types). The module is still numpy-only in third-party terms.
3. The harm verdict of spec 4.10a (health bar lower, or death / level end / game end when there is no bar) is not a function here; the brief defines the inputs as harmed: bool, so the caller must compute it. WP10 should own that rule, or a later package should add it.
4. AdverseBook has no state()/load(); the brief did not ask for it. Spec 4.10b item 5 says the network is retrained on the rule store's evidence, so WP3 or WP10 will need the book's tallies persisted (or rebuilt from the long-term store) for that to hold across games.
5. The doubled -q suppressed pytest's summary line in the required log; the pass count above comes from counting markers in the log, not from a printed total. A rerun without the extra -q would print the usual "N passed" line.


## wp3 (v2/wp3-longterm)

Module arcagent/mind/longterm.py (standard library plus facts.Effect/FactKey/State and the facts constants BAND_CERTAIN, BAND_LOW, PRIOR_A, PRIOR_B, PROMOTE_HITS, SPAWN; kinds.base_sig; core.types.COORD_ACTION_ID).

Type aliases:
- RuleKind = Literal["object", "action", "event", "experience"]
- RuleState = facts.State, i.e. Literal["hypothesis", "fact", "retired"]
- Scope = Literal["working", "long"]
- Source = Literal["observed", "loaded", "label"]
- CanonicalKey = tuple[str, str, str, str]  (action_role, shape_class, context_tag, effect_kind)

Constants: STORE_VERSION = 1; MAX_RULES = 5000; MAX_SEQUENCES = 2000; MAX_COMPLETIONS_PER_LEVEL = 50; MAX_GAMES_SEEN = 500 (game ids kept per rule, oldest dropped); STATEMENT_MAX_CHARS = 120; CLICK_ROLE = "click"; NO_HARM = "none".

Dataclasses (slots, mutable, every field required, no defaults):
- RuleRecord(id: str, kind: RuleKind, statement: str, canonical_key: CanonicalKey, effect: str, probability: float, confirmations: int, contradictions: int, state: RuleState, scope: Scope, games_seen: list[str], created_game: str, created_level: int, last_revised: float, harm: str, source: Source): one rule in the shared record shape of spec 4.2.
- SequenceRecord(id: str, goal_id: str, moves: list[list[int]], game: str, level: int, runs: int, successes: int, deterministic: bool): a move sequence ([id, x, y] triples) that served a goal.

Functions:
- rule_probability(confirmations: int, contradictions: int) -> float: Beta mean (c + 1) / (c + x + 2) to four decimals; ValueError on a negative count.
- rule_state(probability: float, confirmations: int, contradictions: int) -> RuleState: retired below 0.3; fact at or above 0.8 with confirmations >= 5 and contradictions == 0; else hypothesis.
- role_of(label: str) -> str: the action role of a fact label; "A6", "A6:none" and "A6:<sig>@<ring>" give "click"; "A3", "event", "any" stay.
- shape_of(subject: str) -> str: the colour-free shape of a sig or kind name ("c3:<shape>" and "kc3:<shape>" give "<shape>", "c3:*" gives "*"); "self", "any", "spawn" and prefix-free subjects stay.
- tag_of(context: str) -> str: "touch:<c>" and "near:<c>" give "touch" and "near"; a "c<colour>:" prefix is dropped; other contexts stay.
- canonical_key(fact_key: FactKey, effects: Sequence[Effect]) -> CanonicalKey: (role_of(label), shape_of(subject), tag_of(context), first effect's kind or "none").
- rule_id(key: CanonicalKey) -> str: stable id, "r" plus twelve hex digits of the key's SHA-1.
- statement_for(fact_key: FactKey, effects: Sequence[Effect]) -> str: one plain sentence from the canonical key and the first effect, e.g. "A3 moves a ab12cdef01 thing by (1, 0)", "clicking a thing recolours it", "any action ends the game for a ab12cdef01 thing while touching something"; no colour index; cut to 120 chars with "...".
- effect_text(effects: Sequence[Effect]) -> str: colour-free words for RuleRecord.effect, e.g. "move (1, 0); reshape ff00ff00ff; recolor"; "none" when empty.

class LongTermStore:
- __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None
- attributes: path: Path; lock: threading.RLock (every public method holds it; hold it yourself across load, changes and save to make them one step); rules: dict[CanonicalKey, RuleRecord]; sequences: dict[str, SequenceRecord]; completions: dict[str, dict[int, list[int]]] (game -> level -> moves per completion); net_state: dict[str, object] | None (opaque, for the inhibition network); load_error: str | None (why the last load treated the file as corrupt, None otherwise).
- load(self) -> None: replaces the content with the file's; missing file -> empty store; unreadable or malformed file -> moved to "<name>.bad", store empty, load_error set; other OSErrors raise.
- save(self) -> None: json.dumps to a temp file in the same directory, fsync, os.replace; creates parent dirs; temp removed on failure.
- promote(self, record: RuleRecord) -> RuleRecord: insert (scope long, probability and state recomputed, games_seen united with created_game, last_revised = clock()) or merge by canonical_key (confirmations and contradictions summed, games_seen united, probability and state recomputed, harm kept once not "none", id/statement/created_* of the stored record kept); returns the stored record; drops the lowest (probability, confirmations, id) when over MAX_RULES; ValueError when a key part still carries a colour.
- rules_for(self, *, action_role: str | None = None, shape_class: str | None = None, context_tag: str | None = None, effect_kind: str | None = None, limit: int = 20) -> list[RuleRecord]: key-field match (None matches anything), by probability desc then id.
- lessons(self, limit: int = 12) -> list[str]: "<statement> (p=0.8571, 3 games)" ("1 game" for one), by probability desc, games count desc, id.
- record_completion(self, game: str, level: int, moves: int) -> None: appends, keeps the last MAX_COMPLETIONS_PER_LEVEL; ValueError on a negative level or moves.
- typical_moves(self, game: str, level: int) -> float | None: mean over that game's completions of the level (the caller rounds up for the horizon).
- typical_for_level(self, level: int) -> float | None: pooled mean over every game's completions of that level index.
- add_sequence(self, record: SequenceRecord) -> None: insert, or sum runs and successes into the record with the same id (its moves and goal stay); drops the fewest (successes, runs, id) when over MAX_SEQUENCES; ValueError when successes > runs, a count is negative, or a move is not a 3-item list.
- sequences_for(self, goal_id: str, limit: int = 5) -> list[SequenceRecord]: by successes desc then id.

JSON file: {"version": 1, "rules": [...asdict, sorted by id...], "sequences": [...], "completions": {game: {"<level>": [moves...]}}, "net_state": ... or null}; canonical_key is a list in the file and a tuple in memory; completion level keys are strings in the file and ints in memory.

### Implementer concerns
1. Signature deviation: canonical_key and statement_for take effects: Sequence[Effect] (facts.Effect, the repo's own type) instead of Sequence[object], because the statement reads dx, dy and sig. Fact.effects tuples pass unchanged.
2. Information loss the brief prescribes: a click label becomes "click" and a "self" subject stays "self", so ("A6:c3:<shape>@2", "self", "any") -> ("click", "self", "any", "recolor") and every "clicking a thing recolours it" rule in every game merges into one key. The brief's example statement "clicking a <shape> thing recolours it" only arises when the subject is a sig; keeping the clicked shape for self subjects would be a one-line change in canonical_key (fold the clicked sig's shape into shape_class). Tests pin the literal behaviour, so a change there means updating them.
3. promote adds counts. WP10 must pass the evidence gathered since the last promote of the same rule (deltas), or confirmations double-count at every level consolidation (spec 4.12c writes at each level completion and at game end).
4. promote raises ValueError on a key part that still carries a colour ("c<digits>:" prefix, or "touch:<n>"/"near:<n>"), enforcing D1 at the boundary; keys must come from canonical_key.
5. lessons() pluralizes ("1 game" vs "<n> games"); the brief's literal format reads "<n> games".
6. Public names beyond the brief: rule_id, effect_text, role_of, shape_of, tag_of, rule_probability, rule_state, MAX_GAMES_SEEN, CLICK_ROLE, NO_HARM, LongTermStore.load_error, LongTermStore.lock and the clock keyword. The helper names avoid shadowing the rules_for keyword parameters action_role/shape_class/context_tag, which the brief fixes.
7. rule_state reads "no contradictions since the last revision" as lifetime contradictions == 0, as the brief directed; the store tracks no refinement.
8. load treats any structural problem (wrong version, duplicate key, bad or missing field, non-numeric level) as corrupt: the whole file moves to .bad and the store starts empty; nothing is partially recovered. A newer file version would also be moved aside.
9. git add printed "LF will be replaced by CRLF" for both files; that is the repo's autocrlf setting at work, nothing to fix.


## wp4 (v2/wp4-rollout)

arcagent/mind/rollout.py (C:/arc-agent-wp4/arcagent/mind/rollout.py):
- RolloutStop = Literal["goal", "unknown", "death", "horizon", "end"]: why a rollout ended.
- AdverseFn = Callable[[Scene, Action], float]: harm probability of one action from one scene.
- PRIOR_STEP = 0.5 (from facts.PRIOR_A/PRIOR_B): step probability when the prediction used no fact (spec 4.4c rule 5).
- MIN_STEP_P = 0.05: floor under a step probability when counting expected tries.
- EXPECTED_MOVES_HORIZONS = 3: expected moves are capped at this many horizons.
- DECIMALS = 4: probabilities are reported to four places.
- @dataclass(frozen=True, slots=True) class Rollout: scenes: tuple[Scene, ...]; step_probabilities: tuple[float, ...]; path_probability: float; reached: Goal | None; unknown_at: int | None; harm: tuple[tuple[int, float], ...]; stop: RolloutStop; expected_moves: float; assumed: int; deaths_at: int | None. One scene and one probability per recorded step (same length); path_probability is their product to 4 decimals (1.0 for an empty path); harm holds (step, probability) pairs above zero; deaths_at is the step index at which health reached zero.
- step_probability(mem: FactMemory, used: Iterable[FactKey]) -> float: product of the distinct used facts' mem.probability(key) to 4 decimals, PRIOR_STEP when used is empty (public so the live executor can price a single predicted step the same way).
- rollout(scene: Scene, actions: Sequence[Action], mem: FactMemory, goals: Sequence[Goal], protect: frozenset[str], *, horizon: int, health: int = 1, adverse: AdverseFn | None = None, rule: RuleFn | None = None) -> Rollout: predicts up to min(len(actions), horizon) steps with predict(assume_still=protect) and the rule exactly as game_model.play_out applies it; stops "unknown" at i when no scene and no rule changes (nothing recorded for that step), "death" when health reaches zero (a Prediction.game_over or an adverse value >= facts.BAND_CERTAIN each cost one health; the fatal scene is still appended), "goal" when a goal holds (no grid, as play_out); after the loop "horizon" if the horizon cut the plan, else "end". adverse(scene_before, action) is clamped to [0, 1] and recorded when > 0. expected_moves = sum of 1 / max(p_step, 0.05) over recorded steps, capped at 3 * horizon, 2 decimals. assumed = total assumed-still pairs. Raises ValueError for horizon < 0 or health < 1. Pure apart from the rule call; logs nothing.

arcagent/mind/hypotheses.py (C:/arc-agent-wp4/arcagent/mind/hypotheses.py):
- Source = Literal["planner", "search", "skill"].
- LEVEL_REWARD = 100.0; MIN_EXPECTED_MOVES = 1.0; DECIMALS = 4; ID_CHARS = 10.
- @dataclass(frozen=True, slots=True) class Hypothesis: id: str; goal_id: str; source: Source; plan: tuple[Action, ...]; path_probability: float; expected_moves: float; harm_probability: float; expected_value: float; stop: str.
- expected_value(path_p: float, expected_moves: float, typical: float, harm_p: float, restart_moves: float) -> float: plan decision D-C, eff = min(1, typical / max(expected_moves, 1)) ** 2, value = path_p * 100 * eff - harm_p * restart_moves, 4 decimals; raises ValueError when typical <= 0.
- hypothesis_id(goal_id: str, source: str, plan: Sequence[Action]) -> str: first 10 hex chars of SHA-1 over goal id, source and the plan's (id,x,y) triples; deterministic.
- from_rollout(goal_id: str, source: Source, plan: Sequence[Action], ro: Rollout, *, typical: float, restart_moves: float) -> Hypothesis: id via hypothesis_id, plan as a tuple, path_probability/expected_moves/stop copied from the rollout, harm_probability = max over ro.harm (0.0 when none), expected_value = expected_value(path_p, ro.expected_moves, typical, harm_probability, restart_moves) where path_p is ro.path_probability when ro.reached is not None, else 0.0 (see concerns).
- rank(hyps: Sequence[Hypothesis]) -> list[Hypothesis]: expected_value desc, then round(path_probability, 4) desc, then id asc.

### Implementer concerns
1. from_rollout gives the reward term only when the rollout reached a goal (ro.reached is not None); a path cut by the horizon, stuck at an unknown step or ended without a goal is priced at -harm_probability * restart_moves while its path_probability still reports the recorded steps. The brief did not state this; I took it from spec 4.7b rule 2 (reward is the value of the goals the path reaches), because otherwise a long planner segment that reaches nothing would outrank a shorter search plan that completes the level. If WP10 wants the literal formula on every hypothesis, drop the reward_p line in from_rollout (one line, one test to flip: test_from_rollout_without_a_reached_goal_earns_no_reward).
2. The "unknown" stop records nothing for the unknown step, so a rollout that is unknown at step 0 has path_probability 1.0 (empty product) and expected_moves 0.0. The integrator should read stop/unknown_at and not the probability alone for such a rollout.
3. Two boundary validations the brief did not ask for: rollout raises ValueError for horizon < 0 or health < 1, and expected_value raises ValueError for typical <= 0. Both are documented and tested; remove them if the integrator prefers a clamped result.
4. The adverse probability is clamped to [0, 1] before use and recording; a NaN becomes 0.0.
5. Death accounting continues the simulation from the predicted scene after a non-fatal death (health > 1), as the brief specified; the real game would reset the level, so the scenes after a death are a model simplification.
6. git printed LF-to-CRLF warnings on add for the four new files (repo autocrlf setting); the committed content is unaffected.


## wp5 (v2/wp5-plantree)

Module arcagent/mind/plantree.py (standard library plus arcagent.core.types.Action and arcagent.mind.types.ActionKey = tuple[int, int, int]). Logs nothing.

Type aliases and constants:
- PathStatus = Literal["untried", "active", "failed", "achieved"]: status of one path on a fork.
- FinalStatus = Literal["failed", "achieved"]: the two statuses that never change again.
- MAX_FORKS = 200: forks kept per level instance; beyond it the oldest (lowest created) is dropped.
- MAX_PATHS = 16: paths kept per fork; later ones are ignored.
- MAX_MOVES_HERE = 20_000: cap on moves_here (the game's action cap, so never hit in play).
- ROUTE_LIMIT = 20_000: default visit limit of return_route.
- SUMMARY_LINES = 5: default line count of summary().

@dataclass(slots=True) class Fork: id: str; state_key: bytes; goal_id: str; paths: list[tuple[Action, ...]]; status: list[PathStatus]; moves_here: tuple[Action, ...]; created: int. One fork record; paths and status run in parallel; moves_here is the move list from level start as first recorded; created is the tree's running counter (higher = newer). Returned by reference; change it only through PlanTree methods.

class PlanTree:
- __init__(self) -> None: empty tree. Public attribute misses: int, requests ignored because the fork id, index or status change was invalid (lifetime tally, never reset).
- open_fork(self, state_key: bytes, goal_id: str, paths: Sequence[Sequence[Action]], moves_here: Sequence[Action]) -> str: records a fork and returns its id ("f1", "f2", ...); the same (state_key, goal_id) reuses the fork and adds only new, non-empty paths (statuses and first moves_here kept); paths read in order until MAX_PATHS; a new fork beyond MAX_FORKS drops the oldest (and the active path if it was there).
- start_path(self, fork_id: str, index: int) -> None: marks the path active; any other active path in the tree goes back to untried; restarting the active path is a no-op; a failed or achieved path cannot be started (miss).
- fail_branch(self, fork_id: str, index: int) -> None: marks the path failed (final) and releases it if active; repeating is a no-op; changing a final status is a miss.
- achieve(self, fork_id: str, index: int) -> None: marks the path achieved (final), releases it if active; the fork closes, so its sibling paths are no longer reported as untried.
- active(self) -> tuple[str, int] | None: the (fork id, path index) being followed.
- untried(self, fork_id: str) -> list[int]: indices of untried paths in path order; [] for an unknown fork (miss) or a fork with an achieved path.
- newest_with_untried(self) -> Fork | None: the fork with the highest created that still has an untried path (replan order of spec 3.6).
- fork(self, fork_id: str) -> Fork | None: the record by id, or None plus a miss when unknown. (The brief wrote "-> Fork"; None is needed to honor rule 4, never raising on an unknown id.)
- reset_level(self) -> None: drops every fork and the active path; the id counter and misses keep counting.
- summary(self, limit: int = SUMMARY_LINES) -> list[str]: one line per open fork (no achieved path, and some path untried or active), newest first, at most limit lines (limit <= 0 gives []): "fork <id> at goal <goal_id>: <n> untried of <m> paths, <k> moves from start".
- counts(self) -> dict[str, int]: {"forks", "untried", "failed", "achieved"}, raw tallies over every path of every fork.

def return_route(edges: Mapping[bytes, Mapping[ActionKey, bytes]], current: bytes, target: bytes, *, limit: int = ROUTE_LIMIT) -> list[Action] | None: breadth-first search over the explorer Graph.edges mapping from current to target; [] when current == target; None when target is not reached within limit distinct states (start counts as one); each state's moves are expanded in sorted key order, so results are deterministic and ties fall to the smaller action key; actions rebuilt with Action(*key).

### Implementer concerns
1. Signature deviation: fork(fork_id) returns Fork | None, not Fork, because rule 4 (never raise on an unknown id) needs a None return. Everything else matches the brief's signatures; return_route spells the key type as the repo alias ActionKey (= tuple[int, int, int]).
2. Decisions the wiring agent should know, all documented in the module docstring: an achieved path closes its fork (siblings are not offered by untried, newest_with_untried or summary); failed and achieved are final (a change to a final status is a miss, a repeat is a no-op); reopening a fork keeps its first moves_here; reset_level keeps the id counter running (no repeated fork id within a game log) and keeps misses as a lifetime tally; misses counts unknown ids, out-of-range indexes and forbidden status changes.
3. Plan section 5 lists "ruff format --check ." as a gate. On this branch it reports 4 files that would be reformatted, all Markdown plan docs with fenced code blocks (docs/superpowers/plans/2026-10-23-fluid-agent-slice-1.md, 2026-10-23-fluid-agent-slice-2.md, 2026-10-23-stack-mind-llm.md, 2026-10-31-v2-code-map.md). They are pre-existing, outside my ownership, and untouched; every .py file is formatted. Ruff 0.17.1 formats code blocks inside Markdown, which is why they appear.
4. Bounds: moves_here is cut at MAX_MOVES_HERE = 20,000 (the game's action cap) so the bound never changes a real record; paths beyond MAX_PATHS = 16 per fork are dropped without a signal other than the fork's path count. If the planner can hand over more than 16 alternatives for one fork, raise the constant.


## wp6 (v2/wp6-controller)

arcagent/mind/selfrecord.py (spec 4.5a, 4.5b; imports only COORD_ACTION_ID from arcagent.core.types)
- AGENT: str = "agent": the one referent of "I".
- LINE_CHARS = 200, GOAL_CHARS = 36, TEXT_CHARS = 16, WORD_CHARS = 10: the focus line's bound and the caps on its goal id, kind/sig and status/mode/verdict fields (cuts end in ".." like fact lines).
- @dataclass(frozen=True, slots=True, kw_only=True) class SelfRecord: i: str = "agent"; click: bool; my_character: str | None. Who "I" is at this move.
  - SelfRecord.from_state(available: Sequence[int], lone_mover: str | None) -> SelfRecord: click = COORD_ACTION_ID (6) in available; my_character = lone_mover.
  - SelfRecord.line() -> str: "I am the agent. Click: yes|no. My character: <kind>|none."
- @dataclass(frozen=True, slots=True) class FocusRecord: level: int; goal_id: str | None; goal_status: str | None; target: str | None; mode: str; budget_left: int | None; verdict: str; fallbacks_in_row: int; segment_left: int; tokens_this_move: int; seconds_this_move: float. What the agent is doing now, rewritten after each move.
  - FocusRecord.line() -> str: one line, always under 200 chars: "L<level>; goal <id> (<status>); target <t>; mode <m>; budget <n|unknown>; verdict <v>; fallbacks <n>; segment <n>; tokens <n>; <s>s" (goal/target print "none" when None).
  - FocusRecord.as_dict() -> dict[str, object]: every field by name, for the move log.

arcagent/mind/controller.py (spec 4.5c; standard library only)
- Method = Literal["keep", "fallback", "return_to_fork", "explore", "replan"].
- STALL_MOVES = 25 (H-e limit, mirrors agent/hybrid.py), NO_PROGRESS_MOVES = 40 (H-c limit), BUDGET_MARGIN = 2 (H-d spare moves), DEADLINE_RULE = "deadline", NO_RULE = "none".
- DEFAULT_ORDER: tuple[str, ...] = ("H-a", "H-b", "H-e", "H-c", "H-d"); RULES: tuple[str, ...] = ("H-a", "H-b", "H-c", "H-d", "H-e"): the names an order may use.
- @dataclass(frozen=True, slots=True) class Signals: gate_tripped: bool; segment_broke: bool; fallbacks_in_row: int; no_progress_moves: int; budget_left: int | None; segment_left: int; deadline_passed: bool. The signal values for one move.
- @dataclass(frozen=True, slots=True) class Decision: method: Method; rule: str; reason: str. rule is a RULES name, "deadline" or "none"; reason quotes the signal values that fired.
- class Controller(order: Sequence[str] = DEFAULT_ORDER, *, stall_moves: int = STALL_MOVES, no_progress_moves: int = NO_PROGRESS_MOVES): raises ValueError for an unknown or repeated rule name or a limit below 1.
  - properties order -> tuple[str, ...], stall_moves -> int, no_progress_moves -> int.
  - Controller.decide(s: Signals) -> Decision: deadline_passed -> ("fallback", "deadline") before the order; then first rule in order that fires: H-a gate_tripped -> fallback; H-b segment_broke -> return_to_fork; H-e fallbacks_in_row >= stall_moves -> explore; H-c no_progress_moves >= no_progress_moves -> replan; H-d budget_left is not None and budget_left < segment_left + 2 -> replan; else Decision("keep", "none", "no rule fired"). Pure, no state between moves, logs nothing.

arcagent/mind/flags.py (spec 4.11; standard library only)
- State = Literal["new", "check", "checked", "ignored"]; Source = Literal["novel_look", "unknown_pair", "changed", "label"].
- MAX_FLAGS = 200, LABEL_CHARS = 30, OUTCOME_MAX = 20, NEW_LOOK_VALUE = 1.0, CHANGED_VALUE = 0.5, MIN_COST = 1.0.
- @dataclass(slots=True) class Flag: kind: str; state: State; source: Source; value: float; cost: float; label: str; outcome: tuple[str, ...]. Mutable; the book owns it, get() returns the live record.
- class FlagBook(max_flags: int = MAX_FLAGS): ValueError below 1.
  - note(kind: str, *, new_look: bool, unknown_pairs: int, changed: bool = False) -> Flag: value = unknown_pairs + 1.0 if new_look + 0.5 if changed; state -> "check" when any holds unless the flag is "ignored"; a new flag's source is the first true of novel_look, unknown_pair, changed (novel_look when none holds and the kind is new to the book, state "new", value 0); an existing flag keeps source, label, outcome; ValueError for unknown_pairs < 0; evicts the lowest-value flag (ties by kind) when full.
  - set_cost(kind: str, cost: float) -> None: moves to reach plus one; ValueError below 0; KeyError for an unknown kind.
  - checked(kind: str, outcome: Sequence[str]) -> None: state "checked", outcome = first OUTCOME_MAX keys; KeyError unknown kind; TypeError for a bare string.
  - ignore(kind: str) -> None: state "ignored" until reset_level; KeyError unknown kind.
  - label(kind: str, text: str) -> None: stripped, cut to 30 chars; source stays; an unknown kind gets a flag (state "new", source "label").
  - get(kind: str) -> Flag | None.
  - top(n: int) -> list[Flag]: state "check" only, by value / max(cost, 1.0) desc then kind; n <= 0 gives [].
  - lines(n: int) -> list[str]: "check <kind>[ '<label>']: value <v>, cost <c>, <source>" for top(n), numbers in :g form.
  - reset_level() -> None: every flag to state "new", value 0, cost 0; kinds, sources, labels, outcomes kept.

### Implementer concerns
1. The plan's section 5 gate form "ruff format --check ." fails on four pre-existing docs files that ruff treats as containing Python code blocks (docs/superpowers/plans/2026-10-23-fluid-agent-slice-1.md, 2026-10-23-stack-mind-llm.md, 2026-10-31-v2-code-map.md and one more); they are outside this package and untouched. The brief's form (format the changed files, then "ruff check .") passes.
2. Judgment calls beyond the brief's letter, for the integrator (WP10) to confirm: FlagBook.note leaves an "ignored" flag ignored (otherwise ignore() would be undone on the next move); reset_level also zeroes cost (it is level-specific); a kind new to the book with nothing to check gets source "novel_look"; set_cost, checked and ignore raise KeyError for a kind with no flag (guard with get() if a flag may have been evicted); checked raises TypeError for a bare string; negative unknown_pairs or cost raise ValueError; Controller rejects duplicate rule names and limits below 1.
3. STALL_MOVES = 25 is defined in controller.py rather than imported from arcagent.agent.hybrid, to keep mind free of the agent package; two copies of the number exist until hybrid.py is rewired to the controller as the plan intends.
4. SelfRecord is kw_only so that i can keep its default while click and my_character have none; construct it as SelfRecord(click=..., my_character=...) or through from_state.
5. The build plan's WP6 text named decide(focus, signals) and from_state(available_actions, controlled_kinds, lone_mover); the package brief's signatures (decide(s: Signals), from_state(available, lone_mover)) were followed as the later, more specific contract.


## wp7 (v2/wp7-briefing)

arcagent/mind/briefing.py (C:/arc-agent-wp7/arcagent/mind/briefing.py):
- LINE_WIDTH = 160: characters per rendered line (whitespace collapsed, "..." when clipped).
- MAX_BRIEFING_CHARS = 12000: default cap on the whole rendered text.
- CUT_LINE = "(briefing cut)": final line when the text was cut.
- HORIZON_TOKEN = "<horizon>": placeholder in PLANNER_SYSTEM_PROMPT that system_prompt fills.
- MODEL_TITLE = "Your notes from the last turn (unverified)": title of the planner's own notes, rendered last.
- SECTION_ORDER: tuple[str, ...] = ("state","actions","goals","tasks","level_rules","game_rules","hypotheses","sequence","forks","flags","self_focus","recent","lessons","questions"): the list sections in render order.
- SECTION_TITLES: dict[str, str]: field name to title ("Current state","Actions","Goals","Tasks","Level rules","Game rules","Hypotheses","Sequence","Forks","Flags","Self and focus","Recent moves","Lessons","Questions").
- SECTION_CAPS: dict[str, int]: lines kept per list (state 40, actions 1, goals 6, tasks 8, level_rules 16, game_rules 16, hypotheses 6, sequence 4, forks 5, flags 8, self_focus 2, recent 12, lessons 8, questions 7).
- @dataclass(frozen=True, slots=True, kw_only=True) class Briefing: state, actions, goals, tasks, level_rules, game_rules, hypotheses, sequence, forks, flags, self_focus, recent, lessons: tuple[str, ...]; questions: tuple[tuple[str, str], ...] (question id, computed answer); model: str (planner's notes from its last reply); wake: str; level: int; horizon: int. No defaults; every field is given by keyword. Builder orders each list with the lines to keep first (newest first for recent) because render keeps the head.
- render(b: Briefing, *, cap_chars: int = MAX_BRIEFING_CHARS) -> str: header "Wake: ..", "Level: n", "Horizon: n moves", then one titled block per SECTION_ORDER entry (questions as "Q-ID: answer"), then the model notes under MODEL_TITLE (cut at contract.MAX_MODEL_CHARS); each list cut at its cap with a trailing "(+N more)" line; each line clipped to LINE_WIDTH; empty list reads "(none)"; whole text at most cap_chars, cut on the last line boundary that fits and ended with CUT_LINE.
- PLANNER_SYSTEM_PROMPT: str (2165 chars): the planner's job (few moves; score min(human/agent,1) squared), the briefing sections, and the plan section 3 reply JSON (model, facts, goals, segment up to <horizon> steps, skill with S-AROUND/S-COLLECT/S-EXPLORE/S-ATTACK/S-WAIT, question with the seven Q ids, labels, notes); segment and skill are alternatives (segment kept when both), ids bare integers valid this turn only, simulator checks the segment before any real move, a rejected segment returns with the failing step named. Facts and goals shapes are byte-identical fragments of DRIVE_SYSTEM_PROMPT (tested).
- system_prompt(horizon: int) -> str: PLANNER_SYSTEM_PROMPT with HORIZON_TOKEN replaced by max(1, horizon).
- planner_messages(b: Briefing, grid: Grid, prev_grid: Grid | None) -> list[VMessage]: [system, user]; system is system_prompt(b.horizon); user content is [text part render(b), image_url png(grid), and image_url diff_png(prev_grid, grid) when prev_grid is not None], built the way turn.build_messages does.

arcagent/mind/contract.py (C:/arc-agent-wp7/arcagent/mind/contract.py), additions:
- SKILL_NAMES = ("S-AROUND", "S-COLLECT", "S-EXPLORE", "S-ATTACK", "S-WAIT"); QUESTION_IDS = ("Q-BLOCK", "Q-REWARD", "Q-HARM", "Q-FORK", "Q-CHECK", "Q-SAFE", "Q-UNKNOWN"); MAX_LABELS = 24; MAX_LABEL_CHARS = 30; MAX_SKILL_ARGS = 16.
- @dataclass(frozen=True, slots=True) class SkillCall: name: str; args: dict[str, object] (field(hash=False): args is compared but kept out of the hash because a dict cannot be hashed).
- Reply gains skill: SkillCall | None = None; question: str = ""; labels: tuple[tuple[int, str], ...] = (); plan keeps its meaning (the segment). Existing fields and their order unchanged.
- parse_reply(text: str, scene: Scene, available: Sequence[int], *, max_steps: int = MAX_PLAN_STEPS) -> Reply. New rules (all in the module docstring): "segment" is read as the plan, "plan" accepted as alias, a null key counts as absent, both given keeps segment and records "plan ignored: segment given"; steps beyond max_steps (below 1 reads as 1) dropped with one reason "<key> capped at N steps"; "skill" with name in SKILL_NAMES (case and _ for - forgiven) and args an object (missing/null = {}, at most 16 entries) becomes Reply.skill, else dropped with "skill is not an object" / "unknown skill: X" / "skill args is not an object"; a valid skill alongside a non-empty valid segment is dropped with "skill dropped: segment given"; "question" in QUESTION_IDS (same forgiveness) becomes Reply.question, unknown id gives "unknown question: X", non-string gives "question is not a string", missing/empty gives "" with no reason; "labels" entries (first 24; lone object counts as a list of one) with an integer on-screen id and non-empty string label become (id, label[:30]) with whitespace collapsed and exact duplicates kept once, others dropped with "label is not an object" / "label needs an integer object id" / "label object not on screen: #N" / "label for #N needs text". Every pre-existing rule and drop reason is unchanged.

### Implementer concerns
1. The full-suite run had not printed its final line when the harness forced the return (it was at about 92% with no failure markers). The background process is terminated with this return, so the log at C:/Users/Sagac/AppData/Local/Temp/claude/wp7-pytest.txt may stay incomplete; rerun before merge: cd C:/arc-agent-wp7 && C:/arc-agent-venv/Scripts/python.exe -m pytest -q -p no:cacheprovider. The three test files my change touches (test_briefing, test_contract, test_turn) are fully green and whole-tree ruff and pyright are green. 2. "ruff format --check ." on the whole worktree fails on four pre-existing Markdown files under docs/superpowers/plans (ruff 0.17 formats Python code blocks inside Markdown): 2026-10-23-fluid-agent-slice-1.md, 2026-10-23-stack-mind-llm.md, 2026-10-31-v2-code-map.md and one more. They are untouched by me (git diff agent-v2 -- docs is empty) and outside my ownership; plan section 5 lists this as a gate, so whoever owns docs should exclude *.md in pyproject [tool.ruff] extend-exclude or reformat the blocks. 3. Choices beyond the brief's letter, each documented in the module docstrings: Briefing is kw_only=True and slots=True in addition to frozen=True (18 same-typed fields make positional construction an error magnet); render keeps the head of each list and appends "(+N more)", so the builder must order each list with the lines to keep first (newest first for recent) to honor spec 4.9a's "truncated from the least recent end"; render has a keyword cap_chars defaulting to MAX_BRIEFING_CHARS (plan section 4 named render(briefing, cap_chars)); the total cut falls on the last line boundary that fits so no half line reaches the planner; the skill drop reason is "skill dropped: segment given" (contains the brief's phrase); a plan ignored for a segment is recorded as "plan ignored: segment given"; skill args are capped at 16 entries and labels at 24 with a reason; skill and question ids forgive case and underscores. 4. The pre-existing _plan loop still records one drop reason per invalid step without a bound on the list length (unchanged behavior, noted only). 5. Git printed "LF will be replaced by CRLF" warnings for the two new files (core.autocrlf on this machine); content is unaffected.


## wp8 (v2/wp8-skills)

arcagent/mind/skills.py (C:/arc-agent-wp8/arcagent/mind/skills.py):
- Role = Literal["harm", "collectible", "blocker", "mover", "controlled", "unknown"]: the role the agent gave a kind.
- Flag = tuple[str, float, float]: a curiosity flag as (kind, value, cost).
- SKILL_NAMES: tuple[str, ...] = ("S-AROUND", "S-COLLECT", "S-EXPLORE", "S-ATTACK", "S-WAIT"): the names run_skill accepts (the contract's skill list).
- MAX_COLLECT = 8: collectibles routed per S-COLLECT call, nearest by cell gap first.
- MAX_FLAGS = 100: flags read per S-EXPLORE call.
- MAX_AVOID_CELLS = GRID_SIZE * GRID_SIZE: cells read from args["avoid_cells"].
- UNKNOWN_SKILL = "unknown skill"; NOT_BUILT = "not built in this version": the fixed why strings.
- @dataclass(frozen=True) class SkillContext: scene: Scene; mem: FactMemory; kin: Kinematics; goals: Sequence[Goal]; roles: Mapping[str, Role]; flags: Sequence[Flag]; can_click: bool; mover: str | None; click_cell: Callable[[Thing], tuple[int, int]]; max_steps: int. goals are the open goals in the agent's order; roles map kind (sig when no kind class) to role; mover is the controlled thing's name as Kinematics knows it.
- @dataclass(frozen=True) class SkillResult: name: str; actions: tuple[Action, ...]; why: str; applies: bool. applies False carries the reason and no actions.
- run_skill(name: str, args: Mapping[str, object], ctx: SkillContext) -> SkillResult: dispatch by name (stripped, upper-cased, "_" read as "-"); unknown name -> applies False, why "unknown skill"; ctx.max_steps < 1 -> applies False; S-ATTACK and S-WAIT -> applies False, why "not built in this version"; every result holds at most ctx.max_steps actions (why gains ", cut to N" when cut); never raises on a scene a skill cannot handle. Args read: S-AROUND takes args["target"] (int thing id; wins when on screen, else the nearest target of the first reach/cover goal whose a the mover answers to and whose target is on screen) and args["avoid_cells"] (list of [x, y] int pairs, malformed entries skipped). S-COLLECT and S-EXPLORE read no args. S-AROUND contact rule: overlap or 4-adjacent. S-COLLECT: collectible-role things (nearest MAX_COLLECT by cell gap, those with no path from the start dropped) ordered by route.route over kinematic legs, moves to the first, harm cells avoided, overlap preferred then touch. S-EXPLORE: first flag with a thing on screen; one Action(6, *ctx.click_cell(thing)) when ctx.can_click, else the moves to touch it, harm cells avoided. "Already touching" returns applies False with that reason.

arcagent/mind/kinematics.py (C:/arc-agent-wp8/arcagent/mind/kinematics.py), additive:
- Kinematics.reachable_avoiding(self, scene: Scene, name: str, avoid: frozenset[Cell], *, limit: int = KIN_LIMIT) -> dict[Cell, tuple[int, Cell | None, Action | None]]: forward breadth-first search over the subject's gaits with the same fit test as reachable; each top-left maps to (moves, previous top-left, action taken); the start maps to (0, None, None) and is never tested; a fixed step must fit and keep every body cell out of avoid; a slide is refused when any position on its way puts a body cell in avoid; a gait whose label is not A1..A5/A7 is skipped; with an empty avoid the move counts equal reachable's; empty when the subject is not on screen; at most limit top-lefts. reachable, steps_to, field and _walk_back are unchanged. Private additions: _step_avoiding, _action_of, _SIMPLE_IDS; one sentence added to the module docstring.

### Implementer concerns
1. S-COLLECT leg lengths deviate from the brief's literal "steps_to or field": steps_to measures only from the mover's current place, and field names its target by sig, so two same-sig fruits (the pacman case) collapse into one target and the legs between them read 0. Legs are measured instead with reachable_avoiding from a scene where the mover stands at the previous collectible's top-left and that collectible is removed, cell gap as the fallback. Cost: up to MAX_COLLECT + 1 BFS walks of up to 4096 top-lefts per S-COLLECT call; worth measuring against the per-move time gate (4.10d) once wired.
2. Slide semantics in reachable_avoiding: a slide whose path crosses an avoided cell is refused outright rather than stopped short of it (the brief's literal fit-and-avoid reading). Reason: a slide does not stop for a thing that is not a blocker, so stopping it short would plan a move the game will not make. When the harm thing is also a blocker the plain fit already stops the slide before it, so both cases come out right. Move counts equal reachable's when avoid is empty (tested).
3. S-COLLECT contact prefers a top-left whose body overlaps the collectible (how one is taken) and falls back to touching; S-AROUND and S-EXPLORE use the touch rule (overlap or 4-adjacent) as the brief says. A skill whose path is empty (already in contact) returns applies False with "already touching ..." so the executor is never handed an empty segment.
4. SkillContext has no goal status, so ctx.goals is read as the open goals; "the first open reach or cover goal" is read as the first whose target is on screen (goals with no target on screen are skipped). Roles are read by t.kind then t.sig. args["target"] must be an int (bools rejected, a float such as 12.0 counts as absent).
5. The plan's section 4 sketch of WP8 named a Skill protocol (trigger/steps); the package brief's run_skill/SkillContext/SkillResult contract was built instead, as the brief directs.
6. Repo-wide "ruff format --check ." (plan section 5) fails on four pre-existing Markdown files under docs/superpowers/plans (ruff formats their Python code blocks); all 177 Python files are formatted. Those files predate this package on agent-v2 and are outside my ownership, so they were left alone. The brief's own gate (format the changed files) passes.
7. SkillContext.mem is kept per the brief but no skill reads it directly (Kinematics carries the memory).


## wp9 (v2/wp9-logging)

arcagent/mind/runlog.py: no new code names. The module docstring gains the section "Planner agent (v2) fields", which is the log contract. Action-record fields: self_rec (dict of SelfRecord fields; named self_rec because RunLog.action's first parameter is self and a field named self raises TypeError), focus (dict of FocusRecord fields), hypothesis_id, segment_id, fork_id, planner_call (ids, null when none), p_step (probability of this move's prediction, 4 decimals, null when none), p_path (the segment's path probability, null when none), prev_matched (whether the previous move's prediction matched, null on the first move of a level instance), gate ({"seconds": float, "tokens": int, "trip": str | null}). Events: segment (step, id, source, length, p_path, ev, goal), handoff (step, rule, method, reason), harm (step, kind, label, p), consolidate (level, promoted, sequences, typical), longterm (game_id, rules, sequences; game_id because game is an envelope key), fork_open (step, id, goal, paths), branch_failed (step, fork, index, reason), return_route (step, fork, length), replan_root (step, reason), gate_trip (step, gate, seconds, tokens), planner_call (step, call, seconds, tokens, segment_len, skill, question, dropped), flag (step, kind, state, source, value, cost). No code change was needed: json.dumps encodes nested dicts and lists natively and hands only numpy leaves to _plain (proved by test).

scripts/mind_metrics.py (CLI unchanged: python scripts/mind_metrics.py LOG_DIR... [--json out.json]):
- v2_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]: the planner-agent metrics block for one game's records taken in file order; every key present whatever the records hold.
- compute(records: Iterable[Rec], human: dict[str, list[int]] | None = None) -> dict[str, Any]: signature unchanged; every per-game dict and "overall" now carry key "v2" (per game it equals v2_metrics(that game's records); overall pools the raw tallies, recomputes rates and means, concatenates moves_vs_typical in first-seen game order).
- v2 block shape: {"calibration": [10 x {"bin": "0.0-0.1".."0.9-1.0", "n": int, "matched": int, "rate": float | None}], "moves_vs_typical": [{"level", "moves": int, "typical": number | None, "ratio": float | None}], "gates": {"trips": int, "by_gate": {name: int}, "seconds_total": float, "tokens_total": int}, "handoffs": {rule: int}, "planner": {"calls": int, "tokens": int, "segments": int, "mean_segment_len": float | None, "skills": {name: int}, "questions": {id: int}}, "forks": {"opened", "failed", "returns", "root_replans"}, "harm": {"events": int, "by_kind": {kind: int}}}.
- Definitions (module docstring, "Planner agent (v2) metrics"): a calibration pair is action i's p_step with action i+1's prev_matched when both carry the same level and no level_up or game_over event lies between them in file order; skipped when p_step is not a number in [0, 1] or prev_matched is not a bool; bin i holds i/10 <= p < (i+1)/10 and p == 1 joins the last bin. moves_vs_typical: one entry per consolidate event; moves counts the action records of the level instance the event closes (records since the previous level_up, game_over or consolidate; a level_up directly before the consolidate hands over the instance it closed); ratio = moves / typical when typical is a positive number. gates: trips and by_gate count distinct (step, gate name) pairs over gate_trip events and action records whose gate.trip is not null (a trip logged both ways counts once); seconds_total and tokens_total sum the action records' numeric gate.seconds and gate.tokens. handoffs: handoff events by rule. planner: calls and summed numeric tokens over planner_call events, skills by skill (a string, or a dict's "name"), questions by question; segments counts segment events of every source, mean_segment_len the mean of their numeric length. forks: counts of fork_open, branch_failed, return_route, replan_root. harm: harm events and their kinds.
- New module constants: CAL_BINS = 10; LEVEL_BOUNDARIES = frozenset({"level_up", "game_over"}); FORK_EVENTS = {"fork_open": "opened", "branch_failed": "failed", "return_route": "returns", "replan_root": "root_replans"}.
- Private: class _V2Tally (add(rec), raw() -> v2_-prefixed additive tallies), _v2_summary(raw), _bin_index(p), _bin_label(i), _skill_name(value), _sorted(counts). Existing helper _is_num(x: object) now returns TypeGuard[int | float] (runtime unchanged).

### Implementer concerns
No gate failed and nothing is blocked; these are notes for the integrator and reviewer. (1) Contract choices WP10 must honor when wiring the agent: calibration pairs are taken in file order and need the fluid agent's existing order (action record, then the level_up or game_over event with that step, then the next action record) plus the same level value on both records; a gate trip is deduplicated by (step, gate name) across the action's gate.trip and the gate_trip event, so both can be logged for one move; planner.segments and mean_segment_len count segment events of every source (planner, search, skill); consolidate's moves counts the action records of the instance it closes, so the RESET after a death belongs to the new instance; moves_vs_typical entries carry no game id (the brief's shape), so in the pooled list only order tells games apart. (2) pyright's include is ["arcagent", "tests"], so the project pyright run never checks scripts/mind_metrics.py; I checked it explicitly (clean) and had to make _is_num a TypeGuard for that. Adding "scripts" to the include is a pyproject change outside my files. (3) Repo-wide "ruff format --check ." names 4 files, all docs/superpowers/plans/*.md with fenced code (the three the code map already listed plus 2026-10-31-v2-code-map.md itself); none is mine and I left them alone. (4) The runlog tests pass without a code change (RunLog already round-trips nested dicts and lists), so Part A is docstring only. (5) In tests, a harm event cannot be built with the ev() helper because its first parameter is named kind; a literal-dict helper harm() is used instead. (6) The pytest summary count line is absent at the effective -qq verbosity; the count above was taken from the outcome marks in the saved output, and exit code 0 is the gate.
