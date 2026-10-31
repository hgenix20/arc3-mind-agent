export const meta = {
  name: 'v2-wave1-modules',
  description: 'Build the eight wave-1 modules of agent v2 in parallel worktrees, each with tests, gates, a review and one fix round',
  phases: [
    { title: 'Implement', detail: 'eight implementers, one worktree each' },
    { title: 'Review', detail: 'one reviewer per package reads the branch diff' },
    { title: 'Fix', detail: 'one fix round where the reviewer asked for changes, then a re-review' },
  ],
}

const PY = 'C:/arc-agent-venv/Scripts/python.exe'
const MAIN = 'C:/arc-agent'
const PLAN = `${MAIN}/docs/superpowers/plans/2026-10-31-agent-v2-build.md`
const MAP = `${MAIN}/docs/superpowers/plans/2026-10-31-v2-code-map.md`
const SPEC = `${MAIN}/docs/superpowers/specs/2026-10-31-agent-spec-v2.md`
const SESSION = 'Claude-Session: https://claude.ai/code/session_01ED1JLjSNaMyvkVda5AScAg'

const common = (p) => `You are implementing work package ${p.key} of the agent v2 build. Your working copy is the git worktree at ${p.wt} on branch ${p.branch} (already checked out from agent-v2; confirm with "git -C ${p.wt} branch --show-current"). Every tool call uses absolute paths under ${p.wt}; the shell's default directory is not the repo. Never edit anything under ${MAIN} or any other worktree. The project interpreter is ${PY}; never use system Python. Files you own (create or change only these): ${p.files}. If a gate can only pass by changing another file, stop, do not change it, and report BLOCKED with the reason.

Read first, in this order: ${PLAN} (section 4, your package; section 2 for the architecture; section 5 gates), ${MAP} (the sections named in your brief; it was produced by readers of this code and may contain errors, so confirm against the files), the spec sections named below in ${SPEC}, and the existing files your brief names.

Method: write the failing tests first and run them (they must fail for the right reason), then implement, then run them again. Small units, explicit contracts, types on every signature (pyright strict is a gate), docstrings that state intent, contract, failure modes and the spec section; comments say why, not what. Plain English in every docstring and comment: no em-dashes, no "not X but Y" phrasing. Bound every loop and every list. Log nothing yourself unless the brief says so; the agent that wires your module logs.

Gates, all run from ${p.wt} with the cwd set there (cd in the same command): "${PY} -m ruff format <only the files you changed>", "${PY} -m ruff check ." (All checks passed), "${PY} -m pyright" (0 errors), and the full suite "${PY} -m pytest -q -p no:cacheprovider" with output redirected to C:/Users/Sagac/AppData/Local/Temp/claude/${p.key}-pytest.txt (read its last 3 lines; it takes a few minutes). Then commit on ${p.branch}: "git -C ${p.wt} add <your files>" and a message of the form:

<imperative title under 72 chars>

<two or three lines on what and why>

${SESSION}

No Co-Authored-By line. Do not spawn subagents. Return: status (DONE, DONE_WITH_CONCERNS or BLOCKED), the commit hash(es), the public interface you produced (every public name with its exact signature and a one-line meaning), the pytest command's last line, and concerns (empty string if none).`

const PACKAGES = [
  {
    key: 'wp2', wt: 'C:/arc-agent-wp2', branch: 'v2/wp2-adverse',
    files: 'arcagent/mind/adverse.py (new), tests/mind/test_adverse.py (new)',
    brief: `Spec sections: 4.10a, 4.10b, 4.10c and 4.4 (bands 0.8 and 0.3). Code map sections: facts-predict (for the shape of a FactKey and the sig format "c<color>:<shape>"), fluid-loop (how vetoes are used), goals-search (tester.veto reasons).

Build arcagent/mind/adverse.py, pure functions and small classes, numpy only:

1. AdverseKey = tuple[str, str, str]: (action label, subject kind, context). The kind is a Thing.kind string (a sig-shaped string); callers pass it.
2. class AdverseBook: record(key, harmed: bool) -> None; trials(key) -> int; harms(key) -> int; probability(key) -> float = (harms + PRIOR_A) / (trials + PRIOR_A + PRIOR_B) with PRIOR_A = PRIOR_B = 1, rounded to 4 decimals, 0.5 for an unknown key; band(key) -> Literal["certain", "medium", "low"] with BAND_CERTAIN = 0.8 (>=) and BAND_LOW = 0.3 (<); evidenced(key) -> bool when trials(key) >= RULE_TRIALS (5); keys() -> list of keys sorted. Bounded: at most MAX_KEYS = 5000 keys; beyond that record() drops the least-tried key and returns normally.
3. features(label: str, kind: str, shape_class: str, context: str, roles: frozenset[str], health_bar: bool) -> np.ndarray of shape (WIDTH,), WIDTH = 256, float64. Hashing trick: each token ("label=<label>", "kind=<kind>", "shape=<shape_class>", "ctx=<context>", one "role=<r>" per role, "hb=<0|1>") sets index zlib.crc32(token.encode()) % (WIDTH - 1) to 1.0; index WIDTH - 1 is the bias, always 1.0. Deterministic across processes (never Python's hash()).
4. class InhibitionNet(width: int = WIDTH, lr: float = 0.1, l2: float = 0.001): weights float64 zeros; predict(x) -> float = sigmoid(w . x) rounded to 4 decimals; update(x, harmed: bool) -> None does one SGD step on the logistic loss with L2 on every weight but the bias; n_updates: int; state() -> dict[str, list[float] | int] and load(state) for persistence. Guard: predict on a wrong-length x raises ValueError.
5. harm_probability(book: AdverseBook, net: InhibitionNet, key: AdverseKey, x: np.ndarray) -> tuple[float, Literal["rule", "net"]]: the book's probability when book.evidenced(key), else the net's prediction.
6. class Perseverance: allow(p: float, *, no_other_way: bool, ev_gain: float, trials_used: int, max_trials: int = 3) -> tuple[bool, str]. Rules (spec 4.10c): p >= 0.8 -> (False, "certain"); p < 0.3 -> (True, "low"); medium: trials_used >= max_trials -> (False, "trials exhausted"); no_other_way -> (True, "no other way"); ev_gain > 0 -> (True, "expected value"); else (False, "no reason").
7. band_of(p: float) -> the band literal, shared by the book and the perseverance rule.

Tests (tests/mind/test_adverse.py): unknown key is 0.5 and medium; probability after 4 harms in 5 trials is 0.7143; bands at 0.79, 0.8, 0.29, 0.3; evidenced flips at 5 trials; features are deterministic, have the bias set, and differ when the label differs; the net moves its prediction toward 1 after repeated updates with harmed=True and toward 0 with harmed=False; harm_probability picks the rule only when evidenced; every Perseverance branch; MAX_KEYS eviction keeps the most-tried keys.`,
  },
  {
    key: 'wp3', wt: 'C:/arc-agent-wp3', branch: 'v2/wp3-longterm',
    files: 'arcagent/mind/longterm.py (new), tests/mind/test_longterm.py (new)',
    brief: `Spec sections: 4.2 (record types, stores, status and clear rules), 4.12b, 4.12c, 4.7 horizon rule. Code map sections: facts-predict (FactKey, Effect kinds, sig format "c<color>:<shape>", action_label), goals-search (Goal.id()).

Build arcagent/mind/longterm.py: the long-term store that persists across games in one run (and across runs when the file is kept). Standard library plus the repo's own types only.

1. @dataclass RuleRecord: id: str; kind: Literal["object", "action", "event", "experience"]; statement: str (plain words); canonical_key: tuple[str, str, str, str] (action_role, shape_class, context_tag, effect_kind); effect: str; probability: float; confirmations: int; contradictions: int; state: Literal["hypothesis", "fact", "retired"]; scope: Literal["working", "long"]; games_seen: list[str]; created_game: str; created_level: int; last_revised: float (time.time()); harm: str ("none" or a harm kind); source: Literal["observed", "loaded", "label"].
2. @dataclass SequenceRecord: id: str; goal_id: str; moves: list[list[int]] (each [id, x, y]); game: str; level: int; runs: int; successes: int; deterministic: bool.
3. canonical_key(fact_key: tuple[str, str, str], effects: Sequence[object]) -> tuple[str, str, str, str]: action_role is the label with any click target removed (read facts.action_label and clicked_sig in facts.py to see the label formats; a click label becomes "click", a simple action label stays, "event" stays); shape_class is the subject's sig with its colour prefix removed (sig format "c<colour>:<shape>"; subjects "self", "any", and the spawn subject stay as they are); context_tag is the context string with any colour index removed the same way; effect_kind is the first effect's kind attribute (or "none" when there are no effects). Never include a colour index (spec 4.2, Part 1 D1).
4. statement_for(fact_key, effects) -> str: one plain sentence from the canonical key, for example "A3 moves a <shape> thing by (1, 0)" or "clicking a <shape> thing recolours it". Keep under 120 characters.
5. class LongTermStore(path: Path): rules: dict[canonical_key, RuleRecord]; sequences: dict[str, SequenceRecord]; completions: dict[str, dict[int, list[int]]] (game -> level -> moves per completion); net_state: dict | None (opaque, for the inhibition network); load() -> None (missing file is an empty store; a corrupt file is logged to a .bad copy and treated as empty); save() -> None (write to a temp file in the same directory and os.replace it, under a threading.RLock held for the whole load-modify-save); promote(record) -> RuleRecord: insert, or merge by canonical_key (sum confirmations and contradictions, union games_seen, probability = (confirmations + 1) / (confirmations + contradictions + 2) rounded to 4 decimals, state by bands 0.8 and 0.3 with "fact" needing confirmations >= 5 and no contradictions since last_revised is not tracked here so use contradictions == 0, scope "long", last_revised now); rules_for(*, action_role=None, shape_class=None, context_tag=None, effect_kind=None, limit=20) -> list[RuleRecord] ranked by probability desc then id; lessons(limit=12) -> list[str] each "<statement> (p=<4 decimals>, <n> games)" ranked by probability then games count; record_completion(game, level, moves) -> None; typical_moves(game, level) -> float | None (mean over that game's completions of that level); typical_for_level(level) -> float | None (mean over every game's completions of that level index); add_sequence(record) -> None (merge by id: runs and successes summed); sequences_for(goal_id, limit=5) -> list[SequenceRecord] by successes desc.
6. Bounds: at most MAX_RULES = 5000 rules (drop the lowest probability when full), MAX_SEQUENCES = 2000, MAX_COMPLETIONS_PER_LEVEL = 50.
7. JSON format: one object {"version": 1, "rules": [...], "sequences": [...], "completions": {...}, "net_state": ...}. Tuples become lists on save and back on load.

Tests (tests/mind/test_longterm.py): round trip through save and load keeps every field; a missing file is an empty store; a corrupt file is renamed and treated as empty; promote merges counts and recomputes probability and state; canonical_key strips colours from the subject and context and never contains "c<digits>:" ; statement_for has no colour index; typical_moves and typical_for_level means; lessons format and order; two threads promoting 200 records each leave 400 or merged records with no exception and a loadable file.`,
  },
  {
    key: 'wp4', wt: 'C:/arc-agent-wp4', branch: 'v2/wp4-rollout',
    files: 'arcagent/mind/rollout.py (new), arcagent/mind/hypotheses.py (new), tests/mind/test_rollout.py (new), tests/mind/test_hypotheses.py (new)',
    brief: `Spec sections: 4.8a (rollout), 4.4c (path probability by the chain rule), 4.7a, 4.7b (hypothesis list, expected value), plan section 2 decision D-C. Code map sections: facts-predict (Prediction fields: scene, effects, sources, lent, assumed, unknown, used, game_over; predict signature and assume_still), goals-search (play_out in game_model.py, Goal.holds, Goal.id), and read arcagent/mind/game_model.py and arcagent/mind/predictor.py yourself. Facts already have Fact.probability and FactMemory.probability(key) (added by WP1; read facts.py for the exact names).

Build arcagent/mind/rollout.py:

1. @dataclass(frozen=True) Rollout: scenes: tuple[Scene, ...]; step_probabilities: tuple[float, ...]; path_probability: float (4 decimals); reached: Goal | None; unknown_at: int | None; harm: tuple[tuple[int, float], ...] (step index, harm probability); stop: Literal["goal", "unknown", "death", "horizon", "end"]; expected_moves: float; assumed: int (count of assumed-still predictions across steps); deaths_at: int | None.
2. rollout(scene, actions: Sequence[Action], mem: FactMemory, goals: Sequence[Goal], protect: frozenset[str], *, horizon: int, health: int = 1, adverse: Callable[[Scene, Action], float] | None = None, rule: RuleFn | None = None) -> Rollout. Logic, per step i up to min(len(actions), horizon): predict with assume_still=protect (as game_model.play_out does, including the optional rule function exactly as play_out applies it); if the prediction has no scene and the rule gives nothing, stop "unknown" at i; step probability = product over the distinct fact keys in the prediction's used set of mem.probability(key), rounded to 4 decimals, 0.5 when the set is empty; path probability = product of step probabilities (the chain rule over the simulated states, spec 4.4c); assumed += number of assumed pairs; harm probability h = adverse(scene_before, action) when adverse is given else 0.0; record (i, h) when h > 0; death when the prediction predicts game over (read how predict reports it: Prediction.game_over) or when h >= 0.8: health decreases by 1 and, when it reaches 0, stop "death" with deaths_at = i (the scene is still appended); if a goal holds in the new scene, stop "goal" with reached set; after the loop, stop "horizon" when the horizon cut the actions short, else "end". expected_moves = sum over steps of 1 / max(p_step, 0.05), capped at 3 * horizon, rounded to 2 decimals.
3. Keep rollout pure apart from calling the rule function. Do not change game_model.py; its play_out stays for the DRIVE path.

Build arcagent/mind/hypotheses.py:

4. @dataclass(frozen=True) Hypothesis: id: str; goal_id: str; source: Literal["planner", "search", "skill"]; plan: tuple[Action, ...]; path_probability: float; expected_moves: float; harm_probability: float (max over the rollout's harm entries, 0.0 when none); expected_value: float; stop: str.
5. expected_value(path_p: float, expected_moves: float, typical: float, harm_p: float, restart_moves: float) -> float: eff = min(1.0, typical / max(expected_moves, 1.0)) ** 2; value = path_p * 100.0 * eff - harm_p * restart_moves; rounded to 4 decimals. Document D-C from the plan in the docstring.
6. hypothesis_id(goal_id: str, source: str, plan: Sequence[Action]) -> str: a 10-character hex digest of the three (sha1), deterministic.
7. from_rollout(goal_id, source, plan, ro: Rollout, *, typical: float, restart_moves: float) -> Hypothesis.
8. rank(hyps: Sequence[Hypothesis]) -> list[Hypothesis]: expected_value desc, then path_probability (4 decimals) desc, then id asc.

Tests: build small scenes and facts the way tests/mind/test_game_model.py and tests/mind/test_predictor.py do (read them). Cover: path probability is the product of step probabilities and each step's is the product of its distinct facts' probabilities; a step with no fact for a protected thing stops "unknown"; a predicted game over stops "death" with health 1 and only decrements with health 2; adverse >= 0.8 counts as death and < 0.8 is recorded in harm; the horizon cuts a long plan with stop "horizon"; a goal that holds stops "goal"; expected_moves formula and cap; expected_value examples (typical 20, expected 20 gives eff 1; expected 40 gives 0.25); rank order and ties.`,
  },
  {
    key: 'wp5', wt: 'C:/arc-agent-wp5', branch: 'v2/wp5-plantree',
    files: 'arcagent/mind/plantree.py (new), tests/mind/test_plantree.py (new)',
    brief: `Spec sections: 4.7c (forks and the plan tree), 3.6 (fork handling), Kameron's answers in section 6 (a fork is a plan or path that could go two ways toward the same goal; keep the moves that got there; the explorer's backtracking is the replan step). Code map section: runtime-kaggle (ExplorerPolicy and Graph: edges is dict[StateKey, dict[ActionKey, StateKey]] with StateKey = bytes and ActionKey = tuple[int, int, int] = (id, x, y); read arcagent/agent/explore.py lines 60 to 160 yourself).

Build arcagent/mind/plantree.py, standard library plus the repo's types:

1. @dataclass Fork: id: str; state_key: bytes; goal_id: str; paths: list[tuple[Action, ...]]; status: list[Literal["untried", "active", "failed", "achieved"]]; moves_here: tuple[Action, ...] (the moves from the level start to this fork, kept per Kameron); created: int (a counter, not time).
2. class PlanTree: open_fork(state_key: bytes, goal_id: str, paths: Sequence[Sequence[Action]], moves_here: Sequence[Action]) -> str (returns the fork id; a second call with the same state_key and goal_id adds any new paths to the existing fork instead of making a new one; at most MAX_FORKS = 200 forks, oldest dropped); start_path(fork_id, index) -> None (marks active; only one path in the whole tree is active at a time; starting one makes any other active path "untried" again); fail_branch(fork_id, index) -> None; achieve(fork_id, index) -> None; active() -> tuple[str, int] | None; untried(fork_id) -> list[int]; newest_with_untried() -> Fork | None (highest created with at least one untried path); fork(fork_id) -> Fork; reset_level() -> None (clears everything; a new level instance starts); summary(limit: int = 5) -> list[str] one line per open fork for a briefing ("fork <id> at goal <goal_id>: <n> untried of <m> paths, <k> moves from start"); counts() -> dict[str, int] (forks, untried, failed, achieved).
3. return_route(edges: Mapping[bytes, Mapping[tuple[int, int, int], bytes]], current: bytes, target: bytes, *, limit: int = 20000) -> list[Action] | None: breadth-first search over the known transitions from current to target, returning the action list (rebuild each Action from its key with Action(id, x, y); read arcagent/core/types.py for the constructor) or None when unreachable within limit visited states; an empty list when current == target. Deterministic: iterate a state's actions in sorted key order.
4. Every mutation returns normally and the tree never raises on an unknown fork id (it returns None or an empty list and records the miss in a counter misses: int so the wiring agent can log it).

Tests: open two forks and the newest_with_untried is the second; starting a path deactivates the other active path; failing all paths makes untried empty and newest_with_untried skips it; the same state and goal reuse one fork and add new paths only; return_route finds the shortest action list on a small synthetic edge map, returns [] for the same state, None when unreachable, and respects the visit limit; reset_level empties everything; summary lines are bounded and contain the counts.`,
  },
  {
    key: 'wp6', wt: 'C:/arc-agent-wp6', branch: 'v2/wp6-controller',
    files: 'arcagent/mind/selfrecord.py (new), arcagent/mind/controller.py (new), arcagent/mind/flags.py (new), tests/mind/test_selfrecord.py (new), tests/mind/test_controller.py (new), tests/mind/test_flags.py (new)',
    brief: `Spec sections: 4.5a (self record; Kameron: "I" is always the agent, click present means action 6 is available, labels reach both the briefing and the decisions), 4.5b (focus record), 4.5c (controller; the handoff rules H-a to H-e; Kameron will order them after seeing the workflow, so the order is a constructor argument with the default given here), 4.11 (flags: derived from new looks and unknown pairs, agent-made labels allowed, value and cost, the planner decides). Code map sections: fluid-loop (what signals exist: progress verdicts, fallback branch, budget), goals-search (ProgressMonitor verdicts, BudgetReader.remaining).

Build three small modules, standard library only:

arcagent/mind/selfrecord.py
1. @dataclass(frozen=True) SelfRecord: i: str = "agent"; click: bool; my_character: str | None. from_state(available: Sequence[int], lone_mover: str | None) -> SelfRecord with click = 6 in available. line() -> str: "I am the agent. Click: yes|no. My character: <kind>|none."
2. @dataclass(frozen=True) FocusRecord: level: int; goal_id: str | None; goal_status: str | None; target: str | None; mode: str (the branch of the last move); budget_left: int | None; verdict: str (progress verdict of the last move); fallbacks_in_row: int; segment_left: int; tokens_this_move: int; seconds_this_move: float. line() -> str one line, under 200 characters. as_dict() -> dict[str, object] for the log.

arcagent/mind/controller.py
3. @dataclass(frozen=True) Signals: gate_tripped: bool; segment_broke: bool; fallbacks_in_row: int; no_progress_moves: int; budget_left: int | None; segment_left: int; deadline_passed: bool.
4. @dataclass(frozen=True) Decision: method: Literal["keep", "fallback", "return_to_fork", "explore", "replan"]; rule: str; reason: str.
5. class Controller(order: Sequence[str] = DEFAULT_ORDER, *, stall_moves: int = 25, no_progress_moves: int = 40): DEFAULT_ORDER = ("H-a", "H-b", "H-e", "H-c", "H-d"). decide(s: Signals) -> Decision: evaluate the rules in order and return the first that fires: H-a gate_tripped -> "fallback"; H-b segment_broke -> "return_to_fork"; H-e fallbacks_in_row >= stall_moves -> "explore"; H-c no_progress_moves >= no_progress_moves -> "replan"; H-d budget_left is not None and budget_left < segment_left + 2 -> "replan"; deadline_passed always wins before the order -> "fallback" with rule "deadline"; otherwise Decision("keep", "none", "no rule fired"). An unknown rule name in order raises ValueError at construction. Each Decision carries a reason string with the signal values that fired.

arcagent/mind/flags.py
6. @dataclass Flag: kind: str; state: Literal["new", "check", "checked", "ignored"]; source: Literal["novel_look", "unknown_pair", "changed", "label"]; value: float; cost: float; label: str; outcome: tuple[str, ...].
7. class FlagBook(max_flags: int = 200): note(kind: str, *, new_look: bool, unknown_pairs: int, changed: bool = False) -> Flag (sets state "check" when new_look or unknown_pairs > 0 or changed, with value = unknown_pairs + (1 if new_look else 0) + (0.5 if changed else 0), source by the first true of novel_look, unknown_pair, changed; an existing flag keeps its label and outcome and gets its value refreshed); set_cost(kind, cost: float) -> None; checked(kind, outcome: Sequence[str]) -> None (state "checked"); ignore(kind) -> None; label(kind, text: str) -> None (text cut to 30 characters; source stays); get(kind) -> Flag | None; top(n: int) -> list[Flag] of state "check" ordered by value / max(cost, 1) desc then kind; lines(n: int) -> list[str] for the briefing: "check <kind>[ '<label>']: value <v>, cost <c>, <source>"; reset_level() keeps labels and outcomes, resets states to "new" and values to 0; eviction drops the lowest value when full.

Tests: SelfRecord four cases (click and character, character only, click only, neither); FocusRecord line length and as_dict keys; Controller: each rule fires alone, deadline wins over everything, order change changes the winner when two fire, unknown rule name raises; FlagBook: note sets check and the source precedence, checked clears, label truncation, top ordering by value per cost, lines format, reset keeps labels, eviction.`,
  },
  {
    key: 'wp7', wt: 'C:/arc-agent-wp7', branch: 'v2/wp7-briefing',
    files: 'arcagent/mind/briefing.py (new), arcagent/mind/contract.py (extend), tests/mind/test_briefing.py (new), tests/mind/test_contract.py (extend)',
    brief: `Spec sections: 4.9a (briefing lists), 4.9c (question library; Kameron: propose questions so the planner decides faster instead of being told what to do), 4.9f (planner call), plan section 3 (the v2 reply contract). Code map section: llm-infra (turn.py TurnInput, _text and its caps, build_messages and the images, DRIVE_SYSTEM_PROMPT; contract.py parse_reply rules and Reply fields; screen.py png, diff_png, data_url). Read arcagent/mind/turn.py and arcagent/mind/contract.py in full.

Part A, arcagent/mind/briefing.py (new):
1. @dataclass(frozen=True) Briefing with these fields, each tuple[str, ...] unless said: state (current state list: one line per thing with id, kind, role, known rules), actions (valid actions and whether click is present), goals (level goal and game goal with status), tasks (ordered tasks of the chosen hypothesis), level_rules, game_rules, hypotheses (top hypotheses with path probability and expected value), sequence (the current segment and status), forks, flags, self_focus (the self record line and the focus record line), recent (last moves with predicted and observed), lessons (long-term rules in words), questions: tuple[tuple[str, str], ...] (question id, computed answer), model: str (the planner's own model text from its last reply), wake: str, level: int, horizon: int.
2. SECTION_CAPS: dict[str, int] (lines per list; defaults: state 40, actions 1, goals 6, tasks 8, level_rules 16, game_rules 16, hypotheses 6, sequence 4, forks 5, flags 8, self_focus 2, recent 12, lessons 8, questions 7) and MAX_BRIEFING_CHARS = 12000.
3. render(b: Briefing) -> str: fixed section titles in the order above (the model text's title is exactly "Your notes from the last turn (unverified)", because it is the planner's own prior reply and spec 4.9a says the planner must not read its own guess as a fact), each list clipped to its cap with a trailing "(+N more)" line when clipped, each line clipped to 160 characters, the whole text cut at MAX_BRIEFING_CHARS with a final line "(briefing cut)" when cut. The questions render as "Q-ID: answer".
4. PLANNER_SYSTEM_PROMPT: a system prompt that states the planner's job (finish the level in few moves; the score is min(human/agent, 1)^2 so moves matter), the input sections, and the reply JSON contract from plan section 3 exactly: keys model, facts, goals, segment (up to <horizon> steps, actions "A1".."A7" or "click #<id>"), skill ({"name": one of S-AROUND, S-COLLECT, S-EXPLORE, S-ATTACK, S-WAIT, "args": {...}}), question (one of Q-BLOCK, Q-REWARD, Q-HARM, Q-FORK, Q-CHECK, Q-SAFE, Q-UNKNOWN), labels ([{"object": id, "label": text}]), notes. Say that segment and skill are alternatives, that ids are valid this turn only, that the simulator checks the segment before any real move, and that a rejected segment comes back with the failing step named. Keep the facts and goals shapes identical to DRIVE_SYSTEM_PROMPT so the existing parser applies. Under 2200 characters. Plain English, no em-dashes.
5. planner_messages(b: Briefing, grid, prev_grid) -> list[VMessage]: [system, user] where the user content is the rendered text, the screen image, and the diff image when prev_grid is given (reuse screen.png, screen.diff_png, screen.data_url the way turn.build_messages does). Substitute the horizon into the system prompt.

Part B, arcagent/mind/contract.py (extend; every existing test keeps passing):
6. @dataclass(frozen=True, slots=True) SkillCall: name: str; args: dict[str, object]. Constants SKILL_NAMES = ("S-AROUND", "S-COLLECT", "S-EXPLORE", "S-ATTACK", "S-WAIT") and QUESTION_IDS = ("Q-BLOCK", "Q-REWARD", "Q-HARM", "Q-FORK", "Q-CHECK", "Q-SAFE", "Q-UNKNOWN").
7. Reply gains: skill: SkillCall | None = None; question: str = ""; labels: tuple[tuple[int, str], ...] = (). Keep plan with its meaning (the segment).
8. parse_reply(text, scene, available, *, max_steps: int = MAX_PLAN_STEPS) -> Reply: "segment" is read as the plan (when both "segment" and "plan" are present, "segment" wins); steps beyond max_steps are dropped with one reason; "skill" with a name in SKILL_NAMES and args a dict (or missing, then {}) becomes Reply.skill, else dropped with a reason; when both a plan and a skill are valid, keep the plan and drop the skill with the reason "segment given"; "question" in QUESTION_IDS becomes Reply.question, else dropped with a reason; "labels" entries with an integer object id on screen and a non-empty string label become (id, label[:30]), others dropped with a reason. Document every rule in the module docstring's contract list.

Tests: render clips each section at its cap with the "(+N more)" line, cuts at MAX_BRIEFING_CHARS, keeps section order, and the questions render as "Q-ID: answer"; planner_messages carries two images with prev_grid and one without, and the system prompt contains the horizon; parse_reply: segment alias, segment wins over plan, max_steps cut, skill parsing and the unknown-name drop, "segment given" drop, question parsing and drop, labels parsing with truncation and the off-screen drop; every existing test in tests/mind/test_contract.py and tests/mind/test_turn.py still passes.`,
  },
  {
    key: 'wp8', wt: 'C:/arc-agent-wp8', branch: 'v2/wp8-skills',
    files: 'arcagent/mind/skills.py (new), arcagent/mind/kinematics.py (additive public method only), tests/mind/test_skills.py (new)',
    brief: `Spec section 4.9d (strategy skills, Kameron's examples: "enemy blocks path, then kill it", "pitfall blocks path, then replan route around it", collect like pacman's fruit before completing the level), 4.6b (route over targets), 4.11b (checks). Code map sections: facts-predict (Kinematics API: gaits, body, reachable, steps_to, field, _walk_back, KIN_LIMIT; Thing fields; click_cell in objects.py), goals-search (route.route signature; Goal fields; goal_sigs). Read arcagent/mind/kinematics.py, arcagent/mind/route.py and tests/mind/test_kinematics.py (for how gaits are set up through facts) in full.

Build arcagent/mind/skills.py:
1. Role = Literal["harm", "collectible", "blocker", "mover", "controlled", "unknown"].
2. @dataclass(frozen=True) SkillContext: scene: Scene; mem: FactMemory; kin: Kinematics; goals: Sequence[Goal]; roles: Mapping[str, Role] (thing kind -> role); flags: Sequence[tuple[str, float, float]] (kind, value, cost) best first; can_click: bool; mover: str | None (the controlled thing's kind or sig name that Kinematics knows); click_cell: Callable[[Thing], tuple[int, int]]; max_steps: int.
3. @dataclass(frozen=True) SkillResult: name: str; actions: tuple[Action, ...]; why: str; applies: bool.
4. run_skill(name: str, args: Mapping[str, object], ctx: SkillContext) -> SkillResult. Unknown name -> applies False with why "unknown skill". Every skill returns at most ctx.max_steps actions and never raises on a scene it cannot handle (applies False with a reason).
5. S-AROUND: target = the thing named by args["target"] (an id) if present and on screen, else the target of the first open reach or cover goal in ctx.goals whose mover is ctx.mover; avoid = every cell of every thing whose kind has role "harm" plus the cells in args.get("avoid_cells", []) if given; path = breadth-first search over the mover's top-lefts using the gaits from ctx.kin.gaits(ctx.mover), where a top-left fits only if ctx.kin's fit test passes AND no cell of the body is in avoid (add one additive public method to Kinematics for this: reachable_avoiding(scene, name, avoid: frozenset[Cell], *, limit: int = KIN_LIMIT) -> dict[Cell, tuple[int, Cell | None, Action | None]] mapping each top-left to (moves, previous top-left, action taken), built from the same fit and step helpers reachable uses; do not change reachable); the actions are the first max_steps moves of the shortest path to any top-left that touches the target (overlaps or 4-adjacent, the same rule field uses). applies False when there is no mover, no target, or no path.
6. S-COLLECT: targets = things whose kind has role "collectible"; order them with route.route(start, targets, dist) where dist(a, b) is the kinematic distance between top-lefts via ctx.kin (steps_to or field; read which fits; cells apart as the fallback when None); the actions are the moves to the first target in that order (through reachable_avoiding with the harm cells avoided), cut to max_steps. applies False when there are no collectibles or no path.
7. S-EXPLORE: the first flag in ctx.flags whose kind is on screen; if ctx.can_click, one click at ctx.click_cell(thing); else the moves to touch it (as S-AROUND does toward it). applies False when there are no flags on screen.
8. S-ATTACK and S-WAIT: return applies False with why "not built in this version" (they need the health model); keep the names so the contract's skill list stays valid.

Tests: build scenes and facts the way tests/mind/test_kinematics.py does so the mover has gaits; S-AROUND finds a path that avoids a harm thing's cells and returns no path when the harm thing blocks the only corridor; S-COLLECT orders two collectibles by distance and returns the moves to the nearer one; S-EXPLORE clicks when it can and walks when it cannot; unknown skill and the unbuilt skills return applies False; max_steps cuts; reachable_avoiding equals reachable (move counts) when avoid is empty.`,
  },
  {
    key: 'wp9', wt: 'C:/arc-agent-wp9', branch: 'v2/wp9-logging',
    files: 'arcagent/mind/runlog.py (docstring and any additive helper), scripts/mind_metrics.py (extend), tests/mind/test_runlog.py (extend), tests/test_mind_metrics.py (extend)',
    brief: `Spec section 4.13 (move record, summaries, reports; calibration in bins of 0.1; moves against typical; gate trips; handoffs; planner calls and tokens). Code map section: runtime-kaggle (RunLog.action and RunLog.event contracts; what mind_metrics.py reads and computes and how its tests are built). Read arcagent/mind/runlog.py, scripts/mind_metrics.py and tests/test_mind_metrics.py in full.

Part A, arcagent/mind/runlog.py: the module docstring is the log contract. Add to it, under a heading "Planner agent (v2) fields", the new action-record fields: self_rec (dict from SelfRecord; the name is self_rec because RunLog.action's first parameter is self and a field named self would collide), focus (dict from FocusRecord), hypothesis_id, segment_id, fork_id, planner_call, p_step (the probability of this move's prediction, 4 decimals, null when none), p_path, prev_matched (whether the previous move's prediction matched, null on the first move of a level instance), gate ({"seconds": float, "tokens": int, "trip": str | null}); and the new events: segment (step, id, source, length, p_path, ev, goal), handoff (step, rule, method, reason), harm (step, kind, label, p), consolidate (level, promoted, sequences, typical), longterm (game_id, rules, sequences; game_id because game is an envelope key and RunLog raises ValueError on it), fork_open (step, id, goal, paths), branch_failed (step, fork, index, reason), return_route (step, fork, length), replan_root (step, reason), gate_trip (step, gate, seconds, tokens), planner_call (step, call, seconds, tokens, segment_len, skill, question, dropped), flag (step, kind, state, source, value, cost). RunLog already accepts any keyword fields, so the code change is the docstring plus, if _plain does not already handle them, dict and list values of plain types (check with a test that an action record with a nested dict round-trips).

Part B, scripts/mind_metrics.py: add a function v2_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] computed from one game's records and merged into the per-game output under the key "v2", with: calibration: a list of 10 bins {"bin": "0.0-0.1" .., "n": int, "matched": int, "rate": float | None} pairing each action record's p_step with the next action record's prev_matched (same level instance; skip pairs across a level_up or game_over event); moves_vs_typical: per completed level {"level", "moves", "typical", "ratio"} from consolidate events (moves is the count of action records in that level instance); gates: {"trips": int, "by_gate": {name: count}, "seconds_total": float, "tokens_total": int} from the gate field and gate_trip events; handoffs: {rule: count}; planner: {"calls": int, "tokens": int, "segments": int, "mean_segment_len": float | None, "skills": {name: count}, "questions": {id: count}}; forks: {"opened": int, "failed": int, "returns": int, "root_replans": int}; harm: {"events": int, "by_kind": {kind: count}}. Every field present even when the log has none of the records (zeros and empty dicts), so a report never KeyErrors. Wire it into the existing per-game and run-level summary where the other metrics are assembled; read the current structure and follow it. Keep the script's CLI unchanged.

Tests: runlog: an action record with the new nested fields round-trips as JSON; mind_metrics: a synthetic JSONL game with a handful of action records, one level_up, one consolidate, two gate trips, one handoff, one planner_call and two forks gives the expected v2 numbers, and a game with no v2 records gives the zero shape.`,
  },
]

const IMPL_SCHEMA = { type: 'object', properties: { status: { type: 'string' }, commits: { type: 'string' }, interface: { type: 'string' }, tests: { type: 'string' }, concerns: { type: 'string' } }, required: ['status', 'commits', 'interface', 'tests'] }
const REVIEW_SCHEMA = { type: 'object', properties: { verdict: { type: 'string' }, findings: { type: 'string' }, gates: { type: 'string' } }, required: ['verdict', 'findings', 'gates'] }

const reviewPrompt = (p, impl) => `Review the branch ${p.branch} in the worktree ${p.wt} (absolute paths only). The implementer reported: status ${impl.status}; commits ${impl.commits}; interface: ${impl.interface}; concerns: ${impl.concerns || 'none'}. Read the diff with "git -C ${p.wt} diff agent-v2...HEAD" (and "git -C ${p.wt} log --oneline agent-v2..HEAD"). The contract is the package brief below, plus ${PLAN} section 4 (${p.key}) and the spec sections the brief names in ${SPEC}. Check every numbered requirement against the code and the tests: names and signatures exact, formulas and constants as stated, bounds present, failure modes handled as stated, tests assert the stated values (a test that only checks "no exception" does not count), docstrings state intent and contract without em-dashes, and no file outside the ownership list changed. Run from ${p.wt} with cd: "${PY} -m ruff check .", "${PY} -m pyright", and the package's own test file(s) with "${PY} -m pytest -q -p no:cacheprovider <test files>". Return verdict APPROVE or CHANGES, a numbered list of findings (file:line, what is wrong, what the contract says), most severe first (an APPROVE may still list minor notes), and the gate outputs' last lines. Do not edit files.\n\nPackage brief:\n${p.brief}`

const fixPrompt = (p, impl, review) => `You are fixing review findings on branch ${p.branch} in the worktree ${p.wt} (absolute paths only; interpreter ${PY}; files you may change: ${p.files}). The implementer's report: ${impl.interface}. The reviewer's findings, most severe first:\n${review.findings}\n\nFix every finding that names a contract violation (wrong formula, missing requirement, wrong signature, missing or weak test). For a finding you believe is wrong, leave the code and say why in your report. Keep the same method: failing test first where a test is missing, then the fix. Gates from ${p.wt} with cd: "${PY} -m ruff format <changed files>", "${PY} -m ruff check .", "${PY} -m pyright", full "${PY} -m pytest -q -p no:cacheprovider" to C:/Users/Sagac/AppData/Local/Temp/claude/${p.key}-fix-pytest.txt (read the last 3 lines). Commit on ${p.branch} with a message in the repo's style (imperative title, two lines of why, then the line "${SESSION}", no Co-Authored-By). No subagents. Return status, commit hash, what you changed per finding, what you declined and why, and the pytest last line.\n\nOriginal brief for reference:\n${p.brief}`

const results = await pipeline(
  PACKAGES,
  p => agent(`${common(p)}\n\nPackage brief:\n${p.brief}`, { label: `implement:${p.key}`, phase: 'Implement', schema: IMPL_SCHEMA })
    .then(impl => ({ p, impl })),
  ({ p, impl }) => {
    if (!impl || impl.status === 'BLOCKED') return { p, impl, review: null }
    return agent(reviewPrompt(p, impl), { label: `review:${p.key}`, phase: 'Review', schema: REVIEW_SCHEMA })
      .then(review => ({ p, impl, review }))
  },
  ({ p, impl, review }) => {
    if (!impl || !review || review.verdict.toUpperCase().startsWith('APPROVE')) return { p, impl, review, fix: null, rereview: null }
    return agent(fixPrompt(p, impl, review), { label: `fix:${p.key}`, phase: 'Fix' })
      .then(fix => agent(reviewPrompt(p, { ...impl, concerns: `after a fix round: ${fix}` }), { label: `rereview:${p.key}`, phase: 'Fix', schema: REVIEW_SCHEMA })
        .then(rereview => ({ p, impl, review, fix, rereview })))
  },
)

const rows = results.filter(Boolean).map(r => ({
  key: r.p.key, branch: r.p.branch, worktree: r.p.wt,
  status: r.impl ? r.impl.status : 'NO_RESULT',
  commits: r.impl ? r.impl.commits : '',
  interface: r.impl ? r.impl.interface : '',
  tests: r.impl ? r.impl.tests : '',
  concerns: r.impl ? r.impl.concerns : '',
  verdict: r.rereview ? `${r.review.verdict} then ${r.rereview.verdict}` : (r.review ? r.review.verdict : 'not reviewed'),
  findings: r.rereview ? r.rereview.findings : (r.review ? r.review.findings : ''),
  fix: r.fix || '',
}))
log(`${rows.length}/${PACKAGES.length} packages reported; verdicts: ${rows.map(r => `${r.key}=${r.verdict}`).join(', ')}`)
return rows
