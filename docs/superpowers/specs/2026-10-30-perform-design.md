# Perform Design: Sequence List, Inhibition, Performance (Part 6 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Perform" section, pasted 2026-10-30. The source governs where it
differs from the current code. Parts 1 to 5 cover the stores, retrieval, metacognition,
spatial tools, and planning that this part uses.

## 1. Requirements (from the source)

The section has three parts.

**Sequence list**

- **Q1.** A list of sequential steps, in order, that completes a task.
- **Q2.** The list acts as the action list the agent uses.

**Inhibition**

- **I1.** A neural network runs alongside the rest of the system.
- **I2.** It connects objects to their adverse events, actions, and experiences that
  produce negative outcomes for the player.
- **I3.** It stops simulations and planning from taking an action that leads to an
  immediate, near-future negative outcome.

**Performance**

- **R1.** The agent writes code and performs actions to make progress in the level.
- **R2.** It uses the sequence list to write the code or to choose the actions.

The source names a neural network for inhibition and does not say how big it is, what it
is trained on, or whether it is learned online. Section 5 covers that.

## 2. What the current code does

**Sequence list.** The fluid planner keeps a plan as a list of steps (`_plan`, a list of
`PlanStep`). Each move pops the next step (`fluid.py`, step 4). The indicator adds a
sequence goal (`SEQ_SOURCE = "indicator"`), and its branches `sequence step` and
`sequence try` choose clicks in order (`fluid.py`, step 6). A level change clears the plan
(step 2). So the plan is the action list, but it lives for one level and is then dropped.
Nothing carries a successful sequence into a later level or game.

**Inhibition.** There are two implementations.

- **Program path.** `DisconfirmInhibition` (`modules/inhibition.py`) records each
  (frame, action) pair that went badly, for one of three reasons: `game_over`, `noop`, or
  `disconfirmed`. Veto sets are kept in working memory under a frame hash with a TTL. This
  is a lookup table, not a network.
- **Fluid path.** The fluid agent asks its inhibition module for vetoes on each option
  (`modules.inhibition.veto`, `fluid.py`). Plan steps honor four reasons: `game_over`,
  `no_effect`, `seen_no_effect`, and `undoes_progress` (`_PLAN_VETOES`). The search never
  expands a predicted game over (`search.py`, the first branch rule). The search also
  checks each predicted scene against the scenes already seen.

So inhibition runs in two places, search and the move loop. It is not a network. It tracks
four kinds of outcome, and the first three are the only ones learned from the game itself:
death, no change, and disconfirmed prediction. `undoes_progress` is the fourth.

**Performance.**

- **Scored path (hybrid).** The fluid agent picks each action and writes no code. The
  explorer picks actions from its graph. Nothing in the scored path writes code.
- **Code path (LLM).** The synthesizer (`synthesizer.py`), the program sandbox
  (`sandbox.py`, with allowlisted imports and a timeout), `toolbox.py`, and `tool_fix.py`
  let the LLM write programs and tools. `llm_drive.py` runs them. This path is separate
  from the hybrid.

Status: partial for Q1 and Q2 (a plan exists, but it doesn't persist). Partial for I2 and
I3 (rules, not a network, and only four outcome types). For R1 and R2, the scored path
does not write code. The code path does, but it is a separate mode.

## 3. Proposed structure

**Sequence record.** A sequence is an ordered list of steps that completed a goal, kept
with what it did.

- `id`: stable within the store.
- `goal`: the goal it completes, as a canonical goal record (Part 1, section 3).
- `steps`: the moves in order, each as an action key plus the features it was taken on.
- `runs`, `successes`: how often it has been tried and how often it completed the goal.
- `scope`: `level`, `game`, or `long`, as in Part 1.

A sequence is written when a plan completes its goal. It is read when the same goal comes
up again, and it is a candidate in the retrieval from Part 2.

**Inhibition store.** Inhibition becomes a set of adverse rules, kept in the Part 1 store
and retrieved through Part 2's index.

- `subject`: a thing kind (Part 1, D1), or `any`.
- `action`: an action key, or `any`.
- `event`: the adverse outcome: `death`, `harm`, `no_change`, `undoes_progress`, or
  `disconfirmed`.
- `evidence`: how often the rule's pattern led to the event.
- `status`: `candidate` or `validated`, as in Part 1.

A rule vetoes an action when its subject is present, its action matches, and its event is
expected within the search horizon. A veto is logged with the rule it came from.

**Network hook.** The inhibition store exposes one function: given a scene and an action,
return the predicted adverse events and the rules behind them. A network, if built later,
replaces the function and keeps the same return shape. This keeps the source's network
behind an interface we can test first.

**Performance.** The move loop writes a record of the step that produced each action:
which list (plan, sequence, explorer) and which step. Code written by the LLM path stays
in the sandbox and is logged the same way.

## 4. Behavior

- **Sequence write.** When a plan completes its goal, the sequence record is written with
  `runs` and `successes` set to 1. A level change does not clear it.
- **Sequence read.** When a goal is proposed and a sequence for it exists, the sequence is
  tried first, as a plan. A break ends it, and the failure is counted in `runs`.
- **Sequence validation.** A sequence that succeeds on three runs in a row is validated
  (D3).
- **Inhibition veto.** A vetoed action is skipped unless it is the only move left, in which
  case the forced branch takes it and logs `forced`. This is the same rule the fluid agent
  already follows.
- **Search.** Search prunes branches where an adverse rule fires with an event the horizon
  reaches. The predicted game over stays the first and hardest rule.
- **Log.** Each veto writes its rule id and event. Each sequence step writes its sequence
  id.

## 5. Decisions I made

These are my calls, changeable without changing section 1.

- **D1. Sequences persist beyond one level.** A sequence is written on success and kept
  in the store. Reason: Q1 says the list completes a task, and a task can recur at a later
  level or game. Cost if wrong: stale sequences that the validation step must retire.
- **D2. Inhibition is rules first, network later.** The source names a network. This build
  stores adverse rules behind the interface in section 3. Reason: we have no training set
  of adverse events for any game, and a rule can be checked in a run log. Cost if wrong:
  a network that would have generalized better is delayed. Q2 asks whether to build it
  now.
- **D3. A sequence is validated after three successes in a row.** The source gives no
  number. Three matches the solid threshold used elsewhere, scaled down for sequences,
  which are checked against a whole goal rather than one fact.
- **D4. Harm is an adverse event only when a counter or bar falls.** This needs a
  definition of a counter. The fluid path already masks move counters as noise
  (`noise.py`), so their cells are known. Reason: it is the only harm signal the games show that is not death.
  Cost if wrong: harm events that are missed or invented. Q3 asks for a definition.
- **D5. The scored path stays action-only in this build.** The code path stays a separate
  mode. Reason: the scored entry is model-free, and merging the code path would change its
  eligibility. The open Host question on the code path's open-weights status is part of
  this decision. Cost if wrong: R1 is unmet on the scored path until the decision changes.

## 6. Test plan

**Sequence persistence.** A test that a plan completes its goal, the sequence is written,
and it survives a level change. A second test that a recurring goal tries the stored
sequence first.

**Sequence validation.** A test that three successes validate a sequence and a failure
resets the run count.

**Inhibition.** A fixture with a hazard that causes a death. The test asserts the rule is
written, the same action is vetoed on the next visit, and the forced branch still takes it
when it is the only move.

**Search.** A test that the search does not expand a branch whose predicted adverse event
is in the horizon, and does expand it when the rule is not yet validated.

**Log.** A test that each veto and each sequence step names its source.

**Measured effect.** On recorded runs, count repeats of a known adverse action on the same
scene, before and after the rule store. That is the measure of I3. Count moves to the
first win on games with sequences, before and after, for Q1.

**Held-out games.** Same rule as Parts 1 to 5.

## 7. Open questions for Kameron

- **Q1.** Is a sequence one ordered list per goal, or one list per level that holds every
  goal's steps in order? The first is what section 3 builds.
- **Q2.** Should the inhibition network be built now, or after the rule store has run on
  the held-out games? The source names a network; D2 defers it.
- **Q3.** What counts as harm? Options are a counter or bar falling, a thing disappearing,
  or only death and no-change. D4 uses a falling counter or bar.
- **Q4.** Should the code path (LLM writes programs) run beside the model-free agent, or
  replace the part of it that performs actions? D5 keeps them separate until the Host
  question is answered.
