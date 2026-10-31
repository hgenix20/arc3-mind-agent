# Curiosity Design (Part 9 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Curiosity Questions" section and the "Planning Process" loop, pasted
2026-10-30. The source governs where it differs from the current code. Parts 1 to 8
cover the stores, retrieval, metacognition, spatial tools, planning, perform, grit, and
prediction that curiosity draws on.

## 1. Requirements (from the source)

**Curiosity flags**

- **C1.** Each item on the map carries flags.
- **C2.** A flag can be labeled `check`. A mouse click or the character touching a
  `check` flag identifies the outcome.

**Four stages of curiosity**

- **H. Hook (attention and recognition).** The agent detects something novel or out of
  place, then checks memory to see whether the item matches a known category.
- **I. Investigation (decoding).** The agent works out what the item does, how it works
  (buttons, textures, moving parts), and why it is there (context, design, material).
- **A. Personal assessment (value).** The agent weighs the item's value against its cost.
  The source adds a social check: what others think of the item.
- **M. Mental simulation (projection).** The agent places the item in its routine and asks
  what problem it solves.

**Planning process loop (given with this section)**

Framing, then objectives and rules, then alternative generation, then modeling and
trade-offs, then execution, then evaluation and learning, then back to framing.

Two of the source's ideas do not fit a single-player game agent, and section 5 says so.
The social check (what others think) has no counterpart here, because the agent plays
alone. The source also leaves open where the planning loop runs. This draft treats it as
the loop that the agent's own planning follows, and Q4 asks you to confirm that reading.

## 2. What the current code does

**Flags.** Nothing on a map item is labeled `check`. The closest thing is the search's
frontier: `SearchResult.frontier` holds the unknown `(label, subject)` pairs the search met,
in first-met order (`search.py`). A frontier pair is an unchecked outcome, which is what a
`check` flag would be. The frontier is not stored on the item, and it is cleared on a level
change (`fluid.py` step 2).

**Hook.** Novelty is detected in three places.

- `kinds.KindBook` gives each thing an identity class across looks, so a changed look is
  matched to a known kind. This is "have I seen this before" for appearance.
- A pair with no fact is an unknown, so a new sig is novel by default.
- The novelty fallback (`fluid.py` step 7, `explore novel`) searches for states the agent
  has not seen on this level.

**Investigation.** Touch and near contexts are learned (`facts.touch_contexts`,
`facts._learn_touch`). A touch that changes something is recorded as a fact keyed by the
action, the subject, and the context. Modes (`modes.py`) learn what a click on a thing
does to its selection marks. Blockers and walkable colors are learned per sig. The
investigation is done by moves that touch things, which is what the source describes.

**Assessment.** `GoalBook.ranked` weighs a goal by its prior, its salience, and its
distance (`prior * salience / (1 + cost)`). That is a value and a cost, computed for goals
rather than for items. Nothing scores an item by its own value.

**Simulation.** `search` predicts a child scene for each action (`predict`), and the goal
distance decides which scenes it expands. A goal whose plan reaches the item is the
closest thing to "how would I use this." The search does not ask what problem the item
solves for the agent; it asks whether a goal holds.

**The planning loop.** The fluid agent runs the same sequence on each move, in its
decision ladder (`fluid.py`, steps 1 to 7). Framing is the level model
(`FactMemory.new_level`, the goal book). Objectives and rules are the goals and the facts.
Alternatives are the goal guesses and the frontier. Modeling and trade-offs are `predict`
and `search`. Execution is the move. Evaluation and learning are `learn`, `check`,
`fluid_sleep`, and `learn_win`. A level change starts framing again. The loop exists; it
is implicit in the code rather than named.

**Test choice.** The fallback test (`tester.choose_test`, `fluid.py` step 6) picks an
unknown pair from the frontier to test, using the Voi module's disagreement measure
(`modules/voi.py`). It runs only after the plan, the sequence steps, and the partial plan
have had a chance. A `check` moves through this same path.

Status: partial. The stages are present in the code as separate mechanisms. There is no
item-level record that ties them together, and no flag state.

## 3. Proposed structure

**Flag record.** Each thing on screen has a flag record, kept in the level model.

- `thing`: the thing's id (tracker id) and its kind (Part 1).
- `state`: `new`, `check`, `checked`, or `ignored`.
- `source`: `novel_look`, `unknown_pair`, or `changed`.
- `outcome`: the fact keys learned from the check, once there are any.

A flag is set to `check` when the thing's look is new (`novel_look`), or when a frontier
pair names it (`unknown_pair`). A flag is set to `checked` when a touch or click on the
thing has written a fact, and the outcome is the fact keys.

**Assessment.** A `check` flag gets two numbers: a value and a cost.

- `value`: the expected information from the check, from the Voi measure. Where the flag's
  thing is a goal sig, the goal's prior adds to it (the source's "do I need this").
- `cost`: the moves to reach the thing (kinematics distance), and one move for the check
  itself.

**Projection.** Before a check is taken, the search runs once with the thing's assumed
outcome, to see whether the outcome reaches a goal in a few moves. A check whose projected
outcome reaches a goal is ranked above one that does not. This is the source's mental
simulation (M), carried out by the search the planner already has.

**Planning loop.** The loop is written as the stage names of `fluid.py` with each stage's
owner, so a run log shows which stage made each move. No new logic is needed for the loop
itself. Section 4 gives the log events.

## 4. Behavior

- **Set.** A new look or a frontier pair sets a flag to `check` and logs `flag_set` with
  its source.
- **Check.** A move that touches or clicks a `check` thing logs `check` with the thing, the
  action, and the outcome facts. The flag becomes `checked`.
- **Ranking.** Among `check` flags, the one with the highest value per cost is taken first,
  but only after the plan and sequence steps (`fluid.py` steps 4 and 5) have had their
  turn, as today (D3).
- **Stage in the log.** Each move's log entry names its loop stage: framing, objectives,
  alternatives, modeling, execution, or evaluation.
- **Level change.** Flags reset with the level model. Facts from checks stay (Part 1).

## 5. Decisions I made

These are my calls, changeable without changing section 1.

- **D1. Flags are derived, not designer-set.** The source has a designer label the flag
  `check`. The agent can't see a designer, so the flag is set from what the agent does not
  know yet: a new look or an unknown pair. Reason: this is the only way to set flags
  without a human in the loop. Cost if wrong: flags for things that don't matter.
- **D2. The social check is not built.** Reason: the agent plays alone and has no other
  players' opinions. The closest analogue is a cross-game store (Part 1), and that is a
  store of the agent's own experience, not social validation. Cost if wrong: none for the
  agent; the source's stage is simply absent.
- **D3. Checks wait behind the plan.** A `check` is taken only after the plan and sequence
  steps. Reason: the current ladder already puts exploration after exploitation, and
  changing that order is a larger change than the requirements need. Cost if wrong: an
  item that would have changed the plan is checked late.
- **D4. Value is the Voi measure, plus the goal prior when the item is a goal sig.** Reason:
  Voi is already computed for the fallback test, and the goal prior is already computed for
  goals. Cost if wrong: value is a rough sum of two measures on different scales. The
  calibration test (section 6) checks the sum.
- **D5. The loop is named in the logs, not rebuilt.** Reason: the stages exist. Rebuilding
  them would break the tests that already cover them. Cost if wrong: a loop that is named
  but not enforced.

## 6. Test plan

**Flag set.** A fixture with a new look and a frontier pair. The test asserts each sets a
`check` flag with the right source.

**Check outcome.** A fixture where a click on a `check` thing changes a sprite. The test
asserts the flag becomes `checked` and the outcome fact exists.

**Ranking.** Two `check` flags, one with higher value per cost. The higher one is taken
first, after a plan step is available.

**Projection.** A fixture where a check's assumed outcome reaches a goal in two moves, and
a second check whose outcome reaches nothing. The first is ranked above the second.

**Calibration.** On recorded runs, compare the value of each check with the information the
check gave (the number of facts it wrote). A value that rises with the facts written is a
sign the Voi-plus-prior sum is ranking usefully (D4).

**Loop log.** A test that each move's log entry names exactly one loop stage.

**Held-out games.** Same rule as Parts 1 to 8.

## 7. Open questions for Kameron

- **Q1.** Should flags be derived from unknowns only (D1), or should a designer-style label
  also be possible for a game where the agent is told which items to check?
- **Q2.** Do you want the social check (D2) at all? If so, what would a stand-in for it be
  in a game with no other players?
- **Q3.** Should a `check` ever run before the plan (D3)? The current order favors the plan.
- **Q4.** Is the planning loop meant for the agent's own planning, as this draft reads it,
  or for the process that produces this spec? The two need different owners.
