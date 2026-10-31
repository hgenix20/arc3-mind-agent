# Grit and Perseverance Design (Part 7 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Grit" and "Perseverance" sections, pasted 2026-10-30. The source
governs where it differs from the current code. Part 6 (`2026-10-30-perform-design.md`)
defines the adverse rules that perseverance reads, and Part 5 (`2026-10-30-planning-design.md`)
defines the plan tree that a gate would stop.

## 1. Requirements (from the source)

**Grit: gates**

- **G1.** Stop the agent from spending all its time on one action or task.
- **G2.** Stop the agent from spending an excessive number of tokens on one action or
  task. The source calls this a sign of over-engineering.

**Perseverance**

- **P1.** An adverse outcome can be predicted with some probability. Perseverance can
  override a medium probability and let the agent try the action anyway, because the
  good outcome (the example is a critical hit) also has a probability and may happen.
- **P2.** Perseverance needs a reason. The source gives two: there is no other way around
  (the example is an enemy blocking the path), or going through the adverse outcome gives a
  higher score.
- **P3.** The example: a sword fight where losing is certain unless the player lands a
  critical hit. Perseverance overrides inhibition and attacks anyway, even though the
  player may die.

The source gives no numbers for either the gates or the probability bands. Section 5
states the values I used, and each one is marked as a placeholder until measured.

## 2. What the current code does

**Time gates.** Each gate is per move or per turn, not per task.

- `search.py`: `SEARCH_TIME_S = 1.0` wall-clock cap per search, and `SEARCH_MAX_NODES =
  3000` predictions.
- `fluid.py`: `TURN_DEADLINE_S = 30.0` (the longest a move waits for its turn), and
  `MAX_TESTS_PER_ATTEMPT = 200` test moves per level attempt before the tester is skipped.
- `llm_drive.py`: `DRIVE_TURN_S = 90.0` per drive or fix turn.
- `budget.py`: the move budget is read from the game's own counter bar. It is the game's
  limit, not ours.
- `fluid.py` `STALL_AFTER = 2`: after two failed attempts at one goal in one level, the
  goal is set aside. This is a per-task gate, and it counts attempts, not time.
- `hybrid.py` `STALL_MOVES = 25`: after 25 fallback moves in a row, the explorer takes the
  level. This is a per-task gate on moves.

**Token gates.** `TURN_MAX_TOKENS = 1536` (fluid turn) and `DRIVE_MAX_TOKENS = 4096`
(drive turn) cap each reply. `llm_drive.py` `DRIVE_CUTOFF = 0.1` stops a turn from starting
when less than a tenth of the game's budget is left. I searched `fluid.py`,
`llm_drive.py`, and `budget.py` for a cumulative token count per task and found none. The
caps are per reply, and nothing counts tokens across a task.

**Perseverance.** The closest mechanisms are these.

- **Forced move.** `fluid.py` `_forced` takes the one action left after plan vetoes. It
  never takes an action whose veto reason is in `_PLAN_VETOES`, and that set includes
  `game_over`. So a predicted death is never forced, even if it is the only move.
- **Commit probes.** `fluid.py` step 3 skips the `seen no effect` veto when a goal is
  reached but not yet confirmed. It still honors a predicted game over.
- **Stall handover.** The explorer takes over after a stall. This is a change of method,
  not a override of an adverse prediction.

Vetoes are binary. A rule vetoes or it doesn't. Nothing records a probability for the
adverse outcome, and nothing records a reason for overriding one. The forced move and the
commit probe are the only overrides, and neither logs a probability.

Status: gates are partial (time and attempt gates exist, token gates are per reply only).
Perseverance is missing as a concept. The code has two narrow overrides with no reason
record.

## 3. Proposed structure

**Task record.** Every goal attempt and every level attempt has a record.

- `wall_s`: wall-clock time spent.
- `tokens`: tokens spent on LLM turns for the task (zero on the scored path).
- `moves`: moves spent.
- `progress`: whether a goal distance fell (from the progress monitor, Part 3).

**Gates.** The task record is checked after each move. A gate that trips ends the task
and hands it to the next method (the explorer, or the next goal in the ranked list). Each
trip is logged with the gate's name and the record's values.

**Over-engineering signal.** A task whose `tokens` exceed the expected value for its
goal kind, with no `progress`, is flagged in the log. The flag does not stop the task. It
is an input to the controller in Part 3.

**Adverse estimate.** For each adverse rule (Part 6), the estimate is a Beta posterior on
the probability of the adverse event, given the trials in which the rule's pattern
occurred.

- `hits`: trials where the event happened.
- `trials`: trials where the pattern occurred.
- `p`: the posterior mean under a Beta(1, 1) prior.

**Override.** Perseverance is a rule that returns an override decision. It needs both:

1. A probability band for the adverse event: medium (see D2).
2. A reason, one of:
   - **No other way.** The search with the vetoed move removed returns no plan to the goal.
   - **Score.** The goal's score gain (Section 5, D3) exceeds the expected cost of the
     adverse event.

**Override trials.** A permitted override is a trial. Its result updates the adverse
estimate. The override is allowed again only while the estimate stays in the medium band,
and at most `PERSIST_TRIALS` times per rule per level (a placeholder value, D4).

## 4. Behavior

- **Gate trip.** A gate that trips logs `gate_trip` with its name, the task record, and
  the next method. Nothing is silently skipped.
- **Override.** An override logs `perseverance` with the probability, the reason, and the
  rule id. The move is taken as a normal move.
- **Certain adverse event.** A predicted event at or above the certain band is never
  overridden, and a predicted game over stays in `_PLAN_VETOES`. This keeps the current
  rule that death is never forced, even when it is the only move.
- **Low adverse event.** A rule below the medium band is never overridden. The veto
  holds as it does now.
- **Repeated failure.** When an override ends in the adverse event, the estimate rises. If
  it crosses into the certain band, further overrides stop.

## 5. Decisions I made

These are my calls. The values marked as placeholders have not been measured.

- **D1. Gates are per task, checked per move.** Reason: the source names time and tokens
  per action or task, and the current gates are per move or per reply. Cost if wrong: the
  check runs on every move even when no task is near its limit.
- **D2. Probability bands (placeholder).** Certain: p at or above 0.9. Medium: p from 0.2
  to 0.9. Low: p below 0.2. Reason: the source says "some probability" and "medium level,"
  so the bands need edges, and these are a first guess. Cost if wrong: overrides at the
  wrong probabilities, which the estimate and the trial cap are meant to catch.
- **D3. Score as the reason.** The score for a level is `min(115, (human / agent)^2 * 100)`,
  weighted by level depth, as in the competition scoring. An override on score is allowed
  when the gain in that score from the goal exceeds the adverse cost. The adverse cost is
  the moves lost to a restart, which feeds the same formula. Reason: P2 names score as a
  reason, and this is the score. Cost if wrong: the estimate of moves lost is rough.
- **D4. Trial cap (placeholder).** `PERSIST_TRIALS = 3` override trials per rule per
  level. Reason: enough to see a probability event happen (the critical hit) without
  letting a bad estimate repeat unbounded. Cost if wrong: a real critical hit could need
  more trials.
- **D5. Tokens are logged per task, not gated, until measured.** The token gate is the
  over-engineering flag only. Reason: nothing in the runs tells us the expected token count
  per goal kind, so a gate would use a guessed number. Cost if wrong: a task that
  over-engineers is flagged but not stopped.

## 6. Test plan

**Gate trips.** Fixtures where a task passes its time limit, its move limit, and its
token limit. Each trips its gate, logs it, and hands off to the next method.

**Probability estimate.** A test that the Beta posterior moves toward the observed rate,
and that a rule with no trials stays at the prior.

**Override bands.** A test that a certain event is not overridden, a low event is not
overridden, and a medium event with a reason is overridden and logged.

**Reason.** A test where no plan exists without the vetoed move (the "no other way" reason)
and the override is allowed. A test where a plan exists and the score gain is below the
cost, and the override is refused.

**Repeated failure.** A test that the estimate rises after each failure and that overrides
stop once it crosses into the certain band.

**Sword fixture.** The source's example: a fixture with a critical-hit probability of
about 0.3 and a certain death without it. The override is allowed for up to
`PERSIST_TRIALS`, and it stops after the estimate crosses into the certain band.

**Measured effect.** On recorded runs, count gate trips per task, and count overrides and
how many of them ended the level. The second number is the measure of P3.

**Held-out games.** Same rule as Parts 1 to 6.

## 7. Open questions for Kameron

- **Q1.** Is the time gate per move, per level attempt, or per whole game? The competition
  rules set a runtime limit that I haven't checked for this spec.
- **Q2.** Should the token gate count LLM tokens only, or every model call in the run,
  including any model we ship?
- **Q3.** Are the probability bands (D2) right? I used 0.2 and 0.9 as edges. Give numbers
  or a rule and I'll replace them.
- **Q4.** Is "no other way around" enough as a reason, or should it also need the score
  check in D3?
- **Q5.** Should an override trial count against the game's move budget from `budget.py`?
  A trial that dies costs a life, and a life is a budget unit in some games.
