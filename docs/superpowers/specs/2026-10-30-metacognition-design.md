# Metacognition and Self-Identity Design (Part 3 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Metacognition" section, pasted 2026-10-30. The source governs where it
differs from the current code. Parts 1 and 2 (`2026-10-30-agent-memory-design.md`,
`2026-10-30-fluency-retrieval-design.md`) define the stores and retrieval that this part
reads.

## 1. Requirements (from the source)

**Self-identity**

- **S1.** The agent is a player. It is represented as "player", "I", or "Me", depending on
  what information it receives.
- **S2.** When a mouse click and a character are both present, "I" and "Me" stand for the
  mouse click, and "My character" stands for the actionable figure the agent moves
  through controls.
- **S3.** When a mouse click is missing and a character is present, "I" and "Me" stand
  for the character.
- **S4.** When a mouse click is present and a character is not, "I" and "Me" stand for the
  mouse click.

**Metacognition**

- **M1.** Metacognition identifies how the agent is thinking, what it is thinking about,
  and adjusts in real time.

The source does not define "mouse click present," "character present," what "how the
agent is thinking" covers, or which adjustments are allowed. Section 5 proposes
definitions and marks them as decisions.

## 2. What the current code does

**Self-identity**

- There is no self record. The code never labels anything as "I," "me," or "my."
- **Controlled things** are learned from movement, not declared. `goals.controlled_sigs`
  returns the sigs whose things moved under actions.
- **Active controller.** `ModeTracker` (`modes.py`) tags one controlled thing as active
  after a non-click action moves exactly one controlled thing, and only when two or more
  controlled things are on screen. With one controlled thing, its facts stay in context
  `any` and no active tag is set.
- **Clicks.** Action 6 with coordinates is handled by `clicked_thing` and `remap_click`
  (`fluid.py`). A click is an action, not a self.
- **Mouse cursor.** I searched `arcagent/mind` for "cursor" and found nothing. The code
  does not detect a cursor drawn on screen.

Status: partial. The active controller covers S3 when several controlled things exist. S2
and S4 have no implementation, and nothing in the code states which of the two is "I."

**Metacognition**

- **Program confidence.** `ConfidenceMetacognition` (`modules/metacognition.py`) reports
  programs to drop on misses in a row or low fit. It works on the belief of the LLM and
  program path, which the hybrid does not use.
- **Fact breaking.** In the fluid path, `FluidMetacognition` (`fluid_modules.py`) sets the
  fact break threshold, read once when the agent is built (`MissMetacognition` returns
  `BROKEN_MISSES`; its ablation null returns `NEVER_BROKEN`). Facts break after that many
  misses in a row. `ConfidenceMetacognition` is a different module, on the program path.
- **Progress monitor.** `progress.py` names each move `milestone`, `progress`, `regress`,
  or `motion` from goal distances.
- **Stall rule.** `hybrid.py` hands the level to the explorer after `STALL_MOVES` = 25
  fallback moves in a row, or after a fixed number of mind moves per level.
- **Branch labels.** Each chosen move records how it was chosen: `plan`, `commit`,
  `fallback`, `forced`, `reset`, and the tester's labels. The label is logged, and the
  agent reads `last_branch` once, for the hybrid's stall count.
- **Indicator tracking.** `indicator.py` tracks progress indicators and records each
  step as right or wrong.

Status: partial. Each signal is computed and logged, and a few drive a single decision
(the fact break threshold, the stall handover). No component holds a current picture of
what the agent is thinking about, and no component weighs several signals together to
choose a strategy. The branch label is written down but not read by a decision, except
the stall count.

## 3. Proposed structure

Two records, updated once per observation, plus one controller that reads them.

**Self record.** One per observation.

- `role`: `character_and_click`, `character`, `click`, or `none`. This is the S2 to S4
  rule applied to the observation.
- `I`: the referent of "I" and "me" for this observation. `click` in the first case,
  `character` in the second, `click` in the third, `none` in the fourth.
- `my_character`: the tracked id of the controlled thing when one is present, otherwise
  none. Drawn from the active controller when set, else from the single controlled thing.
- `click_at`: the last click coordinate when the action was a click, otherwise none.

**Focus record.** One per observation, the agent's current picture.

- `level`, `goal` (the goal being pursued and its status), `target` (the thing or
  region the goal is about), `mode` (the branch that chose the last move), `budget` (from
  the budget module), and `signals` (the progress, regress, stall, and miss counts).

**Controller.** A fixed table of rules. Each rule reads the focus record and returns one
of: keep the current strategy, switch goal, switch from the mind to the explorer, reset
fact breaks, or re-probe a goal. Every returned action is logged with the rule that
fired.

## 4. Behavior

- **Self record.** Recomputed every observation. The three cases in S2 to S4 are checked
  in the order the source gives. The record is written to the run log so each choice can
  be audited against its self.
- **Click present.** A game counts as having a click when its available actions include
  action 6 (this is the reading in D2).
- **Character present.** A game counts as having a character when `controlled_sigs` is
  non-empty on the current scene.
- **Focus record.** Updated after the goal and mode are chosen for the move, before the
  next move.
- **Controller rules.** Evaluated once per move, in a fixed order. The first rule that
  fires wins. Rules are written as code, with thresholds in one place. No rule is learned
  in this build.

## 5. Decisions I made

These are my calls, changeable without changing section 1.

- **D1. The source's rule is followed as written.** When click and character are both
  present, "I" is the click and "my character" is the controlled thing. This is the
  reading that the source states, even though the character is the thing the agent moves.
- **D2. "Mouse click present" means action 6 is available.** The code has no cursor
  detector, and a cursor is not needed to know whether clicks are a legal action. Reason:
  the action list is exact and already exposed. Cost if wrong: a game that draws a cursor
  but lacks action 6 would be misread. Q1 asks for confirmation.
- **D3. "Character present" means the controlled-thing set is non-empty.** Reason: the
  same set already drives the active controller, so the two cannot disagree.
- **D4. The controller is rule-based in this build.** Reason: a rule table can be read in
  a run log and tested by hand. A learned controller would need the phase 2 scorer from
  Part 2 and more training games than we have.
- **D5. Real time means per move.** Reason: the source says "in real-time," and every
  decision the code makes already happens per move. The controller adds no delay beyond
  the rule checks.

## 6. Test plan

**Self record (S2 to S4).** Fixture scenes for each of the four cases, with expected
`role`, `I`, and `my_character`. The fixtures include the edge cases: a click on empty
ground with a character present, and two controlled things with one active.

**Click and character presence on the real games.** Run the detector on the 25 games and
compare with the action survey from 2026-10-29. The six games whose only action is 6
(`ft09`, `lp85`, `r11l`, `s5i5`, `tn36`, `vc33`) should come out as click-only or
character-free. Any other result is a finding to explain.

**Controller decisions.** For each rule, count how often it fires on a run and whether the
move it caused was followed by progress. A rule that fires often and rarely helps is a
candidate for removal.

**Stall rule.** Compare the current 25-move stall rule with a focus-based rule on the
same runs: moves spent after the goal stopped progressing. This is the measure that
decides whether the controller earns its place.

**Held-out games.** Same rule as Parts 1 and 2.

**Unit tests.** Each controller rule has a test with a focus record that fires it and one
that does not. Each log line names the rule that fired.

## 7. Open questions for Kameron

- **Q1.** Is "mouse click present" the same as action 6 being available (D2), or is there
  a visible cursor you want the agent to see?
- **Q2.** When click and character are both present, does the agent still call the click
  "I" on keyboard moves? The source's rule (D1) says yes. Confirm that this is what you
  mean.
- **Q3.** Should the "I" and "my character" labels reach the LLM prompt, the decisions, or
  both? Right now the decision code has no self concept, and I haven't checked how the LLM
  prompt labels things. Labels that only shape wording would not change behavior.
- **Q4.** Which signals should trigger a strategy switch, and in what order? I haven't
  written the rule table yet. Once you give the order, it becomes the table.
- **Q5.** Should the controller also run per level, to decide whether to revisit a goal
  the agent has dropped?
