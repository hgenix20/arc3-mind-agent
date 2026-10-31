## 4. Modules, by section

Each module has a card: status, file, purpose, inputs, outputs, logic, stores, logs, tests.
The logic lines are the build contract. A line that says PLACEHOLDER is a value to set
(section 7).

### 4.1 Perception and scene

**Built modules**

- `arcagent/mind/screen.py` (built, keep). Palette and PNG rendering for review images.
- `arcagent/mind/objects.py` (built, keep). Things, scenes, diffs.
- `arcagent/mind/regions.py` (built, keep). Regions of the grid.
- `arcagent/mind/kinds.py` (built, keep). Identity of a thing across looks.
- `arcagent/mind/noise.py` (built, keep). Masks for counters and moving noise. Kameron,
  2026-10-30: "This seems useful for the new build."
- `arcagent/mind/modes.py` (built, changes). Selection marks and the active controller.
  Changes: its active-controller output becomes the source for "my character" (4.5).

**Not yet built**

- Flag labels for items (4.11).

**M1.1 Frame reader**

- Purpose: turn a frame into the observation the agent uses.
- Inputs: frame layers, game state, level counts, available actions.
- Outputs: grid (64 by 64, values 0 to 15, padding -1), phase, levels completed, levels to win, available actions.
- Logic:
  1. IF the frame has layers THEN the grid is the last layer, padded to 64 by 64 with -1.
  2. IF the frame has no layers THEN the grid is all -1.
  3. Phase is one of: not played, in play, won, game over.
- Stores: none.
- Logs: none (the observation is logged by the move record).
- Tests: a layer smaller than 64 by 64 keeps its values in the top-left and pads the rest with -1.

**M1.2 Scene builder**

- Purpose: find the things in the grid.
- Inputs: grid, the mask of noise cells.
- Outputs: a scene: a list of things, each with id, colour, shape, cells, and bounding box.
- Logic:
  1. Ignore noise cells (M1.4).
  2. Group connected cells of one colour into a thing.
  3. Cells that cover more than a quarter of the grid are background or terrain, not things.
- Stores: none.
- Tests: two touching squares of different colours make two things; a grid of one colour makes one background.

**M1.3 Tracker (identity over time)**

- Purpose: keep the same id for a thing across frames.
- Inputs: previous scene, new scene, the action taken.
- Outputs: the same ids, with new things given new ids.
- Logic:
  1. Match things of the same colour and shape across frames, nearest first.
  2. A thing that changed shape or colour keeps its id if its kind (M1.5) is the same.
  3. A thing with no match gets a new id and is marked new.
- Built from: `objects.py` tracker functions.

**M1.4 Noise mask**

- Purpose: find cells that change for reasons other than the agent's actions (counters, timers, animated backgrounds).
- Inputs: the history of frames on this level.
- Outputs: a set of noise cells, and any counter cells found.
- Logic:
  1. A cell that changes on most frames regardless of the action is noise.
  2. A thin strip whose size shrinks or grows one step per move is a counter (kept for the budget, M4.10).
- Built from: `noise.py`.

**M1.5 Kind (identity class)**

- Purpose: group looks that belong to one kind of thing.
- Inputs: a thing's look (colour, shape) and its history.
- Outputs: a kind id.
- Logic:
  1. A new look is matched to an existing kind if it shares shape and at least one behaviour seen so far.
  2. Otherwise a new kind is created.
- Built from: `kinds.py`.
- Note: the kind is what cross-game rules use (4.2). A look is only the first thing a rule sees.

**M1.6 Mode reader (selection and active controller)**

- Purpose: find what is selected, and which thing the agent is moving.
- Inputs: the previous and current scene, the action, the clicked thing if the action was a click.
- Outputs: the selected thing (if any), the active controlled thing (if any).
- Logic: as built in `modes.py` (a selection mark appears on a clicked thing and vanishes on a second click; the active controlled thing is the one that moves after a non-click action when two or more controlled things exist).
- Change: the output feeds the self record (4.5), not only the facts.

### 4.2 Memory

Memory has five record types and four stores. The records are the same shape across
stores, so one rule can move between stores without being rewritten.

**Built modules**

- `arcagent/mind/facts.py` (built, changes). The current fact table is the start of the
  rule store. Changes: add scope, status, probability, and the canonical key (below).
- `arcagent/mind/goals.py` (built, changes). The goal book is the start of the hypothesis
  store. Changes: each goal gets a path probability (4.7).
- `arcagent/mind/fluid_sleep.py` (built, changes). Level-up bookkeeping. Changes: becomes
  the level consolidation step (4.12).
- `arcagent/mind/memory.py` (built, off path, dropped for this agent). Its working memory
  class is the LLM path's version. Replaced by the stores below.

**Not yet built**

- Long-term store (file or table, per game and across games).
- Sequence record store.
- Canonical key function.
- Rule and hypothesis records in one schema.
- Flag store (4.11).

**Record types (logic)**

- **Rule record.** Fields:
  - `id`: stable.
  - `kind`: object, action, event, or experience.
  - `statement`: the regularity in words, for the log.
  - `conditions`: what must hold for the rule to apply (for example, health above zero, or the object touching a wall).
  - `effect`: what happens, in the effect vocabulary of `facts.py`.
  - `status`: hypothesis or fact (section 4.2 status rules).
  - `confirmations`, `contradictions`: counts of observed cases.
  - `probability`: from 4.4.
  - `scope`: working (this game), or long (across games).
  - `key`: the canonical key (below).
  - `source`: observed, loaded from long-term, or agent-made label.
  - `created_level`, `created_game`, `last_revised`.
  - `harm`: none, or the kind of harm the rule's effect causes (4.10).
- **Canonical key.** A rule's key names the kind of the subject and the role of the action, never the colour index. Example of the form, not a value: (kind of thing, action role, event type). Two games with the same mechanic get the same key even if their colours differ. (Part 1, D1; kept.)
- **Sequence record.** Fields: `id`, `goal` (the hypothesis it serves), `moves` (ordered), `segment_of` (the level), `runs`, `successes`, `status` (live, or completed), `deterministic` (true when the sequence is a named skill, 4.9).
- **Hypothesis record.** Fields: `id`, `goal` (kind and targets), `path_probability`, `expected_value`, `status` (open, tried, failed, achieved), `fork_id` (if on a fork).
- **Flag record.** Fields: `thing_id`, `label` (check, or an agent label), `source`, `value`, `cost`, `outcome`. (4.11.)

**Stores and what they hold**

| Store | Holds | Lives | Cleared when |
|---|---|---|---|
| Working memory | Rule records for this game, hypothesis records, sequence records, flags, fork tree, adverse rules seen | One game | The game ends (won, stopped, or out of budget). Not on death. Not on level reset. |
| Level state | Scene history, tracker, noise mask, current level's plan | One level instance | A new level instance starts (death, reset) or the level is complete. The level model is rebuilt from working memory. |
| Long-term store | Rule records with scope long, sequence records with scope long, the index | Across games and runs | Never cleared. Only records are retired (status retired). |
| Log | Move records, level and game summaries | Per run, on disk | Never. |

**Status rules (logic)**

1. IF a rule is created from one observation THEN status is hypothesis.
2. IF a hypothesis meets the promotion test (4.4) THEN status is fact.
3. IF a fact is contradicted AND a new condition explains the contradiction THEN the rule is refined: the condition is added, the counts are kept, status becomes hypothesis again until it meets the promotion test with the new condition.
4. IF a fact is contradicted AND no condition explains it THEN the rule's probability drops (4.4), and status goes to hypothesis.
5. IF the probability is in the low band (4.4) THEN status is retired. The record is kept.

Kameron's example, 2026-10-30, is the test for rules 1 to 4: "if the player touches an
enemy and dies instantly, that's a hypothesis; then the player touches an enemy and dies
after 2 seconds, that's still a hypothesis but the rule is updated; then the player dies
after touching the enemy for 5 seconds, the rule is updated to enemies kill if touched but
only the player dies after hit points drop to 0." In the rule record, that is one rule
refined twice: first with a time condition, then with a health condition. It is never
deleted.

**Clear rules (logic)**

1. IF a game ends THEN working memory is cleared, after the game consolidation (4.12) has written what is promoted.
2. IF a level instance ends by death or reset THEN level state is cleared and working memory is kept.
3. IF a level is completed THEN level state is cleared, working memory is kept, and the level consolidation (4.12) runs.
4. Long-term records are never cleared by a game ending.

Logs: every create, refine, promote, contradict, retire, and clear writes a log event with the record id.

Tests:
- The Kameron example above, as a fixture: three observations, two refinements, one rule id throughout.
- A game death keeps working memory: a rule learned before the death is still present after the death.
- A game end clears working memory and keeps long-term records.

### 4.3 Retrieval and index

**Built modules**

- `arcagent/mind/facts.py` (built, changes): exact-key lookup, a table of contents by behaviour tag, and a sibling index by shape. The lookup order is fixed (first match wins). Changes: ranked by probability (below).

**Not yet built**

- Index over canonical features (all rule and hypothesis records).
- Ranking by probability.
- Real-time cut.

**M4.3 Retrieval**

- Purpose: find the rules most likely to be right for a situation, quickly, without reading every record.
- Inputs: a query: the canonical features of the situation (kinds present, action role, event types of interest, the goal kind).
- Outputs: a ranked list of records, each with its probability of being correct.
- Logic:
  1. For each feature in the query, read its posting list from the index. The index maps each feature to the record ids that carry it.
  2. The candidate set is the union of posting lists, and each candidate's score counts how many query features it matches.
  3. Each candidate's probability is its probability from 4.4. Kameron, 2026-10-30 (Q3): the learned scorer is a network, because it "will get better" as levels and games happen. So from the point the index holds 200 records (PLACEHOLDER), a small network reads the query's features and the candidate's features and outputs a second probability. The candidate's score is the mean of the two probabilities. Before 200 records, the score is the rule's probability alone.
  4. Compute the cut for this query (rule 5).
  5. Keep candidates whose probability is at or above the cut. Sort best first. Ties sort by the four-decimal probability, then by id.
  6. IF no candidate is kept THEN return an empty list. The caller uses exploration or asks the planner a question (4.9). Retrieval never scans the whole store without a log event. Each scan is logged.
  7. Safety bound: at most 50 candidates are returned, to bound time. This is a time bound, not a choice of how many to use.
  8. The cut is set in real time: cut = the larger of the medium-band floor (4.4) and half of the best candidate's probability. PLACEHOLDER: the half-of-best factor is a proposal for review (Kameron, 2026-10-30: "it must be established in real-time").
- Stores: reads working and long-term stores.
- Logs: query features, candidate count, cut, returned ids, probabilities.
- Tests:
  - Equivalence: the indexed result equals a full scan's result on recorded queries.
  - A query with no matching features returns empty and logs a scan event.
  - Tie order is stable.

### 4.4 Probability and validation

**Built modules**

- `arcagent/mind/facts.py` (built, changes): hits, misses, and status by threshold (solid after 5 agreeing observations, broken after 2 misses in a row).

**Not yet built**

- Probability from evidence, with bands.
- Probability for paths (4.7) and for adverse events (4.10).

**M4.4 Probability of a rule**

- Purpose: turn evidence into the likelihood that a rule holds.
- Inputs: confirmations and contradictions of the rule.
- Outputs: a probability between 0 and 1, reported to four decimal places.
- Logic:
  1. probability = (confirmations + 1) / (confirmations + contradictions + 2). This is the Beta mean with a flat prior, PLACEHOLDER prior.
  2. A new rule has probability 0.5.
  3. Bands: certain band is at or above 0.8. Medium band is from 0.3 up to 0.8. Low band is below 0.3. (Kameron, 2026-10-30: "tighten them to 0.8 and 0.3".)
  4. Probability is reported with four decimal places, so ties are rare (Kameron, 2026-10-31: "there won't be any ties if the probability uses 4 trailing decimals").

**M4.4b Promotion test (hypothesis to fact)**

- Purpose: decide when a hypothesis is a fact.
- Logic:
  1. IF confirmations since the last refinement are at least 5 (PLACEHOLDER) AND contradictions since the last refinement are zero AND the probability is in the certain band THEN promote.
  2. The promotion test counts only observations made under the rule's conditions. A refinement resets the count for the new conditions and keeps the old counts in the log.
  3. Kameron's definition, 2026-10-30: "validated just means that the rule is a fact rather than a hypothesis." So the test is about repeated, unbroken observation, not about a count alone.

**M4.4c Path probability**

- Used in 4.7. Kameron, 2026-10-30, answered "no" to the question whether the independence approximation (multiplying move probabilities as if they were independent) is acceptable. So the path probability is not that product.
- Logic (chain rule over the simulated states):
  1. Simulate the path one move at a time. At step i, the simulator is in a known state S(i-1).
  2. The probability of step i is the probability of the rule that predicts it, conditioned on S(i-1) by that rule's conditions (4.2). Rule probabilities are conditional on their conditions, so each factor already depends on the state it is evaluated in.
  3. Path probability = the product of these conditional factors, taken in path order.
  4. Two factors that come from the same rule in the same state are one fact, counted once.
  5. Where a step has no rule, its factor is the prior 0.5 and the path is marked uncertain in the log.
- Note: this is the chain rule, which is exact for the model the agent holds. It does not claim the moves are independent in the world. The calibration check (4.13) measures whether the model's path probabilities match the observed rates.

Tests:
- Probability moves toward the observed rate.
- A new rule is at 0.5.
- A rule at 0.79 is medium, at 0.81 is certain, at 0.29 is low.
- Two hypotheses with the same probability to four places cannot tie, by construction, unless their paths are equal.

### 4.5 Self, focus, metacognition, and controller

**Built modules**

- `arcagent/mind/modes.py` (built, changes). Active controller.
- `arcagent/mind/progress.py` (built, keep). Milestone, progress, regress, motion, settled. These are signals for the controller.
- `arcagent/mind/indicator.py` (built, keep). Progress indicators. A signal for the controller.
- `arcagent/mind/budget.py` (built, keep). Moves left. A signal for the controller.
- `arcagent/agent/hybrid.py` (built, changes). The current switching rule (mind moves, stall count). Becomes the controller's first handoff rule. Its uncommitted stall rule must be benched first (as-built spec, section 2).

**Not yet built**

- Self record.
- Focus record.
- Controller as one module with a single owner of handoffs.
- Metacognition labels in the planner briefing.

**M4.5a Self record**

- Purpose: say who "I" is at each move, so the agent's own actions have one referent.
- Inputs: available actions, the controlled things (from 4.5 and 1.6), the last click.
- Outputs: `i` (always the agent), `my_character` (the controlled thing, or none), `click` (present or not).
- Logic:
  1. `i` is always the agent. (Kameron, 2026-10-30, answering Q2: "I would represent the agent manipulating those actions and the character is an extension of oneself." This changes the earlier source rule that said "I" is the mouse click when both exist. The change is recorded here for Kameron to confirm.)
  2. `click` is present IF action 6 is in the available actions. (Kameron, 2026-10-30: "Click present means exactly that, the action 6 is available.")
  3. `my_character` is the controlled thing IF one is identified, otherwise none.
  4. The self record is written to the planner briefing and to the decision code. (Kameron, 2026-10-30: the labels "should reach both since it might be talking to itself.")

**M4.5b Focus record**

- Purpose: say what the agent is doing right now, in one record.
- Outputs: level, goal and its status, target thing, mode (the branch that chose the last move), budget left, and the signal values.
- Logic: rewritten after each move. It is the first thing the planner briefing reads.

**M4.5c Controller**

- Purpose: decide when to change method, and say why.
- Inputs: the focus record, the signals, the gate state (4.10).
- Outputs: the method to use next (keep, switch goal, switch mode, reset fact breaks, re-probe a goal), and the reason.
- Logic:
  1. The controller reads its handoff rules in the order given in section 7 (order to be set after review of the workflow, Kameron, 2026-10-30: "Once you create it, we can answer that question").
  2. The first rule that fires wins. The rest are not checked for that move.
  3. Every handoff is logged with the rule name and the signal values.
  4. The controller does not run per level as a separate pass (Kameron, 2026-10-30: "the controller is meant for its purpose and not more").
- Handoff rules to be ordered (names only; the logic is in the cards that own the signals):
  - H-a: a gate tripped (4.10).
  - H-b: a plan broke at a miss that matters (3.6).
  - H-c: progress has stopped for N moves on the active goal (4.13 signal).
  - H-d: budget is low for the active hypothesis (4.13 signal).
  - H-e: the fallback count in a row is at the stall limit (as-built spec, STALL_MOVES).

**M4.5d Metacognition labels**

- Purpose: make the agent's own state visible to itself and to the planner.
- Logic: the focus record, the self record, and the list of open hypotheses are written into the briefing (4.9) at each planner call. The record of what the agent thinks it is doing (its branch and why) is part of the move log (4.13).

Tests:
- Self record: four fixtures (click and character, character only, click only, neither) give the expected `i`, `click`, and `my_character`.
- Controller: a fixture where two handoff rules would both fire gives the first one's name in the log.

### 4.6 Visual-spatial tools

**Built modules**

- `arcagent/mind/kinematics.py` (built, changes). `reachable` (forward search over positions the controlled thing can reach) and `field` (distance from each position to a target). Changes: shared breadth-first search with a truncation flag (below). Its walk currently stops at 4096 positions with no flag.
- `arcagent/mind/route.py` (built, keep, reason given below). Ordering of visits over several targets, exact up to 6 targets, then 2-opt.
- `arcagent/mind/search.py` (built, changes). Best-first search. Its heuristic is not guaranteed to be admissible. Changes: bounded by path probability (4.7).
- `arcagent/agent/explore.py`, `arcagent/agent/guided.py` (built, off path for the new agent, but the breadth-first search is a reuse source).

**Not yet built**

- A shared breadth-first search used by all callers (currently four copies).
- A truncation flag on every search.
- Jump point search: conditional (below).
- Theta-star: out of scope (below).
- Value estimation by Q-learning: new, gated (below).

**M4.6a Breadth-first search (shared)**

- Purpose: shortest move count over a graph of positions.
- Inputs: start, a neighbour function (the moves the agent can make from a position), a target test, a visit limit.
- Outputs: the path, the depth, and a flag: found, not found, or stopped at the limit.
- Logic:
  1. Visit positions in order of depth. Stop when the target is found or the limit is reached.
  2. Return the flag. An empty result is not the same as a truncated result.
- Replaces: the four copies in `kinematics.py` (two loops), `explore.py`, `guided.py`.
- Tests: equivalence with each current copy on the same inputs; a limit below the reachable area returns the truncated flag.

**M4.6b Route over targets (kept)**

- Purpose: the order to visit several targets in the shortest total path.
- Why it is kept: Kameron's example, 2026-10-30 (Q4 on the earlier spec), "pacman collects fruit then completes the level": a level with several collectible things needs an order, and this is the tool for it. Kameron, 2026-10-31: "I have no idea what route.py does or if its worth keeping." This section is the answer: it orders visits to collectibles and to goal-relevant things. It is kept.
- Logic: as built (exact for up to 6 targets, then nearest neighbour with 2-opt).

**M4.6c Best-first search**

- Purpose: find a plan toward a goal by predicting moves.
- Logic: as built, with two changes:
  1. Each expanded branch carries its path probability (4.4c). Branches below the probability floor (0.3, the medium-band floor, as a PLACEHOLDER floor) are not expanded further.
  2. Kameron, 2026-10-30 (Q1 admissible heuristic): an admissible heuristic is used only if it makes the agent take fewer moves and still complete levels. This is a measured choice (4.13), not a rule. Until measured, the current heuristic stays.

**M4.6d Jump point search (conditional)**

- Purpose: faster search on open grids with unit steps.
- Logic: built only if all three hold on recorded runs: (a) a game's learned moves are all unit steps; (b) search time is a large share of move time; (c) plan length matches breadth-first search. Otherwise not built.
- Why conditional: Part 4 (visual-spatial) gave this reason. The learned moves in these games can be multi-cell steps or slides, which the pruning rules do not cover.

**M4.6e Theta-star (out of scope)**

- Reason: the moves are discrete. Any-angle paths would contain moves the game does not allow. Revisit only if a game has continuous movement. Kameron, 2026-10-30 (Q2): "I have no idea because I don't see them, only the agent does." So the agent must check for continuous movement from observation (4.8 rule 6). Until it does, theta-star stays out.

**M4.6f Value estimation (Q-learning)**

- Purpose: a learned value for states or moves, used in the expected-value choice (3.4).
- Kameron, 2026-10-30: "Q-learning is a highly specialized math engine used to automate and optimize sequential decisions under deep uncertainty. Should help significantly with decision making."
- Logic:
  1. The value is a function of canonical features (not scene keys), so it generalises across games.
  2. It is trained from the agent's own moves, rewards being level completion, progress, harm, and the score.
  3. It is an input to the expected-value choice, not a replacement for it.
  4. Gate: built only after the rule store and the index (4.2, 4.3) exist and produce recorded transitions. Reason: with a few hundred moves per level, a value learned from scratch needs the shared features to learn anything across levels.
- Risk: tabular Q-learning over scenes will not converge in the move counts we have. The first version uses features and a small function approximator, measured against the plain expected-value choice (4.13).

Tests:
- Shared breadth-first search equals each current copy on fixtures.
- Route on 3 targets: the order's total length equals the exact minimum.
- Search: a branch below the probability floor is not expanded.
