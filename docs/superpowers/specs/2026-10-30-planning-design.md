# Planning Design (Part 5 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Planning" section, pasted 2026-10-30. The source governs where it
differs from the current code. Parts 1 to 4 define the stores, retrieval, self and
metacognition, and spatial tools that a plan draws on. Part 4 (`2026-10-30-visual-spatial-design.md`)
covers the BFS and best-first search this part reuses.

## 1. Requirements (from the source)

- **P1. Decision science.** Planning is a way of deciding what to do in order to carry out
  tasks. Each move is a decision, and the decision rule should be stated, not implied.
- **P2. Decompose from known things.** Planning breaks a goal into tasks, using the
  attributes, events, rules, and experiences the agent already knows.
- **P3. Replan from the last fork.** When something unexpected happens mid-task, the agent
  replans from its last successful branch, not from its last step. Its example is a maze.
  If the newest fork is exhausted without success, the agent returns to the fork before
  it. This repeats until the level is complete.
- **P4. Simulate before moving.** Planning runs in simulation before any real move.
- **P5. Plan parts, not the whole.** A plan covers part of the end-to-end process, not all
  of it. The source says this gives the agent adaptability.

The source does not define a fork, what "exhausted" means, or the size of a plan part.
Section 5 proposes those and marks them as decisions.

## 2. What the current code does

The planner is `arcagent/mind/fluid.py` with the search in `search.py`. Its decision order
is written in the module docstring, steps 1 to 7 and the sub-steps under them.

| Requirement | Current code | Status | Gap |
|---|---|---|---|
| P1 decision rule | A fixed priority ladder in `fluid.py` `_decide`: game over, level change, forced move, bump, plan or follow, budget-tight step, indicator step, tester test, best partial plan, novelty. | Partial | The order is explicit and logged (each move records its branch and why), but no move is scored against the others. The ladder is not a utility or expected-value rule. |
| P2 decomposition | Goals come from heuristics in `GoalBook`, the executive turn, and the indicator. `search.py` composes steps toward a goal by predicting moves. | Partial | Goals are not split into subgoals. A goal is reached by one search, not by a tree of tasks built from rules. |
| P3 replan from fork | After a plan breaks (`plan_breaks`), `_plan` is cleared, and the next move runs a fresh search from the current scene. | Missing | No fork stack. The planner does not return to an earlier branch point after the newest one fails. The fresh search can rediscover the old branch, but nothing guarantees it. |
| P3 backtrack in a maze | `explore.py` `Graph.path_to_frontier` walks back to the nearest state with an untried action. That is backtracking, in the explorer. | Partial, explorer only | The fluid planner has no equivalent. The hybrid hands the level to the explorer after a stall, which is the only backtracking path in play. |
| P4 simulate first | `search` expands scenes with `predict` before any real move. `SEARCH_MAX_NODES = 3000` bounds the predictions. | Built | The simulation covers the goal search. The tester and novelty moves are not simulated. |
| P5 plan parts | `search` returns a full plan to a goal. When no full plan is found, the best partial plan's first step is taken and the rest is held as a plan toward its goal (`toward <goal id>`). | Partial | A plan is a path to the whole goal, not a bounded segment. Partial plans are held in full. |

**Stop rule.** A plan stops only on a miss that matters (`plan_breaks`, the rule in
`fluid.py` step 4). A miss about a mover or a goal sig stops it. Other misses are logged
and learned. After `STALL_AFTER = 2` failed attempts at one goal in one level, the goal is
set aside. This is the closest the code has to a replanning trigger, and it is
goal-level, not fork-level.

**Why the gap matters.** When a plan breaks in a maze, the planner searches again from
where it stands. If the first branch was the wrong one, the fresh search may choose it
again, because nothing records that it failed. Recording failed branches is what P3 asks
for.

## 3. Proposed structure

**Plan tree.** A plan is a tree of branch points, not a list. Each node records:

- `scene_key`: the scene the branch started from.
- `steps`: the moves tried from that node, in order.
- `status`: `open`, `exhausted`, or `succeeded`.
- `parent`: the node it branched from.

The tree is per level, reset on a level change. A game over keeps the tree and marks the
current branch failed, because the level restarts from the same frame.

**Replan rule.** When a plan breaks:

1. Mark the current branch `exhausted` if its goal did not hold.
2. Return to the newest ancestor branch with an untried step, and resume there.
3. If no ancestor has an untried step, search from the root.

This is the P3 maze rule, applied to the plan tree.

**Plan parts.** A plan is cut to a bounded segment of steps (the size is decision D3). The
segment ends at a checkpoint, a scene where a fact the segment depends on is first tested
or a goal sub-condition first holds. The next segment is planned from the checkpoint.

**Simulation.** Every segment is simulated with `predict` before any real move. The
simulation reuses `search`, with its node and time bounds.

## 4. Behavior

- **Branch record.** Each branch point is logged (`branch_open`) with its scene key and
  the moves it will try.
- **Backtrack.** A break logs `branch_exhausted` or `replan_from`, with the node resumed.
- **Termination.** Replanning stops at level completion or when the root has no untried
  step. The second case falls back to the explorer, as today.
- **Segment size.** A segment ends at its size bound or its checkpoint, whichever comes
  first. A segment that ends at the bound is logged as `segment_cut`.
- **Interaction with the stall rule.** `STALL_AFTER` counts failed segments at one goal,
  not failed plans. A segment that backtracks is not a failed attempt until its tree is
  exhausted.

## 5. Decisions I made

These are my calls, changeable without changing section 1.

- **D1. The decision ladder stays.** The fixed order from `_decide` is kept, with the
  logging it already has. Reason: it is explicit and tested, and changing it to an
  expected-value rule is a larger change than the requirements need. Cost if wrong: a
  decision rule that is wrong in an order we don't see. Q1 asks for a yes or no on
  expected value.
- **D2. Plan tree per level.** Branch points live in one tree per level, reset on a level
  change. Reason: P3 is about the same level, and a level change makes old branches
  invalid. Cost if wrong: failed branches in one level do not inform the next.
- **D3. Segment bound.** A segment is at most the depth of a goal search found in a
  typical run, which I have not measured. I'll set the bound from the recorded plan
  lengths before choosing a number. Reason: a bound that is too short makes the planner
  replan constantly, and one that is too long is the whole-process plan P5 rejects.
- **D4. Game over keeps the tree, marks the branch failed.** Reason: the level restarts
  from the same frame, so the failed branch is still the wrong one to try first.
- **D5. Simulation covers the goal search only.** The tester and novelty moves stay
  unsimulated. Reason: they exist to learn facts, and simulating them would use the facts
  they are meant to test.

## 6. Test plan

**Backtrack.** A maze fixture where the first branch dead-ends. The test asserts the
planner returns to the fork and tries the second branch before searching from the root.

**Exhaustion.** A fixture where every branch dead-ends. The test asserts the planner falls
back to the explorer after the root is exhausted, and logs each `branch_exhausted`.

**Segment bound.** A test that checks no segment exceeds the bound and that each cut is
logged. The bound itself is set in D3 from recorded data.

**Game over.** A test that checks the tree survives a game over, the current branch is
marked failed, and the next move resumes an older branch.

**Simulation.** A test that the goal search makes no real move before its plan is
complete, and that tester and novelty moves are not simulated.

**Measured effect.** On recorded runs, count moves spent on branches that had already
failed in the same level, before and after the tree. That is the measure of P3.

**Held-out games.** Same rule as Parts 1 to 4.

## 7. Open questions for Kameron

- **Q1.** Should the decision ladder (D1) be replaced by an expected-value rule, where
  each move is scored by the chance it advances a goal, or is the fixed order enough?
- **Q2.** Is a fork in your sense any point where the planner had more than one untried
  step, or only a point where the plan could go two different ways toward the same goal?
- **Q3.** When the planner returns to an older fork, should it keep the moves that worked
  on the way to the fork, or replay them from the start of the level?
- **Q4.** Should the explorer's backtracking (`path_to_frontier`) become the fluid
  planner's replan step, or should the two stay separate with the hybrid as the switch?
