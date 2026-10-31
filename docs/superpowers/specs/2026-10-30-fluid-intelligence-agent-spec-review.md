# Fluid-Intelligence Agent Spec: Review of the Current Build

Status: draft, 2026-10-30. Part of the master spec `2026-10-30-fluid-intelligence-agent-spec.md`.
This section reviews the code and the nine parts. It does not change any requirement that
Kameron wrote. Where a part's analysis was wrong, the correction is noted in section 3.

## 1. Purpose (Kameron, 2026-10-30)

Build an agent that can demonstrate fluid intelligence, with the research line kept true to
HCQM so the work can continue. The current scored entry is the hybrid of the fluid agent
and the graph explorer (`arcagent/agent/hybrid.py`). Its best public score is about 0.39.
The competition's final submission is 2026-11-02 23:59 UTC. The agent should be finished by
2026-11-01 (Kameron's date).

## 2. What is built

### 2.1 The scored path

The hybrid plays the fluid agent first on each level. It hands the level to the explorer
after `mind_moves` moves or after `STALL_MOVES = 25` fallback moves in a row. The fluid agent
writes no code and makes no LLM call. The explorer covers the screens it can reach.

The fluid agent is built around the paper's four modules, each with a null for ablation
(`arcagent/mind/fluid_modules.py`):

| Paper module | Real | Null |
|---|---|---|
| Voi (choose the next test) | `tester.choose_test` | first untried action with an unknown pair |
| Inhibition (refuse an action) | `tester.veto` | never refuses |
| Metacognition (fact break threshold) | `MissMetacognition` (`BROKEN_MISSES`) | `NEVER_BROKEN` |
| Transfer (carry facts across levels) | `CarryTransfer` | `FactMemory.forget` |

### 2.2 Modules in the fluid agent

- Perception and identity: `screen.py`, `objects.py` (things, scenes, diffs), `kinds.py`
  (identity across looks), `noise.py` (masks for counters and noise), `modes.py` (selection
  and the active controller), `regions.py`.
- Facts: `facts.py` (`FactMemory`, effects, touch and near contexts, toc, sibling borrowing),
  `predictor.py` (`predict`, read-only), `kinematics.py` (gaits, reachable, fields).
- Goals and search: `goals.py` (`GoalBook`, reach, clear, count, cover, match, sequence,
  match_thing), `search.py` (best-first over predicted scenes), `route.py` (visiting order).
- Learning and checks: `tester.py`, `progress.py`, `indicator.py`, `budget.py`,
  `fluid_sleep.py`, `learn_win` and `generalize_forms` in `goals.py`.
- Outside the fluid path: `agent.py`, `memory.py`, `sleep.py`, `wisdom.py`,
  `sandbox.py`, `synthesizer.py` (program path); `turn.py`, `llm_drive.py`, `toolbox.py`,
  `tool_fix.py`, `vchat.py` (LLM path).
- Explorers: `agent/explore.py` (state-graph explorer), `agent/guided.py`.

### 2.3 Benchmarks on record

From `docs/experiment-log.md` (Kaggle CPU notebooks, 4 seeds): hybrid with 150 mind moves
averaged 26.75 levels with score 0.473. Adding learn-from-win gave 27.25 and 0.474. The
explorer alone at 100k moves reached 37 levels of about 180. The DRIVE path (LLM) was 11
levels offline at 0.577 and 0.39 public. The numbers in this section are taken from the log
and are not re-run for this review.

## 3. Coverage of the nine parts

Counts are over the requirements each part lists.

| Part | Built | Partial | Missing | Notes |
|---|---|---|---|---|
| 1 Memory | 2 | 4 | 2 | Game boundary and level persistence built. Long-term store and escalation missing. |
| 2 Fluency retrieval | 2 | 2 | 2 | Indexed lookups exist. Ranking by probability and matching by meaning missing. |
| 3 Metacognition and self | 1 | 1 | 2 | No self record. Metacognition signals computed, not combined. |
| 4 Visual-spatial | 2 | 0 | 2 | BFS and best-first search exist. JPS, Theta*, Q-learning are gated or out of scope. |
| 5 Planning | 2 | 2 | 2 | Simulation and the decision ladder exist. Plan tree and segments missing. |
| 6 Perform | 1 | 3 | 1 | Plan as action list built. Inhibition is rules, not a network. Scored path writes no code. |
| 7 Grit and perseverance | 0 | 2 | 2 | Time and attempt gates partial. Token totals and perseverance missing. |
| 8 Prediction and simulation | 2 | 3 | 2 | `predict` is read-only. No probabilities. Hypothesis list is a formula. |
| 9 Curiosity | 1 | 3 | 1 | Four stages exist as separate mechanisms. No flag record. |

The counts are my reading of each part's section 2. A reviewer should check the table
against those sections, not against this summary.

### 3.1 Corrections to the parts

- **Part 3 (fixed in the file).** The fact break threshold comes from `FluidMetacognition`
  (`fluid_modules.py`), not from `ConfidenceMetacognition`.
- **Part 4.** The BFS copies are four, not three: kinematics `reachable`, kinematics
  `_walk_back`, the explorer's `path_to_frontier`, and the guided explorer's path search.
- **Part 2.** The claim that sig keys carry the color is confirmed by `_SIG_RE`
  (`turn.py`) and `_sig_color` (`goals.py`). It was not an assumption.

## 4. Conflicts between parts

Each conflict needs one owner before build. These are my proposals, not rulings.

- **C1. Sequence record (Parts 6 and 8).** Part 6 stores completed sequences per goal. Part
  8 keeps a live sequence list, the first moves of the top hypothesis. Proposal: one record
  type with `status` set to `live` or `completed`.
- **C2. Two confidence systems (Parts 1, 7, 8).** Rule status (`candidate`, `validated`,
  `broken`) and the Beta probability both describe reliability. Proposal: status is derived
  from the probability and the evidence thresholds, so one number decides.
- **C3. Index fields (Parts 1 and 2).** The index in Part 2 must cover every field of the rule
  record in Part 1, including `scope` and the canonical key. Build Part 1's schema first.
- **C4. Strategy switches (Parts 3, 7).** The Part 3 controller and the Part 7 gates both end
  a task and hand it to another method. Proposal: the controller owns every handoff, and the
  gates only report.
- **C5. Horizon (Parts 5 and 8).** Part 5's segment bound and Part 8's horizon `H = 5` are
  two names for one value. Proposal: one constant, one owner.
- **C6. Prediction writes (Part 8 against the source).** The source says prediction writes
  only two lists. Part 8 keeps learning as a separate writer (D1). Part 9's flags are also
  written by the learning step. This stands until Kameron rules on Part 8 Q1.
- **C7. Inhibition and adverse estimates (Parts 6, 7).** Part 7 reads adverse probabilities
  from Part 6's rules. Part 6's rules carry evidence counts, not a probability. Proposal:
  the Part 7 Beta is a field of the Part 6 rule record.
- **C8. Self and controller (Parts 3 and existing code).** The "I" and "my character" rule in
  Part 3 overlaps `modes.py` (the active controller). Proposal: `modes.py` supplies the
  character, and the self record is built on it.
- **C9. Shared BFS and the silent stop (Part 4 and `kinematics.py`).** The shared BFS changes
  the function that stops silently at `KIN_LIMIT`. Build the truncation flag before the
  shared function, so the change is one step.
- **C10. Ordering of checks (Parts 5, 7, 9).** Parts 5, 7, and 9 each say a lower-priority
  move waits behind the plan. The ladder in `fluid.py` already does that. Proposal: one
  ladder, with each part's addition written as a step in it.

## 5. Proposed build order

The order follows dependencies. Measure before change: each step starts with the
measurement that decides whether the step is needed.

1. **Measure.** Profile the move loop and report time in `kinematics`, `search`, `explore`,
   and `guided` (Part 4). Log loop stage and gate records (Parts 7, 9). Count predicted and
   observed move outcomes in bins (Part 8).
2. **Schema.** Rule record, goal record, sequence record, flag record, with the probability
   field and the status rule (Parts 1, 6, 7, 8, 9; C1, C2, C7).
3. **Index and retrieval.** Indexed retrieval with the equivalence test against a brute-force
   scan (Part 2). Shared BFS and truncation flag (Part 4; C9).
4. **Plan tree and segments.** Branch record and replan rule, with the horizon constant
   (Part 5; C5).
5. **Adverse rules and perseverance.** Rules in the store, veto with reasons, override with
   a logged reason (Parts 6, 7).
6. **Self record and controller.** Self record on `modes.py`, controller as the one owner of
   handoffs (Parts 3, 7; C4, C8).
7. **Curiosity flags.** Flag records and check ranking behind the plan (Part 9).
8. **Long-term store and the H1 test.** Store across games, then the held-out comparison of
   moves to first win with and without it (Part 1).
9. **Learned components, only if the gates pass.** Inhibition network (Part 6), JPS (Part 4),
   learned retrieval scorer (Part 2), Q-learning (Part 4).

### 5.1 Deadline check

Steps 1 to 6 are a multi-week program. The agent has to be finished by 2026-11-01, which
is two days from this review. My estimate is that steps 1 and 2 are the most that can be
built and verified before then. The rest needs either a later submission or a smaller first
build. I have not timed the steps, so the estimate is a judgment, not a measurement. This is
Kameron's call, and it is the first open question in section 7.

## 6. What was verified and what was not

**Checked against code this session:**

- The four paper modules and their nulls (`fluid_modules.py`).
- `predict` is read-only (`predictor.py`).
- `GoalBook.ranked` formula (`goals.py`).
- Sibling borrowing and the toc (`facts.py`).
- `gaits` sorts the whole fact table on each call (`kinematics.py`).
- `reachable` stops silently at `KIN_LIMIT` (`kinematics.py`).
- Plan stop rule and forced move (`fluid.py`).
- Game over is in `_PLAN_VETOES` (`fluid.py`).
- Fresh modules per game (`config.py`).

**Not checked, and needed before any spec item is built:**

- Whether the Kaggle runtime serializes frames as JSON. The offline path does not.
- Continuous movement in any game. The survey recorded colors and actions only.
- Enemy movement in any game (Part 8 Q4).
- The competition's runtime limit (Part 7 Q1).
- Whether a cursor is drawn on screen (Part 3 Q1).
- Cumulative token counts per task (Part 7). None were found in code, and none were measured.
- Calibration of any probability (Parts 7 and 8). None exists yet.
- The per-level grid size and color set after level 1 (survey covered level 1 only).

## 7. Open questions, all unanswered as of 2026-10-30

- **Schedule.** Which of build-order steps 1 to 9 must finish by 2026-11-01, and which can
  wait for a later submission? (Section 5.1.)
- **Part 1.** Q1 RESET after game over; Q2 long-term memory during a scored run; Q3
  "validated" threshold for escalation; Q4 held-out game set.
- **Part 2.** Q1 "most probable" with age or recency; Q2 k = 5 or threshold; Q3 learned
  scorer as network or tree; Q4 HCQM construct for retrieval.
- **Part 3.** Q1 "click present" as action 6 or a visible cursor; Q2 "I" on keyboard moves
  when a click exists; Q3 self labels in the prompt, decisions, or both; Q4 signal order for
  switches; Q5 controller per level.
- **Part 4.** Q1 admissible heuristic; Q2 continuous movement in any game; Q3 Q-learning as
  policy or value estimate; Q4 `route.py` as a visual-spatial tool.
- **Part 5.** Q1 expected-value decision rule; Q2 definition of a fork; Q3 keep or replay
  moves to a fork; Q4 explorer backtracking as the replan step.
- **Part 6.** Q1 one list per goal or per level; Q2 build the network now or later; Q3 what
  counts as harm; Q4 code path beside or replacing action performance.
- **Part 7.** Q1 time gate scope; Q2 which tokens count; Q3 probability bands; Q4 "no other
  way" reason alone or with the score check; Q5 override trials and the move budget.
- **Part 8.** Q1 strict read-only rule; Q2 independence approximation; Q3 horizon; Q4 enemies
  that move on their own; Q5 hypothesis ranking by path probability alone.
- **Part 9.** Q1 designer label as well as derived flags; Q2 social check; Q3 checks before the
  plan; Q4 which owner runs the planning loop.
