# Prediction, Level Modeling, and Simulation Design (Part 8 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Prediction" section, pasted 2026-10-30. The source governs where it
differs from the current code. Parts 1 to 7 cover the stores, retrieval, metacognition,
spatial tools, planning, perform, and grit that this part draws on. Part 6 defines the
sequence record, and Part 7 defines the adverse-event probabilities this part reuses.

## 1. Requirements (from the source)

**Prediction**

- **R1.** Prediction writes to nothing except two lists: the sequence list and the
  hypothesis list. It only reads everything else.

**Level modeling**

- **L1.** Hypothesis list: a guess list of ways to complete the level, ranked by the
  highest probability of completing the level task.

**Simulation**

- **S1.** A copy of the level is rebuilt as the current state.
- **S2.** The player plays the level in simulation, pulling from memory to test hypotheses.
- **S3.** The agent predicts only a few moves at a time, each with a probability estimate.
- **S4.** The simulation also predicts enemy movement and adverse events and experiences.
- **S5.** The simulation uses the visual-spatial tools (Part 4) to aid its predictions.

**Sequence list (second definition)**

- **Q1.** The sequence list holds the highest probable next move or moves that let the
  player reach the next task.

The source puts a probability on the hypothesis list and on each predicted move. It does
not say how those probabilities are computed. Section 5 proposes a way and marks it as a
decision.

## 2. What the current code does

| Requirement | Current code | Status | Gap |
|---|---|---|---|
| R1 read-only prediction | `predict` (`predictor.py`) builds a new scene from facts and reports unknowns. It writes nothing. Learning (`facts.learn`, `record`) and the goal book (`GoalBook.level_up`, `note_failed`) are separate steps in the move loop. | Built for `predict` | The source says prediction writes to the sequence and hypothesis lists. The code keeps those writes separate from `predict`, and nothing writes the hypothesis list at all (see L1). |
| L1 hypothesis list | `GoalBook.ranked` (`goals.py`) returns confirmed forms first, then open goals scored by `prior * salience / (1 + cost)`. | Partial | The score is a formula, not a probability of completing the level. Ranking is by goal, not by a path of moves. |
| Level model | Spread across `fluid.py` fields: the tracker, `FactMemory` (positions cleared, facts kept), `GoalBook`, the plan, and the level's scene history. | Partial | No single level-model record. The level model is the set of fields, and a new level clears some of them (`fluid.py` step 2). |
| S1 copy of the level | `Scene` objects (`objects.py`) are built per frame and per prediction. Search builds each child with `predict` (`search.py`). | Built | Scenes are copies, so search is a simulation in this sense. |
| S2 test hypotheses in memory | `search` tests goal guesses against predicted scenes, and `tester.py` runs fact tests. | Built | The search tests one goal at a time. It does not hold several hypotheses at once. |
| S3 few moves, with probability | `search` plans to the goal's depth (up to `SEARCH_MAX_NODES = 3000` predictions). Facts carry `hits`, `misses`, and `status` (`facts.py`), but no probability is computed from them. | Partial | No probability per predicted move. The search plans a full path, not a few moves (Part 5 covers the segment cut). |
| S4 enemy movement and adverse events | Side objects with no usable fact are predicted still (`assume_still`, `predictor.py`). Adverse events are only game over (Part 6 covers the rest). | Partial | Enemy motion is predicted only if a fact has learned it. Otherwise the enemy is assumed still, and nothing records that the assumption was wrong. |
| S5 visual-spatial tools | `kinematics.py` (`reachable`, `field`) feeds goal distances. `route.py` orders targets. | Built | See Part 4 for the tool gaps. |
| Q1 sequence list | The plan's first step (`fluid.py`) is the next move. The Part 6 sequence record stores completed sequences. | Partial | No ranked list of the next probable moves with probabilities. |

**Where the probability is missing.** Facts already record evidence (`hits`, `misses`), and
the fact status (`candidate`, `solid`, `thin`, `broken`) is a threshold on that evidence.
Nothing turns the evidence into a probability, and nothing multiplies probabilities along a
path. Without that, the source's ranking by probability cannot be computed.

## 3. Proposed structure

**Fact probability.** Every fact has a probability of being right, from its evidence.

- `p_fact = (hits + a) / (hits + misses + a + b)`, a Beta posterior mean with prior
  `a, b` (D2).
- The same form is used for adverse rules (Part 7) and for sequence success (Part 6).

**Move probability.** A predicted move has the probability of the fact that produced it.
A move predicted from a sibling fact (`lent`) uses the sibling's probability (`lent` is
already recorded in `Prediction`).

**Path probability.** A hypothesis (a path to a goal) has the product of its move
probabilities (D3).

**Hypothesis list.** The hypothesis list is built as follows.

1. Run the search over the goal guesses, as now.
2. For each goal the search reaches, compute the path probability of its plan.
3. Rank the goals by path probability, ties by the current score.
4. Keep the top `k` (D5).

The list is written after each search and read by the planner (Part 5). It is the
hypothesis list the source asks for.

**Sequence list (live).** The sequence list is the first `h` moves of the top hypothesis,
with each move's probability. It is written with the hypothesis list. Part 6's sequence
record is the completed version: a sequence list that reached its task is saved.

**Enemy motion.** An uncontrolled mover with no usable fact is predicted still, as now, but
its prediction is marked `assumed`. When the observed frame shows it moved, the miss is
logged against the assumption. Enough of those, and the mover gets a fact by observation.

## 4. Behavior

- **Prediction.** `predict` stays read-only. The hypothesis and sequence lists are written
  by a separate step after the search.
- **Probability bounds.** A path whose probability falls below `P_MIN` is not expanded
  further (D3). The search logs the bound it hit.
- **Horizon.** The sequence list holds at most `H` moves (D4). The planner (Part 5) cuts
  its segments at the same `H`.
- **Adverse events.** A predicted adverse event with probability from Part 7 is attached to
  its move. A move whose adverse probability is in the certain band is never followed
  (Part 7).
- **Assumed still.** Every assumed-still prediction is counted. A miss on an assumed-still
  mover is logged as `assumption_missed`.

## 5. Decisions I made

These are my calls. The values marked as placeholders have not been measured.

- **D1. Learning writes facts, not prediction.** The source says prediction writes only the
  two lists. This build keeps `predict` read-only and has the learning step write the facts,
  because learning needs the observed frame and prediction does not have it. Cost if wrong:
  the source's rule is broken on a technicality. Q1 asks whether the rule is strict.
- **D2. Beta prior of 1 and 1.** A new fact starts at 0.5, the same uninformative prior as
  Part 7. Reason: one set of numbers for every probability in the store.
- **D3. Independence and the bound.** Path probability is the product of move
  probabilities, which assumes the moves are independent. They are not, but the product is
  a stated, testable approximation. `P_MIN = 0.05` (placeholder). Cost if wrong: paths with
  correlated risk are over- or under-counted, which the calibration test in section 6 will
  show.
- **D4. Horizon `H = 5` (placeholder).** Reason: it matches the segment bound I'd set in
  Part 5 (D3), and five moves is short enough to check against the observed frame.
- **D5. Hypothesis list `k = 5` (placeholder).** Reason: the planner reads the top few, not
  the full list. Same reasoning as Part 2's D3.
- **D6. Enemy motion is assumed still until observed.** Reason: no fact means no evidence,
  and the current code already does this. The change is that the assumption is logged.

## 6. Test plan

**Read-only prediction.** A test that `predict` changes no field of `FactMemory`, the goal
book, or the plan, on a set of random scenes and actions.

**Probability.** A test that `p_fact` moves toward the observed rate, and that a fact with
no evidence stays at 0.5.

**Calibration.** On recorded runs, compare predicted move probabilities with the observed
rate, in bins. A well-calibrated store has the observed rate near the predicted one in each
bin. This is the test of D3's independence approximation.

**Hypothesis ranking.** A fixture with two goals where the cheaper one has the lower path
probability. The ranking must put the higher path probability first.

**Sequence list.** A test that the sequence list has at most `H` moves and that each move's
probability is the product of its facts'.

**Enemy assumption.** A fixture where an uncontrolled mover moves with no fact. The test
asserts the prediction is marked `assumed` and the miss is logged.

**Held-out games.** Same rule as Parts 1 to 7.

## 7. Open questions for Kameron

- **Q1.** Is the rule that prediction writes only the two lists strict? If it is, learning
  needs its own module, and D1 is the answer. If not, the learning step can stay in the
  prediction module.
- **Q2.** Is the independence approximation (D3) acceptable for the first build, with the
  calibration test as the check? Or should path probability use a joint model?
- **Q3.** Is `H = 5` the right horizon for the sequence list, or should it follow the games'
  typical move count per goal?
- **Q4.** Which of the 25 games have enemies that move on their own? I haven't checked that
  from the survey, which recorded action lists and colors only. The answer decides how much
  the enemy assumption matters.
- **Q5.** Should the hypothesis list rank by path probability alone, dropping the current
  prior-times-salience formula, or keep salience as a tie-breaker?
