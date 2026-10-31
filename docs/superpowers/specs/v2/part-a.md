# Agent Specification v2: Process, Modules, and Build Path

Status: draft for review, 2026-10-31. Owner: Kameron. Prepared by Claude.
Supersedes the refactor parts (1 to 9) and the as-built section as the working spec. Those
documents stay as the record of the comparison. Every decision Kameron made in the review
comments of 2026-10-30 and 2026-10-31 is in section 6. Nothing here is built.

## 0. How to read this document

**Status labels on every module.**

- **Built.** The code exists and is on the current path. The file is named.
- **Built, changes.** The code exists. This spec changes its contract or its use.
- **Built, off path.** The code exists and is not on the current path. Section 5 says whether it is kept.
- **New.** No code exists. The spec describes it fully enough to build.
- **Dropped.** Kameron decided it is not part of the new agent (section 6).

**Workflow graphs.** Each process has a graph. Graphs come at three depths:

- **Depth 0.** The whole game, from start to end. Section 2.
- **Depth 1.** One stage of the depth 0 graph, for example the per-move loop. Section 3.1 to 3.4.
- **Depth 2.** A sub-process inside a stage, for example how a hypothesis is simulated. Section 3.5 onward.

In the graphs, rectangles are steps, diamonds are decisions, and rounded boxes are stores
(memory and logs). Arrows are the order of control. Labels on arrows give the condition.

**Logic, not code.** Each module is written as numbered rules of the form IF condition
THEN result, plus its inputs, outputs, stores, and tests. The rules are the contract.
Implementation picks the data structures. Where a number is a placeholder, it is marked
PLACEHOLDER and listed in section 7.

**Terms.**

- **Scene.** The objects on screen at one moment, with their colours, positions, and kinds.
- **Move.** One action sent to the game. A click is a move.
- **Level instance.** One attempt at a level. A death or a reset starts a new level instance
  of the same level. Working memory belongs to the game, not to the level instance.
- **Rule.** A stated regularity about objects, actions, events, or experiences. It has a
  status: hypothesis or fact (section 4.2).
- **Hypothesis (goal hypothesis).** One way to complete a level task, with a probability.
  Different from a rule hypothesis, which is a candidate regularity. The text says which.
- **Sequence.** An ordered list of moves that serves one task.
- **Fork.** A point where the agent can reach the same goal by two or more paths
  (Kameron, 2026-10-31: "a plan or path that could go two ways toward the same goal").

## 1. Scope, goals, and constraints

**Goal.** An agent that completes levels in games it has not seen, by learning how the game
works from play, and that can show this learning is real. Kameron's words: the agent "must
have true learning and fluid intelligence capabilities."

**Scoring.** Per level, the score is `min(human_actions / agent_actions, 1)^2`
(`docs/agent-submission.md`). The game score weights levels by level index, and the total is the
mean over games. The agent never reads the human action counts at runtime. The counts exist
in the local game metadata (`baseline_actions`) and the scorer uses them. So every "moves
per level" target in this spec is an estimate the agent builds from its own completions
(section 4.8, horizon rule).

**Runtime and environment.** 110 private games. 9 hours of runtime. No internet during the
run. Offline model weights must be open (competition rules, section 6 of the rules as pasted
2026-10-29). Each level gives a move counter in some games and none in others.

**Constraints that govern every module.**

- **C1. No tuning on the public games.** The 25 public games are observation only. Their logs
  show gaps in learning, intelligence, and code. They are never used to pick a threshold,
  a move, or a rule for a named game (Kameron, 2026-10-30: "games should not be used for
  tuning ... We can use the played games as observations to understand gaps").
- **C2. No hard-coded moves and no per-game answers.** No module names a game, a level, or a
  sprite. Rules come from observation.
- **C3. Concepts are learned, not coded.** Mechanics are learned as rules. The agent should
  combine learned concepts (for example, jump plus run gives a running jump). Where a
  general strategy is written down, it is a skill with a trigger on object and event kinds
  (section 4.9), not a rule for one game.
- **C4. The model decides, the simulator checks.** The language model proposes plans. The
  simulator predicts each plan before a real move. The executor checks each real move against
  the prediction (Kameron, 2026-10-31: the LLM "uses a simulated model of the game to check
  the next few moves and then ... write the next few sequence of moves before it has to
  remodel and simulate again").
- **C5. Every step is logged.** A move without a log record is a defect.

**Final submission date.** Kameron's comment (2026-10-30): "11-1 is the last and final
submission allowed." An earlier note in this project said 2026-11-02 23:59 UTC. This spec
uses 2026-11-01. Open item O1 asks for confirmation against the competition page.

## 2. The whole game, end to end (depth 0)

Each game is one run. A run plays one game from start to end. The end of a game is a win,
or the game ending with no moves left, or the run deadline.

```mermaid
flowchart TD
  A["Start game: read game id and seed"] --> B["Load long-term store: rules, sequences, index"]
  B --> C["Working memory empty for this game"]
  C --> D["Start level instance: read frame, build scene"]
  D --> E["Planning loop for this level instance (section 3.2)"]
  E --> F{"Outcome of the move loop"}
  F -->|level complete| G["Level consolidation: promote rules, record sequence"]
  G --> H{"More levels in this game?"}
  H -->|yes| D
  H -->|no, game won| Z["Game consolidation: write long-term, clear working memory"]
  F -->|death or reset| I["Adverse event: record harm, keep working memory"]
  I --> D
  F -->|no moves left or deadline| J["Stop game: write partial log, clear working memory"]
  J --> Z
  Z --> K["End of game: scores and log written"]
```

Reading the graph. The loop from "Start level instance" back to itself is the level cycle.
A death does not clear working memory (Kameron, 2026-10-30: "working memory should not clear
on a game lost since that memory will be needed to replay"). Consolidation writes to
long-term memory during the run, which Kameron allowed (2026-10-30).

## 3. Process graphs (depth 1 and 2)

### 3.1 Depth 1: the per-move loop

This runs once for every move. It is the heart of the agent.

```mermaid
flowchart TD
  M0["Observe frame and build scene"] --> M1{"Level changed?"}
  M1 -->|yes| L1["Go to level consolidation"]
  M1 -->|no| M2{"Game over or reset?"}
  M2 -->|yes| L2["Go to adverse event handling"]
  M2 -->|no| M3["Check last prediction against this scene"]
  M3 --> M4{"Plan step ready and still valid?"}
  M4 -->|yes| M5["Execute next plan step"]
  M4 -->|no| M6["Replan: planning loop (section 3.2)"]
  M6 --> M7{"Plan found with a fork or a single path?"}
  M7 --> M5
  M5 --> M8{"Gate tripped? time per move or LLM tokens"}
  M8 -->|yes| M9["Fallback move from simulator, log the trip"]
  M8 -->|no| M10["Send action to game"]
  M9 --> M10
  M10 --> M11["Record move: prediction, observed, self, stage, branch"]
  M11 --> M0
```

Notes. "Check last prediction" is the miss test: a miss that matters stops the plan
(section 4.10). "Replan" is the planning loop. A gate trip never stops the run. It sends a
fallback move and logs it (section 4.10).

### 3.2 Depth 1: the planning loop

This is Kameron's planning process, used by the agent for its own planning (Kameron,
2026-10-31: "its for the agent's own planning"). It runs on every replan and at each level start.

```mermaid
flowchart LR
  P1["Framing: build level model and flags"] --> P2["Objectives and rules: goal hypotheses from rules"]
  P2 --> P3["Alternatives: generate hypothesis list with probabilities"]
  P3 --> P4["Modeling and trade-offs: simulate each, score by expected value"]
  P4 --> P5["Execution: write sequence segment, execute move by move"]
  P5 --> P6["Evaluation and learning: check, update rules and probabilities"]
  P6 --> P1
```

Reading the graph. Evaluation returns to Framing, but a miss that matters returns to
Alternatives first, from the last fork (section 3.6). Framing is repeated only when the
level changes or when the level model needs rebuilding.

### 3.3 Depth 2: how alternatives are generated

```mermaid
flowchart TD
  G1["Read rules for this level and this game"] --> G2["Read goal kinds the scene supports"]
  G2 --> G3["For each goal kind, find candidate targets from flags and kinds"]
  G3 --> G4["Ask retrieval for the rules that link mover to target"]
  G4 --> G5{"Any link with probability above the cut?"}
  G5 -->|no| G6["Add an exploration hypothesis: test the nearest unknown"]
  G5 -->|yes| G7["Build a candidate path per link using the simulator"]
  G6 --> G8["Store hypothesis list with each path probability"]
  G7 --> G8
```

### 3.4 Depth 2: the expected-value choice

```mermaid
flowchart TD
  V1["Take one hypothesis from the list"] --> V2["Simulate its path to the goal, up to horizon"]
  V2 --> V3["Get path probability and expected moves"]
  V3 --> V4["Get expected adverse cost: probability of harm times restart cost"]
  V4 --> V5["Expected value = reward times probability, minus expected moves cost, minus adverse cost"]
  V5 --> V6{"More hypotheses?"}
  V6 -->|yes| V1
  V6 -->|no| V7["Choose highest expected value; ties by four-decimal probability"]
  V7 --> V8["Store choice and its numbers in the move record"]
```

### 3.5 Depth 2: the sequence write and execute cycle

```mermaid
flowchart TD
  S1["Take the chosen hypothesis"] --> S2["Build the planner briefing (section 4.9)"]
  S2 --> S3["Planner proposes a segment: up to horizon moves, or a named skill"]
  S3 --> S4["Simulator checks the segment move by move"]
  S4 --> S5{"Segment reaches goal or stays safe?"}
  S5 -->|no, unsafe| S6["Ask planner again with the failed step noted, once"]
  S6 --> S4
  S5 -->|no, twice| S7["Drop hypothesis, take next one from the list"]
  S5 -->|yes| S8["Store sequence segment with ids"]
  S8 --> S9["Execute first move, check it, then next"]
  S9 --> S10{"Observed matches prediction?"}
  S10 -->|yes, more in segment| S9
  S10 -->|yes, segment done| S11["Replan only if goal not reached"]
  S10 -->|no, matters| S12["Stop segment: go to fork handling"]
```

### 3.6 Depth 2: fork handling and replan

Kameron, 2026-10-31: "it should keep the moves that got there" and the agent "would take the
shortest path or least dangerous path, whichever is more favorable for the score."

```mermaid
flowchart TD
  F1["Stop segment after a miss that matters"] --> F2["Mark current branch failed for this goal instance"]
  F2 --> F3{"Newest fork has an untried path?"}
  F3 -->|yes| F4["Plan a return route to that fork using known moves"]
  F4 --> F5["Execute return route, checking each move"]
  F5 --> F6["Take the untried path from the fork"]
  F3 -->|no| F7{"Older fork has an untried path?"}
  F7 -->|yes| F4
  F7 -->|no| F8["Replan from the root: current scene, all hypotheses"]
  F6 --> F9["Back to sequence write (section 3.5)"]
  F8 --> F9
```

### 3.7 Depth 2: learning from one transition

```mermaid
flowchart TD
  L1["Transition: before scene, action, after scene"] --> L2["Find rules that predicted something about this action or these objects"]
  L2 --> L3{"Each rule: did its prediction hold?"}
  L3 -->|held| L4["Add confirmation to the rule"]
  L3 -->|missed| L5{"Does a new condition explain the miss?"}
  L5 -->|yes| L6["Refine the rule: add the condition, keep its counts"]
  L5 -->|no| L7["Add contradiction; lower probability"]
  L3 -->|no rule covers it| L8["Create a hypothesis rule from the transition"]
  L4 --> L9{"Hypothesis meets promotion test?"}
  L6 --> L9
  L7 --> L10{"Probability below low band?"}
  L8 --> L9
  L9 -->|yes| L11["Promote hypothesis to fact"]
  L9 -->|no| L12["Keep as hypothesis"]
  L10 -->|yes| L13["Retire rule: keep the record, stop using it"]
  L10 -->|no| L12
```

### 3.8 Depth 2: retrieval

```mermaid
flowchart TD
  R1["Query: canonical features of the situation"] --> R2["Read postings for each feature from the index"]
  R2 --> R3["Score each candidate: probability of being correct"]
  R3 --> R4["Compute cut for this query in real time (section 4.3)"]
  R4 --> R5["Keep candidates at or above the cut, best first"]
  R5 --> R6{"Any candidate kept?"}
  R6 -->|yes| R7["Return ranked list with probabilities"]
  R6 -->|no| R8["Return empty; caller uses exploration or a question to the planner"]
```

### 3.9 Depth 2: harm, inhibition, and perseverance

```mermaid
flowchart TD
  H1["Candidate action in a simulated path"] --> H2{"Game over predicted within horizon?"}
  H2 -->|yes| H3["Adverse path: probability is certain-band or higher"]
  H2 -->|no| H4["Read adverse rules and inhibition network for this action"]
  H4 --> H5{"Adverse probability band"}
  H5 -->|certain| H6["Veto: never take it, and the path is dropped"]
  H5 -->|medium| H7{"Override reason present?"}
  H7 -->|no other way to goal| H8["Allow with logged reason"]
  H7 -->|expected value of goal beats harm| H8
  H7 -->|neither| H6
  H5 -->|low| H9["Allow"]
```

### 3.10 Depth 2: curiosity checks

```mermaid
flowchart TD
  K1["Scene: new look or unknown pair found"] --> K2["Set flag to check, with source"]
  K2 --> K3["Give the flag an information value and a cost"]
  K3 --> K4["Simulate the assumed outcome: does it reach a goal?"]
  K4 --> K5["Put flags and their numbers in the planner briefing as questions"]
  K5 --> K6["Planner chooses a check or a plan step"]
  K6 --> K7["Check move: touch or click; record outcome as rule evidence"]
```

### 3.11 Depth 2: self, focus, and controller

```mermaid
flowchart TD
  C1["Each move: read available actions and controlled things"] --> C2["Self record: I, my character, click"]
  C2 --> C3["Focus record: level, goal, target, mode, budget, signals"]
  C3 --> C4["Controller reads signals in its order"]
  C4 --> C5{"A handoff rule fires?"}
  C5 -->|no| C6["Keep current method"]
  C5 -->|yes| C7["Hand off to the named method, log rule and reason"]
  C6 --> C8["Self and focus records go into the planner briefing"]
  C7 --> C8
```

### 3.12 Depth 2: the planner (model-first performance)

```mermaid
flowchart TD
  Q1["Build briefing: lists from section 4.9"] --> Q2["Planner model reads briefing and tool list"]
  Q2 --> Q3["Planner writes a short program or a skill call"]
  Q3 --> Q4["Sandbox runs it: tools only, time and token limits"]
  Q4 --> Q5{"Program asked for an action?"}
  Q5 -->|yes| Q6["Executor: simulator checks it, then sends it"]
  Q5 -->|no, asked a question| Q7["Tool answers from store or simulator"]
  Q7 --> Q2
  Q6 --> Q8{"Segment finished, missed, or limit hit?"}
  Q8 -->|no| Q2
  Q8 -->|yes| Q9["Return to the per-move loop"]
```

