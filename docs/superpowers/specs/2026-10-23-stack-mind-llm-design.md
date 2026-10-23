# ARC-AGI-3 entry: the stack + Genix Mind + offline LLM

Status: draft for Kameron's review, 2026-10-23. Written on Opus; Fable reviews and
redesigns during the week of 2026-10-26.

## Why

Kameron's research architecture is Genix Mind. The ARC Prize 2026 ARC-AGI-3
competition is the public test for it, and a high score funds further research.
The entry must be our own work: the stack (planning and exploration layers), Genix
Mind's mechanisms on top, and an open-weight LLM used only on the slow layer, to
write the rules of a game. Franzen's public notebook was studied as reference
reading; none of its code, harness, settings or prompts go into this entry, and the
paper says so.

Winning comes first. Where the paper's pre-registered design and winning pull apart,
winning wins and the deviation is recorded for the paper.

What the measurements so far say about where points are (sources in
`docs/experiment-log.md`):

- On the 25 public games, an LLM agent lost about 54 of 100 points per game to levels
  it never finished and about 4 to extra actions. Finishing levels is the goal;
  action efficiency is second.
- An open-weight LLM (Qwen3.6-27B) can write a Python `step()` that predicts the next
  frame: held-out exact match ft09 0.89 to 1.00, vc33 0.53, ls20 up to 0.25. It works
  on some games, not yet on most.
- A model-free explorer alone scored 0.36 offline and our probe agent 0.11 on the
  leaderboard. Exploration without a model doesn't finish levels.
- An unfinished level scores 0 no matter how many actions it took. A level finished
  at 3x the human action count still scores 11 of 100. So the agent keeps trying a
  level while time remains; there's no action-count give-up rule.

## Goals

1. A Kaggle submission of our own agent that finishes as many levels as possible
   across the 110 private games within 9 hours, offline, on one RTX PRO 6000.
2. The agent's structure is the stack + Genix Mind + offline LLM, with the four
   HCQM-mapped modules (metacognition, inhibition, VOI, transfer) as swappable parts.
3. Every run writes logs complete enough to compute the paper's metrics afterwards.
4. Final submission by 2026-11-02 23:59 UTC; ablation runs 2026-11-03 to 11-07 for
   the paper due 2026-11-08.

## Non-goals (for the competition build)

- The 10-config ablation with 3 seeds and tuned/reported game splits. The structure
  supports it (null modules, logs); it runs after the final submission.
- Fine-tuning the model weights. Revisit in the Fable week only if logs show wrong
  rules are the main cause of lost levels.
- Minimax with NNUE or hybrid evaluators, ASP/clingo, MCTS, and a cross-game mechanics
  library. Considered in the 2026-10-19 comparison page; cut for time.

## Anti-goals (a failure even if it runs)

- Any Franzen code, prompt or configuration in the entry.
- A submission that wasn't measured on the 25 public games first.
- A run that ends at 0.00 because the gateway closed the scorecard (15 minutes with
  no action closes it).
- Repeating an action already shown to do nothing or to kill the agent.

## Constraints

- Kaggle: notebook only, internet off, under 9 hours, RTX PRO 6000 (97 GB), one
  submission per day, two final selections, deadline 2026-11-02 23:59 UTC.
- Prize eligibility: open-source system and open weights. Public pretrained weights
  are fine (Kameron, 2026-10-23).
- Scoring: per level min(human/agent actions, 1) squared, weighted by level index
  within a game, mean over games. Human counts for private games are unknown to the
  agent.
- Agent interface: the toolkit's `Arcade` `reset`/`step` path under one scorecard,
  as `arcagent/agent/submit.py` already does (it plays games one after another today;
  the scheduler extends it to concurrent games, and the first build checks that one
  scorecard accepts them).
- Kaggle GPU quota: about 30 GPU hours per week for interactive runs, shared by every
  eval kernel (one 25-game eval is about 2.3 GPU hours). Competition reruns of
  submissions don't draw on it. Evals are rationed accordingly (see Evaluation).
- Model and serving: Qwen3.6-27B-FP8 under vLLM 0.19 from the public Kaggle datasets
  the rule-synthesis probe used (`driessmit1/arc3-vllm-h100-wheelhouse-v3`,
  `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`). Thinking off. Measured median
  82 to 233 s per synthesis call when called one at a time; vLLM batches concurrent
  calls.
- Engineering rules from Kameron's global instructions: ruff, pyright, pytest,
  small units, explicit contracts, bounded everything.

## Decided (by Kameron)

- Structure: the stack + Genix Mind + offline LLM, LLM on the slow layer only.
- Nothing from Franzen.
- Public open weights.
- The four HCQM modules plug in through interfaces; switching one off means using a
  do-nothing version, never editing agent logic.
- B0 is a competent baseline: systematic explorer plus greedy use of learned action
  effects.
- Append-only JSONL logs per run are the paper's dataset; metrics are computed from
  logs in one place.
- Ablation after the final submission.
- Both final selections on 2026-11-02 are this agent. No Franzen-based submission
  (duck v1 or v2) is selected, even as a backup (Kameron, 2026-10-23).
- The agent carries Genix Mind's own structure (the three loops, the wisdom function,
  working memory, sleep, drive state), mapped below from the design of record
  `genix-mind/docs/superpowers/specs/2026-09-03-mind-architecture-design.md`.

## Design

### Genix Mind in the agent

The live mind runs on genix-server with a Claude session as its executive; Kaggle is
offline and bans API models, so the entry re-implements the mind's structure in Python
inside the notebook. Each part of the agent belongs to one of the mind's loops, and the
mind's principles carry over: no generative model decides salience, forgetting or
consolidation; the predictive loop never acts; every channel into an executor passes
the wisdom function.

| Mind part (design of record section) | In the agent |
|---|---|
| Conscious loop, the executive (5) | The controller, the planner and the LLM synthesizer. The executive reasons (the LLM writes rule programs), plans (A*) and acts. It pushes each accepted program to the predictive loop as an expectation. Metacognition and inhibition live here, as in section 10.1. |
| Unconscious loop, predictive (5) | Belief. Every live program predicts the next frame for the action taken; the frame arrives; each prediction is compared; prediction error and running fit update. It never chooses an action. Perception (objects, frame hash) is its encoder. |
| Subconscious awake, salience (5) | Scores every transition deterministically: prediction-error magnitude (novelty), level-up or game over (urgency), the region that changed, cost of a response. Above the wake threshold it wakes the executive, which means an LLM synthesis call. Below it, the event is logged and B0 or the plan continues. VOI runs here as curiosity credit when no plan exists. |
| Subconscious asleep, sleep (8) | Runs at every level-up and when a game ends. Compaction of the level's transitions; consolidation of each surviving program into the game's procedural store; verification of every program against every stored transition; replay, the only point where programs are re-scored offline; goal evidence stored. Transfer is the hand-off from sleep to the next level's belief. |
| Working memory (4) | The current level's transitions, live programs, plan and goal hypotheses, each with a timer in actions. Expired items drop to the game's cold store; a re-seen state restores them. Sleep empties working memory into the stores. |
| Wisdom function (10.2) | Deterministic gate on three channels: an LLM program into belief (sandbox pass, fit threshold), a module vote into the controller (inhibition vetoes), and a plan into action (the program still predicts correctly). Outputs pass, pass with caveat, hold for sleep, or drop; every drop is a logged event. |
| Drive state (7) | Per game: wall time left, actions spent on the level, GPU queue depth, synthesis calls left. As time runs short, salience raises its wake threshold and the executive gets smaller planner bounds; the scheduler moves time between games. |

The four paper modules are the HCQM constructs that this structure exercises
(metacognition and inhibition in the executive, VOI in salience, transfer in sleep),
each a swappable `Protocol`. Turning a module off for the ablation leaves the loops
in place.

### The per-game loop

Each game runs this loop until it's won, the game's wall budget ends, or the run
deadline arrives:

```
observe frame -> perceive (objects, frame hash) -> update belief with the transition
  -> modules vote -> controller picks one action -> act -> log
```

The controller's order of preference each step:

1. **Follow the plan.** If a plan exists and the active rule program predicted the
   last frame correctly, take the plan's next action.
2. **Plan.** If belief holds a rule program above the confidence threshold with a goal
   test, run the planner to its goal and follow the result.
3. **Learn.** Otherwise VOI picks the action that best separates the live rule
   programs (or, with one program and no goal, reaches a state the program hasn't
   seen). Inhibition removes vetoed actions first.
4. **Fall back.** With no rule program at all, B0 acts.

A level-up resets the plan and hands the old level's belief to transfer.

### Components

All live under `arcagent/`. Each module is a `Protocol` with a null implementation.

| Component | Responsibility |
|---|---|
| Perception | Grid to objects (connected components, colors, bounding boxes) and a frame hash. Logs its object list so perception errors can be told apart from decision errors. |
| Belief | The live rule programs (each a `step()` and optional `goal()` written by the LLM), each with its fit over every observed transition in the current level, plus B0's action-effect table. |
| LLM synthesizer | The slow layer. Asks the model for K candidate programs (seeded samples) from the transitions so far; verifies each against every observed transition in a sandbox; adds those above a fit threshold to belief. Reuses `arcagent/synth` (prompt, verify, repair). |
| Planner | Breadth-first or A* over the best program's `step()` to its `goal()`, over the game's simple actions plus one click per object. Bounded by nodes and seconds. Starts from `scripts/duck_planner.py` (commit e0794ae), moved into the package as `arcagent/plan/`. That file is our own generic search code; its name and docstring date from the Franzen period and both get rewritten so the entry carries no reference to that work (my call). |
| Metacognition (HCQM 2.1) | Tracks each program's confidence as its running prediction accuracy; drops programs below the threshold; when belief empties, asks the synthesizer for new candidates. |
| Inhibition (HCQM 2.5) | Vetoes an action in a state where its predicted effect was disconfirmed, or where it did nothing or caused game over before. |
| VOI (HCQM 5.1, 8.4) | When no plan exists, scores candidate actions by how much the live programs disagree on their outcome (expected information gain) and picks the best. |
| Transfer (HCQM 6.3) | At level-up, carries the surviving programs, the goal evidence and the action-effect priors into the next level as starting belief, re-verified on the new level's first transitions. |
| B0 | Systematic explorer (state graph, untried actions first, routes to the frontier; `arcagent/agent/explore.py`) plus greedy use of learned action effects: prefer actions whose past effect moved toward a changed region. Always available; the fallback and the ablation baseline. |
| Controller | The preference order above. Pure decision logic; no I/O. |
| Run log | Append-only JSONL per game: frame hash, action, controller branch taken, module votes, belief state (program ids, fits), LLM calls, planner results, level events. |
| Scheduler | Runs games in parallel threads against one model server, shares the run deadline, writes `submission.parquet`. Extends `arcagent/agent/submit.py`. |
| Metrics | One script computes every metric from the logs: levels completed, actions per completed level, model prediction accuracy, and the four module metrics. |

### Goal inference

A* needs a goal, and on a game's first level the agent has never seen one end. The
plan:

- The synthesis prompt asks for `goal()` as a hypothesis from the layout (targets,
  matching colors, an object reaching a marked cell). VOI and the planner pursue goal
  hypotheses like any other.
- When a level ends, the frames just before and after are stored as goal evidence and
  every program's `goal()` is re-scored against it. Transfer carries the evidence, so
  from level 2 on the goal is grounded in an observed win.
- With no goal hypothesis, the planner targets states the program predicts but the
  agent hasn't visited (model-based exploration), which beats B0's blind wandering.

### LLM budget

The GPU is shared by every game. Salience decides when the executive wakes, so it
decides when the LLM is called: at a new game (after a short B0 exploration to gather
transitions), when prediction error crosses the wake threshold and metacognition
empties belief, and when transfer fails on a new level. Games run concurrently so vLLM batches their calls.
Each game gets a cap on synthesis calls and a wall budget; the scheduler moves time
from finished games to unfinished ones. Exact numbers are the builder's to set from
the first public-game run.

### Keeping the scorecard open

The gateway closes a scorecard after 15 minutes with no action. The scorecard opens
only after the model server answers its health check, and games start with B0
exploration, so actions flow from the first minute. `scripts/duck_keepalive.py` isn't
used (it patches Franzen's session class).

### Logging and metrics

Each game writes `logs/<run>/<game>.jsonl`, one record per action and one per event
(LLM call, plan, level-up, program dropped, veto). Records carry enough to tag
failure patterns: loops (repeated frame hashes), repeated disconfirmed actions,
stuck states, perception mismatches. `scripts/mind_metrics.py` computes every number
from the logs; nothing is computed inline in the agent.

### Paper metrics (pre-registered 2026-08-08)

| Module | Metric | Prediction when removed |
|---|---|---|
| Metacognition | abandonment latency | latency and retests of dead hypotheses rise |
| Inhibition | disconfirmed-action repeat rate | repeat rate and actions per level rise |
| VOI | belief-entropy change per action | yield falls; more wandering before level 1 completes |
| Transfer | later-level actions vs cold start | level 1 unchanged; later levels degrade (key test) |

Every config also reports levels completed, actions per completed level, and model
prediction accuracy.

The ablation configs (from the paper's design): B0; B0 plus each module (4); FULL;
FULL minus each module (4). The null-module set follows: each of the four modules has
a do-nothing version, and B0 runs with all four null.

The pre-registration gives the metrics by name only. So the log stores the raw
ingredients and the metric definitions live in `scripts/mind_metrics.py`, where they
can be fixed before the ablation without re-running anything (my call). Per action:
every live program's id, its prediction for the taken action (hash), whether it
matched, and its running fit; programs dropped and why; vetoes; VOI's scores for the
candidates. Belief entropy is computed over programs weighted by running fit,
normalized (my call, changeable in the metrics script).

## Evaluation and submission gates

- **Offline loop**: the 25 public games through the local engine, CPU only, with the
  LLM replaced by replies saved from earlier runs, for fast development of everything
  except synthesis quality.
- **Public-game eval kernel**: the full agent with the live model on Kaggle, all 25
  public games, one pass, logs saved as kernel output. It's the measure for every
  change.
- **Eval rationing** (my call): the offline loop runs on every change; the GPU eval
  kernel runs at most once a day, about 16 GPU hours over the build, leaving the rest
  of the weekly quota for the ablation week, which reuses saved LLM replies where it
  can so most configs run on CPU.
- **Submission rule**: the comparison baseline restarts at our own agents (explorer
  0.36 offline, probe agent 0.11 on the leaderboard); duck v2's 27.22 isn't the bar.
  Submit on a day when the eval shows more levels completed than our last submitted
  version. The first submission goes in once the agent finishes levels the explorer
  alone can't. Daily slots and both final selections go to this agent only
  (Kameron, 2026-10-23).

## Schedule

| Date (UTC) | Milestone |
|---|---|
| 10-23 to 10-24 | Spec approved, plan written; package skeleton, interfaces, null modules, run log, B0 controller; offline loop runs end to end |
| 10-25 to 10-26 | Synthesizer in the loop (K candidates, verify), planner wired to belief, metacognition and inhibition; first public-game eval kernel; entry + team merge deadline 10-26 |
| 10-27 to 10-29 | VOI, transfer, goal inference; daily evals; first submission |
| 10-27 to 11-01 | Fable review and redesign; daily eval and submission |
| 11-02 | Final two selections by 23:59 UTC |
| 11-03 to 11-07 | Ablation: 10 configs, at least 3 seeds, tuned and reported game splits; paper due 11-08 |

## Plan horizon

The plan covers everything through a submittable agent with all four modules and goal
inference, because the paper needs all four. The Fable week may redesign components;
the interfaces stay, so a redesigned module replaces one implementation (my call).

## Left to the builder

- Confidence threshold, K candidates per call, synthesis call caps, planner node and
  time bounds, per-game wall budget, concurrency: set from the first eval run.
- Object extraction details and click candidate choice.
- VOI's exact information measure (outcome disagreement across programs is the
  starting point).
- File layout inside `arcagent/` beyond the components named above.
- Salience weights, wake thresholds and working-memory timer lengths: start from
  simple fixed values and tune from the first eval's logs.
