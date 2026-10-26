# LLM-driven agent: the model of the game plays first, the tools improve themselves

Status: design, from Kameron's answers on 2026-10-26. Builds on the fluid agent
spec (`2026-10-23-fluid-agent-design.md`) and keeps its components; what changes
is who decides.

## Why

Main scores 0.42 offline on the 25 public games and v5 scored 0.39 on the public
leaderboard. Duck, an LLM tool loop we didn't write, scored 43.7 on the same 25
games offline and 27.22 on the leaderboard. The score weights level k by k and
squares the human-to-agent action ratio, so it rewards depth. Duck cleared 106
levels and won 6 games outright; we clear 11 levels, and only vc33 gets past
level 1.

Our agent stalls because hand-written code decides most moves. It only
understands mechanics we've written perception and goal kinds for, and each new
kind took about a day and moved one or two games. The LLM, which could read what
a level means, is called rarely (263 calls over 470,704 actions in v1).

Kameron's direction: the LLM drives, and the deterministic code is its toolkit.
The LLM builds a model of the game, the tools play moves out inside that model
before any real move, and when a tool fails inside the model the LLM rewrites
the tool.

## Goal and measure

- **Success:** a public leaderboard score of 60 or higher.
- **Submission rule (Kameron):** no submission until we expect 60 or more. The
  evidence is the notebook's interactive commit, which plays the 25 public games
  offline and prints a scorecard. Threshold: an interactive-commit scorecard of
  75 or higher. *Claude's call:* duck dropped from 43.7 offline to 27.2 on the
  leaderboard, so the offline number needs headroom over 60. Kameron may change
  the threshold.
- Kernel v6 (main a5b76d7) stays unsubmitted under this rule.
- **Time (Kameron):** the gate is absolute. The schedule is Kameron's to manage;
  the plan builds the whole design and doesn't trade parts away for the date.
- **Branch (Claude's call):** the build lives on its own branch until all three
  modes exist, then gets one full comparison against main. Slices are checked on
  CPU with recorded replies; the LLM-driven loop will likely score below main
  until it's whole, so slice-by-slice merging would block it.

## Non-goals and anti-goals

- **No Franzen code or structure** (standing rule). Prompts, the loop and the
  tool interface are ours.
- **No game-specific code in our repo, and no memorized solutions.** The LLM's
  own knowledge of what grids and objects tend to mean is allowed: it's the
  model's, not code we wrote for a genre.
- **A tool fix must stay general.** A fix may change how a tool works (how
  pushing blocks a path, how a route is ordered). It must never hard-code a
  game's layout, coordinates or move list. Failing this is an anti-goal even if
  the score rises.
- **No real move the model hasn't played out,** except in the experiment mode
  below.
- No new model and no internet: Qwen3.6-27B-FP8 served in the notebook stays
  the LLM.

## The three modes

Every real move comes from one of these.

1. **Simulator plays.** The LLM's model of the game holds. The deterministic
   side searches and checks moves inside the model and sends to the real game
   only moves that played out correctly there. No LLM call per move.
2. **Tool fix.** A tool fails inside the model, for example the pathfinder finds
   no path the LLM can see. The LLM reads the tool's code and the failing case
   and writes a replacement. Play returns to mode 1 with the fixed tool.
3. **LLM experiments.** The simulator can't play the situation out because the
   model is wrong or doesn't cover it. The LLM picks the next real move itself as
   an experiment. The result feeds back into its model, and play returns to mode
   1 once the model predicts correctly again.

## The loop

```
level start
  -> LLM first look (before any move): writes the model of the game
mode 1: simulator plays
  -> search inside the model toward the LLM's goal
  -> plan found: play it out in the model, then send the first move for real
  -> after each real move: compare the real frame with the model's prediction
       match    -> next move of the plan
       mismatch -> stop at once (one wrong move at most); LLM revises the model
  -> a tool fails inside the model -> mode 2
  -> the model can't play it out    -> mode 3
mode 2: LLM rewrites the failing tool; the fix is checked, then shared (below)
mode 3: LLM chooses one real move as an experiment; the result goes to the model
level-up / game end -> sleep (fluid spec section 9), which keeps the model's
  confirmed parts as the first hypothesis for the next level
```

## Components

### 1. The model of the game (written by the LLM)

The LLM's model replaces what our deterministic goal inference and fact learner
try to work out alone. It holds what the fluid spec's executive turn already
returns (section 6):

- goal guesses as predicates over objects (reach, clear, match, count, cover,
  sequence, or a short function over the object list);
- facts in the fact format: what each action does to which objects, in which
  context;
- a plan with a predicted effect per move.

*Claude's call:* where the fact format can't express a mechanic, the LLM may
add a rule function over the object list that the predictor calls for the
objects it names: it takes the objects (as plain dicts: id, sig, color, cells)
and the action, and returns the changed objects, which the predictor merges into
its prediction. It works on objects, not the raw grid, so the simulator and
perception describe the same things. It runs in the tools worker (section 4). The predictor
plus search over this model is the simulator.

The deterministic perception (objects, noise, diff), fact learning from observed
transitions and goal-rule proposals keep running. They become evidence the LLM
sees in its prompt, and its model is the one the simulator uses. *Claude's call.*

### 2. When the LLM is called

- level start (the first look, before any move);
- a mismatch between a real frame and the model's prediction;
- a tool failure inside the model (mode 2);
- no plan the simulator can play out (mode 3);
- a plan finished without a level-up.

Not called while the plan holds and predictions match. Each turn keeps the fluid
spec's bounds: thinking off, capped output, per-turn deadline.

*Claude's call, a change from the fluid spec:* on a turn timeout the game waits
for the LLM instead of taking the tester's move, because the score counts
actions, not seconds. Only when the game's own time budget is nearly spent does
the tester's best targeted test go out, logged as a mode 3 move chosen by the
tester.

### 3. Who plans, and detecting a tool failure

*Claude's call:* the LLM's reply carries the goal, the model and its own rough
plan of moves with a predicted effect each (the fluid spec's plan of up to 8
moves). The deterministic search plans toward the goal inside the model; the LLM
doesn't compose tool calls, so there's no tool API to expose in the prompt.

A tool fails inside the model when:

- the LLM's own plan reaches the goal when played out in the model, but the
  search (with its pathfinder and route ordering) found no plan or a longer one;
- a tool raises or times out;
- an explicit expectation in the reply ("a path from #3 to #7 exists, about 9
  moves") disagrees with the tool's answer. Optional; the first test needs none.

Failures are detected mechanically, never inferred from the score. When the
LLM's plan works in the model and the search's doesn't, the LLM's plan is played
while the fix is made.

### 4. Tool fixes (self-improvement)

- **Patchable tools.** *Claude's call:* the planning tools: the search
  (`mind/search.py`), the pathfinder and planner (`plan/planner.py`), the route
  ordering (`mind/route.py`), goal distance and goal checks, and the predictor's
  movement rules. Not patchable: perception, wisdom vetoes, the sandbox, the
  budget and scheduler, and the scorecard and submit code. Every other tool
  depends on what perception reports, and a broken veto or sandbox can lose a
  game or the whole run.
- **What the LLM sees:** the tool's current source, the failing input, its
  expectation and the tool's actual answer.
- **What it returns:** a full replacement for one function with the same
  signature.
- **Checks before adoption:**
  - the replacement gives the expected answer on the failing case;
  - it agrees with the current version on that tool's regression cases. These
    are recorded inputs and outputs from calls whose plans played out
    correctly, kept per tool and growing during the run. *Claude's call:* this
    stands in for unit tests, which can't run cheaply inside the notebook.
    Seeded at build time from the repo's existing tests for that tool.
  - it stays inside a time bound per call.
- **Where it runs:** a second sandbox worker, a separate process with a
  deadline per request, never the agent's process. *Claude's call:* the existing
  worker runs `python -I` with the standard library only and can't import
  `arcagent`, while `search.py` needs the scene, fact memory, predictor and
  goals. The tools worker is started from `/kaggle/working`, imports `arcagent`,
  and receives the scene, memory and goals once per tool call (one round trip
  per search, not per node). On a crash or timeout the original tool answers.

### 5. Sharing fixes across the run

Kameron's decision: a fix stays for the rest of the run and carries into later
games. The notebook plays 8 games at a time in one process, so this is a shared
toolbox:

- one versioned record per tool: source, the game and case that produced it,
  the reason, and the time;
- every game calls the newest adopted version, including games already running;
- if a fixed version fails where the version before it would have answered
  correctly (checked by running both on the failing case), the toolbox rolls
  back to the earlier version and records why;
- adoption and rollback are logged.

Rules: we couldn't confirm whether the competition allows state shared between
games in one run. Kameron's ruling: if it can't be confirmed, it's allowed.

### 6. Experiments in the real game (mode 3)

The LLM names one move and what it expects the move to show. The real result
is written as facts at once and goes into the next prompt. The wisdom vetoes
still apply: no move predicted to end the game, no repeat of a settled test.

## Metrics we log

All the fluid spec's metrics, plus per game:

- moves per mode (simulator, experiment) and LLM calls per mode;
- model mismatches, and moves lost to them;
- tool failures, fixes proposed, fixes adopted, rollbacks, per tool;
- for every adopted fix: which later games called it, and whether their plans
  played out.

## Constraints

- Kaggle notebook, internet off, under 9 h (`kaggle_mind_notebook.py` budget
  logic stays).
- Qwen3.6-27B-FP8 through vLLM, through the existing `vchat` client.
- The program sandbox is the only place model-written code runs.
- Standing repo rules: ruff, pyright, pytest; the experiment log stays current;
  merging to main only on more levels or a higher score in a full 25-game run.

## Left to the builder

- Prompt wording and the reply's JSON shape, within the fluid spec's executive
  input and output.
- How the regression cases are sampled and capped.
- How the model's rule functions plug into the predictor.
- The per-turn deadlines and token caps, within the existing budget scheduler.
- The offline test harness: recorded replies (`vchat` record and replay) for CPU
  tests, the interactive commit for real measurement.
