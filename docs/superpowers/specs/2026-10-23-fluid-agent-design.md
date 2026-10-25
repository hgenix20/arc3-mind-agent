# Fluid agent: the stack + Genix Mind playing by prediction

Status: draft for Kameron's review, 2026-10-23. Replaces how the mind entry
chooses moves (`docs/superpowers/specs/2026-10-23-stack-mind-llm-design.md`
stays the record of the build it modifies).

## Why

Eval v1 of the mind entry finished 13 levels on the 25 public games (score
estimate 0.32), and the planner given the game's true rules finished 13 of
183 (2.91). The current design asks the LLM for a program that imitates the
game, then searches blindly for a level change. Blind search never gets deep
enough, and every wasted action is squared against the human count.

ARC-AGI-3 measures fluid intelligence. HCQM 1.2 defines it as the "Ability to
solve new problems without relying on prior learned knowledge", with rule
inference, pattern abstraction, relational reasoning and novel problem solving
as its parts (HCQM_v1.2.md:1065). The agent has to do what a person does on a
new game: look at the screen, form a guess about the goal, learn how the
environment behaves from a few deliberate tests, predict each move before
making it, and carry what it learned forward.

## Goal and measure

- Kameron's target: a score of 80 on the 25 public games, run by us with
  `scripts/mind_metrics.py` (which reproduces the toolkit scorecard). Get as
  close as possible before the next submission (2026-10-24); keep iterating
  to the final selection on 2026-11-02.
- The same code varies 4 to 7 points between runs, so a change counts as
  better only when it wins on paired runs (same games, two runs each, or a
  per-game comparison), never on one run.
- Each agent move should be worth one to two human moves. Actions taken is
  the cost the design minimizes, second only to finishing levels.

## Non-goals and anti-goals

- No Franzen code or structure, and no other team's harness. Ideas from
  published work are allowed (recorded in `.superpowers/recon/`).
- No heavy tooling around the model. Plain move logs searched directly beat
  summarized notes.
- No memorized solutions. The agent memorizes how the environment works.
- No genre priors. A game is never treated as a familiar game (Tetris,
  Sokoban, Breakout). Shared root mechanics (move, click, jump, push,
  collect, toggle, rotate, spawn, destroy) may be proposed as hypotheses, and
  each must pass a test before it is trusted.
- The laptop's RTX 4080 runs nothing outside a window Kameron grants.
- Anti-goal: an agent that thinks for minutes per move. Time lost to thinking
  costs more than one well-chosen test move.

## Three failures the design plans against

From ARC Prize's analysis of frontier models, per Kameron:

1. An effect noticed but never turned into a rule. Every unexplained change
   in a frame must end as a fact or as an open question with a planned test.
2. A game treated as a familiar one. Prompt and memory carry only root
   mechanics as hypotheses; nothing names a known game.
3. A level won on a misread mechanic that breaks later levels. After every
   level-up, the facts used by the winning moves are checked against the
   move log, and any fact with a miss is demoted before the next level.

## The loop, per move

```
frame in
  -> perception: drop noise, objects, diff            (subconscious, no LLM)
  -> predictive loop: did the last prediction hold?   (unconscious, no LLM)
       hit  -> next step of the current plan, predicted from facts
       miss -> stop the plan, record the miss, revise the fact
  -> salience: is an LLM turn needed?                 (subconscious, no LLM)
       no  -> act
       yes -> executive turn (LLM): goal, plan with predictions, or a test
  -> wisdom: veto moves predicted to lose, repeat a dead test, or do nothing
  -> act, log the transition
level-up / game end -> sleep: consolidate facts, check the winning facts, transfer
```

Most moves come from stored facts with no LLM call. The LLM is called when
something new needs a decision.

## Components

Each maps onto a Genix Mind part. Where a module exists in `arcagent/mind/`
it is changed in place; the list says what is kept and what is replaced.

### 1. Perception (subconscious salience): noise, objects, diff

- Frames: the last frame of each step's animation is the state; the
  animation frames are kept in the move log for inspection.
- Noise: a cell region is noise when it changes regardless of which action
  was taken (seen changing under two different actions with the same change,
  or changing on a fixed period). Timers, score bars and animations land
  here. Noise is masked from diffs and from goal checks but stays visible in
  the image. A region that changes only after some actions is never noise.
  The mask is per game and can be revised.
- Objects: connected components per color over the masked grid, each with an
  id, color, cell count, bounding box, and a shape signature invariant to
  position. Ids persist across frames by matching signature and nearest
  position.
- Diff: per object, moved (dx, dy), appeared, disappeared, recolored,
  reshaped; plus cells changed outside any object.
- Click candidates come from objects (one per object, plus its centroid),
  never the 4,096 cells.

### 2. Fact memory (working and long-term memory): a dictionary with a table of contents

- A fact is short: a key, a one-line effect, a structured effect the
  predictor can apply, an evidence list of move ids, a hit and miss count,
  and a status: `thin` under 5 observations (HCQM threshold, architecture
  paper :622), `solid`, or `broken` after a miss it can't absorb.
- Keys are structured so lookup is one step:
  `(action, object signature or "any", context)` for action effects, for
  example `("ACTION3", sig_blue_2x2, "free")` -> "moves 1 cell left";
  `("ACTION3", sig_blue_2x2, "wall_left")` -> "no change".
  Event facts use `("event", name)`: what triggered level-up, game over.
- Table of contents: a second dictionary from behavior tags to fact keys.
  Tags are root-mechanic words (move, block, push, collect, toggle, die,
  win, spawn, rotate, select, no_effect) plus behavior phrases the agent
  defines itself ("moves until blocked"). Tags are behavior keys, not word
  matching: the architecture paper found lexical matching misses paraphrases
  (:515, :668). Reusable nodes: a tag node lists every fact sharing that
  behavior, so a new object that behaves like a known one is predicted from
  the node at once and confirmed with a single move.
- Kinds (Kameron's decision, 2026-10-24: identity apart from look). A thing's
  look (color and shape) is state; its kind is its identity, learned when one
  tracked thing is seen switching between two looks twice. Facts stay keyed
  by look, and every observation of a thing with a kind is also recorded
  under the kind, with a move normalized to "and its look becomes X". The
  predictor uses the kind fact when no usable look fact exists, so a sprite
  seen moving while facing one way is predicted moving while facing another.
  A state-relative look change (a tile that toggles) keeps its look facts and
  its kind fact simply breaks. Goals, protection and novelty name kinds.
- Facts live per game; sleep carries them across levels within the game.
  Nothing crosses games except the root-mechanic tag vocabulary.
- The plain move log (every transition: frame hashes, action, object diff,
  prediction, outcome) is the record. The executive can search it with a
  small query (by action, object, or outcome); it is never summarized.

### 3. Predictive loop (unconscious): predict before acting

- Given the object state and an action, the predictor applies the matching
  facts and returns the predicted diff, or "unknown" when no fact covers it.
  Pure Python over objects, microseconds per call.
- After each move, the actual diff is compared with the prediction. A hit
  adds evidence; a miss stops the current plan, records the miss on the fact
  (with the move id), and creates an open question.
- The same predictor runs short searches: a bounded search over predicted
  object states toward a goal predicate, depth-limited and time-limited (well
  under a second). This replaces the blind planner over LLM programs. Search
  only covers transitions the facts can predict; an unknown transition ends
  that branch and becomes a test candidate.

### 4. Goal inference (conscious executive, with the predictive loop)

- Goal guesses are predicates over objects: reach (object A on object B),
  clear (no objects of a kind left), match (a region equals a template
  region), count (n objects of a kind), and compositions of two.
- Guesses come from the LLM's reading of the screen and from signs of
  progress (a HUD or progress region that changes, a level-up event).
- A guess the move log already rules out is thrown out: if its predicate was
  true on a logged frame with no level-up, it is not the goal.
- The cheapest remaining guess (fewest predicted moves to make it true, by
  the short search) is pursued first.
- A guess whose state is reached without a level-up is excluded for the rest
  of the game and never proposed again.
- On level-up, the guess that was true becomes a confirmed goal form, carried
  to the next level as the first hypothesis, never as a certainty.

### 5. Testing (conscious executive): targeted, one move at a time

- A test is taken only when prediction can't decide: the move needed is
  "unknown", or failed predictions on the current question reach the
  threshold (2 misses on the same fact, a tunable), or the search runs out.
- The test is the single move that settles the most open questions:
  untried (action, object) pairs first, then pairs whose facts are thin,
  preferring a test that also moves toward the cheapest goal guess. Never
  random, and never a repeat of a settled test.
- The answer is written as a fact at once and used on the next move.

### 6. Executive turn (conscious executive, the LLM)

- Input: the screen as an image (upscaled), an image of what changed, and a
  short text: the object list, the facts the table of contents returns for
  the objects on screen, the goal guess list with statuses, open questions,
  and the last few move-log lines.
- Output (JSON): either a plan of up to 8 moves, each with its predicted
  effect and the goal guess it serves, or one test move with the question it
  answers; plus new goal guesses and new or revised facts in the fact format.
- The LLM may return a short function over the object list (run in the
  existing sandbox) to compute positions or targets exactly; the image is for
  seeing, the code is for exact details.
- Thinking off, a capped output length, and a per-turn deadline. On a
  timeout the agent takes the best targeted test instead of waiting.

### 7. Salience (subconscious): when to wake the executive

Wake on: level start, a prediction miss, a plan finished or blocked, a goal
guess excluded, a fact turning `broken`, or no executable plan. Do not wake
while the plan holds and predictions hit.

### 8. Wisdom and inhibition (conscious executive): gate every move

Veto a move predicted to cause game over, a move with a known no-effect fact
in the current context, and a repeat of a settled test. Every veto is
logged with its reason.

### 9. Sleep (at level-up and game end)

Merge duplicate facts, promote facts with 5 or more hits, demote facts with
misses, re-check the winning moves' facts against the log (failure 3),
record the confirmed goal form, and hand the fact dictionary and table of
contents to the next level (transfer). Bounded time.

### 10. Drive and time

Per-game budget from the existing scheduler. Time spent thinking is charged
against moves: a turn that would exceed its deadline yields to a test. The
executive turn and the search are the only places time grows, and both are
bounded.

### 11. Fact effects and contexts (Claude's call)

- A structured effect is one of a small fixed set of effect types the
  predictor applies directly: `move(dx, dy)`, `slide(dx, dy)` (moves until
  blocked), `appear(sig, at)`, `disappear`, `recolor(c)`, `reshape(sig)`,
  `toggle_group(sig)` (every object of a signature changes together),
  `teleport(to)`, `none`, `game_over`, `level_up`. Effects compose as a list.
- Rare effects the set can't express may be a short sandboxed function over
  the object list. The predictor tries the fixed types first and calls the
  sandbox only for those facts, so the common path stays fast.
- Contexts are a fixed set computed by perception: blocked or free in each
  direction, touching a color, inside a region, selected. The LLM may add a
  context as a predicate over objects (sandboxed); it is then computed like
  the built-in ones.
- Hidden mode (Kameron's decision, 2026-10-24: tags on tracked things, no
  global mode slot, no LLM). A selection marker is learned from clicks (a mark
  that follows the clicked thing or goes away on a second click), and the
  active controller is the one controlled thing that answered a non-click
  action when two or more are on screen. Both live on the scene and in its
  key. A click made while a thing is selected gives contexts relative to the
  selection: `selected:dx,dy` for the selected thing, `between:dx,dy` for
  things on the cells between it and the click, `sel:dx,dy` for the clicked
  thing (offsets past 8 cells are `far`); the active thing's facts for
  non-click actions use context `active`. The predictor predicts the next
  selection: a click on a kind seen selected selects it, any other click
  clears it.

### 12. The paper's four modules (Claude's call)

The pre-registered modules stay as swappable protocols with do-nothing
versions, so the ablation and the paper plan keep working: the targeted test
chooser is VOI, the wisdom vetoes are Inhibition, the threshold that decides
"test now" from fact confidence is Metacognition, and sleep's hand-off of
facts and goal forms is Transfer. Log fields `scripts/mind_metrics.py` reads
are kept; new fields are added for the metrics below.

## Delivery in slices (Claude's call)

The build is sequenced so each slice can play and be scored:

1. Perception, noise mask, objects, fact dictionary and table of contents,
   predictor, targeted tests and heuristic goal guesses, with no LLM. Plays
   on CPU, so it iterates offline on all 25 public games with no GPU.
2. The executive turn: images, JSON contract, goal guesses and plans from
   the LLM, salience wake rules.
3. Sleep, transfer, the winning-facts check, metrics.

Tomorrow's submission (2026-10-24) is whichever of v1 or a finished slice
scores higher on the public games. Offline tests use a scripted stand-in for
the LLM that returns contract-shaped replies.

## Metrics we log

From the HCQM vault (`.superpowers/recon/hcqm-fluid.md`), each per level and
per game:

- moves to the first correct rule for a mechanic, and tests per mechanic;
- prediction hit rate, and later-half vs earlier-half hit rate per level;
- rule reuse: fraction of moves predicted by facts from earlier levels;
- revision after a miss (time and moves to a corrected fact);
- impasses (executive wakes with no plan) and search depth;
- goal guesses proposed, ruled out by the log, excluded by reaching, confirmed;
- actions per level against the human count.

## What stays, what goes

- Stays: scheduler, submit runner, notebook builder (vision enabled in the
  request), runlog, sandbox (now for code over object lists), salience,
  wisdom, sleep, drive, metrics script, offline loop, B0 explorer as the
  fallback when the model server fails.
- Replaced: belief over LLM-written `step()` programs, VOI over programs,
  the blind planner as the main search, the synthesizer's program prompts.
- New: noise mask, object tracker, fact dictionary and table of contents,
  fact predictor and short search, goal guess list, targeted test chooser,
  executive prompt and JSON contract.

## Constraints

- Model: Qwen3.6-27B-FP8 on vLLM 0.19, served as now
  (`Qwen3_5ForConditionalGeneration`, image and text input). Qwen3.8-Flash-Next
  (Intel W4A16 AutoRound) is the planned swap once it serves offline on
  Kaggle with our own wheelhouse and our own copy of the weights; that work
  runs separately and doesn't block this design.
- Kaggle: notebook only, internet off, under 9 hours for 110 games, RTX PRO
  6000 (97 GB), one submission per day, scorecard closes after 15 minutes
  without an action.
- Iteration: Kaggle GPU evals on the 25 public games, up to 5 GPU hours a
  day (Kameron, 2026-10-23); paired comparisons for any claimed gain. Quick
  checks of the executive's replies may run first on the laptop's RTX 4080
  with a smaller model, only in windows Kameron grants (50 minutes granted
  on 2026-10-23).
- Repo rules: ruff, pyright strict, pytest, commit trailer as in
  `CLAUDE.md`, no em-dashes in prose, the toolkit's own scoring.

## Left to the builder

The exact shape signature, the object-id matching rule, the noise test's
period detection, the search algorithm and its depth, the prompt wording,
the image scale, and the move-log query form, within the contracts above.
