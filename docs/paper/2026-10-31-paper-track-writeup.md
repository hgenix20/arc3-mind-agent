# Paper Track write-up: the agent v2 planner on ARC-AGI-3

DRAFT v0.1, 2026-10-31. Structure, figures and numbers by the harness; Kameron
rewrites the prose before anything is pasted into Kaggle. Every number here has
a source line in `docs/experiment-log.md`, `docs/superpowers/plans/2026-10-31-v2-gaps.md`
or `bench/evals/2026-10-31-planner-v1/`. Do not change a number here without
changing it there. Text in `[K: ...]` is a decision or a sentence only Kameron can
supply. This file supersedes `docs/paper/draft.md` (the rule-synthesis probe,
2026-10-20), which is kept as an appendix candidate only.

## 0. What the paper track requires

Read on 2026-10-31 from the Kaggle paper-track Overview and Rules tabs (in the
browser, logged in) and from arcprize.org/competitions/2026/paper.

| Item | Requirement | Source |
|---|---|---|
| Deadline | 2026-11-09 23:59 UTC (Kaggle CLI and the Kaggle page agree; arcprize.org's timeline says November 8) | Kaggle Overview, `kaggle competitions list` |
| Form | A Kaggle Writeup: title, subtitle, body, with a Track selected (ARC-AGI-3) | Kaggle Overview, Submission Requirements |
| Length | "Your Writeup should not exceed 1,500 words. Submissions over this limit may be subject to penalty." | Kaggle Overview |
| Required attachments | Media Gallery with a cover image (required), and a public notebook in Project Links | Kaggle Overview |
| Optional | A PDF of a longer paper through a public Project Link, in place of the Writeup body | Kaggle Overview |
| Notebook | Public, no login or paywall; a private notebook is made public after the deadline | Kaggle Overview |
| Submission ID | Judges ask for the code-competition submission ID to match the notebook; its leaderboard score is the Accuracy criterion | Kaggle Evaluation, arcprize.org |
| Rubric | Six criteria, 0 to 5 each, averaged: Accuracy, Universality, Progress, Theory, Completeness, Novelty | Kaggle Evaluation |
| Prizes | $50k, $20k, $5k; papers above 4.5/5 share a $375k pool at the host's discretion | Kaggle Overview, Rules 1.5 |
| Winner license | CC-BY 4.0 on the submission and its source; pretrained models with an incompatible license are exempt | Kaggle Rules 2.5 |
| Team | Maximum 8; must match the team that made the ARC-AGI-3 submission; one Writeup per team | Kaggle Rules 2.1, 2.2 |
| Suggested sections | Abstract, Intro, Prior work, Approach (algorithm level), Results (leaderboard and public eval, never train-set performance), Conclusion | arcprize.org paper page |
| Tone | "Shorter and clearer is always better." No filler, no unnecessary equations | arcprize.org paper page |
| Status | Kameron has joined the paper track and accepted its rules; no Writeup exists yet | Kaggle page banner |

Consequences for this draft:

- One Writeup per team means one paper for the account. It documents one ARC-AGI-3
  submission. `[K: which one. The candidates are 57844971 (agent v2 planner, score
  pending) and 57746426 (DRIVE v11, 0.39). The curiosity agent cannot have its own
  paper from this account.]`
- The 1,500-word limit covers the Writeup body. The word budget below is for that
  body. A longer version can go up as a PDF through a Project Link; the full detail
  (spec v2, gaps report, logs) stays in the repository and is linked.
- The notebook `hgenix20/arc3-mind-agent` is private. It must be public, or attached
  so Kaggle publishes it after the deadline. `[K: make it public before submitting,
  or attach it private and let Kaggle flip it.]`
- The repository has no LICENSE file. The winner license is CC-BY 4.0; arcprize.org
  asks for a permissive license on code (it names CC0 and MIT-0). `[K: pick the
  license and add the file before the Writeup links the repo.]`
- The model (Qwen3.6-27B-FP8, from the `vrfai` snapshot on Kaggle) is a third-party
  pretrained model. Rules 2.5 exempt incompatible-license models from the grant.
  `[K: verify the weights' license on the model card and state it in section 11.]`

## 1. Word budget for the Kaggle Writeup body

| Section | Words | Carries which rubric criteria |
|---|---|---|
| Abstract | 120 | all |
| Introduction | 180 | Progress, Novelty |
| Prior work | 150 | Novelty |
| Approach | 450 | Completeness, Theory, Universality |
| Results | 250 | Accuracy, Completeness |
| Why it worked where it did, and why it did not elsewhere | 250 | Theory, Progress |
| Conclusion | 100 | all |
| Total | 1,500 | |

The reproduction section (section 11 below) goes in the notebook description and the
Project Link, not in the word count. Sections 3 to 9 as drafted run about 1,800 words
with the `[K]` notes and section labels in; the paste into Kaggle has to come in under
1,500 after the rewrite.

---

## 2. Title and subtitle

Working title: **Plan, simulate, check: a model-first planning agent that learns the
rules of an ARC-AGI-3 game while it plays**

Subtitle: An offline 27B model proposes short move segments, a learned simulator rolls
them out before any move, and the agent keeps a probability on every rule it believes.

`[K: title and subtitle are yours. Alternatives: "Facts with probabilities: planning
under a learned game model on ARC-AGI-3"; "An agent that keeps a book on itself".]`

## 3. Abstract (120 words)

We describe the ARC-AGI-3 entry behind Kaggle submission 57844971. The agent plays
each unseen game by learning its rules as facts with Beta probabilities, planning
against a simulator built from those facts, and asking an offline 27B language model
for short move segments that the simulator checks step by step before any real move.
Every move is logged with its predicted and observed outcome, so the agent's
calibration, its harm avoidance and its hand-offs to a graph explorer are measurable.
On the 25 public games the agent finished 33 levels in one run on competition
hardware (offline scorecard 0.59; public leaderboard `[K: fill in when scored]`).
We report where the planning paid (2 of 33 levels won below the human move count)
and the twelve measured gaps between the design and the play.

## 4. Introduction (180 words)

ARC-AGI-3 gives an agent a 64x64 grid of colour indices, a list of legal action ids
with no meanings, and a score that rewards finishing levels in as few moves as a
human did: per level `min(human_moves / agent_moves, 1)^2`, weighted by level index.
Nothing tells the agent what the actions do, what counts as progress, or what kills it.

Our question was narrower than "solve the games". We asked what an agent needs to
learn how a game works from play, and to show that the learning is real: a memory of
rules with probabilities, a simulator that uses them, a planner that proposes and a
checker that refuses, a record of harm, and a log of every prediction against every
outcome. The entry is the result of building those parts to a written specification
(`[K: link spec v2]`) and then measuring the gap between the specification and the
play on the public games, with no per-game tuning and no hard-coded moves.

`[K: one or two sentences on why this matters to you: fluid intelligence as the
thing being measured, and the HCQM line if you want it cited.]`

## 5. Prior work (150 words)

Every entry below is a candidate to verify at the primary source before one sentence
about it is written. No characterizations until then (house rule: open the source at
the page, not the abstract).

- ARC-AGI-3 Milestone 1 and 2 writeups on arcprize.org (2026-06-30 and 2026-09-30).
  `[verify: what the leading agents do; whether any keeps a probabilistic rule store or
  a simulator; the Tufa Labs notebook whose wheelhouse and model snapshot we reuse.]`
- ARC Prize 2025 paper-track winners (Kaggle "View Writeups"). `[verify: which used a
  world model or test-time learning; how they framed novelty.]`
- Model-based reinforcement learning with learned dynamics (Dyna-style planning,
  MuZero, Dreamer). `[verify: what each learns and plans over; ours is symbolic rules
  with Beta counts, not a neural dynamics model.]`
- LLM agents with a verifier or a tool-checked plan (ReAct-style loops, code-as-policy
  work). `[verify: whether any check a plan against a learned simulator before acting.]`
- Beta-Bernoulli belief tracking and calibration (the textbook basis of our fact
  probabilities). `[verify the citation you prefer.]`

What is different in ours, pending that check: the model never moves the agent
directly. It writes a segment; the simulator rolls the segment out under the current
facts; the executor compares each real frame with the prediction and breaks the segment
on the first miss. Learning is on the rule store, not on the model's weights.

## 6. Approach (450 words)

Algorithm level, one game at a time. Module names in parentheses point at
`arcagent/mind/`.

**Perception.** Each frame becomes a scene of things: connected same-colour
components with a kind id from shape and colour (`objects.py`, `kinds.py`). The diff
of two scenes is the observed effect of the move between them.

**Facts with probabilities** (`facts.py`). A fact is "action A on a thing of kind K
in context C has effect E". Each fact carries hits and misses and a probability
`(hits + 1) / (hits + misses + 2)`, the mean of a Beta(1,1) posterior. Probabilities
fall in three bands: certain at 0.8 and above, low below 0.3, medium between. A
hypothesis becomes a fact after five hits in a row in the certain band. Misses are
never forgotten.

**Simulator and rollout** (`predictor.py`, `rollout.py`). Prediction is read-only:
given a scene and a move, the facts predict the next scene and a step probability
`p_step`. A segment of moves gets a path probability by the chain rule,
`p_path = product of p_step`, with 0.5 for a step no fact covers. Treating steps as
independent was considered and rejected.

**Hypotheses and expected value** (`hypotheses.py`, `plantree.py`). Candidate paths
to each open goal are ranked by `EV = p_path * 100 * efficiency - p_harm * restart_cost`,
where efficiency is the agent's own estimate of moves per level from its completions,
and the restart cost is the moves lost if the path kills the agent. The chosen path is
a node in a plan tree; where paths diverge the node is a fork, so a broken segment
returns to the last fork and replans from there, not from scratch.

**The planner** (`planner_agent.py`, `briefing.py`, `skills.py`). The language model
(Qwen3.6-27B-FP8 served by vLLM on the competition GPU, no internet) reads a briefing:
the scene, the rules and their probabilities, the open goals, the current hypothesis,
the plan tree, recent harm, lessons from earlier games, and its own notes from the
last turn marked unverified. It answers in a fixed contract: its model of the game,
new facts, goals, a segment of at most 12 moves or a skill call (go around, collect,
explore), a question from a fixed list, and labels. Every segment is rolled out before
the first move is sent. Each real move is compared with its prediction; the first miss
breaks the segment.

**Harm, inhibition, perseverance** (`adverse.py`). Deaths and health-bar drops are
recorded against (label, kind) keys with the same Beta counts. A small online logistic
network over situation features (256 weights, learning rate 0.1) gives a harm
probability for situations not yet keyed. A move in the harm band is refused unless
perseverance allows it: no other way to the goal, and fewer than three trials of that
rule in this level.

**Controller and hand-off** (`controller.py`, `agent/hybrid.py`). Fixed order: harm
gate, segment broke, 25 fallback moves in a row, 40 moves without progress, budget. A
per-move time gate of 60 s and a token gate of 4,096 bound every call. After 150 mind
moves on a level, or 25 fallbacks in a row, the level goes to a state-graph explorer
that plays without the model; the explorer's wins are still logged and consolidated.

**Long-term store** (`longterm.py`). Rules confirmed at a level win are written to a
store keyed without colour, read back as lessons by later games in the same run.

**Logging.** Every move writes its branch, hypothesis, segment, fork, predicted and
observed outcome, and gate state. The metrics and the gaps report are computed from
these records alone.

## 7. Results (250 words)

Report leaderboard and public-eval numbers only. Nothing here comes from a training
set; there is none.

**Leaderboard.** Submission 57844971 (notebook `hgenix20/arc3-mind-agent` version 13,
submitted 2026-10-31 16:44 UTC): public score `[K: pending at 2026-10-31 17:30 UTC]`.
Earlier submissions of the same code line: 57746426 (DRIVE v11, the model drives the
fluid agent directly) 0.39; 57637984 (fluid agent v5, no model) 0.39; 57588171 (fluid
agent v1) 0.23.

**Public evaluation, 25 games, one run on the competition hardware** (RTX PRO 6000,
8 game workers, 9-hour budget, 5,511 s used): 33 levels finished, offline scorecard
0.5895 (the same metric as the leaderboard, computed from the local game metadata).
Six games finished no level. Best games: lp85 5 of 8 levels, tu93 5 of 9, vc33 3 of 7.
Per-game scores above 1 point: r11l 4.00, tn36 3.57, lf52 2.09, lp85 1.97, ls20 1.41,
tu93 1.39.

**Who made the moves.** Of 476,787 actions, the planning agent made 6,268 (1.3%) and
the explorer the rest. The agent won 6 of the 33 levels; the explorer won 27. Two of
the six were won under the human move count (lf52 level 1 in 10 moves against 32;
tn36 level 1 in 14 against 32). Mean per-level score on the six: 0.595.

**Model use.** 1,077 planner calls, 11.0 s per call on average, 42% of all game
seconds. 1,077 segments proposed, mean length 5.5 moves; 208 were rejected by the
simulator at step 0.

**Calibration.** Observed match rate ran 0.10 to 0.20 above the predicted bin
everywhere below 0.7: the simulator is underconfident. Table in the gaps report,
section 5.

Figures: see section 10.

## 8. Why it worked where it did, and why it did not elsewhere (250 words)

This section carries the Theory and Progress criteria. The claims are about mechanism,
each with the measurement behind it.

**What the design got right.** A move the simulator predicts correctly is a move that
cannot surprise the agent, so prediction accuracy is the right first metric, and it was
high where facts existed (per-game prediction hit rates 0.46 to 1.00, 20 of 25 games
above 0.75; `metrics.json`, `hit_rate`). Planning under a learned
model let two levels finish below the human count with zero deaths. The log made every
failure below diagnosable from records, with no replay.

**Why the score stayed low.** Three mechanisms, in order of cost:

1. *Harm never reached the veto band.* In tu93 the agent died 68 times to one kind
   while that kind's harm probability sat between 0.05 and 0.14, because trials were
   counted on every move near the kind, not only on moves the simulator predicted
   would touch it. Dilution kept the probability low and the veto off.
2. *A stale counter locked the planner out.* After the explorer won a level, the
   fallback counter was not reset; ten later level instances got 25 fallback moves and
   no planner call before being handed back. Control state must reset on every change
   of who is playing.
3. *The first planner call set the whole level.* Nine games lost call 0 to a timeout
   on the shared model server at cold start, with no retry; two games never got a
   successful call. A single unretried request should not decide a level.

Smaller mechanisms with the same shape (a rule is right, its trigger is wrong): the
fork tree never produced a win because return routes were searched rather than
recorded; perseverance allowed everything when no goal path existed; the long-term
store filled with "does nothing" rules from one game.

**The general lesson** `[K: your words]`: the learning parts worked; the control parts
around them decided the score. That is a statement about where to look, which is what
the Progress criterion asks for.

## 9. Conclusion (100 words)

`[K: write last. Content to carry: what the entry is (a rule store with probabilities,
a simulator, a model that proposes and a checker that refuses); what it did (33 levels,
2 under the human count, 1.3% of moves); what the measurement showed (control, not
learning, set the score); what comes next (the twelve gaps, ranked, in the linked
report).]`

---

## 10. Figures to make (cover image required)

The Media Gallery needs a cover image. Candidates, in order:

1. **Cover.** One game frame with the agent's overlay: things outlined by kind, the
   current segment drawn as arrows, the rolled-out prediction beside the real frame.
   Source: any `logs/<game>.jsonl` record with `p_step` and a frame; render with the
   grid palette.
2. **Per-move loop.** The depth-1 graph from spec v2 section 3.1 (observe, predict,
   plan, gate, act, learn). Render from the mermaid source.
3. **Where the moves went.** A stacked bar per game: planner, learn, fallback, follow,
   explorer. Data: gaps report section 1 table.
4. **Calibration.** Predicted probability bin against observed match rate, all games
   pooled, with the identity line. Data: `metrics.json`, `v2.calibration`.
5. **Harm that never vetoed.** tu93: deaths over moves against the harm probability of
   the killing kind, with the 0.3 and 0.8 bands drawn. Data: `logs/tu93-*.jsonl` harm
   events.

Figure 1 is the cover. Figures 2 to 5 go in the gallery and are referenced from the
body by number.

## 11. Reproduction and links (for the notebook description and Project Link)

- Notebook: `hgenix20/arc3-mind-agent` version 13 (the submitted version). Sources
  ship inside the notebook as one compressed blob; `scripts/kaggle_mind_notebook.py`
  builds it from the repository.
- Repository: https://github.com/hgenix20/arc3-mind-agent (public, default branch `agent-v2`, pushed 2026-11-01), commits 4c24d08 (agent)
  and 4351b0c (notebook). 22,160 lines of Python in `arcagent/`, 64 modules under
  `arcagent/mind/`, 72 test files, 1,077 tests passing, pyright and ruff clean.
- Run: `python scripts/kaggle_mind_notebook.py --games all --config PLANNER --hybrid-moves 150`,
  then push the kernel. Datasets: `driessmit1/arc3-vllm-h100-wheelhouse-v3` and
  `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (both public on Kaggle).
- Model: Qwen3.6-27B-FP8 `[K: license from the model card]`, served by vLLM at
  127.0.0.1:1234, no internet. If the server fails the notebook plays the model-free
  `FLUID` configuration.
- Specification: `docs/superpowers/specs/2026-10-31-agent-spec-v2.md`; review of the
  build it replaced: `docs/superpowers/specs/2026-10-30-fluid-intelligence-agent-spec-review.md`.
- Measurements: `docs/experiment-log.md` (every run with its scorecard),
  `docs/superpowers/plans/2026-10-31-v2-gaps.md` (the twelve gaps with evidence),
  `bench/evals/2026-10-31-planner-v1/` (results, metrics, logs, replies, long-term store).
- License: `[K: add LICENSE to the repository first.]`

## 12. Checklist before the Kaggle Submit button

- [ ] Public score of 57844971 read and written into sections 3 and 7.
- [ ] `[K]` decisions above resolved: submission to document, title, license, model
      license, notebook visibility. Repository URL done (pushed 2026-11-01).
- [ ] Prior-work entries verified at the primary source or cut.
- [ ] Body under 1,500 words (count the pasted text, not this file).
- [ ] Cover image and figures 2 to 5 uploaded to the Media Gallery.
- [ ] Public notebook attached in Project Links; repository link added.
- [ ] Track set to ARC-AGI-3; submission ID 57844971 at hand for the form.
- [ ] Prose pass: no em-dashes, no stock vocabulary, contractions where natural,
      ragged paragraph lengths, at most one "not X but Y" in the whole text.
- [ ] Saved, then Submit (a draft left unsubmitted is not judged).
