# Can a 27B open-weight model write the rules of an ARC-AGI-3 game from what an agent sees?

DRAFT v0.1, 2026-10-20. Structure and data by the harness; prose to be
rewritten by Kameron before submission. Numbers come from
`docs/experiment-log.md`; do not edit them here without updating the log.
Required sections per the ARC Prize 2026 Paper Prize page: Abstract, Intro,
Prior work, Approach (algorithm level), Results, Conclusion. Results must
cite the Kaggle leaderboard and public eval and must not report training-set
performance.

## Abstract

[150 words max. Claim: an open-weight 27B model running offline on the
competition hardware can, from observed transitions alone, write an
executable next-frame model that exactly predicts 100% of held-out
transitions on one public ARC-AGI-3 game (ft09), partially on a second
(vc33, 0.53), and fails on a third (ls20, 0.00). The repair loop only works
when the feedback shows the model what its own code did, not just where it
was wrong. State the leaderboard link once the agent submission exists.]

## 1. Introduction

- ARC-AGI-3 games are interactive: the agent sees 64x64 frames and chooses
  actions; nothing tells it the rules.
- Question: can a rule *program* be inferred from a short random probe, so
  that an agent can plan against a simulator instead of the live game?
- Why a program and not a learned predictor: exact, inspectable, cheap to
  run, and its failures are legible (a wrong rule, not a wrong weight).
- What this paper contributes: a harness (trace, prompt, sandboxed verify,
  repair), five diagnostic runs on the competition hardware, and the finding
  that repair feedback must include the candidate's own prediction.

## 2. Prior work

[Every entry below is a candidate to verify at the primary source before a
single sentence about it is written. No characterizations until then.]

- Program synthesis for ARC-AGI-1/2 with LLMs (search over DSL programs;
  test-time training). [verify: which papers, what exactly they report]
- ARC-AGI-3 milestone 1 and 2 winners' write-ups (the Tufa Labs notebook
  this harness reuses the wheelhouse and model snapshot from). [verify: what
  their agent does; whether any synthesizes rules]
- World-model learning from pixels for games (model-based RL). [verify]
- LLM self-repair / execution-feedback loops for code generation. [verify:
  which report that execution feedback helps, and under what conditions]

## 3. Approach

Algorithm, one game at a time:

1. **Trace.** Play `steps=120` actions from RESET with a probe policy:
   with probability 0.5 a uniform random action, else cycle through the
   available actions in order. A click action expands to one target per
   connected single-color component (up to 8), chosen nearest the
   centroid. Record every transition `(grid_before, action, grid_after)`.
   Split by time: first 70% train, remaining 30% held-out.
2. **Prompt.** System message states the interface and the task: write a
   pure Python module `step(grid, action, x, y) -> grid`, standard library
   only, that models objects and rules rather than a lookup table. User
   message shows the starting grid in full and the first 40 training
   transitions as cell diffs (a full grid when a transition changes more
   than 60 cells, at most twice). Grids render as run-length rows of hex
   digits; the format is invertible.
3. **Verify.** Run the candidate in `python -I` in a fresh directory with
   no network and a 20 s wall limit for all transitions. Score exact-match
   rate per split and cell accuracy. For each training mismatch record the
   true change, the candidate's own change, and the wrong cells.
4. **Repair** (up to two more rounds). Append the prior code as the
   assistant turn and a user turn with: how many training transitions it
   reproduced, up to 10 mismatches (true change, your change, wrong cells),
   and an instruction to diagnose in at most five lines before the code.
   Stop when held-out exact reaches 0.95, when the reply has no code (after
   one re-sample with the next seed), or when the new module's AST equals
   the previous one.
5. **Select.** Keep the round with the best training fit (exact rate, then
   cell accuracy, earliest on a tie). Selection never reads held-out.

Model: Qwen3.6-27B-FP8 via vLLM, thinking disabled at the chat template,
temperature 0.2, max 16000 completion tokens, per-request seed
`trace_seed*1000 + round*10 + attempt`. Everything runs inside a Kaggle
notebook with no internet on one RTX PRO 6000.

## 4. Results

[Report held-out numbers only. Leaderboard: the probe-policy agent
(`hgenix20/arc3-agent-probe-baseline`, submission 57437483) scored 0.11 on
the public leaderboard on 2026-10-20; the model-free explorer scores 0.36
offline on the 25 public games (not yet submitted); the LLM-layer kernel
`hgenix20/arc3-duck-agent` is pending.]

Held-out exact-match rate of the selected candidate, 36 held-out
transitions per game, identity baseline in parentheses:

| game | v6 | v7 | v8 (seeded) | v9 (seeded, train-fit final) |
|---|---|---|---|---|
| ft09 (0.89) | 0.86 | 0.69 | **1.00** | 0.89 |
| vc33 (0.00) | 0.42 | 0.50 | 0.53 | 0.53 |
| ls20 (0.00) | 0.22 | 0.00 | 0.00 | **0.25** |

v8 and v9 ran the same code, prompts and per-request seeds; every round-1
reply differed byte for byte, and ft09's exact solution did not recur. The
server (vLLM 0.19, FP8) is not reproducible across runs, so each column is
one sample and the differences between columns are within sampling noise.

Three findings, each with its evidence in the log:

1. **Thinking must be off for this budget.** With the reasoning channel on
   and 6000 tokens, all 75 replies in run v4 spent the whole budget thinking
   and returned no code. With it off, every reply in v6–v8 stopped normally.
2. **Repair needs the candidate's own prediction.** With feedback limited to
   "expected X, got Y", the model returned its module with comment-only
   edits in every repair round (v6, 3 of 3 games). Adding a score line and
   the candidate's own change per mismatch, and asking for a diagnosis
   first, produced logic changes in every repair round (v7, v8) and the
   ft09 solution.
3. **Repair is not monotone.** vc33 fell from 0.53 to 0.39 over two repairs
   in v8; selecting the candidate by training fit keeps the 0.53 round.
   Run-to-run variance without a per-request seed was as large as the
   repair gains (ls20 round 1: 0.22 vs 0.00 on a byte-identical prompt).

Costs: 9 model calls per 3-game run, 1.5–4 minutes per call at 37 tokens/s;
server startup 9–31 minutes depending on disk speed.

## 5. Conclusion

[What holds: one game solved exactly from 120 random steps and three model
calls; the harness is deterministic end to end and reproduces Kaggle
numbers locally. What does not: two of three games stay unsolved; the probe
policy rarely leaves level 0, so the rules of later levels are never seen.
Next: feed the synthesized `step` to a planner and submit the agent; a
different feedback form for the unsolved games (fewer, larger mismatches,
or a full predicted frame).]

## Reproducibility

Code: `C:\arc-agent` at commit 4e9f424 (to be published under MIT-0 before
private scores). Kaggle kernel `hgenix20/arc3-rule-synthesis-probe`
versions 4–9; outputs (results.json, replies, synthesized code) archived
per run. Local verification: `python -m arcagent.synth.run --fake` for the
pipeline, and the v6 ls20 round-1 candidate reproduces its Kaggle score
(0.32 train / 0.22 held-out) on a locally collected trace.
