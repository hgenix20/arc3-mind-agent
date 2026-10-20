# Experiment log: rule synthesis on ARC-AGI-3

Running record of every Kaggle run of `hgenix20/arc3-rule-synthesis-probe`,
what changed between runs, and the numbers. Source of truth for the paper
track write-up. Dates are UTC. Model: Qwen3.6-27B-FP8 behind vLLM 0.19 on a
Kaggle RTX PRO 6000 (97 GB), no internet, image pinned to Python 3.12.
Harness: `arcagent/synth/` (see `synth-experiment.md` for the contract).

Protocol in every run unless noted: `--steps 120 --seed 0 --rounds 3
--max-transitions 40`; a trace of 120 transitions per game, split 84 train /
36 held-out by time; round 1 synthesizes, later rounds repair; repair stops
at 0.95 held-out exact. "Exact" is exact-match rate on next-frame prediction.
Held-out transitions never enter any prompt.

## Run history

| kernel | pushed | change under test | outcome |
|---|---|---|---|
| v1–v3 | 2026-10-19 | wheelhouse install, Python 3.12 image, wheel tree under /tmp | died in vLLM startup; fixed in commits 09a1666, 1468425 |
| v4 | 2026-10-19 | first full run, 25 games, thinking on, max_tokens 6000 | 0.00 on every game: all 75 replies used exactly 6000 tokens, empty `content` (thinking consumed the budget); 50 of 75 calls were repeat repairs of the stub |
| v5 | 2026-10-19 16:18 local | thinking off (`enable_thinking: false`), max_tokens 16000, record `finish_reason`/`reasoning_chars`, save replies, stop after a no-code round | ERROR: weight load 1284 s vs ~400 s in v4; the notebook's 1500 s server deadline fired with warmup finished |
| v6 | 2026-10-19 21:49 | server deadline 3600 s; 3 games | **first working run**: every reply `finish_reason=stop`, real code; repair rounds 2 and 3 returned comment-only edits on all three games |
| v7 | 2026-10-20 02:08 | repair turn: score line, `your change` per mismatch, diagnose-first instruction; AST-identical repair ends the game | repair rounds now change logic with a written diagnosis; one reply looped to the 16000 cap with no fence; identical round-1 prompt vs v6 gave a different reply (no per-request seed) |
| v8 | 2026-10-20 02:58 | per-request `seed`; one re-sample on a no-fence reply; `final` skips no-code rounds | ft09 solved (1.00 held-out); retry fired once and recovered; vc33 regressed under repair |
| (code) | 2026-10-20 | `final` = round with the best training fit (exact, then cell accuracy, earliest on tie) | not yet run on Kaggle; on v8's data it selects ft09 r2, vc33 r1, ls20 r2 |

## Held-out exact match per round

Identity baseline (predict "nothing changes"): ls20 0.00, ft09 0.89, vc33 0.00.

| run | ls20 r1 | r2 | r3 | ft09 r1 | r2 | r3 | vc33 r1 | r2 | r3 |
|---|---|---|---|---|---|---|---|---|---|
| v4 (thinking on) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| v6 | 0.22 | 0.22 | 0.22 | 0.69 | 0.86 | 0.86 | 0.42 | 0.42 | 0.42 |
| v7 | 0.00 | 0.03 | 0.03* | 0.69 | 0.69 | 0.00† | 0.50 | 0.50 | 0.53 |
| v8 | 0.00 | 0.00 | 0.00 | 0.92‡ | **1.00** | – | 0.53 | 0.47 | 0.39 |

\* round 3 was AST-identical to round 2 (`unchanged`), the game stopped there.
† no code fence after 16000 tokens (two formulas alternating); scored as a load failure.
‡ first attempt had no fence (43k chars of enumeration); the re-sample with the next seed produced this candidate.

Held-out cell accuracy, final candidate: v6 ls20 0.9892 / ft09 0.9987 / vc33 0.9859;
v8 ls20 0.9838 / ft09 1.0000 / vc33 0.9855 (r1).

## Did the repair round change anything?

| run | ls20 r2 | ft09 r2 | vc33 r2 | how measured |
|---|---|---|---|---|
| v6 | comment only | rewrite | comment only | `cmp` of code files; prompt diff r2 vs r3 differed only inside the prior code |
| v7 | logic changed | logic changed | logic changed | `unchanged` flag (AST compare) false on all three |
| v8 | logic changed | logic changed, solved | logic changed, worse | same |

Repair deltas, held-out exact (r1 to best later round): v6 +0.00 / +0.17 / +0.00;
v7 +0.03 / +0.00 / +0.03; v8 +0.00 / +0.08 / −0.06 (vc33 fell each round).

## Reply accounting

| run | calls | replies at the token cap | no-fence replies | retries | median seconds per call |
|---|---|---|---|---|---|
| v4 | 75 | 75 (6000) | 75 | – | 157 |
| v6 | 9 | 0 | 0 | – | 82 |
| v7 | 9 | 1 (16000) | 1 | – | 149 |
| v8 | 9 (+1 retry) | 1 (16000, discarded) | 1 | 1 | 233 |

## Infrastructure timeline (seconds from notebook start)

| run | wheelhouse install | server ready | run total |
|---|---|---|---|
| v4 | 85 | 535 | ~12,490 (25 games) |
| v5 | 298 | killed at 1804 (1500 s deadline) | – |
| v6 | 295 | 1855 | 2835 |
| v7 | – | – | ~2700 |
| v8 | 297 | 1837 | ~3900 |

Weight loading from the Kaggle input disk varied 2.4x between runs; the
server deadline is 3600 s since v6.

## Run-to-run variance

v6 and v7 sent a byte-identical round-1 prompt for ls20 and received
different replies (11,239 vs 4,245 completion tokens; 0.22 vs 0.00 held-out).
vLLM was started with a global seed but requests carried none. From v8 every
request carries `seed = trace_seed*1000 + round*10 + attempt`. Reproducibility
is testable only from v8 onward (v8 vs v9 should match byte for byte).

## Open items

- Paper eligibility needs a linked Kaggle *code submission* on the ARC-AGI-3
  track. The probe is a private notebook, not a submission; an agent that
  uses the synthesized models has to exist and be submitted first.
- The paper may not report training-set performance; this log's train
  columns are for our own diagnosis only.
- ls20 is unsolved at 0.00 in every run; the model never finds the moving
  5x5 pattern. vc33's repair regresses: the diagnoses run to 6–9k characters
  and the rewrite overreaches. Both are candidates for a different feedback
  form (fewer, larger mismatches; or showing the predicted grid in full).
- All code under this repo is to be released under MIT-0/CC0 before private
  scores, per the competition rules.
