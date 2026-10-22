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
| v9 | 2026-10-20 04:20 | `final` = round with the best training fit (exact, then cell accuracy, earliest on tie); same seeds as v8 | best ls20 so far (0.25, repair improved it twice); vc33 regressed again and the policy kept the 0.53 round; ft09 did not reproduce v8's 1.00 (0.89 = identity); every round-1 reply differed from v8's despite identical prompts and seeds |

## Held-out exact match per round

Identity baseline (predict "nothing changes"): ls20 0.00, ft09 0.89, vc33 0.00.

| run | ls20 r1 | r2 | r3 | ft09 r1 | r2 | r3 | vc33 r1 | r2 | r3 |
|---|---|---|---|---|---|---|---|---|---|
| v4 (thinking on) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| v6 | 0.22 | 0.22 | 0.22 | 0.69 | 0.86 | 0.86 | 0.42 | 0.42 | 0.42 |
| v7 | 0.00 | 0.03 | 0.03* | 0.69 | 0.69 | 0.00† | 0.50 | 0.50 | 0.53 |
| v8 | 0.00 | 0.00 | 0.00 | 0.92‡ | **1.00** | – | 0.53 | 0.47 | 0.39 |
| v9 | 0.17 | 0.19 | **0.25** | 0.89 | 0.89 | 0.89 | 0.53 | 0.53 | 0.39 |

Selected `final` under the train-fit policy (v9): ls20 r3, ft09 r3, vc33 r2.
On v8's data the same policy would select ft09 r2 (1.00), vc33 r1 (0.53), ls20 r2.

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
| v9 | logic changed, better twice | logic changed, flat | logic changed, worse in r3 | same |

Repair deltas, held-out exact (r1 to best later round): v6 +0.00 / +0.17 / +0.00;
v7 +0.03 / +0.00 / +0.03; v8 +0.00 / +0.08 / −0.06 (vc33 fell each round);
v9 +0.08 / +0.00 / +0.00 (vc33 r3 fell to 0.39 again; `final` kept r2).

ls20 in v9 is the clearest case of train/held-out divergence: train exact
0.81 → 0.71 → 0.89 while held-out went 0.17 → 0.19 → 0.25. The held-out
split is the last 30% of the trace in time; rules that fit the first 84
transitions do not carry to the last 36.

## Reply accounting

| run | calls | replies at the token cap | no-fence replies | retries | median seconds per call |
|---|---|---|---|---|---|
| v4 | 75 | 75 (6000) | 75 | – | 157 |
| v6 | 9 | 0 | 0 | – | 82 |
| v7 | 9 | 1 (16000) | 1 | – | 149 |
| v8 | 9 (+1 retry) | 1 (16000, discarded) | 1 | 1 | 233 |
| v9 | 9 | 0 | 0 | 0 | 198 |

## Infrastructure timeline (seconds from notebook start)

| run | wheelhouse install | server ready | run total |
|---|---|---|---|
| v4 | 85 | 535 | ~12,490 (25 games) |
| v5 | 298 | killed at 1804 (1500 s deadline) | – |
| v6 | 295 | 1855 | 2835 |
| v7 | – | – | ~2700 |
| v8 | 297 | 1837 | ~3900 |
| v9 | 299 | 1779 | ~3480 |

Weight loading from the Kaggle input disk varied 2.4x between runs; the
server deadline is 3600 s since v6.

## Run-to-run variance

v6 and v7 sent a byte-identical round-1 prompt for ls20 and received
different replies (11,239 vs 4,245 completion tokens; 0.22 vs 0.00 held-out).
vLLM was started with a global seed but requests carried none. From v8 every
request carries `seed = trace_seed*1000 + round*10 + attempt` (the test on
`request_body` proves it is sent; vLLM 0.19 does not log request params by
default, so the server side is unconfirmed).

**v8 vs v9, same code, same prompts, same seeds: all three round-1 replies
differ byte for byte**, and ft09 went from solved (1.00) to the identity
baseline (0.89). A per-request seed does not make this server reproducible
across runs. The likely cause is kernel-level nondeterminism in the FP8
attention and prefix-caching path (logits differ in the last bits, and once
one sampled token differs the rest of the reply diverges); this is not
proven here. Consequence for every comparison in this log: a single run per
condition cannot separate a change in the harness from sampling noise. The
honest unit is several runs per condition, or temperature 0 with the
caveat that greedy decoding is what looped in v7.

## Agent submissions

The competition entry is a separate notebook, `hgenix20/arc3-agent-probe-baseline`
(see `agent-submission.md`). It plays every game through the public action
interface; the competition rerun scores it from the gateway's record.

| version | pushed | agent | offline (25 public games) | leaderboard |
|---|---|---|---|---|
| v1 | 2026-10-20 12:18 | probe policy, 300 actions per game, CPU | 2 levels (r11l, vc33), scorecard 0.19 | **0.11** public (submission 57437483, scored 2026-10-20) |
| explore (not pushed) | 2026-10-20 | `ExplorerPolicy`: state graph, UCB over action kinds, frontier routing | 6 levels at 300 actions, 7 at 1000; scorecard 0.36 either way | not submitted (one slot per day; the duck kernel takes the next) |
| duck v1 | 2026-10-20 13:52 | Franzen's Milestone 2 notebook unchanged except a 3-game interactive smoke run (`hgenix20/arc3-duck-agent`) | smoke (ls20, ft09, vc33; 870 s per game): mean 42.75; ft09 6/6 levels at 87 actions (score 100), vc33 3/7 (21.4), ls20 2/7 (6.8); server ready at 12 min, 27 min total | submitted 2026-10-21 00:04 UTC (submission 57457365): **0.00**, "Succeeded" within an hour of a 9-hour budget. The gateway's 15-minute idle rule ended the run before the model server answered; see "Why duck v1 scored 0.00" below |
| duck v2 | 2026-10-21 01:21 | v1 plus `scripts/duck_keepalive.py` (the warm-up game repeats its RESET every 5 min until the first model action) and a 60-minute first-request grace (was 15) | smoke (same 3 games, 870 s each): mean 24.21; ft09 4/6 (47.6), vc33 3/7 (21.4), ls20 1/7 (3.6); keep-alive RESET at 01:34:05, server ready 01:38:21, first model action 01:38:38, so the second RESET was never needed; 27 min total | submitted 2026-10-22 13:56 UTC (submission 57527194): **27.22 public LB**, complete by 23:08 UTC; the keep-alive fixed the zero, and 27.22 is within the ±14 run-to-run band around Franzen's published 34.3 |

Duck smoke-run detail (agent/human actions per level): ft09 4/43, 7/12,
14/23, 22/28, 24/65, 16/37 (every level under the human count); vc33 10/7,
12/18, 25/44, 19/61 then out of time; ls20 32/22, 145/123 then out of time.
The harness reported 251 generated tokens/s over the job and 220k tokens for
330 actions, about 670 tokens per action.

### Why duck v1 scored 0.00 (2026-10-21)

The rerun was marked "Succeeded" with 0.00 within an hour of submission; a
healthy rerun of this notebook uses most of its 9 hours, and copies of the
same notebook by other users scored 29 to 31 on the same day (competition
Code tab, sorted by last run). Kaggle shows no rerun log, so the cause comes
from the gateway's rules and our own startup timeline.

The rule: a scorecard with no move for 15 minutes is closed and the run ends
with the score so far. The toolkit's `ScorecardManager` has `STALE_MINUTES`
= 15 (arc_agi 0.9.8, `scorecard.py`), the docs say "Scorecards auto close
after 15 minutes", and Jeroen Cottaar (Duck author, first on the
leaderboard) wrote in competition discussion 750173: a run with no move for
15 minutes ends with the score so far, so start a thread that issues a RESET
to one game every 10 minutes until startup. Two other posters in that thread
tie their 0.00 runs to slow dataset and model loading in the rerun (2 to 3
times slower than an interactive run, one run ending after about 30
minutes), and a third thread (750981) reports the same notebook version
dropping from 32.76 to 24.03 between Oct 16 and Oct 19, every run completing
9 hours with a keep-alive move in place.

Our timeline fits. Franzen's notebook releases the benchmark at 12 minutes
(`SERVER_STARTUP_TIMEOUT`) and the harness spends one RESET on one game at
that moment (`ARC3_WARMUP_ACTION_GAMES=1`). The next move is the first model
action, which needs the server up plus one full reply. In our four
interactive runs the server became ready 12.8, 12.5, 17.4 and 15.6 minutes
after notebook start (serve.log "ready to roll" against the benchmark start,
which is 12 minutes after notebook start), leaving five to twelve minutes
for the first reply before the idle rule, and a rerun host that loads the
185 GB of weights twice as slowly has none. The first analyzer
request also gives up after a 900 s grace plus ten failures, which would end
every game within about 35 minutes on a slow host even without the idle
rule.

The fix (duck v2): `scripts/duck_keepalive.py`, copied verbatim into a new
notebook cell by the generator, wraps the harness's game session so the
warm-up game repeats its RESET every 5 minutes until any game executes a
model-chosen action, then plays normally (6 unit tests against a stub
session; the real class cannot be imported off Kaggle). Cost: a few RESET
actions on one game's first level. The first-request grace goes from 900 to
3600 s. Both are in every build, not a variant, because they only act while
the server is loading. The lesson for the paper: the public notebook as
published is one slow disk away from a zero, and the only way to know is the
leaderboard. Duck v2 scored 27.22 on the public leaderboard (submission
57527194, 2026-10-22), up from 0.00 with no change to the agent itself.

An independent noise study by Francise Liang (discussion 750885, 2026-10-19)
matches ours: the same code twice gave means 47.30 and 48.92 at two passes
per game with a per-game difference sd of 24 points, 14 streams instead of
10 gave 29.90 against 47.30, and 2 of their 12 submissions came back
"Succeeded" at 0.00 with the unmodified notebook among them.

### Duck evaluation run v1 (baseline, 2026-10-20 14:35–16:37 UTC)

`hgenix20/arc3-duck-eval` version 1: the unchanged harness on all 25 public
games, one pass, 7254 s per game, 25 games admitted with 10 active streams
(the competition gate). Mean score **41.47**, median 35.71, 6 games won
(cd82, ft09, lp85, m0r0, r11l, sb26), 4592 actions, 4.3M generated tokens at
588 tokens/s job-wide. Every unfinished game stopped on the time cap, so the
score is still rising when the clock runs out; the competition rerun gives
each game about 0.8 stream-hours against 2 here, which is consistent with
the published 34.3 on the private games. Per-game and per-level records:
`duck-runs/eval-v1-baseline-summary.txt`, `duck-runs/eval-v1-baseline-levels.txt`.
Games stuck at level 0 or 1: sc25 (146 actions on level 1, human 36), sk48
(80 actions), bp35, g50t, tn36, wa30, sp80. Efficiency on completed levels is
mostly at or under the human count; the losses are levels never reached.

### Duck experiments (one variable per run, judged on the 25-game evaluation)

| variant | kernel | change | mean score | notes |
|---|---|---|---|---|
| baseline | arc3-duck-eval v1 | none | 41.47 | control, 2026-10-20 |
| streams16 | arc3-duck-eval-s16 v1 | 16 active streams and server requests (CUDA graphs to batch 16) instead of 10 | **30.61** (2 won, 3119 actions, 2.4M tokens, 330 tokens/s) | refuted, and the cause is mechanical, see below |
| baseline repeat | arc3-duck-eval v2 | none | **43.69** (6 won, 5117 actions, 4.1M tokens, 555 tokens/s) | 19:44–21:47 UTC; the noise measurement, see "Noise band" below |
| spec4 | arc3-duck-eval-spec4 v1 | speculative decoding drafts 4 tokens per step instead of 3 (`SPEC_STEPS`) | running, pushed 2026-10-22 23:41 UTC | judged on the mechanism: generated tokens/s and mean accept length against the two baselines (555 and 588 tokens/s, accept 2.67); the score alone can't resolve under ±14 |

**Noise band (two identical baseline runs, 2026-10-20).** The means differ
by 2.2 points, the games do not. The root mean square per-game difference is
36 points; 7 of 25 games moved by more than 30 (ar25 58→100, m0r0 100→29,
r11l 100→14, s5i5 58→3, sc25 0→96, tn36 4→58, vc33 21→54). Only four games
were won in both runs (cd82, ft09, lp85, sb26). From the paired differences:
per-game standard deviation of a single run about 25 points, standard error
of the 25-game mean about 5, and the 95% band for the difference between two
single runs about **±14 points**. A one-pass evaluation therefore cannot
confirm a change worth less than about 14 points on the mean; the
competition rerun, one pass over the private games, carries the same
uncertainty on the leaderboard figure. Script: `scripts/duck_noise.py`
(paths are the two summary files). Records: `duck-runs/eval-v2-baseline-summary.txt`,
`duck-runs/eval-v2-baseline-levels.txt`.

Why the loss pool is stable: the harness's priority gate. With 10 slots for
25 games, Franzen's patch ranks games by the level reached, the value of the
levels remaining and the actions and tokens spent per level (env
`ARC3_PRIORITY_*`, tuned by replaying his own runs) and parks the expensive
ones. In the repeat run sk48 made no model call for 81 minutes between two
analysis steps (20:08 to 21:29 UTC) while holding no slot, then ran out of
clock on level 2; its transcript ends mid-derivation of a world model
(validated 62 of 63 transitions) with a server read timeout. The 19 games
that end as "gave_up" are the time cap, not a decision by the model. A change
aimed at those games has to buy them stream time, which the gate takes from
games that are scoring; the gate's parameters are the lever, not the prompt.

Consequences for the iteration plan: (1) a variant is judged on its
mechanism (tokens generated, levels reached, prefill volume) as much as on
its score; (2) a score comparison needs several passes or a paired design
(same games, same seeds where the server allows); (3) the games that scored
under 6 in all three runs (bp35, lf52, sk48, sp80, wa30, and su15 at 6–13)
are the stable loss pool, about a quarter of the possible mean, and the
only place where a change could show up above the noise in one run.

**streams16 (17:11–19:14 UTC).** Worse by 11 to 13 points against the two
baselines, at the edge of the noise band, so the score alone does not settle
it; the server log does. The Mamba state cache keeps 60 slots at 5
per request, so the server capped running requests at 12 regardless of the
16 asked for, and the extra streams overran the prefix cache:

| run | prefill batches | new prefill tokens | cached prefill tokens | cache hit |
|---|---|---|---|---|
| baseline (10 streams) | 2931 | 10.4M | 157.9M | 94% |
| streams16 | 6672 | 47.5M | 47.1M | 50% |

Each conversation turn re-prefilled from a cold start about half the time,
the GPU spent its hours on prefill instead of decode (generated tokens fell
from 4.3M to 2.4M in the same wall clock), the request queue sat at 4 on
average, and in the final minutes the harness logged analyzer request
timeouts on most games. Decode itself was unaffected (mean accept length
2.64 vs 2.67, similar tokens per decode step). Per-game records:
`duck-runs/eval-s16-summary.txt`, `duck-runs/eval-s16-levels.txt`. Lesson: with this model the stream count
is bounded by the Mamba checkpoint cache, not by batch throughput; raising
streams needs `--max-mamba-cache-size` raised in step, and that memory comes
out of the KV pool (full token usage already reached 0.95 in this run).

One run per variant cannot separate a few points from sampling noise (the
synthesis runs showed this, and the two baseline runs measured it: treat
differences under about 14 points as unresolved until repeated). A 25-game, one-pass
evaluation variant (`hgenix20/arc3-duck-eval`, 7254 s per game, about 2.3
GPU hours) is the per-game control for later changes.

Kameron's instruction on 2026-10-20 was to iterate until the score reaches 80.
The public leaderboard's top entry stands at 68.91 (Tufa Labs), the best public
notebook at 34.3; every scoring entry is an open-weight LLM in a Python tool
loop. The plan of record is therefore: the published harness as the LLM layer
(credited, Apache-2.0), our own changes on top as named-cell patches, one
leaderboard reading per day, the model-free explorer kept as the fallback
agent and as the fast layer of the stack.

## Open items

- Paper eligibility needs a linked Kaggle *code submission* on the ARC-AGI-3
  track. The probe-policy agent above is that submission; the agent that
  plans against synthesized models is the next step.
- The paper may not report training-set performance; this log's train
  columns are for our own diagnosis only.
- ls20 reached 0.25 held-out in v9 (0.89 train) after 0.00 in v7 and v8 on
  the same prompt: the gap between runs is sampling, not the harness. vc33's
  round-3 repair regressed to 0.39 in both v8 and v9 (diagnoses of 6–9k
  characters, the rewrite overreaches); the train-fit `final` absorbs the
  damage but does not prevent it. Both remain candidates for a different
  feedback form (fewer, larger mismatches; or the predicted grid in full).
- Reproducibility: per-request seeds did not reproduce replies across runs
  (see above). Before any claim that a harness change helped, run each
  condition several times; v6–v9 are one run each.
- All code under this repo is to be released under MIT-0/CC0 before private
  scores, per the competition rules.
