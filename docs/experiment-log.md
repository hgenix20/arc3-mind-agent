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

### Mind entry (stack + Genix Mind + offline LLM), from 2026-10-23

The Franzen-derived duck kernel is out of the entry. The agent is our own:
the mind's loops in `arcagent/mind/`, the planner in `arcagent/plan/planner.py`,
and Qwen3.6-27B-FP8 served inside the notebook (kernel `hgenix20/arc3-mind-agent`).

| date | version | run | levels | score est. | notes |
|---|---|---|---|---|---|
| 2026-10-23 | v1 | public eval, 25 games, 8.5 h budget | 13 | 0.32 | 263 LLM calls, 184 programs admitted, 113 dropped, prediction accuracy 0.97; 470,704 actions |
| 2026-10-23 | v1 | submission 57565016 | pending | | Kameron's go; submitted from the v1 commit |

v1 sits at the explorer's level (offline EXPLORER 0.36 on the same 25 games),
so the model-written programs aren't completing levels the explorer can't yet.
Every level it finished took far more actions than the human counts. v1 predates
the final fix wave (fallback parquet when vLLM or the run fails, scaled eval
budget, bounded sleep, sandbox startup). Known open cost: duplicate programs
are admitted under new ids, and VOI scores each copy on every step.
Logs: `bench/evals/2026-10-23-full-v1/`.

Planner ceiling, 2026-10-23 (`scripts/plan_ceiling.py --level-seconds 60
--max-nodes 200000`, true rules as the world model): 13 of 183 levels, mean
score 2.91 over the 25 public games. Every search that failed stopped on the
time limit after expanding 21 to 4,026 nodes, because cloning the real game
engine is slow, and sc25 exposed no usable action. Blind search over even a
perfect model doesn't finish levels; goal-directed play is what's missing.
Results: `bench/ceiling/`.

### Fluid agent, slice 1 (no LLM), from 2026-10-23

Spec `docs/superpowers/specs/2026-10-23-fluid-agent-design.md`, plan
`docs/superpowers/plans/2026-10-23-fluid-agent-slice-1.md`, branch
`fluid-slice-1`. Config `FLUID`: objects and terrain, noise mask, fact
dictionary with a table of contents, predictor, heuristic goal guesses
(reach, clear), short search over predictions, targeted tests, vetoes.

| date | commit | run | levels | score est. | notes |
|---|---|---|---|---|---|
| 2026-10-23 | a4edb72 | offline, 25 games, 120 s each, CPU | 3 (lf52 1, lp85 2) | 0.13 | 6,273 actions; prediction accuracy 0.91; 1,386 no-op actions, 52 game overs |

It finishes fewer levels than v1 (13, 0.32), which had an 8.5 h budget. Most
games show high prediction accuracy (0.67 to 1.0), so the agent learns how
moves change objects; what it lacks is the goal. On ls20, traced move by
move, stepping into the target needs a condition met first, and touching the
switch changes the switch's own signature, so `reach` guesses fail and stall.
The click games (ft09, vc33) need goals like "make this area match that one",
which slice 1's heuristics don't propose. That is slice 2's job: the LLM reads
the screen and proposes goals and staged plans. Only one run so far, so no
paired comparison. Logs: `bench/offline/fluid-s1-run1/`.

| date | commit | run | levels | score est. | notes |
|---|---|---|---|---|---|
| 2026-10-23 | 375426b | Kaggle eval v2 of `arc3-mind-agent`, FLUID-LLM, 25 games, scaled budget | 8 | 0.05 | images accepted by vLLM 0.19 (`limit_mm_per_prompt` image 2); 200 turns, mean 7.6 s, max 16.9 s; 4 exec_failed, 45 stale; first levels on ft09, m0r0, r11l, vc33, sp80, tn36 that slice 1 never reached |

v2 predates the final-review fix wave (turns applied to a moved-on state, lost
level-start wakes, inert match goals). Levels came at 413 to 3,426 actions
each (lf52 892, against 12 offline without the LLM), so the squared action
ratio wiped out the score. Kernel version 3 carries the fixes.
Logs: `bench/evals/2026-10-23-fluid-llm-v2/`.

| date | commit | run | levels | score est. | notes |
|---|---|---|---|---|---|
| 2026-10-24 | 3f2b086 | Kaggle eval v3 of `arc3-mind-agent`, FLUID-LLM after the fix wave, no slice 3, 25 games, scaled budget | 4 | 0.41 | scorecard 0.4065, 6,246 s, 0 errors; levels on lf52 (1.82), r11l (4.76), tn36 (3.57), lp85 (0.01); prediction hit rate 0.92; 593 tests per level overall (cd82 1,958, dc22 1,988) |

v3 finished half of v2's levels and scored eight times as much, because the
levels it did finish cost close to the human count. It's the first version
above v1's 0.32. The test flood is still there: on games with no level,
almost every move is a test. Kernel version 4 is main at a3c123b (slice 3:
sleep, the four modules, novelty search, and the 200-test cap per attempt).
Logs: `bench/evals/2026-10-23-fluid-llm-v3/`.

| date | commit | run | levels | score est. | notes |
|---|---|---|---|---|---|
| 2026-10-24 | a3c123b | Kaggle eval v4 of `arc3-mind-agent`, FLUID-LLM full build (slices 1 to 3), 25 games, scaled budget | 5 | 0.38 | scorecard 0.3774, 6,247 s, 0 errors; levels on r11l (4.76), lp85 (2.78), lf52 (1.82), tn36 (0.07), ft09 (0.005); 30 sleeps; hit rate 0.89; rule reuse 0.02; 593 tests per level |
| 2026-10-25 | f2288a1 | Kaggle eval v5 of `arc3-mind-agent`, FLUID-LLM after the forced-move merge, 25 games, scaled budget | 14 | 0.43 | scorecard 0.4464, 6,246 s, 0 errors; prediction accuracy 0.92; levels on ar25 x2 (211, 399), lp85 x2 (7, 1,890), cd82 (1,065), dc22 (1,035), lf52 (149), ls20 (44), m0r0 (95), r11l (29), sp80 (1,189), su15 (628), tn36 (11), vc33 (120); submitted 2026-10-25 (submission 57637984, Kameron's go) |
| 2026-10-26 | ea4e8b4 (llm-drive, before the review fixes) | Kaggle eval v7 of `arc3-mind-agent`, DRIVE (the LLM drives, code plays plans out in its model), 25 games, scaled budget | 8 | 0.46 | scorecard 0.4555, 6,249 s, 0 errors; levels on r11l (4.76), tn36 (3.57), lp85 (2.78), ka59 (0.17), ar25 (0.08), m0r0, sp80, vc33; 1,753 turns; moves 18,285 fallback, 7,003 sim, 422 experiment; 0 claims seeded in every game; 11 tool fixes tried, 1 adopted (sp80) |
| 2026-10-26 | 2f90da9 | Kaggle eval v8, DRIVE with the review fixes and lenient reply parsing, 25 games, scaled budget | 8 | 0.31 | scorecard 0.3060, 6,247 s, 0 errors; levels on tn36 (3.57), lp85 (2.78), r11l (0.85), su15 (0.26), ar25 (0.10), cn04, m0r0, vc33; 1,500 turns, mean 29.6 s; 550 claims seeded; moves 14,947 fallback, 3,271 sim, 271 experiment |
| 2026-10-26 | 6386c35 | Kaggle eval v9, DRIVE with JSON-only replies, a 4,096-token cap and more reply spellings, 25 games, scaled budget | 8 | 0.52 | scorecard 0.5194, 6,248 s, 0 errors; levels on tn36 (3.57), lp85 (2.78), m0r0 (2.12), lf52 (1.82), ls20 (1.59), r11l (0.73), ar25 (0.37), vc33; 1,488 turns, mean 27.9 s; 642 claims seeded; moves 13,265 fallback, 6,542 sim, 210 experiment |
| 2026-10-26 | fea771b | Kaggle eval v10, DRIVE with wakes only on level start and game over while a modeled move remains, run-on plans and claims replaced by observation, 25 games, scaled budget | 11 | 0.53 | scorecard 0.5329, 6,247 s, 0 errors; levels on vc33 (2, 4.03), tn36 (3.57), lp85 (2, 2.78), m0r0 (1.01), ar25 (0.53), ls20 (0.50), cd82 (0.33), r11l (0.12), lf52 (0.03); 1,626 turns, mean 26.0 s; 622 claims seeded; moves 17,326 fallback, 5,333 sim, 210 experiment |

v4 and v3 are one run each, and 0.38 against 0.41 is inside run to run
variation. The cap of 200 tests per attempt didn't change the test count,
because the count resets with each attempt. v4 was submitted on 2026-10-24
(submission 57588171). v1 (submission 57565016) scored 0.08 on the
leaderboard, a quarter of its 0.32 public eval estimate.
Logs: `bench/evals/2026-10-24-fluid-llm-v4/`.
v4 (submission 57588171) scored 0.23 on the leaderboard. v5 is main at f2288a1 with the
LLM on; offline without it that code scored 10 levels and 0.63, so the LLM adds levels on
ar25, dc22 and m0r0 but plays slower wins. Logs: `bench/evals/2026-10-25-fluid-llm-v5/`.
v7 ran the DRIVE build from before the review fix pass. Its score is v5's, and for a
plain reason: no LLM fact ever reached the model. Qwen sent `facts` as one object instead
of a list in 1,638 of 1,640 replies and wrote ids as `"#3"`, and the parser dropped both,
so play-out had only observed facts and most moves fell through to slice-1 fallback.
2f90da9 reads a lone object as a list of one and integers written as strings; kernel v8 is
d9f726e's review fixes plus that. Logs: `bench/evals/2026-10-26-drive-v7/`.
v8 seeded 550 claims and scored the same. The cause is throughput. Nearly every move woke
the LLM (612 wakes on a miss, 205 on an experiment's result), so each turn bought 1.2
moves on average and a game made 50 to 320 moves before the last 10% of its budget, when
turns stop and B0 plays out the clock (that's where most fallback moves come from: 3,594
of ls20's in v7). Turns were 30 s because 8 games share the GPU and 43% of each reply was
prose before the JSON. v9 (6386c35) asks for JSON-only replies, raises the reply cap to
4,096 and reads more of the model's spellings (`c11`, `cover`, `no change`). c17cf7d then
wakes only on level start and game over while a modeled move remains, lets an LLM plan run
past its experiment step, and lets the first observation of a claimed fact replace the
claim (it used to break the fact). Logs: `bench/evals/2026-10-26-drive-v8/`.

v9 scored 0.52, again with 8 levels: lf52 and ls20 new and su15 and cn04 lost.
JSON-only replies barely moved turn time (27.9 s against 29.6 s): the shared GPU queue sets it, more than
the prose. Sim moves doubled to 6,542. Misses still drove the wakes (601 of
1,517), which c17cf7d removes. The parser still drops a lot: 837 move facts with no
object, 830 reach goals aimed "off screen", and effects spelled `disappears` or
`blocked`, which now parse. Logs: `bench/evals/2026-10-26-drive-v9/`.

v10 reached 11 levels and 0.53, the best DRIVE run so far, and was merged to main. Wake
labels still read "miss that matters" because a wake names its top pending reason; the
trigger was no modeled move. That trigger is now the limit: 2.7 moves a turn, and 18,760
of 23,156 moves came after a game's last turn. Every level the LLM stretch won came in its
first 140 moves. The next build (DRIVE_GAP_MOVES = 5) lets the agent's own policy play 5
unmodeled moves before a turn for no modeled move. Logs: `bench/evals/2026-10-26-drive-v10/`.

v11 (16fe583, 5 unmodeled moves between turns) scored 11 levels and 0.577 on Kaggle.

### Watching the agent, and the explorer as the backbone (2026-10-27)

Ten DRIVE runs moved the scorecard from 0.31 to 0.58 against a target of 60, so the
work turned to watching games instead of tuning. `scripts/watch.py` replays a run log
in the offline toolkit and draws the frames with the agent's reasons;
`scripts/watch_explore.py` does the same live for the explorer. What the 16 zero-level
games of v10 showed:

- Most moves changed nothing but the move counter (ft09: 138 of 155 fallback moves;
  s5i5: 3,312 of 4,420), and with a move budget those moves end in game overs.
- Failed plans repeat: sb26 sent the same 8-click plan for over 200 moves.
- Goals don't hold: ka59 put each piece on its target at different times, never both.
- 80 to 90% of each game's wall time is spent waiting for the LLM.
- Only one of v10's 11 wins confirmed any goal guess, because the frame a winning move
  returns already shows the next level.

The plain explorer (no LLM, 3,000 moves a game, 108 s for all 25 games locally)
finished 8 levels, close to DRIVE's 11 in 2 hours. It keyed states by every pixel, and
nearly every game draws a move counter on an edge, so no state ever repeated. Blanking
edge lines that tick alone or change on 60% of moves (d889830) took it to 15.3 levels,
scorecard 0.457 (3 seeds). At 30,000 moves it reaches about 20 levels: level 1 in 17
games and almost never level 2. The 25 games have about 180 levels.

`GuidedExplorer` (c14ed60) credits color gaps and counts at their level extreme on the
frame before a win and heads for them on later levels: 19.3 levels at 10,000 moves
(3 seeds, local). Its credit is often empty or noisy: on the pre-win frame most
measurements sit at the level's usual value, and color gaps can't see a bar's end
reaching its marker. Later levels also add mechanics (su15 level 2 brings in new
pieces), which no credit from level 1 explains. Benches now run in a Kaggle CPU notebook
(`scripts/kaggle_bench_notebook.py`) after a 6-process local bench ran out of memory.

First Kaggle CPU bench (4 seeds, 10,000 moves a game, c14ed60 code): plain explorer 18.8
levels, guided 19.0, both mean scorecard 0.544. The statistical credit adds nothing
measurable. Level 1 falls in the same 17 games for both, level 2 in about 5 (cd82, sc25,
tu93, vc33, sometimes ar25, ft09, ls20). The scorecard swings from 0.41 to 0.80 across
seeds at about the same levels: it rides on how fast a few first levels (lp85, r11l, tn36)
are won.

Kaggle GPU bench (84144c2, 2 seeds, 10,000 moves): guided 19.0 levels, advised 16.5, both
scorecard 0.42. The advised explorer asks Qwen3.6-27B which color gaps and counts win the
level, in the background. Its goal sentences are often sensible (tu93: "move the yellow
cell to the purple cell"; su15: "connect the black cross to the red square by extending
the green path"), but color gaps and counts lose most of that, it often names counts of
colors that should stay, and choosing which known frontier to visit next is a weak lever
either way. Neither guidance helps a level that coverage can't reach within the move
budget. Logs: `bench/evals/2026-10-27-bench-gpu-advised/`.

Explorer fixes, Kaggle CPU bench, 4 seeds, 10,000 moves a game:

| Code | Change | Mean levels | Mean scorecard |
|---|---|---|---|
| c14ed60 | baseline (counters left out of the state) | 18.8 | 0.544 |
| 1dab62e | exhausted graph starts a clock-keyed round (g50t) | 19.2 | 0.546 |
| cda6968 | no frontier at a game over, moves dead after 2 game overs | 19.0 | 0.455 |
| e2cebdc | game-over frames keyed apart; idle kinds of move wait | 23.5 | 0.467 |

The game-over frame of tu93 looks exactly like a normal frame; with terminal states
getting no candidates, the shared key emptied that state's moves. Keyed apart, tu93 goes
from 0 to 5 levels on every seed, and dc22, s5i5, m0r0 and sp80 win level 1 on all four.
The scorecard doesn't follow the levels: the explorer wins in hundreds or thousands of
moves where humans take tens, and the score is (human/agent moves) squared per level.

At 30,000 moves (e2cebdc, 4 seeds) the explorer averages 28.5 levels, scorecard 0.471;
21 games win a level on some seed, ka59 on all four. re86, sb26, sk48 and wa30 never do.
A game takes about 145 s at that length, so the competition rerun (about 260 s a game)
could afford about 50,000 moves.

GPU bench with object matches (bfbd8a7, 2 seeds, 10,000 moves): explorer 22.5 levels,
advised 23.0. The model named a match in 20 of 181 replies, but only m0r0's ("make the
two halves identical") got through, and m0r0 then won 2 levels on both seeds. The rest
were dropped because single-color components split a picture tile into boxes of
different sizes; objects now include multi-color regions.

With multi-color objects (b0c1044, 2 seeds, 10,000 moves) the advised explorer scored
22.5 levels, the same as the plain explorer. Matches got through on 7 game runs, tr87
among them, but the model's tr87 pair was the white row against the orange row (a copy),
where the legend says each white picture maps to a different orange one. The model reads
layouts but doesn't infer such rules from one screen, so its guidance nets out to zero.

DRIVE v12 (5d6c5ba, the LLM-driven agent with the fixed explorer as its fallback) scored
5 levels and 0.379 on Kaggle. It still made only 100 to 1,900 moves a game, waiting on
the model, so the explorer fixes had no room to matter. DRIVE's last five runs average
about 0.48 (0.31 to 0.58, 5 to 11 levels); the explorer alone averages 0.47 over four
seeds with 23 to 28 levels and no GPU. The explorer is the base from here. Logs:
`bench/evals/2026-10-28-drive-v12/`.

## Open items

- The test flood (v4): on cd82, dc22 and s5i5 almost every test asks a
  question no earlier test asked (2,354 distinct of 2,447 on cd82), while the
  same cell is clicked up to 200 times (dc22 at (10, 40)). The question space
  is click label times subject times context, and click labels carry the
  clicked object's sig and ring color, so it never runs dry. A cap per
  attempt resets on every game over and doesn't bind. The earlier caps
  (reverted in a8ac9ba) lowered the count and the score together.
  Found offline on dc22: the subjects mostly recur (3 of 96 appear once). The
  leak is an action that reshapes a tracked thing. Facts are keyed by shape,
  so the new shape has none, and the same click came back as a test 59 times
  in 400 moves, each time about the newest shape. 5daf0c6 lets the tester use
  the facts a tracked thing has under its earlier shapes (391 tests to 303 on
  that probe). Still open: things that split or merge get new track ids and
  no history, and they make up most of what's left.

| date | commit | run | levels | score est. | notes |
|---|---|---|---|---|---|
| 2026-10-24 | fb765b7 | offline FLUID, 25 games, seed 0, 240 s per game, 4 workers | 5 | 0.13 | lf52, lp85 x2, r11l, tn36; 268 tests per level |
| 2026-10-24 | 5daf0c6 | same, with the reshaped-thing fix | 5 | 0.14 | ar25, g50t, lf52, lp85, r11l; 201 tests per level |
| 2026-10-24 | cd9007e | same, with kin for split, merged and clicked things | 10 | 0.37 | the five above plus bp35, cd82, ft09, m0r0, tn36 |
| 2026-10-24 | d828d73 | same, plus win confirmation on the predicted outcome and no off-board clicks | 10 | 0.37 | same games; lf52 confirms `clear c14`; lp85's 6 agent errors gone |
| 2026-10-24 | 8265321 | same, plus a bound of 3 tests per (label, subject) per level | 8 | 0.35 | lost bp35 and m0r0; reverted |
| 2026-10-25 | c2a7001 | main, offline FLUID, 25 games, 240 s per game, 2 workers, run twice | 11 | 0.37 | ar25, bp35, cd82, ft09, g50t, lf52, lp85, m0r0, r11l, tn36, vc33 (601 actions); both runs identical |
| 2026-10-25 | e175744 | counter mask, refuse seen no-ops, counters kept across lives; same, run twice | 8 | 0.49 | merged: ls20 (116), su15 (373), lp85 x2, lf52 (35), r11l (14), tn36 (14), vc33 (265); lost ar25, bp35, cd82, ft09, g50t, m0r0; both runs identical |
| 2026-10-25 | 2e16328 | forced-move branch (forced moves, anytime and best-first search, commit probe, kinds, mode, progress, budget, goal kinds and rule-book, doomed-life RESET, routes); same, one run | 7 | 0.39 | not merged: prediction accuracy 0.92, but lf52 35 to 298 actions, vc33 265 to 1,519, lost ls20 and su15, gained cd82 |
| 2026-10-25 | ef4c9ca | forced-move without the doomed-life RESET and without the budget filter on goal ranking; same | 9 | 0.39 | lf52 298, ls20 lost; the twelve-goal cut still on |
| 2026-10-25 | 9b0c53a | same, and search gets every open goal instead of the best twelve | 10 | 0.63 | merged (c2c71f1): cd82 (62), lf52 (19), lp85 (26), ls20 (60), r11l (15), sp80 (827), su15 (434), tn36 (15), vc33 x2 (117, 195); lost lp85's second level |
| 2026-10-25 | 15fa5f2 | sequence goal and progress indicators; bump into a reached target (sequence steps and the commit probe); same, one run | 11 | 0.57 | merged (958bb21) on more levels: new sc25 (479); slower lp85 (26 to 88), ls20 (60 to 103), lf52 (19 to 22); the rest unchanged |
| 2026-10-25 | e440d5b | match_thing goal (restyle families, stored reference picture, pinned reference); same, one run | 12 | 0.34 | merged on more levels: new ar25 (372), m0r0 (831); slower tn36 (15 to 3,219), cd82 (62 to 611); the rest unchanged |
| 2026-10-25 | 2783b47 | main with the sequence and match_thing goals together; same, one run | 13 | 0.28 | main as it stands: ar25 (565), sc25 (409), m0r0 (831); tn36 still 3,219 and cd82 611 |
| 2026-10-25 | 705e420 | main plus the route lock (`_route_choice`, `Kinematics.tour`, `_closer`); same, one run | 10 | 0.28 | not merged: loses ar25, cd82 and sc25; m0r0 faster (831 to 698) |
| 2026-10-25 | 213daa1 | main, but search doesn't pin a match_thing reference whose look is a known mover ; same, one run | 12 | 0.28 | not merged: loses ar25; tn36 still 3,219. Correction: the 14-move tn36 result came from turning pinning off entirely; this commit alone takes 283 moves, like main, because the pinned references (c3:1d2f) have no move fact yet. tn36 is seed 0 alone and seed 20 as game 21 of 25 (`make_agent` seeds by list position); with no plan the explorer fallback finds the win at 284 moves with seed 0 and 3,218 with seed 20, which is the whole full-run gap |
| 2026-10-25 | 4c6f4be | composite pieces (Section 7: decorations on a sprite's rim join it, selection shown by use, SEL_RANGE 16); same, one run | 13 | 0.05 | not merged: r11l 15 to 91 and lf52 22 to 771 sink the score; sc25 409 to 123, cd82 611 to 416, tn36 3,219 to 1,785; before the click-cell offset fix (76433c1) |
| 2026-10-25 | e0d371c | search without match_thing pins first, pinned pass only if nothing is found; same, one run | 11 | 0.42 | merged on score: tn36 3,219 to 13; loses ar25 (main's 565 rides on a pinning side effect at move 315) and sc25 (409); the rest unchanged |
| 2026-10-25 | 4bd5735 | composite pieces with the recurrence rule (a piece look shows twice) and click-cell offsets, on the old main; same, one run | 11 | 0.21 | not merged: r11l back to 15, but lf52 190 (main 22), tn36 and sc25 lost; lf52 still burns its opening on sequence tries (slot rows look like an indicator) and records jumps under scattered offsets (clicks land on marker fragments) |
| 2026-10-26 | a5b76d7 | composite pieces shelved on Kameron's call; branch composite-pieces kept as is. Kaggle v6 pushed from main a5b76d7 (pin-fallback, sequence, match_thing) | | | public score pending |

One run each under wall-clock caps. The first two rows are within noise; the
drop in tests is the effect that fix was for.
The 2026-10-25 pair ran each arm twice, alternating, and each pair of runs came out
identical game by game: with no LLM, the offline FLUID agent is deterministic, so
offline comparisons need one run per arm (the 4 to 7 points between runs in the spec
come from the LLM configuration). e175744 is merged under the rule (higher score):
fewer levels, but wins are faster (r11l 53 to 14 actions, tn36 102 to 14, vc33 601 to
265) and ls20 and su15 are new. Four of the six lost levels were won by the explorer
fallback or novelty moves on main. Logs: scratchpad `ab2/`.
The forced-move branch lost to main, so it was bisected on the five games that moved
(lf52, vc33, ls20, su15, lp85) at each section boundary. Two commits did the damage.
The doomed-life RESET compared moves left against goal distances, which estimate moves
and run high when one click slides a piece many cells: su15 reset 1,678 times and
ls20 and su15 lost their level. Goal ranking cut search to the best twelve goals and
dropped goals that looked beyond the budget, which took ls20 from 60 actions to 335.
Both are removed; 9b0c53a is merged under the rule (more levels and a higher score).
One caveat on determinism: the same commit gives the same result for the same game
set, but main on the five-game set won vc33 in 78 actions against 265 in the full run,
so a game's result depends on which games run beside it. Compare full runs only.
Logs: scratchpad `bisect/` and `full3/`.
The sequence branch (Task 5.4) is merged on levels with a lower score. No public probe
trace showed a progress indicator, so the new sc25 level and the slower wins most likely
come from the bump (running into a reached target, now also a commit-probe candidate),
which every game sees. That hasn't been checked by a run without the bump.
The split and merge half of cd9007e changed nothing on the dc22 probe (303
tests either way). The leak that remained was in the click label: it carries
the clicked thing's sig, so a click on a region that keeps changing shape
(the color-0 area on dc22) was a new label each time, and one subject was
tested under 34 of them. cd9007e also tries the clicked thing's label under
its earlier sigs: 303 tests to 210 on the probe, 86 distinct click labels to
42. Offline it doubled the levels and kept every level main had, which is
past what one noisy run explains, so it's merged. Prediction accuracy held
(0.891 to 0.887).
d828d73 fixes two defects found in those logs. No goal had ever been confirmed at a
level-up: the frame a winning move returns already shows the next level, and the check
ran on the frame before the move, so a goal that became true on that move never
counted. And a predicted scene could push a thing past the board's edge, and the
search then built a click off the board (6 agent errors on lp85). Both are fixed and
the score didn't move.
8265321 bounded a loop seen on lf52's level 2: one question asked 48 times, because
its `any` fact had broken and every later observation was filed under `blocked`, so
the fact was never re-seeded. The bound ended the loop and cost two levels (bp35's
win came from the explorer fallback and m0r0's from novelty moves, so this may be
noise), and it's reverted. The loop itself is still there.
Scale, for reading these rows: the scorecard is in percent, so 0.37 is 0.37 of 100.
Level counts barely move it, because a level scores (human ÷ agent actions)²: our won
levels took 12 to 306 actions. No game gets past level 1, and later levels weigh more.
Logs: scratchpad `ab-base/`, `ab-kin/`, `ab-split/` and `ab-confirm/` (not kept in the
repo).

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

## 2026-10-28: reach features (e34bbe9), Kaggle CPU, 4 seeds, 10k moves

The guided explorer now measures the mover's distance to each color (the avatar wears
its exit's color in tu93, so plain color gaps read zero).

| Policy | Levels per seed | Mean | Scorecard mean |
|---|---|---|---|
| explore | 24, 21, 23, 26 | 23.5 | 0.467 |
| guided (reach) | 24, 20, 24, 25 | 23.25 | 0.476 |

- tu93 level 2: 862 to 979 moves became 407 to 602. Level 3: about 1,200 became 439 to
  963. Level 4 got worse (3,300 to 3,950 became 6,400 to 7,100) and one seed lost it:
  the credited exit color misleads there.
- ar25 won level 1 once; cd82 lost level 1 on one seed. Same level count overall.
- Per-level watching of the four never-won games is in `docs/level-notes.md`.

## 2026-10-28: idle-move fix, 100k coverage, hybrid (Kaggle CPU)

| Run | Levels per seed | Mean levels | Scorecard mean |
|---|---|---|---|
| explore, e2a542d (walk to a live untried move first), 10k | 27, 25, 24, 26 | 25.5 | 0.468 |
| guided, e2a542d, 10k | 24, 25, 24, 24 | 24.25 | 0.477 |
| explore, e34bbe9, 100k, seed 0 | 37 | 37 | 0.662 |
| guided, e34bbe9, 100k, seed 0 | 36 | 36 | 0.420 |
| hybrid (FLUID mind 400 moves per level, then explore; 3fbefd2, before the idle fix), 10k | 27, 27, 27, 25 | 26.5 | 0.453 |

- The idle fix adds two levels at 10k (23.5 to 25.5), same score.
- Ten times the moves takes the explorer from about 25 to 37 of 180 levels: coverage
  alone stalls at level 1 or 2 in most games whatever the budget.
- The hybrid's mind wins some first levels at or near human moves (lf52 22, r11l 15,
  tn36 13, m0r0 111 to 116, ls20 103) and one seed took lp85 to level 5. Where the mind
  doesn't win, its 400 moves come first and slow the level down (vc33 level 0: 414 to
  445 moves against the explorer's 29 to 57). Net score unchanged.
- Baseline submission 57746426 (DRIVE v11) was made today; public score pending.

## 2026-10-28 evening: learning from wins (Kaggle CPU, hybrid, 4 seeds, 10k)

| Run | Levels per seed | Mean levels | Scorecard mean | Wins the mind learned from |
|---|---|---|---|---|
| hybrid 150 mind moves per level, veto fix (2b48c31) | 25, 30, 26, 26 | 26.75 | 0.473 | 5 of 30 |
| + credit probe-excluded goals, learn_win (eb350a1) | 25, 31, 27, 26 | 27.25 | 0.474 | 25 of 30 |

- Learning from a win went from 17% to 83% of wins. The levels barely moved, because
  the learned goals named exact shapes and didn't hold on the next level's pieces, and
  search preferred cheap fresh guesses over them. Both addressed next: confirmed forms
  searched first (e5a218e), forms carried by color too (4a64461); bench running.
- Hybrid 150 runs in about 1.5 h per 25 games against 4 h for hybrid 400.
