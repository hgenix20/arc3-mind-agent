# Agent submission: how the competition entry runs

The competition is a Kaggle code competition: a notebook plays the private
games through a gateway inside Kaggle's rerun, and the score comes from the
gateway's own record of that play. The notebook also has to write
`/kaggle/working/submission.parquet`, and the Submit button only activates
after a committed run that produced it. Facts read from the competition pages
on 2026-10-20: 110 private games (55 public leaderboard, 55 private), 9 hours
of runtime, no internet, per-level score `min(human_actions / agent_actions,
1)^2`, per-game score the level-index-weighted mean, total the mean over games.

## Pieces

| piece | file | contract |
|---|---|---|
| frame conversion | `arcagent/core/frames.py` | `FrameDataRaw`-like object to `Observation`; no toolkit import |
| play loop | `arcagent/agent/play.py` | `play_game(env, policy, game_id, max_actions, deadline)`; uses only `reset`/`step`/`observation_space`; stops on WIN, budget, deadline, or three consecutive failed steps; returns `GameResult` |
| policy `probe` | `arcagent/synth/trace.py` `ProbePolicy` | the trace collector's probe policy as a stateful chooser, so a trace and a baseline game are the same action sequence for the same seed |
| policy `explore` | `arcagent/agent/explore.py` `ExplorerPolicy` | model-free state-graph exploration: untried (state, action) pairs first, ranked by a UCB over action kinds (clicks keyed by the color under the cursor), shortest known path back to a frontier state otherwise; the graph is cleared at each level change, the priors are kept |
| runner | `arcagent/agent/submit.py` | one scorecard, every game in sorted order, `results.json` + `submission.parquet`; `--mode auto` picks competition inside `KAGGLE_IS_COMPETITION_RERUN` |
| notebook | `scripts/kaggle_agent_notebook.py` | builds `build/kaggle-agent/` (CPU, no internet, Python 3.12 image, competition data attached) |

The oracle (`arcagent/core/oracle.py`) is never imported by the agent; the
notebook ships only `core/`, `synth/trace.py` and `agent/`.

## Running it

Offline on the public games (what the Kaggle commit run does):

```
python -m arcagent.agent.submit --mode offline --policy explore --env-dir C:\arc-study\data\environment_files --out bench/agent
```

Build, push, watch, submit:

```
python scripts/kaggle_agent_notebook.py
kaggle kernels push -p build/kaggle-agent
kaggle kernels status hgenix20/arc3-agent-probe-baseline
kaggle competitions submit -c arc-prize-2026-arc-agi-3 -k hgenix20/arc3-agent-probe-baseline -v <version> -f submission.parquet -m "<message>"
```

`kaggle competitions submissions -c arc-prize-2026-arc-agi-3` lists the
results. One submission per day counts; the user's go is required before any.

## Competition-mode details that matter

- `ARC_BASE_URL` defaults to `http://gateway:8001`; the runner strips a
  trailing slash because the toolkit joins `/api/...` onto it. `ARC_API_KEY`
  defaults to `test-key-123`, the value the milestone-winning notebook used.
- The runner polls `<base>/api/games` for up to 600 s before building the
  `Arcade`; the gateway starts asynchronously.
- Only one scorecard can be open in competition mode, so every game is played
  under the one opened at the start. Closing it is in a `finally`, and the
  gateway creates a zero-score entry for any game that was never played.
- `arcade.make` already resets a remote game; the play loop reuses that frame
  rather than spending a RESET.

## Budget

Defaults: 300 actions per game, 240 s per game, 8 h for the run. Offline the
25 public games take about 10 s in total; a gateway step is one HTTP round
trip, so 110 games at 300 actions fit the limit by a wide margin.

## Baseline numbers (offline, 2026-10-20)

Scorecard scores are percentages (the Kaggle leaderboard shows a random agent
at 0.18). Seed 0, 25 public games:

| policy | actions per game | levels completed | scorecard |
|---|---|---|---|
| probe | 300 | 2 (r11l, vc33) | 0.19 |
| explore | 300 | 6 (g50t, lf52, lp85, r11l, tn36, vc33) | 0.36 |
| explore | 1000 | 7 (+ m0r0) | 0.36 |

The explorer plateaus: more actions find almost no further levels, and the
levels it does finish cost several times the human count, so each is worth
a few percent of its game at most. A blind explorer has no goal; the LLM
layer (`duck` below) is where the leaderboard points are.

## The LLM layer: `hgenix20/arc3-duck-agent`

`scripts/kaggle_duck_notebook.py` derives a second kernel from Daniel
Franzen's public Milestone 2 notebook (`third_party/franzen-m2/`, Apache-2.0
at github.com/da-fr/arc-agi-3-solution; a Duck-harness fork serving
Qwen3.8-Flash-Next INT4 through the Pennyroyal SGLang build, 34.3 public
leaderboard as published). Every change we make is a named-cell patch in that
script. The baseline build changes only the interactive commit: three games,
one pass, so a version costs about half a GPU hour instead of nine; the
competition rerun branch is untouched. Attach the same two datasets and two
models as the reference (the generator writes them into the metadata).
