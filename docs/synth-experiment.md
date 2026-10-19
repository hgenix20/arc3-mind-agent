# Rule-synthesis experiment

Question: given only observed transitions of an ARC-AGI-3 game, can an
open-weight LLM write an executable Python rule model that predicts the next
frame? Metric: exact-match rate on held-out transitions, with a bounded repair
loop. Code lives in `arcagent/synth/`, tests in `tests/test_synth.py`.

The harness sees only what an agent sees through the action interface. It
never reads the game source files under `environment_files` and never shows
them to the model.

## Usage

```
python -m arcagent.synth.run --env-dir C:\arc-study\data\environment_files ^
    --games ls20 ft09 vc33 --steps 120 --rounds 3 ^
    --endpoint http://127.0.0.1:8001/v1 --model <name> ^
    --max-transitions 40 --out bench/synth/results.json
```

- `--dry-run` collects traces and writes the round-1 prompt plus a sample
  repair prompt (built from the identity candidate's mismatches) to
  `prompts/`. No endpoint is called.
- `--fake` answers every round with the identity candidate, so the whole
  pipeline (trace, prompt, subprocess verify, JSON, table) runs offline.
- `--env-dir` defaults to `ARC_ENV_DIR` or the path above. `--seed` (default
  0) seeds the probe policy and the engine.
- If `SYNTH_API_KEY` is set it is sent as a bearer token.

All outputs go next to `--out`: `traces/` (npz + json sidecar, git-ignored),
`prompts/` (json + readable txt, git-ignored), `code/<game>-r<round>.py`.

## Pipeline

1. **Trace** (`trace.py`). `collect_trace(env_dir, game, steps, seed)`. At each
   step: RESET if the phase is GAME_OVER or WIN; otherwise with probability
   0.5 a uniform random action, else the next entry of a cycle through every
   available action in order. ACTION6 expands into click targets: one per
   4-connected single-color component of non-background color (background =
   most frequent color), up to 8 sampled per expansion, at the component cell
   nearest its centroid; a 4x4 lattice when there are no components. Every
   action, RESET included, is a `Transition` (step, grid_before, action,
   grid_after, phase_after, levels_before, levels_after). Split by time:
   `train` is the first floor(0.7 n), `heldout` the rest.
2. **Prompt** (`prompt.py`). The system message states the interface (64x64
   ints 0..15 with -1 padding, `grid[y][x]`, actions 0..7, ACTION6 x/y,
   levels, phases), the rendering format, and the task: a module defining
   `step(grid, action, x=-1, y=-1) -> grid`, pure, deterministic, standard
   library only, plus optional `goal(grid) -> bool`. The user message shows
   the first `--max-transitions` training transitions: the starting grid in
   full, then one summary line and a cell diff per transition. A transition
   that changes more than 60 cells is shown as a full next grid instead, at
   most twice per prompt. Repair turns append the prior code as an assistant
   message and up to 10 training mismatches (true change, wrong cells with
   expected and predicted color, or the exception) with an instruction to fix
   the rules and not special-case examples.
3. **Verify** (`verify.py`). The candidate runs in `python -I` in a fresh temp
   dir with a 20 s wall limit for all transitions, environment reduced to
   `SYSTEMROOT`, and `socket` replaced with a stub that raises. Results stream
   to a JSON-lines file, so a timeout keeps finished transitions and marks the
   rest `timeout`. Candidate code never runs in the harness process.
4. **Loop** (`run.py`). Round 1 synthesizes; while held-out exact rate is
   below 0.95 and rounds remain, the next round repairs. Requests go to
   `<endpoint>/chat/completions` with `requests`, temperature 0.2, max_tokens
   6000, timeout 600 s.

## Rendering

Colors print as hex `0`-`f`, padding as `.`. A full grid is a header
`crop x=<x0> y=<y0> w=<w> h=<h>` (bounding box of non -1 cells) and one line
per run of identical rows, `<y0>[-<y1>]: <tokens>`, each token `c` or `c*n`.
The format is exact: `render.parse_grid` inverts it (tested). A single-color
64x64 grid renders in 33 characters. Diffs list `(x,y) old->new` row-major,
capped at 60 with `... and N more`.

## Results JSON

```
{
  "config": {games, steps, seed, rounds, max_transitions, endpoint, model,
             dry_run, temperature, max_tokens, repair_target},
  "games": {
    "<game>": {
      "trace":    {steps, train, heldout, seed, path},
      "baseline": {identity_exact_rate_train, identity_exact_rate_heldout},
      "rounds": [{
        round, error,                       # error: str when the call failed
        exact_rate_train, exact_rate_heldout,
        cell_accuracy,                      # held-out cells, crashes count as all wrong
        crashed,                            # transitions with an exception, bad shape or timeout, both splits
        timed_out, load_error,
        prompt_tokens, completion_tokens,   # from the response usage field, 0 if absent
        seconds,                            # wall time of the HTTP call
        prompt_chars, code_path
      }],
      "final": <last scored round or null>,
      "dry_run": {prompt_chars_r1, prompt_chars_repair_sample}   # --dry-run only, no rounds
    }
  }
}
```

A round with `error` set has none of the score fields, and the game stops
there. Rates over an empty split are 0.0.

## Baseline

The identity model (next grid = current grid) is scored in numpy for free and
reported as `baseline`; `--fake` runs the same model through the subprocess
verifier, and the two must agree (a test checks this). Observed at
`--steps 120 --seed 0` (84 train, 36 held-out):

| game | identity held-out exact | identity held-out cell accuracy |
|------|-------------------------|---------------------------------|
| ls20 | 0.00                    | 0.9884                          |
| ft09 | 0.89                    | 0.9990                          |
| vc33 | 0.00                    | 0.9849                          |

ls20 and vc33 change some cells on every step (a step counter or animation),
so exact match demands real rules. Most ft09 clicks change nothing, so its
exact rate is a weak signal on this trace; read the cell accuracy and the
train mismatches alongside it.

Round-1 prompt size at `--max-transitions 40`: ls20 21,499 characters, vc33
11,631, ft09 8,515. `PROMPT_CHAR_CAP` (40,000) is the tested upper bound.

## Failure modes and limits

- HTTP errors, a non-JSON reply or a reply missing `choices` end that game's
  rounds and are recorded; other games continue.
- A reply without a code fence is saved as a stub module and scores as a load
  failure.
- The sandbox is best-effort: it blocks `socket` and kills at the time limit,
  but a candidate can still touch the file system. Treat model output as
  untrusted and run on a disposable machine if that matters.
- The 20 s limit covers all transitions together, so a slow but correct model
  can be cut off; `verify(..., timeout_s=...)` raises it.
- The probe policy is shallow. It rarely completes a level, so traces mostly
  cover level 0 dynamics.
