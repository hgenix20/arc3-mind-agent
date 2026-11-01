# Planner-comparison figures

Rendered 2026-10-20 from the artifact "The 34.3 notebook against a game-tree
planner" (https://claude.ai/artifact/1GtkHS5cnQ3tZNVZA2odi5, version
1792425742-7b4a, dated 2026-10-19 on the page). Every chart on that page is
drawn by the page's own JavaScript from a model of each strategy's cost, so
these PNGs are screenshots of the page rendered by headless Chrome at 2x,
light theme, and the CSVs are that model's output dumped from the page
itself. Nothing here is re-derived.

Reproduce: `python scripts/render_planner_figures.py <index.html> <out_dir>`
with the page saved as `index.html`.

## What is measured and what is modeled

Only the leaderboard bars are measurements (Kaggle public leaderboard,
2026-10-19: random 0.18, CNN policy 0.25, the studied notebook 34.3, 5th
place 49.85, 1st place 68.91). Every planner number is a stated assumption
from the page's model (50 M ops/s, alpha-beta node counts, assumed plan
times); the page says so in each bar's tooltip, and the CSV carries those
notes in the `*_note` columns. The paper must present them as modeled, not
measured, until the agent runs on the 25 public games.

## Files

- `full-page-bf4.png`, `full-page-bf64.png`: the whole page at minimax
  branching factor 4 (keyboard games) and 64 (8x8 click lattice).
- `00-header.png` … `08-mechanism-by-mechanism.png`: one PNG per section,
  in page order; `06-five-metrics-bf64.png` is the five-metric section at
  branching factor 64.
- `chart-*.png`: one chart each. The five metric charts exist at `-bf4` and
  `-bf64`; the leaderboard and the score curve do not depend on it.
- `chart-data.json`: the page's `strategies()` output for both branching
  factors, every field including the tooltip notes.
- `chart-data-bf4.csv`, `chart-data-bf64.csv`: one row per strategy with
  seconds per action, simulated plies, action-multiple range, compute
  minutes per game and actions affordable per game, computed exactly as the
  page computes them (800 actions per game; 48.4 GPU-stream minutes and
  235.6 CPU core-minutes available per game).

## Export-only layout change

The page's horizontal bar charts clip their longest row label and let the
hardware column overlap the value on the longest bars. The render script
widens the label gutter (168 to 240 px in the SVG's coordinate space) and
reserves a right gutter for the hardware column (16 to 110 px). Data,
scales and bar lengths are unchanged.
