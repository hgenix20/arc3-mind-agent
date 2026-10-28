"""Watch the graph explorer play one game live and draw what it saw.

Why: the explorer writes no run log, so ``watch.py`` can't replay it. This plays
``ExplorerPolicy`` on one offline game and writes, to ``--out``:

- ``sheet-<n>.png``: every Nth frame, labelled with move, level and action, plus the
  frame each winning move was made from (and ``--before`` frames ahead of it).
- ``summary.txt``: per level the moves taken, the states and no-ops the graph holds,
  the clock lines found, and the game overs.

    python scripts/watch_explore.py s5i5 --moves 3000 --every 100 --out build/watch/x-s5i5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from watch import PER_SHEET, sheet, tile  # noqa: E402  # pyright: ignore[reportMissingImports]

from arcagent.agent.explore import ExplorerPolicy  # noqa: E402
from arcagent.agent.guided import GuidedExplorer  # noqa: E402
from arcagent.agent.play import to_game_action  # noqa: E402
from arcagent.agent.submit import DEFAULT_ENV_DIR, make_arcade, select_games  # noqa: E402
from arcagent.core.frames import observation_from_frame  # noqa: E402


def main() -> None:  # noqa: PLR0915  (one linear watch loop)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("game", help="4-char game id")
    p.add_argument("--moves", type=int, default=3000)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guided", action="store_true", help="play GuidedExplorer")
    p.add_argument("--before", type=int, default=0, help="also draw this many frames before a win")
    p.add_argument("--from-level", type=int, default=0, help="draw no frames before this level")
    p.add_argument("--env-dir", default=DEFAULT_ENV_DIR)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arcade = make_arcade("offline", a.env_dir, a.out)
    game = select_games([e.game_id for e in arcade.available_environments], [a.game])[0]
    env = arcade.make(game, save_recording=False)
    obs = observation_from_frame(env.reset())
    pol = GuidedExplorer(a.seed) if a.guided else ExplorerPolicy(a.seed)
    tiles: list[Image.Image] = []
    lines: list[str] = []
    level_start, overs = 0, 0
    recent: list[Image.Image] = []
    for move in range(a.moves):
        if obs.won:
            lines.append(f"won at move {move}")
            break
        act = pol.choose(obs)
        before = obs
        label = f"{move} L{obs.levels_completed} {act}"
        if move % a.every == 0 and obs.levels_completed >= a.from_level:
            tiles.append(tile(np.asarray(obs.grid), label))
        if a.before:
            recent = (
                [*recent[-(a.before - 1) :], tile(np.asarray(obs.grid), label)]
                if a.before > 1
                else [tile(np.asarray(obs.grid), label)]
            )
        frame = env.step(*to_game_action(act))
        if frame is None:
            lines.append(f"move {move}: no frame")
            break
        obs = observation_from_frame(frame)
        if obs.terminal and not obs.won:
            overs += 1
        if obs.levels_completed != before.levels_completed:
            g = pol.graph
            lines.append(
                f"level {before.levels_completed} done at move {move} "
                f"({move - level_start} moves, {len(g.edges)} states, "
                f"{sum(map(len, g.noop.values()))} no-ops, {overs} game overs)"
            )
            tiles.extend(recent[:-1])
            tiles.append(
                tile(np.asarray(before.grid), f"{move} L{before.levels_completed} WON by {act}")
            )
            level_start, overs = move, 0
    g = pol.graph
    lines.append(
        f"end: level {obs.levels_completed}, {len(g.edges)} states, {len(g.cands)} seen, "
        f"{sum(map(len, g.noop.values()))} no-ops, {overs} game overs this level, "
        f"clock lines {sorted(pol.clock)}"
    )
    if isinstance(pol, GuidedExplorer):
        top = sorted(pol.credit.items(), key=lambda kv: -abs(kv[1]))[:6]
        lines.append(f"credit: {[(f, round(w, 2)) for f, w in top]}")
    for i in range(0, len(tiles), PER_SHEET):
        sheet(tiles[i : i + PER_SHEET]).save(a.out / f"sheet-{i // PER_SHEET}.png")
    (a.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(game, *lines, sep="\n")


if __name__ == "__main__":
    main()
