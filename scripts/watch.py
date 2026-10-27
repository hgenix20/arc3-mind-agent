"""Watch a logged game: replay its moves locally and draw what the agent saw.

Why: the logs say what the agent chose and why; this shows the frames those choices
were made on, so a person can see where a game goes wrong. Development only.

It replays the ``action`` events of one run log (``bench/evals/<run>/logs/<game>.jsonl``)
in the offline toolkit, checks each frame against the logged ``frame_hash``, and writes
to ``--out``:

- ``sheet-<n>.png``: frames at the chosen steps, each labelled with its step, level and
  the agent's reason for the move made from it.
- ``timeline.txt``: one line per move (step, level, branch, why) with each LLM turn's
  model text and plan length inline.

    python scripts/watch.py bench/evals/2026-10-26-drive-v10/logs/ft09-0d8bbf25.jsonl \\
        --every 10 --out build/watch/ft09
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arcagent.agent.play import to_game_action  # noqa: E402
from arcagent.agent.submit import DEFAULT_ENV_DIR, make_arcade  # noqa: E402
from arcagent.core.frames import observation_from_frame  # noqa: E402
from arcagent.core.types import Action  # noqa: E402
from arcagent.mind.perception import frame_hash  # noqa: E402
from arcagent.mind.screen import PALETTE  # noqa: E402

CELL = 4  # pixels per grid cell on the sheet
LABEL_H = 34
COLS = 6
PER_SHEET = 36


def load(log: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def tile(grid: np.ndarray, label: str) -> Image.Image:
    table = np.array([*PALETTE, (0, 0, 0)], dtype=np.uint8)
    idx = np.where((grid >= 0) & (grid < len(PALETTE)), grid, len(PALETTE)).astype(np.intp)
    img = Image.fromarray(table[idx]).resize(
        (grid.shape[1] * CELL, grid.shape[0] * CELL), Image.Resampling.NEAREST
    )
    out = Image.new("RGB", (img.width, img.height + LABEL_H), "white")
    out.paste(img, (0, LABEL_H))
    d = ImageDraw.Draw(out)
    for i, chunk in enumerate((label[:44], label[44:88])):
        d.text((2, 2 + 15 * i), chunk, fill="black")
    return out


def sheet(tiles: list[Image.Image]) -> Image.Image:
    w, h = tiles[0].size
    rows = (len(tiles) + COLS - 1) // COLS
    out = Image.new("RGB", (COLS * (w + 4), rows * (h + 4)), "gray")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % COLS) * (w + 4), (i // COLS) * (h + 4)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("log", type=Path)
    p.add_argument("--every", type=int, default=10, help="draw every Nth step")
    p.add_argument("--steps", type=int, nargs="*", default=None, help="draw these steps too")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=10**9)
    p.add_argument("--env-dir", default=DEFAULT_ENV_DIR)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    recs = load(a.log)
    game = next(r["game"] for r in recs if "game" in r)
    acts = [r for r in recs if r.get("type") == "action"]
    turns = {r["step"]: r for r in recs if r.get("type") == "exec_turn"}
    a.out.mkdir(parents=True, exist_ok=True)
    arcade = make_arcade("offline", a.env_dir, a.out)
    env = arcade.make(game, save_recording=False)
    frame = env.reset()
    want = set(a.steps or [])
    tiles: list[Image.Image] = []
    lines: list[str] = []
    mismatch = 0
    for r in acts:
        step = r["step"]
        obs = observation_from_frame(frame)
        grid = np.asarray(obs.grid)
        if frame_hash(grid) != r.get("frame_hash"):
            mismatch += 1
        why = str(r.get("why", ""))
        lines.append(f"{step:5d} L{obs.levels_completed} {r.get('branch', '')}: {why[:100]}")
        if step in turns:
            t = turns[step]
            lines.append(f"      TURN plan_len={t.get('plan_len')} goals={t.get('goals', [])[:3]}")
        if a.first <= step <= a.last and (step % a.every == 0 or step in want):
            tiles.append(tile(grid, f"{step} L{obs.levels_completed} {r['action']} {why}"))
        act = r["action"]
        frame = env.step(*to_game_action(Action(act[0], *act[1:])))
        if frame is None:
            lines.append(f"      step {step} returned no frame")
            break
    for i in range(0, len(tiles), PER_SHEET):
        sheet(tiles[i : i + PER_SHEET]).save(a.out / f"sheet-{i // PER_SHEET}.png")
    (a.out / "timeline.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{game}: {len(acts)} moves, {mismatch} hash mismatches, {len(tiles)} frames drawn")


if __name__ == "__main__":
    main()
