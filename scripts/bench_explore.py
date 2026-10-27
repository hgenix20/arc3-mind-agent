"""Local bench: the explorer policy on all public games over several seeds, no LLM.

Why: one run per condition is too noisy to compare explorer changes. This runs
``submit --mode offline --policy explore`` once per seed, in parallel processes, and
prints per game the levels each seed completed, then the mean levels and scorecard.

    python scripts/bench_explore.py --seeds 0 1 2 --max-actions 3000 --out build/local/b1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    procs = []
    for seed in a.seeds:
        out = a.out / f"seed{seed}"
        cmd = [
            sys.executable, "-m", "arcagent.agent.submit", "--mode", "offline",
            "--policy", "explore", "--seed", str(seed), "--max-actions", str(a.max_actions),
            "--game-seconds", "600", "--out", str(out),
        ]  # fmt: skip
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "run.log").open("w", encoding="utf-8")
        procs.append((seed, out, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log)))
    per_game: dict[str, list[int]] = {}
    scores: list[float] = []
    for seed, out, proc in procs:
        proc.wait()
        res = json.loads((out / "results.json").read_text(encoding="utf-8"))
        for g in res["games"]:
            per_game.setdefault(g["game_id"][:4], []).append(int(g["levels_completed"]))
        tail = (out / "run.log").read_text(encoding="utf-8").strip().splitlines()[-1]
        score = float(tail.split("scorecard score ")[1].split(",")[0]) if "scorecard" in tail else 0
        scores.append(score)
        print(f"seed {seed}: {tail}")
    for game, levels in sorted(per_game.items()):
        if any(levels):
            print(f"  {game} {levels}")
    n = len(scores)
    total = sum(sum(v) for v in per_game.values()) / n
    print(f"mean levels {total:.1f}, mean scorecard {sum(scores) / n:.3f} over {n} seeds")


if __name__ == "__main__":
    main()
