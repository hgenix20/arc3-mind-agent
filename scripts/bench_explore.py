"""Bench: explorer policies on all public games over several seeds, no LLM.

Why: one run per condition is too noisy to compare explorer changes. This runs
``submit --mode offline --policy P --seed S`` for every policy and seed, at most
``--jobs`` processes at a time, and prints per policy the levels each seed completed
on each game, then the mean levels and scorecard. It runs on the laptop or in the
Kaggle CPU bench kernel (``scripts/kaggle_bench_notebook.py``); results also go to
``<out>/bench.json``.

    python scripts/bench_explore.py --policies explore guided --seeds 0 1 2 \\
        --max-actions 10000 --jobs 3 --out build/local/b1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def run_jobs(cmds: list[tuple[str, int, Path, list[str]]], jobs: int) -> None:
    """Run each command with its log in ``<out>/run.log``, at most ``jobs`` at a time."""
    pending = list(cmds)
    live: list[subprocess.Popen[bytes]] = []
    while pending or live:
        live = [p for p in live if p.poll() is None]
        while pending and len(live) < jobs:
            _, _, out, cmd = pending.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = (out / "run.log").open("w", encoding="utf-8")
            live.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log))
        time.sleep(1)


def summarize(policy: str, runs: list[tuple[int, Path]]) -> dict[str, Any]:
    """Per-game levels by seed, mean levels and mean scorecard for one policy."""
    per_game: dict[str, list[int]] = {}
    scores: list[float] = []
    for seed, out in runs:
        res = json.loads((out / "results.json").read_text(encoding="utf-8"))
        for g in res["games"]:
            per_game.setdefault(g["game_id"][:4], []).append(int(g["levels_completed"]))
        tail = (out / "run.log").read_text(encoding="utf-8").strip().splitlines()[-1]
        score = float(tail.split("scorecard score ")[1].split(",")[0]) if "scorecard" in tail else 0
        scores.append(score)
        print(f"{policy} seed {seed}: {tail}")
    n = len(scores)
    mean_levels = sum(sum(v) for v in per_game.values()) / n
    for game, levels in sorted(per_game.items()):
        if any(levels):
            print(f"  {game} {levels}")
    print(
        f"{policy}: mean levels {mean_levels:.1f}, mean scorecard {sum(scores) / n:.3f}, {n} seeds"
    )
    return {"per_game": per_game, "scores": scores, "mean_levels": mean_levels}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--policies", nargs="+", default=["explore"], choices=["explore", "guided"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--jobs", type=int, default=3, help="processes at a time")
    p.add_argument("--env-dir", default=None, help="default: submit's")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cmds: list[tuple[str, int, Path, list[str]]] = []
    for policy in a.policies:
        for seed in a.seeds:
            out = a.out / policy / f"seed{seed}"
            cmd = [
                sys.executable, "-m", "arcagent.agent.submit", "--mode", "offline",
                "--policy", policy, "--seed", str(seed), "--max-actions", str(a.max_actions),
                "--game-seconds", "600", "--out", str(out),
            ]  # fmt: skip
            if a.env_dir:
                cmd += ["--env-dir", a.env_dir]
            cmds.append((policy, seed, out, cmd))
    run_jobs(cmds, a.jobs)
    report = {
        policy: summarize(policy, [(s, o) for pol, s, o, _ in cmds if pol == policy])
        for policy in a.policies
    }
    (a.out / "bench.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
