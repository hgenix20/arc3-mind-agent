"""Planner ceiling: plan each level with the TRUE rules (the oracle) as the world model.

Why: the stack's planning layer can only finish levels its world model and search
budget allow. Swapping the LLM-written ``step()`` for the oracle measures the best
the planner could do with perfect rules, and how large the searches get, before any
LLM time is spent. Offline only (the oracle needs the public game files).

Per level: breadth-first search (``arcagent.plan.planner``) from the current state
until ``levels_completed`` rises, over the game's simple actions plus one click per
component; the found plan is executed on the real game. A level with no plan within
budget ends the game. Score uses the human action counts in ``data/public_human_actions.json``.

    python scripts/plan_ceiling.py --games ls20 ft09 --level-seconds 30 --out bench/ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arcagent.agent.explore import click_candidates  # noqa: E402
from arcagent.core.oracle import OracleModel  # noqa: E402
from arcagent.core.types import COORD_ACTION_ID, RESET_ID, Action, Phase  # noqa: E402
from arcagent.plan import planner  # noqa: E402

ENV_DIR = Path("C:/arc-study/data/environment_files")
HUMANS = ROOT / "data" / "public_human_actions.json"


def human_counts() -> dict[str, list[int]]:
    return json.loads(HUMANS.read_text(encoding="utf-8"))


def game_score(actions_per_level: list[int], humans: list[int]) -> float:
    total = sum(range(1, len(humans) + 1))
    got = sum(
        i * min(h / a, 1.0) ** 2
        for i, (a, h) in enumerate(zip(actions_per_level, humans, strict=False), 1)
        if a > 0
    )
    return 100.0 * got / total


def candidate_actions(m: OracleModel) -> list[Action]:
    obs = m.observe()
    acts = [Action(a) for a in obs.available_actions if a not in (RESET_ID, COORD_ACTION_ID)]
    if COORD_ACTION_ID in obs.available_actions:
        acts += [Action(COORD_ACTION_ID, x, y) for x, y in click_candidates(obs.grid)]
    return acts


def play(game: str, level_seconds: float, max_nodes: int) -> dict[str, Any]:
    real = OracleModel.load(ENV_DIR, game)
    per_level: list[int] = []
    searches: list[dict[str, Any]] = []
    t0 = time.monotonic()
    while not real.observe().terminal:
        level = real.observe().levels_completed

        def step(s: OracleModel, a: Action) -> OracleModel | None:
            c = s.clone()
            obs = c.step(a)
            return None if obs.phase == Phase.GAME_OVER else c

        def goal(s: OracleModel, level: int = level) -> bool:
            o = s.observe()
            return o.levels_completed > level or o.won

        r = planner.plan(
            step,
            real,
            goal,
            candidate_actions,
            key=lambda s: s.key(),
            max_nodes=max_nodes,
            time_limit_s=level_seconds,
        )
        searches.append({"level": level + 1, "result": r.summary()})
        if not r.actions:
            break
        for a in r.actions:
            real.step(a)
        per_level.append(len(r.actions))
    return {
        "game": game,
        "levels": len(per_level),
        "per_level": per_level,
        "searches": searches,
        "seconds": round(time.monotonic() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--level-seconds", type=float, default=30.0)
    ap.add_argument("--max-nodes", type=int, default=20_000)
    ap.add_argument("--out", default=str(ROOT / "bench" / "ceiling"))
    args = ap.parse_args()
    humans = human_counts()
    games = args.games or sorted(humans)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for g in games:
        try:
            rec = play(g, args.level_seconds, args.max_nodes)
        except Exception as e:  # one broken game must not stop the benchmark
            rec = {"game": g, "error": f"{type(e).__name__}: {e}", "levels": 0, "per_level": []}
        rec["score"] = (
            round(game_score(rec["per_level"], humans.get(g, [])), 2) if g in humans else None
        )
        rec["total_levels"] = len(humans.get(g, []))
        (out / f"{g}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(
            f"{g}: {rec['levels']}/{rec['total_levels']} levels, score {rec['score']}, "
            f"{rec.get('seconds')}s; {rec.get('error', '')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
