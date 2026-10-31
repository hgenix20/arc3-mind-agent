"""Write the gaps report for one planner-agent eval: per game, then the gaps it shows.

Reads ``<eval dir>/results.json`` (the run's games and scorecard) and
``<eval dir>/metrics.json`` (``scripts/mind_metrics.py --json``; its per-game ``v2`` block
holds the planner's calls, segments, handoffs, forks, harm, gates and calibration) and
writes a Markdown table plus a summary of the gaps. Missing fields print as "-" so the
report never fails on a partial run. Human action counts come from
``data/public_human_actions.json`` when the game is a public one.

Usage: python scripts/v2_gaps.py <eval dir> [--out docs/superpowers/plans/<date>-v2-gaps.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HUMAN_FILE = ROOT / "data" / "public_human_actions.json"


def _get(d: Mapping[str, Any] | None, *keys: str, default: Any = "-") -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _human(humans: Mapping[str, Any], game_id: str) -> list[int]:
    for key in (game_id, game_id.split("-", 1)[0]):
        val = humans.get(key)
        if isinstance(val, list):
            return [int(v) for v in val]
        if isinstance(val, Mapping) and isinstance(val.get("baseline_actions"), list):
            return [int(v) for v in val["baseline_actions"]]
    return []


def game_rows(
    results: Mapping[str, Any], metrics: Mapping[str, Any], humans: Mapping[str, Any]
) -> list[dict[str, Any]]:
    envs = {
        e["id"]: e
        for e in (_get(results, "scorecard", "environments", default=[]) or [])
        if isinstance(e, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for g in results.get("games", []):
        gid = str(g.get("game_id", "?"))
        env = envs.get(gid, {})
        m = metrics.get("games", {}).get(gid, {})
        v2 = m.get("v2", {}) if isinstance(m, Mapping) else {}
        human = _human(humans, gid)
        rows.append(
            {
                "game": gid.split("-", 1)[0],
                "levels": f"{g.get('levels_completed', '-')}/{g.get('win_levels', '-')}",
                "actions": g.get("actions", "-"),
                "human_l1": human[0] if human else "-",
                "score": round(float(env["score"]), 3)
                if isinstance(env.get("score"), (int, float))
                else "-",
                "stop": g.get("stop", "-"),
                "planner_calls": _get(v2, "planner", "calls"),
                "tokens": _get(v2, "planner", "tokens"),
                "segments": _get(v2, "planner", "segments"),
                "handoffs": _count(_get(v2, "handoffs", default={})),
                "forks": _get(v2, "forks", "opened"),
                "harm": _get(v2, "harm", "events"),
                "gate_trips": _get(v2, "gates", "trips"),
                "calibration": _calib(_get(v2, "calibration", default=[])),
            }
        )
    return rows


def _count(d: Any) -> Any:
    return sum(int(v) for v in d.values()) if isinstance(d, Mapping) else "-"


def _calib(bins: Any) -> str:
    if not isinstance(bins, Sequence) or not bins:
        return "-"
    parts = [
        f"{b.get('bin')}:{b.get('rate')}" for b in bins if isinstance(b, Mapping) and b.get("n")
    ]
    return " ".join(parts[:4]) if parts else "-"


def render(rows: Sequence[Mapping[str, Any]], results: Mapping[str, Any], eval_dir: Path) -> str:
    cols = [
        "game",
        "levels",
        "actions",
        "human_l1",
        "score",
        "stop",
        "planner_calls",
        "tokens",
        "segments",
        "handoffs",
        "forks",
        "harm",
        "gate_trips",
        "calibration",
    ]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    total_levels = sum(
        int(str(r["levels"]).split("/")[0]) for r in rows if str(r["levels"])[0].isdigit()
    )
    score = _get(results, "scorecard", "score")
    out = [
        f"# Agent v2 gaps report: {eval_dir.name}",
        "",
        f"Source: `{eval_dir}` (results.json, metrics.json). "
        f"Levels completed: {total_levels} across {len(rows)} games. Scorecard score: {score}.",
        "",
        "## Per game",
        "",
        head + body,
        "## Gaps (fill from the logs)",
        "",
        "- Learning: which rules were created, promoted, refined; where the planner's claims "
        "were wrong.",
        "- Intelligence: where segments broke, what the forks did, which goals the planner missed.",
        "- Efficiency: moves per completed level against the human count; where the moves "
        "went (fallback, planner, explorer).",
        "- Each gap tied to a spec section.",
        "",
    ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    results = json.loads((a.eval_dir / "results.json").read_text(encoding="utf-8"))
    mpath = a.eval_dir / "metrics.json"
    metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    humans = json.loads(HUMAN_FILE.read_text(encoding="utf-8")) if HUMAN_FILE.exists() else {}
    text = render(game_rows(results, metrics, humans), results, a.eval_dir)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        print("wrote", a.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
