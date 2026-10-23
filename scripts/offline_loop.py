"""Offline loop: the public games on CPU with LLM replies replayed from disk.

Why: fast development without the GPU. This is a thin wrapper that builds an
argv for ``arcagent.agent.submit.main`` (``--mode offline --policy mind
--replay``) and calls it in-process; it adds no agent logic. It never calls
``submit.cli`` because that ends the process with ``os._exit``.

``submit`` has a per-game ``--game-seconds`` flag, but the mind policy is
bounded by ``--total-seconds`` alone. The wrapper therefore derives
``--total-seconds = game_seconds * n_games / workers + TOTAL_MARGIN_S``.

Replies: ``--replies`` or else the newest ``bench/evals/*/replies`` by
directory mtime. With none, replay answers with the identity program and the
script says so. Out: ``bench/offline/<UTC stamp>`` unless ``--out`` is given.
If ``scripts/mind_metrics.py`` exposes ``compute``, its summary over
``<out>/logs`` is printed; otherwise that step is skipped.

    python scripts/offline_loop.py --games ls20 ft09 --game-seconds 20
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arcagent.agent import submit  # noqa: E402

TOTAL_MARGIN_S = 30.0  # scorecard open/close and agent start-up, outside the per-game budget


def newest_replies(root: Path) -> Path | None:
    """The ``replies`` directory under ``root/bench/evals`` with the latest mtime, or None."""
    found = [p for p in (root / "bench" / "evals").glob("*/replies") if p.is_dir()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--games", nargs="*", default=None, help="4-char ids; default all public")
    p.add_argument("--config", choices=sorted(submit.CONFIGS), default="FULL")
    p.add_argument("--replies", default=None, help="default: newest bench/evals/*/replies")
    p.add_argument("--game-seconds", type=float, default=120.0)
    p.add_argument("--workers", type=int, default=submit.DEFAULT_WORKERS)
    p.add_argument("--env-dir", default=None)
    p.add_argument("--out", default=None, help="default: bench/offline/<UTC stamp>")
    args = p.parse_args(argv)
    if args.game_seconds <= 0 or args.workers < 1:
        p.error("need positive --game-seconds and --workers >= 1")
    return args


def build_submit_argv(
    args: argparse.Namespace, out: Path, replies: Path, n_games: int
) -> list[str]:
    """The argv handed to ``submit.main``; pure so the budget arithmetic is checkable."""
    total = args.game_seconds * n_games / args.workers + TOTAL_MARGIN_S
    argv = [
        "--mode",
        "offline",
        "--policy",
        "mind",
        "--replay",
        "--config",
        args.config,
        "--workers",
        str(args.workers),
        "--game-seconds",
        str(args.game_seconds),
        "--total-seconds",
        str(total),
        "--out",
        str(out),
        "--replies-dir",
        str(replies),
    ]
    if args.games:
        argv += ["--games", *args.games]
    if args.env_dir:
        argv += ["--env-dir", args.env_dir]
    return argv


def metrics_summary(logs: Path) -> Any | None:
    """``scripts/mind_metrics.compute(logs)`` when that module exists, else None."""
    path = ROOT / "scripts" / "mind_metrics.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("mind_metrics", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compute = getattr(module, "compute", None)
    return compute(logs) if callable(compute) else None


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    out = Path(args.out or ROOT / "bench" / "offline" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    replies = Path(args.replies) if args.replies else newest_replies(ROOT)
    if replies is None or not replies.is_dir():
        print("no saved replies found: running on the identity fallback")
        replies = out / "replies"
    else:
        print(f"replaying replies from {replies}")
    n_games = len(args.games) if args.games else 25
    doc = submit.main(build_submit_argv(args, out, replies, n_games))
    for g in doc["games"]:
        print(
            f"{g['game_id']}: levels {g['levels_completed']}/{g['win_levels']} "
            f"actions {g['actions']} stop {g['stop']}"
        )
    summary = metrics_summary(out / "logs")
    if summary is not None:
        print(summary)
    return doc


if __name__ == "__main__":
    main()
