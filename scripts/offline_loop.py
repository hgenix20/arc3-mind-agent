"""Offline loop: the public games on CPU with LLM replies replayed from disk.

Why: fast development without the GPU. This is a thin wrapper that builds an
argv for ``arcagent.agent.submit.main`` (``--mode offline --policy mind
--replay``) and calls it in-process; it adds no agent logic. It never calls
``submit.cli`` because that ends the process with ``os._exit``.

The mind policy ignores ``--game-seconds``; the scheduler splits
``--total-seconds`` itself (``scheduler.game_budget``: time left minus
``RESERVE_S``, spread over the games still to start). The wrapper derives
``--total-seconds = RESERVE_S + game_seconds * n_games / workers`` so the
first wave of games gets ``--game-seconds`` each. ``MIN_GAME_S`` (120 s) is a
floor in the scheduler, so a ``--game-seconds`` below 120 gives each game 120 s
(capped by the run's deadline). ``n_games`` is the count ``submit`` resolves
from ``--env-dir`` and ``--games``.

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
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arcagent.agent import submit  # noqa: E402
from arcagent.agent.scheduler import RESERVE_S  # noqa: E402


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
    p.add_argument("--max-actions", type=int, default=None, help="default: submit's mind default")
    p.add_argument("--env-dir", default=None)
    p.add_argument("--out", default=None, help="default: bench/offline/<UTC stamp>")
    args = p.parse_args(argv)
    if args.game_seconds <= 0 or args.workers < 1:
        p.error("need positive --game-seconds and --workers >= 1")
    return args


def derive_total_seconds(game_seconds: float, n_games: int, workers: int) -> float:
    """Run budget so the scheduler's first-wave game budget equals ``game_seconds``."""
    return RESERVE_S + game_seconds * n_games / workers


def count_games(env_dir: str, out: Path, wanted: Sequence[str] | None) -> int:
    """Games ``submit`` would play: its own arcade and ``select_games``, no scorecard opened."""
    arcade = submit.make_arcade("offline", env_dir, out)
    available = [e.game_id for e in arcade.available_environments]
    return len(submit.select_games(available, wanted))


def build_submit_argv(
    args: argparse.Namespace, out: Path, replies: Path, n_games: int
) -> list[str]:
    """The argv handed to ``submit.main``; pure so the budget arithmetic is checkable."""
    total = derive_total_seconds(args.game_seconds, n_games, args.workers)
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
    if args.max_actions is not None:
        argv += ["--max-actions", str(args.max_actions)]
    return argv


def metrics_summary(logs: Path) -> Any | None:
    """``mind_metrics.compute(load([logs]))`` when that module exists, else None."""
    path = ROOT / "scripts" / "mind_metrics.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("mind_metrics", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compute = getattr(module, "compute", None)
    load = getattr(module, "load", None)
    if not callable(compute) or not callable(load):
        return None
    records, malformed = load([logs])
    summary = compute(records)
    summary["malformed_lines"] = malformed
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    out = Path(args.out or ROOT / "bench" / "offline" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    replies = Path(args.replies) if args.replies else newest_replies(ROOT)
    if replies is None or not replies.is_dir():
        print("no saved replies found: running on the identity fallback")
        replies = out / "replies"
    else:
        print(f"replaying replies from {replies}")
    env_dir = args.env_dir or os.environ.get("ARC_ENV_DIR", submit.DEFAULT_ENV_DIR)
    n_games = count_games(env_dir, out, args.games)
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
