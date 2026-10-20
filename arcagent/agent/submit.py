"""Play every game the Arcade offers and write the submission outputs.

    python -m arcagent.agent.submit --mode offline --env-dir <environment_files> \\
        --out bench/agent [--games ls20 ft09] [--max-actions 300]

Modes:
- ``offline``: the public games under ``--env-dir`` through the local wrapper.
- ``competition``: the private games served by the Kaggle gateway
  (``ARC_BASE_URL``, default ``http://gateway:8001``; ``ARC_API_KEY``,
  default ``test-key-123``), after waiting for the gateway to answer.
- ``auto``: ``competition`` when ``KAGGLE_IS_COMPETITION_RERUN`` is truthy,
  else ``offline``. The Kaggle notebook uses this.

Flow: open one scorecard, play each game in sorted order with a fresh policy
(``--policy``: ``probe`` or ``explore``; seed ``--seed + index``), close the scorecard, then write
``results.json`` (every ``GameResult``, the scorecard as returned, timings) and
``submission.parquet`` (one row per game: ``row_id``, ``game_id``,
``end_of_game``, ``score``) under ``--out``.

Bounds: ``--max-actions`` per game, ``--game-seconds`` per game and
``--total-seconds`` for the whole run; a game that cannot start before the
run deadline is recorded with ``stop = "skipped"``. An exception inside one
game is recorded in that game's ``error`` field and the run continues; the
scorecard is closed even if everything fails. The competition scores a
submission from the gateway's own record, so the parquet is the required
output artifact, not the score source.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from arcagent.agent.explore import ExplorerPolicy
from arcagent.agent.play import GameResult, Policy, play_game
from arcagent.synth.trace import ProbePolicy

MODES = ("auto", "offline", "competition")
POLICIES = ("probe", "explore")
DEFAULT_ENV_DIR = r"C:\arc-study\data\environment_files"
DEFAULT_MAX_ACTIONS = 300
DEFAULT_GAME_SECONDS = 240.0
DEFAULT_TOTAL_SECONDS = 8 * 3600.0
GATEWAY_URL = "http://gateway:8001"
GATEWAY_KEY = "test-key-123"
GATEWAY_WAIT_S = 600.0
GATEWAY_POLL_S = 5.0
PARQUET_COLUMNS = ("row_id", "game_id", "end_of_game", "score")
TAGS = ["agent", "arcagent-probe"]

_LOG = logging.getLogger("arcagent.agent")


def resolve_mode(mode: str, environ: Mapping[str, str] = os.environ) -> str:
    """``auto`` -> competition inside a Kaggle rerun, offline otherwise."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode != "auto":
        return mode
    flag = environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
    return "competition" if flag in {"1", "true"} else "offline"


def wait_for_gateway(
    base_url: str,
    timeout_s: float = GATEWAY_WAIT_S,
    *,
    poll_s: float = GATEWAY_POLL_S,
    fetch: Callable[[str], Any] = lambda url: urlopen(url, timeout=10),
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Poll ``<base_url>/api/games`` until it answers; return the seconds waited.

    Raises ``TimeoutError`` with the last error after ``timeout_s``.
    """
    t0 = clock()
    last = ""
    while clock() - t0 < timeout_s:
        try:
            with fetch(f"{base_url}/api/games"):
                return clock() - t0
        except Exception as exc:  # any failure means "not yet"
            last = repr(exc)
        sleep(poll_s)
    raise TimeoutError(f"gateway at {base_url} did not answer in {timeout_s:.0f}s: {last}")


def make_arcade(mode: str, env_dir: str, out: Path, environ: Mapping[str, str] = os.environ) -> Any:
    """Construct the toolkit's ``Arcade`` for ``mode`` (imports the wheels here)."""
    from arc_agi import Arcade, OperationMode  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    quiet = logging.getLogger("arc_agi.arcade")
    quiet.setLevel(logging.WARNING)
    recordings = str(out / "recordings")
    if mode == "offline":
        return Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=env_dir,
            recordings_dir=recordings,
            logger=quiet,
        )
    base_url = environ.get("ARC_BASE_URL", GATEWAY_URL).rstrip("/")
    key = environ.get("ARC_API_KEY", GATEWAY_KEY)
    waited = wait_for_gateway(base_url)
    _LOG.info("gateway %s answered after %.0fs", base_url, waited)
    return Arcade(
        arc_api_key=key,
        arc_base_url=base_url,
        operation_mode=OperationMode.COMPETITION,
        environments_dir=str(out / "no-local-environments"),
        recordings_dir=recordings,
        logger=quiet,
    )


def select_games(available: Sequence[str], wanted: Sequence[str] | None) -> list[str]:
    """Sorted game ids; with ``wanted`` only those whose 4-char prefix is listed.

    Raises ``ValueError`` naming any wanted prefix with no match.
    """
    ids = sorted(available)
    if not wanted:
        return ids
    chosen = [g for g in ids if g.split("-", 1)[0] in set(wanted)]
    missing = sorted(set(wanted) - {g.split("-", 1)[0] for g in chosen})
    if missing:
        raise ValueError(f"games not available: {missing}")
    return chosen


def scorecard_dict(card: Any) -> dict[str, Any] | None:
    """The closed scorecard as plain JSON data, or None if the toolkit returned none."""
    if card is None:
        return None
    dump = getattr(card, "model_dump", None)
    data = dump(mode="json") if callable(dump) else dict(card)
    if not isinstance(data, dict):
        raise TypeError(f"scorecard dumped to {type(data).__name__}, expected a dict")
    return {str(k): v for k, v in data.items()}


def submission_rows(
    results: Sequence[GameResult], scorecard: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """One row per played game; score from the scorecard when it has the game, else 0."""
    scores: dict[str, float] = {}
    for env in (scorecard or {}).get("environments", []) or []:
        scores[str(env.get("id"))] = float(env.get("score") or 0.0)
    return [
        {
            "row_id": f"{i}_0",
            "game_id": r.game_id,
            "end_of_game": True,
            "score": scores.get(r.game_id, 0.0),
        }
        for i, r in enumerate(results)
    ]


def write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write the submission table with a fixed schema (needs ``pyarrow``)."""
    import pyarrow as pa  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    import pyarrow.parquet as pq  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("game_id", pa.string()),
            ("end_of_game", pa.bool_()),
            ("score", pa.float64()),
        ]
    )
    table = pa.Table.from_pylist([dict(r) for r in rows], schema=schema)
    pq.write_table(table, str(path))


def make_policy(name: str, seed: int) -> Policy:
    """A fresh policy by name (``POLICIES``) for one game; raises ``ValueError`` otherwise."""
    if name == "probe":
        return ProbePolicy(seed)
    if name == "explore":
        return ExplorerPolicy(seed)
    raise ValueError(f"unknown policy {name!r}; choose from {POLICIES}")


def play_all(
    arcade: Any,
    games: Sequence[str],
    *,
    seed: int,
    max_actions: int,
    game_seconds: float,
    total_seconds: float,
    policy: str = "probe",
    clock: Callable[[], float] = time.monotonic,
    report: Callable[[str], None] = print,
) -> tuple[list[GameResult], list[dict[str, Any]], dict[str, Any] | None]:
    """Play ``games`` under one scorecard; return results, per-game errors, the scorecard."""
    t0 = clock()
    run_deadline = t0 + total_seconds
    results: list[GameResult] = []
    errors: list[dict[str, Any]] = []
    card = arcade.open_scorecard(tags=list(TAGS))
    try:
        for i, game_id in enumerate(games):
            now = clock()
            if now >= run_deadline:
                results.append(GameResult(game_id, 0, 0, 0, "NOT_PLAYED", 0.0, "skipped"))
                report(f"{game_id}: skipped, run budget spent")
                continue
            try:
                env = arcade.make(game_id, scorecard_id=card, save_recording=False)
                if env is None:
                    raise RuntimeError("arcade.make returned None")
                r = play_game(
                    env,
                    make_policy(policy, seed + i),
                    game_id,
                    max_actions=max_actions,
                    deadline=min(now + game_seconds, run_deadline),
                    clock=clock,
                )
            except Exception as exc:  # one game must not end the run
                errors.append({"game_id": game_id, "error": f"{type(exc).__name__}: {exc}"})
                results.append(GameResult(game_id, 0, 0, 0, "NOT_PLAYED", 0.0, "error"))
                report(f"{game_id}: error {type(exc).__name__}: {exc}")
                continue
            results.append(r)
            report(
                f"{game_id}: {r.levels_completed}/{r.win_levels} levels, {r.actions} actions, "
                f"{r.seconds:.1f}s, stop={r.stop}"
            )
    finally:
        try:
            scorecard = scorecard_dict(arcade.close_scorecard(card))
        except Exception as exc:  # report, keep the results
            errors.append({"game_id": None, "error": f"close_scorecard: {exc!r}"})
            scorecard = None
    return results, errors, scorecard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m arcagent.agent.submit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=MODES, default="auto")
    p.add_argument("--env-dir", default=os.environ.get("ARC_ENV_DIR", DEFAULT_ENV_DIR))
    p.add_argument("--games", nargs="*", default=None, help="4-char ids; offline only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=POLICIES, default="probe")
    p.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    p.add_argument("--game-seconds", type=float, default=DEFAULT_GAME_SECONDS)
    p.add_argument("--total-seconds", type=float, default=DEFAULT_TOTAL_SECONDS)
    p.add_argument("--out", default="bench/agent")
    args = p.parse_args(argv)
    if args.max_actions < 1 or args.game_seconds <= 0 or args.total_seconds <= 0:
        p.error("need --max-actions >= 1 and positive --game-seconds / --total-seconds")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the agent; returns (and writes) the results document."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = resolve_mode(args.mode)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    arcade = make_arcade(mode, args.env_dir, out)
    available = [e.game_id for e in arcade.available_environments]
    if not available:
        raise RuntimeError(f"no games available in {mode} mode")
    games = select_games(available, args.games if mode == "offline" else None)
    print(f"mode={mode} games={len(games)} max_actions={args.max_actions}")
    results, errors, scorecard = play_all(
        arcade,
        games,
        seed=args.seed,
        max_actions=args.max_actions,
        game_seconds=args.game_seconds,
        total_seconds=args.total_seconds,
        policy=args.policy,
    )
    rows = submission_rows(results, scorecard)
    write_parquet(rows, out / "submission.parquet")
    doc: dict[str, Any] = {
        "config": {
            "mode": mode,
            "seed": args.seed,
            "max_actions": args.max_actions,
            "game_seconds": args.game_seconds,
            "total_seconds": args.total_seconds,
            "policy": args.policy,
        },
        "started_unix": started,
        "elapsed_seconds": round(time.time() - started, 1),
        "games": [asdict(r) for r in results],
        "errors": errors,
        "scorecard": scorecard,
        "submission_rows": rows,
    }
    (out / "results.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    levels = sum(r.levels_completed for r in results)
    score = float((scorecard or {}).get("score") or 0.0)
    print(
        f"played {len(results)} games, {levels} levels completed, scorecard score {score:.4f}, "
        f"{len(errors)} errors, {doc['elapsed_seconds']}s; wrote {out / 'submission.parquet'}"
    )
    return doc


if __name__ == "__main__":
    main()
