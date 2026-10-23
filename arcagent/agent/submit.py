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

Mind policy (``--policy mind``): games run ``--workers`` at a time under one
scorecard (``arcagent.agent.scheduler``), each with a fresh ``MindAgent`` on
the ``--config`` ablation config. All games share one ``Synthesizer`` over the
model server at ``--endpoint`` (every reply recorded under ``--replies-dir``),
one synthesis thread pool of ``--workers`` threads and one ``ActivityClock``;
each game gets its own sandbox and its run log at ``<log-dir>/<game>.jsonl``.
Before the scorecard opens, ``wait_for_model`` polls ``<endpoint>/models``
until the server answers, so the 15-minute idle clock starts only once
actions can flow. ``--replay`` answers from the recorded replies instead and
skips the wait, as does the model-free ``EXPLORER`` config. Per-game
deadlines come from ``scheduler.game_budget``; ``--game-seconds`` is unused.
``--max-actions`` defaults to 20000 for the mind policy, 300 otherwise.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from arcagent.agent.explore import ExplorerPolicy
from arcagent.agent.play import GameResult, Policy, play_game
from arcagent.agent.scheduler import DEFAULT_MAX_ACTIONS as MIND_MAX_ACTIONS
from arcagent.agent.scheduler import MIND_TAGS, play_concurrent
from arcagent.core.types import Action, Observation
from arcagent.mind.agent import ActivityClock, MindAgent
from arcagent.mind.chat_store import recording_chat, replay_chat
from arcagent.mind.config import CONFIGS, MindConfig
from arcagent.mind.runlog import RunLog
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.synthesizer import Synthesizer
from arcagent.synth.run import Chat, http_chat
from arcagent.synth.trace import ProbePolicy

MODES = ("auto", "offline", "competition")
POLICIES = ("probe", "explore", "mind")
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
DEFAULT_ENDPOINT = "http://127.0.0.1:1234/v1"
# Must match SERVED_NAME in scripts/kaggle_synth_notebook.py (the name vLLM serves).
DEFAULT_MODEL = "vrfai/Qwen3.6-27B-FP8"
DEFAULT_WORKERS = 8
MODEL_WAIT_S = 3600.0

_LOG = logging.getLogger("arcagent.agent")


def resolve_mode(mode: str, environ: Mapping[str, str] = os.environ) -> str:
    """``auto`` -> competition inside a Kaggle rerun, offline otherwise."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode != "auto":
        return mode
    flag = environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
    return "competition" if flag in {"1", "true"} else "offline"


def _poll_until_answers(
    url: str,
    *,
    what: str,
    timeout_s: float,
    poll_s: float,
    fetch: Callable[[str], Any],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> float:
    """GET ``url`` every ``poll_s`` until it answers; the seconds waited, or ``TimeoutError``."""
    t0 = clock()
    last = ""
    while clock() - t0 < timeout_s:
        try:
            with fetch(url):
                return clock() - t0
        except Exception as exc:  # any failure means "not yet"
            last = repr(exc)
        sleep(poll_s)
    raise TimeoutError(f"{what} did not answer in {timeout_s:.0f}s: {last}")


def _urlopen(url: str) -> Any:
    return urlopen(url, timeout=10)


def wait_for_gateway(
    base_url: str,
    timeout_s: float = GATEWAY_WAIT_S,
    *,
    poll_s: float = GATEWAY_POLL_S,
    fetch: Callable[[str], Any] = _urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Poll ``<base_url>/api/games`` until it answers; return the seconds waited.

    Raises ``TimeoutError`` with the last error after ``timeout_s``.
    """
    return _poll_until_answers(
        f"{base_url}/api/games",
        what=f"gateway at {base_url}",
        timeout_s=timeout_s,
        poll_s=poll_s,
        fetch=fetch,
        clock=clock,
        sleep=sleep,
    )


def wait_for_model(
    endpoint: str,
    timeout_s: float = MODEL_WAIT_S,
    *,
    poll_s: float = GATEWAY_POLL_S,
    fetch: Callable[[str], Any] = _urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Poll ``<endpoint>/models`` until the model server answers; return the seconds waited.

    vLLM answers ``/models`` only once the weights are loaded, so a pass means
    synthesis requests can be served. Raises ``TimeoutError`` naming the
    endpoint and the last error after ``timeout_s``.
    """
    base = endpoint.rstrip("/")
    return _poll_until_answers(
        f"{base}/models",
        what=f"model server at {base}",
        timeout_s=timeout_s,
        poll_s=poll_s,
        fetch=fetch,
        clock=clock,
        sleep=sleep,
    )


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
    """A fresh ``probe`` or ``explore`` policy for one game; raises ``ValueError`` otherwise.

    ``mind`` is not built here: it needs shared run resources (``MindRun``).
    """
    if name == "probe":
        return ProbePolicy(seed)
    if name == "explore":
        return ExplorerPolicy(seed)
    raise ValueError(f"unknown policy {name!r}; choose 'probe' or 'explore'")


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


class MindGame:
    """One game's ``MindAgent`` plus the run log it writes; ``close`` releases both."""

    def __init__(self, agent: MindAgent, log: RunLog) -> None:
        self.agent = agent
        self._log = log

    def choose(self, obs: Observation) -> Action:
        return self.agent.choose(obs)

    def close(self) -> None:
        """Close the agent (it logs ``game_end``), then the log, even if the agent fails."""
        try:
            self.agent.close()
        finally:
            self._log.close()


class MindRun:
    """The resources every game of one mind run shares, and the per-game factory.

    Shared: one ``Synthesizer`` (none for a config without the LLM), one
    synthesis thread pool of ``workers`` threads so each running game can have
    a call in flight, and one ``ActivityClock``. Per game: a fresh
    ``ProgramSandbox`` and a ``RunLog`` at ``<log_dir>/<game>.jsonl``. A game's
    seed is ``seed`` plus its index in ``games``. Call ``close`` once every
    game is done.
    """

    def __init__(
        self,
        cfg: MindConfig,
        *,
        chat: Chat | None,
        games: Sequence[str],
        workers: int,
        log_dir: Path,
        seed: int,
        run_deadline: float,
    ) -> None:
        if cfg.llm and chat is None:
            raise ValueError(f"config {cfg.name} uses the LLM but no chat was given")
        self.cfg = cfg
        self.synth = Synthesizer(chat, k=cfg.k) if cfg.llm and chat is not None else None
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="synth")
        self.activity = ActivityClock()
        self._index = {g: i for i, g in enumerate(games)}
        self._log_dir = log_dir
        self._seed = seed
        self._run_deadline = run_deadline

    def make_agent(self, game: str, deadline: float) -> MindGame:
        """A fresh agent for ``game`` that stops thinking at ``deadline``."""
        seed = self._seed + self._index.get(game, 0)
        log = RunLog(self._log_dir / f"{game}.jsonl", game, self.cfg.name, seed)
        sandbox: ProgramSandbox | None = None
        try:
            sandbox = ProgramSandbox()
            agent = MindAgent(
                game,
                self.cfg,
                seed=seed,
                synth=self.synth,
                pool=self.pool,
                sandbox=sandbox,
                log=log,
                activity=self.activity,
                game_deadline=deadline,
                run_deadline=self._run_deadline,
            )
        except BaseException:
            if sandbox is not None:
                sandbox.close()
            log.close()
            raise
        return MindGame(agent, log)

    def close(self) -> None:
        """Cancel queued synthesis calls; a call already running finishes on its own thread."""
        self.pool.shutdown(wait=False, cancel_futures=True)


def mind_chat(args: argparse.Namespace) -> Chat:
    """The model client: recorded replies with ``--replay``, else HTTP with every reply saved."""
    replies = Path(args.replies_dir)
    if args.replay:
        return replay_chat(replies)
    return recording_chat(http_chat(args.endpoint, args.model), replies)


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
    p.add_argument(
        "--max-actions", type=int, default=None, help="default 20000 for mind, 300 otherwise"
    )
    p.add_argument("--game-seconds", type=float, default=DEFAULT_GAME_SECONDS)
    p.add_argument("--total-seconds", type=float, default=DEFAULT_TOTAL_SECONDS)
    p.add_argument("--out", default="bench/agent")
    p.add_argument("--config", choices=sorted(CONFIGS), default="FULL", help="mind config")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible model server")
    p.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="games at once")
    p.add_argument("--k", type=int, default=None, help="programs per call; default the config's")
    p.add_argument("--log-dir", default=None, help="default <out>/logs")
    p.add_argument("--replies-dir", default=None, help="default <out>/replies")
    p.add_argument("--replay", action="store_true", help="answer from recorded replies")
    args = p.parse_args(argv)
    if args.max_actions is None:
        args.max_actions = MIND_MAX_ACTIONS if args.policy == "mind" else DEFAULT_MAX_ACTIONS
    if args.max_actions < 1 or args.game_seconds <= 0 or args.total_seconds <= 0:
        p.error("need --max-actions >= 1 and positive --game-seconds / --total-seconds")
    if args.workers < 1 or (args.k is not None and args.k < 1):
        p.error("need --workers >= 1 and --k >= 1")
    out = Path(args.out)
    args.log_dir = args.log_dir or str(out / "logs")
    args.replies_dir = args.replies_dir or str(out / "replies")
    return args


def play_mind(
    arcade: Any, games: Sequence[str], args: argparse.Namespace
) -> tuple[list[GameResult], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    """Run ``games`` with the mind policy; also returns the run's config record.

    The model wait comes first so the scorecard opens only once the server can
    answer. Raises ``TimeoutError`` from ``wait_for_model`` without opening it.
    """
    cfg = CONFIGS[args.config]
    if args.k is not None:
        cfg = replace(cfg, k=args.k)
    waits = cfg.llm and not args.replay
    waited = wait_for_model(args.endpoint) if waits else 0.0
    if waits:
        _LOG.info("model server %s answered after %.0fs", args.endpoint, waited)
    run = MindRun(
        cfg,
        chat=mind_chat(args) if cfg.llm else None,
        games=games,
        workers=args.workers,
        log_dir=Path(args.log_dir),
        seed=args.seed,
        run_deadline=time.monotonic() + args.total_seconds,
    )
    try:
        results, errors, scorecard = play_concurrent(
            arcade,
            games,
            workers=args.workers,
            make_agent=run.make_agent,
            total_seconds=args.total_seconds,
            max_actions=args.max_actions,
            tags=MIND_TAGS,
        )
    finally:
        run.close()
    record = {
        "mind_config": cfg.name,
        "workers": args.workers,
        "k": cfg.k if cfg.llm else 0,
        "endpoint": args.endpoint,
        "model": args.model,
        "replay": bool(args.replay),
        "model_wait_seconds": round(waited, 1),
    }
    return results, errors, scorecard, record


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
    print(f"mode={mode} policy={args.policy} games={len(games)} max_actions={args.max_actions}")
    extra: dict[str, Any] = {}
    if args.policy == "mind":
        results, errors, scorecard, extra = play_mind(arcade, games, args)
    else:
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
            **extra,
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
