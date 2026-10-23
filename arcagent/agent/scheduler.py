"""Play many games at once under one scorecard.

Intent: the mind policy spends most of a game's wall time waiting on the model
server, so games run side by side on a thread pool and share one competition
scorecard. The model server batches their requests.

Contract of ``play_concurrent``:
- One ``open_scorecard`` before any game and one ``close_scorecard`` after
  every game, even when every game fails or the pool raises.
- At most ``workers`` games run at once; they start in the order given. Each
  game's deadline is fixed when it starts, from ``game_budget`` with the games
  not yet started (this one included), and ``play_game`` enforces it and
  ``max_actions``.
- A game that would start at or past the run deadline is recorded with
  ``stop = "skipped"`` and never built.
- Per game: ``arcade.make``, then ``make_agent(game, deadline)``, then
  ``play_game``. An exception from any of them is that game's ``error`` (the
  run goes on); an agent that was built is closed in a ``finally``. A failing
  ``close`` is recorded as an error and keeps the game's result.
- Results come back in the order of ``games``; errors sorted the same way.
- Bounded wait: the games get until ``run_deadline + hang_grace_s``. A game
  still running then (a stuck gateway call, ``choose`` or ``close``) is
  recorded with ``stop = "hung"`` plus an error, a game that never started is
  ``skipped``, and the scorecard is closed anyway. Hung threads are abandoned,
  never joined, so the caller should end the process when it is done.

Threading: ``make_agent`` and ``report`` are called from pool threads; each
agent is used only by the thread that built it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Protocol

from arcagent.agent.play import GameResult, play_game
from arcagent.core.types import Action, Observation

MIND_TAGS = ("agent", "arcagent-mind")
DEFAULT_MAX_ACTIONS = 20_000
RESERVE_S = 300.0  # left at the end of the run to close the scorecard and write outputs
MIN_GAME_S = 120.0
HANG_GRACE_S = 120.0  # past the run deadline before an unfinished game counts as hung


class GamePolicy(Protocol):
    """A per-game ``Policy`` that holds resources: ``close`` releases them."""

    def choose(self, obs: Observation) -> Action: ...

    def close(self) -> None: ...


def game_budget(
    now: float,
    run_deadline: float,
    games_left: int,
    workers: int,
    *,
    reserve_s: float = RESERVE_S,
    min_s: float = MIN_GAME_S,
) -> float:
    """Seconds a game starting at ``now`` may play.

    The time left before the reserve, spread over the games still to start as
    if ``workers`` of them run side by side, and never more than that time
    itself, so the reserve holds. The ``min_s`` floor may eat into the reserve
    but never reaches past ``run_deadline``. Never negative.
    """
    usable = run_deadline - reserve_s - now
    share = min(usable * workers / max(games_left, 1), usable)
    return max(0.0, min(max(min_s, share), run_deadline - now))


def _error(game_id: str | None, exc: BaseException, prefix: str = "") -> dict[str, Any]:
    return {"game_id": game_id, "error": f"{prefix}{type(exc).__name__}: {exc}"}


def _play_one(  # noqa: PLR0917  # private helper, every argument is required
    arcade: Any,
    card: Any,
    game_id: str,
    deadline: float,
    make_agent: Callable[[str, float], GamePolicy],
    max_actions: int,
    clock: Callable[[], float],
) -> tuple[GameResult, list[dict[str, Any]], str]:
    """Make, build, play and close one game; its result, errors and report line."""
    errors: list[dict[str, Any]] = []
    agent: GamePolicy | None = None
    try:
        env = arcade.make(game_id, scorecard_id=card, save_recording=False)
        if env is None:
            raise RuntimeError("arcade.make returned None")
        agent = make_agent(game_id, deadline)
        r = play_game(env, agent, game_id, max_actions=max_actions, deadline=deadline, clock=clock)
    except Exception as exc:  # one game must not end the run
        errors.append(_error(game_id, exc))
        r = GameResult(game_id, 0, 0, 0, "NOT_PLAYED", 0.0, "error")
        line = f"{game_id}: error {type(exc).__name__}: {exc}"
    else:
        line = (
            f"{game_id}: {r.levels_completed}/{r.win_levels} levels, {r.actions} actions, "
            f"{r.seconds:.1f}s, stop={r.stop}"
        )
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception as exc:  # the game is over; keep its result
                errors.append(_error(game_id, exc, "close: "))
    return r, errors, line


def play_concurrent(
    arcade: Any,
    games: Sequence[str],
    *,
    workers: int,
    make_agent: Callable[[str, float], GamePolicy],
    total_seconds: float,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    run_deadline: float | None = None,
    hang_grace_s: float = HANG_GRACE_S,
    tags: Sequence[str] = MIND_TAGS,
    clock: Callable[[], float] = time.monotonic,
    report: Callable[[str], None] = print,
) -> tuple[list[GameResult], list[dict[str, Any]], dict[str, Any] | None]:
    """Play ``games`` on ``workers`` threads under one scorecard; see the module docstring.

    ``run_deadline`` (a ``clock`` value) overrides ``clock() + total_seconds``
    so the caller can anchor the budget at its own start. ``clock`` must tick
    in seconds: the hang wait converts its remainder to real time.

    Returns the results, the per-game errors and the closed scorecard (None if
    closing failed, which is then an error with ``game_id`` None). Raises
    ``ValueError`` for ``workers < 1`` or ``max_actions < 1`` before opening
    anything.
    """
    # Imported here: submit imports this module.
    from arcagent.agent.submit import scorecard_dict  # noqa: PLC0415

    if workers < 1 or max_actions < 1:
        raise ValueError(f"need workers >= 1 and max_actions >= 1, got {workers}, {max_actions}")
    deadline_at = clock() + total_seconds if run_deadline is None else run_deadline
    results: list[GameResult | None] = [None] * len(games)
    errors: list[list[dict[str, Any]]] = [[] for _ in games]
    lock = threading.Lock()
    started = [0]

    def run_one(i: int, game_id: str) -> None:
        with lock:
            games_left = len(games) - started[0]
            started[0] += 1
        now = clock()
        if now >= deadline_at:
            results[i] = GameResult(game_id, 0, 0, 0, "NOT_PLAYED", 0.0, "skipped")
            line = f"{game_id}: skipped, run budget spent"
        else:
            deadline = now + game_budget(now, deadline_at, games_left, workers)
            results[i], errors[i], line = _play_one(
                arcade, card, game_id, deadline, make_agent, max_actions, clock
            )
        with lock:
            report(line)

    card = arcade.open_scorecard(tags=list(tags))
    scorecard: dict[str, Any] | None = None
    tail: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="game")
    try:
        futures = [pool.submit(run_one, i, g) for i, g in enumerate(games)]
        _, pending = wait(futures, timeout=max(0.0, deadline_at - clock()) + hang_grace_s)
        for i, f in enumerate(futures):
            if f not in pending:
                f.result()  # run_one catches every Exception; this re-raises only bugs
            elif f.cancel():
                results[i] = GameResult(games[i], 0, 0, 0, "NOT_PLAYED", 0.0, "skipped")
            else:
                results[i] = GameResult(games[i], 0, 0, 0, "NOT_PLAYED", 0.0, "hung")
                errors[i] = [{"game_id": games[i], "error": "hung: still running at the deadline"}]
                with lock:
                    report(f"{games[i]}: hung, abandoned after the run deadline")
    finally:
        # Never join: a hung game thread would block the scorecard close forever.
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            scorecard = scorecard_dict(arcade.close_scorecard(card))
        except Exception as exc:  # report, keep the results
            tail.append({"game_id": None, "error": f"close_scorecard: {exc!r}"})
    done = [
        r if r is not None else GameResult(g, 0, 0, 0, "NOT_PLAYED", 0.0, "error")
        for r, g in zip(results, games, strict=True)
    ]
    return done, [e for per in errors for e in per] + tail, scorecard
