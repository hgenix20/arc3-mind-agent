"""Keep the competition scorecard alive while the model server loads.

Why: the Kaggle gateway closes a scorecard that has had no move for 15 minutes
(arc_agi ``ScorecardManager``, ``STALE_MINUTES`` = 15) and the run ends with
the score so far; the Duck author confirmed the rule on the competition forum
(discussion 750173). Franzen's harness spends one RESET on one game at the
12-minute mark and the next move is the first model action, so a server that
takes more than 15 minutes from that point ends the run at 0.00. That is what
happened to submission 57457365 (duck v1, 2026-10-21).

What: ``install`` patches ``_HarnessGameSession`` in place so the warm-up game
repeats its RESET every ``period_s`` seconds until any game executes a
model-chosen action, then plays normally. Each RESET costs that one game one
action on its first level.

Contract: ``session_cls`` must provide ``_play_inner()``,
``_execute_action(action, *, automatic=..., **kw)``, ``_execute_auto_reset()``,
``should_stop()``, ``solver._claim_warmup_slot()`` and
``game.game_run.game_id``. Installing twice on one class is a no-op. The file
is copied verbatim into the Kaggle notebook by ``scripts/kaggle_duck_notebook.py``,
so it uses the standard library only and has no ``from __future__`` import
(it runs after other statements in the cell).
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger("duck_keepalive")

PERIOD_S = 300.0
MAX_WAIT_S = 3 * 3600.0


def install(
    session_cls: Any,
    *,
    period_s: float = PERIOD_S,
    max_wait_s: float = MAX_WAIT_S,
    clock: Callable[[], float] = time.monotonic,
) -> threading.Event:
    """Wrap the session class; returns the event set by the first model-chosen action."""
    if getattr(session_cls, "_duck_keepalive_installed", False):
        return session_cls._duck_keepalive_event
    first_model_action = threading.Event()
    orig_execute = session_cls._execute_action
    orig_play_inner = session_cls._play_inner

    def execute_action(self: Any, action: Any, **kw: Any) -> Any:
        result = orig_execute(self, action, **kw)
        if not kw.get("automatic", False):
            first_model_action.set()
        return result

    def play_inner(self: Any) -> None:
        if self.solver._claim_warmup_slot():
            game_id = self.game.game_run.game_id
            started = clock()
            resets = 0
            while True:
                resets += 1
                log.warning("%s: keep-alive RESET %d, no model action yet", game_id, resets)
                self._execute_auto_reset()
                if first_model_action.wait(period_s):
                    break
                if self.should_stop() or clock() - started >= max_wait_s:
                    log.warning("%s: keep-alive stopped after %d RESETs", game_id, resets)
                    break
            # The slot is spent, so the original single warm-up RESET below
            # is skipped.
        orig_play_inner(self)

    session_cls._execute_action = execute_action
    session_cls._play_inner = play_inner
    session_cls._duck_keepalive_installed = True
    session_cls._duck_keepalive_event = first_model_action
    return first_model_action
