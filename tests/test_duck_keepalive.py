"""The keep-alive wrap for Franzen's game session (scripts/duck_keepalive.py)."""

import importlib.util
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODULE = Path(__file__).resolve().parent.parent / "scripts" / "duck_keepalive.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("duck_keepalive", MODULE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeSolver:
    def __init__(self, warmup: int = 1) -> None:
        self.remaining = warmup
        self.lock = threading.Lock()

    def _claim_warmup_slot(self) -> bool:
        with self.lock:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True


def make_session_cls() -> type:
    """A fresh class per test, so each ``install`` starts clean."""

    class Session:
        def __init__(self, solver: FakeSolver, game_id: str, stop_after: int | None = None):
            self.solver = solver
            self.game = SimpleNamespace(game_run=SimpleNamespace(game_id=game_id))
            self.resets = 0
            self.executed: list[tuple[str, bool]] = []
            self.played = 0
            self.stop_after = stop_after

        def should_stop(self) -> bool:
            return self.stop_after is not None and self.resets >= self.stop_after

        def _execute_auto_reset(self) -> None:
            self.resets += 1
            self._execute_action("RESET", batch_index=1, batch_size=1, automatic=True)

        def _execute_action(
            self,
            action: str,
            *,
            batch_index: int,
            batch_size: int,
            generated_tokens: int | None = None,
            automatic: bool = False,
        ) -> dict[str, Any]:
            self.executed.append((action, automatic))
            return {"executed": True, "batch": (batch_index, batch_size, generated_tokens)}

        def _play_inner(self) -> None:
            self.played += 1

    return Session


def test_warmup_game_repeats_reset_until_a_model_action_anywhere():
    mod = load_module()
    session = make_session_cls()
    event = mod.install(session, period_s=0.01)
    solver = FakeSolver()
    warm = session(solver, "aaaa")
    other = session(solver, "bbbb")
    thread = threading.Thread(target=warm._play_inner)
    thread.start()
    time.sleep(0.08)
    assert warm.resets >= 2
    assert warm.played == 0
    other._execute_action("UP", batch_index=1, batch_size=1)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert event.is_set()
    assert warm.played == 1
    assert all(auto for _, auto in warm.executed)


def test_games_without_the_warmup_slot_play_at_once():
    mod = load_module()
    session = make_session_cls()
    mod.install(session, period_s=0.01)
    solver = FakeSolver()
    first = session(solver, "aaaa", stop_after=1)
    second = session(solver, "bbbb")
    first._play_inner()
    second._play_inner()
    assert first.resets == 1
    assert second.resets == 0
    assert second.played == 1


def test_automatic_actions_do_not_count_as_a_model_action():
    mod = load_module()
    session = make_session_cls()
    event = mod.install(session, period_s=0.01)
    solver = FakeSolver()
    warm = session(solver, "aaaa", stop_after=3)
    other = session(solver, "bbbb")
    other._execute_action("RESET", batch_index=1, batch_size=1, automatic=True)
    warm._play_inner()
    assert not event.is_set()
    assert warm.resets == 3
    assert warm.played == 1


def test_keepalive_gives_up_after_max_wait():
    mod = load_module()
    session = make_session_cls()
    ticks = iter(range(0, 100_000, 1000))
    mod.install(session, period_s=0.001, max_wait_s=1500, clock=lambda: float(next(ticks)))
    warm = session(FakeSolver(), "aaaa")
    warm._play_inner()
    assert warm.resets == 2
    assert warm.played == 1


def test_install_twice_is_a_no_op():
    mod = load_module()
    session = make_session_cls()
    first = mod.install(session, period_s=0.01)
    second = mod.install(session, period_s=0.01)
    assert first is second
    warm = session(FakeSolver(), "aaaa", stop_after=1)
    warm._play_inner()
    assert warm.resets == 1
    assert warm.played == 1


def test_source_is_safe_to_append_to_a_notebook_cell():
    src = MODULE.read_text(encoding="utf-8")
    assert not re.search(r"^from __future__ import", src, re.MULTILINE)
    compile("x = 1\n" + src, str(MODULE), "exec")
