"""Parent side of the program sandbox: one worker subprocess per game.

Model-written rule programs (``step(grid, action, x, y)`` and an optional
``goal(grid)``) never run in the agent's process. ``ProgramSandbox`` starts
``python -I _worker.py PLANNER_PATH`` and talks to it over stdin/stdout JSON
lines; the worker holds every loaded program and also runs the planner on one,
so a search costs one round trip.

Contract:
- Every request has a deadline (``timeout_s``, or ``plan_timeout_s`` for
  ``plan``). Replies are read by a thread into a queue, because Windows pipes
  do not support ``select``.
- On a deadline or a dead worker, the process is killed, ``restarts`` goes up
  by one, and a new worker is started with every loaded program reloaded except
  the one to blame, which is unloaded and recorded in ``dropped`` with reason
  ``timeout`` (or ``crash`` when the worker died). When one request covered
  several programs the culprit is unknown, so the request is re-run one
  program at a time and only the programs that fail alone are dropped. Each
  kill counts, so a batch with one hang costs two restarts.
- Program misbehavior is a value, never an exception: ``None`` for a crash, a
  timeout or an invalid grid (anything but 64 rows of 64 ints in -1..15).
- Failing to start the worker at all raises ``OSError``: that is a broken
  environment, not model behavior.
- Thread-safe: every public method holds one re-entrant lock, because the
  game's thread and its synthesis call (on a pool thread) share the sandbox.
  Without it two requests on the one pipe could take each other's replies.

Isolation is best-effort (restricted builtins and imports, reduced
environment, a 2 GB address-space cap on Linux), not a security boundary.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal

import numpy as np

import arcagent.plan.planner as _planner
from arcagent.core.types import GRID_SIZE, Action, Grid
from arcagent.mind.perception import frame_hash
from arcagent.mind.types import ActionKey, Step

WORKER_PATH = Path(__file__).with_name("_worker.py")
PLANNER_PATH = Path(_planner.__file__)
PLAN_LIMIT_FRACTION = 0.8  # leave the hard deadline headroom over the search's own limit
KILL_WAIT_S = 5.0
SCORE_CHUNK = 200  # steps per score request: ~0.85 ms/step must fit the per-request deadline

Reply = dict[str, Any]


@dataclass(frozen=True, slots=True)
class SandboxPlan:
    """A planner result from the worker. ``actions`` is None when no plan was found.

    ``stopped`` is one of the planner's reasons (``goal``, ``goal_at_start``,
    ``exhausted``, ``max_nodes``, ``time_limit``) or a sandbox one:
    ``timeout``, ``crash``, ``not_loaded``, ``no_goal``, ``no_planner``.
    """

    actions: tuple[Action, ...] | None
    stopped: str
    expanded: int
    seconds: float
    errors: int


def _rows(grid: Grid) -> list[list[int]]:
    return grid.astype(int).tolist()


def _wire(a: Action) -> list[int]:
    return [a.id, a.x, a.y]


def _pump(stream: IO[bytes], q: queue.Queue[bytes | None]) -> None:
    """Forward lines until EOF, then a None sentinel so a dead worker is seen at once."""
    try:
        for line in stream:
            q.put(line)
    except (OSError, ValueError):
        pass  # pipe closed by a kill
    q.put(None)


class ProgramSandbox:
    """A persistent worker subprocess holding one game's programs. Use as a context manager."""

    def __init__(self, *, timeout_s: float = 5.0, plan_timeout_s: float = 30.0) -> None:
        self.timeout_s = timeout_s
        self.plan_timeout_s = plan_timeout_s
        self.restarts = 0
        self.dropped: dict[str, str] = {}
        self._codes: dict[str, str] = {}
        self._tmp = tempfile.TemporaryDirectory(prefix="arc-sandbox-")
        self._proc: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._next_id = 0
        self._lock = threading.RLock()
        self._spawn()

    def __enter__(self) -> ProgramSandbox:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop the worker and remove its working directory. Safe to call twice."""
        with self._lock:
            self._kill()
            self._tmp.cleanup()

    # -- process management ------------------------------------------------

    def _spawn(self) -> None:
        """Start a worker and reload every program; a program that fails to reload is dropped."""
        env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        cmd = [sys.executable, "-I", str(WORKER_PATH), str(PLANNER_PATH)]
        self._proc = subprocess.Popen(  # fixed argv, no shell
            cmd,
            cwd=self._tmp.name,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert self._proc.stdout is not None  # set by stdout=PIPE
        self._queue = queue.Queue()
        threading.Thread(target=_pump, args=(self._proc.stdout, self._queue), daemon=True).start()
        for pid, code in list(self._codes.items()):
            reply = self._roundtrip({"op": "load", "pid": pid, "code": code}, self.timeout_s, pid)
            if reply is None:
                return  # the failure path dropped pid and spawned a worker with the rest
            if "error" in reply:
                del self._codes[pid]
                self.dropped[pid] = "reload_error"

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.kill()
        # on TimeoutExpired the OS reaps it; nothing more we can do from here
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(KILL_WAIT_S)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                stream.close()

    def _roundtrip(self, req: Reply, timeout_s: float, blame: str | None) -> Reply | None:
        """Send ``req`` and wait for its reply until ``timeout_s``.

        On a deadline or a dead worker: kill, count a restart, drop ``blame``
        (if given) with the reason, start a fresh worker, and return None.
        """
        if self._proc is None:
            self._spawn()
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        self._next_id += 1
        rid = self._next_id
        reason = "crash"
        try:
            proc.stdin.write(json.dumps({**req, "id": rid}).encode("utf-8") + b"\n")
            proc.stdin.flush()
            deadline = time.monotonic() + timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "timeout"
                    break
                try:
                    line = self._queue.get(timeout=remaining)
                except queue.Empty:
                    reason = "timeout"
                    break
                if line is None:
                    break  # EOF: the worker died
                reply: Reply = json.loads(line)
                if reply.get("id") == rid:
                    return reply
        except (OSError, ValueError):
            pass  # broken pipe or a garbled line: treat as a dead worker
        self._kill()
        self.restarts += 1
        if blame is not None:
            self._codes.pop(blame, None)
            self.dropped[blame] = reason
        self._spawn()
        return None

    def _multi(self, req: Reply, pids: Sequence[str]) -> Reply:
        """Run a request over the loaded ``pids``; the per-pid ``results`` that came back.

        A multi-program failure is re-run one program at a time so only the
        culprit is dropped.
        """
        live = [p for p in dict.fromkeys(pids) if p in self._codes]
        if not live:
            return {}
        blame = live[0] if len(live) == 1 else None
        reply = self._roundtrip({**req, "pids": live}, self.timeout_s, blame)
        if reply is not None:
            return reply.get("results") or {}
        out: Reply = {}
        if blame is None:
            for p in live:
                one = self._roundtrip({**req, "pids": [p]}, self.timeout_s, p)
                if one is not None:
                    out.update(one.get("results") or {})
        return out

    # -- public API --------------------------------------------------------

    def load(self, pid: str, code: str) -> str | None:
        """Load (or replace) a program. Returns the load error, ``"timeout"``, or None."""
        with self._lock:
            self._codes.pop(pid, None)
            self.dropped.pop(pid, None)
            reply = self._roundtrip({"op": "load", "pid": pid, "code": code}, self.timeout_s, None)
            if reply is None:
                return "timeout"
            if "error" in reply:
                return str(reply["error"])
            self._codes[pid] = code
            return None

    def unload(self, pid: str) -> None:
        """Forget a program; unknown ids are ignored."""
        with self._lock:
            if self._codes.pop(pid, None) is not None:
                self._roundtrip({"op": "unload", "pid": pid}, self.timeout_s, None)

    def predict_many(
        self, pids: Sequence[str], grid: Grid, actions: Sequence[Action]
    ) -> dict[tuple[str, ActionKey], str | None]:
        """Each program's predicted frame hash for each action on ``grid``; None is a miss."""
        with self._lock:
            req: Reply = {
                "op": "predict",
                "grid": _rows(grid),
                "actions": [_wire(a) for a in actions],
            }
            results = self._multi(req, pids)
            out: dict[tuple[str, ActionKey], str | None] = {}
            for pid in pids:
                hashes = results.get(pid)
                for i, a in enumerate(actions):
                    h = hashes[i] if isinstance(hashes, list) and i < len(hashes) else None
                    out[(pid, a.key())] = h if isinstance(h, str) else None
            return out

    def predict(self, pids: Sequence[str], grid: Grid, action: Action) -> dict[str, str | None]:
        """Each program's predicted frame hash for ``action`` on ``grid``; None is a miss."""
        many = self.predict_many(pids, grid, [action])
        return {pid: many[(pid, action.key())] for pid in pids}

    def predict_grid(self, pid: str, grid: Grid, action: Action) -> Grid | None:
        """The predicted next grid, or None on a crash, timeout or invalid output."""
        with self._lock:
            req: Reply = {
                "op": "predict",
                "grid": _rows(grid),
                "actions": [_wire(action)],
                "want_grid": True,
            }
            got = self._multi(req, [pid]).get(pid)
            if not isinstance(got, list) or not got or got[0] is None:
                return None
            arr = np.asarray(got[0], dtype=np.int8)
            return arr if arr.shape == (GRID_SIZE, GRID_SIZE) else None

    def goal(self, pids: Sequence[str], grid: Grid) -> dict[str, bool | None]:
        """Whether each program's ``goal`` holds on ``grid``; None if it has none or failed."""
        with self._lock:
            results = self._multi({"op": "goal", "grid": _rows(grid)}, pids)
            out: dict[str, bool | None] = {}
            for pid in pids:
                v = results.get(pid)
                out[pid] = v if isinstance(v, bool) else None
            return out

    def score(self, pid: str, steps: Sequence[Step]) -> tuple[int, int]:
        """``(exact_hits, trials)``: how many ``steps`` the program predicts cell for cell.

        Steps go to the worker in chunks of ``SCORE_CHUNK`` so each request gets its own
        deadline and a long history cannot time out a correct program. A chunk that fails
        (timeout, crash, unknown pid) stops scoring and the result is ``(0, len(steps))``,
        the same as a single failed request; a timeout also drops the program as usual.
        """
        with self._lock:
            if not steps:
                return (0, 0)
            total = 0
            for lo in range(0, len(steps), SCORE_CHUNK):
                wire = [
                    [_rows(s.before), _wire(s.action), frame_hash(s.after)]
                    for s in steps[lo : lo + SCORE_CHUNK]
                ]
                hits = self._multi({"op": "score", "steps": wire}, [pid]).get(pid)
                if not isinstance(hits, int):
                    return (0, len(steps))
                total += hits
            return (total, len(steps))

    def plan(
        self,
        pid: str,
        grid: Grid,
        actions: Sequence[Action],
        *,
        target: Literal["goal", "novel"],
        visited: frozenset[str],
        max_nodes: int,
        time_limit_s: float,
    ) -> SandboxPlan:
        """Shortest action list under program ``pid`` from ``grid`` to the target.

        ``target="goal"`` searches for the program's own ``goal``;
        ``target="novel"`` for any frame whose hash is not in ``visited``. The
        search limit is capped below ``plan_timeout_s`` so a slow but healthy
        search stops itself before the hard deadline drops the program.
        """
        with self._lock:
            if pid not in self._codes:
                return SandboxPlan(None, "not_loaded", 0, 0.0, 0)
            req: Reply = {
                "op": "plan",
                "pid": pid,
                "grid": _rows(grid),
                "actions": [_wire(a) for a in actions],
                "target": target,
                "visited": sorted(visited),
                "max_nodes": max_nodes,
                "time_limit_s": min(time_limit_s, self.plan_timeout_s * PLAN_LIMIT_FRACTION),
            }
            t0 = time.monotonic()
            reply = self._roundtrip(req, self.plan_timeout_s, pid)
            if reply is None:
                return SandboxPlan(
                    None, self.dropped.get(pid, "crash"), 0, time.monotonic() - t0, 0
                )
            raw = reply.get("actions")
            acts = (
                None if raw is None else tuple(Action(int(a[0]), int(a[1]), int(a[2])) for a in raw)
            )
            return SandboxPlan(
                actions=acts,
                stopped=str(reply.get("stopped", "crash")),
                expanded=int(reply.get("expanded", 0)),
                seconds=float(reply.get("seconds", 0.0)),
                errors=int(reply.get("errors", 0)),
            )
