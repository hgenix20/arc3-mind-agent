"""The shared toolbox: LLM-patched versions of the planning tools, shared across the run.

Contract:
- ``TOOLS`` names the patchable tools (``search.search``, ``route.route``) and, per tool,
  ``agrees(expected, got)``: whether a candidate's answer is at least as good as a
  recorded one (used to check a fix against regression cases). Perception, wisdom, the
  sandbox, the budget, the scheduler and submit are never patchable.
- ``TOOLBOX`` is one process-wide ``Toolbox`` (games run on threads of one process, so a
  fix adopted in one game reaches every game's next call). Every method takes its lock:
  versions are numbered in adoption order and no update is lost. ``current`` is None
  for the original. ``rollback`` drops the newest version and returns the number now
  current (0 for the original). Each tool keeps its last ``MAX_CASES`` regression cases.
- ``ToolWorker`` is one game's tools process (``_tool_worker.py``). ``call`` and
  ``run_rule`` never raise: they return ``(ok, result or error text)``. A reply later
  than ``timeout_s`` or a dead process kills the worker, counts a restart, and starts a
  new one on the next request.
- ``call_tool`` is how agents call a patchable tool: the original in-process when no
  version is adopted (or no worker is given), else the worker; a failed worker call
  answers with the original. It returns ``(result, version, error)`` with ``version``
  0 for the original and ``error`` the worker's failure text or "".
- ``SceneActions`` is a picklable stand-in for ``FluidAgent._actions`` so search's
  arguments can cross to the worker.
"""

from __future__ import annotations

import base64
import contextlib
import importlib
import json
import pickle
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, Action
from arcagent.mind.objects import Cell, Scene, Thing, click_cell

MAX_CASES = 8  # regression cases kept per tool
CALL_TIMEOUT_S = 5.0  # default per-call deadline in the worker
SPAWN_TIMEOUT_S = 30.0  # interpreter start plus the first imports
WORKER_PATH = Path(__file__).with_name("_tool_worker.py")


def _search_agrees(expected: Any, got: Any) -> bool:
    """A search answer agrees when it finds a plan wherever the recorded one did, no longer."""
    if getattr(expected, "stop", None) != "found":
        return True
    return getattr(got, "stop", None) == "found" and len(got.plan) <= len(expected.plan)


def _route_agrees(expected: Any, got: Any) -> bool:
    """A route agrees when it exists wherever the recorded one did, no longer."""
    if expected is None:
        return True
    return got is not None and got[1] <= expected[1]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    module: str
    func: str
    agrees: Callable[[Any, Any], bool]

    def original(self) -> Callable[..., Any]:
        return getattr(importlib.import_module(self.module), self.func)


TOOLS: dict[str, ToolSpec] = {
    "search": ToolSpec("search", "arcagent.mind.search", "search", _search_agrees),
    "route": ToolSpec("route", "arcagent.mind.route", "route", _route_agrees),
}


@dataclass(frozen=True, slots=True)
class ToolVersion:
    no: int
    source: str
    game: str
    reason: str
    at: float


@dataclass(frozen=True, slots=True)
class Case:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any


@dataclass
class Toolbox:
    """Adopted versions per tool (newest last), the regression cases, and the history."""

    _versions: dict[str, list[ToolVersion]] = field(default_factory=dict[str, list[ToolVersion]])
    _cases: dict[str, deque[Case]] = field(default_factory=dict[str, deque[Case]])
    _history: list[tuple[str, str, int, str]] = field(
        default_factory=list[tuple[str, str, int, str]]
    )
    _next: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def current(self, name: str) -> ToolVersion | None:
        with self._lock:
            stack = self._versions.get(name)
            return stack[-1] if stack else None

    def adopt(self, name: str, source: str, game: str, reason: str) -> int:
        with self._lock:
            no = self._next
            self._next += 1
            v = ToolVersion(no, source, game, reason, time.time())
            self._versions.setdefault(name, []).append(v)
            self._history.append(("adopt", name, no, reason))
            return no

    def rollback(self, name: str, reason: str) -> int:
        with self._lock:
            stack = self._versions.get(name, [])
            if stack:
                gone = stack.pop()
                self._history.append(("rollback", name, gone.no, reason))
            return stack[-1].no if stack else 0

    def add_case(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> None:
        with self._lock:
            self._cases.setdefault(name, deque(maxlen=MAX_CASES)).append(Case(args, kwargs, result))

    def cases(self, name: str) -> list[Case]:
        with self._lock:
            return list(self._cases.get(name, ()))

    def history(self) -> list[tuple[str, str, int, str]]:
        with self._lock:
            return list(self._history)

    def clear(self) -> None:
        with self._lock:
            self._versions.clear()
            self._cases.clear()
            self._history.clear()
            self._next = 1


TOOLBOX = Toolbox()


def _pump(stream: IO[bytes], q: queue.Queue[bytes | None]) -> None:
    try:
        for line in stream:
            q.put(line)
    except (OSError, ValueError):
        pass
    q.put(None)


def _dump(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


class ToolWorker:
    """One game's tools process; see the module docstring."""

    def __init__(self) -> None:
        self.restarts = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False

    def _spawn(self) -> None:
        self._q = queue.Queue()
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert self._proc.stdout is not None
        threading.Thread(target=_pump, args=(self._proc.stdout, self._q), daemon=True).start()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)

    def _request(self, req: dict[str, Any], timeout_s: float) -> tuple[bool, Any]:  # noqa: PLR0911  (one return per failure kind)
        with self._lock:
            if self._closed:
                return False, "worker closed"
            first = self._proc is None
            try:
                if first:
                    self._spawn()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
                wait = timeout_s + (SPAWN_TIMEOUT_S if first else 0.0)
                line = self._q.get(timeout=wait)
            except queue.Empty:
                self.restarts += 1
                self._kill()
                return False, "timeout"
            except OSError as exc:
                self.restarts += 1
                self._kill()
                return False, f"worker died: {exc}"
            if line is None:
                self.restarts += 1
                self._kill()
                return False, "crash"
        try:
            reply = json.loads(line)
            if not reply.get("ok"):
                return False, str(reply.get("error", "error"))
            return True, pickle.loads(base64.b64decode(reply["result"]))
        except Exception as exc:  # a garbled reply costs the call only
            return False, f"bad reply: {type(exc).__name__}"

    def call(
        self,
        spec: ToolSpec,
        version: ToolVersion,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout_s: float = CALL_TIMEOUT_S,
    ) -> tuple[bool, Any]:
        req = {
            "op": "call",
            "name": spec.name,
            "version": version.no,
            "module": spec.module,
            "func": spec.func,
            "source": version.source,
            "args": _dump((args, kwargs)),
        }
        return self._request(req, timeout_s)

    def run_rule(
        self, source: str, objects: list[dict[str, Any]], action: dict[str, int], timeout_s: float
    ) -> tuple[bool, Any]:
        return self._request(
            {"op": "rule", "source": source, "args": _dump(((objects, action), {}))}, timeout_s
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._kill()


def call_tool(
    name: str,
    worker: ToolWorker | None,
    *args: Any,
    box: Toolbox = TOOLBOX,
    timeout_s: float = CALL_TIMEOUT_S,
    **kwargs: Any,
) -> tuple[Any, int, str]:
    """Call the newest adopted version of tool ``name``; see the module docstring."""
    spec = TOOLS[name]
    version = box.current(name)
    if version is not None and worker is not None:
        ok, got = worker.call(spec, version, args, kwargs, timeout_s)
        if ok:
            return got, version.no, ""
        return spec.original()(*args, **kwargs), 0, str(got)
    return spec.original()(*args, **kwargs), 0, ""


@dataclass
class SceneActions:
    """Search's actions for a scene: the simple ones, then one click per on-board thing."""

    simple: tuple[Action, ...]
    can_click: bool
    offsets: dict[str, Cell] = field(default_factory=dict[str, Cell])

    def cell(self, t: Thing) -> Cell:
        off = self.offsets.get(t.shape)
        if off is None:
            x, y = click_cell(t)
            off = (x - t.x, y - t.y)
            self.offsets[t.shape] = off
        return t.x + off[0], t.y + off[1]

    def __call__(self, scene: Scene) -> list[Action]:
        out = list(self.simple)
        if self.can_click:
            for t in scene.things:
                x, y = self.cell(t)
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    out.append(Action(COORD_ACTION_ID, x, y))
        return out
