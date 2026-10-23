"""Sandbox worker: holds model-written programs and answers requests on stdin.

Run as ``python -I _worker.py PLANNER_PATH``, never imported by the agent for
execution. Standard library only, with no ``from __future__`` import and no
``arcagent`` imports, because ``-I`` leaves the package off the path.

Protocol: one JSON object per line in each direction. Every request carries an
``id`` that the reply echoes. Ops: ``ping``, ``load``, ``unload``, ``predict``, ``goal``,
``score``, ``plan`` (see ``_handle``). Grids travel as 64 lists of 64 ints,
actions as ``[id, x, y]``.

Contract:
- A program is executed with restricted builtins: no ``open``, ``exec``,
  ``eval``, ``compile``, ``input``, ``breakpoint``, ``help``, ``exit`` or
  ``quit``, and ``__import__`` admits only ``ALLOWED_MODULES``. ``socket`` is
  stubbed before any program loads, as in ``arcagent/synth/verify.py``.
- Program output is valid only as 64 rows of 64 plain ints in -1..15. Anything
  else, and any exception (``BaseException`` included), is a miss reported as
  ``null``; the worker never dies on program behavior it can catch.
- ``print`` inside a program goes to the null device, so it cannot corrupt the
  reply channel.
- On Linux the address space is capped at 2 GB. A program that loops forever
  is not caught here: the parent's deadline kills the whole process.

Isolation is best-effort, not a security boundary.
"""

import builtins
import contextlib
import hashlib
import importlib.util
import json
import os
import socket
import sys
from collections.abc import Callable
from typing import Any

N = 64
MEMORY_CAP_BYTES = 2 * 1024**3
ALLOWED_MODULES = frozenset(
    {"math", "itertools", "collections", "functools", "heapq", "copy", "re"}
)
REMOVED_BUILTINS = frozenset(
    {"open", "exec", "eval", "compile", "input", "breakpoint", "help", "exit", "quit"}
)
MAX_ERROR_CHARS = 400

Rows = tuple[tuple[int, ...], ...]


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise OSError("network is disabled in the sandbox")


def _restricted_import(
    name: str,
    globals_: Any = None,
    locals_: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    if level != 0 or name.split(".", maxsplit=1)[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is blocked in the sandbox")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def safe_builtins() -> dict[str, Any]:
    """The builtins a program sees: the real ones minus file, process and eval access."""
    out = {k: v for k, v in vars(builtins).items() if k not in REMOVED_BUILTINS}
    out["__import__"] = _restricted_import
    return out


def frame_hash(rows: Any) -> str:
    """Same value as ``arcagent.mind.perception.frame_hash`` for the equal int8 grid."""
    data = bytes(v & 0xFF for row in rows for v in row)
    return hashlib.sha1(data).hexdigest()[:16]


def valid_grid(g: Any) -> Rows | None:
    """``g`` as a tuple of tuples if it is 64 rows of 64 plain ints in -1..15, else None."""
    if type(g) not in (list, tuple) or len(g) != N:
        return None
    rows: list[tuple[int, ...]] = []
    for r in g:
        if type(r) not in (list, tuple) or len(r) != N:
            return None
        if not all(type(v) is int and -1 <= v <= 15 for v in r):
            return None
        rows.append(tuple(r))
    return tuple(rows)


def as_bool(v: Any) -> bool | None:
    """A goal answer: True/False, or 1/0; anything else is None."""
    if type(v) is bool:
        return v
    if type(v) is int and v in (0, 1):
        return bool(v)
    return None


def _same(s: Rows) -> Rows:
    return s


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


class Program:
    """A loaded program: its ``step`` and optional ``goal``."""

    def __init__(self, step: Callable[..., Any], goal: Callable[..., Any] | None) -> None:
        self.step = step
        self.goal = goal

    def predict(self, rows: Any, action: Any) -> Rows | None:
        """The validated next grid, or None on any exception or invalid output."""
        try:
            got = self.step([list(r) for r in rows], action[0], action[1], action[2])
        except BaseException:  # model-written code: any failure is a miss
            return None
        return valid_grid(got)

    def check_goal(self, rows: Any) -> bool | None:
        if self.goal is None:
            return None
        try:
            return as_bool(self.goal([list(r) for r in rows]))
        except BaseException:  # model-written code
            return None


def load_program(code: str) -> Program:
    """Execute ``code`` with restricted builtins. Raises on any load failure."""
    ns: dict[str, Any] = {"__builtins__": safe_builtins(), "__name__": "program"}
    exec(compile(code, "<program>", "exec"), ns)  # running model code is the sandbox's purpose
    step = ns.get("step")
    if not callable(step):
        raise TypeError("program defines no step() function")
    goal = ns.get("goal")
    return Program(step, goal if callable(goal) else None)


def load_planner(path: str) -> Any:
    """The planner module loaded from ``path``; registered so dataclasses resolve it."""
    spec = importlib.util.spec_from_file_location("sandbox_planner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load planner from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_planner"] = mod
    spec.loader.exec_module(mod)
    return mod


class Worker:
    """Request handlers over the loaded programs."""

    def __init__(self, planner: Any, planner_error: str) -> None:
        self.programs: dict[str, Program] = {}
        self.planner = planner
        self.planner_error = planner_error

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        op = req.get("op")
        if op == "load":
            try:
                self.programs[req["pid"]] = load_program(req["code"])
            except BaseException as exc:  # model-written code
                return {"error": _err(exc)}
            return {"ok": True}
        if op == "unload":
            self.programs.pop(req["pid"], None)
        if op in ("ping", "unload"):
            return {"ok": True}
        batch: dict[str, Callable[[dict[str, Any]], Any]] = {
            "predict": self.predict,
            "goal": self.goal,
            "score": self.score,
        }
        if op in batch:
            return {"results": batch[op](req)}
        if op == "plan":
            return self.plan(req)
        return {"error": f"unknown op {op!r}"}

    def predict(self, req: dict[str, Any]) -> dict[str, list[Any]]:
        grid, actions, want_grid = req["grid"], req["actions"], bool(req.get("want_grid"))
        out: dict[str, list[Any]] = {}
        for pid in req["pids"]:
            prog = self.programs.get(pid)
            res: list[Any] = []
            for a in actions:
                got = prog.predict(grid, a) if prog is not None else None
                if got is None:
                    res.append(None)
                else:
                    res.append([list(r) for r in got] if want_grid else frame_hash(got))
            out[pid] = res
        return out

    def goal(self, req: dict[str, Any]) -> dict[str, bool | None]:
        out: dict[str, bool | None] = {}
        for pid in req["pids"]:
            prog = self.programs.get(pid)
            out[pid] = prog.check_goal(req["grid"]) if prog is not None else None
        return out

    def score(self, req: dict[str, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for pid in req["pids"]:
            prog = self.programs.get(pid)
            hits = 0
            if prog is not None:
                for before, action, after_hash in req["steps"]:
                    got = prog.predict(before, action)
                    hits += got is not None and frame_hash(got) == after_hash
            out[pid] = hits
        return out

    def plan(self, req: dict[str, Any]) -> dict[str, Any]:
        empty: dict[str, Any] = {"actions": None, "expanded": 0, "seconds": 0.0, "errors": 0}
        prog = self.programs.get(req["pid"])
        if prog is None:
            return {**empty, "stopped": "not_loaded"}
        if self.planner is None:
            return {**empty, "stopped": "no_planner", "error": self.planner_error}
        start = valid_grid(req["grid"])
        if start is None:
            return {**empty, "stopped": "bad_start"}
        if req["target"] == "goal":
            if prog.goal is None:
                return {**empty, "stopped": "no_goal"}
            goal_fn = prog.goal

            def at_goal(s: Rows) -> bool:
                try:
                    v = goal_fn([list(r) for r in s])
                except BaseException as exc:  # re-raised as Exception so the planner counts it
                    raise RuntimeError(_err(exc)) from None
                return as_bool(v) is True

        else:
            visited = frozenset(req["visited"])

            def at_goal(s: Rows) -> bool:
                return frame_hash(s) not in visited

        def step(s: Rows, a: tuple[int, int, int]) -> Rows:
            try:
                got = prog.step([list(r) for r in s], a[0], a[1], a[2])
            except BaseException as exc:  # re-raised as Exception so the planner counts it
                raise RuntimeError(_err(exc)) from None
            rows = valid_grid(got)
            if rows is None:
                raise ValueError("step() returned an invalid grid")
            return rows

        actions = [(int(a[0]), int(a[1]), int(a[2])) for a in req["actions"]]
        res = self.planner.plan(
            step,
            start,
            at_goal,
            actions,
            key=_same,  # tuples of tuples are already hashable, so no freeze()
            max_nodes=int(req["max_nodes"]),
            time_limit_s=float(req["time_limit_s"]),
        )
        return {
            "actions": None if res.actions is None else [list(a) for a in res.actions],
            "stopped": res.stopped,
            "expanded": res.expanded,
            "seconds": res.seconds,
            "errors": res.errors,
        }


def _cap_memory() -> None:
    try:
        # Linux only; on Windows the parent's timeout is the only bound
        resource: Any = importlib.import_module("resource")
    except ImportError:
        return
    # ValueError/OSError: a hard limit below the cap is already stricter
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))


def main() -> None:
    _cap_memory()
    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    channel = sys.stdout.buffer
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115 # lives as long as the process
    planner: Any = None
    planner_error = ""
    try:
        planner = load_planner(sys.argv[1])
    except Exception as exc:
        planner_error = _err(exc)
    worker = Worker(planner, planner_error)
    for line in sys.stdin.buffer:
        req: Any = None
        try:
            req = json.loads(line)
            reply = worker.handle(req)
            reply["id"] = req.get("id")
        except Exception as exc:  # a malformed request must not kill the worker
            # Echo the id whenever the request parsed, so the parent gets the error
            # at once instead of waiting out its deadline and calling it a timeout.
            rid = req.get("id") if isinstance(req, dict) else None
            reply = {"id": rid, "error": _err(exc)}
        channel.write(json.dumps(reply).encode("utf-8") + b"\n")
        channel.flush()


if __name__ == "__main__":
    main()
