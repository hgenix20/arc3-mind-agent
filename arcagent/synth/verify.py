"""Score a candidate rule module against observed transitions, out of process.

Contract:
- Candidate code is never executed in the harness process. ``verify`` writes
  the code, the transitions (JSON) and a small runner into a fresh temp dir
  and runs ``python -I runner.py`` there with a wall-clock limit (default
  20 s for the whole split set).
- The runner imports the candidate, calls ``step(grid, action, x, y)`` on a
  fresh copy of every ``grid_before`` and writes one JSON line per
  transition as soon as it is done, so a timeout keeps the finished results
  and marks only the rest as crashed.
- A prediction is exact only if it is a 64x64 list of lists equal to
  ``grid_after`` cell for cell, padding (-1) included.

Isolation is best-effort, not a security boundary: ``-I`` drops user site
and ``PYTHON*`` variables, the environment is reduced to ``SYSTEMROOT``,
``socket`` is replaced with a stub that raises before the candidate loads,
and the process is killed at the time limit. Run untrusted models on a
machine you are willing to lose.

Failure modes reported, never raised: candidate import error (every
transition crashed with that text), exception inside ``step``, wrong output
shape or type, and timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from arcagent.core.types import GRID_SIZE
from arcagent.synth.trace import Trace, Transition

DEFAULT_TIMEOUT_S = 20.0
MAX_MISMATCHES = 10
MAX_WRONG_CELLS = 20
CELLS = GRID_SIZE * GRID_SIZE

_RUNNER = r"""
import importlib.util, json, socket, sys, traceback

def _blocked(*args, **kwargs):
    raise OSError("network is disabled in the verifier")

socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

N = 64
MAX_WRONG = __MAX_WRONG__
data_path, code_path, out_path = sys.argv[1:4]

def _err(exc):
    tb = traceback.format_exception_only(type(exc), exc)
    return "".join(tb).strip()[-400:]

with open(out_path, "w", encoding="utf-8") as out:
    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
    try:
        spec = importlib.util.spec_from_file_location("candidate", code_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        step = mod.step
    except BaseException as exc:
        emit({"load_error": _err(exc)})
        sys.exit(0)
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    for i, tr in enumerate(data):
        rec = {"i": i, "exact": False, "cells_wrong": N * N, "error": None, "wrong": [],
               "predicted_changes": 0, "predicted": []}
        try:
            got = step([row[:] for row in tr["before"]], tr["a"], tr["x"], tr["y"])
            if not (isinstance(got, list) and len(got) == N
                    and all(isinstance(r, list) and len(r) == N for r in got)):
                raise TypeError("step() must return a 64x64 list of lists of int")
            exp, bef = tr["after"], tr["before"]
            wrong, predicted = [], []
            for y in range(N):
                gr, er, br = got[y], exp[y], bef[y]
                for x in range(N):
                    g = gr[x] if isinstance(gr[x], int) else -99
                    if g != er[x]:
                        wrong.append([x, y, er[x], g])
                    if g != br[x]:
                        predicted.append([x, y, br[x], g])
            rec["cells_wrong"] = len(wrong)
            rec["exact"] = not wrong
            rec["wrong"] = wrong[:MAX_WRONG]
            rec["predicted_changes"] = len(predicted)
            rec["predicted"] = predicted[:MAX_WRONG]
        except BaseException as exc:
            rec["error"] = _err(exc)
        emit(rec)
""".replace("__MAX_WRONG__", str(MAX_WRONG_CELLS))


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Outcome of one prediction. ``wrong`` holds up to 20 ``(x, y, expected, got)``;
    ``predicted`` holds up to 20 ``(x, y, before, got)`` cells the candidate changed and
    ``predicted_changes`` counts all of them, so a repair turn can show the model what
    its own code did, not only where it was wrong."""

    split: str
    index: int
    exact: bool
    cells_wrong: int
    error: str | None
    wrong: tuple[tuple[int, int, int, int], ...] = ()
    predicted_changes: int = 0
    predicted: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(slots=True)
class VerifyResult:
    """Aggregate scores. Rates over an empty split are 0.0.

    ``cell_accuracy`` is over held-out cells, crashed transitions counting
    every cell as wrong. ``crashed`` counts both splits. ``mismatches`` are
    train-split failures only (the held-out split never feeds the repair
    prompt), at most 10, in time order.
    """

    exact_rate_train: float
    exact_rate_heldout: float
    cell_accuracy: float
    crashed: int
    mismatches: list[TransitionResult]
    results: list[TransitionResult] = field(default_factory=list[TransitionResult])
    timed_out: bool = False
    load_error: str | None = None


def _payload(ts: list[Transition]) -> list[dict[str, object]]:
    return [
        {
            "before": t.grid_before.astype(int).tolist(),
            "after": t.grid_after.astype(int).tolist(),
            "a": t.action.id,
            "x": t.action.x,
            "y": t.action.y,
        }
        for t in ts
    ]


def run_candidate(
    code: str, transitions: list[Transition], timeout_s: float = DEFAULT_TIMEOUT_S
) -> tuple[list[dict[str, object]], bool, str | None]:
    """Run ``code`` on ``transitions`` in a subprocess.

    Returns ``(records, timed_out, load_error)`` where ``records`` has one
    dict per finished transition (keys ``i, exact, cells_wrong, error, wrong``).
    """
    with tempfile.TemporaryDirectory(prefix="synth-verify-") as tmp:
        d = Path(tmp)
        (d / "candidate.py").write_text(code, encoding="utf-8")
        (d / "runner.py").write_text(_RUNNER, encoding="utf-8")
        (d / "data.json").write_text(json.dumps(_payload(transitions)), encoding="utf-8")
        out = d / "out.jsonl"
        env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        cmd = [sys.executable, "-I", "runner.py", "data.json", "candidate.py", str(out)]
        timed_out = False
        try:
            subprocess.run(  # fixed argv, no shell
                cmd,
                cwd=d,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        records: list[dict[str, object]] = []
        load_error: str | None = None
        if out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by the kill
                if "load_error" in rec:
                    load_error = str(rec["load_error"])
                else:
                    records.append(rec)
        return records, timed_out, load_error


def _rate(rs: list[TransitionResult]) -> float:
    return sum(r.exact for r in rs) / len(rs) if rs else 0.0


def _cells(raw: object) -> tuple[tuple[int, int, int, int], ...]:
    """A runner cell list ``[[x, y, a, b], ...]`` as a tuple of int 4-tuples; absent -> ()."""
    if not isinstance(raw, list):
        return ()
    return tuple(
        (int(w[0]), int(w[1]), int(w[2]), int(w[3]))
        for w in raw  # pyright: ignore[reportUnknownVariableType]
    )


def verify(code: str, trace: Trace, timeout_s: float = DEFAULT_TIMEOUT_S) -> VerifyResult:
    """Score ``code`` on the train and held-out splits of ``trace`` in one subprocess."""
    train, held = trace.train, trace.heldout
    records, timed_out, load_error = run_candidate(code, train + held, timeout_s)
    by_index = {int(str(r["i"])): r for r in records}
    results: list[TransitionResult] = []
    for k in range(len(train) + len(held)):
        split, idx = ("train", k) if k < len(train) else ("heldout", k - len(train))
        rec = by_index.get(k)
        if rec is None:
            err = load_error or ("timeout" if timed_out else "no result (runner died)")
            results.append(TransitionResult(split, idx, False, CELLS, err))
            continue
        err_val = rec.get("error")
        results.append(
            TransitionResult(
                split=split,
                index=idx,
                exact=bool(rec["exact"]),
                cells_wrong=int(str(rec["cells_wrong"])),
                error=None if err_val is None else str(err_val),
                wrong=_cells(rec.get("wrong")),
                predicted_changes=int(str(rec.get("predicted_changes") or 0)),
                predicted=_cells(rec.get("predicted")),
            )
        )
    tr = [r for r in results if r.split == "train"]
    ho = [r for r in results if r.split == "heldout"]
    cell_acc = 1.0 - sum(r.cells_wrong for r in ho) / (len(ho) * CELLS) if ho else 0.0
    return VerifyResult(
        exact_rate_train=_rate(tr),
        exact_rate_heldout=_rate(ho),
        cell_accuracy=cell_acc,
        crashed=sum(r.error is not None for r in results),
        mismatches=[r for r in tr if not r.exact][:MAX_MISMATCHES],
        results=results,
        timed_out=timed_out,
        load_error=load_error,
    )


def identity_exact_rate(transitions: list[Transition]) -> float:
    """Exact rate of the identity model (next grid = current grid); 0.0 if empty."""
    if not transitions:
        return 0.0
    same = sum(bool(np.array_equal(t.grid_before, t.grid_after)) for t in transitions)
    return same / len(transitions)


IDENTITY_CODE = '''"""Identity baseline: predicts that nothing changes."""


def step(grid, action, x=-1, y=-1):
    return [row[:] for row in grid]
'''
