"""The tool-fix turn: the LLM rewrites a planning tool that failed inside its model.

Contract:
- A ``ToolFailure`` names the tool, the failing call's arguments, what the LLM expected
  and what the tool gave (in words, for the prompt), and ``check``: whether a candidate's
  result solves the failing case (run in the agent's process on the unpickled result).
- ``fix_messages`` shows the tool's current source (the adopted version's, else the
  original's ``inspect.getsource``), the names the module offers, the failing case and
  both answers, and asks for one general function with the same name and signature.
- ``try_fix`` asks once, takes the first ```` ```python ```` fence (or any fence) that
  defines the function, and adopts it in ``box`` only when, in the tools worker, it solves
  the failing case (``check``) and agrees with every stored regression case
  (``ToolSpec.agrees``), each call within ``FIX_CALL_S``. It never raises: any failure is
  ``FixResult(None, reason)``. The fix is adopted only while the version it was written
  against is still current (``tool changed meanwhile`` otherwise). Candidates run under
  negative version numbers so the worker never mistakes one for an adopted version.
"""

from __future__ import annotations

import inspect
import itertools
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from arcagent.mind.toolbox import TOOLBOX, TOOLS, Toolbox, ToolVersion, ToolWorker
from arcagent.mind.vchat import VisionChat, VMessage

FIX_CALL_S = 5.0  # deadline per candidate call in the worker
MAX_SOURCE_CHARS = 6000
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_candidates = itertools.count(1)

FIX_SYSTEM = """\
You repair one Python function used by a game-playing agent's planner.
The function gave a wrong answer on a case where the right answer is known.
Rewrite it so it handles this case and every case it already handled.
Keep it general: never hard-code coordinates, object names, sizes or move lists
from the case. Keep the same name, signature and return type.
You may use any name the module already defines.
Reply with one ```python fenced block holding the complete function and nothing else."""


@dataclass(frozen=True, slots=True)
class ToolFailure:
    tool: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    expected: str
    got: str
    check: Callable[[Any], bool] = field(compare=False)


@dataclass(frozen=True, slots=True)
class FixResult:
    adopted: int | None
    reason: str


def current_source(tool: str, box: Toolbox = TOOLBOX) -> str:
    v = box.current(tool)
    if v is not None:
        return v.source
    return inspect.getsource(TOOLS[tool].original())


def _module_names(tool: str) -> str:
    import importlib  # noqa: PLC0415  (only the prompt needs the module's names)

    mod = importlib.import_module(TOOLS[tool].module)
    names = sorted(n for n in vars(mod) if not n.startswith("__"))
    return ", ".join(names)[:1500]


def fix_messages(failure: ToolFailure, box: Toolbox = TOOLBOX) -> list[VMessage]:
    spec = TOOLS[failure.tool]
    text = "\n".join(
        [
            f"Function: {spec.func} in module {spec.module}",
            "Current source:",
            current_source(failure.tool, box)[:MAX_SOURCE_CHARS],
            f"Names available in the module: {_module_names(failure.tool)}",
            f"Failing case: {failure.expected}",
            f"What it returned: {failure.got}",
        ]
    )
    return [
        {"role": "system", "content": FIX_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def extract_function(text: str, func: str) -> str | None:
    """The first fenced block that defines ``func``, or None."""
    for block in _FENCE.findall(text):
        if re.search(rf"^def {re.escape(func)}\s*\(", block, re.MULTILINE):
            return block.strip() + "\n"
    return None


def try_fix(  # noqa: PLR0911  (one return per rejection reason)
    chat: VisionChat,
    failure: ToolFailure,
    worker: ToolWorker,
    *,
    game: str,
    seed: int,
    box: Toolbox = TOOLBOX,
) -> FixResult:
    """Ask for a fix, check it, adopt it; see the module docstring."""
    spec = TOOLS[failure.tool]
    start = box.current(failure.tool)
    base = 0 if start is None else start.no
    try:
        text = chat(fix_messages(failure, box), seed).text
    except Exception as exc:  # the fix is lost, the game goes on
        return FixResult(None, f"chat failed: {type(exc).__name__}")
    source = extract_function(text, spec.func)
    if source is None:
        return FixResult(None, "no function in reply")
    cand = ToolVersion(-next(_candidates), source, game, "candidate", time.time())
    ok, got = worker.call(spec, cand, failure.args, failure.kwargs, FIX_CALL_S)
    if not ok:
        return FixResult(None, f"failing case: {str(got)[:200]}")
    try:
        solved = failure.check(got)
    except Exception as exc:  # a malformed result fails the check
        return FixResult(None, f"check raised {type(exc).__name__}")
    if not solved:
        return FixResult(None, "does not solve the failing case")
    for i, case in enumerate(box.cases(failure.tool)):
        ok, got = worker.call(spec, cand, case.args, case.kwargs, FIX_CALL_S)
        if not ok or not spec.agrees(case.result, got):
            return FixResult(None, f"regression case {i} disagrees")
    no = box.adopt(failure.tool, source, game, failure.expected[:120], base=base)
    if no is None:
        return FixResult(None, "tool changed meanwhile")
    return FixResult(no, "adopted")
