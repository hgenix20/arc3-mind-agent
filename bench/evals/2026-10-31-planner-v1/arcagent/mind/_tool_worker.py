"""Tools worker: runs LLM-patched tools and LLM rule functions out of the agent's process.

Started by ``toolbox.ToolWorker`` as ``python -u _tool_worker.py`` with the repo root
(the directory holding ``arcagent``) put on ``sys.path`` here, so a patched tool can use
the module it replaces. One JSON object per line in, one per line out; arguments and
results travel as base64 pickles from and to the parent only.

Requests:
- ``{"op": "call", "name", "version", "module", "func", "source", "args"}``: exec
  ``source`` into a copy of ``module``'s globals (once per name and version), then call
  ``func`` with the unpickled ``(args, kwargs)``.
- ``{"op": "rule", "source", "args"}``: exec ``source`` into a fresh namespace and call
  its ``rule(objects, action)``.
Replies: ``{"ok": true, "result": <b64 pickle>}`` or ``{"ok": false, "error": <text>}``,
each echoing the request's ``rid``. Replies go to the original stdout only: ``sys.stdout``
is pointed at stderr before any model code runs, so a ``print`` there can't be read as a
reply.
A crash or hang is the parent's to handle (it kills and restarts this process).
"""

from __future__ import annotations

import base64
import importlib
import json
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_loaded: dict[str, tuple[int, Any]] = {}
_rules: dict[str, Any] = {}


def _load(name: str, version: int, module: str, func: str, source: str) -> Any:
    got = _loaded.get(name)
    if got is not None and got[0] == version:
        return got[1]
    ns = dict(vars(importlib.import_module(module)))
    exec(compile(source, f"<patched {name} v{version}>", "exec"), ns)
    fn = ns[func]
    _loaded[name] = (version, fn)
    return fn


def _rule(source: str) -> Any:
    fn = _rules.get(source)
    if fn is None:
        ns: dict[str, Any] = {}
        exec(compile(source, "<rule>", "exec"), ns)
        fn = ns["rule"]
        _rules.clear()  # one rule per game at a time
        _rules[source] = fn
    return fn


def _handle(req: dict[str, Any]) -> dict[str, Any]:
    args, kwargs = pickle.loads(base64.b64decode(req["args"]))
    if req["op"] == "call":
        fn = _load(req["name"], req["version"], req["module"], req["func"], req["source"])
    elif req["op"] == "rule":
        fn = _rule(req["source"])
    else:
        return {"ok": False, "error": f"unknown op {req['op']!r}"}
    result = fn(*args, **kwargs)
    return {"ok": True, "result": base64.b64encode(pickle.dumps(result)).decode("ascii")}


def main() -> None:
    out = sys.stdout
    sys.stdout = sys.stderr  # model code's prints never reach the reply stream
    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        reply: dict[str, Any]
        try:
            req = json.loads(line)
            rid = req.get("rid")
            reply = _handle(req)
        except Exception as exc:  # model code: any failure is a reply, not a crash
            tb = traceback.format_exc(limit=3)
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}\n{tb}"[:2000]}
        reply["rid"] = rid
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
