"""Record live model replies and replay them offline.

Contract:
- ``recording_chat`` wraps a live ``Chat`` and writes every successful reply to
  ``<directory>/<sha1>.json``, ``sha1`` over ``json.dumps(messages, sort_keys=True) + str(seed)``.
  Writes go to a temp file first and then ``os.replace``, so games writing at the same
  time never leave a half file. A failed inner call writes nothing and re-raises.
- The ``{"game", "call", "k"}`` metadata comes from ``REQUEST_META``, a context variable
  the synthesizer sets around each chat call in the calling thread. Outside a synthesizer
  call the metadata is empty and the fields are stored as ``""`` / ``-1``.
- ``replay_chat`` indexes the directory once, at construction. A request is answered by
  the exact prompt hash, else by the same ``(game, call, k)`` from ``REQUEST_META``, else
  by the identity program, so an offline run never fails for lack of a recording.
  Unreadable or malformed files are skipped.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from arcagent.synth.prompt import Message
from arcagent.synth.run import Chat, Completion
from arcagent.synth.verify import IDENTITY_CODE

REQUEST_META: ContextVar[dict[str, Any] | None] = ContextVar("REQUEST_META", default=None)
"""``{"game": str, "call": int, "k": int}`` for the chat call in progress, else None."""

_META_KEY = tuple[str, int, int]


def messages_sha1(messages: list[Message], seed: int) -> str:
    """The reply-file key: sha1 of the sorted-key messages JSON followed by the seed."""
    return hashlib.sha1(
        (json.dumps(messages, sort_keys=True) + str(seed)).encode()
    ).hexdigest()  # a content key, not a security use


def _meta() -> dict[str, Any]:
    return REQUEST_META.get() or {}


def _write_atomic(path: Path, doc: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def recording_chat(inner: Chat, directory: Path) -> Chat:
    """Wrap ``inner`` so every reply is saved under ``directory`` (created if absent)."""
    directory.mkdir(parents=True, exist_ok=True)

    def call(messages: list[Message], seed: int) -> Completion:
        comp = inner(messages, seed)
        meta = _meta()
        key = messages_sha1(messages, seed)
        _write_atomic(
            directory / f"{key}.json",
            {
                "messages_sha1": key,
                "seed": seed,
                "game": str(meta.get("game", "")),
                "call": int(meta.get("call", -1)),
                "k": int(meta.get("k", -1)),
                "text": comp.text,
                "prompt_tokens": comp.prompt_tokens,
                "completion_tokens": comp.completion_tokens,
                "seconds": comp.seconds,
                "finish_reason": comp.finish_reason,
            },
        )
        return comp

    return call


def _completion(doc: dict[str, Any]) -> Completion:
    return Completion(
        text=str(doc["text"]),
        prompt_tokens=int(doc.get("prompt_tokens", 0)),
        completion_tokens=int(doc.get("completion_tokens", 0)),
        seconds=float(doc.get("seconds", 0.0)),
        finish_reason=str(doc.get("finish_reason", "")),
    )


def replay_chat(directory: Path) -> Chat:
    """A ``Chat`` that answers from the replies recorded in ``directory``."""
    by_hash: dict[str, Completion] = {}
    by_meta: dict[_META_KEY, Completion] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            comp = _completion(doc)
            by_hash[str(doc["messages_sha1"])] = comp
            by_meta.setdefault((str(doc["game"]), int(doc["call"]), int(doc["k"])), comp)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    fallback = Completion(f"```python\n{IDENTITY_CODE}```", 0, 0, 0.0, "stop", 0)

    def call(messages: list[Message], seed: int) -> Completion:
        hit = by_hash.get(messages_sha1(messages, seed))
        if hit is not None:
            return hit
        meta = _meta()
        if meta:
            hit = by_meta.get((str(meta.get("game", "")), int(meta["call"]), int(meta["k"])))
            if hit is not None:
                return hit
        return fallback

    return call
