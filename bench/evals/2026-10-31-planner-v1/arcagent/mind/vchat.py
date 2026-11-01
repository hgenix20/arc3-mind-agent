"""Vision chat: messages with image parts, plus record and replay for offline runs.

Contract:
- ``VisionChat(messages, seed) -> Completion``. A message's ``content`` is a string or a
  list of parts: ``{"type": "text", "text": ...}`` or
  ``{"type": "image_url", "image_url": {"url": <data url>}}``.
- ``http_vision_chat`` posts to an OpenAI-compatible ``/chat/completions`` with the body
  of ``arcagent.synth.run.request_body`` (same temperature, thinking off) and the given
  ``max_tokens``; with ``json_reply`` it also asks for ``response_format`` ``json_object``
  (the reply is one JSON object, with no prose before it). HTTP 400 whose body mentions
  images raises ``ImagesRejected``; any other failure
  (including another 400) raises ``requests`` or parse errors. ``timeout_s`` is the socket
  timeout of ``requests``, not a wall-clock bound; the caller enforces the deadline.
- ``recording`` saves every reply to ``<directory>/<sha1>.json`` (sha1 over
  ``json.dumps(messages, sort_keys=True) + str(seed)``), written atomically. A failed
  inner call writes nothing and re-raises.
- ``replaying`` answers by that key only; a missing or unreadable recording answers
  ``{}`` so the run continues on slice 1 logic.
- Replay reproduces a prompt only when every earlier reply arrived at the same move as
  in the recorded run, since each reply changes the moves, scene and facts the next
  prompt shows. The fluid agent waits for its turn inside the move that asked, but a
  turn that timed out live (its reply recorded later, or never) gets an instant answer
  or ``{}`` on replay, and any timing-dependent choice (deadlines, the 75% turn cutoff)
  can differ. So a live run can't be replayed exactly; replay is exact only for a run
  that was itself recorded offline with no timeouts.
- ``scripted`` returns fixed replies in order, then ``{}``; for offline tests.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import requests

from arcagent.synth.run import Completion, parse_completion, request_body

Part = dict[str, object]
VMessage = dict[str, object]
VisionChat = Callable[[list[VMessage], int], Completion]

EMPTY_REPLY = "{}"


class ImagesRejected(RuntimeError):  # noqa: N818 (name fixed by the plan contract)
    """The server answered HTTP 400 to a request carrying images."""


def _empty() -> Completion:
    return Completion(text=EMPTY_REPLY, prompt_tokens=0, completion_tokens=0, seconds=0.0)


def vision_key(messages: list[VMessage], seed: int) -> str:
    """The recording key; same formula as ``chat_store.messages_sha1``."""
    return hashlib.sha1(
        (json.dumps(messages, sort_keys=True) + str(seed)).encode()
    ).hexdigest()  # a content key, not a security use


_IMAGE_WORDS = ("image", "multimodal", "mm_", "vision")


def _mentions_images(body: str) -> bool:
    low = body.lower()
    return any(w in low for w in _IMAGE_WORDS)


def http_vision_chat(
    endpoint: str, model: str, *, max_tokens: int, timeout_s: float, json_reply: bool = False
) -> VisionChat:
    """A ``VisionChat`` over an OpenAI-compatible endpoint."""
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("SYNTH_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def call(messages: list[VMessage], seed: int) -> Completion:
        body = request_body(model, cast("Any", messages), seed)
        body["max_tokens"] = max_tokens
        if json_reply:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
        secs = time.perf_counter() - t0
        if resp.status_code == 400 and _mentions_images(resp.text):
            raise ImagesRejected(resp.text[:200])
        resp.raise_for_status()
        return parse_completion(resp.json(), secs)

    return call


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


def recording(inner: VisionChat, directory: Path) -> VisionChat:
    """Wrap ``inner`` so every reply is saved under ``directory`` (created if absent)."""
    directory.mkdir(parents=True, exist_ok=True)

    def call(messages: list[VMessage], seed: int) -> Completion:
        comp = inner(messages, seed)
        key = vision_key(messages, seed)
        _write_atomic(
            directory / f"{key}.json",
            {
                "key": key,
                "seed": seed,
                "text": comp.text,
                "prompt_tokens": comp.prompt_tokens,
                "completion_tokens": comp.completion_tokens,
                "seconds": comp.seconds,
                "finish_reason": comp.finish_reason,
            },
        )
        return comp

    return call


def replaying(directory: Path) -> VisionChat:
    """A ``VisionChat`` that answers from ``directory`` by key; a miss answers ``{}``."""

    def call(messages: list[VMessage], seed: int) -> Completion:
        path = directory / f"{vision_key(messages, seed)}.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            return Completion(
                text=str(doc["text"]),
                prompt_tokens=int(doc.get("prompt_tokens", 0)),
                completion_tokens=int(doc.get("completion_tokens", 0)),
                seconds=float(doc.get("seconds", 0.0)),
                finish_reason=str(doc.get("finish_reason", "")),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return _empty()

    return call


def scripted(replies: Sequence[str]) -> VisionChat:
    """Replies in order, then ``{}``."""
    queue = list(replies)

    def call(messages: list[VMessage], seed: int) -> Completion:
        del messages, seed
        text = queue.pop(0) if queue else EMPTY_REPLY
        return Completion(text=text, prompt_tokens=0, completion_tokens=0, seconds=0.0)

    return call
