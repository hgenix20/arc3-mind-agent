"""Vision chat: scripted order, record and replay, and the HTTP client against a stub."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from arcagent.mind.vchat import (
    ImagesRejected,
    VMessage,
    http_vision_chat,
    recording,
    replaying,
    scripted,
)
from arcagent.synth.run import Completion

MSGS: list[VMessage] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }
]


def test_scripted_order_then_empty() -> None:
    chat = scripted(["a", "b"])
    assert [chat(MSGS, 0).text for _ in range(4)] == ["a", "b", "{}", "{}"]


def test_recording_writes_one_file_and_replay_returns_it(tmp_path: Path) -> None:
    rec = recording(scripted(['{"notes": "x"}']), tmp_path / "rec")
    assert rec(MSGS, 3).text == '{"notes": "x"}'
    assert len(list((tmp_path / "rec").glob("*.json"))) == 1
    assert replaying(tmp_path / "rec")(MSGS, 3).text == '{"notes": "x"}'
    assert list((tmp_path / "rec").glob("*.tmp")) == []


def test_replay_missing_recording_is_empty_object(tmp_path: Path) -> None:
    assert replaying(tmp_path)(MSGS, 0).text == "{}"
    recording(scripted(["z"]), tmp_path)(MSGS, 1)
    assert replaying(tmp_path)(MSGS, 2).text == "{}"


def test_failed_inner_call_records_nothing(tmp_path: Path) -> None:
    def boom(messages: list[VMessage], seed: int) -> Completion:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        recording(boom, tmp_path)(MSGS, 0)
    assert list(tmp_path.glob("*.json")) == []


@contextmanager
def _stub(status: int) -> Generator[tuple[str, list[dict[str, Any]]]]:
    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            n = int(self.headers["Content-Length"])
            seen.append(json.loads(self.rfile.read(n)))
            body = json.dumps(
                {
                    "choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            ).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", seen
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


def test_http_sends_image_parts_and_parses_reply() -> None:
    with _stub(200) as (url, seen):
        comp = http_vision_chat(url, "m", max_tokens=77, timeout_s=5)(MSGS, 4)
    assert comp.text == '{"ok": 1}' and comp.prompt_tokens == 5
    body = seen[0]
    assert body["max_tokens"] == 77 and body["seed"] == 4
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    content = body["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png")


def test_http_400_raises_images_rejected() -> None:
    with _stub(400) as (url, _):
        chat = http_vision_chat(url, "m", max_tokens=10, timeout_s=5)
        with pytest.raises(ImagesRejected):
            chat(MSGS, 0)
