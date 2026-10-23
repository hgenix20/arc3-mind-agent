"""Run log: one JSON line per call, plain types, readable before close."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arcagent.mind.runlog import RunLog


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_runlog_writes_one_json_line_per_call_with_plain_types(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "g.jsonl"
    log = RunLog(path, game="ls20", config="full", seed=3)
    log.action(
        step=np.int64(4),
        entropy=np.float32(0.5),
        done=np.bool_(True),
        action=(6, 1, 2),
        grid=np.array([[1, 2]]),
    )
    log.event("sleep", n=np.int32(2))
    log.close()
    rows = _lines(path)
    assert [r["type"] for r in rows] == ["action", "sleep"]
    a = rows[0]
    assert a["game"] == "ls20" and a["config"] == "full" and a["seed"] == 3
    assert isinstance(a["t"], float)
    assert a["step"] == 4 and type(a["step"]) is int
    assert a["entropy"] == 0.5 and a["done"] is True
    assert a["action"] == [6, 1, 2] and a["grid"] == [[1, 2]]
    assert rows[1]["n"] == 2
    e = rows[1]
    assert (e["game"], e["config"], e["seed"]) == ("ls20", "full", 3)
    assert isinstance(e["t"], float)


def test_runlog_envelope_fields_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    log = RunLog(path, game="g", config="c", seed=0)
    for name in ("type", "t", "game", "config", "seed"):
        with pytest.raises(ValueError, match=name):
            log.event("plan", **{name: "x"})
        with pytest.raises(ValueError, match=name):
            log.action(**{name: "x"})
    log.close()
    assert path.read_text(encoding="utf-8") == ""


def test_runlog_lines_survive_without_close(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    log = RunLog(path, game="g", config="c", seed=0)
    log.event("llm_call", ok=True)
    assert len(_lines(path)) == 1
    log.close()


def test_runlog_rejects_unserializable_fields(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "g.jsonl", game="g", config="c", seed=0)
    with pytest.raises(TypeError):
        log.event("plan", bad=object())
    log.close()
