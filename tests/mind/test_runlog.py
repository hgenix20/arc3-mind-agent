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


def test_runlog_refuses_an_existing_file(tmp_path: Path) -> None:
    # Appending would mix two runs in one game's log.
    path = tmp_path / "g.jsonl"
    path.write_text('{"type": "action"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError):
        RunLog(path, game="g", config="c", seed=0)
    assert path.read_text(encoding="utf-8") == '{"type": "action"}\n'


def test_runlog_action_with_v2_nested_fields_round_trips(tmp_path: Path) -> None:
    """The planner agent's nested self, focus and gate records come back as plain JSON."""
    path = tmp_path / "g.jsonl"
    log = RunLog(path, game="ls20", config="PLANNER", seed=1)
    log.action(
        step=7,
        level=1,
        self_rec={"i": "agent", "click": np.bool_(True), "my_character": None},
        focus={"level": 1, "goal_id": "g3", "budget_left": np.int64(12), "mode": "follow"},
        hypothesis_id="h2",
        segment_id="s5",
        fork_id=None,
        planner_call=3,
        p_step=np.float64(0.8125),
        p_path=0.5,
        prev_matched=True,
        gate={"seconds": np.float32(0.25), "tokens": 120, "trip": None},
    )
    log.event("fork_open", step=7, id="f1", goal="g3", paths=[[(1, 0, 0), (2, 0, 0)], [(3, 0, 0)]])
    log.event("longterm", game_id="ls20", rules=4, sequences=1)
    log.close()
    rows = _lines(path)
    a = rows[0]
    assert a["self_rec"] == {"i": "agent", "click": True, "my_character": None}
    assert a["focus"] == {"level": 1, "goal_id": "g3", "budget_left": 12, "mode": "follow"}
    assert a["gate"] == {"seconds": 0.25, "tokens": 120, "trip": None}
    ids = (a["hypothesis_id"], a["segment_id"], a["fork_id"], a["planner_call"])
    assert ids == ("h2", "s5", None, 3)
    assert a["p_step"] == 0.8125 and a["p_path"] == 0.5 and a["prev_matched"] is True
    assert rows[1]["paths"] == [[[1, 0, 0], [2, 0, 0]], [[3, 0, 0]]]
    assert rows[2]["game_id"] == "ls20" and rows[2]["game"] == "ls20"


def test_runlog_self_keyword_collides_with_the_method_so_the_field_is_self_rec(
    tmp_path: Path,
) -> None:
    path = tmp_path / "g.jsonl"
    log = RunLog(path, game="g", config="c", seed=0)
    with pytest.raises(TypeError):
        log.action(**{"self": {"i": "agent"}})
    log.close()
    assert path.read_text(encoding="utf-8") == ""


def test_runlog_rejects_unserializable_values_inside_nested_fields(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    log = RunLog(path, game="g", config="c", seed=0)
    with pytest.raises(TypeError):
        log.action(focus={"target": object()})
    with pytest.raises(TypeError):
        log.event("fork_open", step=1, paths=[{1, 2}])
    log.close()
    assert path.read_text(encoding="utf-8") == ""
