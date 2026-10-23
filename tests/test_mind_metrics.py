"""Metrics script: one hand-built log per metric, each answer worked out by hand."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mind_metrics.py"
GAME = "zz99-00000000"
Rec = dict[str, Any]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mind_metrics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mm = load_script()


def act(
    step: int,
    level: int,
    frame: str,
    *,
    action: tuple[int, int, int] = (1, 0, 0),
    entropy: float | None = None,
    programs: list[Rec] | None = None,
    active: str | None = None,
    game: str = GAME,
) -> Rec:
    return {
        "type": "action",
        "game": game,
        "level": level,
        "step": step,
        "frame_hash": frame,
        "action": list(action),
        "programs": programs or [],
        "entropy": entropy,
        "active": active,
    }


def ev(kind: str, step: int, game: str = GAME, **fields: Any) -> Rec:
    return {"type": kind, "game": game, "step": step, **fields}


def per_game(records: list[Rec], game: str = GAME) -> Rec:
    return mm.compute(records, human={})["games"][game]


def test_score_estimate_matches_competition_formula() -> None:
    # 3 levels, humans [10, 20, 30]; agent used 10 then 40; level 3 unfinished.
    records: list[Rec] = []
    for i in range(10):
        records.append(act(i, 0, f"a{i}"))
    records.append(ev("level_up", 9, level=1))
    for i in range(10, 50):
        records.append(act(i, 1, f"b{i}"))
    records.append(ev("level_up", 49, level=2))
    records.append(act(50, 2, "c0"))
    out = mm.compute(records, human={"zz99": [10, 20, 30]})["games"][GAME]
    assert out["levels_completed"] == 2
    assert out["actions_per_completed_level"] == [10, 40]
    assert out["score_estimate"] == pytest.approx(25.0)
    assert out["transfer_actions"] == {"level_1": 10.0, "levels_2_plus": 40.0}


def test_score_estimate_unknown_game_is_none() -> None:
    assert per_game([act(0, 0, "a")])["score_estimate"] is None


def test_repeat_rate_counts_only_after_a_bad_outcome() -> None:
    # s0 (f1,A) is a noop (next frame is f1 again), so s1 (f1,A) repeats a bad outcome.
    # s4 (f2,B) repeats s2 (f2,B), which moved on to f3: not a repeat of a bad one.
    # s5 (f3,C) ends in game_over; after the RESET at s6, s7 (f3,C) repeats it.
    a, b, c = (1, 0, 0), (2, 0, 0), (3, 0, 0)
    records = [
        act(0, 0, "f1", action=a),
        act(1, 0, "f1", action=a),
        act(2, 0, "f2", action=b),
        act(3, 0, "f3", action=(4, 9, 9)),
        act(4, 0, "f2", action=b),
        act(5, 0, "f3", action=c),
        ev("game_over", 5, level=0),
        act(6, 0, "f4", action=(0, 0, 0)),
        act(7, 0, "f3", action=c),
    ]
    out = per_game(records)
    # Repeats: s1 and s7. Non-reset actions: 7 (s6 is the RESET).
    assert out["disconfirmed_repeat_rate"] == pytest.approx(2 / 7)
    assert out["outcomes"] == {"noop": 1, "game_over": 1, "disconfirmed": 0}


def test_repeat_rate_is_per_level() -> None:
    records = [
        act(0, 0, "f1"),
        act(1, 0, "f1"),
        act(2, 1, "f1"),
    ]
    out = per_game(records)
    # s0 is a noop; s2 is on another level so its pair is fresh. s1 is the only repeat.
    assert out["disconfirmed_repeat_rate"] == pytest.approx(1 / 3)


def test_disconfirmed_outcome_reads_next_records_programs() -> None:
    miss = [{"id": "p1", "pred": "x", "matched": False, "fit": 0.5}]
    hit = [{"id": "p1", "pred": "x", "matched": True, "fit": 0.5}]
    other = [{"id": "p2", "pred": "x", "matched": False, "fit": 0.5}]
    records = [
        act(0, 0, "a", active="p1"),
        act(1, 0, "b", programs=miss, active="p1"),  # scores s0: disconfirmed
        act(2, 0, "c", programs=hit, active="p1"),  # scores s1: confirmed
        act(3, 0, "d", programs=other),  # p2 missed but s2 had no active program
    ]
    out = per_game(records)
    assert out["outcomes"]["disconfirmed"] == 1
    assert out["prediction_accuracy"] == pytest.approx(1 / 3)


def test_latency_from_drop_events() -> None:
    records = [
        act(0, 0, "a"),
        ev("program_dropped", 10, pid="p1", reason="low_fit", first_miss_step=4),
        ev("program_dropped", 20, pid="p2", reason="misses_in_row", first_miss_step=17),
        ev("program_dropped", 30, pid="p3", reason="low_fit", first_miss_step=None),
        ev("program_dropped", 40, pid="p4", reason="timeout", first_miss_step=1),
        ev("program_dropped", 41, pid="p5", reason="level_end", first_miss_step=None),
    ]
    out = per_game(records)
    assert out["abandonment_latency"] == pytest.approx((6 + 3) / 2)
    assert out["abandonment_latency_n"] == 2
    assert out["drop_reasons"] == {
        "low_fit": 2,
        "misses_in_row": 1,
        "timeout": 1,
        "level_end": 1,
    }


def test_latency_none_without_drops() -> None:
    out = per_game([act(0, 0, "a")])
    assert out["abandonment_latency"] is None
    assert out["abandonment_latency_n"] == 0


def test_entropy_change_within_levels_only() -> None:
    records = [
        act(0, 0, "a", entropy=5.0),
        act(1, 0, "b", entropy=4.0),  # +1.0
        act(2, 0, "c", entropy=3.5),  # +0.5
        act(3, 1, "d", entropy=9.0),  # crosses a level: excluded
        act(4, 1, "e", entropy=8.0),  # +1.0
    ]
    assert per_game(records)["entropy_change_per_action"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)


def test_events_are_attributed_by_game_and_games_are_separate() -> None:
    other = "yy88-11111111"
    records = [
        act(0, 0, "a"),
        act(0, 0, "a", game=other),
        ev("level_up", 0, game=other, level=1),
    ]
    out = mm.compute(records, human={})
    assert out["games"][GAME]["levels_completed"] == 0
    assert out["games"][other]["levels_completed"] == 1
    assert out["overall"]["levels_completed"] == 1


def test_load_skips_and_counts_malformed_lines_and_stamps_events(tmp_path: Path) -> None:
    path = tmp_path / f"{GAME}.jsonl"
    lines = [
        json.dumps(act(0, 0, "a")),
        "{not json",
        json.dumps([1, 2]),
        json.dumps({"type": "level_up", "step": 0, "level": 1}),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records, bad = mm.load([tmp_path])
    assert bad == 2
    assert [r["type"] for r in records] == ["action", "level_up"]
    assert records[1]["game"] == GAME
    assert per_game(records)["levels_completed"] == 1


def test_load_keeps_a_game_already_on_the_record(tmp_path: Path) -> None:
    """Only records missing ``game`` are stamped; new-style events keep their own."""
    path = tmp_path / "file-stem.jsonl"
    lines = [
        json.dumps(act(0, 0, "a")),
        json.dumps({"type": "level_up", "game": "zz99", "step": 0, "level": 1}),
        json.dumps({"type": "sleep", "step": 1}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records, bad = mm.load([path])
    assert bad == 0
    assert [r["game"] for r in records] == [GAME, "zz99", GAME]
