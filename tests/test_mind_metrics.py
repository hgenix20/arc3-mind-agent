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


def entry(pid: str, matched: bool, level: int | None = None) -> Rec:
    out: Rec = {"id": pid, "pred": "p", "matched": matched, "fit": 1.0}
    if level is not None:
        out["level"] = level
    return out


def slice3_log() -> list[Rec]:
    """Level 0: 4 actions; level 1: 2 actions. Every answer is worked out in the tests."""
    return [
        act(0, 0, "a0", programs=[entry("x", False, 0), entry("y", True, 0)]),
        act(1, 0, "a1", programs=[entry("x", False, 0)]),
        act(2, 0, "a2", programs=[entry("x", True, 0)]),
        act(3, 0, "a3", programs=[entry("x", True, 0), entry("y", True, 0)]),
        ev("level_up", 3, level=1),
        act(4, 1, "b0", programs=[entry("x", True, 0), entry("z", True, 1), entry("w", False, 0)]),
        act(5, 1, "b1", programs=[entry("x", True, 0)]),
        ev("search", 0, stop="goal", nodes=5, depth=4, goals=1, frontier=2, protected=0),
        ev("search", 4, stop="cap", nodes=9, depth=2, goals=1, frontier=0, protected=0),
        ev("wake", 1, reason="no plan and no test"),
        ev("wake", 2, reason="new fact"),
        ev("test", 0, action=[1, 0, 0], question="q"),
        ev("test", 4, action=[2, 0, 0], question="q"),
        ev("test", 5, action=[3, 0, 0], question="q"),
        ev("goal", 0, id="g1", status="candidate", source="s"),
        ev("goal", 1, id="g1", status="confirmed", source="s"),
        ev("goal", 1, id="g2", status="refuted", source="s"),
    ]


def test_hit_rate_and_halves() -> None:
    out = per_game(slice3_log())
    # 10 entries, 7 matched.
    assert out["hit_rate"] == pytest.approx(7 / 10)
    # level 0 first half = records 0,1 (1 of 3), second = records 2,3 (3 of 3);
    # level 1 first half = record 0 (2 of 3), second = record 1 (1 of 1).
    assert out["hit_rate_halves"]["first"] == pytest.approx(3 / 6)
    assert out["hit_rate_halves"]["second"] == pytest.approx(4 / 4)


def test_rule_reuse_counts_matched_entries_from_earlier_levels() -> None:
    # matched entries with a level: 4 on level 0 (none reused); level 1 has
    # x(0) reused, z(1) not, x(0) reused. 7 total, 2 reused.
    assert per_game(slice3_log())["rule_reuse"] == pytest.approx(2 / 7)


def test_tests_impasses_depth_goals() -> None:
    out = per_game(slice3_log())
    assert out["tests_per_level"] == pytest.approx(3 / 2)
    assert out["impasses"] == 1
    assert out["search_depth_mean"] == pytest.approx(3.0)
    assert out["goals"] == {
        "final": {"confirmed": 1, "refuted": 1},
        "transitions": {"candidate": 1, "confirmed": 1, "refuted": 1},
    }


def test_actions_vs_human_per_completed_level() -> None:
    out = mm.compute(slice3_log(), human={"zz99": [2, 5]})["games"][GAME]
    assert out["actions_vs_human"] == [pytest.approx(2.0)]  # level 0 only: 4 / 2
    assert per_game(slice3_log())["actions_vs_human"] is None


def test_overall_pools_new_metrics_across_games() -> None:
    other = "yy88-00000000"
    records = slice3_log() + [{**r, "game": other} for r in slice3_log() if r["type"] != "level_up"]
    overall = mm.compute(records, human={})["overall"]
    assert overall["hit_rate"] == pytest.approx(14 / 20)
    assert overall["rule_reuse"] == pytest.approx(4 / 14)
    assert overall["tests_per_level"] == pytest.approx(6 / 4)
    assert overall["impasses"] == 2
    assert overall["search_depth_mean"] == pytest.approx(3.0)
    assert overall["goals"] is not None
    assert overall["goals"]["transitions"] == {"candidate": 2, "confirmed": 2, "refuted": 2}
    assert overall["hit_rate_halves"]["first"] == pytest.approx(3 / 6)


def test_old_logs_have_none_for_event_based_metrics() -> None:
    """Pre-slice-3: no level in entries, no search/wake/test/goal events."""
    records = [
        act(0, 0, "a0", programs=[{"id": "x", "pred": "p", "matched": True, "fit": 1.0}]),
        act(1, 0, "a1", programs=[{"id": "x", "pred": "p", "matched": False, "fit": 1.0}]),
        ev("level_up", 1, level=1),
    ]
    result = mm.compute(records, human={})
    for out in (result["games"][GAME], result["overall"]):
        for key in (
            "rule_reuse",
            "tests_per_level",
            "impasses",
            "search_depth_mean",
            "goals",
            "actions_vs_human",
        ):
            assert out[key] is None, key
    mm.format_table(result)
    assert mm.compute([], human={})["overall"]["hit_rate"] is None
