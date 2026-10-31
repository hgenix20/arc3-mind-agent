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


def test_metrics_count_modes() -> None:
    recs = [
        act(0, 0, "a"),
        act(1, 0, "b"),
        ev("drive_end", 2, modes={"sim": 5, "experiment": 2, "fallback": 1}),
        act(0, 0, "a", game="other"),
        ev("drive_end", 1, game="other", modes={"sim": 1}),
    ]
    out = mm.compute(recs, human={})
    assert out["games"][GAME]["modes"] == {"experiment": 2, "fallback": 1, "sim": 5}
    assert out["overall"]["modes"] == {"experiment": 2, "fallback": 1, "sim": 6}
    assert mm.compute([act(0, 0, "a")], human={})["games"][GAME]["modes"] is None


# Planner agent (v2) metrics: "Planner agent (v2) metrics" in the script's docstring.

BIN_LABELS = [
    "0.0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0",
]
ZERO_V2: Rec = {
    "calibration": [{"bin": b, "n": 0, "matched": 0, "rate": None} for b in BIN_LABELS],
    "moves_vs_typical": [],
    "gates": {"trips": 0, "by_gate": {}, "seconds_total": 0.0, "tokens_total": 0},
    "handoffs": {},
    "planner": {
        "calls": 0,
        "tokens": 0,
        "segments": 0,
        "mean_segment_len": None,
        "skills": {},
        "questions": {},
    },
    "forks": {"opened": 0, "failed": 0, "returns": 0, "root_replans": 0},
    "harm": {"events": 0, "by_kind": {}},
}


def v2act(
    step: int,
    level: int,
    *,
    p_step: float | None = None,
    prev_matched: bool | None = None,
    gate: Rec | None = None,
    action: tuple[int, int, int] = (1, 0, 0),
    game: str = GAME,
) -> Rec:
    """An action record with the planner agent's fields; ``gate`` None leaves the field out."""
    rec = act(step, level, f"f{step}", action=action, game=game)
    rec["p_step"] = p_step
    rec["prev_matched"] = prev_matched
    if gate is not None:
        rec["gate"] = gate
    return rec


def gate(seconds: float, tokens: int, trip: str | None = None) -> Rec:
    return {"seconds": seconds, "tokens": tokens, "trip": trip}


def consolidate(level: int, typical: float | None, game: str = GAME) -> Rec:
    return {
        "type": "consolidate",
        "game": game,
        "level": level,
        "promoted": 2,
        "sequences": 1,
        "typical": typical,
    }


def harm(step: int, kind: str, label: str, p: float, game: str = GAME) -> Rec:
    """A ``harm`` event; its ``kind`` field would collide with ``ev``'s first parameter."""
    return {"type": "harm", "game": game, "step": step, "kind": kind, "label": label, "p": p}


def v2_log() -> list[Rec]:
    """Level 0: 4 moves then a level up; level 1: 2 moves. Answers are worked out in the tests."""
    return [
        v2act(0, 0, p_step=0.95, gate=gate(0.5, 100)),
        v2act(1, 0, p_step=0.55, prev_matched=True, gate=gate(1.5, 200, "time")),
        ev("gate_trip", 1, gate="time", seconds=1.5, tokens=200),
        ev("fork_open", 1, id="f1", goal="g1", paths=2),
        v2act(2, 0, p_step=0.5, prev_matched=False, gate=gate(0.25, 0)),
        ev("handoff", 2, rule="H-b", method="replan", reason="segment broke"),
        ev(
            "planner_call",
            2,
            call=1,
            seconds=3.2,
            tokens=900,
            segment_len=3,
            skill=None,
            question="Q-FORK",
            dropped=[],
        ),
        ev("segment", 2, id="seg1", source="planner", length=3, p_path=0.42, ev=12.5, goal="g1"),
        v2act(3, 0, p_step=None, prev_matched=True, gate=gate(0.25, 50)),
        ev("branch_failed", 3, fork="f1", index=0, reason="miss that matters"),
        ev("return_route", 3, fork="f1", length=2),
        harm(3, "death", "A3", 0.9),
        ev("level_up", 3, level=1),
        consolidate(0, 8.0),
        v2act(4, 1, p_step=1.0, prev_matched=True, gate=gate(2.0, 300, "tokens")),
        ev("gate_trip", 4, gate="tokens", seconds=2.0, tokens=300),
        ev("fork_open", 4, id="f2", goal="g2", paths=3),
        v2act(5, 1, p_step=0.2, prev_matched=True),
        ev("replan_root", 5, reason="no untried fork"),
        harm(5, "health", "A1", 0.4),
        {"type": "longterm", "game": GAME, "game_id": GAME, "rules": 4, "sequences": 1},
    ]


def bins(out: Rec) -> dict[str, tuple[int, int, float | None]]:
    return {b["bin"]: (b["n"], b["matched"], b["rate"]) for b in out["calibration"]}


def test_v2_calibration_pairs_p_step_with_the_next_prev_matched() -> None:
    got = bins(mm.v2_metrics(v2_log()))
    assert list(got) == BIN_LABELS
    # (s0 .95, s1 hit) and (s4 1.0, s5 hit) fill the last bin; (s1 .55, s2 miss) and
    # (s2 .5, s3 hit) fill 0.5-0.6. s3 has no p_step and the level_up follows it.
    assert got["0.9-1.0"] == (2, 2, 1.0)
    assert got["0.5-0.6"] == (2, 1, 0.5)
    rest = [b for b in BIN_LABELS if b not in ("0.9-1.0", "0.5-0.6")]
    assert all(got[b] == (0, 0, None) for b in rest)


def test_v2_typical_gates_handoffs_planner_forks_harm() -> None:
    out = mm.v2_metrics(v2_log())
    assert out["moves_vs_typical"] == [{"level": 0, "moves": 4, "typical": 8.0, "ratio": 0.5}]
    # The trips at steps 1 and 4 sit in the gate field and in an event: each counts once.
    assert out["gates"] == {
        "trips": 2,
        "by_gate": {"time": 1, "tokens": 1},
        "seconds_total": 4.5,
        "tokens_total": 650,
    }
    assert out["handoffs"] == {"H-b": 1}
    assert out["planner"] == {
        "calls": 1,
        "tokens": 900,
        "segments": 1,
        "mean_segment_len": 3.0,
        "skills": {},
        "questions": {"Q-FORK": 1},
    }
    assert out["forks"] == {"opened": 2, "failed": 1, "returns": 1, "root_replans": 1}
    assert out["harm"] == {"events": 2, "by_kind": {"death": 1, "health": 1}}


def test_v2_zero_shape_without_planner_records() -> None:
    assert mm.v2_metrics([]) == ZERO_V2
    assert mm.v2_metrics(slice3_log()) == ZERO_V2
    result = mm.compute(slice3_log(), human={})
    assert result["games"][GAME]["v2"] == ZERO_V2
    assert result["overall"]["v2"] == ZERO_V2
    assert mm.compute([], human={})["overall"]["v2"] == ZERO_V2


def test_v2_calibration_half_right_reports_bin_half() -> None:
    """Spec 4.13: predictions right half the time give the 0.5 bin a rate of 0.5."""
    records = [v2act(0, 0, p_step=0.5)]
    records += [v2act(i, 0, p_step=0.5, prev_matched=i % 2 == 1) for i in range(1, 21)]
    assert bins(mm.v2_metrics(records))["0.5-0.6"] == (20, 10, 0.5)


def test_v2_calibration_skips_boundaries_and_unusable_values() -> None:
    not_bool = v2act(7, 1, p_step=0.3)
    not_bool["prev_matched"] = 1
    records = [
        v2act(0, 0, p_step=0.3),
        v2act(1, 0, p_step=0.3, prev_matched=True),  # pairs s0: hit
        ev("game_over", 1, level=0),
        v2act(2, 0, action=(0, 0, 0)),  # the RESET after a game over: across the event
        v2act(3, 0, p_step=0.3, prev_matched=True),  # s2 has no p_step
        v2act(4, 0, p_step=1.5, prev_matched=False),  # pairs s3: miss
        v2act(5, 0, p_step=0.3, prev_matched=True),  # s4's p_step is out of range
        ev("level_up", 5, level=1),
        v2act(6, 1, p_step=0.3, prev_matched=True),  # across the level_up
        not_bool,  # prev_matched is not a bool
        v2act(8, 2, p_step=0.3, prev_matched=True),  # the level changed with no event
    ]
    got = bins(mm.v2_metrics(records))
    assert got["0.3-0.4"] == (2, 1, 0.5)
    assert sum(n for n, _, _ in got.values()) == 2


def test_v2_calibration_bin_edges() -> None:
    probabilities = [0.0, 0.1, 0.95, 1.0, 0.3, 0.7]
    records = [v2act(i, 0, p_step=p, prev_matched=i > 0) for i, p in enumerate(probabilities)]
    records.append(v2act(len(probabilities), 0, prev_matched=True))
    got = bins(mm.v2_metrics(records))
    assert got["0.0-0.1"] == (1, 1, 1.0)
    assert got["0.1-0.2"] == (1, 1, 1.0)
    assert got["0.3-0.4"] == (1, 1, 1.0)
    assert got["0.7-0.8"] == (1, 1, 1.0)
    assert got["0.9-1.0"] == (2, 2, 1.0)


def test_v2_moves_vs_typical_in_either_event_order_and_after_a_death() -> None:
    records = [
        v2act(0, 0),
        v2act(1, 0),
        v2act(2, 0),
        ev("level_up", 2, level=1),
        consolidate(0, 6),  # the level_up closed the instance: 3 moves
        v2act(3, 1),
        v2act(4, 1),
        ev("game_over", 4, level=1),
        v2act(5, 1, action=(0, 0, 0)),
        v2act(6, 1),
        v2act(7, 1),
        consolidate(1, 4),  # before its level_up: the records since the death, 3
        ev("level_up", 7, level=2),
        v2act(8, 2),
        consolidate(2, None),  # no typical yet
        ev("level_up", 8, level=3),
        v2act(9, 3),
        v2act(10, 3),
        ev("level_up", 10, level=4),
        consolidate(3, 0),  # a zero typical gives no ratio
    ]
    assert mm.v2_metrics(records)["moves_vs_typical"] == [
        {"level": 0, "moves": 3, "typical": 6, "ratio": 0.5},
        {"level": 1, "moves": 3, "typical": 4, "ratio": 0.75},
        {"level": 2, "moves": 1, "typical": None, "ratio": None},
        {"level": 3, "moves": 2, "typical": 0, "ratio": None},
    ]


def test_v2_moves_vs_typical_falls_back_to_the_instance_a_game_over_closed() -> None:
    """A boundary directly before the consolidate hands over the instance it closed."""
    records = [v2act(0, 0), v2act(1, 0), ev("game_over", 1, level=0), consolidate(0, 4)]
    assert mm.v2_metrics(records)["moves_vs_typical"] == [
        {"level": 0, "moves": 2, "typical": 4, "ratio": 0.5}
    ]


def test_v2_events_missing_their_name_field_count_under_none() -> None:
    """The script's convention (as ``drop_reasons``): a missing name is the key ``"None"``."""
    records = [ev("gate_trip", 1), ev("handoff", 1), {"type": "harm", "game": GAME, "step": 1}]
    out = mm.v2_metrics(records)
    assert out["gates"]["trips"] == 1 and out["gates"]["by_gate"] == {"None": 1}
    assert out["handoffs"] == {"None": 1}
    assert out["harm"] == {"events": 1, "by_kind": {"None": 1}}


def test_v2_gate_trips_count_once_per_step_and_gate() -> None:
    records = [
        v2act(0, 0, gate=gate(1.0, 10, "time")),
        ev("gate_trip", 0, gate="time", seconds=1.0, tokens=10),  # the same trip as the field
        v2act(1, 0, gate=gate(0.5, 5, "tokens")),  # field only
        v2act(2, 0, gate=gate(0.5, 5)),
        ev("gate_trip", 2, gate="time", seconds=0.5, tokens=5),  # event only
        v2act(3, 0, gate={"seconds": "slow", "tokens": None, "trip": None}),  # unusable
    ]
    assert mm.v2_metrics(records)["gates"] == {
        "trips": 3,
        "by_gate": {"time": 2, "tokens": 1},
        "seconds_total": 2.0,
        "tokens_total": 20,
    }


def test_v2_planner_skills_questions_and_segment_lengths() -> None:
    def call(step: int, tokens: int | None, skill: object, question: str | None) -> Rec:
        return ev(
            "planner_call",
            step,
            call=step + 1,
            seconds=2.0,
            tokens=tokens,
            segment_len=2,
            skill=skill,
            question=question,
            dropped=[],
        )

    records = [
        v2act(0, 0),
        call(0, 500, "S-AROUND", None),
        call(1, 700, {"name": "S-COLLECT", "args": {}}, "Q-HARM"),
        call(2, None, None, "Q-HARM"),
        ev("segment", 1, id="a", source="planner", length=4, p_path=0.5, ev=1.0, goal="g"),
        ev("segment", 2, id="b", source="search", length=2, p_path=0.9, ev=2.0, goal="g"),
        ev("segment", 3, id="c", source="skill", length=None, p_path=None, ev=None, goal="g"),
    ]
    assert mm.v2_metrics(records)["planner"] == {
        "calls": 3,
        "tokens": 1200,
        "segments": 3,
        "mean_segment_len": 3.0,
        "skills": {"S-AROUND": 1, "S-COLLECT": 1},
        "questions": {"Q-HARM": 2},
    }


def test_v2_in_compute_matches_v2_metrics_and_pools_overall() -> None:
    other = "yy88-00000000"
    records = v2_log() + [{**r, "game": other} for r in v2_log()]
    result = mm.compute(records, human={})
    assert result["games"][GAME]["v2"] == mm.v2_metrics(v2_log())
    assert result["games"][other]["v2"] == mm.v2_metrics(v2_log())
    overall = result["overall"]["v2"]
    assert bins(overall)["0.9-1.0"] == (4, 4, 1.0)
    assert bins(overall)["0.5-0.6"] == (4, 2, 0.5)
    assert overall["moves_vs_typical"] == [
        {"level": 0, "moves": 4, "typical": 8.0, "ratio": 0.5},
        {"level": 0, "moves": 4, "typical": 8.0, "ratio": 0.5},
    ]
    assert overall["gates"] == {
        "trips": 4,
        "by_gate": {"time": 2, "tokens": 2},
        "seconds_total": 9.0,
        "tokens_total": 1300,
    }
    assert overall["handoffs"] == {"H-b": 2}
    assert overall["planner"]["calls"] == 2
    assert overall["planner"]["tokens"] == 1800
    assert overall["planner"]["mean_segment_len"] == 3.0
    assert overall["forks"] == {"opened": 4, "failed": 2, "returns": 2, "root_replans": 2}
    assert overall["harm"] == {"events": 4, "by_kind": {"death": 2, "health": 2}}
    mm.format_table(result)


def test_main_writes_v2_into_the_json_report(tmp_path: Path) -> None:
    path = tmp_path / f"{GAME}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in v2_log()) + "\n", encoding="utf-8")
    out = tmp_path / "metrics.json"
    result = mm.main([str(path), "--json", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["games"][GAME]["v2"]["gates"]["trips"] == 2
    assert saved["overall"]["v2"]["planner"]["calls"] == 1
    assert result["games"][GAME]["v2"] == mm.v2_metrics(v2_log())
