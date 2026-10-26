"""Reply contract: valid replies, fences, drops with reasons, caps."""

from __future__ import annotations

import json

import numpy as np

from arcagent.core.types import Action
from arcagent.mind.contract import (
    MAX_MODEL_CHARS,
    MAX_PLAN_STEPS,
    MAX_RULE_CHARS,
    FactClaim,
    parse_reply,
)
from arcagent.mind.facts import Effect, action_label
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Scene, Thing, click_cell, shape_of

AVAIL = [1, 2, 3, 4, 5, 6]


def _thing(i: int, color: int, x: int, y: int, *, w: int = 2, h: int = 2) -> Thing:
    cells = frozenset((x + dx, y + dy) for dx in range(w) for dy in range(h))
    return Thing(i, color, shape_of(cells), f"c{color}:{i:010d}", x, y, cells)


SCENE = Scene(
    (_thing(1, 1, 10, 10), _thing(4, 2, 20, 30, w=3, h=1), _thing(7, 4, 0, 0)),
    background=0,
    blocked=frozenset(),
    terrain=np.zeros((64, 64), dtype=np.int8),
)


def test_valid_reply_with_reach_goal_and_three_step_plan() -> None:
    text = json.dumps(
        {
            "goals": [{"kind": "reach", "a": 1, "b": 4, "why": "x"}],
            "plan": [
                {"action": "A3", "expect": "moves left"},
                {"action": "click #4", "expect": "toggles"},
                {"action": "A1", "expect": ""},
            ],
            "notes": "ok",
            "extra": 5,
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    assert r.goals == (Goal("reach", a=SCENE.things[0].sig, b=SCENE.things[1].sig),)
    x, y = click_cell(SCENE.things[1])
    assert [p.action for p in r.plan] == [Action(3), Action(6, x, y), Action(1)]
    assert r.plan[0].expect == "moves left" and r.notes == "ok" and r.dropped == ()


def test_fenced_json_with_prose() -> None:
    text = 'Sure! Here you go:\n```json\n{"plan": [{"action": "A2"}]}\n```\nHope it helps {x}.'
    assert parse_reply(text, SCENE, AVAIL).plan[0].action == Action(2)


def test_unknown_id_dropped_with_reason() -> None:
    text = json.dumps(
        {
            "goals": [{"kind": "reach", "a": 1, "b": 99}],
            "plan": [{"action": "click #99"}, {"action": "A1"}],
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    assert r.goals == () and len(r.plan) == 1
    assert len(r.dropped) == 2 and any("#99" in d for d in r.dropped)


def test_unavailable_and_invalid_actions_dropped() -> None:
    text = json.dumps(
        {"plan": [{"action": "A5"}, {"action": "A0"}, {"action": "A6"}, {"action": "jump"}]}
    )
    r = parse_reply(text, SCENE, [1, 2])
    assert r.plan == () and len(r.dropped) == 4
    r2 = parse_reply('{"plan": [{"action": "click #1"}]}', SCENE, [1, 2])
    assert r2.plan == () and "not available" in r2.dropped[0]


def test_plan_wins_over_test() -> None:
    text = json.dumps({"plan": [{"action": "A1"}], "test": {"action": "A2", "question": "q?"}})
    r = parse_reply(text, SCENE, AVAIL)
    assert len(r.plan) == 1 and r.test is None
    only = parse_reply('{"test": {"action": "A2", "question": "q?"}}', SCENE, AVAIL)
    assert only.test is not None and only.test.action == Action(2) and only.test.expect == "q?"


def test_garbage_and_empty_return_empty_reply() -> None:
    for text in ["", "no json here", "{broken", "[1, 2]", "{}", '{"goals": 5, "plan": "x"}']:
        r = parse_reply(text, SCENE, AVAIL)
        assert r.goals == () and r.plan == () and r.test is None


def test_plan_capped_at_eight() -> None:
    text = json.dumps({"plan": [{"action": "A1"}] * 12})
    r = parse_reply(text, SCENE, AVAIL)
    assert len(r.plan) == MAX_PLAN_STEPS and r.dropped


def test_match_goal_keeps_regions_and_bad_regions_drop() -> None:
    good = {"kind": "match", "region_a": [0, 0, 8, 8], "region_b": [10, 0, 18, 8]}
    bad = {"kind": "match", "region_a": [0, 0, 80, 8], "region_b": [10, 0, 18, 8]}
    r = parse_reply(json.dumps({"goals": [good, bad]}), SCENE, AVAIL)
    assert r.goals == (Goal("match", region_a=(0, 0, 8, 8), region_b=(10, 0, 18, 8)),)
    assert len(r.dropped) == 1


def test_clear_and_count_map_colors() -> None:
    goals = [
        {"kind": "clear", "color": 9},
        {"kind": "count", "color": 8, "n": 0},
        {"kind": "clear", "color": 16},
        {"kind": "count", "color": 8},
        {"kind": "clear", "color": True},
        {"kind": "wat"},
    ]
    r = parse_reply(json.dumps({"goals": goals}), SCENE, AVAIL)
    assert r.goals == (Goal("clear", color=9), Goal("count", color=8, n=0))
    assert len(r.dropped) == 4


def test_overlong_click_id_drops_only_that_step() -> None:
    text = json.dumps({"plan": [{"action": "click #" + "9" * 5000}, {"action": "A1"}]})
    r = parse_reply(text, SCENE, AVAIL)
    assert [p.action for p in r.plan] == [Action(1)] and len(r.dropped) == 1


def test_deeply_nested_object_is_skipped_for_a_later_valid_one() -> None:
    text = "{" * 100000 + ' junk {"plan": [{"action": "A2"}]}'
    r = parse_reply(text, SCENE, AVAIL)
    assert r.plan[0].action == Action(2)


def test_facts_parse_into_claims() -> None:
    text = json.dumps(
        {
            "facts": [
                {"action": "A1", "object": 1, "effect": "move", "dx": 0, "dy": -1},
                {"action": "click #4", "object": 4, "effect": "recolor", "color": 9},
                {"action": "A2", "object": "any", "effect": "none"},
            ]
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    sig1, sig4 = SCENE.things[0].sig, SCENE.things[1].sig
    assert r.claims[0] == FactClaim("A1", sig1, (Effect("move", 0, -1),))
    x, y = click_cell(SCENE.things[1])
    assert r.claims[1] == FactClaim(
        action_label(Action(6, x, y), SCENE), sig4, (Effect("recolor", color=9),)
    )
    assert r.claims[2] == FactClaim("A2", "any", (Effect("none"),))
    assert r.dropped == ()


def test_bad_fact_dropped_with_reason() -> None:
    text = json.dumps(
        {
            "facts": [
                {"action": "A1", "object": 99, "effect": "move", "dx": 1, "dy": 0},
                {"action": "A1", "object": 1, "effect": "fly"},
                {"action": "A0", "object": 1, "effect": "none"},
                {"action": "A1", "object": 1, "effect": "move", "dx": "x"},
                "nope",
            ]
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    assert r.claims == () and len(r.dropped) == 5
    assert any("#99" in d for d in r.dropped)


def test_model_and_rule_clipped() -> None:
    text = json.dumps({"model": "m" * 5000, "rule": "def rule(o, a):\n" + "x" * 9000})
    r = parse_reply(text, SCENE, AVAIL)
    assert len(r.model) == MAX_MODEL_CHARS and len(r.rule) == MAX_RULE_CHARS
    assert r.rule.startswith("def rule")
