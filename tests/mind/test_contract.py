"""Reply contract: valid replies, fences, drops with reasons, caps."""

from __future__ import annotations

import json

import numpy as np

from arcagent.core.types import Action
from arcagent.mind.contract import (
    MAX_LABEL_CHARS,
    MAX_MODEL_CHARS,
    MAX_PLAN_STEPS,
    MAX_RULE_CHARS,
    QUESTION_IDS,
    SKILL_NAMES,
    FactClaim,
    Reply,
    SkillCall,
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


def test_single_object_facts_and_goals_count_as_one_item_lists() -> None:
    # Qwen often sends one fact or goal as a bare object instead of a list of one.
    text = json.dumps(
        {
            "facts": {"action": "A1", "object": 1, "effect": "move", "dx": 0, "dy": -1},
            "goals": {"kind": "reach", "a": 1, "b": 4},
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    assert r.claims == (FactClaim("A1", SCENE.things[0].sig, (Effect("move", 0, -1),)),)
    assert r.goals == (Goal("reach", a=SCENE.things[0].sig, b=SCENE.things[1].sig),)
    assert r.dropped == ()


def test_ids_written_as_hash_strings_parse() -> None:
    text = json.dumps(
        {
            "facts": [{"action": "A1", "object": "#1", "effect": "move", "dx": "1", "dy": 0}],
            "goals": [{"kind": "reach", "a": "#1", "b": "4"}],
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    assert r.claims == (FactClaim("A1", SCENE.things[0].sig, (Effect("move", 1, 0),)),)
    assert r.goals == (Goal("reach", a=SCENE.things[0].sig, b=SCENE.things[1].sig),)
    assert r.dropped == ()


def test_move_claim_needs_an_object() -> None:
    # A wide move fact would move every unmodeled thing, walls included, in play-outs.
    text = json.dumps(
        {
            "facts": [
                {"action": "A1", "effect": "move", "dx": 0, "dy": -1},
                {"action": "A2", "object": "any", "effect": "none"},
            ]
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    assert r.claims == (FactClaim("A2", "any", (Effect("none"),)),)
    assert r.dropped == ("fact: a move names its object",)


def test_colors_in_screen_notation_cover_and_no_change_parse() -> None:
    text = json.dumps(
        {
            "facts": [
                {"action": "click #4", "object": 4, "effect": "recolor", "color": "c9"},
                {"action": "A2", "object": "any", "effect": "no change"},
            ],
            "goals": [{"kind": "clear", "color": "c11"}, {"kind": "cover", "a": 1, "b": 4}],
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    x, y = click_cell(SCENE.things[1])
    assert r.claims == (
        FactClaim(
            action_label(Action(6, x, y), SCENE), SCENE.things[1].sig, (Effect("recolor", color=9),)
        ),
        FactClaim("A2", "any", (Effect("none"),)),
    )
    assert r.goals == (
        Goal("clear", color=11),
        Goal("reach", a=SCENE.things[0].sig, b=SCENE.things[1].sig),
    )
    assert r.dropped == ()


def test_disappears_and_blocked_effects_parse() -> None:
    text = json.dumps(
        {
            "facts": [
                {"action": "A1", "object": 1, "effect": "disappears"},
                {"action": "A2", "object": 1, "effect": "blocked"},
            ]
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    sig = SCENE.things[0].sig
    assert r.claims == (
        FactClaim("A1", sig, (Effect("disappear"),)),
        FactClaim("A2", sig, (Effect("none"),)),
    )
    assert r.dropped == ()


# The v2 planner contract (plan section 3): segment, skill, question, labels.


def test_new_reply_fields_default_empty_and_constants_are_fixed() -> None:
    r = Reply()
    assert r.skill is None and r.question == "" and r.labels == ()
    assert SKILL_NAMES == ("S-AROUND", "S-COLLECT", "S-EXPLORE", "S-ATTACK", "S-WAIT")
    assert QUESTION_IDS == (
        "Q-BLOCK",
        "Q-REWARD",
        "Q-HARM",
        "Q-FORK",
        "Q-CHECK",
        "Q-SAFE",
        "Q-UNKNOWN",
    )


def test_segment_is_read_as_the_plan() -> None:
    text = json.dumps({"segment": [{"action": "A3", "expect": "left"}, {"action": "click #4"}]})
    r = parse_reply(text, SCENE, AVAIL)
    x, y = click_cell(SCENE.things[1])
    assert [p.action for p in r.plan] == [Action(3), Action(6, x, y)]
    assert r.plan[0].expect == "left" and r.dropped == ()


def test_segment_wins_over_plan_when_both_are_given() -> None:
    text = json.dumps({"segment": [{"action": "A3"}], "plan": [{"action": "A1"}, {"action": "A2"}]})
    r = parse_reply(text, SCENE, AVAIL)
    assert [p.action for p in r.plan] == [Action(3)]
    assert r.dropped == ("plan ignored: segment given",)


def test_segment_steps_beyond_max_steps_drop_with_one_reason() -> None:
    text = json.dumps({"segment": [{"action": "A1"}] * 6})
    r = parse_reply(text, SCENE, AVAIL, max_steps=4)
    assert len(r.plan) == 4
    assert r.dropped == ("segment capped at 4 steps",)
    full = parse_reply(text, SCENE, AVAIL, max_steps=6)
    assert len(full.plan) == 6 and full.dropped == ()
    assert len(parse_reply(json.dumps({"plan": [{"action": "A1"}] * 12}), SCENE, AVAIL).plan) == 8
    assert len(parse_reply(text, SCENE, AVAIL, max_steps=0).plan) == 1  # below 1 reads as 1


def test_skill_parses_with_args_and_missing_args_become_empty() -> None:
    text = json.dumps({"skill": {"name": "S-AROUND", "args": {"goal": "g1", "avoid": [7]}}})
    r = parse_reply(text, SCENE, AVAIL)
    assert r.skill == SkillCall("S-AROUND", {"goal": "g1", "avoid": [7]})
    assert r.plan == () and r.dropped == ()
    bare = parse_reply(json.dumps({"skill": {"name": "s_collect"}}), SCENE, AVAIL)
    assert bare.skill == SkillCall("S-COLLECT", {}) and bare.dropped == ()


def test_unknown_skill_and_bad_skill_shapes_drop_with_a_reason() -> None:
    for raw, reason in (
        ({"name": "S-FLY", "args": {}}, "unknown skill: S-FLY"),
        ({"args": {}}, "unknown skill: None"),
        ("S-AROUND", "skill is not an object"),
        ({"name": "S-WAIT", "args": [1, 2]}, "skill args is not an object"),
    ):
        r = parse_reply(json.dumps({"skill": raw}), SCENE, AVAIL)
        assert r.skill is None and r.dropped == (reason,), raw


def test_skill_is_dropped_when_a_segment_is_given() -> None:
    text = json.dumps({"segment": [{"action": "A1"}], "skill": {"name": "S-EXPLORE"}})
    r = parse_reply(text, SCENE, AVAIL)
    assert [p.action for p in r.plan] == [Action(1)]
    assert r.skill is None and r.dropped == ("skill dropped: segment given",)
    # An empty segment leaves the skill standing: there is no plan to prefer.
    none_valid = json.dumps({"segment": [{"action": "A0"}], "skill": {"name": "S-EXPLORE"}})
    r2 = parse_reply(none_valid, SCENE, AVAIL)
    assert r2.plan == () and r2.skill == SkillCall("S-EXPLORE", {}) and len(r2.dropped) == 1


def test_question_parses_and_unknown_question_drops() -> None:
    r = parse_reply(json.dumps({"question": "Q-HARM"}), SCENE, AVAIL)
    assert r.question == "Q-HARM" and r.dropped == ()
    lower = parse_reply(json.dumps({"question": " q_fork "}), SCENE, AVAIL)
    assert lower.question == "Q-FORK" and lower.dropped == ()
    bad = parse_reply(json.dumps({"question": "Q-WHY"}), SCENE, AVAIL)
    assert bad.question == "" and bad.dropped == ("unknown question: Q-WHY",)
    not_text = parse_reply(json.dumps({"question": 7}), SCENE, AVAIL)
    assert not_text.question == "" and not_text.dropped == ("question is not a string",)
    assert parse_reply(json.dumps({"question": ""}), SCENE, AVAIL).dropped == ()


def test_labels_parse_truncate_and_drop_off_screen_or_empty() -> None:
    text = json.dumps(
        {
            "labels": [
                {"object": 1, "label": "player"},
                {"object": "#4", "label": "  door   frame " + "x" * 40},
                {"object": 99, "label": "ghost"},
                {"object": 7, "label": ""},
                {"object": "seven", "label": "wall"},
                "player",
                {"object": 1, "label": "player"},
            ]
        }
    )
    r = parse_reply(text, SCENE, AVAIL)
    assert r.labels == ((1, "player"), (4, ("door frame " + "x" * 40)[:MAX_LABEL_CHARS]))
    assert len(r.labels[1][1]) == MAX_LABEL_CHARS
    assert r.dropped == (
        "label object not on screen: #99",
        "label for #7 needs text",
        "label needs an integer object id",
        "label is not an object",
    )
    lone = parse_reply(json.dumps({"labels": {"object": 7, "label": "corner"}}), SCENE, AVAIL)
    assert lone.labels == ((7, "corner"),) and lone.dropped == ()


def test_full_planner_reply_parses_every_part() -> None:
    text = json.dumps(
        {
            "model": "A1 moves #1 up.",
            "facts": [{"action": "A1", "object": 1, "effect": "move", "dx": 0, "dy": -1}],
            "goals": [{"kind": "reach", "a": 1, "b": 4}],
            "segment": [{"action": "A1", "expect": "#1 up"}, {"action": "A1", "expect": "#1 up"}],
            "question": "Q-SAFE",
            "labels": [{"object": 4, "label": "door"}],
            "notes": "two up then check",
        }
    )
    r = parse_reply(text, SCENE, AVAIL, max_steps=5)
    assert r.model == "A1 moves #1 up." and len(r.claims) == 1 and len(r.goals) == 1
    assert [p.action for p in r.plan] == [Action(1), Action(1)]
    assert r.skill is None and r.question == "Q-SAFE" and r.labels == ((4, "door"),)
    assert r.notes == "two up then check" and r.dropped == ()
