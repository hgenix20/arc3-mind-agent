"""Tests for the long-term store (spec 4.2, 4.12b, 4.12c and the 4.7 horizon rule)."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from arcagent.mind import longterm
from arcagent.mind.facts import Effect, FactKey
from arcagent.mind.longterm import (
    STATEMENT_MAX_CHARS,
    STORE_VERSION,
    CanonicalKey,
    LongTermStore,
    RuleRecord,
    SequenceRecord,
    canonical_key,
    effect_text,
    role_of,
    rule_id,
    rule_probability,
    rule_state,
    shape_of,
    statement_for,
    tag_of,
)

COLOUR = re.compile(r"c\d+:")
SHAPE = "ab12cdef01"
OTHER = "ff00ff00ff"
MOVE = (Effect("move", 1, 0),)


def rule(key: CanonicalKey, **over: object) -> RuleRecord:
    base = RuleRecord(
        id=rule_id(key),
        kind="action",
        statement=f"{key[0]} does {key[3]} to a {key[1]} thing",
        canonical_key=key,
        effect=key[3],
        probability=0.5,
        confirmations=0,
        contradictions=0,
        state="hypothesis",
        scope="working",
        games_seen=["g1"],
        created_game="g1",
        created_level=0,
        last_revised=1.0,
        harm="none",
        source="observed",
    )
    return replace(base, **over)


def sequence(sid: str, goal: str, runs: int, successes: int) -> SequenceRecord:
    return SequenceRecord(
        id=sid,
        goal_id=goal,
        moves=[[1, -1, -1], [6, 3, 4]],
        game="g1",
        level=1,
        runs=runs,
        successes=successes,
        deterministic=False,
    )


def store_at(tmp_path: Path, clock: float = 50.0) -> LongTermStore:
    return LongTermStore(tmp_path / "lt.json", clock=lambda: clock)


# --- file round trip -------------------------------------------------------


def test_round_trip_keeps_every_field(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    r = rule(
        ("A3", SHAPE, "any", "move"),
        kind="object",
        probability=0.8333,
        confirmations=4,
        contradictions=0,
        scope="long",
        games_seen=["g1", "g2"],
        created_level=2,
        last_revised=123.5,
        harm="death",
        source="loaded",
    )
    store.rules[r.canonical_key] = r
    s = sequence("seq1", "reach a>b", 3, 2)
    store.sequences[s.id] = s
    store.completions = {"g1": {0: [12, 14], 1: [30]}}
    store.net_state = {"w": [0.1, 0.2], "bias": 0.0}
    store.save()

    again = store_at(tmp_path)
    again.load()
    assert again.rules == {r.canonical_key: r}
    assert isinstance(next(iter(again.rules)), tuple)
    assert isinstance(next(iter(again.rules.values())).canonical_key, tuple)
    assert again.sequences == {"seq1": s}
    assert again.completions == {"g1": {0: [12, 14], 1: [30]}}
    assert again.net_state == {"w": [0.1, 0.2], "bias": 0.0}
    assert again.load_error is None


def test_file_is_one_versioned_object(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.promote(rule(("A3", SHAPE, "any", "move")))
    store.save()
    data = json.loads((tmp_path / "lt.json").read_text(encoding="utf-8"))
    assert data["version"] == STORE_VERSION
    assert set(data) == {"version", "rules", "sequences", "completions", "net_state"}
    assert data["rules"][0]["canonical_key"] == ["A3", SHAPE, "any", "move"]
    assert data["net_state"] is None


def test_missing_file_is_an_empty_store(tmp_path: Path) -> None:
    store = store_at(tmp_path / "deeper")
    store.promote(rule(("A3", SHAPE, "any", "move")))
    store.load()
    assert store.rules == {}
    assert store.sequences == {}
    assert store.completions == {}
    assert store.net_state is None
    assert store.load_error is None
    assert not (tmp_path / "deeper" / "lt.json").exists()


def test_save_creates_parent_and_leaves_no_temp_file(tmp_path: Path) -> None:
    store = store_at(tmp_path / "a" / "b")
    store.record_completion("g1", 0, 7)
    store.save()
    assert [p.name for p in (tmp_path / "a" / "b").iterdir()] == ["lt.json"]


@pytest.mark.parametrize(
    "content",
    [
        b"\x00garbage{",
        b"[]",
        json.dumps({"version": 99, "rules": [], "sequences": [], "completions": {}}).encode(),
        json.dumps(
            {
                "version": 1,
                "rules": [{"id": "x"}],
                "sequences": [],
                "completions": {},
                "net_state": None,
            }
        ).encode(),
        json.dumps(
            {
                "version": 1,
                "rules": [],
                "sequences": [],
                "completions": {"g1": {"zero": [1]}},
                "net_state": None,
            }
        ).encode(),
    ],
)
def test_corrupt_file_is_renamed_and_treated_as_empty(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "lt.json"
    path.write_bytes(content)
    store = store_at(tmp_path)
    store.promote(rule(("A3", SHAPE, "any", "move")))
    store.load()
    assert store.rules == {} and store.sequences == {} and store.completions == {}
    assert not path.exists()
    assert (tmp_path / "lt.json.bad").read_bytes() == content
    assert store.load_error
    store.save()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == STORE_VERSION


def test_duplicate_rule_keys_in_a_file_are_corrupt(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    r = store.promote(rule(("A3", SHAPE, "any", "move")))
    store.save()
    path = tmp_path / "lt.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["rules"].append(data["rules"][0])
    path.write_text(json.dumps(data), encoding="utf-8")
    store.load()
    assert store.rules == {}
    assert store.load_error and "duplicate" in store.load_error
    assert r.canonical_key == ("A3", SHAPE, "any", "move")


def test_load_trims_an_oversized_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = store_at(tmp_path)
    for i, p in enumerate((0.5, 0.75, 0.6667)):
        hits = {0.5: 0, 0.75: 2, 0.6667: 1}[p]
        store.promote(rule(("A1", f"s{i}", "any", "move"), confirmations=hits))
    store.save()
    monkeypatch.setattr(longterm, "MAX_RULES", 2)
    again = store_at(tmp_path)
    again.load()
    assert sorted(r.probability for r in again.rules.values()) == [0.6667, 0.75]


# --- probability, state and promote ------------------------------------------


def test_rule_probability_and_state() -> None:
    assert rule_probability(0, 0) == 0.5
    assert rule_probability(5, 0) == 0.8571
    assert rule_probability(4, 0) == 0.8333
    assert rule_probability(0, 5) == 0.1429
    assert rule_state(0.5, 0, 0) == "hypothesis"
    assert rule_state(0.8571, 5, 0) == "fact"
    assert rule_state(0.8333, 4, 0) == "hypothesis"
    assert rule_state(0.8571, 6, 1) == "hypothesis"
    assert rule_state(0.1429, 0, 5) == "retired"
    assert rule_state(0.8, 5, 0) == "fact"
    assert rule_state(0.2999, 0, 0) == "retired"
    with pytest.raises(ValueError, match="negative"):
        rule_probability(-1, 0)


def test_promote_inserts_with_derived_fields(tmp_path: Path) -> None:
    store = store_at(tmp_path, clock=77.0)
    given = rule(
        ("A3", SHAPE, "any", "move"),
        probability=0.99,
        confirmations=2,
        contradictions=0,
        state="fact",
        scope="working",
        games_seen=[],
        created_game="g7",
        last_revised=1.0,
    )
    stored = store.promote(given)
    assert store.rules[("A3", SHAPE, "any", "move")] is stored
    assert stored.probability == 0.75
    assert stored.state == "hypothesis"
    assert stored.scope == "long"
    assert stored.games_seen == ["g7"]
    assert stored.last_revised == 77.0
    assert stored.id == given.id and stored.statement == given.statement
    assert given.probability == 0.99  # the caller's record is left alone


def test_promote_merges_counts_and_recomputes_probability_and_state(tmp_path: Path) -> None:
    store = store_at(tmp_path, clock=10.0)
    key: CanonicalKey = ("A3", SHAPE, "any", "move")
    first = store.promote(rule(key, confirmations=3, games_seen=["g1"], id="first"))
    assert (first.probability, first.state) == (0.8, "hypothesis")

    second = store.promote(
        rule(key, confirmations=2, games_seen=["g2"], created_game="g2", id="second")
    )
    assert second.id == "first"
    assert (second.confirmations, second.contradictions) == (5, 0)
    assert (second.probability, second.state) == (0.8571, "fact")
    assert second.games_seen == ["g1", "g2"]
    assert second.last_revised == 10.0
    assert len(store.rules) == 1

    third = store.promote(rule(key, contradictions=1, games_seen=["g1"]))
    assert (third.confirmations, third.contradictions) == (5, 1)
    assert (third.probability, third.state) == (0.75, "hypothesis")

    fourth = store.promote(rule(key, contradictions=20))
    assert (fourth.probability, fourth.state) == (0.2143, "retired")
    assert fourth.games_seen == ["g1", "g2"]


def test_promote_keeps_a_harm_kind_once_seen(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    key: CanonicalKey = ("any", SHAPE, "touch", "game_over")
    store.promote(rule(key, harm="none"))
    assert store.promote(rule(key, harm="death")).harm == "death"
    assert store.promote(rule(key, harm="none")).harm == "death"


def test_promote_rejects_a_key_that_names_a_colour(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match="colour"):
        store.promote(rule(("A3", f"c1:{SHAPE}", "any", "move")))
    with pytest.raises(ValueError, match="colour"):
        store.promote(rule(("any", SHAPE, "touch:3", "game_over")))
    assert store.rules == {}


def test_rules_are_bounded_by_lowest_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(longterm, "MAX_RULES", 2)
    store = store_at(tmp_path)
    store.promote(rule(("A1", "s0", "any", "move"), confirmations=2))  # 0.75
    store.promote(rule(("A1", "s1", "any", "move"), confirmations=0))  # 0.5
    store.promote(rule(("A1", "s2", "any", "move"), confirmations=6))  # 0.875
    assert set(store.rules) == {("A1", "s0", "any", "move"), ("A1", "s2", "any", "move")}
    # A newcomer that is itself the lowest is the one dropped.
    low = store.promote(rule(("A1", "s3", "any", "move"), contradictions=3))
    assert low.canonical_key not in store.rules
    assert len(store.rules) == 2


# --- canonical key and statements -----------------------------------------------


@pytest.mark.parametrize(
    ("fact_key", "effects", "expected"),
    [
        (("A3", f"c1:{SHAPE}", "any"), MOVE, ("A3", SHAPE, "any", "move")),
        (
            (f"A6:c3:{SHAPE}@2", "self", "any"),
            (Effect("recolor", color=5),),
            ("click", "self", "any", "recolor"),
        ),
        (("A6:none", f"c2:{SHAPE}", "any"), (Effect("none"),), ("click", SHAPE, "any", "none")),
        (("A6", "self", "any"), (), ("click", "self", "any", "none")),
        (
            ("any", f"kc1:{SHAPE}", "touch:9"),
            (Effect("game_over"),),
            ("any", SHAPE, "touch", "game_over"),
        ),
        (("A2", f"c4:{SHAPE}", "near:3"), MOVE, ("A2", SHAPE, "near", "move")),
        (
            ("event", "level_up", "any"),
            (Effect("level_up"),),
            ("event", "level_up", "any", "level_up"),
        ),
        (
            ("A3", "spawn", "any"),
            (Effect("appear", sig=f"c2:{OTHER}", at=(1, 1)),),
            ("A3", "spawn", "any", "appear"),
        ),
        (("A1", f"c7:{SHAPE}", "selected:1,0"), (), ("A1", SHAPE, "selected:1,0", "none")),
        (("A1", "c3:*", "blocked"), (Effect("none"),), ("A1", "*", "blocked", "none")),
        (("A5", "level", "any"), (Effect("reset_level"),), ("A5", "level", "any", "reset_level")),
        (
            ("A4", f"c12:{SHAPE}", "touch:12"),
            (Effect("move", 0, 1), Effect("reshape", sig=f"c3:{OTHER}")),
            ("A4", SHAPE, "touch", "move"),
        ),
    ],
)
def test_canonical_key_strips_colours(
    fact_key: FactKey, effects: tuple[Effect, ...], expected: CanonicalKey
) -> None:
    key = canonical_key(fact_key, effects)
    assert key == expected
    assert not any(COLOUR.search(part) for part in key)
    assert not any(re.search(r"(touch|near):\d", part) for part in key)


def test_key_parts() -> None:
    assert role_of("A6:c3:ab@1") == "click"
    assert role_of("A6:none") == "click"
    assert role_of("A6") == "click"
    assert role_of("A3") == "A3"
    assert role_of("event") == "event"
    assert role_of("any") == "any"
    assert shape_of(f"c3:{SHAPE}") == SHAPE
    assert shape_of(f"kc3:{SHAPE}") == SHAPE
    assert shape_of("c12:*") == "*"
    assert shape_of("self") == "self"
    assert shape_of("any") == "any"
    assert shape_of("spawn") == "spawn"
    assert shape_of("level") == "level"
    assert tag_of("touch:3") == "touch"
    assert tag_of("near:12") == "near"
    assert tag_of("any") == "any"
    assert tag_of("blocked") == "blocked"
    assert tag_of("between:2,0") == "between:2,0"


@pytest.mark.parametrize(
    ("fact_key", "effects", "expected"),
    [
        (("A3", f"c1:{SHAPE}", "any"), MOVE, f"A3 moves a {SHAPE} thing by (1, 0)"),
        (
            (f"A6:c3:{SHAPE}@2", "self", "any"),
            (Effect("recolor", color=5),),
            "clicking a thing recolours it",
        ),
        (
            (f"A6:c3:{SHAPE}@2", f"c4:{OTHER}", "any"),
            (Effect("recolor", color=5),),
            f"clicking recolours a {OTHER} thing",
        ),
        (
            ("any", f"c1:{SHAPE}", "touch:9"),
            (Effect("game_over"),),
            f"any action ends the game for a {SHAPE} thing while touching something",
        ),
        (
            ("A3", "spawn", "any"),
            (Effect("appear", sig=f"c2:{OTHER}", at=(1, 1)),),
            f"A3 makes a {OTHER} thing appear",
        ),
        (("event", "level_up", "any"), (Effect("level_up"),), "an event completes the level"),
        (("A5", "level", "any"), (Effect("reset_level"),), "A5 puts the level back as it started"),
        (
            ("A2", f"c4:{SHAPE}", "blocked"),
            (Effect("none"),),
            f"A2 does nothing to a {SHAPE} thing when blocked",
        ),
        (
            ("A1", f"c7:{SHAPE}", "any"),
            (Effect("slide", 0, -1),),
            f"A1 slides a {SHAPE} thing by (0, -1) until blocked",
        ),
        (
            ("A1", f"c7:{SHAPE}", "any"),
            (Effect("reshape", sig=f"c7:{OTHER}"),),
            f"A1 reshapes a {SHAPE} thing to {OTHER}",
        ),
        (
            ("A1", f"c7:{SHAPE}", "any"),
            (Effect("toggle_group", sig=f"c7:{OTHER}", color=2),),
            f"A1 recolours every {OTHER} thing",
        ),
        (
            ("A1", f"c7:{SHAPE}", "selected:1,0"),
            (Effect("disappear"),),
            f"A1 makes a {SHAPE} thing disappear while selected",
        ),
        (("A1", f"c7:{SHAPE}", "any"), (), f"A1 has no recorded effect on a {SHAPE} thing"),
    ],
)
def test_statement_for_is_plain_and_colour_free(
    fact_key: FactKey, effects: tuple[Effect, ...], expected: str
) -> None:
    text = statement_for(fact_key, effects)
    assert text == expected
    assert not COLOUR.search(text)
    assert "color" not in text
    assert len(text) <= STATEMENT_MAX_CHARS


def test_statement_for_is_capped() -> None:
    text = statement_for(("A1", "c1:" + "x" * 300, "any"), MOVE)
    assert len(text) == STATEMENT_MAX_CHARS
    assert text.endswith("...")


def test_effect_text_is_colour_free() -> None:
    effects = (
        Effect("move", 1, 0),
        Effect("reshape", sig=f"c3:{OTHER}"),
        Effect("recolor", color=4),
    )
    assert effect_text(effects) == f"move (1, 0); reshape {OTHER}; recolor"
    assert effect_text(()) == "none"
    assert effect_text((Effect("appear", sig=f"c9:{OTHER}", at=(2, 3)),)) == f"appear {OTHER}"
    assert effect_text((Effect("appear"),)) == "appear"
    assert not COLOUR.search(effect_text(effects))


def test_rule_id_is_stable_and_distinct() -> None:
    a = rule_id(("A3", SHAPE, "any", "move"))
    assert a == rule_id(("A3", SHAPE, "any", "move"))
    assert a != rule_id(("A3", SHAPE, "any", "none"))
    assert a != rule_id(("click", SHAPE, "any", "move"))
    assert a.startswith("r") and len(a) == 13


# --- lookups ------------------------------------------------------------------------


def test_rules_for_filters_on_key_fields_and_ranks(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.promote(rule(("A3", SHAPE, "any", "move"), confirmations=2, id="b"))  # 0.75
    store.promote(rule(("A3", OTHER, "any", "move"), confirmations=2, id="a"))  # 0.75
    store.promote(rule(("click", SHAPE, "any", "recolor"), confirmations=6, id="c"))  # 0.875
    store.promote(rule(("A3", SHAPE, "touch", "game_over"), contradictions=3, id="d"))  # 0.2

    assert [r.id for r in store.rules_for()] == ["c", "a", "b", "d"]
    assert [r.id for r in store.rules_for(action_role="A3")] == ["a", "b", "d"]
    assert [r.id for r in store.rules_for(shape_class=SHAPE)] == ["c", "b", "d"]
    assert [r.id for r in store.rules_for(context_tag="touch")] == ["d"]
    assert [r.id for r in store.rules_for(effect_kind="move")] == ["a", "b"]
    assert [r.id for r in store.rules_for(action_role="A3", shape_class=OTHER)] == ["a"]
    assert [r.id for r in store.rules_for(limit=2)] == ["c", "a"]
    assert store.rules_for(action_role="A9") == []


def test_lessons_format_and_order(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.promote(
        rule(
            ("A3", SHAPE, "any", "move"), confirmations=8, games_seen=["g1", "g2"], statement="two"
        )
    )
    store.promote(
        rule(
            ("A2", SHAPE, "any", "move"),
            confirmations=8,
            games_seen=["g1", "g2", "g3"],
            statement="three",
        )
    )
    store.promote(rule(("A1", SHAPE, "any", "none"), games_seen=["g1"], statement="one"))
    assert store.lessons() == [
        "three (p=0.9000, 3 games)",
        "two (p=0.9000, 2 games)",
        "one (p=0.5000, 1 game)",
    ]
    assert store.lessons(limit=2) == ["three (p=0.9000, 3 games)", "two (p=0.9000, 2 games)"]
    assert LongTermStore(tmp_path / "empty.json").lessons() == []


# --- completions and the horizon ---------------------------------------------------


def test_typical_moves_and_typical_for_level(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    assert store.typical_moves("g1", 0) is None
    assert store.typical_for_level(0) is None
    store.record_completion("g1", 0, 10)
    store.record_completion("g1", 0, 20)
    store.record_completion("g2", 0, 30)
    store.record_completion("g2", 1, 7)
    assert store.typical_moves("g1", 0) == 15.0
    assert store.typical_moves("g2", 0) == 30.0
    assert store.typical_moves("g1", 1) is None
    assert store.typical_for_level(0) == 20.0
    assert store.typical_for_level(1) == 7.0
    assert store.typical_for_level(5) is None


def test_record_completion_rejects_negative_values(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match="moves"):
        store.record_completion("g1", 0, -1)
    with pytest.raises(ValueError, match="level"):
        store.record_completion("g1", -1, 3)
    assert store.completions == {}


def test_completions_per_level_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(longterm, "MAX_COMPLETIONS_PER_LEVEL", 3)
    store = store_at(tmp_path)
    for moves in (1, 2, 3, 4, 5):
        store.record_completion("g1", 2, moves)
    assert store.completions["g1"][2] == [3, 4, 5]
    assert store.typical_moves("g1", 2) == 4.0


# --- sequences ------------------------------------------------------------------------


def test_sequences_merge_by_id_and_rank_by_successes(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    store.add_sequence(sequence("s1", "reach a>b", 2, 1))
    store.add_sequence(sequence("s1", "reach a>b", 3, 3))
    store.add_sequence(sequence("s2", "reach a>b", 1, 1))
    store.add_sequence(sequence("s3", "reach a>b", 9, 9))
    store.add_sequence(sequence("s4", "clear c3", 1, 1))
    merged = store.sequences["s1"]
    assert (merged.runs, merged.successes) == (5, 4)
    assert merged.moves == [[1, -1, -1], [6, 3, 4]]
    assert [s.id for s in store.sequences_for("reach a>b")] == ["s3", "s1", "s2"]
    assert [s.id for s in store.sequences_for("reach a>b", limit=1)] == ["s3"]
    assert [s.id for s in store.sequences_for("clear c3")] == ["s4"]
    assert store.sequences_for("count c1=2") == []


def test_add_sequence_validates_counts_and_moves(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match="successes"):
        store.add_sequence(sequence("s1", "g", 1, 2))
    with pytest.raises(ValueError, match="moves"):
        store.add_sequence(replace(sequence("s1", "g", 1, 1), moves=[[1, 2]]))
    assert store.sequences == {}


def test_sequences_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(longterm, "MAX_SEQUENCES", 2)
    store = store_at(tmp_path)
    store.add_sequence(sequence("s1", "g", 4, 2))
    store.add_sequence(sequence("s2", "g", 4, 1))
    store.add_sequence(sequence("s3", "g", 4, 4))
    assert set(store.sequences) == {"s1", "s3"}


# --- threads ------------------------------------------------------------------------------


def test_two_threads_promoting_leave_a_loadable_store(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    errors: list[BaseException] = []

    def work(prefix: str) -> None:
        try:
            for i in range(200):
                store.promote(rule(("A1", f"{prefix}{i}", "any", "move"), confirmations=i % 7))
                if i % 40 == 39:
                    store.save()
        except BaseException as exc:  # the test reports every failure, whatever its type
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(p,)) for p in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert not any(t.is_alive() for t in threads)
    assert len(store.rules) == 400
    store.save()
    again = store_at(tmp_path)
    again.load()
    assert again.load_error is None
    assert len(again.rules) == 400
    assert again.rules == store.rules


def test_lock_is_reentrant_for_a_whole_load_modify_save(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with store.lock:
        store.load()
        store.promote(rule(("A3", SHAPE, "any", "move")))
        store.save()
    again = store_at(tmp_path)
    again.load()
    assert len(again.rules) == 1
