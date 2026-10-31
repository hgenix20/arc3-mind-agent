"""Tests for curiosity flags: the flag book, labels, ranking and the briefing lines (spec 4.11)."""

from __future__ import annotations

import pytest

from arcagent.mind.flags import LABEL_CHARS, OUTCOME_MAX, Flag, FlagBook


def test_new_look_sets_check_with_source_novel_look() -> None:
    flag = FlagBook().note("kc3:ab12", new_look=True, unknown_pairs=0)
    assert flag == Flag("kc3:ab12", "check", "novel_look", 1.0, 0.0, "", ())


def test_source_precedence_is_novel_look_then_unknown_pair_then_changed() -> None:
    book = FlagBook()
    assert book.note("a", new_look=True, unknown_pairs=2, changed=True).source == "novel_look"
    assert book.note("b", new_look=False, unknown_pairs=2, changed=True).source == "unknown_pair"
    assert book.note("c", new_look=False, unknown_pairs=0, changed=True).source == "changed"
    for kind in "abc":
        flag = book.get(kind)
        assert flag is not None and flag.state == "check", kind


def test_value_adds_unknown_pairs_new_look_and_change() -> None:
    book = FlagBook()
    assert book.note("a", new_look=True, unknown_pairs=2, changed=True).value == 3.5
    assert book.note("b", new_look=False, unknown_pairs=2, changed=True).value == 2.5
    assert book.note("c", new_look=False, unknown_pairs=0, changed=True).value == 0.5
    assert book.note("d", new_look=True, unknown_pairs=0).value == 1.0


def test_nothing_to_check_makes_a_new_flag_with_value_zero() -> None:
    book = FlagBook()
    flag = book.note("quiet", new_look=False, unknown_pairs=0)
    assert (flag.state, flag.value) == ("new", 0.0)
    assert book.top(10) == []
    assert book.lines(10) == []


def test_note_refreshes_the_value_and_keeps_label_outcome_and_source() -> None:
    book = FlagBook()
    book.note("a", new_look=True, unknown_pairs=3)
    book.label("a", "door")
    book.checked("a", ["A3 a any"])
    flag = book.note("a", new_look=False, unknown_pairs=1)
    assert flag == Flag("a", "check", "novel_look", 1.0, 0.0, "door", ("A3 a any",))


def test_checked_records_the_outcome_and_clears_the_flag() -> None:
    book = FlagBook()
    book.note("a", new_look=True, unknown_pairs=2)
    book.checked("a", ("A1 a any", "A2 a any"))
    flag = book.get("a")
    assert flag is not None
    assert (flag.state, flag.outcome) == ("checked", ("A1 a any", "A2 a any"))
    assert book.top(5) == []
    assert book.lines(5) == []


def test_outcome_is_bounded() -> None:
    book = FlagBook()
    book.note("a", new_look=True, unknown_pairs=0)
    book.checked("a", [f"k{i}" for i in range(OUTCOME_MAX + 30)])
    flag = book.get("a")
    assert flag is not None
    assert len(flag.outcome) == OUTCOME_MAX
    assert flag.outcome[0] == "k0"


def test_ignore_holds_through_later_notes() -> None:
    book = FlagBook()
    book.note("a", new_look=True, unknown_pairs=2)
    book.ignore("a")
    flag = book.note("a", new_look=False, unknown_pairs=2)
    assert (flag.state, flag.value) == ("ignored", 2.0)
    assert book.top(5) == []


def test_label_is_cut_and_keeps_the_source() -> None:
    book = FlagBook()
    book.note("a", new_look=True, unknown_pairs=0)
    book.label("a", "x" * (LABEL_CHARS + 15))
    flag = book.get("a")
    assert flag is not None
    assert flag.label == "x" * LABEL_CHARS
    assert flag.source == "novel_look"


def test_label_on_an_unknown_kind_makes_a_new_flag_with_source_label() -> None:
    book = FlagBook()
    book.label("z", "  wall ")
    assert book.get("z") == Flag("z", "new", "label", 0.0, 0.0, "wall", ())
    assert book.top(5) == []


def test_top_orders_by_value_per_cost_then_kind() -> None:
    book = FlagBook()
    book.note("a", new_look=False, unknown_pairs=4)  # 4 / 4 = 1.0
    book.set_cost("a", 4.0)
    book.note("b", new_look=False, unknown_pairs=3)  # 3 / 1 = 3.0
    book.set_cost("b", 1.0)
    book.note("c", new_look=False, unknown_pairs=2)  # 2 / max(0, 1) = 2.0
    book.note("d", new_look=False, unknown_pairs=3)  # 3 / 1 = 3.0, ties with b, kind order
    book.set_cost("d", 1.0)
    book.note("e", new_look=False, unknown_pairs=0)  # state new: never in top
    assert [f.kind for f in book.top(10)] == ["b", "d", "c", "a"]
    assert [f.kind for f in book.top(2)] == ["b", "d"]
    assert book.top(0) == []


def test_lines_format() -> None:
    book = FlagBook()
    book.note("kc3:ab12", new_look=True, unknown_pairs=2)
    book.set_cost("kc3:ab12", 3.0)
    book.label("kc3:ab12", "door")
    book.note("kc1:ff00", new_look=False, unknown_pairs=0, changed=True)
    assert book.lines(5) == [
        "check kc3:ab12 'door': value 3, cost 3, novel_look",
        "check kc1:ff00: value 0.5, cost 0, changed",
    ]
    assert book.lines(1) == ["check kc3:ab12 'door': value 3, cost 3, novel_look"]


def test_reset_level_keeps_labels_outcomes_and_sources() -> None:
    book = FlagBook()
    book.note("a", new_look=True, unknown_pairs=3)
    book.set_cost("a", 2.0)
    book.label("a", "door")
    book.checked("a", ["A3 a any"])
    book.note("b", new_look=False, unknown_pairs=1)
    book.reset_level()
    assert book.get("a") == Flag("a", "new", "novel_look", 0.0, 0.0, "door", ("A3 a any",))
    assert book.get("b") == Flag("b", "new", "unknown_pair", 0.0, 0.0, "", ())
    assert book.top(5) == []


def test_eviction_drops_the_lowest_value_when_full() -> None:
    book = FlagBook(max_flags=3)
    book.note("a", new_look=False, unknown_pairs=3)
    book.note("b", new_look=False, unknown_pairs=1)
    book.note("c", new_look=False, unknown_pairs=2)
    book.note("d", new_look=False, unknown_pairs=5)
    assert book.get("b") is None
    assert sorted(f.kind for f in book.top(10)) == ["a", "c", "d"]
    book.label("e", "new label")  # a label-made flag also evicts the lowest (c)
    assert book.get("c") is None
    assert book.get("e") is not None
    assert len(book.top(10)) == 2


def test_note_on_a_known_kind_never_evicts() -> None:
    book = FlagBook(max_flags=2)
    book.note("a", new_look=False, unknown_pairs=1)
    book.note("b", new_look=False, unknown_pairs=2)
    book.note("a", new_look=False, unknown_pairs=9)
    assert sorted(f.kind for f in book.top(10)) == ["a", "b"]


def test_unknown_kind_raises_key_error() -> None:
    book = FlagBook()
    with pytest.raises(KeyError):
        book.set_cost("nope", 1.0)
    with pytest.raises(KeyError):
        book.checked("nope", ())
    with pytest.raises(KeyError):
        book.ignore("nope")


def test_max_flags_below_one_raises() -> None:
    with pytest.raises(ValueError, match="max_flags"):
        FlagBook(max_flags=0)
