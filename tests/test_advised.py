"""The advised explorer: reply parsing, background questions, credit from the reply."""

from __future__ import annotations

import json
import time

import numpy as np

from arcagent.agent.advised import LLM_WEIGHT, AdvisedExplorer, parse_measures, screen_notes
from arcagent.core.types import GRID_SIZE
from arcagent.mind.vchat import Completion, VMessage
from tests.test_guided import ReachEnv


def test_parse_keeps_valid_measures_on_screen_colors() -> None:
    text = "sure: " + json.dumps(
        {
            "goal": "block to post",
            "measures": [
                {"gap": ["c3", 9]},
                {"count": 4, "want": "zero"},
                {"count": "c5", "want": "more"},
                {"gap": [3, 3]},
                {"gap": [3, 12]},
                {"count": 4, "want": "sideways"},
                "nonsense",
            ],
        }
    )
    goal, credit = parse_measures(text, present={3, 4, 5, 9})
    assert goal == "block to post"
    assert credit == {
        ("gap", 3, 9): LLM_WEIGHT,
        ("count", 4, -1): LLM_WEIGHT,
        ("count", 5, -1): -LLM_WEIGHT,
    }


def test_parse_of_a_broken_reply_gives_nothing() -> None:
    assert parse_measures("no json here", {1}) == ("", {})
    assert parse_measures('{"measures": 3}', {1}) == ("", {})


def test_screen_notes_list_colors_with_boxes() -> None:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    g[2:4, 5:9] = 7
    assert "c7: 8 cells, x 5-8, y 2-3" in screen_notes(g)


def test_a_reply_becomes_credit_without_the_explorer_waiting() -> None:
    asked: list[list[VMessage]] = []

    def chat(messages: list[VMessage], seed: int) -> Completion:
        del seed
        asked.append(messages)
        time.sleep(0.05)
        return Completion(json.dumps({"goal": "x", "measures": [{"gap": [3, 9]}]}), 0, 0, 0.0)

    env, p = ReachEnv(), AdvisedExplorer(0, chat=chat)
    o = env.observe()
    o = env.step(p.choose(o))  # asks at the level start and moves on at once
    assert len(asked) == 1 and p.llm_credit == {}
    deadline = time.monotonic() + 5
    while not p.llm_credit and time.monotonic() < deadline:
        o = env.step(p.choose(o))
        time.sleep(0.01)
    p.close()
    assert p.llm_credit == {("gap", 3, 9): LLM_WEIGHT}
    assert p.credit[("gap", 3, 9)] >= LLM_WEIGHT
    assert p.history[0]["goal"] == "x"


def test_a_failed_call_changes_nothing() -> None:
    def chat(messages: list[VMessage], seed: int) -> Completion:
        raise TimeoutError("slow")

    env, p = ReachEnv(), AdvisedExplorer(0, chat=chat)
    o = env.observe()
    for _ in range(20):
        o = env.step(p.choose(o))
        time.sleep(0.005)
    p.close()
    assert p.llm_credit == {} and "error" in p.history[0]


def test_match_measures_name_same_size_objects_and_score_their_difference() -> None:
    from arcagent.agent.advised import mismatch, objects, parse_matches  # noqa: PLC0415

    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    g[2:5, 2:5] = 3  # a 3x3 key
    g[3, 3] = 5
    g[20:23, 30:33] = 3  # a 3x3 slot, not yet showing the key's centre
    g[40:42, 40:46] = 7  # a 6x2 bar, the wrong size to match
    boxes = objects(g)
    key = next(i for i, b in enumerate(boxes) if b == (2, 2, 4, 4))
    slot = next(i for i, b in enumerate(boxes) if b == (30, 20, 32, 22))
    bar = next(i for i, b in enumerate(boxes) if b == (40, 40, 45, 41))
    text = json.dumps({"measures": [{"match": [f"#{slot}", key]}, {"match": [slot, bar]}]})
    pairs = parse_matches(text, boxes)
    assert pairs == [(boxes[slot], boxes[key])]
    assert mismatch(g, *pairs[0]) == 1
    g[21, 31] = 5
    assert mismatch(g, *pairs[0]) == 0


def test_score_reads_match_pairs_from_the_state_key() -> None:
    p = AdvisedExplorer(0)
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    g[2:5, 2:5] = 3
    g[20:23, 30:33] = 4
    p.matches = [((30, 20, 32, 22), (2, 2, 4, 4))]
    p._shape = g.shape  # pyright: ignore[reportPrivateUsage]
    far = g.tobytes() + bytes([0, 0])
    g[20:23, 30:33] = 3
    near = g.tobytes() + bytes([0, 0])
    p.close()
    assert p.score(near) < p.score(far)
