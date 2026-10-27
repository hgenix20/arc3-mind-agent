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
