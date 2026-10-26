"""The LLM's model: claims never override observed facts; play-out reaches, stops, rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.contract import FactClaim
from arcagent.mind.facts import Effect, FactMemory
from arcagent.mind.game_model import CLAIM_MOVE, play_out, seed_claims
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Scene, Tracker, shape_of

DOT = shape_of([(0, 0)])
PSIG = f"c1:{DOT}"
TSIG = f"c4:{DOT}"


def scene_of(cells: dict[tuple[int, int], int]) -> Scene:
    g: Grid = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return Tracker().update(g, frozenset())[0]


def claim_moves() -> FactMemory:
    mem = FactMemory()
    seed_claims(
        mem,
        [
            FactClaim("A4", PSIG, (Effect("move", 1, 0),)),
            FactClaim("A4", TSIG, (Effect("none"),)),
        ],
    )
    return mem


def test_seed_skips_observed_key() -> None:
    mem = FactMemory()
    mem.record(("A1", PSIG, "any"), (Effect("move", 0, -1),), 3)
    added = seed_claims(
        mem,
        [
            FactClaim("A1", PSIG, (Effect("move", 0, 1),)),
            FactClaim("A2", PSIG, (Effect("move", 0, 1),)),
        ],
    )
    assert added == [("A2", PSIG, "any")]
    fact = mem.get(("A1", PSIG, "any"))
    assert fact is not None and fact.effects == (Effect("move", 0, -1),)
    claimed = mem.get(("A2", PSIG, "any"))
    assert claimed is not None and claimed.evidence == [CLAIM_MOVE]


def test_play_out_reaches_goal() -> None:
    s = scene_of({(2, 2): 1, (6, 2): 4})
    goal = Goal("reach", a=PSIG, b=TSIG)
    out = play_out(s, [Action(4)] * 6, claim_moves(), [goal], frozenset({PSIG}))
    assert out.reached == goal and out.steps == 3 and out.unknown_at is None


def test_play_out_stops_at_unknown() -> None:
    s = scene_of({(2, 2): 1, (6, 2): 4})
    goal = Goal("reach", a=PSIG, b=TSIG)
    out = play_out(s, [Action(4), Action(1), Action(4)], claim_moves(), [goal], frozenset({PSIG}))
    assert out.reached is None and out.unknown_at == 1 and out.steps == 1


def test_rule_moves_object() -> None:
    s = scene_of({(2, 2): 1, (6, 2): 4})
    goal = Goal("reach", a=PSIG, b=TSIG)

    def rule(objects: list[dict[str, Any]], action: dict[str, int]) -> list[dict[str, Any]]:
        if action["id"] != 1:
            return []
        player = next(o for o in objects if o["color"] == 1)
        return [{"id": player["id"], "x": player["x"] + 3}]

    out = play_out(s, [Action(1)], claim_moves(), [goal], frozenset({PSIG}), rule=rule)
    assert out.reached == goal and out.steps == 1


def test_failing_rule_adds_nothing() -> None:
    s = scene_of({(2, 2): 1, (6, 2): 4})

    def rule(_o: list[dict[str, Any]], _a: dict[str, int]) -> list[dict[str, Any]]:
        raise ValueError("bad")

    out = play_out(s, [Action(1)], claim_moves(), [], frozenset({PSIG}), rule=rule)
    assert out.unknown_at == 0
