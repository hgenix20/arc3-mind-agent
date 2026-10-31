"""Tests for the rollout: path probability by the chain rule, harm, death, horizon, goal."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from arcagent.core.types import Action, Grid
from arcagent.mind.facts import Effect, FactKey, FactMemory
from arcagent.mind.goals import Goal
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.rollout import rollout

DOT = shape_of([(0, 0)])
PAIR = shape_of([(0, 0), (1, 0)])
PSIG = f"c1:{DOT}"  # the player: one color-1 cell
TSIG = f"c4:{DOT}"  # the target: one color-4 cell
LSIG = f"c9:{DOT}"  # lava: one color-9 cell
SSIG = f"c7:{PAIR}"  # a side thing nothing has a fact for; its shape has no sibling to borrow from

GOAL = Goal("reach", a=PSIG, b=TSIG)
PROTECT = frozenset({PSIG})
P_KEY: FactKey = ("A4", PSIG, "any")
T_KEY: FactKey = ("A4", TSIG, "any")


def scene_of(cells: dict[tuple[int, int], int]) -> Scene:
    g: Grid = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return Tracker().update(g, frozenset())[0]


def fact(
    mem: FactMemory, key: FactKey, effects: tuple[Effect, ...], *, hits: int, misses: int = 0
) -> FactKey:
    """Record ``key`` once, then set its counts so its probability is exact."""
    mem.record(key, effects, 0)
    stored = mem.facts[key]
    stored.hits = hits
    stored.misses = misses
    return key


def walk_mem(*, p_hits: int = 3, t_hits: int = 1, t_misses: int = 1) -> FactMemory:
    """A4 moves the player one cell right (p 0.8 by default); the target stays (p 0.5)."""
    mem = FactMemory()
    fact(mem, P_KEY, (Effect("move", 1, 0),), hits=p_hits)
    fact(mem, T_KEY, (Effect("none"),), hits=t_hits, misses=t_misses)
    return mem


def player_x(scene: Scene) -> int:
    return next(t.x for t in scene.things if t.sig == PSIG)


# player at (2, 2), target at (6, 2): three A4 steps put the player next to the target
START = scene_of({(2, 2): 1, (6, 2): 4})


# --- probability ----------------------------------------------------------------------


def test_path_probability_is_the_product_of_step_probabilities() -> None:
    mem = walk_mem()
    assert mem.probability(P_KEY) == 0.8 and mem.probability(T_KEY) == 0.5
    ro = rollout(START, [Action(4)] * 6, mem, [GOAL], PROTECT, horizon=10)
    p_step = round(mem.probability(P_KEY) * mem.probability(T_KEY), 4)
    assert ro.step_probabilities == (p_step, p_step, p_step) == (0.4, 0.4, 0.4)
    assert ro.path_probability == round(p_step**3, 4) == 0.064
    assert ro.stop == "goal" and ro.reached == GOAL
    assert len(ro.scenes) == 3 and ro.unknown_at is None and ro.deaths_at is None


def test_a_fact_used_for_two_things_counts_once_per_step() -> None:
    # two targets share one fact: the step's probability has that fact once, not squared
    two_targets = scene_of({(2, 2): 1, (6, 2): 4, (6, 10): 4})
    mem = walk_mem()
    ro = rollout(two_targets, [Action(4)] * 3, mem, [GOAL], PROTECT, horizon=10)
    assert ro.step_probabilities == (0.4, 0.4, 0.4)


def test_a_step_with_no_fact_for_a_protected_thing_stops_unknown() -> None:
    ro = rollout(START, [Action(4), Action(1), Action(4)], walk_mem(), [GOAL], PROTECT, horizon=10)
    assert ro.stop == "unknown" and ro.unknown_at == 1
    assert len(ro.scenes) == 1 and len(ro.step_probabilities) == 1
    assert ro.reached is None and ro.path_probability == 0.4


def test_an_unprotected_thing_with_no_fact_is_assumed_still_and_counted() -> None:
    with_side = scene_of({(2, 2): 1, (6, 2): 4, (10, 10): 7, (11, 10): 7})
    ro = rollout(with_side, [Action(4)] * 3, walk_mem(), [GOAL], PROTECT, horizon=10)
    assert ro.stop == "goal" and ro.assumed == 3
    assert all(any(t.sig == SSIG and (t.x, t.y) == (10, 10) for t in s.things) for s in ro.scenes)


def test_an_empty_plan_is_the_empty_product() -> None:
    ro = rollout(START, [], walk_mem(), [GOAL], PROTECT, horizon=5)
    assert ro.stop == "end" and ro.scenes == () and ro.step_probabilities == ()
    assert ro.path_probability == 1.0 and ro.expected_moves == 0.0 and ro.assumed == 0


# --- death and harm -------------------------------------------------------------------


def lava_mem() -> FactMemory:
    """Stepping onto color 9 is death, learned the way ``facts.learn`` records it."""
    mem = FactMemory()
    fact(mem, P_KEY, (Effect("move", 1, 0),), hits=3)
    fact(mem, ("A4", LSIG, "any"), (Effect("none"),), hits=3)
    fact(mem, ("any", PSIG, "touch:9"), (Effect("game_over"),), hits=3)
    return mem


def test_a_predicted_game_over_is_death_with_health_one() -> None:
    next_to_lava = scene_of({(5, 5): 1, (6, 5): 9})
    ro = rollout(next_to_lava, [Action(4), Action(4)], lava_mem(), [], PROTECT, horizon=10)
    assert ro.stop == "death" and ro.deaths_at == 0
    assert len(ro.scenes) == 1 and len(ro.step_probabilities) == 1  # the fatal scene stays
    assert ro.harm == () and ro.reached is None


def deadly_mem() -> FactMemory:
    """Every A4 is predicted fatal outright, so deaths are deterministic per step."""
    mem = FactMemory()
    fact(mem, P_KEY, (Effect("game_over"),), hits=3)
    fact(mem, T_KEY, (Effect("none"),), hits=3)
    return mem


def test_health_two_survives_one_death_and_stops_at_the_second() -> None:
    plan = [Action(4)] * 3
    one = rollout(START, plan, deadly_mem(), [], PROTECT, horizon=10, health=1)
    assert one.stop == "death" and one.deaths_at == 0 and len(one.scenes) == 1
    two = rollout(START, plan, deadly_mem(), [], PROTECT, horizon=10, health=2)
    assert two.stop == "death" and two.deaths_at == 1 and len(two.scenes) == 2
    three = rollout(START, plan, deadly_mem(), [], PROTECT, horizon=10, health=3)
    assert three.stop == "death" and three.deaths_at == 2 and len(three.scenes) == 3
    four = rollout(START, plan, deadly_mem(), [], PROTECT, horizon=10, health=4)
    assert four.stop == "end" and four.deaths_at is None and len(four.scenes) == 3


def test_adverse_in_the_certain_band_is_death() -> None:
    plan = [Action(4)] * 3
    ro = rollout(START, plan, walk_mem(), [GOAL], PROTECT, horizon=10, adverse=lambda _s, _a: 0.9)
    assert ro.stop == "death" and ro.deaths_at == 0 and ro.harm == ((0, 0.9),)
    ro2 = rollout(
        START, plan, walk_mem(), [GOAL], PROTECT, horizon=10, health=2, adverse=lambda _s, _a: 0.8
    )
    assert ro2.stop == "death" and ro2.deaths_at == 1 and ro2.harm == ((0, 0.8), (1, 0.8))


def test_adverse_below_the_certain_band_is_recorded_not_death() -> None:
    ro = rollout(
        START, [Action(4)] * 3, walk_mem(), [GOAL], PROTECT, horizon=10, adverse=lambda _s, _a: 0.4
    )
    assert ro.stop == "goal" and ro.deaths_at is None
    assert ro.harm == ((0, 0.4), (1, 0.4), (2, 0.4))


def test_adverse_sees_the_scene_before_each_step() -> None:
    seen: list[tuple[int, int]] = []

    def adverse(scene: Scene, action: Action) -> float:
        seen.append((player_x(scene), action.id))
        return 0.0

    ro = rollout(START, [Action(4)] * 3, walk_mem(), [GOAL], PROTECT, horizon=10, adverse=adverse)
    assert ro.harm == () and seen == [(2, 4), (3, 4), (4, 4)]


# --- horizon, goal, expected moves ----------------------------------------------------


def test_the_horizon_cuts_a_long_plan() -> None:
    far = scene_of({(2, 2): 1, (30, 2): 4})
    cut = rollout(far, [Action(4)] * 6, walk_mem(), [GOAL], PROTECT, horizon=2)
    assert cut.stop == "horizon" and len(cut.scenes) == 2 and cut.reached is None
    whole = rollout(far, [Action(4)] * 2, walk_mem(), [GOAL], PROTECT, horizon=2)
    assert whole.stop == "end" and len(whole.scenes) == 2


def test_a_goal_that_holds_stops_the_rollout() -> None:
    ro = rollout(START, [Action(4)] * 6, walk_mem(), [GOAL], PROTECT, horizon=10)
    assert ro.stop == "goal" and ro.reached == GOAL and len(ro.scenes) == 3
    assert player_x(ro.scenes[-1]) == 5


def test_expected_moves_sums_the_expected_tries_per_step() -> None:
    ro = rollout(START, [Action(4)] * 3, walk_mem(), [GOAL], PROTECT, horizon=10)
    assert ro.expected_moves == round(3 * (1 / 0.4), 2) == 7.5


def test_expected_moves_floors_each_step_and_caps_at_three_horizons() -> None:
    doubtful = walk_mem(p_hits=0, t_hits=0, t_misses=20)  # step probability far below 0.05
    doubtful.facts[P_KEY].misses = 20
    ro = rollout(START, [Action(4)] * 3, doubtful, [GOAL], PROTECT, horizon=100)
    assert ro.step_probabilities[0] < 0.05
    assert ro.expected_moves == 60.0  # 3 steps at the floor, 1 / 0.05 each
    capped = rollout(START, [Action(4)] * 3, doubtful, [GOAL], PROTECT, horizon=10)
    assert capped.expected_moves == 30.0


# --- the rule function, as play_out applies it ----------------------------------------


def test_a_rule_stands_in_for_an_unknown_step_at_the_prior() -> None:
    def rule(objects: list[dict[str, Any]], action: dict[str, int]) -> list[dict[str, Any]]:
        if action["id"] != 1:
            return []
        player = next(o for o in objects if o["color"] == 1)
        return [{"id": player["id"], "x": player["x"] + 3}]

    ro = rollout(START, [Action(1)], walk_mem(), [GOAL], PROTECT, horizon=10, rule=rule)
    assert ro.stop == "goal" and ro.reached == GOAL
    assert ro.step_probabilities == (0.5,) and ro.path_probability == 0.5


def test_rule_changes_apply_on_top_of_a_prediction() -> None:
    def rule(objects: list[dict[str, Any]], _action: dict[str, int]) -> list[dict[str, Any]]:
        player = next(o for o in objects if o["color"] == 1)
        return [{"id": player["id"], "x": player["x"] + 3}]

    ro = rollout(START, [Action(4)], walk_mem(), [GOAL], PROTECT, horizon=10, rule=rule)
    assert ro.stop == "goal" and player_x(ro.scenes[0]) == 5
    assert ro.step_probabilities == (0.4,)


def test_a_failing_rule_adds_nothing() -> None:
    def rule(_o: list[dict[str, Any]], _a: dict[str, int]) -> list[dict[str, Any]]:
        raise ValueError("bad")

    ro = rollout(START, [Action(1)], walk_mem(), [GOAL], PROTECT, horizon=10, rule=rule)
    assert ro.stop == "unknown" and ro.unknown_at == 0


# --- boundaries -----------------------------------------------------------------------


def test_rejects_a_negative_horizon_and_no_health() -> None:
    with pytest.raises(ValueError, match="horizon"):
        rollout(START, [Action(4)], walk_mem(), [GOAL], PROTECT, horizon=-1)
    with pytest.raises(ValueError, match="health"):
        rollout(START, [Action(4)], walk_mem(), [GOAL], PROTECT, horizon=1, health=0)
