"""Tests for the hypothesis list: expected value (plan D-C), ids, mapping from a rollout, rank."""

from __future__ import annotations

from arcagent.core.types import Action
from arcagent.mind.goals import Goal
from arcagent.mind.hypotheses import Hypothesis, expected_value, from_rollout, hypothesis_id, rank
from arcagent.mind.rollout import Rollout, RolloutStop

GOAL = Goal("reach", a="c1:aa", b="c4:bb")
PLAN = (Action(4), Action(4), Action(1))


def rollout_of(
    *,
    path: float,
    moves: float,
    harm: tuple[tuple[int, float], ...] = (),
    reached: Goal | None = GOAL,
    stop: RolloutStop = "goal",
) -> Rollout:
    """A rollout record with the numbers under test; scenes are not needed here."""
    return Rollout(
        scenes=(),
        step_probabilities=(path,),
        path_probability=path,
        reached=reached,
        unknown_at=None,
        harm=harm,
        stop=stop,
        expected_moves=moves,
        assumed=0,
        deaths_at=None,
    )


def hyp(hid: str, value: float, path: float) -> Hypothesis:
    return Hypothesis(hid, GOAL.id(), "search", PLAN, path, 10.0, 0.0, value, "goal")


# --- expected value -------------------------------------------------------------------


def test_expected_value_examples_from_the_plan() -> None:
    assert expected_value(1.0, 20.0, 20.0, 0.0, 0.0) == 100.0  # typical reached: eff 1
    assert expected_value(1.0, 40.0, 20.0, 0.0, 0.0) == 25.0  # twice typical: eff 0.25
    assert expected_value(1.0, 10.0, 20.0, 0.0, 0.0) == 100.0  # faster than typical caps at 1
    assert expected_value(0.5, 10.0, 20.0, 0.2, 30.0) == 44.0  # 50 minus 0.2 * 30


def test_expected_value_floors_expected_moves_at_one() -> None:
    assert expected_value(1.0, 0.0, 20.0, 0.0, 0.0) == 100.0


def test_expected_value_rounds_to_four_decimals() -> None:
    assert expected_value(1 / 3, 20.0, 20.0, 0.0, 0.0) == 33.3333
    assert expected_value(0.0, 20.0, 20.0, 1.0, 12.3456789) == -12.3457


# --- ids -----------------------------------------------------------------------------


def test_hypothesis_id_is_a_deterministic_ten_hex_digest() -> None:
    a = hypothesis_id(GOAL.id(), "search", PLAN)
    assert a == hypothesis_id(GOAL.id(), "search", list(PLAN))
    assert len(a) == 10 and int(a, 16) >= 0
    assert a != hypothesis_id(GOAL.id(), "planner", PLAN)
    assert a != hypothesis_id(GOAL.id(), "search", PLAN[:2])
    assert a != hypothesis_id("clear c4", "search", PLAN)
    assert a != hypothesis_id(GOAL.id(), "search", (*PLAN[:2], Action(6, 3, 4)))


# --- from a rollout -------------------------------------------------------------------


def test_from_rollout_takes_the_rollout_numbers_and_the_worst_harm() -> None:
    ro = rollout_of(path=0.5, moves=10.0, harm=((0, 0.2), (2, 0.6), (3, 0.1)))
    h = from_rollout(GOAL.id(), "search", PLAN, ro, typical=20.0, restart_moves=7.0)
    assert h.id == hypothesis_id(GOAL.id(), "search", PLAN)
    assert h.goal_id == GOAL.id() and h.source == "search" and h.plan == PLAN
    assert h.path_probability == 0.5 and h.expected_moves == 10.0 and h.stop == "goal"
    assert h.harm_probability == 0.6
    assert h.expected_value == expected_value(0.5, 10.0, 20.0, 0.6, 7.0) == 45.8


def test_from_rollout_with_no_harm_has_harm_probability_zero() -> None:
    h = from_rollout(
        GOAL.id(),
        "planner",
        list(PLAN),
        rollout_of(path=0.9, moves=3.0),
        typical=20.0,
        restart_moves=5.0,
    )
    assert h.harm_probability == 0.0 and h.plan == PLAN and h.expected_value == 90.0


def test_from_rollout_without_a_reached_goal_earns_no_reward() -> None:
    # spec 4.7b: reward is the value of the goals the path reaches; a cut or stuck path reaches none
    cut = rollout_of(path=0.9, moves=3.0, harm=((1, 0.5),), reached=None, stop="horizon")
    h = from_rollout(GOAL.id(), "planner", PLAN, cut, typical=20.0, restart_moves=8.0)
    assert h.stop == "horizon" and h.path_probability == 0.9
    assert h.expected_value == -4.0  # only the harm cost, 0.5 * 8


# --- rank ------------------------------------------------------------------------------


def test_rank_orders_by_value_then_path_probability_then_id() -> None:
    by_value = hyp("cc", 50.0, 0.1)
    tie_high_p = hyp("bb", 40.0, 0.7)
    tie_low_p = hyp("aa", 40.0, 0.6)
    assert rank([tie_low_p, by_value, tie_high_p]) == [by_value, tie_high_p, tie_low_p]
    # a difference below four decimals is a tie; the id breaks it
    z = hyp("zz", 40.0, 0.70004)
    assert rank([z, tie_high_p]) == [tie_high_p, z]
    assert rank([]) == []
