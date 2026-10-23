"""DisagreementVoi against a scripted sandbox: weights, buckets, ties, bounds."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from arcagent.core.types import GRID_SIZE, Action, Grid, Observation, Phase
from arcagent.mind.modules.voi import DisagreementVoi
from arcagent.mind.predictive import Belief
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.types import ActionKey, Program


class FakeSandbox:
    """Loads anything; ``predict_many`` answers from a (pid, action id) table."""

    def __init__(self, table: dict[tuple[str, int], str | None]) -> None:
        self.table = table
        self.dropped: dict[str, str] = {}
        self.asked: list[list[ActionKey]] = []

    def load(self, pid: str, code: str) -> str | None:
        return None

    def unload(self, pid: str) -> None:
        return None

    def predict(self, pids: Sequence[str], grid: Grid, action: Action) -> dict[str, str | None]:
        return dict.fromkeys(pids)

    def predict_many(
        self, pids: Sequence[str], grid: Grid, actions: Sequence[Action]
    ) -> dict[tuple[str, ActionKey], str | None]:
        self.asked.append([a.key() for a in actions])
        return {(p, a.key()): self.table.get((p, a.id)) for p in pids for a in actions}


def belief_of(**states: tuple[int, int]) -> Belief:
    """Each value is (hits, trials)."""
    b = Belief(FakeSandbox({}))
    for pid, (hits, trials) in states.items():
        b.add(Program(pid, "", False, "llm"), hits, trials)
    return b


OBS = Observation(np.zeros((GRID_SIZE, GRID_SIZE), np.int8), Phase.NOT_FINISHED, 0, 1, (1, 2, 3))
ACTS = [Action(1), Action(2), Action(3)]


def pick(
    voi: DisagreementVoi, belief: Belief, table: dict[tuple[str, int], str | None]
) -> tuple[Action | None, dict[ActionKey, float]]:
    return voi.pick(
        belief,
        FakeSandbox(table),  # pyright: ignore[reportArgumentType]  # only predict_many is used
        OBS,
        ACTS,
        frozenset(),
    )


def test_picks_action_where_programs_disagree() -> None:
    belief = belief_of(a=(9, 10), b=(9, 10))
    table: dict[tuple[str, int], str | None] = {
        ("a", 1): "x",
        ("b", 1): "x",
        ("a", 2): "x",
        ("b", 2): "y",
        ("a", 3): "x",
        ("b", 3): "x",
    }
    got, scores = pick(DisagreementVoi(), belief, table)
    assert got == Action(2)
    assert math.isclose(scores[Action(2).key()], math.log(2))
    assert scores[Action(1).key()] == 0.0


def test_returns_none_when_all_agree() -> None:
    belief = belief_of(a=(5, 10), b=(9, 10))
    table: dict[tuple[str, int], str | None] = {(p, i): "same" for p in "ab" for i in (1, 2, 3)}
    got, scores = pick(DisagreementVoi(), belief, table)
    assert got is None
    assert scores == {a.key(): 0.0 for a in ACTS}


def test_none_predictions_form_their_own_bucket() -> None:
    belief = belief_of(a=(1, 1), b=(1, 1), c=(1, 1))
    table: dict[tuple[str, int], str | None] = {
        ("a", 1): None,
        ("b", 1): None,
        ("c", 1): "x",  # buckets {None: 2, x: 1}
        ("a", 2): "x",
        ("b", 2): "y",
        ("c", 2): "z",  # three buckets, higher entropy
        ("a", 3): None,
        ("b", 3): None,
        ("c", 3): None,  # one bucket
    }
    got, scores = pick(DisagreementVoi(), belief, table)
    assert got == Action(2)
    assert scores[Action(1).key()] > 0.0
    assert scores[Action(3).key()] == 0.0


def test_respects_max_candidates() -> None:
    belief = belief_of(a=(1, 1), b=(1, 1))
    table: dict[tuple[str, int], str | None] = {("a", 3): "x", ("b", 3): "y"}
    sb = FakeSandbox(table)
    got, scores = DisagreementVoi(max_candidates=2).pick(
        belief,
        sb,  # pyright: ignore[reportArgumentType]  # only predict_many is used
        OBS,
        ACTS,
        frozenset(),
    )
    assert got is None
    assert set(scores) == {Action(1).key(), Action(2).key()}
    assert sb.asked == [[Action(1).key(), Action(2).key()]]


def test_fewer_than_two_programs_means_no_opinion() -> None:
    voi = DisagreementVoi()
    assert pick(voi, belief_of(), {}) == (None, {})
    assert pick(voi, belief_of(a=(1, 1)), {("a", 1): "x"}) == (None, {})


def test_weights_follow_fit_and_fresh_programs_get_the_mean() -> None:
    # a has fit 0.5, b has no trials so it weighs 0.5 too: an even split.
    table: dict[tuple[str, int], str | None] = {("a", 1): "x", ("b", 1): "y"}
    _, scores = pick(DisagreementVoi(), belief_of(a=(1, 2), b=(0, 0)), table)
    assert math.isclose(scores[Action(1).key()], math.log(2))
    # Fits 0.9 and 0.1: skewed, so below ln 2.
    _, skew = pick(DisagreementVoi(), belief_of(a=(9, 10), b=(1, 10)), table)
    h = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert math.isclose(skew[Action(1).key()], h)
    # No program has trials: every weight is 1.0.
    _, fresh = pick(DisagreementVoi(), belief_of(a=(0, 0), b=(0, 0)), table)
    assert math.isclose(fresh[Action(1).key()], math.log(2))


def test_all_zero_fit_still_scores_with_equal_weights() -> None:
    table: dict[tuple[str, int], str | None] = {("a", 1): "x", ("b", 1): "y"}
    got, _ = pick(DisagreementVoi(), belief_of(a=(0, 4), b=(0, 4)), table)
    assert got == Action(1)


def test_ties_go_to_the_earlier_candidate() -> None:
    belief = belief_of(a=(1, 1), b=(1, 1))
    table: dict[tuple[str, int], str | None] = {(p, i): p for p in "ab" for i in (1, 2, 3)}
    got, _ = pick(DisagreementVoi(), belief, table)
    assert got == Action(1)


def test_real_sandbox_scores_two_disagreeing_programs() -> None:
    left = "def step(grid, action, x=-1, y=-1):\n    return [r[:] for r in grid]\n"
    right = (
        "def step(grid, action, x=-1, y=-1):\n"
        "    g = [r[:] for r in grid]\n"
        "    g[0][0] = action\n"
        "    return g\n"
    )
    with ProgramSandbox(timeout_s=2.0) as sb:
        belief = Belief(sb)
        assert belief.add(Program("l", left, False, "llm")) is None
        assert belief.add(Program("r", right, False, "llm")) is None
        got, scores = DisagreementVoi().pick(belief, sb, OBS, ACTS, frozenset())
    assert got == Action(1)
    assert all(math.isclose(v, math.log(2)) for v in scores.values())
