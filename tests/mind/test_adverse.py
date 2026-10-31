"""Tests for the adverse book, the inhibition network and the perseverance rule (spec 4.10)."""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from arcagent.mind.adverse import (
    BAND_CERTAIN,
    BAND_LOW,
    L2,
    LEARNING_RATE,
    MAX_KEYS,
    MAX_ROLES,
    MAX_TRIALS,
    PRIOR_A,
    PRIOR_B,
    RULE_TRIALS,
    WIDTH,
    AdverseBook,
    AdverseKey,
    InhibitionNet,
    Perseverance,
    band_of,
    features,
    harm_probability,
)

KEY: AdverseKey = ("A3", "kc9:ab12cd34ef", "touch:9")
OTHER: AdverseKey = ("A4", "kc9:ab12cd34ef", "touch:9")


def feature_vector(label: str = "A3", *, roles: frozenset[str] = frozenset({"harm"})) -> np.ndarray:
    return features(label, "kc9:ab12cd34ef", "ab12cd34ef", "touch:9", roles=roles, health_bar=False)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# --- constants ---------------------------------------------------------------------


def test_constants_match_the_spec_and_the_brief() -> None:
    assert (PRIOR_A, PRIOR_B) == (1, 1)
    assert (BAND_CERTAIN, BAND_LOW) == (0.8, 0.3)
    assert (RULE_TRIALS, MAX_TRIALS) == (5, 3)
    assert (WIDTH, MAX_KEYS) == (256, 5000)
    assert (LEARNING_RATE, L2) == (0.1, 0.001)


# --- bands -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "band"),
    [
        (0.79, "medium"),
        (0.8, "certain"),
        (0.29, "low"),
        (0.3, "medium"),
        (0.0, "low"),
        (1.0, "certain"),
        (0.5, "medium"),
    ],
)
def test_band_of_edges_at_point_eight_and_point_three(p: float, band: str) -> None:
    assert band_of(p) == band


@pytest.mark.parametrize("p", [-0.01, 1.01, math.nan, math.inf])
def test_band_of_rejects_values_that_are_not_probabilities(p: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        band_of(p)


# --- adverse book ------------------------------------------------------------------


def test_unknown_key_is_half_medium_and_not_evidenced() -> None:
    book = AdverseBook()
    assert book.probability(KEY) == 0.5
    assert book.band(KEY) == "medium"
    assert (book.trials(KEY), book.harms(KEY)) == (0, 0)
    assert not book.evidenced(KEY)
    assert book.keys() == []


def test_probability_after_four_harms_in_five_trials_is_point_seven_one_four_three() -> None:
    book = AdverseBook()
    for harmed in (True, True, False, True, True):
        book.record(KEY, harmed)
    assert (book.trials(KEY), book.harms(KEY)) == (5, 4)
    assert book.probability(KEY) == 0.7143  # (4 + 1) / (5 + 2)
    assert book.band(KEY) == "medium"


def test_probability_tracks_the_harm_rate_into_both_outer_bands() -> None:
    safe, deadly = AdverseBook(), AdverseBook()
    for _ in range(3):
        safe.record(KEY, False)
        deadly.record(KEY, True)
    assert (safe.probability(KEY), safe.band(KEY)) == (0.2, "low")  # 1 / 5
    assert (deadly.probability(KEY), deadly.band(KEY)) == (0.8, "certain")  # 4 / 5


def test_book_band_agrees_with_band_of() -> None:
    book = AdverseBook()
    for harmed in (True, False, True):
        book.record(KEY, harmed)
    assert book.band(KEY) == band_of(book.probability(KEY))


def test_evidenced_flips_at_rule_trials() -> None:
    book = AdverseBook()
    for _ in range(RULE_TRIALS - 1):
        book.record(KEY, True)
    assert not book.evidenced(KEY)
    book.record(KEY, False)
    assert book.evidenced(KEY)
    assert book.trials(KEY) == RULE_TRIALS


def test_keys_are_sorted_and_independent() -> None:
    book = AdverseBook()
    book.record(OTHER, True)
    book.record(KEY, False)
    assert book.keys() == [KEY, OTHER]
    assert (book.harms(KEY), book.harms(OTHER)) == (0, 1)


def test_eviction_drops_the_least_tried_key_and_keeps_the_most_tried() -> None:
    book = AdverseBook(max_keys=3)
    a, b, c, d, e = (("A1", f"k{i}", "any") for i in range(5))
    for key, n in ((a, 3), (b, 2), (c, 1)):
        for _ in range(n):
            book.record(key, True)
    book.record(d, True)  # full: c has the fewest trials and goes
    assert book.keys() == [a, b, d]
    assert (book.trials(a), book.trials(b), book.trials(d)) == (3, 2, 1)
    book.record(e, False)  # d has one trial, the fewest, and goes
    assert book.keys() == [a, b, e]
    assert book.trials(c) == 0  # an evicted key reads as unknown again


def test_eviction_breaks_a_trial_tie_by_key_order() -> None:
    book = AdverseBook(max_keys=2)
    low, high, new = ("A1", "k0", "any"), ("A1", "k1", "any"), ("A1", "k2", "any")
    book.record(high, True)
    book.record(low, True)
    book.record(new, True)
    assert book.keys() == [high, new]


def test_recording_a_known_key_at_capacity_evicts_nothing() -> None:
    book = AdverseBook(max_keys=2)
    book.record(KEY, True)
    book.record(OTHER, True)
    book.record(KEY, False)
    assert book.keys() == [KEY, OTHER]
    assert book.trials(KEY) == 2


def test_default_bound_is_max_keys() -> None:
    book = AdverseBook()
    for i in range(MAX_KEYS + 1):
        book.record(("A1", f"k{i}", "any"), True)
    assert len(book.keys()) == MAX_KEYS


def test_book_rejects_a_bound_below_one() -> None:
    with pytest.raises(ValueError, match="max_keys"):
        AdverseBook(max_keys=0)


# --- features ----------------------------------------------------------------------


def test_features_are_deterministic_and_carry_the_bias() -> None:
    x, again = feature_vector(), feature_vector()
    assert x.shape == (WIDTH,)
    assert x.dtype == np.float64
    assert np.array_equal(x, again)
    assert x[WIDTH - 1] == 1.0
    assert set(np.unique(x).tolist()) <= {0.0, 1.0}
    # six tokens plus the bias; collisions can only lower the count
    assert 2 <= int(x.sum()) <= 7


def test_features_differ_when_the_label_differs() -> None:
    assert not np.array_equal(feature_vector("A3"), feature_vector("A4"))


def test_features_use_crc32_so_the_index_is_the_same_in_every_process() -> None:
    x = feature_vector("A3", roles=frozenset({"harm", "blocker"}))
    for token in (
        "label=A3",
        "kind=kc9:ab12cd34ef",
        "shape=ab12cd34ef",
        "ctx=touch:9",
        "role=harm",
        "role=blocker",
        "hb=0",
    ):
        assert x[zlib.crc32(token.encode()) % (WIDTH - 1)] == 1.0, token


def test_features_health_bar_flag_changes_the_vector() -> None:
    with_bar = features("A3", "k", "s", "any", roles=frozenset(), health_bar=True)
    without = features("A3", "k", "s", "any", roles=frozenset(), health_bar=False)
    assert not np.array_equal(with_bar, without)


def test_features_reject_more_roles_than_the_cap() -> None:
    roles = frozenset(f"r{i}" for i in range(MAX_ROLES + 1))
    with pytest.raises(ValueError, match="roles"):
        features("A3", "k", "s", "any", roles=roles, health_bar=False)
    at_cap = frozenset(f"r{i}" for i in range(MAX_ROLES))
    features("A3", "k", "s", "any", roles=at_cap, health_bar=False)


# --- inhibition network ------------------------------------------------------------


def test_net_starts_at_half_with_zero_weights() -> None:
    net = InhibitionNet()
    assert net.predict(feature_vector()) == 0.5
    assert net.n_updates == 0


def test_net_moves_toward_one_after_repeated_harm() -> None:
    net, x = InhibitionNet(), feature_vector()
    seen: list[float] = []
    for _ in range(25):
        net.update(x, harmed=True)
        seen.append(net.predict(x))
    assert seen == sorted(seen)
    assert seen[0] > 0.5
    assert seen[-1] > 0.9
    assert net.n_updates == 25


def test_net_moves_toward_zero_after_repeated_safety() -> None:
    net, x = InhibitionNet(), feature_vector()
    seen: list[float] = []
    for _ in range(25):
        net.update(x, harmed=False)
        seen.append(net.predict(x))
    assert seen == sorted(seen, reverse=True)
    assert seen[0] < 0.5
    assert seen[-1] < 0.1


def test_net_generalizes_through_shared_tokens() -> None:
    # The same kind and context under another label share most features, so harm learned
    # on A3 raises the prediction for A4 a little, and a fresh label elsewhere not at all.
    net = InhibitionNet()
    for _ in range(10):
        net.update(feature_vector("A3"), harmed=True)
    far = features("A9", "kc1:zz", "zz", "any", roles=frozenset(), health_bar=True)
    assert net.predict(feature_vector("A4")) > 0.5
    assert net.predict(feature_vector("A4")) < net.predict(feature_vector("A3"))
    assert net.predict(far) >= 0.5  # it still shares the bias, which rose


def test_net_l2_decays_every_weight_but_the_bias() -> None:
    net = InhibitionNet(width=4, lr=0.1, l2=0.001)
    net.load({"weights": [1.0, 1.0, 1.0, 1.0], "n_updates": 0})
    x = np.array([0.0, 0.0, 0.0, 1.0])  # the bias alone is active
    net.update(x, harmed=False)
    w = net.state()["weights"]
    assert isinstance(w, list)
    assert w[:3] == pytest.approx([1.0 - 0.1 * 0.001] * 3)  # L2 only
    assert w[3] == pytest.approx(1.0 - 0.1 * sigmoid(1.0), abs=1e-9)  # gradient only
    assert net.n_updates == 1


def test_net_prediction_is_rounded_to_four_decimals() -> None:
    net = InhibitionNet(width=2)
    net.load({"weights": [0.0, 0.123456], "n_updates": 3})
    p = net.predict(np.array([0.0, 1.0]))
    assert p == round(sigmoid(0.123456), 4)
    assert net.n_updates == 3


def test_net_is_stable_at_extreme_logits() -> None:
    net = InhibitionNet(width=2)
    net.load({"weights": [0.0, -1000.0], "n_updates": 0})
    assert net.predict(np.array([0.0, 1.0])) == 0.0
    net.load({"weights": [0.0, 1000.0], "n_updates": 0})
    assert net.predict(np.array([0.0, 1.0])) == 1.0


@pytest.mark.parametrize(
    "bad",
    [np.zeros(WIDTH - 1), np.zeros(WIDTH + 1), np.zeros((WIDTH, 1)), np.zeros((1, WIDTH))],
)
def test_net_rejects_a_vector_of_the_wrong_shape(bad: np.ndarray) -> None:
    net = InhibitionNet()
    with pytest.raises(ValueError, match="shape"):
        net.predict(bad)
    with pytest.raises(ValueError, match="shape"):
        net.update(bad, harmed=True)
    assert net.n_updates == 0


def test_net_state_round_trips_through_load() -> None:
    net, x = InhibitionNet(), feature_vector()
    for harmed in (True, True, False, True):
        net.update(x, harmed=harmed)
    state = net.state()
    assert set(state) == {"weights", "n_updates"}
    weights, n = state["weights"], state["n_updates"]
    assert isinstance(weights, list) and len(weights) == WIDTH
    assert all(isinstance(w, float) for w in weights)
    assert n == 4
    copy = InhibitionNet()
    copy.load(state)
    assert copy.predict(x) == net.predict(x)
    assert copy.n_updates == 4
    copy.update(x, harmed=True)
    assert net.n_updates == 4  # load copies; the two nets do not share weights


@pytest.mark.parametrize(
    "state",
    [
        {"weights": [0.0] * (WIDTH - 1), "n_updates": 0},
        {"weights": [0.0] * WIDTH},
        {"n_updates": 0},
        {"weights": [0.0] * (WIDTH - 1) + [math.nan], "n_updates": 0},
        {"weights": [0.0] * WIDTH, "n_updates": -1},
        {"weights": [0.0] * WIDTH, "n_updates": 1.5},
        {"weights": "not a list", "n_updates": 0},
        {"weights": ["x"] * WIDTH, "n_updates": 0},
    ],
)
def test_net_load_rejects_a_bad_state_and_keeps_its_weights(state: dict[str, object]) -> None:
    net, x = InhibitionNet(), feature_vector()
    net.update(x, harmed=True)
    before = net.predict(x)
    with pytest.raises(ValueError):
        net.load(state)
    assert net.predict(x) == before
    assert net.n_updates == 1


@pytest.mark.parametrize(
    ("width", "lr", "l2"),
    [(1, 0.1, 0.001), (0, 0.1, 0.001), (4, 0.0, 0.001), (4, -0.1, 0.001), (4, 0.1, -0.001)],
)
def test_net_rejects_bad_hyperparameters(width: int, lr: float, l2: float) -> None:
    with pytest.raises(ValueError):
        InhibitionNet(width=width, lr=lr, l2=l2)


# --- harm probability --------------------------------------------------------------


def test_harm_probability_uses_the_rule_only_once_it_is_evidenced() -> None:
    book, net, x = AdverseBook(), InhibitionNet(), feature_vector()
    for _ in range(3):
        net.update(x, harmed=True)
    for _ in range(RULE_TRIALS - 1):
        book.record(KEY, False)
    assert harm_probability(book, net, KEY, x) == (net.predict(x), "net")
    book.record(KEY, False)
    assert book.evidenced(KEY)
    assert harm_probability(book, net, KEY, x) == (book.probability(KEY), "rule")
    assert harm_probability(book, net, KEY, x)[0] == 0.1429  # 1 / 7, the rule, not the net


def test_harm_probability_of_an_unknown_key_comes_from_the_net() -> None:
    book, net, x = AdverseBook(), InhibitionNet(), feature_vector()
    assert harm_probability(book, net, KEY, x) == (0.5, "net")


# --- perseverance ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "no_other_way", "ev_gain", "trials_used", "expected"),
    [
        (0.8, True, 100.0, 0, (False, "certain")),  # certain: never, whatever the reasons
        (0.95, False, 0.0, 0, (False, "certain")),
        (0.29, False, -1.0, 3, (True, "low")),  # low: always, with no reason needed
        (0.0, False, 0.0, 0, (True, "low")),
        (0.5, True, 100.0, 3, (False, "trials exhausted")),  # medium: the cap comes first
        (0.5, True, -5.0, 0, (True, "no other way")),  # reason A alone
        (0.5, False, 0.1, 2, (True, "expected value")),  # reason B alone
        (0.5, True, 0.1, 0, (True, "no other way")),  # both reasons: A is named
        (0.5, False, 0.0, 0, (False, "no reason")),
        (0.79, False, -0.1, 0, (False, "no reason")),
        (0.3, False, 0.0, 0, (False, "no reason")),
    ],
)
def test_perseverance_every_branch(
    p: float, no_other_way: bool, ev_gain: float, trials_used: int, expected: tuple[bool, str]
) -> None:
    result = Perseverance.allow(
        p, no_other_way=no_other_way, ev_gain=ev_gain, trials_used=trials_used
    )
    assert result == expected
    assert (
        Perseverance().allow(p, no_other_way=no_other_way, ev_gain=ev_gain, trials_used=trials_used)
        == expected
    )


def test_perseverance_trial_cap_is_a_parameter() -> None:
    allow = Perseverance.allow
    assert allow(0.5, no_other_way=True, ev_gain=0.0, trials_used=1, max_trials=1) == (
        False,
        "trials exhausted",
    )
    assert allow(0.5, no_other_way=True, ev_gain=0.0, trials_used=1, max_trials=2) == (
        True,
        "no other way",
    )
    assert allow(0.5, no_other_way=True, ev_gain=0.0, trials_used=0, max_trials=0) == (
        False,
        "trials exhausted",
    )


def test_perseverance_rejects_negative_counts_and_bad_probabilities() -> None:
    with pytest.raises(ValueError, match="trials_used"):
        Perseverance.allow(0.5, no_other_way=False, ev_gain=0.0, trials_used=-1)
    with pytest.raises(ValueError, match="max_trials"):
        Perseverance.allow(0.5, no_other_way=False, ev_gain=0.0, trials_used=0, max_trials=-1)
    with pytest.raises(ValueError, match="probability"):
        Perseverance.allow(1.5, no_other_way=False, ev_gain=0.0, trials_used=0)
