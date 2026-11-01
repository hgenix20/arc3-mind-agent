"""Adverse rules, the inhibition network and the perseverance rule (spec 4.10a, 4.10b, 4.10c).

Harm (spec 4.10a) is decided by the caller: with a health bar it is any move after which
the bar is lower, without one it is any move that ends the life, the level or the game.
This module takes that verdict as ``harmed: bool`` and turns it into the probability that
a move harms, in three pieces:

- ``AdverseBook`` keeps one tally per ``AdverseKey`` (action label, subject kind, context)
  and reads it as the Beta mean of spec 4.4, ``(harms + 1) / (trials + 2)`` to four
  decimals, with the same bands as a fact (certain at or above ``BAND_CERTAIN``, low below
  ``BAND_LOW``). The prior and the band edges are imported from ``facts`` so a fact and an
  adverse rule read the same numbers. A key becomes evidence the rollout may trust on its
  own after ``RULE_TRIALS`` trials. The book is bounded to ``MAX_KEYS`` keys; a new key
  beyond that evicts the least-tried key.
- ``InhibitionNet`` is the learned function of spec 4.10b: an online logistic regression
  over ``WIDTH`` weights (a few hundred, as the spec asks), fed by ``features``, which
  hashes the canonical tokens of a situation (label, kind, shape class, context, roles,
  health bar present) with CRC-32 so the same situation gives the same vector in every
  process. One SGD step per observed transition, L2 on every weight but the bias.
- ``harm_probability`` picks the rule when the book has ``RULE_TRIALS`` trials for the
  key and the network otherwise (spec 4.10b item 4).
- ``Perseverance.allow`` is the medium-band rule of spec 4.10c: certain is never allowed,
  low always, medium needs no other way to the goal or an expected-value gain, under a
  per-rule trial cap. It returns the reason so the caller can log it.

Architecture decision (spec 4.10b item 6): a logistic model with hashed binary features
was chosen over a hidden layer because the training set is the run's own transitions, a
few thousand at most, and the prediction must fit inside a move's time gate. The caller
that wires this module logs the choice; nothing here logs or does I/O. Pure, numpy only.

Failure modes: a probability outside [0, 1] (NaN included) raises ``ValueError`` in
``band_of`` and ``Perseverance.allow``; a feature vector whose shape is not ``(width,)``
raises ``ValueError`` in the network; a malformed persisted state raises ``ValueError``
from ``load`` and leaves the network unchanged; more than ``MAX_ROLES`` roles raise
``ValueError`` in ``features``.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from arcagent.mind.facts import BAND_CERTAIN, BAND_LOW, PRIOR_A, PRIOR_B, Band

__all__ = [
    "BAND_CERTAIN",
    "BAND_LOW",
    "L2",
    "LEARNING_RATE",
    "MAX_KEYS",
    "MAX_ROLES",
    "MAX_TRIALS",
    "PRIOR_A",
    "PRIOR_B",
    "RULE_TRIALS",
    "WIDTH",
    "AdverseBook",
    "AdverseKey",
    "Band",
    "Features",
    "InhibitionNet",
    "Perseverance",
    "Reason",
    "Source",
    "band_of",
    "features",
    "harm_probability",
]

AdverseKey = tuple[str, str, str]  # (action label, subject kind, context)
Features = npt.NDArray[np.float64]  # shape (width,), the last entry is the bias
Source = Literal["rule", "net"]  # which estimate harm_probability returned
Reason = Literal[
    "certain", "low", "trials exhausted", "no other way", "expected value", "no reason"
]

WIDTH = 256  # feature vector length; a few hundred weights (spec 4.10b item 6)
RULE_TRIALS = 5  # trials after which the adverse rule's own probability is used (4.10b item 4)
MAX_KEYS = 5000  # adverse keys the book keeps; beyond it the least-tried key is evicted
MAX_TRIALS = 3  # medium-band trials per rule per level instance (spec 4.10c item 7)
MAX_ROLES = 16  # roles one situation may carry; the role vocabulary is far smaller
LEARNING_RATE = 0.1  # SGD step of the inhibition network
L2 = 0.001  # weight decay of the inhibition network, never applied to the bias


def band_of(p: float) -> Band:
    """The band a harm probability falls in (spec 4.4): ``certain`` at or above
    ``BAND_CERTAIN``, ``low`` below ``BAND_LOW``, else ``medium``.

    Shared by ``AdverseBook.band`` and ``Perseverance.allow`` so the two cannot drift.
    Raises ``ValueError`` when ``p`` is not in [0, 1]; NaN fails that test too.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p!r}")
    if p >= BAND_CERTAIN:
        return "certain"
    return "low" if p < BAND_LOW else "medium"


@dataclass(slots=True)
class _Tally:
    trials: int = 0
    harms: int = 0


class AdverseBook:
    """Per-key tallies of trials and harms, read as probabilities (spec 4.10b items 1, 4).

    A key is ``(action label, subject kind, context)``; the kind is a ``Thing.kind`` string
    that the caller passes. ``record`` is the only write. ``probability`` is the Beta mean
    with the flat prior of spec 4.4, so an unknown key reads 0.5 and sits in the medium
    band. ``evidenced`` says whether the key has ``RULE_TRIALS`` trials, the point from
    which ``harm_probability`` prefers the rule to the network.

    Bounded to ``max_keys`` keys (default ``MAX_KEYS``): a new key beyond that evicts the
    key with the fewest trials, ties broken by key order, so the most-tried evidence stays.
    An evicted key reads as unknown again. Raises ``ValueError`` for ``max_keys`` below 1.
    """

    def __init__(self, max_keys: int = MAX_KEYS) -> None:
        if max_keys < 1:
            raise ValueError(f"max_keys must be at least 1, got {max_keys}")
        self._max_keys = max_keys
        self._tallies: dict[AdverseKey, _Tally] = {}

    def record(self, key: AdverseKey, harmed: bool) -> None:
        """Count one trial of ``key``, and one harm when ``harmed``; evict if full."""
        tally = self._tallies.get(key)
        if tally is None:
            if len(self._tallies) >= self._max_keys:
                # The eviction scan is linear in the book, which the bound keeps small.
                victim = min(self._tallies.items(), key=lambda kv: (kv[1].trials, kv[0]))[0]
                del self._tallies[victim]
            tally = self._tallies[key] = _Tally()
        tally.trials += 1
        if harmed:
            tally.harms += 1

    def trials(self, key: AdverseKey) -> int:
        """Trials recorded against ``key``; 0 for an unknown key."""
        tally = self._tallies.get(key)
        return 0 if tally is None else tally.trials

    def harms(self, key: AdverseKey) -> int:
        """Harms recorded against ``key``; 0 for an unknown key."""
        tally = self._tallies.get(key)
        return 0 if tally is None else tally.harms

    def probability(self, key: AdverseKey) -> float:
        """``(harms + PRIOR_A) / (trials + PRIOR_A + PRIOR_B)`` to four decimals; 0.5 unknown."""
        tally = self._tallies.get(key, _Tally())
        return round((tally.harms + PRIOR_A) / (tally.trials + PRIOR_A + PRIOR_B), 4)

    def band(self, key: AdverseKey) -> Band:
        """``band_of`` the key's probability."""
        return band_of(self.probability(key))

    def evidenced(self, key: AdverseKey) -> bool:
        """True once ``key`` has at least ``RULE_TRIALS`` trials (spec 4.10b item 4)."""
        return self.trials(key) >= RULE_TRIALS

    def keys(self) -> list[AdverseKey]:
        """Every known key, sorted."""
        return sorted(self._tallies)


def features(
    label: str,
    kind: str,
    shape_class: str,
    context: str,
    *,
    roles: frozenset[str],
    health_bar: bool,
) -> Features:
    """The canonical feature vector of one situation and action (spec 4.10b item 2).

    Each token (``label=<label>``, ``kind=<kind>``, ``shape=<shape_class>``,
    ``ctx=<context>``, one ``role=<r>`` per role, ``hb=<0|1>``) sets index
    ``zlib.crc32(token) % (WIDTH - 1)`` to 1.0; index ``WIDTH - 1`` is the bias, always
    1.0. Two tokens may share an index (the hashing trick); the entry stays 1.0, so the
    vector is binary and the same from any token order. CRC-32 is used because Python's
    ``hash`` is salted per process and the weights persist across games.

    Raises ``ValueError`` for more than ``MAX_ROLES`` roles, which can only be a caller bug.
    """
    if len(roles) > MAX_ROLES:
        raise ValueError(f"at most {MAX_ROLES} roles, got {len(roles)}")
    tokens = [
        f"label={label}",
        f"kind={kind}",
        f"shape={shape_class}",
        f"ctx={context}",
        *(f"role={role}" for role in sorted(roles)),
        f"hb={int(health_bar)}",
    ]
    x = np.zeros(WIDTH, dtype=np.float64)
    for token in tokens:
        x[zlib.crc32(token.encode()) % (WIDTH - 1)] = 1.0
    x[WIDTH - 1] = 1.0
    return x


def _sigmoid(z: float) -> float:
    # exp(-log(1 + exp(-z))) is 1 / (1 + exp(-z)) without overflow at either extreme.
    return float(np.exp(-np.logaddexp(0.0, -z)))


class InhibitionNet:
    """Online logistic regression from a feature vector to a harm probability (spec 4.10b).

    Weights start at zero, so a fresh network predicts 0.5 everywhere. ``update`` takes one
    SGD step on the logistic loss with learning rate ``lr`` and L2 weight decay ``l2`` on
    every weight but the bias (the last entry). ``predict`` is the sigmoid of the dot
    product, rounded to four decimals. ``n_updates`` counts the steps taken. ``state`` and
    ``load`` move the weights and the count through plain JSON-safe values for the
    long-term store.

    Raises ``ValueError`` for ``width`` below 2 (a bias needs at least one feature), ``lr``
    not above 0, ``l2`` below 0, a vector whose shape is not ``(width,)``, and a state
    that is malformed, the wrong length, non-finite, or has a negative or non-integer
    count. A failed ``load`` leaves the network as it was.
    """

    def __init__(self, width: int = WIDTH, lr: float = LEARNING_RATE, l2: float = L2) -> None:
        if width < 2:
            raise ValueError(f"width must be at least 2, got {width}")
        if not lr > 0.0:
            raise ValueError(f"lr must be above 0, got {lr}")
        if not l2 >= 0.0:
            raise ValueError(f"l2 must be at least 0, got {l2}")
        self.width = width
        self.lr = lr
        self.l2 = l2
        self.n_updates = 0
        self._w: Features = np.zeros(width, dtype=np.float64)

    def _check(self, x: Features) -> None:
        if x.shape != (self.width,):
            raise ValueError(f"feature vector shape must be ({self.width},), got {x.shape}")

    def predict(self, x: Features) -> float:
        """Harm probability for ``x``, four decimals. ``ValueError`` on a wrong shape."""
        self._check(x)
        return round(_sigmoid(float(np.dot(self._w, x))), 4)

    def update(self, x: Features, harmed: bool) -> None:
        """One SGD step toward ``harmed`` on the logistic loss, L2 on all but the bias."""
        self._check(x)
        error = _sigmoid(float(np.dot(self._w, x))) - (1.0 if harmed else 0.0)
        decay = self.l2 * self._w
        decay[-1] = 0.0
        self._w -= self.lr * (error * x + decay)
        self.n_updates += 1

    def state(self) -> dict[str, list[float] | int]:
        """``{"weights": [...], "n_updates": n}``, plain values for persistence."""
        return {"weights": [float(w) for w in self._w], "n_updates": self.n_updates}

    def load(self, state: Mapping[str, object]) -> None:
        """Replace the weights and the count from ``state`` (the shape ``state`` returns).

        Validates before writing, so a ``ValueError`` leaves the network unchanged.
        """
        raw_weights = state.get("weights")
        raw_count = state.get("n_updates")
        if not isinstance(raw_weights, list):
            raise ValueError("state['weights'] must be a list")
        try:
            weights = np.array(raw_weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("state['weights'] must hold numbers") from exc
        if weights.shape != (self.width,):
            raise ValueError(f"state['weights'] must have {self.width} entries")
        if not np.all(np.isfinite(weights)):
            raise ValueError("state['weights'] must be finite")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError("state['n_updates'] must be a non-negative integer")
        self._w = weights
        self.n_updates = raw_count


def harm_probability(
    book: AdverseBook, net: InhibitionNet, key: AdverseKey, x: Features
) -> tuple[float, Source]:
    """The harm probability of a move and where it came from (spec 4.10b item 4).

    The book's probability when ``book.evidenced(key)``, else the network's prediction
    for ``x``. The caller builds ``x`` with ``features`` from the same situation as ``key``.
    """
    if book.evidenced(key):
        return book.probability(key), "rule"
    return net.predict(x), "net"


class Perseverance:
    """The medium-band rule of spec 4.10c, stateless; the caller counts trials.

    ``allow`` reads the band of the harm probability ``p``. Certain is never allowed, low
    always. Medium is allowed with a reason: no other way to the goal within the horizon
    (reason A) or an expected-value gain from the move (reason B); either suffices, and A
    is named when both hold. Before the reasons, the trial cap: once ``trials_used`` has
    reached ``max_trials`` (default ``MAX_TRIALS``) no medium move is allowed this level
    instance. The second element is the reason, for the log.
    """

    @staticmethod
    def allow(
        p: float,
        *,
        no_other_way: bool,
        ev_gain: float,
        trials_used: int,
        max_trials: int = MAX_TRIALS,
    ) -> tuple[bool, Reason]:
        """Whether a move with harm probability ``p`` may be taken, and why.

        Raises ``ValueError`` for a negative ``trials_used`` or ``max_trials``, or a ``p``
        outside [0, 1].
        """
        if trials_used < 0:
            raise ValueError(f"trials_used must be at least 0, got {trials_used}")
        if max_trials < 0:
            raise ValueError(f"max_trials must be at least 0, got {max_trials}")
        band = band_of(p)
        if band == "certain":
            return False, "certain"
        if band == "low":
            return True, "low"
        if trials_used >= max_trials:
            return False, "trials exhausted"
        if no_other_way:
            return True, "no other way"
        if ev_gain > 0.0:
            return True, "expected value"
        return False, "no reason"
