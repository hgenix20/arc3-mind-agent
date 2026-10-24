"""Targeted tests and wisdom vetoes.

``choose_test`` picks the single untried action whose prediction leaves the most
``(label, subject)`` pairs unknown (a simple action answers every subject at once, a
click answers its clicked sig). Ties break by: an action touching a broken fact, then a
thin one, then the earliest ``frontier`` pair it answers, then ``toward`` (the first step
of the best partial plan), then the lowest ``Action.key()``. An action whose
``(scene.key(), action.key())`` is in ``tried`` is never chosen, and neither is one
predicted to end the game. ``None`` means nothing is open.

Countable pairs (slice 3): with ``protected`` given, every action is one question,
keyed ``(action label, "self")``, with ``settles`` 1. A click asks it when it has any
open pair. A simple action asks it only when it has a countable pair: one whose subject
is in ``protected`` or is ``self``, or that was never observed (no fact has that label
and subject, so a fresh side object still gets one look). A broken side-object fact
keeps its pair unknown forever; without this rule every new scene made the same
question look fresh (``tried`` is keyed by scene). Once a label has been tested
(``tested``, the caller's per-level counts by key), only its protected and ``self``
pairs count, for clicks too: a new side sig is no reason to repeat it. A key in
``exhausted`` (the caller's per-level cap) never counts. Ties go first by fewest tests
of the label this level (every action is tried once before any repeats), then by
fragility, frontier, ``toward`` and key, as above. ``protected=None`` keeps the slice 1
rule: every open pair counts, one each.

For a click, the things whose prediction was borrowed from a same-shape sibling
(``Prediction.lent``) count as open pairs too: each click label (clicked sig and ring)
gets its own test. Search doesn't do this; borrowing is fine for planning.

``veto`` returns the reason to refuse an action, or None: ``"predicted game_over"``,
``"known no effect here"`` (no unknown pairs, the scene is unchanged, every used fact
solid; thin facts still need observations) or ``"settled test"`` (already tried and no
unknown pairs remain).

Pure; one ``predict`` per action.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from arcagent.core.types import COORD_ACTION_ID, Action
from arcagent.mind.facts import FactMemory, action_label
from arcagent.mind.objects import Scene
from arcagent.mind.predictor import Prediction, predict

Tried = set[tuple[tuple[tuple[str, int, int], ...], tuple[int, int, int]]]

SETTLED = "settled test"
NO_EFFECT = "known no effect here"
GAME_OVER = "predicted game_over"


@dataclass(frozen=True, slots=True)
class Test:
    """A chosen action, the question it asks, and the countable open pairs it answers."""

    action: Action
    question: str
    settles: int
    pairs: tuple[tuple[str, str], ...] = ()


def _short(subject: str) -> str:
    color, sep, shape = subject.partition(":")
    return subject if not sep or len(shape) <= 4 else f"{color}:{shape[:4]}.."


def question(label: str, subject: str) -> str:
    """The question a test of ``(label, subject)`` asks, e.g. ``what does A3 do?``."""
    if subject in ("any", "self"):
        return f"what does {label} do?"
    return f"what does {label} do to {_short(subject)}?"


def open_pairs(scene: Scene, action: Action, pred: Prediction) -> list[tuple[str, str]]:
    """The prediction's unknown pairs, plus for a click the pairs of borrowed predictions."""
    pairs = list(pred.unknown)
    if action.id != COORD_ACTION_ID or not pred.lent:
        return pairs
    label = action_label(action, scene)
    by_id = scene.by_id()
    point = (action.x, action.y)
    for tid in sorted(pred.lent):
        t = by_id.get(tid)
        if t is None:
            continue
        pair = (label, "self" if point in t.cells else t.sig)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _fragility(pred: Prediction, mem: FactMemory) -> int:
    """2 when the prediction leans on a broken fact, 1 on a thin one, else 0."""
    rank = 0
    for key in pred.used:
        fact = mem.get(key)
        status = "thin" if fact is None else fact.status
        if status == "broken":
            return 2
        if status == "thin":
            rank = 1
    return rank


def choose_test(  # noqa: PLR0917  # the six-argument signature is the Task 7 contract
    scene: Scene,
    actions: list[Action],
    mem: FactMemory,
    frontier: Sequence[tuple[str, str]],
    tried: Tried,
    toward: Action | None,
    *,
    protected: frozenset[str] | None = None,
    exhausted: Collection[tuple[str, str]] = (),
    tested: Mapping[tuple[str, str], int] | None = None,
) -> Test | None:
    """The one action that answers the most countable open questions, or None."""
    order = {pair: i for i, pair in reversed(list(enumerate(frontier)))}
    skey = scene.key()
    seen = {(k[0], k[1]) for k in mem.facts} if protected is not None else set[tuple[str, str]]()

    done = tested if tested is not None else {}

    def counts(pair: tuple[str, str], fresh: bool) -> bool:
        if protected is None:
            return pair not in exhausted
        subject = pair[1]
        return subject in protected or subject == "self" or (fresh and pair not in seen)

    best: tuple[tuple[int, int, int, int, int, tuple[int, int, int]], Test] | None = None
    for action in actions:
        if (skey, action.key()) in tried:
            continue
        pred = predict(scene, action, mem)
        found = open_pairs(scene, action, pred)
        key = (action_label(action, scene), "self") if protected is not None else ("", "")
        times = done.get(key, 0)
        # Untested: a click counts any open pair, a simple action its countable pairs.
        # Tested: only protected or self pairs count, so a new side sig is no repeat.
        if action.id == COORD_ACTION_ID and protected is not None and not times:
            live = found
        else:
            live = [p for p in found if counts(p, fresh=not times)]
        # slice 1: every open pair is its own question; slice 3: the action is one,
        # keyed (action label, "self").
        one = [key] if live and key not in exhausted else []
        pairs = live if protected is None else one
        if not pairs or pred.game_over:
            continue
        rank = (
            -len(pairs),
            times,
            -_fragility(pred, mem),
            min(order.get(p, len(order)) for p in live),
            0 if toward is not None and action.key() == toward.key() else 1,
            action.key(),
        )
        if best is None or rank < best[0]:
            label, subject = live[0]
            best = (rank, Test(action, question(label, subject), len(pairs), tuple(pairs)))
    return None if best is None else best[1]


def veto(scene: Scene, action: Action, mem: FactMemory, tried: Tried) -> str | None:
    """Why to refuse ``action`` here, or None."""
    pred = predict(scene, action, mem)
    if pred.game_over:
        return GAME_OVER
    if open_pairs(scene, action, pred):
        return None
    if (scene.key(), action.key()) in tried:
        return SETTLED
    fresh = pred.scene
    if fresh is not None and fresh.key() == scene.key():
        solid = all((f := mem.get(k)) is not None and f.status == "solid" for k in pred.used)
        if solid:
            return NO_EFFECT
    return None
