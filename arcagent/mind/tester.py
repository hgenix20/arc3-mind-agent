"""Targeted tests and wisdom vetoes.

``choose_test`` picks the single untried action whose prediction leaves the most
``(label, subject)`` pairs unknown (a simple action answers every subject at once, a
click answers its clicked sig). Ties break by: an action touching a broken fact, then a
thin one, then the earliest ``frontier`` pair it answers, then ``toward`` (the first step
of the best partial plan), then the lowest ``Action.key()``. An action whose
``(scene.key(), action.key())`` is in ``tried`` is never chosen, and neither is one
predicted to end the game. ``None`` means nothing is open.

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

from collections.abc import Sequence
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
    """A chosen action, the question it asks, and how many open pairs it answers."""

    action: Action
    question: str
    settles: int


def _short(subject: str) -> str:
    color, sep, shape = subject.partition(":")
    return subject if not sep or len(shape) <= 4 else f"{color}:{shape[:4]}.."


def _question(label: str, subject: str) -> str:
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
) -> Test | None:
    """The one action that answers the most open questions, or None."""
    order = {pair: i for i, pair in reversed(list(enumerate(frontier)))}
    skey = scene.key()
    best: tuple[tuple[int, int, int, int, tuple[int, int, int]], Test] | None = None
    for action in actions:
        if (skey, action.key()) in tried:
            continue
        pred = predict(scene, action, mem)
        pairs = open_pairs(scene, action, pred)
        if not pairs or pred.game_over:
            continue
        rank = (
            -len(pairs),
            -_fragility(pred, mem),
            min(order.get(p, len(order)) for p in pairs),
            0 if toward is not None and action.key() == toward.key() else 1,
            action.key(),
        )
        if best is None or rank < best[0]:
            label, subject = pairs[0]
            best = (rank, Test(action, _question(label, subject), len(pairs)))
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
