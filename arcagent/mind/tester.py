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

A pair whose subject is a reshaped thing is not open when the same tracked thing has
its own usable fact for that label under an earlier sig (``FactMemory.kin``). Without
this, an action that reshapes a thing opens a new question about it every time it's
taken, and the tests never run out (the test flood of eval v4).

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


def question(label: str, subject: str) -> str:
    """The question a test of ``(label, subject)`` asks, e.g. ``what does A3 do?``."""
    if subject in ("any", "self"):
        return f"what does {label} do?"
    return f"what does {label} do to {_short(subject)}?"


def _kin_known(scene: Scene, mem: FactMemory, label: str, subject: str) -> bool:
    """Whether every thing carrying ``subject`` has a usable fact for ``label`` under a sig
    its own track carried before (a reshaped thing, already tested in its old shape)."""
    if subject in ("any", "self"):
        return False
    carriers = [t for t in scene.things if t.sig == subject]
    if not carriers:
        return False
    for t in carriers:
        found = None
        for old in mem.earlier_sigs(t):
            found = mem.lookup(label, old, "any")
            if found is not None and not found[1] and found[0].status != "broken":
                break
            found = None
        if found is None:
            return False
    return True


def open_pairs(
    scene: Scene, action: Action, pred: Prediction, mem: FactMemory | None = None
) -> list[tuple[str, str]]:
    """The prediction's unknown pairs, plus for a click the pairs of borrowed predictions,
    less (given ``mem``) the pairs of reshaped things whose earlier shape has the fact."""
    pairs = _open_pairs(scene, action, pred)
    if mem is None or not mem.kin:
        return pairs
    return [p for p in pairs if not _kin_known(scene, mem, p[0], p[1])]


def _open_pairs(scene: Scene, action: Action, pred: Prediction) -> list[tuple[str, str]]:
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
        pairs = open_pairs(scene, action, pred, mem)
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
            best = (rank, Test(action, question(label, subject), len(pairs)))
    return None if best is None else best[1]


def veto(scene: Scene, action: Action, mem: FactMemory, tried: Tried) -> str | None:
    """Why to refuse ``action`` here, or None."""
    pred = predict(scene, action, mem)
    if pred.game_over:
        return GAME_OVER
    if open_pairs(scene, action, pred, mem):
        return None
    if (scene.key(), action.key()) in tried:
        return SETTLED
    fresh = pred.scene
    if fresh is not None and fresh.key() == scene.key():
        solid = all((f := mem.get(k)) is not None and f.status == "solid" for k in pred.used)
        if solid:
            return NO_EFFECT
    return None
