"""Kinds: classes of sigs that one tracked thing has carried, so identity survives a look.

A sprite that turns, a tile that toggles color or a piece that shows a selection mark gets
a new sig each time its look changes. ``KindBook`` joins two sigs into one kind once it has
seen a tracked thing switch between them ``KIN_EVIDENCE`` times, counting only matches that
say "same object, new look": the predicted look (``prior``), a recolor in place, a ``near``
match (same color and size, close by), or a same-color overlap whose size changed by at
most ``objects.NEAR_SIZE``. A strip that grows, or a different thing appearing where one
vanished (``appeared`` plus ``disappeared``), is never evidence.

A sig never joined is its own kind. A class of two or more sigs is named ``k`` plus the
first of its sigs the book saw (``KIND_PREFIX``), so a kind fact never shares a key with
the facts of that first look; ``base_sig`` strips the prefix for code that reads a color
from a subject. Kinds last for the whole game: the same sigs come back on later levels.
Pure; one union-find over sigs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

from arcagent.mind.objects import NEAR_SIZE, MatchHow, Scene, SceneDiff, Thing

KIN_EVIDENCE = 2  # observations of one switch before two sigs become one kind
KIND_PREFIX = "k"  # marks a kind of two or more sigs: "k" + its first-seen sig
_SAME_OBJECT: frozenset[MatchHow] = frozenset({"prior", "recolor", "near"})


def base_sig(subject: str) -> str:
    """``subject`` without the kind prefix: the first-seen sig of a kind, else unchanged."""
    if subject.startswith(KIND_PREFIX + "c"):
        return subject[len(KIND_PREFIX) :]
    return subject


def _same_object(how: MatchHow | None, b: Thing, a: Thing) -> bool:
    if how in _SAME_OBJECT:
        return True
    small, large = sorted((len(a.cells), len(b.cells)))
    return how == "reshape" and large - small <= NEAR_SIZE * large


class KindBook:
    """Union-find over sigs, joined on repeated evidence (module docstring)."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._order: dict[str, int] = {}
        self._evidence: Counter[tuple[str, str]] = Counter()
        self._size: dict[str, int] = {}  # root sig -> sigs in its class

    def _see(self, sig: str) -> None:
        if sig not in self._order:
            self._order[sig] = len(self._order)
            self._parent[sig] = sig

    def kind_of(self, sig: str) -> str:
        """The kind of ``sig``: ``k`` + its class's first-seen sig; a sig never joined is
        its own kind."""
        root = self._root(sig)
        return KIND_PREFIX + root if self._size.get(root, 1) > 1 else sig

    def _root(self, sig: str) -> str:
        root = sig
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        while sig != root:  # path compression
            sig, self._parent[sig] = self._parent[sig], root
        return root

    def observe(
        self, before: Scene, after: Scene, diff: SceneDiff, how: Mapping[int, MatchHow]
    ) -> list[str]:
        """Count each look switch in ``diff``; return the kinds joined by this call."""
        olds, news = before.by_id(), after.by_id()
        joined: list[str] = []
        for c in diff.changes:
            b, a = olds.get(c.id), news.get(c.id)
            if b is None or a is None or b.sig == a.sig or not _same_object(how.get(c.id), b, a):
                continue
            self._see(b.sig)
            self._see(a.sig)
            pair = (min(b.sig, a.sig), max(b.sig, a.sig))
            self._evidence[pair] += 1
            if self._evidence[pair] >= KIN_EVIDENCE and self._union(b.sig, a.sig):
                joined.append(self.kind_of(b.sig))
        return joined

    def _union(self, x: str, y: str) -> bool:
        rx, ry = self._root(x), self._root(y)
        if rx == ry:
            return False
        first, second = sorted((rx, ry), key=self._order.__getitem__)
        self._parent[second] = first
        self._size[first] = self._size.get(first, 1) + self._size.pop(second, 1)
        return True

    def label(self, scene: Scene) -> Scene:
        """``scene`` with every thing's ``kind`` set."""
        things = tuple(replace(t, kind=self.kind_of(t.sig)) for t in scene.things)
        return Scene(things, scene.background, scene.blocked, scene.terrain)
