"""Noise mask: object tracks that change for reasons the player does not control.

Noise is judged per object track, never per cell. A track is noise when either
rule fires:

* its (non-move) change descriptor was identical under two different action ids, or
* it had a non-move change on each of the last ``WINDOW`` transitions, and those
  transitions span two or more action ids.

Moves never count toward either rule: masking a player that two actions move the same
way would blind the agent.

The descriptor is ``(kind, dx, dy, color)``. A reshape descriptor leaves out the new
signature and the size delta: ``Change`` carries only the new sig (a hash), so the
size delta is not derivable from ``before`` and the diff. Two reshapes of one track
with the same shift and color therefore count as the same change.

The mask cells are the bboxes of noise tracks' things in ``before``, so the mask only
grows. ``reset_tracks`` drops track history and flagged ids (ids restart on a level
change) but keeps the cells.
"""

from __future__ import annotations

from arcagent.mind.objects import Cell, Change, Scene, SceneDiff

WINDOW = 4

Descriptor = tuple[str, int, int, int]


def _descriptor(c: Change) -> Descriptor:
    return (c.kind, c.dx, c.dy, c.color)


class NoiseMask:
    """Accumulates per-track evidence and the union of noisy bboxes."""

    def __init__(self) -> None:
        self._cells: set[Cell] = set()
        self._noisy: set[int] = set()
        self._by_descriptor: dict[int, dict[Descriptor, set[int]]] = {}
        self._recent: dict[int, list[tuple[int, str]]] = {}

    @property
    def cells(self) -> frozenset[Cell]:
        return frozenset(self._cells)

    @property
    def noisy_ids(self) -> frozenset[int]:
        return frozenset(self._noisy)

    def observe(self, before: Scene, action_id: int, diff: SceneDiff) -> list[int]:
        """Record one transition; return the ids newly flagged as noise."""
        old = before.by_id()
        flagged: list[int] = []
        for c in diff.changes:
            if c.kind not in ("same", "moved"):
                self._by_descriptor.setdefault(c.id, {}).setdefault(_descriptor(c), set()).add(
                    action_id
                )
            recent = self._recent.setdefault(c.id, [])
            recent.append((action_id, c.kind))
            del recent[:-WINDOW]
            if c.id in self._noisy or not self._is_noise(c.id):
                continue
            self._noisy.add(c.id)
            flagged.append(c.id)
            thing = old.get(c.id)
            if thing is not None:
                xs = [x for x, _ in thing.cells]
                ys = [y for _, y in thing.cells]
                self._cells |= {
                    (x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)
                }
        return flagged

    def _is_noise(self, tid: int) -> bool:
        if any(len(acts) >= 2 for acts in self._by_descriptor.get(tid, {}).values()):
            return True
        recent = self._recent.get(tid, [])
        return (
            len(recent) == WINDOW
            and all(k not in ("moved", "same") for _, k in recent)
            and len({a for a, _ in recent}) >= 2
        )

    def reset_tracks(self) -> None:
        """Level change: forget track history and flagged ids, keep the cells."""
        self._noisy.clear()
        self._by_descriptor.clear()
        self._recent.clear()
