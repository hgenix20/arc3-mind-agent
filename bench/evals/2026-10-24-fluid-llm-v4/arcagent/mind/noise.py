"""Noise mask: object tracks that change for reasons the player does not control.

Noise is judged per object track, never per cell. A track is noise when either
rule fires:

* its (non-move) change descriptor was identical under two different action ids, on
  transitions where nothing moved, or
* it had a non-move change on each of the last ``WINDOW`` transitions, and those
  transitions span two or more action ids.

Moves never count toward either rule: masking a player that two actions move the same
way would blind the agent. The first rule only counts transitions where no track moved:
when something moved, a change elsewhere may be caused by the mover's contact (stepping
on a switch changes a display the same way whichever action made the step), and masking
it would hide the very objects the agent has to reach. Timers that change every step
are still caught by the second rule.

A reshape with a nonzero shift and an unchanged cell count is motion (a walking sprite
that animates while it moves) and is treated as a move. A reshape that changes the cell
count (``Change.size`` against the thing in ``before``) is never motion, whatever its
shift, so a bar that shrinks one cell per move counts toward both rules. A ``size`` of -1
(unknown) counts as unchanged. The descriptor is ``(kind, dx, dy, color, sig)``, where
``sig`` is the new signature, so a sprite that turns left under one action and right
under another has two different descriptors and is not noise. A timer that changes
every step through new states is caught by the second rule, which needs the last
``WINDOW`` descriptors to be all different (a sprite flipping between two poses is not
noise; a two-state blink that alternates exactly with the actions is missed).

The mask cells are the bboxes of noise tracks' things in ``before``, so the mask only
grows within a level. ``reset_tracks`` drops history, flagged ids and cells.
"""

from __future__ import annotations

from arcagent.mind.objects import Cell, Change, Scene, SceneDiff, Thing

WINDOW = 4

Descriptor = tuple[str, int, int, int, str]


def _kind(c: Change, old: Thing | None) -> str:
    """A same-size reshape with a shift is motion (an animated walking sprite): a move."""
    if c.kind != "reshaped" or not (c.dx or c.dy):
        return c.kind
    resized = old is not None and c.size >= 0 and c.size != len(old.cells)
    return "reshaped" if resized else "moved"


def _descriptor(c: Change) -> Descriptor:
    return (c.kind, c.dx, c.dy, c.color, c.sig)


class NoiseMask:
    """Accumulates per-track evidence and the union of noisy bboxes."""

    def __init__(self) -> None:
        self._cells: set[Cell] = set()
        self._noisy: set[int] = set()
        self._by_descriptor: dict[int, dict[Descriptor, set[int]]] = {}
        self._recent: dict[int, list[tuple[int, str, Descriptor]]] = {}

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
        kinds = {c.id: _kind(c, old.get(c.id)) for c in diff.changes}
        still = "moved" not in kinds.values()  # nothing moved: no contact could cause it
        for c in diff.changes:
            kind = kinds[c.id]
            if still and kind not in ("same", "moved"):
                self._by_descriptor.setdefault(c.id, {}).setdefault(_descriptor(c), set()).add(
                    action_id
                )
            recent = self._recent.setdefault(c.id, [])
            recent.append((action_id, kind, _descriptor(c)))
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
            and all(k not in ("moved", "same") for _, k, _ in recent)
            and len({d for _, _, d in recent}) == WINDOW
            and len({a for a, _, _ in recent}) >= 2
        )

    def reset_tracks(self) -> None:
        """Level change: forget history, flagged ids and cells.

        A new level may put real objects where the old animation was; a timer is
        flagged again within a few steps.
        """
        self._cells.clear()
        self._noisy.clear()
        self._by_descriptor.clear()
        self._recent.clear()
