"""Drive state: the mind's budget and fatigue reading.

``fatigue`` is the fraction of the time budget already spent, measured against
whichever deadline (game or run) comes first. ``planner_bounds`` shrinks the
planner's node and time limits as fatigue rises, so a tired game still acts.

Contract: pure functions of a frozen ``DriveState``; no clock reads, no side
effects. A span of zero or less (deadline at or before ``started``) counts as
fully fatigued rather than dividing by zero.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_NODES = 2_000
MIN_SECONDS = 1.0
FATIGUE_SHRINK = 0.75


@dataclass(frozen=True)
class DriveState:
    """One reading of the budget; all times are seconds on the same clock."""

    started: float
    game_deadline: float
    run_deadline: float
    now: float
    level_actions: int
    synth_calls_left: int
    queue_depth: int


def fatigue(d: DriveState) -> float:
    """Fraction of the time to the earlier deadline already used, in [0, 1]."""
    span = min(d.game_deadline, d.run_deadline) - d.started
    if span <= 0.0:
        return 1.0
    return min(1.0, max(0.0, (d.now - d.started) / span))


def planner_bounds(
    d: DriveState, base_nodes: int = 50_000, base_seconds: float = 10.0
) -> tuple[int, float]:
    """Node and time limits scaled by ``1 - 0.75 * fatigue``, with floors.

    Nodes truncate with ``int``. Floors are 2_000 nodes and 1.0 s, so they
    apply even when the base is below them.
    """
    scale = 1.0 - FATIGUE_SHRINK * fatigue(d)
    return max(MIN_NODES, int(base_nodes * scale)), max(MIN_SECONDS, base_seconds * scale)
