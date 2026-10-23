"""Drive state: fatigue and the planner bounds it scales."""

from __future__ import annotations

from arcagent.mind.drive import DriveState, fatigue, planner_bounds


def _d(now: float, game_deadline: float = 100.0, run_deadline: float = 1000.0) -> DriveState:
    return DriveState(
        started=0.0,
        game_deadline=game_deadline,
        run_deadline=run_deadline,
        now=now,
        level_actions=0,
        synth_calls_left=1,
        queue_depth=0,
    )


def test_fatigue_clamps_and_uses_earlier_deadline() -> None:
    assert fatigue(_d(-5.0)) == 0.0
    assert fatigue(_d(50.0)) == 0.5
    assert fatigue(_d(50.0, game_deadline=1000.0, run_deadline=200.0)) == 0.25
    assert fatigue(_d(500.0)) == 1.0
    assert fatigue(_d(100.0, game_deadline=0.0)) == 1.0


def test_planner_bounds_shrink_to_floor() -> None:
    assert planner_bounds(_d(0.0)) == (50_000, 10.0)
    assert planner_bounds(_d(100.0)) == (12_500, 2.5)
    assert planner_bounds(_d(100.0), base_nodes=100, base_seconds=0.5) == (2_000, 1.0)
