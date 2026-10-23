from itertools import count

from arcagent.mind.noise import NoiseMask
from arcagent.mind.objects import Change, Scene, SceneDiff, Thing

BAR_CELLS = frozenset({(0, 0), (1, 0), (2, 0)})


def _thing(tid: int, color: int, cells: frozenset[tuple[int, int]]) -> Thing:
    return Thing(
        tid, color, "s", f"c{color}:s", min(x for x, _ in cells), min(y for _, y in cells), cells
    )


def _scene() -> Scene:
    return Scene((_thing(1, 2, frozenset({(5, 5)})), _thing(7, 4, BAR_CELLS)), 0, frozenset())


def _diff(*changes: Change) -> SceneDiff:
    return SceneDiff(changes, 0)


def reshape_diff(tid: int, size_delta: int) -> SceneDiff:
    return _diff(Change(tid, "reshaped", color=4, sig=f"c4:len{3 + size_delta}"))


def move_diff(tid: int, dx: int, dy: int) -> SceneDiff:
    return _diff(Change(tid, "moved", dx, dy, 2, "c2:s"))


def recolor_diff(tid: int, color: int) -> SceneDiff:
    return _diff(Change(tid, "recolored", color=color, sig=f"c{color}:s"))


def test_same_change_under_two_actions_is_noise() -> None:
    m = NoiseMask()
    assert m.observe(_scene(), 1, reshape_diff(7, -1)) == []
    assert m.observe(_scene(), 2, reshape_diff(7, -1)) == [7]
    assert 7 in m.noisy_ids and m.cells >= BAR_CELLS


def test_player_moving_differently_per_action_is_never_noise() -> None:
    m = NoiseMask()
    for a, (dx, dy) in [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))] * 2:
        m.observe(_scene(), a, move_diff(1, dx, dy))
    assert 1 not in m.noisy_ids


def test_animation_every_step_across_actions_is_noise() -> None:
    m = NoiseMask()
    colors = count(3)
    for a in (1, 2, 1, 2):
        m.observe(_scene(), a, recolor_diff(7, next(colors)))
    assert 7 in m.noisy_ids


def test_change_after_only_one_action_is_never_noise() -> None:
    m = NoiseMask()
    for _ in range(6):
        m.observe(_scene(), 5, recolor_diff(7, 3))
    assert m.noisy_ids == frozenset()


def test_reset_tracks_keeps_cells_and_forgets_history() -> None:
    m = NoiseMask()
    m.observe(_scene(), 1, reshape_diff(7, -1))
    m.observe(_scene(), 2, reshape_diff(7, -1))
    m.reset_tracks()
    assert m.noisy_ids == frozenset() and m.cells >= BAR_CELLS
    assert m.observe(_scene(), 3, reshape_diff(7, -1)) == []
