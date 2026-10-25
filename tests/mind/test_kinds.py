"""Tests for kinds: identity classes of sigs that one tracked thing has carried."""

from __future__ import annotations

import numpy as np

from arcagent.core.types import Grid
from arcagent.mind.kinds import KIN_EVIDENCE, KindBook
from arcagent.mind.objects import Scene, Tracker

Cells = dict[tuple[int, int], int]

FACE_R = {(10, 10): 9, (11, 10): 9, (10, 11): 9}  # an L facing one way
FACE_L = {(10, 10): 9, (11, 10): 9, (11, 11): 9}  # the same L turned


def grid_with(cells: Cells) -> Grid:
    g = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g


def run(frames: list[Cells], book: KindBook) -> list[Scene]:
    """Feed frames through a tracker and the book; return the scenes."""
    t = Tracker()
    scenes: list[Scene] = []
    for cells in frames:
        before = scenes[-1] if scenes else None
        scene, diff = t.update(grid_with(cells), frozenset())
        if before is not None and diff is not None:
            book.observe(before, scene, diff, t.how)
        scenes.append(scene)
    return scenes


def sig_of(scene: Scene) -> str:
    return scene.things[0].sig


def test_turning_sprite_becomes_one_kind() -> None:
    book = KindBook()
    scenes = run([FACE_R, FACE_L, FACE_R], book)
    a, b = sig_of(scenes[0]), sig_of(scenes[1])
    assert KIN_EVIDENCE == 2
    assert book.kind_of(a) == book.kind_of(b) == a  # named by the first sig seen


def test_one_observation_is_not_enough() -> None:
    book = KindBook()
    scenes = run([FACE_R, FACE_L], book)
    assert book.kind_of(sig_of(scenes[1])) == sig_of(scenes[1])


def test_new_thing_on_vacated_spot_is_not_kin() -> None:
    book = KindBook()
    other = {(10, 10): 6, (10, 11): 6}
    scenes = run([{(10, 10): 2}, other, {(10, 10): 2}, other], book)
    assert book.kind_of(sig_of(scenes[1])) == sig_of(scenes[1])


def test_a_growing_strip_is_not_kin() -> None:
    book = KindBook()
    strips = [{(x, 20): 4 for x in range(10, 10 + n)} for n in (3, 6, 3, 6)]
    scenes = run(strips, book)
    assert book.kind_of(sig_of(scenes[1])) == sig_of(scenes[1])


def test_recolor_in_place_twice_is_kin() -> None:
    book = KindBook()
    on, off = {(30, 30): 3, (31, 30): 3}, {(30, 30): 8, (31, 30): 8}
    scenes = run([on, off, on], book)
    assert book.kind_of(sig_of(scenes[1])) == sig_of(scenes[0])


def test_kinds_chain_and_label_fills_things() -> None:
    book = KindBook()
    third = {(10, 10): 9, (10, 11): 9, (11, 11): 9}
    scenes = run([FACE_R, FACE_L, FACE_R, FACE_L, third, FACE_L, third], book)
    root = sig_of(scenes[0])
    assert book.kind_of(sig_of(scenes[4])) == root
    labeled = book.label(scenes[4])
    assert labeled.things[0].kind == root and labeled.things[0].sig == sig_of(scenes[4])


def test_unlabeled_things_are_their_own_kind() -> None:
    scene = run([{(5, 5): 2}], KindBook())[0]
    assert scene.things[0].kind == scene.things[0].sig
