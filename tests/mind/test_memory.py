"""Tests for working memory timers and the per-game store."""

from __future__ import annotations

import numpy as np
import pytest

from arcagent.core.types import Action, Grid, Phase
from arcagent.mind.memory import DEFAULT_TTL, GameStore, WorkingMemory, all_steps
from arcagent.mind.types import Step


def _grid() -> Grid:
    return np.zeros((64, 64), dtype=np.int8)


def _step(index: int) -> Step:
    return Step(
        index=index,
        before=_grid(),
        action=Action(1),
        after=_grid(),
        levels_before=0,
        levels_after=0,
        phase_after=Phase.NOT_FINISHED,
    )


def test_default_ttl_is_200() -> None:
    assert DEFAULT_TTL == 200


def test_item_expires_to_cold_after_ttl_ticks() -> None:
    wm, store = WorkingMemory(), GameStore()
    wm.put("k", "v", 3)
    assert wm.tick(store) == []
    assert wm.tick(store) == []
    assert wm.get("k") == "v"  # still readable after ttl-1 ticks
    assert wm.tick(store) == ["k"]  # expires on the 3rd tick
    assert wm.get("k") is None
    assert store.cold == {"k": ("v", 3)}


def test_recall_restores_with_double_ttl() -> None:
    wm, store = WorkingMemory(), GameStore()
    wm.put("k", "v", 2)
    wm.tick(store)
    wm.tick(store)
    assert wm.recall("k", store) == "v"
    assert "k" not in store.cold
    for _ in range(3):
        assert wm.tick(store) == []
    assert wm.tick(store) == ["k"]  # lived 4 ticks = 2 * 2
    assert store.cold["k"] == ("v", 4)


def test_recall_live_and_missing() -> None:
    wm, store = WorkingMemory(), GameStore()
    wm.put("k", "v", 2)
    wm.tick(store)
    assert wm.recall("k", store) == "v"
    assert wm.tick(store) == ["k"]  # live recall did not change the ttl
    assert wm.recall("nope", store) is None


def test_put_replaces_and_validates_ttl() -> None:
    wm, store = WorkingMemory(), GameStore()
    wm.put("k", "a", 1)
    wm.put("k", "b", 5)
    assert wm.tick(store) == []
    assert wm.get("k") == "b"
    with pytest.raises(ValueError):
        wm.put("k", "c", 0)


def test_clear_level_returns_steps_and_empties() -> None:
    wm, store = WorkingMemory(), GameStore()
    wm.steps.extend([_step(0), _step(1)])
    wm.put("k", "v", 5)
    out = wm.clear_level()
    assert [s.index for s in out] == [0, 1]
    assert wm.steps == []
    assert wm.get("k") is None
    assert store.cold == {}


def test_all_steps_orders_episodes_then_current() -> None:
    wm, store = WorkingMemory(), GameStore()
    store.episodes.append([_step(0), _step(1)])
    store.episodes.append([_step(2)])
    wm.steps.append(_step(3))
    assert [s.index for s in all_steps(store, wm)] == [0, 1, 2, 3]
