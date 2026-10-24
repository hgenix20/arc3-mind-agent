"""fluid_sleep: the winning-facts check, consolidation, drops and the time cap."""

from __future__ import annotations

import json
from pathlib import Path

from arcagent.mind.config import CONFIGS
from arcagent.mind.facts import THIN_BELOW, Effect, FactKey, FactMemory
from arcagent.mind.fluid import FluidAgent
from arcagent.mind.fluid_sleep import SLEEP_MAX_S, fluid_sleep
from arcagent.mind.runlog import RunLog
from tests.mind.fakegames import FakeGame, play

NONE = (Effect("none"),)
LEFT = (Effect("move", dx=-1),)
RIGHT = (Effect("move", dx=1),)


def seeded(key: FactKey, effects: tuple[Effect, ...], hits: int) -> FactMemory:
    mem = FactMemory()
    for i in range(hits):
        mem.record(key, effects, i)
    return mem


def test_used_fact_with_a_miss_this_level_is_demoted() -> None:
    key = ("A3", "c1:x", "any")
    mem = seeded(key, LEFT, 7)
    mem.record(key, RIGHT, 8)  # one miss on this level
    assert mem.facts[key].status == "solid"
    report = fluid_sleep(mem, [key], 0)
    assert report.demoted == (key,)
    assert mem.facts[key].hits == THIN_BELOW - 1
    assert mem.facts[key].status == "thin"
    assert not report.truncated


def test_used_fact_without_misses_stays_solid() -> None:
    key = ("A3", "c1:x", "any")
    mem = seeded(key, LEFT, 7)
    report = fluid_sleep(mem, [key], 0)
    assert report.demoted == ()
    assert mem.facts[key].hits == 7 and mem.facts[key].status == "solid"


def test_miss_from_an_earlier_level_does_not_demote() -> None:
    key = ("A3", "c1:x", "any")
    mem = seeded(key, LEFT, 7)
    mem.record(key, RIGHT, 8)
    mem.new_level()
    mem.record(key, LEFT, 9)
    assert fluid_sleep(mem, [key], 1).demoted == ()


def test_sibling_merge_is_logged_and_changes_nothing() -> None:
    a, b = ("A1", "c3:sq", "any"), ("A1", "c5:sq", "any")
    other = ("A1", "c4:sq", "any")
    mem = seeded(a, NONE, 3)
    for i in range(2):
        mem.record(b, NONE, i)
    for i in range(6):
        mem.record(other, (Effect("recolor", color=9),), i)  # the sibling _borrow would lend
    report = fluid_sleep(mem, [], 0)
    assert report.merged == (b,)
    assert set(mem.facts) == {a, b, other}
    found = mem.lookup("A1", "c5:sq", "any")
    assert found is not None and found[0].key == b and not found[1]  # still its own fact


def test_mover_and_click_facts_never_merge() -> None:
    mem = FactMemory()
    mem.record(("A3", "c1:sq", "any"), LEFT, 0)
    mem.record(("A3", "c2:sq", "any"), LEFT, 0)
    mem.record(("A6:c1:sq@0", "c4:dot", "any"), NONE, 0)
    mem.record(("A6:c1:sq@0", "c5:dot", "any"), NONE, 0)
    before = set(mem.facts)
    assert fluid_sleep(mem, [], 0).merged == ()
    assert set(mem.facts) == before


def test_zero_hit_unused_thin_fact_is_dropped() -> None:
    key, kept = ("A2", "c7:x", "any"), ("A2", "c8:y", "any")
    mem = FactMemory()
    mem.record(key, (Effect("recolor", color=3),), 0)
    mem.record(kept, (Effect("recolor", color=4),), 0)
    mem.facts[key].hits = 0
    report = fluid_sleep(mem, [], 0)
    assert report.dropped == (key,)
    assert key not in mem.facts and kept in mem.facts


def test_zero_hit_fact_that_was_used_stays() -> None:
    key = ("A2", "c7:x", "any")
    mem = FactMemory()
    mem.record(key, (Effect("recolor", color=3),), 0)
    mem.facts[key].hits = 0
    assert fluid_sleep(mem, [key], 0).dropped == ()


def test_empty_used_set_changes_nothing() -> None:
    key = ("A3", "c1:x", "any")
    mem = seeded(key, LEFT, 7)
    mem.record(key, RIGHT, 8)
    report = fluid_sleep(mem, set(), 0)
    assert (report.demoted, report.merged, report.dropped) == ((), (), ())
    assert mem.facts[key].hits == 7


def test_time_cap_truncates_on_a_big_memory() -> None:
    mem = FactMemory()
    for i in range(5000):
        mem.record(("A1", f"c{i % 10}:s{i}", "any"), NONE, 0)
    ticks = iter(range(10**6))

    def clock() -> float:
        return next(ticks) * 0.001

    report = fluid_sleep(mem, [], 0, time_s=0.01, clock=clock)
    assert report.truncated
    assert len(mem.facts) == 5000
    assert SLEEP_MAX_S == 0.5


def test_agent_logs_sleep_at_level_up_and_game_end(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "f.jsonl", "fake", "FLUID", 0)
    agent = FluidAgent(
        "fake", CONFIGS["FLUID"], seed=0, log=log, game_deadline=1e12, run_deadline=1e12
    )
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    play(game, agent, max_steps=30, until=lambda: game.levels >= 1)
    play(game, agent, max_steps=2)
    agent.close()
    lines = (tmp_path / "f.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(line) for line in lines]
    sleeps = [r for r in recs if r["type"] == "sleep"]
    assert len(sleeps) == 2
    keys = {"step", "demoted", "merged", "dropped", "seconds", "truncated"}
    assert all(keys <= set(r) for r in sleeps)
    up = next(r for r in recs if r["type"] == "level_up")
    assert sleeps[0]["step"] == up["step"]
    assert recs[-1]["type"] == "game_end"
    levels = [
        p["level"] for r in recs if r["type"] == "action" for p in r["programs"] if "level" in p
    ]
    assert levels and set(levels) <= {0, 1, None}
    assert agent.mem.facts[("event", "level_up", "any")].level == 0
