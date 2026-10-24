"""fluid_modules: each null's behavior, the factory and the ablation configs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from arcagent.core.types import Action, Grid
from arcagent.mind.config import CONFIGS, MODULE_NAMES
from arcagent.mind.facts import BROKEN_MISSES, Effect, FactMemory
from arcagent.mind.fluid import FluidAgent
from arcagent.mind.fluid_modules import (
    NEVER_BROKEN,
    KeepTransfer,
    MissMetacognition,
    NullFluidInhibition,
    NullFluidMetacognition,
    NullFluidTransfer,
    NullFluidVoi,
    TesterInhibition,
    TesterVoi,
    build_fluid_modules,
)
from arcagent.mind.goals import Goal, GoalBook
from arcagent.mind.objects import Scene, Tracker, shape_of
from arcagent.mind.runlog import RunLog
from arcagent.mind.tester import Tried
from tests.mind.fakegames import FakeGame, play

DOT = shape_of([(0, 0)])
PSIG, LSIG = f"c1:{DOT}", f"c9:{DOT}"


def scene_of(cells: dict[tuple[int, int], int]) -> Scene:
    g: Grid = np.zeros((64, 64), dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return Tracker().update(g, frozenset())[0]


def test_ablation_configs_switch_one_module_off() -> None:
    for name in ("FLUID", "FLUID-LLM"):
        cfg = CONFIGS[name]
        assert cfg.metacognition and cfg.inhibition and cfg.voi and cfg.transfer
    assert CONFIGS["FLUID-LLM-voi"].voi is False
    for m in MODULE_NAMES:
        cfg = CONFIGS[f"FLUID-LLM-{m}"]
        assert cfg.fluid and cfg.llm and cfg.name == f"FLUID-LLM-{m}"
        assert {n for n in MODULE_NAMES if not getattr(cfg, n)} == {m}


def test_factory_maps_each_flag_to_real_or_null() -> None:
    real = build_fluid_modules(CONFIGS["FLUID"])
    assert isinstance(real.voi, TesterVoi)
    assert isinstance(real.inhibition, TesterInhibition)
    assert isinstance(real.metacognition, MissMetacognition)
    assert isinstance(real.transfer, KeepTransfer)
    assert isinstance(build_fluid_modules(CONFIGS["FLUID-LLM-voi"]).voi, NullFluidVoi)
    off = build_fluid_modules(CONFIGS["FLUID-LLM-inhibition"]).inhibition
    assert isinstance(off, NullFluidInhibition)
    meta = build_fluid_modules(CONFIGS["FLUID-LLM-metacognition"]).metacognition
    assert isinstance(meta, NullFluidMetacognition)
    assert isinstance(
        build_fluid_modules(CONFIGS["FLUID-LLM-transfer"]).transfer, NullFluidTransfer
    )


def test_null_voi_takes_the_first_untried_action_with_an_unknown() -> None:
    s = scene_of({(2, 2): 1})
    mem = FactMemory()
    mem.record(("A1", PSIG, "any"), (Effect("move", dy=-1),), 0)  # A1 is known
    actions = [Action(4), Action(2), Action(1), Action(3)]
    tried: Tried = {(s.key(), Action(2).key())}
    test = NullFluidVoi().choose(s, actions, mem, (), tried, None)
    assert test is not None and test.action == Action(3)
    assert TesterVoi().choose(s, actions, mem, (), tried, None) is not None
    for a in actions:
        mem.record((f"A{a.id}", PSIG, "any"), (Effect("none"),), 0)
    assert NullFluidVoi().choose(s, actions, mem, (), set(), None) is None


def test_null_inhibition_never_vetoes_a_predicted_death() -> None:
    s = scene_of({(2, 2): 1, (3, 2): 9})
    mem = FactMemory()
    mem.record(("A4", PSIG, "any"), (Effect("move", dx=1),), 0)
    mem.record(("A4", LSIG, "any"), (Effect("none"),), 0)
    mem.record(("any", PSIG, "touch:9"), (Effect("game_over"),), 0)
    assert TesterInhibition().veto(s, Action(4), mem, set()) is not None
    assert NullFluidInhibition().veto(s, Action(4), mem, set()) is None


def test_null_metacognition_never_breaks_a_fact() -> None:
    assert MissMetacognition().broken_after() == BROKEN_MISSES
    assert NullFluidMetacognition().broken_after() == NEVER_BROKEN
    mem = FactMemory()
    key = ("A3", PSIG, "any")
    mem.record(key, (Effect("move", dx=-1),), 0)
    mem.broken_after = NullFluidMetacognition().broken_after()
    for i in range(10):
        mem.record(key, (Effect("none"),), i + 1)
    assert mem.facts[key].status == "thin"
    mem.broken_after = BROKEN_MISSES
    assert mem.facts[key].status == "broken"


def test_null_transfer_clears_facts_and_keeps_goal_forms() -> None:
    mem = FactMemory()
    mem.record(("A3", PSIG, "any"), (Effect("move", dx=-1),), 0)
    mem.blockers[PSIG] = {8}
    mem.walkable[PSIG] = {0}
    mem.safe_touch[PSIG] = {4}
    mem.safe_near[PSIG] = {4}
    mem.landed[("A3", PSIG, "any")] = (1, 1)
    goals = GoalBook()
    form = Goal("reach", a=PSIG, b=LSIG)
    goals.confirmed_forms.append(form)
    goals.status[form] = "confirmed"
    KeepTransfer().carry(mem, goals)
    assert mem.facts
    NullFluidTransfer().carry(mem, goals)
    assert not (mem.facts or mem.toc or mem.blockers or mem.walkable)
    assert not (mem.safe_touch or mem.safe_near or mem.landed)
    assert goals.confirmed_forms == [form]
    assert mem.lookup("A3", PSIG, "any") is None


def test_full_fake_game_with_every_null_module(tmp_path: Path) -> None:
    cfg = replace(CONFIGS["FLUID"], metacognition=False, inhibition=False, voi=False)
    cfg = replace(cfg, transfer=False)
    log = RunLog(tmp_path / "f.jsonl", "fake", cfg.name, 0)
    agent = FluidAgent("fake", cfg, seed=0, log=log, game_deadline=1e12, run_deadline=1e12)
    assert isinstance(agent.modules.transfer, NullFluidTransfer)
    assert agent.mem.broken_after == NEVER_BROKEN
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[(4, 4)])
    acts = play(game, agent, max_steps=120)
    agent.close()
    lines = (tmp_path / "f.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(line) for line in lines]
    assert not [r for r in recs if r["type"] == "agent_error"]
    assert len([r for r in recs if r["type"] == "action"]) == len(acts)
    assert game.levels == 2 or len(acts) == 120
    assert recs[-1]["type"] == "game_end"
