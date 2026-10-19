from pathlib import Path

from arcagent.core.oracle import OracleModel
from arcagent.core.types import GRID_SIZE, Action, Phase


def test_oracle_loads_and_steps(env_dir: Path) -> None:
    m = OracleModel.load(env_dir, "ls20")
    obs = m.observe()
    assert obs.grid.shape == (GRID_SIZE, GRID_SIZE)
    assert obs.phase == Phase.NOT_FINISHED
    assert obs.win_levels == 7
    acts = m.legal_actions()
    assert acts and all(a.id != 0 for a in acts)
    before = m.key()
    m.step(acts[0])
    assert m.steps == 1
    assert isinstance(m.key(), bytes)
    assert len(before) == len(m.key())


def test_oracle_clone_is_independent(env_dir: Path) -> None:
    m = OracleModel.load(env_dir, "ls20")
    c = m.clone()
    k0 = m.key()
    acts = c.legal_actions()[:5]
    for a in acts:
        c.step(a)
    assert m.key() == k0
    assert c.steps == len(acts) and m.steps == 0


def test_coordinate_game_expands_lattice(env_dir: Path) -> None:
    m = OracleModel.load(env_dir, "vc33")
    acts = m.legal_actions()
    assert len(acts) == 64
    assert all(a.id == 6 for a in acts)
    m.step(Action(6, 32, 32))
